// Prefetches WebLLM model weight shards straight into the OPFS store WebLLM
// itself reads from, using an N-way work-stealing pool instead of WebLLM's
// own loader.
//
// Stock WebLLM downloads a model's shard list as four contiguous quarters,
// one fetch() connection per quarter. The tensor-cache.json record list is
// not evenly sized — one oversized first shard routinely carries 34-43% of
// the model's bytes — so whichever quarter holds it dominates the whole
// download and the other three connections idle out early. Four connections
// also just don't saturate a fast link: measured 2026-09-14 against the
// Hugging Face CDN, four connections gave ~40 MB/s and eight ~74 MB/s. Fetching largest-first with a
// work-stealing pool keeps every connection busy until the end and lets the
// caller pick concurrency independent of shard count. Issue #71.
//
// Every shard this module writes is written exactly where WebLLM's own
// `class OPFSStore` (public/vendor/webllm/index.js) would write it — same
// root directory, same scope path, same per-key SHA-256 naming, same
// .bin/.meta.json pair — so WebLLM's own cache lookups see the shard as
// already present and skip fetching it. That layout is private to WebLLM
// 0.2.84 and is not a stable API; if the vendored bundle is ever upgraded
// (scripts/vendor-qwen3-webllm.mjs pins the version), re-check OPFSStore in
// the new bundle before assuming this still matches.

export const OPFS_ROOT = "tvmjs-opfs-store";
export const MODEL_SCOPE = "webllm/model";

export const opfsAvailable = () =>
  typeof navigator !== "undefined" && typeof navigator.storage?.getDirectory === "function";

// Mirrors WebLLM's cleanModelUrl: normalize a model URL (HF repo root or an
// already-resolved URL) to the "resolve/main/"-terminated form every shard
// and the manifest are resolved against.
export function cleanModelUrl(modelUrl) {
  let url = modelUrl.endsWith("/") ? modelUrl : `${modelUrl}/`;
  if (!/.+\/resolve\/.+\//.test(url)) url += "resolve/main/";
  return new URL(url).href;
}

// Lowercase hex SHA-256 of the UTF-8 URL string; this is the OPFS key WebLLM
// looks a shard up by, so it must match byte-for-byte.
export async function shardKey(url, subtle = globalThis.crypto?.subtle) {
  if (!subtle) throw new Error("webllm-store: crypto.subtle is unavailable");
  const digest = await subtle.digest("SHA-256", new TextEncoder().encode(url));
  return Array.from(new Uint8Array(digest)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

// records: tensor-cache.json's `records` array ({ dataPath, nbytes, ... }).
// presentKeys: Set of dataPath values already stored. Largest-first so the
// work-stealing pool below spends its early, most-parallel window on the
// bytes that matter most.
export function planShards(records, presentKeys) {
  const missing = records.filter((r) => !presentKeys.has(r.dataPath)).sort((a, b) => b.nbytes - a.nbytes);
  const totalBytes = records.reduce((sum, r) => sum + r.nbytes, 0);
  const missingBytes = missing.reduce((sum, r) => sum + r.nbytes, 0);
  return { missing, totalBytes, missingBytes };
}

// N-way work-stealing pool: `concurrency` runners each pull the next item
// off the shared queue as soon as they're free, so a runner that finishes an
// early (large) item immediately starts the next one rather than sitting
// idle until its quarter's turn — this is the fix for WebLLM's fixed
// quarter-per-connection split. Results land in input order regardless of
// completion order. On the first failure, stop handing out new items, let
// runners already in flight finish (success or failure), then reject with
// the first error.
export async function runPool(items, concurrency, worker) {
  const results = new Array(items.length);
  let next = 0;
  let firstError = null;
  const n = Math.max(1, Math.min(concurrency, items.length || 1));

  async function run() {
    for (;;) {
      if (firstError) return;
      const i = next++;
      if (i >= items.length) return;
      try {
        results[i] = await worker(items[i], i);
      } catch (err) {
        if (!firstError) firstError = err;
      }
    }
  }

  await Promise.all(Array.from({ length: n }, run));
  if (firstError) throw firstError;
  return results;
}

// Walks down from `storage.getDirectory()` through the fixed OPFS_ROOT and
// then each "/"-split, percent-encoded part of `scope`, creating directories
// as needed — identical traversal to OPFSStore.getScopedDirectory.
export async function openScopeDir(scope = MODEL_SCOPE, storage = navigator.storage) {
  let dir = await storage.getDirectory();
  dir = await dir.getDirectoryHandle(OPFS_ROOT, { create: true });
  for (const part of scope.split("/").filter((p) => p.length > 0)) {
    dir = await dir.getDirectoryHandle(encodeURIComponent(part), { create: true });
  }
  return dir;
}

// Whether a shard for `url` is already stored: WebLLM only requires the
// .bin to exist to treat a key as cached.
export async function hasEntry(dir, url) {
  const key = await shardKey(url);
  try {
    await dir.getFileHandle(`${key}.bin`);
    return true;
  } catch (err) {
    if (err?.name === "NotFoundError") return false;
    throw err;
  }
}

// Writes one shard's body and metadata in OPFSStore's own format: the body
// streamed straight from `response` into `${key}.bin` (counting bytes as
// they pass through so the caller can report progress without buffering the
// whole shard), then `${key}.meta.json` alongside it.
export async function writeEntry(dir, url, response, onBytes) {
  const key = await shardKey(url);
  const dataHandle = await dir.getFileHandle(`${key}.bin`, { create: true });
  const writable = await dataHandle.createWritable();
  const counter = new TransformStream({
    transform(chunk, controller) {
      onBytes?.(chunk.byteLength ?? chunk.length ?? 0);
      controller.enqueue(chunk);
    },
  });
  if (response.body !== null) {
    await response.body.pipeThrough(counter).pipeTo(writable);
  } else {
    const buf = await response.arrayBuffer();
    onBytes?.(buf.byteLength);
    await writable.write(buf);
    await writable.close();
  }
  const metaHandle = await dir.getFileHandle(`${key}.meta.json`, { create: true });
  const metaWritable = await metaHandle.createWritable();
  const contentType = response.headers.get("content-type") ?? undefined;
  await metaWritable.write(JSON.stringify({ url, contentType }));
  await metaWritable.close();
}

// Downloads every shard of `record` (a WebLLM model_list entry; `record.model`
// is the HF repo/resolve URL) that isn't already in the OPFS store, in
// largest-first order across an N-way work-stealing pool, so WebLLM's own
// load finds every key present and does no fetching of its own.
export async function prefetchModel(
  record,
  { concurrency = 8, onProgress = () => {}, signal, fetchImpl = globalThis.fetch, storage = globalThis.navigator?.storage } = {},
) {
  const started = Date.now();
  const base = cleanModelUrl(record.model);
  const dir = await openScopeDir(MODEL_SCOPE, storage);

  // Manifest: fetch and store it if missing, then always read the stored
  // copy back so the records we plan against are the exact bytes WebLLM
  // itself will later read.
  const manifestUrl = new URL("tensor-cache.json", base).href;
  if (!(await hasEntry(dir, manifestUrl))) {
    const res = await fetchImpl(manifestUrl, { method: "GET", signal });
    if (!res.ok) throw new Error(`prefetch tensor-cache.json: HTTP ${res.status}`);
    await writeEntry(dir, manifestUrl, res);
  }
  const manifestKey = await shardKey(manifestUrl);
  const manifestText = await (await (await dir.getFileHandle(`${manifestKey}.bin`)).getFile()).text();
  const records = JSON.parse(manifestText).records;

  // Presence check for every shard, cheap enough to run at higher
  // concurrency than the downloads themselves.
  const presentFlags = await runPool(records, 16, async (r) => {
    const url = new URL(r.dataPath, base).href;
    return (await hasEntry(dir, url)) ? r.dataPath : null;
  });
  const presentKeys = new Set(presentFlags.filter(Boolean));
  const { missing, totalBytes, missingBytes } = planShards(records, presentKeys);

  let loaded = totalBytes - missingBytes;
  let done = records.length - missing.length; // shards already present count as done
  let lastReport = 0;
  const total = totalBytes;
  function report(file, force) {
    const now = Date.now();
    if (!force && now - lastReport < 100) return;
    lastReport = now;
    onProgress({ phase: "prefetch", loaded, total, file, done, count: records.length });
  }
  report(undefined, true);

  await runPool(missing, concurrency, async (r) => {
    const url = new URL(r.dataPath, base).href;
    const res = await fetchImpl(url, { method: "GET", signal });
    if (!res.ok) throw new Error(`prefetch ${r.dataPath}: HTTP ${res.status}`);
    await writeEntry(dir, url, res, (n) => {
      loaded += n;
      report(r.dataPath);
    });
    done += 1;
    report(r.dataPath, true);
  });

  return { downloaded: missing.length, skipped: records.length - missing.length, bytes: missingBytes, ms: Date.now() - started };
}
