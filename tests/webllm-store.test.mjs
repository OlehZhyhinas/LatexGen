// Unit tests for public/webllm-store.js. Run: npm test  (node --test)
// hasEntry/writeEntry/openScopeDir are exercised against an in-memory fake
// of the File System Access API (getFileHandle/getDirectoryHandle/
// createWritable/getFile) rather than real OPFS, which node:test has no
// access to.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";

import {
  cleanModelUrl,
  shardKey,
  planShards,
  runPool,
  openScopeDir,
  hasEntry,
  writeEntry,
  sidecarNames,
  prefetchModel,
  OPFS_ROOT,
  MODEL_SCOPE,
} from "../public/webllm-store.js";

// ---- cleanModelUrl ------------------------------------------------------

test("cleanModelUrl appends resolve/main/ to a bare repo URL", () => {
  assert.equal(cleanModelUrl("https://huggingface.co/mlc-ai/X"), "https://huggingface.co/mlc-ai/X/resolve/main/");
});

test("cleanModelUrl handles a trailing slash the same way", () => {
  assert.equal(cleanModelUrl("https://huggingface.co/mlc-ai/X/"), "https://huggingface.co/mlc-ai/X/resolve/main/");
});

test("cleanModelUrl leaves an already-resolved URL alone apart from the trailing slash", () => {
  assert.equal(
    cleanModelUrl("https://huggingface.co/mlc-ai/X/resolve/main"),
    "https://huggingface.co/mlc-ai/X/resolve/main/",
  );
  assert.equal(
    cleanModelUrl("https://huggingface.co/mlc-ai/X/resolve/main/"),
    "https://huggingface.co/mlc-ai/X/resolve/main/",
  );
});

// ---- shardKey -------------------------------------------------------------

test("shardKey is the lowercase hex sha256 of the URL string", async () => {
  const url = "https://huggingface.co/mlc-ai/X/resolve/main/params_shard_0.bin";
  const expected = createHash("sha256").update(url).digest("hex");
  assert.equal(await shardKey(url), expected);
});

// ---- planShards -------------------------------------------------------------

test("planShards skips present records, sorts the rest by nbytes desc, and totals correctly", () => {
  const records = [
    { dataPath: "a.bin", nbytes: 30 },
    { dataPath: "b.bin", nbytes: 10 },
    { dataPath: "c.bin", nbytes: 20 },
  ];
  const { missing, totalBytes, missingBytes } = planShards(records, new Set(["b.bin"]));
  assert.deepEqual(missing, [
    { dataPath: "a.bin", nbytes: 30 },
    { dataPath: "c.bin", nbytes: 20 },
  ]);
  assert.equal(totalBytes, 60);
  assert.equal(missingBytes, 50);
});

// ---- runPool -------------------------------------------------------------

test("runPool never runs more than `concurrency` tasks at once and returns results in input order", async () => {
  const items = Array.from({ length: 10 }, (_, i) => i);
  let inFlight = 0;
  let maxInFlight = 0;
  const results = await runPool(items, 3, (item) => new Promise((resolve) => {
    inFlight += 1;
    maxInFlight = Math.max(maxInFlight, inFlight);
    setTimeout(() => {
      inFlight -= 1;
      resolve(item * 2);
    }, 5);
  }));
  assert.equal(maxInFlight, 3);
  assert.deepEqual(results, items.map((i) => i * 2));
});

test("runPool stops handing out new items after the first failure and rejects with its error", async () => {
  const items = Array.from({ length: 10 }, (_, i) => i);
  let started = 0;
  await assert.rejects(
    runPool(items, 3, async (item) => {
      started += 1;
      if (item === 0) throw new Error("boom");
      return item;
    }),
    /boom/,
  );
  // Exactly the runners in flight when item 0 failed (one per concurrency
  // slot) ever start; none of the remaining queue is touched.
  assert.equal(started, 3);
});

// ---- fake File System Access API for hasEntry/writeEntry/openScopeDir ----

function notFound() {
  const err = new Error("not found");
  err.name = "NotFoundError";
  return err;
}

function makeFakeDir() {
  const files = new Map();
  const dirs = new Map();
  return {
    files,
    dirs,
    async getFileHandle(name, opts) {
      if (!files.has(name)) {
        if (!opts?.create) throw notFound();
        files.set(name, { bytes: new Uint8Array(0) });
      }
      const entry = files.get(name);
      return {
        async createWritable() {
          const chunks = [];
          const stream = new WritableStream({
            write(chunk) {
              if (typeof chunk === "string") chunks.push(new TextEncoder().encode(chunk));
              else chunks.push(chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk));
            },
            close() {
              const size = chunks.reduce((sum, c) => sum + c.byteLength, 0);
              const bytes = new Uint8Array(size);
              let offset = 0;
              for (const c of chunks) { bytes.set(c, offset); offset += c.byteLength; }
              entry.bytes = bytes;
            },
          });
          // Real FileSystemWritableFileStream instances are WritableStreams
          // that also expose direct write()/close() convenience methods (used
          // by webllm-store.js for the small meta.json write). Acquire the
          // writer lazily so an instance used via pipeTo (the .bin body)
          // never gets locked by a writer nobody asked for.
          let writer;
          stream.write = (data) => (writer ??= stream.getWriter()).write(data);
          stream.close = () => (writer ??= stream.getWriter()).close();
          return stream;
        },
        async getFile() {
          return {
            size: entry.bytes.byteLength,
            async text() { return new TextDecoder().decode(entry.bytes); },
            async arrayBuffer() {
              return entry.bytes.buffer.slice(entry.bytes.byteOffset, entry.bytes.byteOffset + entry.bytes.byteLength);
            },
          };
        },
      };
    },
    async getDirectoryHandle(name, opts) {
      if (!dirs.has(name)) {
        if (!opts?.create) throw notFound();
        dirs.set(name, makeFakeDir());
      }
      return dirs.get(name);
    },
  };
}

// ---- hasEntry / writeEntry -------------------------------------------------

test("writeEntry stores the body and both sidecars the way hasEntry expects to find them", async () => {
  const dir = makeFakeDir();
  const url = "https://huggingface.co/mlc-ai/X/resolve/main/params_shard_0.bin";
  const key = await shardKey(url);

  assert.equal(await hasEntry(dir, url), false);

  let bytesSeen = 0;
  const response = new Response(new Uint8Array(1000), { headers: { "content-type": "application/octet-stream" } });
  const nbytes = await writeEntry(dir, url, response, (n) => { bytesSeen += n; });

  assert.equal(await hasEntry(dir, url), true);
  assert.equal(bytesSeen, 1000);
  assert.equal(nbytes, 1000);

  const dataFile = await (await dir.getFileHandle(`${key}.bin`)).getFile();
  assert.equal(dataFile.size, 1000);

  const metaFile = await (await dir.getFileHandle(`${key}.meta.json`)).getFile();
  assert.deepEqual(JSON.parse(await metaFile.text()), { url, contentType: "application/octet-stream" });

  const recordFile = await (await dir.getFileHandle(`${key}.record.json`)).getFile();
  assert.deepEqual(JSON.parse(await recordFile.text()), { url, nbytes: 1000, contentType: "application/octet-stream" });
});

// ---- hasEntry: size check and cross-bundle sidecar repair ----------------

test("hasEntry returns false when .bin exists but its size doesn't match the given nbytes", async () => {
  const dir = makeFakeDir();
  const url = "https://huggingface.co/mlc-ai/X/resolve/main/params_shard_0.bin";
  await writeEntry(dir, url, new Response(new Uint8Array(1000)), () => {});

  assert.equal(await hasEntry(dir, url, 1000), true);
  assert.equal(await hasEntry(dir, url, 999), false);
});

test("hasEntry repairs a .meta.json-only entry (stock bundle) by adding .record.json", async () => {
  const dir = makeFakeDir();
  const url = "https://huggingface.co/mlc-ai/X/resolve/main/params_shard_0.bin";
  const key = await shardKey(url);
  const { meta, record } = sidecarNames(key);

  // Write only .bin + .meta.json, as the stock OPFSStore does.
  const dataHandle = await dir.getFileHandle(`${key}.bin`, { create: true });
  const writable = await dataHandle.createWritable();
  await writable.write(new Uint8Array(1000));
  await writable.close();
  const metaHandle = await dir.getFileHandle(meta, { create: true });
  const metaWritable = await metaHandle.createWritable();
  await metaWritable.write(JSON.stringify({ url, contentType: "application/octet-stream" }));
  await metaWritable.close();

  assert.equal(await fileExistsInFakeDir(dir, record), false);
  assert.equal(await hasEntry(dir, url, 1000), true);

  const recordFile = await (await dir.getFileHandle(record)).getFile();
  assert.deepEqual(JSON.parse(await recordFile.text()), { url, nbytes: 1000, contentType: "application/octet-stream" });
});

test("hasEntry repairs a .record.json-only entry (catalog bundle) by adding .meta.json", async () => {
  const dir = makeFakeDir();
  const url = "https://huggingface.co/mlc-ai/X/resolve/main/params_shard_0.bin";
  const key = await shardKey(url);
  const { meta, record } = sidecarNames(key);

  // Write only .bin + .record.json, as the tuned catalog bundles' OPFSStore does.
  const dataHandle = await dir.getFileHandle(`${key}.bin`, { create: true });
  const writable = await dataHandle.createWritable();
  await writable.write(new Uint8Array(1000));
  await writable.close();
  const recordHandle = await dir.getFileHandle(record, { create: true });
  const recordWritable = await recordHandle.createWritable();
  await recordWritable.write(JSON.stringify({ url, nbytes: 1000, contentType: "application/octet-stream" }));
  await recordWritable.close();

  assert.equal(await fileExistsInFakeDir(dir, meta), false);
  assert.equal(await hasEntry(dir, url), true);

  const metaFile = await (await dir.getFileHandle(meta)).getFile();
  assert.deepEqual(JSON.parse(await metaFile.text()), { url, contentType: "application/octet-stream" });
});

async function fileExistsInFakeDir(dir, name) {
  try {
    await dir.getFileHandle(name);
    return true;
  } catch (err) {
    if (err?.name === "NotFoundError") return false;
    throw err;
  }
}

// ---- openScopeDir -------------------------------------------------------------

test("openScopeDir walks tvmjs-opfs-store then each encoded scope segment", async () => {
  const root = makeFakeDir();
  const storage = { getDirectory: async () => root };
  const dir = await openScopeDir(MODEL_SCOPE, storage);
  assert.ok(root.dirs.has(OPFS_ROOT));
  const scopeRoot = root.dirs.get(OPFS_ROOT);
  assert.ok(scopeRoot.dirs.has("webllm"));
  assert.ok(scopeRoot.dirs.get("webllm").dirs.has("model"));
  assert.equal(dir, scopeRoot.dirs.get("webllm").dirs.get("model"));
});

// ---- prefetchModel --------------------------------------------------------

test("prefetchModel fetches only the missing shards, largest first, and reports full progress", async () => {
  const root = makeFakeDir();
  const storage = { getDirectory: async () => root };
  const record = { model: "https://huggingface.co/mlc-ai/X" };
  const base = cleanModelUrl(record.model);

  const records = [
    { dataPath: "a.bin", nbytes: 30 },
    { dataPath: "b.bin", nbytes: 10 },
    { dataPath: "c.bin", nbytes: 20 },
  ];

  // Pre-seed "b.bin" as already present in the store.
  const dir = await openScopeDir(MODEL_SCOPE, storage);
  const bUrl = new URL("b.bin", base).href;
  await writeEntry(dir, bUrl, new Response(new Uint8Array(10)), () => {});

  const fetchedUrls = [];
  const fetchImpl = async (url) => {
    if (url === new URL("tensor-cache.json", base).href) {
      return new Response(JSON.stringify({ records }), { headers: { "content-type": "application/json" } });
    }
    fetchedUrls.push(url);
    if (url === new URL("a.bin", base).href) return new Response(new Uint8Array(30));
    if (url === new URL("c.bin", base).href) return new Response(new Uint8Array(20));
    throw new Error(`unexpected fetch: ${url}`);
  };

  let lastProgress;
  const result = await prefetchModel(record, {
    concurrency: 1,
    fetchImpl,
    storage,
    onProgress: (p) => { lastProgress = p; },
  });

  assert.deepEqual(fetchedUrls, [new URL("a.bin", base).href, new URL("c.bin", base).href]);
  assert.equal(lastProgress.loaded, lastProgress.total);
  assert.equal(lastProgress.total, 60);
  assert.deepEqual(
    { downloaded: result.downloaded, skipped: result.skipped, bytes: result.bytes },
    { downloaded: 2, skipped: 1, bytes: 50 },
  );

  // The manifest itself is now in the store too.
  assert.equal(await hasEntry(dir, new URL("tensor-cache.json", base).href), true);

  // The pre-seeded "already present" shard and both freshly fetched shards
  // all end up with both sidecar formats, so either OPFSStore bundle sees
  // them as cached.
  for (const dataPath of ["a.bin", "b.bin", "c.bin"]) {
    const shardUrl = new URL(dataPath, base).href;
    const key = await shardKey(shardUrl);
    const { meta, record } = sidecarNames(key);
    assert.equal(await fileExistsInFakeDir(dir, meta), true, `${dataPath} missing .meta.json`);
    assert.equal(await fileExistsInFakeDir(dir, record), true, `${dataPath} missing .record.json`);
  }
});
