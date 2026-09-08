// Load benchmark: cold start to first generation, ONNX or WebNN.
//
//   bench-load.html?model=intellitex&device=webgpu&dtype=q4&src=local&mode=cold&items=0
//   bench-load.html?model=texo&device=webnn&dtype=fp16&src=hf&mode=cold&items=0
//
//   model  intellitex | texify | texo
//   device webgpu | wasm | webnn
//   dtype  q4 | q8 | fp32 | fp16
//   src    local | hf
//   mode   cold | warm
//   items  0 skips the full suite (still runs warmup + first/second fixture)
//
// WebNN rows include per-graph constantsMs/emitMs/buildMs from the catalog loader.
import { pipeline, env, VisionEncoderDecoderModel, PreTrainedTokenizer, Tensor, cat } from "./vendor/transformers/transformers.min.js";

const q = new URLSearchParams(location.search);
const queue = (q.get("queue") ?? "").split(";").filter(Boolean);
const spec = queue.length ? Object.fromEntries(["model", "device", "dtype", "src", "mode", "items"].map((k, i) => [k, queue[0].split(",")[i]])) : {};
const get = (k) => spec[k] || q.get(k);
const model = get("model") ?? "intellitex";
const device = get("device") ?? (model === "texo" ? "wasm" : "webgpu");
const dtype = get("dtype") ?? (model === "texo" ? (device === "webnn" ? "fp16" : "fp32") : device === "webgpu" || device === "webnn" ? (device === "webnn" ? "fp16" : "q4") : "q8");
const src = get("src") ?? "local";
const rev = q.get("rev") ?? "main";
const mode = get("mode") ?? "cold";
const runItems = get("items") !== "0";
const label = q.get("label") ?? "";
const shareConstants = q.get("shareConstants") !== "0";

const out = document.getElementById("result"), progress = document.getElementById("progress");
const show = (o) => { out.textContent = JSON.stringify(o, null, 2); };

env.allowRemoteModels = true; env.allowLocalModels = false;
if (src === "hf") {
  env.remoteHost = "https://huggingface.co/";
  env.remotePathTemplate = `ozhyhinas/latexgen-models/resolve/${rev}/{model}`;
} else {
  env.remoteHost = new URL("models/", import.meta.url).href;
  env.remotePathTemplate = "{model}";
}
env.backends.onnx.wasm.wasmPaths = new URL("vendor/ort/", import.meta.url).href;
env.backends.onnx.wasm.numThreads = 1;

const realFetch = globalThis.fetch;
const files = {}; let firstProgress = 0, lastProgress = 0;
const progress_callback = (p) => {
  if (p.status === "progress" && p.total) {
    const now = performance.now(); firstProgress ||= now; lastProgress = now;
    files[p.file] = p.total;
    progress.textContent = `${p.file} ${Math.round((p.loaded / p.total) * 100)}%`;
  } else if (p.file && p.total) {
    const now = performance.now(); firstProgress ||= now; lastProgress = now;
    files[p.file] = p.total;
    progress.textContent = `${p.file} ${Math.round((p.loaded / p.total) * 100)}%`;
  }
};

const PREFIX = "Convert natural-language math into a STRICT LaTeX equation\n";
const TEXO_MEAN = 0.7931, TEXO_STD = 0.1738, S = 384;
async function texoPre(blob) {
  const bmp = await createImageBitmap(blob); const c = new OffscreenCanvas(bmp.width, bmp.height); const ctx = c.getContext("2d");
  ctx.fillStyle = "white"; ctx.fillRect(0, 0, c.width, c.height); ctx.drawImage(bmp, 0, 0);
  const { data, width, height } = ctx.getImageData(0, 0, c.width, c.height);
  const g = new Uint8ClampedArray(width * height); for (let i = 0, p = 0; i < g.length; i++, p += 4) g[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
  let dark = 0; for (const v of g) if (v < 200) dark++; if (dark >= g.length - dark) for (let i = 0; i < g.length; i++) g[i] = 255 - g[i];
  let mn = 255, mx = 0; for (const v of g) { if (v < mn) mn = v; if (v > mx) mx = v; }
  let x0 = width, y0 = height, x1 = -1, y1 = -1;
  if (mx > mn) for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) { const n = ((g[y * width + x] - mn) / (mx - mn)) * 255; if (n < 200) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; } }
  if (x1 < x0 || y1 < y0) { x0 = 0; y0 = 0; x1 = width - 1; y1 = height - 1; }
  const cw = Math.max(1, x1 - x0), ch = Math.max(1, y1 - y0);
  const gi = new ImageData(width, height); for (let i = 0, p = 0; i < g.length; i++, p += 4) { gi.data[p] = gi.data[p + 1] = gi.data[p + 2] = g[i]; gi.data[p + 3] = 255; }
  const gc = new OffscreenCanvas(width, height); gc.getContext("2d").putImageData(gi, 0, 0);
  const sc = S / Math.min(ch, cw); let nw = Math.round(cw * sc), nh = Math.round(ch * sc);
  if (nw > S || nh > S) { const r = Math.min(S / nw, S / nh); nw = Math.round(nw * r); nh = Math.round(nh * r); }
  const o = new OffscreenCanvas(S, S).getContext("2d"); o.fillStyle = "black"; o.fillRect(0, 0, S, S);
  o.drawImage(gc, x0, y0, cw, ch, Math.floor((S - nw) / 2), Math.floor((S - nh) / 2), nw, nh);
  const od = o.getImageData(0, 0, S, S).data; const arr = new Float32Array(S * S);
  for (let i = 0, p = 0; i < arr.length; i++, p += 4) arr[i] = (od[p] / 255 - TEXO_MEAN) / TEXO_STD; return arr;
}

function graphRows(stats) {
  if (!stats?.graphs) return [];
  return Object.entries(stats.graphs).map(([name, g]) => ({
    name,
    ops: g.ops,
    constants: g.constants,
    constantBytes: g.constantBytes,
    constantsMs: g.constantsMs,
    emitMs: g.emitMs,
    buildMs: g.buildMs,
  }));
}

async function fixtureFor(kind) {
  if (kind === "text") {
    const items = (await realFetch("bench-data.json").then((r) => r.json()));
    const it = items.find((i) => i.id === "quadratic-formula") ?? items[0];
    return { id: it.id, input: it.input, all: items.map((i) => [i.id, i.input]) };
  }
  const list = await realFetch("bench-images.json").then((r) => r.json());
  const it = list.find((i) => i.id === "quadratic") ?? list[0];
  const blob = await realFetch(it.image.replace(/^\//, "")).then((r) => r.blob());
  const all = [];
  if (runItems) {
    for (const x of list) all.push([x.id, await realFetch(x.image.replace(/^\//, "")).then((r) => r.blob())]);
  }
  return { id: it.id, input: blob, all };
}

const result = { label, model, device, dtype, src, rev, mode, ua: navigator.userAgent, at: new Date().toISOString() };
result.shareConstants = shareConstants;
let peakJSHeapBytes = performance.memory?.usedJSHeapSize ?? null;
result.jsHeapAtStartBytes = peakJSHeapBytes;
const heapTimer = peakJSHeapBytes == null ? null : setInterval(() => {
  peakJSHeapBytes = Math.max(peakJSHeapBytes, performance.memory.usedJSHeapSize);
}, 25);
try {
  if (mode === "cold") {
    for (const k of await caches.keys()) if (/transformers/i.test(k)) await caches.delete(k);
  }
  let predict, kind, webnnHandle = null;
  const t0 = performance.now();
  if (device === "webnn") {
    const webnnOpts = {
      skipWarmup: true,
      preferLocalConstants: src === "local",
      preserveModelSource: src === "hf",
      onProgress: progress_callback,
      shareConstants,
    };
    if (model === "intellitex") {
      const { createIntelliTeXWebNN } = await import("./intellitex-webnn.js");
      webnnHandle = await createIntelliTeXWebNN(webnnOpts);
      predict = (text) => webnnHandle.run(text);
      kind = "text";
    } else if (model === "texify") {
      const { createTexifyWebNN } = await import("./texify-webnn.js");
      webnnHandle = await createTexifyWebNN(webnnOpts);
      predict = (blob) => webnnHandle.run(blob);
      kind = "image";
    } else {
      const { createTexoWebNN } = await import("./texo-webnn.js");
      webnnHandle = await createTexoWebNN(webnnOpts);
      predict = async (blob) => webnnHandle.run(await texoPre(blob));
      kind = "image";
    }
    result.backend = webnnHandle.stats.backend;
    result.graphs = graphRows(webnnHandle.stats);
    result.entry = webnnHandle.stats.entry;
    result.buildMsReported = webnnHandle.stats.buildMs;
    result.constantsMsReported = result.graphs.reduce((sum, graph) => sum + (graph.constantsMs || 0), 0);
    result.compileMsReported = result.graphs.reduce((sum, graph) => sum + (graph.buildMs || 0), 0);
  } else {
    const cfg = { device, dtype, progress_callback };
    if (model === "intellitex") {
      const p = await pipeline("text2text-generation", "intellitex", cfg);
      predict = async (text) => (await p(PREFIX + text, { max_new_tokens: 256 }))[0].generated_text.trim();
      kind = "text";
    } else if (model === "texify") {
      const p = await pipeline("image-to-text", "texify", cfg);
      predict = async (blob) => (await p(blob, { max_new_tokens: 384 }))[0].generated_text.trim();
      kind = "image";
    } else {
      const m = await VisionEncoderDecoderModel.from_pretrained("texo", cfg);
      const tok = await PreTrainedTokenizer.from_pretrained("texo");
      predict = async (blob) => { const a = await texoPre(blob); const t = new Tensor("float32", a, [1, 1, S, S]); const r = await m.generate({ inputs: cat([t, t, t], 1), max_new_tokens: 512 }); return tok.batch_decode(r, { skip_special_tokens: true })[0].trim(); };
      kind = "image";
    }
  }
  const t1 = performance.now();
  if (heapTimer) clearInterval(heapTimer);
  result.peakJSHeapBytes = peakJSHeapBytes;
  result.jsHeapAtLoadEndBytes = performance.memory?.usedJSHeapSize ?? null;
  result.loadMs = Math.round(t1 - t0);
  result.downloadMs = firstProgress ? Math.round(lastProgress - firstProgress) : 0;
  result.sessionMs = Math.round(t1 - (lastProgress || t0));
  result.files = files;
  result.totalBytes = Object.values(files).reduce((a, b) => a + b, 0);
  result.network = globalThis.__benchNet;

  const fixture = await fixtureFor(kind);
  const dummy = device === "webnn"
    ? (kind === "text" ? "x squared" : fixture.input)
    : fixture.input;

  progress.textContent = "warm-up";
  const w0 = performance.now();
  if (device === "webnn" && model === "texo") {
    await webnnHandle.run(new Float32Array(384 * 384).fill(-4.5628));
  } else if (device === "webnn" && model === "texify") {
    await webnnHandle.runPixels(new Float32Array(3 * 420 * 420));
  } else {
    await predict(dummy);
  }
  result.warmupMs = Math.round(performance.now() - w0);

  progress.textContent = `first-gen ${fixture.id}`;
  const f0 = performance.now();
  result.firstOutput = await predict(fixture.input);
  if (webnnHandle?.lastRun) result.firstTokens = Array.from(webnnHandle.lastRun().tokens);
  result.firstGenMs = Math.round(performance.now() - f0);
  result.fixtureId = fixture.id;
  result.coldToFirstMs = result.loadMs + result.warmupMs + result.firstGenMs;

  progress.textContent = `second-gen ${fixture.id}`;
  const s0 = performance.now();
  result.secondOutput = await predict(fixture.input);
  if (webnnHandle?.lastRun) result.secondTokens = Array.from(webnnHandle.lastRun().tokens);
  result.secondGenMs = Math.round(performance.now() - s0);

  if (runItems) {
    result.items = [];
    const items = kind === "text" ? fixture.all : fixture.all;
    for (const [id, input] of items) {
      progress.textContent = id;
      const s = performance.now(); let output = "", err = "", tokens = null;
      try {
        output = await predict(input);
        if (webnnHandle?.lastRun) tokens = Array.from(webnnHandle.lastRun().tokens);
      } catch (e) { err = String(e).slice(0, 200); }
      result.items.push({ id, ms: Math.round(performance.now() - s), output, tokens, err });
    }
  }
  progress.textContent = "done";
  show(result); document.title = "done";
} catch (e) {
  if (heapTimer) clearInterval(heapTimer);
  result.peakJSHeapBytes = peakJSHeapBytes;
  result.jsHeapAtLoadEndBytes = performance.memory?.usedJSHeapSize ?? null;
  result.error = String(e?.stack || e); show(result); document.title = "failed";
}

async function cacheSettled(names, ms = 120000) {
  const t0 = performance.now();
  while (performance.now() - t0 < ms) {
    try {
      const c = await caches.open("transformers-cache");
      const urls = (await c.keys()).map((k) => k.url);
      if (names.every((n) => urls.some((u) => u.endsWith("/" + n)))) return true;
    } catch { return false; }
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}
if (!result.error && device !== "webnn") result.cacheSettled = await cacheSettled(Object.keys(files));

try { await realFetch("api/bench", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ tier: "load", ...result }) }); } catch { /* no collector */ }
if (queue.length > 1) {
  const next = new URL(location.href);
  next.searchParams.set("queue", queue.slice(1).join(";"));
  setTimeout(() => location.replace(next), 1500);
} else if (queue.length) document.title = "queue-done";
