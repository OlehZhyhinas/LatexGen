// Load benchmark: how long one in-browser model takes to become usable, from
// a cold cache (network) and from the Cache API (disk), and what it outputs
// on the bench items so a weight change can be checked for equivalence.
//
//   bench-load.html?model=intellitex&device=webgpu&dtype=q4&src=hf&mode=cold
//
//   model  intellitex | texify | texo
//   device webgpu | wasm          dtype q4 | q8 | fp32
//   src    local (this origin) | hf (the Hugging Face repo the static build uses)
//   rev    Hugging Face revision for src=hf (default main)
//   mode   cold: drop the transformers.js cache and bypass the HTTP cache
//          warm: load from whatever the previous run cached
//   items  0 to skip the bench items (default: run them)
//
// Results land in #result as JSON and the title flips to "done" / "failed".
import { pipeline, env, VisionEncoderDecoderModel, PreTrainedTokenizer, Tensor, cat } from "./vendor/transformers/transformers.min.js";

const q = new URLSearchParams(location.search);
// A queue runs several configs back to back, one page load each (a failed
// WebGPU session poisons the runtime for the page, and each run should start
// from a fresh heap): queue=model,device,dtype,src,mode,items;model,...
const queue = (q.get("queue") ?? "").split(";").filter(Boolean);
const spec = queue.length ? Object.fromEntries(["model", "device", "dtype", "src", "mode", "items"].map((k, i) => [k, queue[0].split(",")[i]])) : {};
const get = (k) => spec[k] || q.get(k);
const model = get("model") ?? "intellitex";
const device = get("device") ?? (model === "texo" ? "wasm" : "webgpu");
const dtype = get("dtype") ?? (model === "texo" ? "fp32" : device === "webgpu" ? "q4" : "q8");
const src = get("src") ?? "local";
const rev = q.get("rev") ?? "main";
const mode = get("mode") ?? "cold";
const runItems = get("items") !== "0";
const label = q.get("label") ?? "";

const out = document.getElementById("result"), progress = document.getElementById("progress");
const show = (o) => { out.textContent = JSON.stringify(o, null, 2); };

// Both sources go through the "remote" loader: with an http localModelPath
// this transformers.js build skips the tokenizer existence check and loads
// no tokenizer at all (see onnx-worker.js for the same arrangement).
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

const realFetch = globalThis.fetch; // already wrapped by bench-fetch-patch.js
const files = {}; let firstProgress = 0, lastProgress = 0;
const progress_callback = (p) => {
  if (p.status === "progress" && p.total) {
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

const result = { label, model, device, dtype, src, rev, mode, ua: navigator.userAgent, at: new Date().toISOString() };
try {
  if (mode === "cold") {
    for (const k of await caches.keys()) if (/transformers/i.test(k)) await caches.delete(k);
  }
  const t0 = performance.now();
  let predict, items;
  const cfg = { device, dtype, progress_callback };
  if (model === "intellitex") {
    const p = await pipeline("text2text-generation", "intellitex", cfg);
    predict = async (text) => (await p(PREFIX + text, { max_new_tokens: 256 }))[0].generated_text.trim();
    items = (await realFetch("bench-data.json").then((r) => r.json())).map((i) => [i.id, i.input]);
  } else if (model === "texify") {
    const p = await pipeline("image-to-text", "texify", cfg);
    predict = async (blob) => (await p(blob, { max_new_tokens: 384 }))[0].generated_text.trim(); // a Blob, as the app does: blob: URLs are outside connect-src
    items = await imageItems();
  } else {
    const m = await VisionEncoderDecoderModel.from_pretrained("texo", cfg);
    const tok = await PreTrainedTokenizer.from_pretrained("texo");
    predict = async (blob) => { const a = await texoPre(blob); const t = new Tensor("float32", a, [1, 1, S, S]); const r = await m.generate({ inputs: cat([t, t, t], 1), max_new_tokens: 512 }); return tok.batch_decode(r, { skip_special_tokens: true })[0].trim(); };
    items = await imageItems();
  }
  const t1 = performance.now();
  result.loadMs = Math.round(t1 - t0);
  result.downloadMs = firstProgress ? Math.round(lastProgress - firstProgress) : 0;
  result.sessionMs = Math.round(t1 - (lastProgress || t0));
  result.files = files;
  result.totalBytes = Object.values(files).reduce((a, b) => a + b, 0);
  result.network = globalThis.__benchNet;

  progress.textContent = "warm-up";
  const w0 = performance.now();
  await predict(items[0][1]);
  result.warmupMs = Math.round(performance.now() - w0);

  if (runItems) {
    result.items = [];
    for (const [id, input] of items) {
      progress.textContent = id;
      const s = performance.now(); let output = "", err = "";
      try { output = await predict(input); } catch (e) { err = String(e).slice(0, 200); }
      result.items.push({ id, ms: Math.round(performance.now() - s), output, err });
    }
  }
  progress.textContent = "done";
  show(result); document.title = "done";
} catch (e) {
  result.error = String(e?.stack || e); show(result); document.title = "failed";
}
// transformers.js writes to the Cache API after resolving the load; leaving
// the page too early aborts the write and the next "warm" run is not warm.
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
if (!result.error) result.cacheSettled = await cacheSettled(Object.keys(files));

// Persist (the server keeps rows in memory; GET api/bench reads them back).
try { await realFetch("api/bench", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ tier: "load", ...result }) }); } catch { /* static build: no collector */ }
if (queue.length > 1) {
  const next = new URL(location.href);
  next.searchParams.set("queue", queue.slice(1).join(";"));
  setTimeout(() => location.replace(next), 1500);
} else if (queue.length) document.title = "queue-done";

async function imageItems() {
  const list = await realFetch("bench-images.json").then((r) => r.json());
  const items = [];
  for (const it of list) items.push([it.id, await realFetch(it.image.replace(/^\//, "")).then((r) => r.blob())]);
  return items;
}
