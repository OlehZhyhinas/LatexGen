// Level A runtime benchmark: for each in-browser model, try every
// (device, dtype) the runtime offers, measure load + per-item latency, and
// keep the outputs so correctness can be judged afterwards.
import { pipeline, env, VisionEncoderDecoderModel, PreTrainedTokenizer, Tensor, cat } from "/vendor/transformers/transformers.min.js";
env.allowRemoteModels = false; env.allowLocalModels = true; env.localModelPath = "/models/";
env.backends.onnx.wasm.wasmPaths = "/vendor/ort/";
env.backends.onnx.wasm.numThreads = 1;

const progress = document.getElementById("progress"), logEl = document.getElementById("log");
const log = (m, c = "") => { const d = document.createElement("div"); d.className = c; d.textContent = m; logEl.appendChild(d); };
const post = (row) => fetch("/api/bench", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(row) });
const withTimeout = (p, ms, label) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(`${label} timeout ${ms}ms`)), ms))]);
const hasGpu = !!navigator.gpu;

const TEXT_ITEMS = [
  ["basel", "the sum from n equals 1 to infinity of 1 over n squared equals pi squared over 6"],
  ["quadratic", "the quadratic formula: x equals minus b plus or minus the square root of b squared minus 4ac, all over 2a"],
  ["schrodinger", "i h bar partial psi over partial t equals minus h bar squared over 2m laplacian psi plus V psi"],
];
const images = (await fetch("/bench-images.json").then((r) => r.json())).filter((i) => ["quadratic", "gaussian", "schrodinger", "newton-text"].includes(i.id));
const blobs = new Map();
for (const it of images) blobs.set(it.id, await fetch(it.image).then((r) => r.blob()));
const PREFIX = "Convert natural-language math into a STRICT LaTeX equation\n";

// ---- Texo custom preprocessing (as in app.js) ----
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
  const out = new OffscreenCanvas(S, S); const o = out.getContext("2d"); o.fillStyle = "black"; o.fillRect(0, 0, S, S);
  o.drawImage(gc, x0, y0, cw, ch, Math.floor((S - nw) / 2), Math.floor((S - nh) / 2), nw, nh);
  const od = o.getImageData(0, 0, S, S).data; const arr = new Float32Array(S * S);
  for (let i = 0, p = 0; i < arr.length; i++, p += 4) arr[i] = (od[p] / 255 - TEXO_MEAN) / TEXO_STD; return arr;
}

async function runConfig(model, cfg) {
  const tag = `${model}:${cfg.device}:${cfg.dtype}`;
  progress.textContent = `loading ${tag}…`;
  const t0 = performance.now();
  let predict, kind;
  try {
    if (model.startsWith("intellitex")) {
      const p = await withTimeout(pipeline("text2text-generation", model, { device: cfg.device, dtype: cfg.dtype }), 90000, "load");
      predict = async (text) => (await p(PREFIX + text, { max_new_tokens: 128 }))[0].generated_text.trim(); kind = "text";
    } else if (model === "texify") {
      const p = await withTimeout(pipeline("image-to-text", "texify", { device: cfg.device, dtype: cfg.dtype }), 90000, "load");
      predict = async (blob) => { const u = URL.createObjectURL(blob); try { return (await p(u, { max_new_tokens: 256 }))[0].generated_text.trim(); } finally { URL.revokeObjectURL(u); } }; kind = "image";
    } else {
      const m = await withTimeout(VisionEncoderDecoderModel.from_pretrained("texo", { device: cfg.device, dtype: cfg.dtype }), 90000, "load");
      const tok = await PreTrainedTokenizer.from_pretrained("texo");
      predict = async (blob) => { const a = await texoPre(blob); const t = new Tensor("float32", a, [1, 1, S, S]); const out = await m.generate({ inputs: cat([t, t, t], 1), max_new_tokens: 256 }); return tok.batch_decode(out, { skip_special_tokens: true })[0].trim(); }; kind = "image";
    }
  } catch (e) {
    log(`${tag}  LOAD FAILED: ${String(e).slice(0, 140)}`, "err");
    await post({ approach: tag, item: "__load__", tier: "runtime", output: "", ms: -1, err: String(e).slice(0, 300) });
    return;
  }
  const loadMs = Math.round(performance.now() - t0);
  const items = kind === "text" ? TEXT_ITEMS : images.map((i) => [i.id, blobs.get(i.id)]);
  try { await withTimeout(predict(items[0][1]), 60000, "warmup"); } catch (e) { log(`${tag}  WARMUP FAILED: ${String(e).slice(0, 140)}`, "err"); await post({ approach: tag, item: "__load__", tier: "runtime", output: "", ms: loadMs, err: "warmup: " + String(e).slice(0, 300) }); return; }
  await post({ approach: tag, item: "__load__", tier: "runtime", output: "", ms: loadMs });
  log(`${tag}  loaded ${loadMs}ms`);
  for (const [id, input] of items) {
    progress.textContent = `${tag} · ${id}`;
    const t = performance.now(); let output = "", err = "";
    try { output = await withTimeout(predict(input), 60000, "predict"); } catch (e) { err = String(e).slice(0, 200); }
    const ms = Math.round(performance.now() - t);
    await post({ approach: tag, item: id, tier: "runtime", output, ms, err });
    log(`  ${id} ${ms}ms → ${(output || err).slice(0, 80)}`, err ? "err" : "");
  }
}

// One config per page load: a failed session poisons the ONNX runtime for the
// rest of the page, so the harness runs config i, then navigates to i+1.
const gpu = (...c) => (hasGpu ? c : []);
const plan = [
  ["intellitex", { device: "wasm", dtype: "q8" }], ["intellitex", { device: "wasm", dtype: "q4" }],
  ...gpu(["intellitex", { device: "webgpu", dtype: "q4" }], ["intellitex", { device: "webgpu", dtype: "fp16" }]),
  ["intellitex-fused", { device: "wasm", dtype: "q8" }],
  ["texo", { device: "wasm", dtype: "fp32" }], ["texo", { device: "wasm", dtype: "q8" }], ["texo", { device: "wasm", dtype: "fp16" }],
  ...gpu(["texo", { device: "webgpu", dtype: "fp32" }], ["texo", { device: "webgpu", dtype: "fp16" }]),
  ["texify", { device: "wasm", dtype: "q8" }], ["texify", { device: "wasm", dtype: "q4" }],
  ...gpu(["texify", { device: "webgpu", dtype: "fp16" }], ["texify", { device: "webgpu", dtype: "q4" }]),
];
const params = new URLSearchParams(location.search);
const i = Number(params.get("i") ?? "0");
// ?only=3,4 runs just those config indices (no reset, no chaining past them)
const only = params.get("only") ? params.get("only").split(",").map(Number) : null;
if (i === 0 && !only) await fetch("/api/bench", { method: "DELETE" });
const nextIndex = (from) => { let n = from + 1; if (only) while (n < plan.length && !only.includes(n)) n++; return n; };
if (i >= plan.length) {
  await post({ approach: "__done__", item: "__done__", tier: "runtime", output: "", ms: 0 });
  progress.textContent = "DONE"; log("DONE", "ok");
} else if (only && !only.includes(i)) {
  location.href = `/bench-runtime.html?i=${nextIndex(i)}&only=${params.get("only")}`;
} else {
  const [model, cfg] = plan[i];
  log(`[${i + 1}/${plan.length}] ${model} ${JSON.stringify(cfg)}`);
  try { await runConfig(model, cfg); } catch (e) { log(`crashed: ${e}`, "err"); }
  location.href = `/bench-runtime.html?i=${nextIndex(i)}${only ? `&only=${params.get("only")}` : ""}`;
}
