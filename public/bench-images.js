// Image-OCR benchmark: Texify (current, ~305MB int8) vs Texo (~77MB fp32),
// both fully in-browser via transformers.js. Posts rows to /api/bench.
// The vendored transformers.js, like bench-runtime.js: the page's CSP is
// script-src 'self', so the CDN import this page started with no longer loads.
import {
  pipeline, env, VisionEncoderDecoderModel, PreTrainedTokenizer, Tensor, cat,
} from "./vendor/transformers/transformers.min.js";

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = new URL("models/", import.meta.url).href;
env.backends.onnx.wasm.wasmPaths = new URL("vendor/ort/", import.meta.url).href;
env.backends.onnx.wasm.numThreads = 1;

const progress = document.getElementById("progress");
const logEl = document.getElementById("log");
const log = (msg, cls = "") => {
  const d = document.createElement("div");
  d.className = cls;
  d.innerHTML = msg;
  logEl.appendChild(d);
  d.scrollIntoView({ block: "end" });
};
const post = (row) =>
  fetch("api/bench", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(row) });

const items = await fetch("bench-images.json").then((r) => r.json());
await fetch("api/bench", { method: "DELETE" });
const blobs = new Map();
for (const it of items) blobs.set(it.id, await fetch(it.image).then((r) => r.blob()));

// ---------------- Texify (stock pipeline, as in app.js) ----------------
async function runTexify() {
  progress.textContent = "loading Texify…";
  const t0 = performance.now();
  const texify = await pipeline("image-to-text", "texify", { dtype: "q8" });
  const loadMs = Math.round(performance.now() - t0);
  await texify(URL.createObjectURL(blobs.get(items[0].id)), { max_new_tokens: 16 }); // warm-up
  await post({ approach: "texify", item: "__load__", tier: "meta", output: "", ms: loadMs });
  log(`texify loaded in ${loadMs} ms (cache-warm)`);
  for (const it of items) {
    progress.textContent = `texify · ${it.id}`;
    const url = URL.createObjectURL(blobs.get(it.id));
    const t = performance.now();
    let output = "", err = "";
    try {
      const out = await texify(url, { max_new_tokens: 384 });
      output = (out[0]?.generated_text ?? "").trim();
    } catch (e) { err = String(e); }
    const ms = Math.round(performance.now() - t);
    URL.revokeObjectURL(url);
    await post({ approach: "texify", item: it.id, tier: it.tier, output, ms, err });
    log(`<img src="${it.image}"> texify ${it.id} ${ms}ms → <code>${escapeHtml(output || err).slice(0, 90)}</code>`, err ? "err" : "");
  }
}

// ---------------- Texo (custom preprocessing, ported from Texo-web) ----------------
const UNIMERNET_MEAN = 0.7931, UNIMERNET_STD = 0.1738, SIZE = 384;

async function texoPreprocess(blob) {
  const bmp = await createImageBitmap(blob);
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = c.getContext("2d");
  ctx.fillStyle = "white"; ctx.fillRect(0, 0, c.width, c.height);
  ctx.drawImage(bmp, 0, 0);
  const { data, width, height } = ctx.getImageData(0, 0, c.width, c.height);

  // grayscale
  const gray = new Uint8ClampedArray(width * height);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    gray[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
  }
  // invert if dark background (heuristic from Texo-web)
  let dark = 0;
  for (const v of gray) if (v < 200) dark++;
  if (dark >= gray.length - dark) for (let i = 0; i < gray.length; i++) gray[i] = 255 - gray[i];

  // crop margin: normalize to [0,255], bbox of pixels < 200
  let min = 255, max = 0;
  for (const v of gray) { if (v < min) min = v; if (v > max) max = v; }
  let x0 = width, y0 = height, x1 = -1, y1 = -1;
  if (max > min) {
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const n = ((gray[y * width + x] - min) / (max - min)) * 255;
      if (n < 200) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    }
  }
  if (x1 < x0 || y1 < y0) { x0 = 0; y0 = 0; x1 = width - 1; y1 = height - 1; }
  const cw = Math.max(1, x1 - x0), ch = Math.max(1, y1 - y0);

  // put grayscale back on a canvas so we can resize via drawImage
  const gImg = new ImageData(width, height);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    gImg.data[p] = gImg.data[p + 1] = gImg.data[p + 2] = gray[i]; gImg.data[p + 3] = 255;
  }
  const gc = new OffscreenCanvas(width, height);
  gc.getContext("2d").putImageData(gImg, 0, 0);

  // resize: scale so min side = 384, then fit within 384x384, center-pad black
  const scale = SIZE / Math.min(ch, cw);
  let nw = Math.round(cw * scale), nh = Math.round(ch * scale);
  if (nw > SIZE || nh > SIZE) {
    const r = Math.min(SIZE / nw, SIZE / nh);
    nw = Math.round(nw * r); nh = Math.round(nh * r);
  }
  const out = new OffscreenCanvas(SIZE, SIZE);
  const octx = out.getContext("2d");
  octx.fillStyle = "black"; octx.fillRect(0, 0, SIZE, SIZE);
  octx.drawImage(gc, x0, y0, cw, ch, Math.floor((SIZE - nw) / 2), Math.floor((SIZE - nh) / 2), nw, nh);
  const od = octx.getImageData(0, 0, SIZE, SIZE).data;
  const arr = new Float32Array(SIZE * SIZE);
  for (let i = 0, p = 0; i < arr.length; i++, p += 4) arr[i] = (od[p] / 255 - UNIMERNET_MEAN) / UNIMERNET_STD;
  return arr;
}

async function runTexo() {
  progress.textContent = "loading Texo…";
  const t0 = performance.now();
  const model = await VisionEncoderDecoderModel.from_pretrained("texo", { dtype: "fp32" });
  const tokenizer = await PreTrainedTokenizer.from_pretrained("texo");
  const loadMs = Math.round(performance.now() - t0);
  const predict = async (blob) => {
    const arr = await texoPreprocess(blob);
    const t = new Tensor("float32", arr, [1, 1, SIZE, SIZE]);
    const pixel_values = cat([t, t, t], 1);
    const outputs = await model.generate({ inputs: pixel_values, max_new_tokens: 512 });
    return tokenizer.batch_decode(outputs, { skip_special_tokens: true })[0].trim();
  };
  await predict(blobs.get(items[0].id)); // warm-up
  await post({ approach: "texo", item: "__load__", tier: "meta", output: "", ms: loadMs });
  log(`texo loaded in ${loadMs} ms (cache-warm)`);
  for (const it of items) {
    progress.textContent = `texo · ${it.id}`;
    const t = performance.now();
    let output = "", err = "";
    try { output = await predict(blobs.get(it.id)); } catch (e) { err = String(e); }
    const ms = Math.round(performance.now() - t);
    await post({ approach: "texo", item: it.id, tier: it.tier, output, ms, err });
    log(`<img src="${it.image}"> texo ${it.id} ${ms}ms → <code>${escapeHtml(output || err).slice(0, 90)}</code>`, err ? "err" : "");
  }
}

// ---------------- Texo through the graph catalog (WebNN / Core ML) ----------------
async function runTexoWebNN() {
  if (!navigator.ml) { log("texo-webnn: navigator.ml missing, skipped", "err"); return; }
  progress.textContent = "loading Texo (WebNN graphs)…";
  const { createTexoWebNN } = await import("./texo-webnn.js");
  const t0 = performance.now();
  const texo = await createTexoWebNN();
  const loadMs = Math.round(performance.now() - t0);
  await post({ approach: "texo-webnn", item: "__load__", tier: "meta", output: "", ms: loadMs });
  log(`texo-webnn loaded in ${loadMs} ms (${texo.stats.backend}; encoder ${texo.stats.graphs.encoder.ops} ops, decoder ${texo.stats.graphs.decoder.ops} ops)`);
  for (const it of items) {
    progress.textContent = `texo-webnn · ${it.id}`;
    const t = performance.now();
    let output = "", err = "";
    try { output = await texo.run(await texoPreprocess(blobs.get(it.id))); } catch (e) { err = String(e); }
    const ms = Math.round(performance.now() - t);
    await post({ approach: "texo-webnn", item: it.id, tier: it.tier, output, ms, err });
    log(`<img src="${it.image}"> texo-webnn ${it.id} ${ms}ms → <code>${escapeHtml(output || err).slice(0, 90)}</code>`, err ? "err" : "");
  }
}

// ---------------- Texify through the graph catalog (WebNN / Core ML) ----------------
async function runTexifyWebNN() {
  if (!navigator.ml) { log("texify-webnn: navigator.ml missing, skipped", "err"); return; }
  progress.textContent = "loading Texify (WebNN graphs)…";
  const { createTexifyWebNN } = await import("./texify-webnn.js");
  const t0 = performance.now();
  const texify = await createTexifyWebNN();
  const loadMs = Math.round(performance.now() - t0);
  await post({ approach: "texify-webnn", item: "__load__", tier: "meta", output: "", ms: loadMs });
  log(`texify-webnn loaded in ${loadMs} ms (${texify.stats.backend}; encoder ${texify.stats.graphs.encoder.ops} ops, decoder ${texify.stats.graphs.decoder.ops} ops)`);
  for (const it of items) {
    progress.textContent = `texify-webnn · ${it.id}`;
    const t = performance.now();
    let output = "", err = "";
    try { output = await texify.run(blobs.get(it.id)); } catch (e) { err = String(e); }
    const ms = Math.round(performance.now() - t);
    await post({ approach: "texify-webnn", item: it.id, tier: it.tier, output, ms, err });
    log(`<img src="${it.image}"> texify-webnn ${it.id} ${ms}ms → <code>${escapeHtml(output || err).slice(0, 90)}</code>`, err ? "err" : "");
  }
}

function escapeHtml(s) { return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }

try {
  // ?only=texo-webnn (or texo, texify, texify-webnn; comma-separated) restricts the run
  const only = new URLSearchParams(location.search).get("only")?.split(",");
  const want = (k) => !only || only.includes(k);
  if (want("texify")) await runTexify();
  if (want("texo")) await runTexo();
  if (want("texo-webnn")) await runTexoWebNN();
  if (want("texify-webnn")) await runTexifyWebNN();
  await post({ approach: "__done__", item: "__done__", tier: "meta", output: "", ms: 0 });
  progress.textContent = "DONE";
  log("DONE", "ok");
} catch (e) {
  progress.textContent = `FAILED: ${e}`;
  log(String(e), "err");
  await post({ approach: "__error__", item: "__error__", tier: "meta", output: String(e), ms: 0 });
}
