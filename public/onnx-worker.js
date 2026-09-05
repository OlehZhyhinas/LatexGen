// ONNX models (IntelliTeX specialist, Texo, Texify) run here, off the main
// thread, so CPU inference never freezes the page — whether the request came
// from the UI or the Tab API. Runs are serialized per model; different
// models may run concurrently.
import {
  pipeline, env, VisionEncoderDecoderModel, PreTrainedTokenizer, Tensor, cat,
} from "./vendor/transformers/transformers.min.js";
import { MODEL_HOST, MODEL_PATH_TEMPLATE } from "./config.js";

// Weights come from this origin in the self-hosted build, and from a CDN in
// the static build. Both go through transformers.js's "remote" loader: with an
// http localModelPath its tokenizer existence check never probes the server,
// so pipelines came back without a tokenizer on a cold cache (issue #3).
// Paths are resolved against this worker's own URL so the app works under any
// deploy prefix (e.g. a GitHub Pages project path).
env.allowRemoteModels = true;
env.allowLocalModels = false;
if (MODEL_HOST) {
  env.remoteHost = MODEL_HOST;
  env.remotePathTemplate = MODEL_PATH_TEMPLATE;
} else {
  env.remoteHost = new URL("models/", import.meta.url).href;
  env.remotePathTemplate = "{model}";
}
env.backends.onnx.wasm.wasmPaths = new URL("vendor/ort/", import.meta.url).href;
env.backends.onnx.wasm.numThreads = 1; // multi-threaded ORT hung on load in testing

const SPECIALIST_PREFIX = "Convert natural-language math into a STRICT LaTeX equation\n";
const models = {};      // key -> async run(input) => string
const loading = {};     // key -> promise
const queues = {};      // key -> promise chain (serialize runs per model)

// ---- Texo preprocessing (ported from Texo-web) ----
const TEXO_MEAN = 0.7931, TEXO_STD = 0.1738, TEXO_SIZE = 384;
async function texoPreprocess(blob) {
  const bmp = await createImageBitmap(blob);
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = c.getContext("2d");
  ctx.fillStyle = "white"; ctx.fillRect(0, 0, c.width, c.height);
  ctx.drawImage(bmp, 0, 0);
  const { data, width, height } = ctx.getImageData(0, 0, c.width, c.height);
  const gray = new Uint8ClampedArray(width * height);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) gray[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
  let dark = 0; for (const v of gray) if (v < 200) dark++;
  if (dark >= gray.length - dark) for (let i = 0; i < gray.length; i++) gray[i] = 255 - gray[i];
  let min = 255, max = 0; for (const v of gray) { if (v < min) min = v; if (v > max) max = v; }
  let x0 = width, y0 = height, x1 = -1, y1 = -1;
  if (max > min) for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const n = ((gray[y * width + x] - min) / (max - min)) * 255;
    if (n < 200) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  }
  if (x1 < x0 || y1 < y0) { x0 = 0; y0 = 0; x1 = width - 1; y1 = height - 1; }
  const cw = Math.max(1, x1 - x0), ch = Math.max(1, y1 - y0);
  const gImg = new ImageData(width, height);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) { gImg.data[p] = gImg.data[p + 1] = gImg.data[p + 2] = gray[i]; gImg.data[p + 3] = 255; }
  const gc = new OffscreenCanvas(width, height); gc.getContext("2d").putImageData(gImg, 0, 0);
  const scale = TEXO_SIZE / Math.min(ch, cw);
  let nw = Math.round(cw * scale), nh = Math.round(ch * scale);
  if (nw > TEXO_SIZE || nh > TEXO_SIZE) { const r = Math.min(TEXO_SIZE / nw, TEXO_SIZE / nh); nw = Math.round(nw * r); nh = Math.round(nh * r); }
  const out = new OffscreenCanvas(TEXO_SIZE, TEXO_SIZE); const octx = out.getContext("2d");
  octx.fillStyle = "black"; octx.fillRect(0, 0, TEXO_SIZE, TEXO_SIZE);
  octx.drawImage(gc, x0, y0, cw, ch, Math.floor((TEXO_SIZE - nw) / 2), Math.floor((TEXO_SIZE - nh) / 2), nw, nh);
  const od = octx.getImageData(0, 0, TEXO_SIZE, TEXO_SIZE).data;
  const arr = new Float32Array(TEXO_SIZE * TEXO_SIZE);
  for (let i = 0, p = 0; i < arr.length; i++, p += 4) arr[i] = (od[p] / 255 - TEXO_MEAN) / TEXO_STD;
  return arr;
}

async function load(key, cfg, reqId) {
  const progress_callback = (p) => {
    if (p.status === "progress" && p.total) self.postMessage({ type: "progress", key, file: p.file, loaded: p.loaded, total: p.total });
  };
  if (key === "intellitex") {
    const p = await pipeline("text2text-generation", "intellitex", { ...cfg, progress_callback });
    await p(`${SPECIALIST_PREFIX}x squared`, { max_new_tokens: 16 }); // warm-up (shader compile etc.)
    models.intellitex = async (text) => (await p(SPECIALIST_PREFIX + text, { max_new_tokens: 256 }))[0]?.generated_text?.trim() ?? "";
  } else if (key === "texify") {
    const p = await pipeline("image-to-text", "texify", { ...cfg, progress_callback });
    models.texify = async (blob) => (await p(blob, { max_new_tokens: 384 }))[0]?.generated_text?.trim() ?? "";
  } else if (key === "texo" && cfg?.device === "webnn") {
    // Hand-built WebNN graphs from the graph catalog (texo-webnn.js): 23 ms per
    // image on Core ML against 0.77 s through ONNX Runtime, identical tokens.
    // Throws on browsers without WebNN or with a non-Core ML backend; models.js
    // remembers that and the next load takes the ONNX Runtime path below.
    const { createTexoWebNN } = await import("./texo-webnn.js");
    const texo = await createTexoWebNN({
      onProgress: (p) => self.postMessage({ type: "progress", key, file: p.file, loaded: p.loaded, total: p.total }),
    });
    models.texo = async (blob) => texo.run(await texoPreprocess(blob));
  } else if (key === "texo") {
    const model = await VisionEncoderDecoderModel.from_pretrained("texo", { dtype: "fp32", progress_callback });
    const tokenizer = await PreTrainedTokenizer.from_pretrained("texo");
    models.texo = async (blob) => {
      const arr = await texoPreprocess(blob);
      const t = new Tensor("float32", arr, [1, 1, TEXO_SIZE, TEXO_SIZE]);
      const outputs = await model.generate({ inputs: cat([t, t, t], 1), max_new_tokens: 512 });
      return tokenizer.batch_decode(outputs, { skip_special_tokens: true })[0].trim();
    };
  } else throw new Error(`unknown model ${key}`);
}

self.onmessage = async ({ data }) => {
  const { type, id, key } = data;
  try {
    if (type === "load") {
      loading[key] ??= load(key, data.cfg, id);
      await loading[key];
      self.postMessage({ type: "loaded", id, key });
    } else if (type === "run") {
      if (!models[key]) throw new Error(`${key} not loaded`);
      // serialize per model
      const prev = queues[key] ?? Promise.resolve();
      const run = prev.catch(() => {}).then(() => models[key](data.input));
      queues[key] = run;
      const output = await run;
      self.postMessage({ type: "result", id, key, output });
    }
  } catch (err) {
    if (type === "load") delete loading[key];
    self.postMessage({ type: "error", id, key, error: String(err?.message || err) });
  }
};
