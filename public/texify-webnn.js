// Texify through hand-built WebNN graphs from the graph catalog (family
// texify-420), instead of ONNX Runtime Web. Used by onnx-worker.js (the
// product) and by the bench pages.
//
// This module builds no graphs itself: it replays the catalog's two recipes
// (a Donut-Swin encoder that also emits the decoder's cross-attention K/V,
// and a decode-step graph that runs one greedy token per dispatch over static
// caches of length 384) through the catalog's generic loader, vendored at
// vendor/webnn-catalog/. The entry's files live under models/texify-webnn/;
// the two constants blobs (~597 MiB) are local copies or catalog URLs.
//
// Only the Core ML backend is accepted. Chromium silently falls back to a
// TFLite CPU path for off-the-record profiles and non-Apple hardware, about
// 50x slower; assertCoreMLFingerprint() turns that into a thrown error, which
// models.js remembers so the next load takes the ONNX Runtime path.

import {
  loadEntry, createEntryTensors, assertCoreMLFingerprint, autoregressive, checkOpSupport,
} from "./vendor/webnn-catalog/loader.js";
import { env, AutoTokenizer } from "./vendor/transformers/transformers.min.js";
import { MODEL_HOST, MODEL_PATH_TEMPLATE, WEBNN_CATALOG_BASE } from "./config.js";

export const TEXIFY_WEBNN_BASE = WEBNN_CATALOG_BASE
  ? `${WEBNN_CATALOG_BASE.replace(/\/$/, "")}/`
  : new URL("models/texify-webnn/", import.meta.url).href;

export function webnnAvailable() {
  return typeof navigator !== "undefined" && !!navigator.ml && typeof navigator.ml.createContext === "function";
}

/** float32 -> float16, round to nearest even; Float16Array where the runtime has it. */
export function toFloat16(f32) {
  if (typeof Float16Array !== "undefined") return Float16Array.from(f32);
  const out = new Uint16Array(f32.length);
  const u32 = new Uint32Array(f32.buffer, f32.byteOffset, f32.length);
  for (let i = 0; i < f32.length; i++) {
    const x = u32[i], sign = (x >>> 16) & 0x8000, exp = (x >>> 23) & 0xff;
    let mant = x & 0x7fffff;
    if (exp === 0xff) { out[i] = sign | 0x7c00 | (mant ? 0x200 : 0); continue; }
    const e = exp - 127 + 15;
    if (e >= 0x1f) { out[i] = sign | 0x7c00; continue; }
    if (e <= 0) {
      if (e < -10) { out[i] = sign; continue; }
      mant |= 0x800000;
      const shift = 14 - e, rem = mant & ((1 << shift) - 1), half = 1 << (shift - 1);
      let h = mant >>> shift;
      if (rem > half || (rem === half && (h & 1))) h++;
      out[i] = sign | h;
      continue;
    }
    let h = (e << 10) | (mant >>> 13);
    const rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem === 0x1000 && (h & 1))) h++;
    out[i] = sign | h;
  }
  return out;
}

const SIZE = 420;
const PIXELS = 3 * SIZE * SIZE;
const IMAGE_MEAN = [0.485, 0.456, 0.406];
const IMAGE_STD = [0.229, 0.224, 0.225];

/**
 * Donut 4.36 geometry (export_texify.py donut_preprocess). Shortest-edge to
 * 420, then thumbnail into 420×420, then center pad. Interpolation is canvas
 * work in donutPreprocess(); this is the integer size plan, which is what
 * the tests pin.
 */
export function donutResizePlan(w, h, size = SIZE) {
  let rw = w, rh = h;
  let first = null, thumb = null;
  if (Math.min(rh, rw) !== size) {
    if (rh < rw) { rh = size; rw = Math.max(1, Math.trunc(w * size / h)); }
    else { rw = size; rh = Math.max(1, Math.trunc(h * size / w)); }
    first = { w: rw, h: rh };
  }
  let tw = Math.min(rw, size), th = Math.min(rh, size);
  if (!(th === rh && tw === rw)) {
    if (rh > rw) tw = Math.max(1, Math.trunc(rw * th / rh));
    else if (rw > rh) th = Math.max(1, Math.trunc(rh * tw / rw));
    thumb = { w: tw, h: th };
    rw = tw; rh = th;
  }
  const top = Math.trunc((size - rh) / 2), left = Math.trunc((size - rw) / 2);
  return { first, thumb, pad: { top, left, bottom: size - rh - top, right: size - rw - left }, w: rw, h: rh };
}

function makeCanvas(w, h) {
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(w, h);
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  return c;
}

async function drawSized(src, w, h, quality) {
  const c = makeCanvas(w, h);
  const ctx = c.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = quality;
  ctx.drawImage(src, 0, 0, w, h);
  src.close?.();
  return createImageBitmap(c);
}

/** RGBA ImageData bytes -> NCHW float32, /255 then ImageNet mean/std. */
export function rgbaToNchw(data, size = SIZE) {
  const out = new Float32Array(3 * size * size);
  const plane = size * size;
  for (let i = 0, j = 0; j < plane; j++, i += 4) {
    out[j] = (data[i] / 255 - IMAGE_MEAN[0]) / IMAGE_STD[0];
    out[plane + j] = (data[i + 1] / 255 - IMAGE_MEAN[1]) / IMAGE_STD[1];
    out[2 * plane + j] = (data[i + 2] / 255 - IMAGE_MEAN[2]) / IMAGE_STD[2];
  }
  return out;
}

/**
 * Donut 4.36 preprocess in the consumer: shortest-edge BILINEAR to 420,
 * thumbnail BICUBIC into 420×420, center pad 0, /255, ImageNet. Returns
 * Float32Array(3*420*420) NCHW. Canvas cannot pick PIL's filters exactly
 * (`high` is the closest to bicubic); the geometry matches export_texify.py.
 */
export async function donutPreprocess(blob) {
  let bmp = await createImageBitmap(blob);
  const plan = donutResizePlan(bmp.width, bmp.height);
  if (plan.first) bmp = await drawSized(bmp, plan.first.w, plan.first.h, "low");
  if (plan.thumb) bmp = await drawSized(bmp, plan.thumb.w, plan.thumb.h, "high");
  const c = makeCanvas(SIZE, SIZE);
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.drawImage(bmp, plan.pad.left, plan.pad.top);
  bmp.close?.();
  return rgbaToNchw(ctx.getImageData(0, 0, SIZE, SIZE).data);
}

/**
 * Build the Texify graphs once. Resolves to {run, stats, close}:
 *   run(blob)  image Blob; Donut 4.36 preprocess (canvas), then greedy decode;
 *              resolves to the decoded LaTeX string (skip_special_tokens).
 * onProgress({file, loaded, total}) mirrors transformers.js's progress events.
 */
export async function createTexifyWebNN({ baseUrl = TEXIFY_WEBNN_BASE, onProgress = null, deviceType = "gpu", skipWarmup = false, preferLocalConstants = false, preserveModelSource = false } = {}) {
  if (!webnnAvailable()) throw new Error("WebNN is not available (navigator.ml missing)");
  // Same rule as onnx-worker.js: an http localModelPath never probes the
  // server for tokenizer.json (issue #3). Always point transformers.js at
  // this origin (or the static-build CDN) before AutoTokenizer runs — its
  // default remoteHost is Hugging Face, which 401s for these files.
  env.allowRemoteModels = true;
  env.allowLocalModels = false;
  if (preserveModelSource) {
    // The benchmark has already selected its remote model host.
  } else if (MODEL_HOST) {
    env.remoteHost = MODEL_HOST;
    env.remotePathTemplate = MODEL_PATH_TEMPLATE;
  } else {
    env.remoteHost = new URL("models/", import.meta.url).href;
    env.remotePathTemplate = "{model}";
  }
  const t0 = performance.now();
  const j = async (u) => { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: HTTP ${r.status}`); return r.json(); };
  const [entry, family] = await Promise.all([j(`${baseUrl}entry.json`), j(`${baseUrl}family.json`)]);
  const spec = family.contract.chaining.autoregressive;

  const ctx = await navigator.ml.createContext({ deviceType });
  const fp = assertCoreMLFingerprint(ctx);

  const manifest = await j(`${baseUrl}${entry.constants}`);
  if (preferLocalConstants) for (const c of Object.values(manifest.constants)) c.url = null;
  const graphNames = Object.keys(entry.graphs).filter((k) => k !== "chain");
  const recipes = Object.fromEntries(await Promise.all(graphNames.map(async (g) => [g, await j(`${baseUrl}${entry.graphs[g].recipe}`)])));
  for (const g of graphNames) {
    const s = checkOpSupport(recipes[g], ctx);
    if (s.missing.length) throw new Error(`WebNN in this browser lacks ${s.missing.join(", ")} (needed by the ${g} recipe)`);
  }
  const tokenizerP = AutoTokenizer.from_pretrained("texify");
  const totals = Object.fromEntries(Object.entries(manifest.constants).map(([k, c]) => [k, c.bytes]));
  const rig = await loadEntry(entry, baseUrl.replace(/\/$/, ""), ctx, {
    baseUrl: baseUrl.replace(/\/$/, ""),
    manifest,
    recipes,
    onProgress: onProgress
      ? (p) => { if (p.phase === "constants") onProgress({ file: manifest.constants[entry.graphs[p.graph].constants].file, loaded: p.bytesRead, total: totals[entry.graphs[p.graph].constants] }); }
      : undefined,
  });
  const tensors = await createEntryTensors(ctx, rig);
  const tokenizer = await tokenizerP;

  const encoderName = graphNames.find((g) => g !== spec.graph);
  const [imageInput] = Object.keys(rig.graphs[encoderName].inputs);
  const imageT = tensors.get(encoderName, imageInput);
  const encIn = tensors.inputsFor(encoderName), encOut = tensors.outputsFor(encoderName);
  const tokensView = new Int32Array(rig.graphs[spec.graph].outputs[spec.tokensOutput].shape.reduce((a, b) => a * b, 1));
  const maxNewTokens = spec.maxNewTokens ?? 384;
  const forcedEos = spec.forcedEosTokenId ?? spec.eos ?? 2;

  const stats = {
    backend: fp.backend, preferredInputLayout: fp.preferredInputLayout,
    graphs: rig.stats, entry: `${entry.family}/${entry.id}`, loaderVersion: entry.producedBy?.workbench ?? null,
    buildMs: +(performance.now() - t0).toFixed(1),
  };

  let last = null;
  const runPixels = async (f32) => {
    if (!(f32 instanceof Float32Array) || f32.length !== PIXELS)
      throw new Error("texify-webnn: expected the preprocessed Float32Array(3*420*420) NCHW");
    const t = performance.now();
    ctx.writeTensor(imageT, toFloat16(f32));
    ctx.dispatch(rig.graphs[encoderName].graph, encIn, encOut);
    const r = await autoregressive(ctx, rig, tensors, spec, { maxNewTokens, tokensView });
    if (!r.endedWithEos && r.tokens.length) r.tokens[r.tokens.length - 1] = forcedEos;
    last = { tokens: r.tokens, dispatches: r.dispatches, endedWithEos: r.endedWithEos, ms: +(performance.now() - t).toFixed(2) };
    return tokenizer.decode(r.tokens, { skip_special_tokens: true }).trim();
  };

  const run = async (blob) => runPixels(await donutPreprocess(blob));

  if (!skipWarmup) await runPixels(new Float32Array(PIXELS));

  return {
    run,
    runPixels,
    stats,
    lastRun: () => last,
    close: () => { for (const g of Object.values(rig.graphs)) g.graph.destroy?.(); for (const t of Object.values(tensors.tensors)) t.destroy?.(); },
  };
}
