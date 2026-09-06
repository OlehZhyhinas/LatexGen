// Texo through hand-built WebNN graphs from the graph catalog (webnn-catalog,
// family texo-384), instead of ONNX Runtime Web. Used by onnx-worker.js (the
// product) and by the bench pages.
//
// This module builds no graphs itself: it replays the catalog's two recipes
// (an HGNetv2 encoder that also emits the decoder's cross-attention K/V, and
// a decode graph that runs 16 greedy steps per dispatch over static caches)
// through the catalog's generic loader, vendored at vendor/webnn-catalog/. The
// entry's files and the two constants blobs live under models/texo-webnn/ next
// to the other model weights; manifest URLs, once the blobs are published,
// take precedence over the local copies.
//
// Only the Core ML backend is accepted. Chromium silently falls back to a
// TFLite CPU path for off-the-record profiles and non-Apple hardware, about
// 50x slower; assertCoreMLFingerprint() turns that into a thrown error, which
// models.js remembers so the next load takes the ONNX Runtime path.

import {
  loadEntry, createEntryTensors, assertCoreMLFingerprint, autoregressive, checkOpSupport,
} from "./vendor/webnn-catalog/loader.js";
import TexoTokenizer from "./models/texo-webnn/tokenizer.js";
import { WEBNN_CATALOG_BASE } from "./config.js";

export const TEXO_WEBNN_BASE = WEBNN_CATALOG_BASE
  ? `${WEBNN_CATALOG_BASE.replace(/\/$/, "")}/`
  : new URL("models/texo-webnn/", import.meta.url).href;

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

/**
 * Build the Texo graphs once. Resolves to {run, stats, close}:
 *   run(normalized)  normalized = Float32Array(384*384), LatexGen's texoPreprocess
 *                    output; resolves to the decoded LaTeX string (same
 *                    decoding as transformers' batch_decode(skip_special_tokens)
 *                    plus clean_up_tokenization, which is what the ORT path
 *                    returns), and .tokens on the returned string's `last` field.
 * onProgress({file, loaded, total}) mirrors transformers.js's progress events.
 */
export async function createTexoWebNN({ baseUrl = TEXO_WEBNN_BASE, onProgress = null, deviceType = "gpu", skipWarmup = false, preferLocalConstants = false } = {}) {
  if (!webnnAvailable()) throw new Error("WebNN is not available (navigator.ml missing)");
  const t0 = performance.now();
  const j = async (u) => { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: HTTP ${r.status}`); return r.json(); };
  const [entry, family] = await Promise.all([j(`${baseUrl}entry.json`), j(`${baseUrl}family.json`)]);
  const spec = family.contract.chaining.autoregressive;

  const ctx = await navigator.ml.createContext({ deviceType });
  const fp = assertCoreMLFingerprint(ctx); // throws on the TFLite/XNNPACK fallback

  const manifest = await j(`${baseUrl}${entry.constants}`);
  if (preferLocalConstants) for (const c of Object.values(manifest.constants)) c.url = null;
  const graphNames = Object.keys(entry.graphs).filter((k) => k !== "chain");
  const recipes = Object.fromEntries(await Promise.all(graphNames.map(async (g) => [g, await j(`${baseUrl}${entry.graphs[g].recipe}`)])));
  for (const g of graphNames) {
    const s = checkOpSupport(recipes[g], ctx);
    if (s.missing.length) throw new Error(`WebNN in this browser lacks ${s.missing.join(", ")} (needed by the ${g} recipe)`);
  }
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
  const tok = new TexoTokenizer(await j(`${baseUrl}tokenizer/tokenizer.json`));

  const encoderName = graphNames.find((g) => g !== spec.graph);
  const [imageInput] = Object.keys(rig.graphs[encoderName].inputs);
  const imageT = tensors.get(encoderName, imageInput);
  const encIn = tensors.inputsFor(encoderName), encOut = tensors.outputsFor(encoderName);
  const tokensView = new Int32Array(rig.graphs[spec.graph].outputs[spec.tokensOutput].shape.reduce((a, b) => a * b, 1));
  const maxNewTokens = family.contract.chaining.autoregressive.maxNewTokens ?? 512;

  const stats = {
    backend: fp.backend, preferredInputLayout: fp.preferredInputLayout,
    graphs: rig.stats, entry: `${entry.family}/${entry.id}`, loaderVersion: entry.producedBy?.workbench ?? null,
    buildMs: +(performance.now() - t0).toFixed(1),
  };

  let last = null;
  const run = async (normalized) => {
    if (!(normalized instanceof Float32Array) || normalized.length !== 384 * 384)
      throw new Error("texo-webnn: expected the preprocessed Float32Array(384*384)");
    const t = performance.now();
    ctx.writeTensor(imageT, toFloat16(normalized));
    ctx.dispatch(rig.graphs[encoderName].graph, encIn, encOut);
    const r = await autoregressive(ctx, rig, tensors, spec, { maxNewTokens, tokensView });
    last = { tokens: r.tokens, dispatches: r.dispatches, endedWithEos: r.endedWithEos, ms: +(performance.now() - t).toFixed(2) };
    return tok.decode(r.tokens); // skips <s>/</s>/<pad>, applies clean_up_tokenization
  };

  // one warm-up: the first dispatch of a freshly built Core ML graph is slow
  if (!skipWarmup) await run(new Float32Array(384 * 384).fill(-4.5628));

  return {
    run,
    stats,
    lastRun: () => last,
    close: () => { for (const g of Object.values(rig.graphs)) g.graph.destroy?.(); for (const t of Object.values(tensors.tensors)) t.destroy?.(); },
  };
}
