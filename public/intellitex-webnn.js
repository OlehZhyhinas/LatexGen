// IntelliTeX through hand-built WebNN graphs from the graph catalog (family
// intellitex-t5-220m), instead of ONNX Runtime Web. Used by onnx-worker.js
// (the product) and by the bench pages.
//
// This module builds no graphs itself: it replays the catalog's six recipes
// (T5 encoder + decode-step at padded lengths 32/64/128) through the catalog's
// generic loader. Compiling three decode graphs in one MLContext kills Chrome,
// so each length bucket gets its own context.
//
// Only the Core ML backend is accepted. Chromium silently falls back to a
// TFLite CPU path for off-the-record profiles, about 50x slower;
// assertCoreMLFingerprint() turns that into a thrown error, which models.js
// remembers so the next load takes the ONNX Runtime path.

import {
  loadEntry, createEntryTensors, assertCoreMLFingerprint, autoregressive, checkOpSupport,
} from "./vendor/webnn-catalog/loader.js";
import { env, AutoTokenizer } from "./vendor/transformers/transformers.min.js";
import { MODEL_HOST, MODEL_PATH_TEMPLATE, WEBNN_CATALOG_BASE } from "./config.js";

export const INTELLITEX_WEBNN_BASE = WEBNN_CATALOG_BASE
  ? `${WEBNN_CATALOG_BASE.replace(/\/$/, "")}/`
  : new URL("models/intellitex-webnn/", import.meta.url).href;

export const SPECIALIST_PREFIX = "Convert natural-language math into a STRICT LaTeX equation\n";

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

export function bucketFor(n, buckets = [32, 64, 128]) {
  for (const b of buckets) if (n <= b) return b;
  throw new Error(`intellitex-webnn: prompt length ${n} exceeds largest bucket ${buckets[buckets.length - 1]}`);
}

export function padIds(ids, L, padId = 0, maskValue = -1e4) {
  const out = new Int32Array(L);
  const bias32 = new Float32Array(L);
  const n = Math.min(ids.length, L);
  for (let i = 0; i < n; i++) out[i] = ids[i];
  for (let i = n; i < L; i++) { out[i] = padId; bias32[i] = maskValue; }
  return { ids: out, padBias: toFloat16(bias32) };
}

/**
 * Reuse one in-flight/full constants buffer for every graph naming the same
 * manifest key. release() deliberately makes this eager-build-only: a future
 * lazy bucket must create a new resolver (and therefore refetch after release).
 */
export function createSharedConstantSource(baseUrl, fetchImpl = fetch, sharedKeys = null) {
  const base = baseUrl.replace(/\/$/, "");
  const sources = new Map();
  const makeSource = (record) => {
    const url = record.url ?? `${base}/${record.file}`;
    let bufferPromise = null;
    return {
      kind: "http-shared",
      totalBytes: record.bytes,
      ranges: [{ byteOffset: 0, byteLength: record.bytes }],
      async fetchRange(byteOffset, byteLength) {
        bufferPromise ??= fetchImpl(url).then((response) => {
          if (!response.ok) throw new Error(`${url} -> HTTP ${response.status}`);
          return response.arrayBuffer();
        });
        return new Uint8Array(await bufferPromise).subarray(byteOffset, byteOffset + byteLength);
      },
      release() { bufferPromise = null; },
    };
  };
  const resolve = (key, record) => {
    if (sharedKeys && !sharedKeys.has(key)) return makeSource(record);
    if (!sources.has(key)) sources.set(key, makeSource(record));
    return sources.get(key);
  };
  resolve.release = () => {
    for (const source of sources.values()) source.release();
    sources.clear();
  };
  return resolve;
}

function pointTokenizerAtOrigin() {
  env.allowRemoteModels = true;
  env.allowLocalModels = false;
  if (MODEL_HOST) {
    env.remoteHost = MODEL_HOST;
    env.remotePathTemplate = MODEL_PATH_TEMPLATE;
  } else {
    env.remoteHost = new URL("models/", import.meta.url).href;
    env.remotePathTemplate = "{model}";
  }
}

/**
 * Build the IntelliTeX graphs once (one Core ML context per length bucket).
 * Resolves to {run, stats, close}:
 *   run(text)  plain-English math (no prefix); the specialist prefix is added
 *              here, matching onnx-worker.js. Resolves to decoded LaTeX.
 */
export async function createIntelliTeXWebNN({ baseUrl = INTELLITEX_WEBNN_BASE, onProgress = null, deviceType = "gpu", skipWarmup = false, preferLocalConstants = false, preserveModelSource = false, shareConstants = true } = {}) {
  if (!webnnAvailable()) throw new Error("WebNN is not available (navigator.ml missing)");
  if (!preserveModelSource) pointTokenizerAtOrigin();
  const t0 = performance.now();
  const j = async (u) => { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: HTTP ${r.status}`); return r.json(); };
  const [entry, family] = await Promise.all([j(`${baseUrl}entry.json`), j(`${baseUrl}family.json`)]);
  const buckets = Object.keys(family.contract.chaining.buckets).map(Number).sort((a, b) => a - b);
  const ar = family.contract.chaining.autoregressive;

  const tokenizerP = AutoTokenizer.from_pretrained("intellitex");
  const manifest = await j(`${baseUrl}${entry.constants}`);
  if (preferLocalConstants) for (const c of Object.values(manifest.constants)) c.url = null;
  const graphNames = Object.keys(entry.graphs).filter((k) => k !== "chain");
  const recipes = Object.fromEntries(await Promise.all(graphNames.map(async (g) => [g, await j(`${baseUrl}${entry.graphs[g].recipe}`)])));

  const totals = Object.fromEntries(Object.entries(manifest.constants).map(([k, c]) => [k, c.bytes]));
  const loaded = {};
  const graphStats = {};
  let fp = null;
  const keyUses = graphNames.reduce((counts, name) => {
    const key = entry.graphs[name].constants;
    counts.set(key, (counts.get(key) ?? 0) + 1);
    return counts;
  }, new Map());
  const sharedKeys = new Set([...keyUses].filter(([, uses]) => uses > 1).map(([key]) => key));
  const constants = shareConstants ? createSharedConstantSource(baseUrl, fetch, sharedKeys) : baseUrl.replace(/\/$/, "");

  try {
    for (const L of buckets) {
      const names = [family.contract.chaining.buckets[String(L)].encoder, family.contract.chaining.buckets[String(L)].decode];
      const ctx = await navigator.ml.createContext({ deviceType });
      fp = assertCoreMLFingerprint(ctx);
      if (L === buckets[0]) {
        for (const g of graphNames) {
          const s = checkOpSupport(recipes[g], ctx);
          if (s.missing.length) throw new Error(`WebNN in this browser lacks ${s.missing.join(", ")} (needed by the ${g} recipe)`);
        }
      }
      const rig = await loadEntry(entry, constants, ctx, {
        baseUrl: baseUrl.replace(/\/$/, ""),
        manifest,
        recipes,
        only: names,
        onProgress: onProgress
          ? (p) => {
            if (p.phase === "constants") {
              const rec = manifest.constants[entry.graphs[p.graph].constants];
              onProgress({ file: rec.file, loaded: p.bytesRead, total: totals[entry.graphs[p.graph].constants] });
            }
          }
          : undefined,
      });
      rig.chain = rig.chain.filter((l) => rig.graphs[l.from.graph] && rig.graphs[l.to.graph]);
      const tensors = await createEntryTensors(ctx, rig);
      const spec = { ...ar, graph: names[1] };
      const encoderName = names[0];
      const idsT = tensors.get(encoderName, "input_ids");
      const encPad = tensors.get(encoderName, "pad_bias");
      const decPad = tensors.get(spec.graph, "pad_bias");
      const encIn = tensors.inputsFor(encoderName), encOut = tensors.outputsFor(encoderName);
      const tokensView = new Int32Array(rig.graphs[spec.graph].outputs[spec.tokensOutput].shape.reduce((a, b) => a * b, 1));
      Object.assign(graphStats, rig.stats);
      loaded[L] = { ctx, rig, tensors, spec, idsT, encPad, decPad, encIn, encOut, encoderName, tokensView };
    }
  } finally {
    if (typeof constants === "function") constants.release();
  }

  const tokenizer = await tokenizerP;
  const maxNewTokens = ar.maxNewTokens ?? 256;
  let last = null;

  const encode = (text) => {
    const enc = tokenizer(SPECIALIST_PREFIX + text, { add_special_tokens: true, padding: false, truncation: false });
    const ids = Array.from(enc.input_ids.data ?? enc.input_ids, Number);
    if (ids.length > buckets[buckets.length - 1])
      throw new Error(`intellitex-webnn: tokenized prompt is ${ids.length} tokens; largest bucket is ${buckets[buckets.length - 1]}`);
    return ids;
  };

  const runIds = async (ids) => {
    const L = bucketFor(ids.length, buckets);
    const b = loaded[L];
    const padded = padIds(ids, L);
    const t = performance.now();
    b.ctx.writeTensor(b.idsT, padded.ids);
    b.ctx.writeTensor(b.encPad, padded.padBias);
    b.ctx.writeTensor(b.decPad, padded.padBias);
    b.ctx.dispatch(b.rig.graphs[b.encoderName].graph, b.encIn, b.encOut);
    const r = await autoregressive(b.ctx, b.rig, b.tensors, b.spec, { maxNewTokens, tokensView: b.tokensView });
    last = { tokens: r.tokens, dispatches: r.dispatches, endedWithEos: r.endedWithEos, bucket: L, ms: +(performance.now() - t).toFixed(2) };
    return tokenizer.decode(r.tokens, { skip_special_tokens: true }).trim();
  };

  const run = async (text) => runIds(encode(text));

  if (!skipWarmup) await run("x squared");

  return {
    run,
    encode,
    lastRun: () => last,
    stats: {
      backend: fp.backend, preferredInputLayout: fp.preferredInputLayout,
      graphs: graphStats, entry: `${entry.family}/${entry.id}`,
      loaderVersion: entry.producedBy?.workbench ?? null,
      buildMs: +(performance.now() - t0).toFixed(1),
    },
    close: () => {
      for (const b of Object.values(loaded)) {
        for (const g of Object.values(b.rig.graphs)) g.graph.destroy?.();
        for (const t of Object.values(b.tensors.tensors)) t.destroy?.();
        b.ctx.destroy?.();
      }
    },
  };
}
