// Main-thread client for onnx-worker.js: runtime selection (WebGPU int4 vs
// CPU int8, remembered per model), load progress plumbing, and promise-based
// run(). All ONNX inference happens in the worker, so callers never block
// the page.
const RUNTIME_PREF_KEY = "latexgen.runtime";
function runtimePrefs() { try { return JSON.parse(localStorage.getItem(RUNTIME_PREF_KEY) || "{}"); } catch { return {}; } }
function rememberRuntime(model, device, ok) { try { const p = runtimePrefs(); p[model] = ok ? device : "wasm"; localStorage.setItem(RUNTIME_PREF_KEY, JSON.stringify(p)); } catch { /* ignore */ } }
// From the Level-A benchmark: IntelliTeX 1.28s->0.45s and Texify 8.0s->0.81s
// on WebGPU int4 with identical outputs; int4 on CPU is 10x slower than int8,
// so int4 is GPU-only. Texo is overhead-bound on ONNX Runtime (0.77 s CPU fp32,
// 0.78 s WebGPU), so it takes the hand-built WebNN graphs from the graph
// catalog when WebNN is present (23 ms per image on Core ML with identical
// tokens, docs/benchmarks.md) and stays CPU fp32 otherwise. Texify does the
// same once its catalog recipes are present (~146 ms vs 0.81 s). Only the
// Core ML backend is accepted: the worker asserts the fingerprint and a
// mismatch (the silent TFLite CPU fallback) is remembered like a failed
// WebGPU session. A failed Texify WebNN load falls back to WebGPU int4, not
// all the way to CPU — WebGPU still wins for this model.
export const webnnAvailable = () => typeof navigator !== "undefined" && !!navigator.ml;
export function pickRuntime(model) {
  const pref = runtimePrefs()[model];
  if (model === "texo") {
    const webnn = { device: "webnn", dtype: "fp16" }, cpu = { dtype: "fp32" };
    if (!webnnAvailable() || pref === "wasm") return cpu;
    return webnn;
  }
  const gpu = { device: "webgpu", dtype: "q4" }, cpu = { device: "wasm", dtype: "q8" };
  if (model === "texify") {
    if (webnnAvailable() && pref !== "wasm" && pref !== "webgpu") return { device: "webnn", dtype: "fp16" };
    if (!navigator.gpu || pref === "wasm") return cpu;
    return gpu;
  }
  if (!navigator.gpu) return cpu;
  return pref === "wasm" ? cpu : gpu;
}

const worker = new Worker(new URL("onnx-worker.js", import.meta.url), { type: "module" });
const pending = new Map();
const progressListeners = new Map(); // key -> Set<fn>
let seq = 0;
worker.onmessage = ({ data }) => {
  if (data.type === "progress") { for (const fn of progressListeners.get(data.key) ?? []) fn(data); return; }
  const p = pending.get(data.id); if (!p) return;
  pending.delete(data.id);
  if (data.type === "error") p.reject(new Error(data.error)); else p.resolve(data);
};
const call = (msg) => new Promise((resolve, reject) => { const id = ++seq; pending.set(id, { resolve, reject }); worker.postMessage({ ...msg, id }); });

export const runtimeUsed = {};
export const loaded = { intellitex: false, texo: false, texify: false };
const loaders = {};

// Loads a model with the preferred runtime; a WebGPU failure is remembered so
// the *next* page load uses CPU (a failed session poisons the runtime).
export function loadModel(key, onProgress) {
  loaders[key] ??= (async () => {
    const cfg = pickRuntime(key);
    if (onProgress) { if (!progressListeners.has(key)) progressListeners.set(key, new Set()); progressListeners.get(key).add(onProgress); }
    try {
      await call({ type: "load", key, cfg });
      if (cfg.device === "webgpu" || cfg.device === "webnn") rememberRuntime(key, cfg.device, true);
      runtimeUsed[key] = cfg; loaded[key] = true;
    } catch (err) {
      if (cfg.device === "webnn") {
        rememberRuntime(key, key === "texify" ? "webgpu" : "wasm", key === "texify");
        console.warn(`${key}: WebNN failed, ${key === "texify" ? "WebGPU" : "CPU"} on next load`, err);
      } else if (cfg.device === "webgpu") { rememberRuntime(key, cfg.device, false); console.warn(`${key}: WebGPU failed, CPU on next load`, err); }
      delete loaders[key];
      throw err;
    } finally {
      if (onProgress) progressListeners.get(key)?.delete(onProgress);
    }
  })();
  return loaders[key];
}
export function onProgress(key, fn) { if (!progressListeners.has(key)) progressListeners.set(key, new Set()); progressListeners.get(key).add(fn); return () => progressListeners.get(key)?.delete(fn); }
export const runModel = async (key, input) => (await call({ type: "run", key, input })).output;
