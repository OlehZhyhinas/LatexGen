const DEFAULT_HOOKS = {
  greedyArgmax: "__webllmGreedyArgmax",
  greedyBurst: "__webllmGreedyBurst",
  deferDecodeCleanup: "__webllmDeferDecodeCleanup",
  batchPass: "__tvmjsWebGPUBatchPass",
  flushEvery: "__tvmjsWebGPUFlushEvery",
  bindGroupCache: "__tvmjsWebGPUBindGroupCache",
  lookahead: "__webllmBurstLookahead",
};

const BOOL_TUNING = new Set([
  "greedyArgmax",
  "deferDecodeCleanup",
  "batchPass",
  "bindGroupCache",
]);
const INT_TUNING = new Set([
  "greedyBurst",
  "flushEvery",
  "lookahead",
]);

const PROBE_NONE = {};
const probeMemo = new WeakMap();

let catalogPromise = null;

export const CATALOG_URL = new URL("models/qwen3-webllm/catalog.json", import.meta.url).href;
export const RUNTIME_KEY = "latexgen.webllm";
export const STOCK_TTL_MS = 7 * 24 * 3600 * 1000;
export const BUNDLE_RE = /^web-llm-0\.2\.84-qwen-[a-z0-9-]+\.js$/;

function toParams(search) {
  if (search instanceof URLSearchParams) return search;
  const text = typeof search === "string" ? search : "";
  return new URLSearchParams(text.startsWith("?") ? text.slice(1) : text);
}

function readRuntimeState(storage) {
  if (!storage || typeof storage.getItem !== "function") return {};
  try { return JSON.parse(storage.getItem(RUNTIME_KEY) || "{}"); } catch { return {}; }
}

function writeRuntimeState(storage, value) {
  if (!storage || typeof storage.setItem !== "function") return;
  try { storage.setItem(RUNTIME_KEY, JSON.stringify(value)); } catch { /* ignore */ }
}

function parseChromeMajor(nav) {
  for (const brand of nav?.userAgentData?.brands ?? []) {
    if (!/(Chromium|Google Chrome)/i.test(brand?.brand || "")) continue;
    const major = Number.parseInt(String(brand.version || "").split(".")[0], 10);
    if (Number.isFinite(major)) return major;
  }
  const m = String(nav?.userAgent || "").match(/Chrom(?:e|ium)\/(\d+)/);
  if (!m) return undefined;
  const major = Number.parseInt(m[1], 10);
  return Number.isFinite(major) ? major : undefined;
}

async function runProbe(catalog, nav) {
  if (!nav?.gpu || typeof nav.gpu.requestAdapter !== "function") {
    return { ok: false, why: "webgpu unavailable" };
  }

  let adapter = null;
  try { adapter = await nav.gpu.requestAdapter(); } catch { return { ok: false, why: "webgpu adapter failed" }; }
  if (!adapter) return { ok: false, why: "webgpu adapter unavailable" };

  const vendor = String(adapter?.info?.vendor || "").toLowerCase();
  const features = [...(adapter?.features ?? [])];
  const requiredVendor = String(catalog?.target?.gpu?.vendor || "").toLowerCase();
  const requiredFeatures = catalog?.target?.gpu?.features ?? [];
  const missing = requiredFeatures.filter((f) => !(adapter.features?.has?.(f)));
  const chromeMajor = parseChromeMajor(nav);
  const minMajor = catalog?.target?.browser?.minMajor;

  if (vendor !== requiredVendor) {
    return { ok: false, why: `gpu vendor ${vendor || "unknown"} is unsupported`, vendor, features, chromeMajor };
  }
  if (missing.length) {
    return { ok: false, why: `missing gpu features: ${missing.join(", ")}`, vendor, features, chromeMajor };
  }
  if (Number.isFinite(minMajor)) {
    if (!Number.isFinite(chromeMajor)) {
      return { ok: false, why: "chromium major version unknown", vendor, features, chromeMajor };
    }
    if (chromeMajor < minMajor) {
      return { ok: false, why: `chromium ${chromeMajor} < ${minMajor}`, vendor, features, chromeMajor };
    }
  }
  return { ok: true, why: "ok", vendor, features, chromeMajor };
}

function stockPlan(why) {
  const plan = { kind: "stock", why, label: "stock WebLLM" };
  plan.workerUrl = workerUrl(plan);
  return plan;
}

export async function loadCatalog(fetchImpl = globalThis.fetch) {
  if (catalogPromise) return catalogPromise;
  if (typeof fetchImpl !== "function") throw new Error("fetch is unavailable");
  catalogPromise = (async () => {
    // models/ is served immutable for a year; the catalog is tiny and its
    // defaults change, so always revalidate it instead of pinning the first copy.
    const r = await fetchImpl(CATALOG_URL, { cache: "no-cache" });
    if (!r?.ok) throw new Error(`catalog HTTP ${r?.status ?? "error"}`);
    return r.json();
  })();
  try {
    return await catalogPromise;
  } catch (err) {
    catalogPromise = null;
    throw err;
  }
}

export function rememberedRuntime(modelId, storage = globalThis.localStorage, now = Date.now()) {
  const state = readRuntimeState(storage);
  const record = state?.[modelId];
  if (!record || record.value !== "stock") return undefined;
  if (!Number.isFinite(record.at)) return undefined;
  if (now - record.at > STOCK_TTL_MS) return undefined;
  return "stock";
}

export function rememberRuntime(modelId, value, why, storage = globalThis.localStorage, now = Date.now()) {
  const state = readRuntimeState(storage);
  state[modelId] = { value, at: now, why };
  writeRuntimeState(storage, state);
}

export function forgetRuntime(modelId, storage = globalThis.localStorage) {
  const state = readRuntimeState(storage);
  if (!(modelId in state)) return;
  delete state[modelId];
  writeRuntimeState(storage, state);
}

export async function probeTarget(catalog, nav = globalThis.navigator) {
  if (!catalog || typeof catalog !== "object") return { ok: false, why: "catalog unavailable" };
  let byNav = probeMemo.get(catalog);
  if (!byNav) {
    byNav = new WeakMap();
    probeMemo.set(catalog, byNav);
  }
  const key = nav && typeof nav === "object" ? nav : PROBE_NONE;
  if (byNav.has(key)) return byNav.get(key);
  const promise = runProbe(catalog, nav);
  byNav.set(key, promise);
  return promise;
}

export function parseForce(search = "") {
  const value = toParams(search).get("webllm");
  if (value === "stock") return "stock";
  if (value === "catalog") return "catalog";
  if (value?.startsWith("catalog:")) {
    const variant = value.slice("catalog:".length);
    if (!variant) return "catalog";
    return { variant };
  }
  return undefined;
}

export async function planEngine(modelId, catalog, { force, nav = globalThis.navigator, storage = globalThis.localStorage, stockRecord } = {}) {
  const model = catalog?.models?.[modelId];
  if (!model) return stockPlan("not in catalog");
  if (force === "stock") return stockPlan("forced");

  const forceCatalog = force === "catalog" || !!(force && typeof force === "object" && force.variant);
  if (forceCatalog) forgetRuntime(modelId, storage);
  else if (rememberedRuntime(modelId, storage) === "stock") return stockPlan("remembered");

  const probe = await probeTarget(catalog, nav);
  if (!probe.ok) return stockPlan(probe.why);

  const variantId = (force && typeof force === "object" && force.variant) ? force.variant : model.default;
  const variant = model.variants?.[variantId];
  if (!variant) throw new Error(`unknown qwen3-webllm variant: ${variantId}`);

  const runtime = catalog.runtimes?.[variant.runtime];
  if (!runtime) throw new Error(`missing qwen3-webllm runtime: ${variant.runtime}`);
  if (!stockRecord) throw new Error(`missing stock record for ${modelId}`);

  const plan = {
    kind: "catalog",
    variant: variantId,
    runtime,
    tuning: variant.tuning ?? {},
    label: `catalog ${variantId}`,
    appConfig: { model_list: [{ ...stockRecord, model_lib: model.modelLib.url }] },
  };
  plan.workerUrl = workerUrl(plan);
  return plan;
}

export function workerUrl(plan, base = import.meta.url) {
  const url = new URL("webllm-worker.js", base);
  if (!plan || plan.kind !== "catalog") return url;
  const file = plan.runtime?.file;
  if (!BUNDLE_RE.test(file || "")) throw new Error(`invalid bundle: ${file}`);
  const params = url.searchParams;
  params.set("bundle", file);
  for (const key of Object.keys(DEFAULT_HOOKS)) {
    if (!(key in (plan.tuning ?? {}))) continue;
    const value = plan.tuning[key];
    if (typeof value === "boolean") params.set(key, value ? "1" : "0");
    else if (Number.isInteger(value)) params.set(key, String(value));
  }
  return url;
}

export function tuningFromSearch(search) {
  const params = toParams(search);
  const tuning = {};
  for (const key of Object.keys(DEFAULT_HOOKS)) {
    if (!params.has(key)) continue;
    const value = params.get(key);
    if (BOOL_TUNING.has(key)) {
      if (value === "1" || value === "true") tuning[key] = true;
      else if (value === "0" || value === "false") tuning[key] = false;
    } else if (INT_TUNING.has(key)) {
      const n = Number.parseInt(value, 10);
      if (Number.isFinite(n)) tuning[key] = n;
    }
  }
  return tuning;
}

export function applyTuning(target, tuning, hooks = DEFAULT_HOOKS) {
  if (!target || !tuning) return target;
  for (const [key, hook] of Object.entries(hooks)) {
    if (!(key in tuning)) continue;
    target[hook] = tuning[key];
  }
  return target;
}
