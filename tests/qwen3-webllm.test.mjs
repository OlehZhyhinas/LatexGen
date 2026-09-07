// Unit tests for the qwen3-webllm catalog runtime: catalog shape, the
// vendored bundle hashes (no network), and the planEngine/probeTarget
// contract against a fake GPU + storage. public/qwen3-webllm.js is owned by
// a concurrent implementer and may not exist yet — module-dependent tests
// skip with a clear message via existsSync + t.skip when it's absent, but
// the catalog-shape and vendored-bundle-hash tests always run since they
// only need catalog.json and public/vendor/webllm/*.js (both already vendored).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const CATALOG_PATH = join(ROOT, "public/models/qwen3-webllm/catalog.json");
const VENDOR_DIR = join(ROOT, "public/vendor/webllm");
const MODULE_PATH = join(ROOT, "public/qwen3-webllm.js");

const catalog = JSON.parse(readFileSync(CATALOG_PATH, "utf8"));

const sha256File = (p) => {
  const h = createHash("sha256");
  h.update(readFileSync(p));
  return h.digest("hex");
};

// --- catalog shape -----------------------------------------------------

test("every variant's runtime exists in runtimes", () => {
  for (const [modelId, model] of Object.entries(catalog.models)) {
    for (const [variantId, variant] of Object.entries(model.variants)) {
      assert.ok(catalog.runtimes[variant.runtime], `${modelId}/${variantId} -> runtime ${variant.runtime}`);
    }
  }
});

test("every model has a default that is one of its variants", () => {
  for (const [modelId, model] of Object.entries(catalog.models)) {
    assert.ok(model.variants[model.default], `${modelId}: default ${model.default} is not a variant`);
  }
});

test("every variant's tuning keys are a subset of hooks keys", () => {
  const hookKeys = new Set(Object.keys(catalog.hooks));
  for (const [modelId, model] of Object.entries(catalog.models)) {
    for (const [variantId, variant] of Object.entries(model.variants)) {
      for (const key of Object.keys(variant.tuning)) {
        assert.ok(hookKeys.has(key), `${modelId}/${variantId} tuning key "${key}" not in hooks`);
      }
    }
  }
});

test("sha256 fields are 64 lowercase hex characters", () => {
  const shas = [
    ...Object.values(catalog.runtimes).map((r) => r.sha256),
    ...Object.values(catalog.models).map((m) => m.modelLib.sha256),
  ];
  assert.ok(shas.length > 0);
  for (const sha of shas) assert.match(sha, /^[0-9a-f]{64}$/);
});

test("runtime and modelLib urls start with source.base", () => {
  const urls = [
    ...Object.values(catalog.runtimes).map((r) => r.url),
    ...Object.values(catalog.models).map((m) => m.modelLib.url),
  ];
  assert.ok(urls.length > 0);
  for (const url of urls) assert.ok(url.startsWith(catalog.source.base), url);
});

test("modelLib file names match qwen3-<size>-argmax-chunk256-sg32-tr32.wasm", () => {
  for (const [modelId, model] of Object.entries(catalog.models)) {
    assert.match(model.modelLib.file, /^qwen3-[a-z0-9.]+-argmax-chunk256-sg32-tr32\.wasm$/, modelId);
  }
});

// --- vendored bundles (no network) -------------------------------------

test("every runtime file exists under public/vendor/webllm/ with the catalog's bytes and sha256", () => {
  for (const [id, r] of Object.entries(catalog.runtimes)) {
    const dest = join(VENDOR_DIR, r.file);
    assert.ok(existsSync(dest), `${id}: ${r.file} missing under public/vendor/webllm/`);
    const bytes = readFileSync(dest).byteLength;
    assert.equal(bytes, r.bytes, `${id}: bytes mismatch`);
    assert.equal(sha256File(dest), r.sha256, `${id}: sha256 mismatch`);
  }
});

// --- module-dependent: planEngine / probeTarget / helpers --------------

const hasModule = existsSync(MODULE_PATH);
const mod = hasModule ? await import("../public/qwen3-webllm.js") : null;

function maybeTest(name, fn) {
  test(name, async (t) => {
    if (!hasModule) {
      t.skip("public/qwen3-webllm.js not present yet (owned by a concurrent implementer) — skipping module-dependent test");
      return;
    }
    await fn(t);
  });
}

const CHROME_152_UA =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.7977.77 Safari/537.36";
const CHROME_141_UA =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36";

const goodNav = () => ({
  gpu: {
    requestAdapter: async () => ({ info: { vendor: "apple" }, features: new Set(["subgroups"]) }),
  },
  userAgent: CHROME_152_UA,
});

function makeStorage() {
  const store = {};
  return {
    getItem: (k) => (k in store ? store[k] : null),
    setItem: (k, v) => { store[k] = String(v); },
    removeItem: (k) => { delete store[k]; },
  };
}

const MODEL_ID = "Qwen3-1.7B-q4f16_1-MLC";
const stockRecordFor = (modelId) => ({ model_id: modelId, model: `stock/${modelId}`, model_lib: "https://stock.example/lib.wasm" });

maybeTest("planEngine: apple+subgroups+Chrome152 picks the model's default catalog variant", async () => {
  const cat = structuredClone(catalog);
  const model = cat.models[MODEL_ID];
  const variant = model.variants[model.default];
  const runtime = cat.runtimes[variant.runtime];

  const plan = await mod.planEngine(MODEL_ID, cat, {
    nav: goodNav(),
    storage: makeStorage(),
    stockRecord: stockRecordFor(MODEL_ID),
  });

  assert.equal(plan.kind, "catalog");
  assert.equal(plan.variant, model.default);
  // plan.runtime may be the runtime id or the resolved runtime record — the
  // spec pins only that workerUrl carries bundle=<runtime.file> (checked below).
  const planRuntimeFile = typeof plan.runtime === "string" ? runtime.file : plan.runtime?.file;
  assert.equal(planRuntimeFile, runtime.file);
  assert.deepEqual(plan.tuning, variant.tuning);
  assert.equal(plan.label, `catalog ${model.default}`);
  assert.equal(plan.appConfig.model_list[0].model_lib, model.modelLib.url);
  assert.equal(plan.appConfig.model_list[0].model_id, MODEL_ID);
  assert.ok(plan.workerUrl.search.includes(`bundle=${runtime.file}`), plan.workerUrl.search);
});

maybeTest("planEngine: a model without catalog libs (the shipped Qwen3.5 and MiniCPM5 sizes) runs stock", async () => {
  const cat = structuredClone(catalog);
  for (const id of ["Qwen3.5-0.8B-q4f16_1-MLC", "MiniCPM5-2B-q4f16_1-MLC", "Qwen3.5-4B-q4f16_1-MLC", "Qwen3.5-9B-q4f16_1-MLC"]) {
    const plan = await mod.planEngine(id, cat, { nav: goodNav(), storage: makeStorage(), stockRecord: stockRecordFor(id) });
    assert.equal(plan.kind, "stock", id);
    assert.equal(plan.why, "not in catalog", id);
  }
});

maybeTest("planEngine: force stock always returns stock", async () => {
  const cat = structuredClone(catalog);
  const plan = await mod.planEngine(MODEL_ID, cat, {
    nav: goodNav(),
    storage: makeStorage(),
    force: "stock",
    stockRecord: stockRecordFor(MODEL_ID),
  });
  assert.equal(plan.kind, "stock");
  assert.equal(plan.why, "forced");
});

maybeTest("planEngine: force catalog:<variant> selects that variant", async () => {
  const cat = structuredClone(catalog);
  const model = cat.models[MODEL_ID];
  const otherVariant = Object.keys(model.variants).find((v) => v !== model.default);
  assert.ok(otherVariant, "fixture needs a second variant to distinguish from default");

  const plan = await mod.planEngine(MODEL_ID, cat, {
    nav: goodNav(),
    storage: makeStorage(),
    force: { variant: otherVariant },
    stockRecord: stockRecordFor(MODEL_ID),
  });
  assert.equal(plan.kind, "catalog");
  assert.equal(plan.variant, otherVariant);
});

maybeTest("planEngine: forcing catalog clears a remembered stock fallback", async () => {
  const cat = structuredClone(catalog);
  const storage = makeStorage();
  mod.rememberRuntime(MODEL_ID, "stock", "prior failure", storage, Date.now());
  assert.equal(mod.rememberedRuntime(MODEL_ID, storage), "stock");

  const plan = await mod.planEngine(MODEL_ID, cat, {
    nav: goodNav(),
    storage,
    force: "catalog",
    stockRecord: stockRecordFor(MODEL_ID),
  });
  assert.equal(plan.kind, "catalog");
  assert.equal(mod.rememberedRuntime(MODEL_ID, storage), undefined);
});

maybeTest("planEngine: non-apple GPU vendor falls back to stock", async () => {
  const cat = structuredClone(catalog);
  const nav = {
    gpu: { requestAdapter: async () => ({ info: { vendor: "nvidia" }, features: new Set(["subgroups"]) }) },
    userAgent: CHROME_152_UA,
  };
  const plan = await mod.planEngine(MODEL_ID, cat, { nav, storage: makeStorage(), stockRecord: stockRecordFor(MODEL_ID) });
  assert.equal(plan.kind, "stock");
});

maybeTest("planEngine: missing subgroups feature falls back to stock", async () => {
  const cat = structuredClone(catalog);
  const nav = {
    gpu: { requestAdapter: async () => ({ info: { vendor: "apple" }, features: new Set([]) }) },
    userAgent: CHROME_152_UA,
  };
  const plan = await mod.planEngine(MODEL_ID, cat, { nav, storage: makeStorage(), stockRecord: stockRecordFor(MODEL_ID) });
  assert.equal(plan.kind, "stock");
});

maybeTest("planEngine: Chromium below minMajor falls back to stock", async () => {
  const cat = structuredClone(catalog);
  const nav = {
    gpu: { requestAdapter: async () => ({ info: { vendor: "apple" }, features: new Set(["subgroups"]) }) },
    userAgent: CHROME_141_UA,
  };
  const plan = await mod.planEngine(MODEL_ID, cat, { nav, storage: makeStorage(), stockRecord: stockRecordFor(MODEL_ID) });
  assert.equal(plan.kind, "stock");
});

maybeTest("planEngine: no navigator.gpu falls back to stock", async () => {
  const cat = structuredClone(catalog);
  const nav = { userAgent: CHROME_152_UA };
  const plan = await mod.planEngine(MODEL_ID, cat, { nav, storage: makeStorage(), stockRecord: stockRecordFor(MODEL_ID) });
  assert.equal(plan.kind, "stock");
});

maybeTest("planEngine: remembered stock within STOCK_TTL_MS stays stock; past it re-probes to catalog", async () => {
  const now = Date.now();

  const catFresh = structuredClone(catalog);
  const storageFresh = makeStorage();
  mod.rememberRuntime(MODEL_ID, "stock", "prior failure", storageFresh, now - 1000);
  const freshPlan = await mod.planEngine(MODEL_ID, catFresh, {
    nav: goodNav(),
    storage: storageFresh,
    stockRecord: stockRecordFor(MODEL_ID),
  });
  assert.equal(freshPlan.kind, "stock");
  assert.equal(freshPlan.why, "remembered");

  const catExpired = structuredClone(catalog); // fresh clone: probeTarget memoises per catalog object
  const storageExpired = makeStorage();
  mod.rememberRuntime(MODEL_ID, "stock", "prior failure", storageExpired, now - mod.STOCK_TTL_MS - 1000);
  const expiredPlan = await mod.planEngine(MODEL_ID, catExpired, {
    nav: goodNav(),
    storage: storageExpired,
    stockRecord: stockRecordFor(MODEL_ID),
  });
  assert.equal(expiredPlan.kind, "catalog");
});

maybeTest("planEngine: model not in the catalog falls back to stock", async () => {
  const cat = structuredClone(catalog);
  const plan = await mod.planEngine("Not-A-Real-Model", cat, {
    nav: goodNav(),
    storage: makeStorage(),
    stockRecord: stockRecordFor("Not-A-Real-Model"),
  });
  assert.equal(plan.kind, "stock");
  assert.equal(plan.why, "not in catalog");
});

maybeTest("parseForce reads ?webllm=stock|catalog|catalog:<variant>", () => {
  assert.equal(mod.parseForce("?webllm=stock"), "stock");
  assert.equal(mod.parseForce("?webllm=catalog"), "catalog");
  assert.deepEqual(mod.parseForce("?webllm=catalog:sg32-burst4-flush32"), { variant: "sg32-burst4-flush32" });
  assert.equal(mod.parseForce(""), undefined);
  assert.equal(mod.parseForce("?foo=bar"), undefined);
  assert.equal(mod.parseForce(new URLSearchParams("webllm=stock")), "stock");
});

maybeTest("tuningFromSearch(workerUrl(plan).search) round-trips the tuning", async () => {
  const cat = structuredClone(catalog);
  const plan = await mod.planEngine(MODEL_ID, cat, {
    nav: goodNav(),
    storage: makeStorage(),
    stockRecord: stockRecordFor(MODEL_ID),
  });
  const roundTripped = mod.tuningFromSearch(plan.workerUrl.search);
  assert.deepEqual(roundTripped, plan.tuning);
});

maybeTest("applyTuning sets the right __webllm*/__tvmjs* names on a plain object", () => {
  const target = {};
  mod.applyTuning(target, {
    greedyArgmax: true,
    greedyBurst: 4,
    deferDecodeCleanup: true,
    batchPass: true,
    flushEvery: 32,
    bindGroupCache: true,
    lookahead: 1,
  });
  assert.equal(target.__webllmGreedyArgmax, true);
  assert.equal(target.__webllmGreedyBurst, 4);
  assert.equal(target.__webllmDeferDecodeCleanup, true);
  assert.equal(target.__tvmjsWebGPUBatchPass, true);
  assert.equal(target.__tvmjsWebGPUFlushEvery, 32);
  assert.equal(target.__tvmjsWebGPUBindGroupCache, true);
  assert.equal(target.__webllmBurstLookahead, 1);
});

maybeTest("BUNDLE_RE only matches bare vendored bundle basenames", () => {
  assert.ok(mod.BUNDLE_RE.test("web-llm-0.2.84-qwen-m5.js"));
  assert.ok(mod.BUNDLE_RE.test("web-llm-0.2.84-qwen-m5-overlap.js"));
  assert.ok(mod.BUNDLE_RE.test("web-llm-0.2.84-qwen-m5-lookahead.js"));
  assert.equal(mod.BUNDLE_RE.test("../index.js"), false);
  assert.equal(mod.BUNDLE_RE.test("index.js"), false);
});
