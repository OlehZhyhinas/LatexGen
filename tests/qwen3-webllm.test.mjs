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

test("modelLib file names match <model>-argmax-chunk256-{sg32|nosg}[-tr<N>].wasm", () => {
  for (const [modelId, model] of Object.entries(catalog.models)) {
    assert.match(model.modelLib.file, /^[a-z0-9.-]+-argmax-chunk256-(sg32|nosg)(-tr\d+)?\.wasm$/, modelId);
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

maybeTest("planEngine: the shipped Qwen3.5 and MiniCPM5 sizes resolve through the catalog", async () => {
  const cat = structuredClone(catalog);
  for (const id of ["Qwen3.5-0.8B-q4f16_1-MLC", "MiniCPM5-2B-q4f16_1-MLC", "Qwen3.5-4B-q4f16_1-MLC", "Qwen3.5-9B-q4f16_1-MLC"]) {
    const plan = await mod.planEngine(id, cat, { nav: goodNav(), storage: makeStorage(), stockRecord: stockRecordFor(id) });
    assert.equal(plan.kind, "catalog", id);
    assert.equal(plan.variant, cat.models[id].default, id);
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

// --- recurrent-state models must never over-fill the decode queue ------
//
// greedyBurst > 1 and lookahead > 0 both issue decode steps past what the CPU
// has consumed and pop the surplus back with vm.builtin.kv_state_popn. On a
// model with space-state layers that pop aborts the WASM runtime, so neither
// knob may reach such a model — not from the catalog, and not from a forced
// variant.

test("no recurrentState model declares an over-filling variant", () => {
  for (const [modelId, model] of Object.entries(catalog.models)) {
    if (!model.recurrentState) continue;
    for (const [variantId, variant] of Object.entries(model.variants)) {
      const burst = variant.tuning.greedyBurst;
      assert.ok(burst === undefined || burst === 1, `${modelId}/${variantId}: greedyBurst ${burst} pops on a recurrent state`);
      assert.equal(variant.tuning.lookahead, undefined, `${modelId}/${variantId}: lookahead pops on a recurrent state`);
    }
  }
});

test("the models known to have space-state layers are flagged", () => {
  for (const id of ["Qwen3.5-4B-q4f16_1-MLC", "Qwen3.5-9B-q4f16_1-MLC"]) {
    assert.equal(catalog.models[id].recurrentState, true, `${id} aborts in RNNStateImpObj::PopN when the queue over-fills`);
  }
});

maybeTest("safeTuning clamps the popping knobs only for recurrent models", () => {
  const unsafe = { greedyArgmax: true, greedyBurst: 4, batchPass: true, flushEvery: 32, lookahead: 1 };

  const clamped = mod.safeTuning({ recurrentState: true }, unsafe);
  assert.equal(clamped.greedyBurst, 1);
  assert.equal("lookahead" in clamped, false);
  // the pop-free knobs survive untouched — they are where the speed comes from
  assert.equal(clamped.greedyArgmax, true);
  assert.equal(clamped.batchPass, true);
  assert.equal(clamped.flushEvery, 32);

  assert.deepEqual(mod.safeTuning({}, unsafe), unsafe);
  assert.deepEqual(mod.safeTuning(undefined, unsafe), unsafe);
  assert.deepEqual(mod.safeTuning({ recurrentState: true }, undefined), {});
});

maybeTest("planEngine clamps a forced over-filling variant on a recurrent model", async () => {
  const cat = structuredClone(catalog);
  const id = "Qwen3.5-4B-q4f16_1-MLC";
  assert.equal(cat.models[id].recurrentState, true);
  // a stale or hand-edited catalog entry, reachable via ?webllm=catalog:<variant>
  cat.models[id].variants["legacy-lookahead"] = {
    runtime: "m5-lookahead",
    tuning: { greedyArgmax: true, greedyBurst: 4, lookahead: 1 },
  };

  const plan = await mod.planEngine(id, cat, {
    nav: goodNav(),
    storage: makeStorage(),
    force: { variant: "legacy-lookahead" },
    stockRecord: stockRecordFor(id),
  });

  assert.equal(plan.kind, "catalog");
  assert.equal(plan.variant, "legacy-lookahead");
  assert.equal(plan.tuning.greedyBurst, 1);
  assert.equal("lookahead" in plan.tuning, false);
  // the worker inherits the clamp through the query string, not just the plan
  assert.equal(plan.workerUrl.searchParams.get("greedyBurst"), "1");
  assert.equal(plan.workerUrl.searchParams.has("lookahead"), false);
});

maybeTest("planEngine leaves a non-recurrent model's lookahead variant alone", async () => {
  const cat = structuredClone(catalog);
  const id = "MiniCPM5-2B-q4f16_1-MLC";
  assert.ok(!cat.models[id].recurrentState);
  const variant = cat.models[id].variants[cat.models[id].default];
  assert.equal(variant.tuning.lookahead, 1, "fixture expects MiniCPM5 to ship lookahead");

  const plan = await mod.planEngine(id, cat, { nav: goodNav(), storage: makeStorage(), stockRecord: stockRecordFor(id) });
  assert.deepEqual(plan.tuning, variant.tuning);
  assert.equal(plan.workerUrl.searchParams.get("lookahead"), "1");
});

// --- prompt-lookup drafting (issue #27, Avenue B) -------------------------

test("prompt-lookup variants carry the spec string and recurrent models carry :fork", () => {
  for (const [modelId, model] of Object.entries(catalog.models)) {
    for (const [variantId, variant] of Object.entries(model.variants)) {
      const spec = variant.tuning.promptLookup;
      if (spec === undefined) continue;
      assert.match(spec, /^[1-9]\d*(?::[1-9]\d*){0,2}(?::fork)?$/, `${modelId}/${variantId}: bad promptLookup spec`);
      if (model.recurrentState) assert.ok(spec.endsWith(":fork"), `${modelId}/${variantId}: a recurrent state needs the fork strategy`);
      assert.ok(catalog.runtimes[variant.runtime].hooks.includes("promptLookup"), `${modelId}/${variantId}: runtime lacks the promptLookup hook`);
    }
  }
});

test("every Qwen3.5 rung is flagged recurrentState", () => {
  // All Qwen3.5 sizes interleave GatedDeltaNet layers with attention
  // (full_attention_interval 4 in every mlc-chat-config), the 0.8B included.
  for (const id of Object.keys(catalog.models).filter((m) => m.startsWith("Qwen3.5-"))) {
    assert.equal(catalog.models[id].recurrentState, true, `${id} has an RNN state`);
  }
});

maybeTest("parsePromptLookup reads the harness spec string", () => {
  assert.deepEqual(mod.parsePromptLookup("5:3:2:fork"), { k: 5, nMax: 3, nMin: 2, hybrid: "fork" });
  assert.deepEqual(mod.parsePromptLookup("5"), { k: 5, nMax: 3, nMin: 2 });
  assert.deepEqual(mod.parsePromptLookup("8:4:4"), { k: 8, nMax: 4, nMin: 4 });
  assert.deepEqual(mod.parsePromptLookup("8:2:3"), { k: 8, nMax: 2, nMin: 2 });
  for (const bad of ["0", "5:x", "5:3:2:pop", "", undefined, 5]) assert.equal(mod.parsePromptLookup(bad), null);
});

maybeTest("safeTuning appends :fork to prompt-lookup on recurrent models and leaves it otherwise", () => {
  const tuning = { greedyArgmax: true, batchPass: true, flushEvery: 32, promptLookup: "5:3:2" };
  assert.equal(mod.safeTuning({ recurrentState: true }, tuning).promptLookup, "5:3:2:fork");
  assert.equal(mod.safeTuning({ recurrentState: true }, { promptLookup: "5:3:2:fork" }).promptLookup, "5:3:2:fork");
  assert.equal(mod.safeTuning({}, tuning).promptLookup, "5:3:2");
});

maybeTest("the prompt-lookup spec survives the worker query string and lands as the runtime object", () => {
  const tuning = { greedyArgmax: true, batchPass: true, flushEvery: 32, promptLookup: "5:3:2:fork" };
  const url = mod.workerUrl({ kind: "catalog", runtime: { file: "web-llm-0.2.84-qwen-m5-prompt-lookup.js" }, tuning }, "http://localhost/public/qwen3-webllm.js");
  assert.equal(url.searchParams.get("promptLookup"), "5:3:2:fork");
  const parsed = mod.tuningFromSearch(url.search);
  assert.equal(parsed.promptLookup, "5:3:2:fork");
  const target = mod.applyTuning({}, parsed);
  assert.deepEqual(target.__webllmPromptLookup, { k: 5, nMax: 3, nMin: 2, hybrid: "fork" });
  assert.equal(target.__tvmjsWebGPUFlushEvery, 32);
  // a malformed spec is dropped rather than forwarded
  const bad = mod.workerUrl({ kind: "catalog", runtime: { file: "web-llm-0.2.84-qwen-m5-prompt-lookup.js" }, tuning: { promptLookup: "nope" } }, "http://localhost/public/qwen3-webllm.js");
  assert.equal(bad.searchParams.has("promptLookup"), false);
});
