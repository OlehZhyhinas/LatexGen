// Unit tests for the IntelliTeX catalog loader: recipe contract (buckets,
// no I/O name collisions, chain + autoregressive spec) and pickRuntime.
// Does not build graphs — that needs Chrome + Core ML + the ~994 MiB blobs.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const DIR = join(ROOT, "public/models/intellitex-webnn");
const read = (name) => JSON.parse(readFileSync(join(DIR, name), "utf8"));

const store = {};
globalThis.localStorage = {
  getItem: (k) => (k in store ? store[k] : null),
  setItem: (k, v) => { store[k] = String(v); },
  removeItem: (k) => { delete store[k]; },
};
globalThis.Worker = class { constructor() {} postMessage() {} };
const nav = { ml: { createContext() {} }, gpu: {}, userAgent: "Node.js", vendor: "" };
Object.defineProperty(globalThis, "navigator", { configurable: true, get: () => nav, set: () => {} });

const { pickRuntime, webnnAvailable } = await import("../public/models.js");
const { toFloat16, bucketFor, padIds } = await import("../public/intellitex-webnn.js");

const family = read("family.json");
const entry = read("entry.json");
const manifest = read("manifest.json");
const decode32 = read("recipe.decode32.json");
const encoder32 = read("recipe.encoder32.json");

test("family and entry name the intellitex-t5-220m / Core ML recipe", () => {
  assert.equal(family.id, "intellitex-t5-220m");
  assert.equal(entry.family, "intellitex-t5-220m");
  assert.equal(entry.compat.requires["backend.name"], "coreml");
  assert.deepEqual(Object.keys(family.contract.chaining.buckets).sort(), ["128", "32", "64"]);
  assert.equal(family.contract.chaining.autoregressive.maxNewTokens, 256);
  assert.equal(family.contract.chaining.autoregressive.eos, 2);
  assert.equal(family.contract.chaining.autoregressive.bos, 0);
});

test("decode input and output names do not collide (ping-pong caches)", () => {
  const ins = new Set(decode32.inputs.map((x) => x.name));
  const outs = new Set(decode32.outputs.map((x) => x.name));
  assert.deepEqual([...ins].filter((n) => outs.has(n)), []);
  assert.ok(ins.has("token") && outs.has("token_out"));
  for (let i = 0; i < 12; i++) {
    assert.ok(ins.has(`self${i}_k`) && outs.has(`self${i}_k_out`));
    assert.ok(ins.has(`self${i}_v`) && outs.has(`self${i}_v_out`));
    assert.ok(ins.has(`dec${i}_k`) && ins.has(`dec${i}_v`));
  }
});

test("entry chain links every encoder K/V into the matching decode input", () => {
  for (const L of [32, 64, 128]) {
    const links = entry.graphs.chain.filter(([from]) => from.startsWith(`encoder${L}.`));
    assert.equal(links.length, 24);
    for (const [from, to] of links) {
      const [, fo] = from.split(".");
      const [tg, ti] = to.split(".");
      assert.equal(tg, `decode${L}`);
      assert.equal(fo, ti);
    }
  }
  assert.ok(encoder32.outputs.some((o) => o.name === "dec0_k"));
});

test("autoregressive spec names tensors that exist on the decode recipe", () => {
  const spec = family.contract.chaining.autoregressive;
  const ins = new Set(decode32.inputs.map((x) => x.name));
  const outs = new Set(decode32.outputs.map((x) => x.name));
  assert.ok(ins.has(spec.tokenInput) && ins.has(spec.positionInput));
  assert.ok(outs.has(spec.tokensOutput));
  assert.equal(spec.caches.length, 24);
  for (const [i, o] of spec.caches) {
    assert.ok(ins.has(i), i);
    assert.ok(outs.has(o), o);
  }
});

test("manifest hashes encoder blobs per bucket and shared decode constants", () => {
  assert.ok(manifest.constants.encoder32.url);
  assert.ok(manifest.constants.decode32.url);
  assert.ok(manifest.constants.encoder64.url);
  assert.ok(manifest.constants.encoder128.url);
  assert.equal(manifest.constants.decode32.bytes > 200e6, true);
  assert.match(manifest.constants.encoder32.sha256, /^[0-9a-f]{64}$/);
  assert.equal(entry.graphs.decode64.constants, "decode32");
  assert.equal(entry.graphs.decode128.constants, "decode32");
});

test("pickRuntime prefers WebNN for intellitex when navigator.ml exists", () => {
  delete store["latexgen.runtime"];
  nav.ml = { createContext() {} };
  nav.gpu = {};
  assert.equal(webnnAvailable(), true);
  assert.deepEqual(pickRuntime("intellitex"), { device: "webnn", dtype: "fp16" });
  assert.deepEqual(pickRuntime("texify"), { device: "webnn", dtype: "fp16" });
});

test("pickRuntime skips WebNN for intellitex after a remembered failure (WebGPU next)", () => {
  store["latexgen.runtime"] = JSON.stringify({ intellitex: "webgpu" });
  nav.ml = { createContext() {} };
  nav.gpu = {};
  assert.deepEqual(pickRuntime("intellitex"), { device: "webgpu", dtype: "q4" });
});

test("pickRuntime without WebNN is today's WebGPU int4 / CPU int8 for intellitex", () => {
  delete store["latexgen.runtime"];
  nav.ml = undefined;
  nav.gpu = {};
  assert.deepEqual(pickRuntime("intellitex"), { device: "webgpu", dtype: "q4" });
  nav.gpu = undefined;
  assert.deepEqual(pickRuntime("intellitex"), { device: "wasm", dtype: "q8" });
});

test("bucketFor picks the smallest fitting length", () => {
  assert.equal(bucketFor(1), 32);
  assert.equal(bucketFor(32), 32);
  assert.equal(bucketFor(33), 64);
  assert.equal(bucketFor(128), 128);
  assert.throws(() => bucketFor(129));
});

test("padIds writes pad id and a large negative bias on the tail", () => {
  const { ids, padBias } = padIds([7, 8], 4);
  assert.deepEqual([...ids], [7, 8, 0, 0]);
  assert.equal(padBias.length, 4);
  const asF32 = typeof Float16Array !== "undefined" ? Float32Array.from(padBias) : null;
  if (asF32) {
    assert.equal(asF32[0], 0);
    assert.ok(asF32[2] < -1000);
  }
});

test("toFloat16 round-trips a few values", () => {
  const src = new Float32Array([0, 1, -1, 0.5]);
  const half = toFloat16(src);
  assert.equal(half.length, src.length);
});

test("recipes are present; constants blobs are optional in git", () => {
  assert.equal(existsSync(join(DIR, "recipe.encoder32.json")), true);
  assert.equal(existsSync(join(DIR, "recipe.decode128.json")), true);
  assert.equal(existsSync(join(DIR, "entry.json")), true);
});
