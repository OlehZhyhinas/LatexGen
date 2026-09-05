// Unit tests for the Texify catalog loader: recipe contract (no I/O name
// collisions, chain + autoregressive spec) and pickRuntime. Does not build
// graphs — that needs Chrome + Core ML + the ~597 MiB constants blobs.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const DIR = join(ROOT, "public/models/texify-webnn");
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
const { toFloat16, donutResizePlan, rgbaToNchw } = await import("../public/texify-webnn.js");

const family = read("family.json");
const entry = read("entry.json");
const manifest = read("manifest.json");
const encoder = read("recipe.encoder.json");
const decoder = read("recipe.decoder.json");

test("family and entry name the texify-420 / Core ML recipe", () => {
  assert.equal(family.id, "texify-420");
  assert.equal(entry.family, "texify-420");
  assert.equal(entry.compat.requires["backend.name"], "coreml");
  assert.equal(family.contract.chaining.autoregressive.graph, "decoder");
  assert.equal(family.contract.chaining.autoregressive.maxNewTokens, 384);
  assert.equal(family.contract.chaining.autoregressive.eos, 2);
});

test("decoder input and output names do not collide (ping-pong caches)", () => {
  const ins = new Set(decoder.inputs.map((x) => x.name));
  const outs = new Set(decoder.outputs.map((x) => x.name));
  assert.deepEqual([...ins].filter((n) => outs.has(n)), []);
  assert.ok(ins.has("token") && outs.has("tokens"));
  for (let i = 0; i < 8; i++) {
    assert.ok(ins.has(`self${i}_k`) && outs.has(`self${i}_k_out`));
    assert.ok(ins.has(`self${i}_v`) && outs.has(`self${i}_v_out`));
    assert.ok(ins.has(`dec${i}_k`) && ins.has(`dec${i}_v`));
  }
});

test("entry chain links every encoder K/V into the matching decoder input", () => {
  assert.equal(entry.graphs.chain.length, 16);
  for (const [from, to] of entry.graphs.chain) {
    const [fg, fo] = from.split(".");
    const [tg, ti] = to.split(".");
    assert.equal(fg, "encoder");
    assert.equal(tg, "decoder");
    assert.ok(encoder.outputs.some((o) => o.name === fo), from);
    assert.ok(decoder.inputs.some((o) => o.name === ti), to);
    assert.equal(fo, ti);
  }
});

test("autoregressive spec names tensors that exist on the decoder recipe", () => {
  const spec = family.contract.chaining.autoregressive;
  const ins = new Set(decoder.inputs.map((x) => x.name));
  const outs = new Set(decoder.outputs.map((x) => x.name));
  assert.ok(ins.has(spec.tokenInput) && ins.has(spec.positionInput));
  assert.ok(outs.has(spec.tokensOutput));
  assert.equal(spec.caches.length, 16);
  for (const [i, o] of spec.caches) {
    assert.ok(ins.has(i), i);
    assert.ok(outs.has(o), o);
  }
});

test("manifest hashes the two constants blobs and both recipes", () => {
  assert.equal(manifest.constants.encoder.file, "texify-encoder.bin");
  assert.equal(manifest.constants.decoder.file, "texify-decoder.bin");
  assert.match(manifest.constants.encoder.sha256, /^[0-9a-f]{64}$/);
  assert.match(manifest.constants.decoder.sha256, /^[0-9a-f]{64}$/);
  assert.ok(manifest.constants.encoder.bytes > 100e6);
  assert.ok(manifest.constants.decoder.bytes > 400e6);
  assert.match(manifest.constants.encoder.url, /\/texify-encoder\.bin$/);
  assert.match(manifest.constants.decoder.url, /\/texify-decode\.bin$/);
  assert.ok(manifest.recipeHashes["recipe.encoder.json"].sha256);
  assert.ok(manifest.recipeHashes["recipe.decoder.json"].sha256);
});

test("pickRuntime prefers WebNN for texify when navigator.ml exists", () => {
  delete store["latexgen.runtime"];
  nav.ml = { createContext() {} };
  nav.gpu = {};
  assert.equal(webnnAvailable(), true);
  assert.deepEqual(pickRuntime("texify"), { device: "webnn", dtype: "fp16" });
  assert.deepEqual(pickRuntime("texo"), { device: "webnn", dtype: "fp16" });
  assert.deepEqual(pickRuntime("intellitex"), { device: "webnn", dtype: "fp16" });
});

test("pickRuntime skips WebNN for texify after a remembered failure (WebGPU next)", () => {
  store["latexgen.runtime"] = JSON.stringify({ texify: "webgpu", texo: "wasm" });
  nav.ml = { createContext() {} };
  nav.gpu = {};
  assert.deepEqual(pickRuntime("texify"), { device: "webgpu", dtype: "q4" });
  assert.deepEqual(pickRuntime("texo"), { dtype: "fp32" });
});

test("pickRuntime without WebNN is today's WebGPU int4 / CPU int8 for texify", () => {
  delete store["latexgen.runtime"];
  nav.ml = undefined;
  nav.gpu = {};
  assert.deepEqual(pickRuntime("texify"), { device: "webgpu", dtype: "q4" });
  nav.gpu = undefined;
  assert.deepEqual(pickRuntime("texify"), { device: "wasm", dtype: "q8" });
});

test("toFloat16 round-trips a few values", () => {
  const src = new Float32Array([0, 1, -1, 0.5, 65504, 1e-8]);
  const half = toFloat16(src);
  assert.equal(half.length, src.length);
  if (typeof Float16Array !== "undefined") {
    assert.equal(+half[1], 1);
    assert.equal(+half[2], -1);
  } else {
    assert.equal(half[0], 0);
    assert.equal(half[1], 0x3c00); // 1.0
    assert.equal(half[2], 0xbc00); // -1.0
  }
});

test("Donut 4.36 resize plan: shortest-edge, thumbnail, center pad", () => {
  // 800x200 screenshot: shortest 200 -> 420x1680, thumbnail 420x105, pad top/bottom
  const wide = donutResizePlan(800, 200);
  assert.deepEqual(wide.first, { w: 1680, h: 420 });
  assert.deepEqual(wide.thumb, { w: 420, h: 105 });
  assert.deepEqual(wide.pad, { top: 157, left: 0, bottom: 158, right: 0 });
  const square = donutResizePlan(420, 420);
  assert.equal(square.first, null);
  assert.equal(square.thumb, null);
  assert.deepEqual(square.pad, { top: 0, left: 0, bottom: 0, right: 0 });
  const alreadyShort = donutResizePlan(800, 420);
  assert.equal(alreadyShort.first, null);
  assert.deepEqual(alreadyShort.thumb, { w: 420, h: 220 });
});

test("rgbaToNchw writes ImageNet-normalized NCHW", () => {
  const data = new Uint8ClampedArray(4);
  data[0] = 255; data[1] = 0; data[2] = 0; data[3] = 255;
  const out = rgbaToNchw(data, 1);
  assert.equal(out.length, 3);
  assert.ok(Math.abs(out[0] - (1 - 0.485) / 0.229) < 1e-5);
  assert.ok(Math.abs(out[1] - (0 - 0.456) / 0.224) < 1e-5);
  assert.ok(Math.abs(out[2] - (0 - 0.406) / 0.225) < 1e-5);
});

test("constants blobs are optional in git but recipes are present", () => {
  assert.equal(existsSync(join(DIR, "recipe.encoder.json")), true);
  assert.equal(existsSync(join(DIR, "recipe.decoder.json")), true);
  assert.equal(existsSync(join(DIR, "entry.json")), true);
});
