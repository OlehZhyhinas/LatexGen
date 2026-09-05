#!/usr/bin/env node
// Turn the workbench Texify IR dumps into the catalog-shaped files LatexGen
// loads (public/models/texify-webnn/). Constants blobs are hashed in place
// and optionally hard-linked; they stay out of git (~597 MiB).
//
//   node scripts/vendor-texify-webnn.mjs \
//     [--ir-dir ../webnn-workbench-texify/bench/webnn/ir] \
//     [--link-bins]
import { readFileSync, writeFileSync, mkdirSync, existsSync, linkSync, symlinkSync, unlinkSync } from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const argv = process.argv.slice(2);
const o = {
  irDir: path.resolve(ROOT, "../webnn-workbench-texify/bench/webnn/ir"),
  out: path.join(ROOT, "public/models/texify-webnn"),
  linkBins: false,
};
for (let i = 0; i < argv.length; i++) {
  const a = argv[i], take = () => argv[++i];
  if (a === "--ir-dir") o.irDir = path.resolve(take());
  else if (a === "--out") o.out = path.resolve(take());
  else if (a === "--link-bins") o.linkBins = true;
  else throw new Error(`unknown arg ${a}`);
}

const readJson = (p) => JSON.parse(readFileSync(p, "utf8"));
const writeJson = (p, v) => writeFileSync(p, JSON.stringify(v, null, 1) + "\n");
const sha256File = (p) => {
  const h = createHash("sha256");
  h.update(readFileSync(p));
  return h.digest("hex");
};
const RECIPE_KEYS = new Set(["version", "label", "recordedLabel", "layout", "constantsBlob", "provenance", "emitterConfig", "note", "inputs", "outputs", "constants", "ops"]);

mkdirSync(o.out, { recursive: true });

function toRecipe(irLabel, graphName, renameOutputs = {}) {
  const irPath = path.join(o.irDir, `${irLabel}.json`);
  const binPath = path.join(o.irDir, `${irLabel}.bin`);
  const ir = readJson(irPath);
  const extras = Object.fromEntries(Object.entries(ir).filter(([k]) => !RECIPE_KEYS.has(k)));
  const outputs = ir.outputs.map((x) => {
    const name = renameOutputs[x.name] ?? x.name;
    return name === x.name ? x : { ...x, name };
  });
  const recipe = {
    version: 1,
    label: `texify-420/${graphName}`,
    recordedLabel: ir.label,
    layout: ir.layout ?? "nchw",
    constantsBlob: `${irLabel}.bin`,
    provenance: {
      source: { model: "vikp/texify", recordedFrom: `webnn-workbench/bench/webnn/ir/${irLabel}.json` },
      recordedFrom: {
        repo: "webnn-workbench",
        recorder: "bench/webnn/ir-record.js (Proxy around MLGraphBuilder; shapes are the backend's)",
        label: ir.label,
        harness: `node bench/webnn/texify/run.mjs --dump-ir ${ir.label === "encoder" ? "encoder" : "decode"} --port 8905`,
        recordedAt: "2026-09-05",
        constantBytes: extras.stats?.constantMiB ? Math.round(extras.stats.constantMiB * 1048576) : (existsSync(binPath) ? readFileSync(binPath).byteLength : null),
      },
    },
    note: "One operand namespace: graph inputs keep their names, constants are k<N>, op results v<N>. Multi-output ops (split) list every result in `outputs`. Decode graph outputs that shared an input name were renamed so createEntryTensors can ping-pong caches.",
    inputs: ir.inputs,
    outputs,
    constants: ir.constants,
    ops: ir.ops,
  };
  const file = path.join(o.out, `recipe.${graphName}.json`);
  writeJson(file, recipe);
  const histo = {};
  for (const op of ir.ops) histo[op.type] = (histo[op.type] ?? 0) + 1;
  const binBytes = existsSync(binPath) ? readFileSync(binPath).byteLength : 0;
  console.log(`${graphName}: ${ir.ops.length} ops, ${Object.keys(ir.constants).length} constants, blob ${(binBytes / 1048576).toFixed(1)} MiB`);
  return { recipe, histo, binPath, binBytes, irLabel };
}

const decodeRenames = { token: "tokens" };
for (let i = 0; i < 8; i++) {
  decodeRenames[`self${i}_k`] = `self${i}_k_out`;
  decodeRenames[`self${i}_v`] = `self${i}_v_out`;
}

const enc = toRecipe("texify-encoder", "encoder");
const dec = toRecipe("texify-decode", "decoder", decodeRenames);

const encBin = "texify-encoder.bin";
const decBin = "texify-decoder.bin";
const encHash = sha256File(enc.binPath);
const decHash = sha256File(dec.binPath);
if (o.linkBins) {
  for (const [src, name] of [[enc.binPath, encBin], [dec.binPath, decBin]]) {
    const dest = path.join(o.out, name);
    try { unlinkSync(dest); } catch { /* missing is fine */ }
    try { linkSync(src, dest); }
    catch { symlinkSync(src, dest); }
  }
  console.log(`linked bins into ${o.out}`);
}

const spec = (arr) => arr.map((x) => ({ name: x.name, dataType: x.dataType, shape: x.shape }));
const encOut = spec(enc.recipe.outputs).map((x, i) => ({ ...x, readable: i === 0, role: `cross-attention ${x.name.endsWith("_k") ? "K" : "V"}, decoder layer ${x.name.match(/\d+/)[0]}` }));
const decIn = spec(dec.recipe.inputs);
const decOut = spec(dec.recipe.outputs).map((x) => ({
  ...x,
  readable: x.name === "tokens",
  role: x.name === "tokens" ? "argMax token of this decode step; the only output a consumer reads" : `updated ${x.name.replace(/_out$/, "")} (bind as next dispatch's input)`,
}));

const chain = enc.recipe.outputs.map((o) => [`encoder.${o.name}`, `decoder.${o.name}`]);
const caches = [];
for (let i = 0; i < 8; i++) {
  caches.push([`self${i}_k`, `self${i}_k_out`]);
  caches.push([`self${i}_v`, `self${i}_v_out`]);
}

const family = {
  id: "texify-420",
  name: "Texify (Donut-Swin + mBART, 300M), 420x420 equation OCR",
  summary: "Texify, LatexGen's second-pass equation OCR: a Donut-Swin encoder over a 420x420 RGB image and an 8-layer mBART decoder that emits LaTeX tokens greedily. Two WebNN graphs: the encoder, which also computes the decoder's cross-attention K/V once per image, and a decode-step graph that runs one greedy step per dispatch over static self-attention caches of length 384. float16 throughout. A preprocessed image goes in, a token sequence comes out.",
  task: "image-to-latex",
  source: {
    model: "vikp/texify (fp32); LatexGen ships the Xenova/texify ONNX export as public/models/texify",
    revision: null,
    components: {
      encoder: { model: "donut-swin (depths [2,2,14,2], heads [4,8,16,32], window 5, patch 4, 420 -> 14x14 = 196 tokens x 1024)", role: "image encoder; hoists decoder EncDecAttention K/V" },
      decoder: { model: "mbart (8 layers, d_model 1024, 16 heads x 64, ffn 4096, vocab 50000, max positions 1536, scale_embedding)", role: "autoregressive LaTeX token decoder, one step per dispatch, cache 384" },
    },
    precision: "float16 throughout; LayerNorm is the native op (a hand-rolled fp16 sum-of-squares overflows at Donut-Swin stage 2). Greedy tokens identical to fp32 on 18/18 LatexGen benchmark images.",
    resolution: 420,
    steps: null,
  },
  contract: {
    graphs: {
      encoder: {
        role: "Donut-Swin encoder + the decoder's cross-attention K/V for all 8 layers, computed once per image",
        inputs: { pixel_values: { dataType: "float16", shape: [1, 3, 420, 420], layout: "nchw", source: "Donut 4.36: shortest-edge resize 420, thumbnail into 420x420, center pad 0, /255, ImageNet mean/std. Preprocessing stays in the consumer." } },
        outputs: Object.fromEntries(encOut.map((x) => [x.name, { dataType: x.dataType, shape: x.shape, role: x.role, readable: x.readable }])),
      },
      decoder: {
        role: "one greedy decode step: embedding, 8 mBART layers over static self-attention caches and the encoder K/V, final LayerNorm, lm head, argMax",
        inputs: Object.fromEntries(decIn.map((x) => [x.name, { dataType: x.dataType, shape: x.shape }])),
        outputs: Object.fromEntries(decOut.map((x) => [x.name, { dataType: x.dataType, shape: x.shape, readable: x.readable, role: x.role }])),
      },
    },
    chaining: {
      contexts: "ONE MLContext for both graphs. navigator.ml.createContext({deviceType: 'gpu'}).",
      autoregressive: {
        graph: "decoder",
        tokenInput: "token",
        positionInput: "step",
        tokensOutput: "tokens",
        caches,
        zeroCachesPerSequence: true,
        bos: 0,
        eos: 2,
        maxNewTokens: 384,
        forcedEosTokenId: 2,
        note: "runtime/loader.js autoregressive() implements the loop from these fields; forced_eos at the 384 cap is applied by the consumer (texify-webnn.js).",
      },
    },
  },
  tokenizer: {
    algorithm: "BPE ByteLevel (NougatTokenizer), vocab 50000; decoding is transformers.js AutoTokenizer skip_special_tokens, matching the ONNX Runtime path",
    files: ["../texify/tokenizer.json", "../texify/tokenizer_config.json"],
    maxLength: 384,
    produces: "decoder.tokens is what it DECODES; nothing is encoded at inference",
  },
  constants: { scope: "per-entry", note: "Encoder blob (~174 MiB) and decoder blob (~423 MiB) are the entry's; they are not in git." },
  verification: {
    case: "LatexGen's 18 image-OCR benchmark items",
    bar: "greedy token ids identical to the fp32 reference on every image",
    runner: "public/bench-images.html?only=texify-webnn",
  },
};
writeJson(path.join(o.out, "family.json"), family);

const texoEntry = readJson(path.join(ROOT, "public/models/texo-webnn/entry.json"));
const ops = [...new Set([...Object.keys(enc.histo), ...Object.keys(dec.histo)])].sort();
const entry = {
  id: "coreml-apple-m5-pro-macos26-chrome152",
  family: "texify-420",
  variant: "exact",
  created: "2026-09-05",
  producedBy: {
    workbench: "webnn-workbench texify IR 2026-09-05",
    date: "2026-09-05",
    recorder: "bench/webnn/ir-record.js",
  },
  title: "Texify (Donut-Swin + mBART, 300M): WebNN / Core ML on Apple M5 Pro",
  summary: "Donut-Swin encoder with the cross-attention K/V hoisted in, and an mBART decode-step (one greedy token per dispatch, static cache 384). 145.7 ms median per LatexGen benchmark image, tokens identical to fp32 on 18/18.",
  target: texoEntry.target,
  compat: {
    requires: { "backend.name": "coreml" },
    ops,
    notes: "Facts, not policy. `requires` is what must hold for these graphs to build and compute correctly; `target` additionally records where they were tuned and timed.",
  },
  graphs: {
    encoder: {
      recipe: "recipe.encoder.json",
      constants: "encoder",
      role: family.contract.graphs.encoder.role,
      layout: "nchw",
      ops: enc.recipe.ops.length,
      opTypes: Object.keys(enc.histo).length,
      constantCount: Object.keys(enc.recipe.constants).length,
      inputs: spec(enc.recipe.inputs),
      outputs: encOut,
    },
    decoder: {
      recipe: "recipe.decoder.json",
      constants: "decoder",
      role: family.contract.graphs.decoder.role,
      layout: "nchw",
      ops: dec.recipe.ops.length,
      opTypes: Object.keys(dec.histo).length,
      constantCount: Object.keys(dec.recipe.constants).length,
      inputs: decIn,
      outputs: decOut,
    },
    chain,
  },
  constants: "manifest.json",
  tokenizerFromFamily: true,
};
writeJson(path.join(o.out, "entry.json"), entry);

const recHash = (file) => sha256File(path.join(o.out, file));
const manifest = {
  entry: entry.id,
  family: entry.family,
  constants: {
    encoder: { file: encBin, bytes: enc.binBytes, sha256: encHash, url: null, chunks: null },
    decoder: { file: decBin, bytes: dec.binBytes, sha256: decHash, url: null, chunks: null },
  },
  recipeHashes: {
    "recipe.encoder.json": { sha256: recHash("recipe.encoder.json"), recordedFrom: "webnn-workbench/bench/webnn/ir/texify-encoder.json" },
    "recipe.decoder.json": { sha256: recHash("recipe.decoder.json"), recordedFrom: "webnn-workbench/bench/webnn/ir/texify-decode.json" },
  },
};
writeJson(path.join(o.out, "manifest.json"), manifest);
console.log(`wrote ${path.relative(ROOT, o.out)} (encoder ${encHash.slice(0, 12)}… decoder ${decHash.slice(0, 12)}…)`);
