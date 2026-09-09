# Benchmarks

Every routing decision in LatexGen comes from a measurement. The harnesses are in the repo, so the numbers can be reproduced or extended.

## Text: which model converts math to LaTeX

15 items with Wikipedia-verified reference LaTeX, four tiers of difficulty, judged for mathematical equivalence by a 27B model. Accuracy at median latency, warm, on an M5 Pro.

| Approach | Easy | Medium | Hard | Multi-line prose |
|---|---|---|---|---|
| IntelliTeX specialist | **100%** 0.8 s | **100%** 1.8 s | 50% 1.6 s | not attempted |
| Qwen3 1.7B | **100%** 0.2 s | **100%** 0.5 s | 50% 0.5 s | 0% |
| Qwen3 4B | 75% 0.5 s | 75% 1.5 s | **100%** 1.1 s | **67%** 2.7 s |
| Qwen3 0.6B | 100% 0.3 s | 25% | 25% | 0% |
| Llama 3.2 3B | 50% | 50% | 25% | 0% |
| SmolLM2 360M | 50% | 0% | 0% | 0% |
| tiny LLM segmenter + specialist | 25% or less | 0% | 0% | 0% |

What the router took from this: the specialist goes first for single equations; prose needs a 4B-class model or the server; the segmenter idea (a tiny model marking math spans for the specialist) failed because tiny models fail at instruction following, not at math.

Reproduce: open `public/bench.html` in a browser with the local server running, then `python3 bench/judge.py`.

## Text, second pass: size-matched candidates against the shipped Qwen3 (2026-09-07)

The first table only had 15 items, all clean spoken math. The second pass adds the inputs people actually paste: 30 PDF pastes of the reference passages (15 clean, 15 with two-column, footnote, ligature and running-head garbling from `pdftotext`) and 60 synthetic prose-with-math passages (PDF-style, Unicode-style and LaTeX-style math; ten are prose only, where the right answer is to change nothing). 105 items, 12 models, one greedy run each with the pipeline's exact prompts and thinking off, through Ollama at 4-bit, judged by Qwen3.8 27B. Each shipped size is compared only with candidates of its own size.

| Class | Model | easy | medium | hard | prose | PDF paste (30) | synth PDF (24) | synth unicode (18) | synth LaTeX (18) | all |
|---|---|---|---|---|---|---|---|---|---|---|
| ~1B | Qwen3 0.6B (was shipped) | 4/4 | 2/4 | 0/4 | 0/3 | 0 | 1 | 2 | 2 | 10% |
| ~1B | **Qwen3.5 0.8B** (ships) | 3/4 | 3/4 | 1/4 | 1/3 | 4 | 6 | 7 | 11 | **34%** |
| ~1B | MiniCPM5 1B | 3/4 | 4/4 | 0/4 | 1/3 | 4 | 4 | 7 | 8 | 30% |
| ~1B | Gemma 3 1B | 2/4 | 0/4 | 0/4 | 0/3 | 0 | 0 | 1 | 0 | 3% |
| ~2B | Qwen3 1.7B (was shipped) | 4/4 | 4/4 | 1/4 | 1/3 | 2 | 0 | 2 | 0 | 13% |
| ~2B | **Qwen3.5 2B** (ships) | 3/4 | 3/4 | 1/4 | 1/3 | 4 | 4 | 8 | 3 | **26%** |
| ~2B | MiniCPM5 2B | 4/4 | 1/4 | 1/4 | 3/3 | 6 | 8 | 15 | 17 | 52% |
| ~2B | Gemma 4 E2B | 2/4 | 2/4 | 3/4 | 1/3 | 9 | 10 | 16 | 18 | 58% |
| ~4B | Qwen3 4B (was shipped) | 3/4 | 1/4 | 3/4 | 2/3 | 10 | 10 | 15 | 18 | 59% |
| ~4B | **Qwen3.5 4B** (ships) | 2/4 | 2/4 | 4/4 | 2/3 | 15 | 9 | 15 | 18 | **64%** |
| ~4B | Gemma 4 E4B | 4/4 | 4/4 | 4/4 | 2/3 | 13 | 9 | 16 | 18 | 67% |
| ~4B | Gemma 3 4B | 3/4 | 3/4 | 3/4 | 2/3 | 3 | 3 | 3 | 3 | 22% |

What changed because of this: the browser ladder is now Qwen3.5 0.8B / MiniCPM5 2B / Qwen3.5 4B / Qwen3.5 9B and the server's default Ollama model is `qwen3.5:4b`. Qwen3.5 gives up a few easy single equations, which the IntelliTeX specialist answers before the LLM is asked, and gains on hard equations, prose and PDF pastes. MiniCPM5 2B had no WebLLM build, so it was quantized to q4f16_1 and compiled for WebGPU here and published at [ozhyhinas/MiniCPM5-2B-q4f16_1-MLC](https://huggingface.co/ozhyhinas/MiniCPM5-2B-q4f16_1-MLC) (build notes in that repo's card). Gemma 4 E2B scores higher still but is 5.1B raw parameters, about 3 GB at 4-bit, and needs a new MLC model type; at that size Qwen3.5 4B is the better pick. Gemma 3 4B wraps whole prose sentences in math mode, so its public instruction-following score does not carry over. Ollama's `qwen3:4b` tag resolves to the 262K-context 2507 Thinking build, not the shipped model; the Qwen3 4B row uses the original weights from the official Qwen GGUF with the `qwen3:1.7b` tag's template (`ollama create qwen3-4b-orig`).

Reproduce: `node bench/run-ollama-text.mjs --data bench/bench-data-extended.json --models <ollama tags> --out bench/results-x.json`, then `python3 bench/judge.py --rows bench/results-x.json --data bench/bench-data-extended.json bench/judged-x.json` and `python3 bench/summarize-text-tiers.py bench/judged-x.json`. Raw rows and verdicts for this run: `bench/results-text-tiers-2026-09-07.json`, `bench/judged-text-tiers-2026-09-07.json`.

## Images: which OCR model reads rendered math

18 rendered images (KaTeX via headless Chrome) across easy, medium, hard, mixed prose and degraded (tiny font, dark background).

| Model | Size | Easy | Medium | Hard | Prose + math | Degraded | Median |
|---|---|---|---|---|---|---|---|
| Texo | 77 MB | 100% | 75%* | 80% | 67% | 50%* | 0.7 s |
| Texify | 305 MB | 50%** | 100% | 100% | 100% | 0% | 3.5 to 4 s |

\* Both misses were the KaTeX-only alias `\infin`, which the app now canonicalizes to `\infty`.
\*\* Texify repeats one line to the token cap on very sparse images; the app collapses exact repeats.

Result: Texo first, Texify on syntax failure or detected prose, plus a manual "read with the other model" button for wrong-but-valid readings such as matrices.

Reproduce: `python3 bench/render-images.py`, open `public/bench-images.html`, then `python3 bench/judge-images.py`.

## Runtime: how to run the same models faster

Every device and precision the runtime offers, one configuration per page load (a failed session poisons the ONNX runtime for the rest of the page).

| Model | CPU int8 | WebGPU int4 | Notes |
|---|---|---|---|
| IntelliTeX | 1.28 s | **0.45 s** | identical outputs; fp16 on GPU corrupted a formula; see the WebNN row |
| IntelliTeX, hand-built WebNN graphs (graph catalog `intellitex-t5-220m`) | | **141 ms** median on 12 single-line text items | Core ML backend only; measured in this repo (`bench.html?only=intellitex-webnn`, 2026-09-05). Catalog workbench figure was 146.2 ms on 15 items. Graph build ~124 s once (one MLContext per length bucket 32/64/128). |
| Texify | 8.0 s | **0.81 s** | fp16 on GPU produced garbage; int4 correct; see the WebNN row |
| Texify, hand-built WebNN graphs (graph catalog `texify-420`) | | **204 ms** median per image | Core ML backend only; measured in this repo (`bench-images.html?only=texify-webnn`, 2026-09-05, 18 images). Catalog workbench figure was 146 ms. Sparse images still hit the token-cap repeat; the app collapses those. |
| Texo | 0.77 s fp32 | 0.78 s | overhead-bound on ONNX Runtime; see the WebNN row |
| Texo, hand-built WebNN graphs (graph catalog `texo-384`) | | **31 ms** median per image, 19 ms warm in the worker | Core ML backend only; identical output to the fp32 reference on 18/18 images |

int4 on the CPU was 10x slower than int8 (no fast WASM kernel), so int4 is GPU-only. Graph fusion (O2) produced contrib ops the browser runtime cannot load.

The WebNN rows are not ONNX Runtime at all: `public/texo-webnn.js`,
`public/texify-webnn.js` and `public/intellitex-webnn.js` replay recipes from
the graph catalog through the catalog's vendored loader. Texo (`texo-384`) is
an HGNetv2 encoder that also emits the decoder's cross-attention K/V plus a
decode graph of 16 greedy steps per dispatch; Texify (`texify-420`) is a
Donut-Swin encoder with the same K/V hoist plus one greedy decode step per
dispatch over a cache of 384; IntelliTeX (`intellitex-t5-220m`) is a T5
encoder-decoder with three static input-length buckets. The 31 ms Texo figure
is `bench-images.html?only=texo-webnn` (canvas preprocess included,
3.8 s to build the graphs once per page load); the 19 ms is a warm second call
through the product's worker. The 204 ms Texify figure is this repo's
`bench-images.html?only=texify-webnn` over the 18 images (workbench e2e was
146 ms). The 141 ms IntelliTeX figure is `bench.html?only=intellitex-webnn` on
the 12 single-line specialist items (catalog e2e was 146.2 ms on 15). All three
families were built in `webnn-workbench`. Chrome 152 with the WebNN flags and a
persistent profile; an off-the-record profile silently lands on a TFLite CPU
path, which the loaders reject so the app falls back to ONNX Runtime and
remembers to.

Reproduce: open `public/bench-runtime.html?i=0` (the WebNN configs are appended
when `navigator.ml` exists), or `public/bench-images.html?only=texo-webnn` /
`?only=texify-webnn` for all 18 images, or `public/bench.html?only=intellitex-webnn`
for the 15 text items.

### WebLLM: Qwen3 through the graph catalog's WebGPU runtime

The same Qwen3 q4f16_1 weights, decoded by the catalog's patched WebLLM 0.2.84
(family `qwen3-webllm`: device-resident greedy argmax, K-token decode bursts,
batched command encoding with periodic flushes, optional lookahead) with a
per-model subgroup-32 model lib, against stock WebLLM 0.2.84. Both sides
greedy (`temperature: 0`), streamed, the 15 text items, two rounds, arms
interleaved per round, fresh engine per arm. Decode is (wall − first token) /
(tokens − 1); medians over the 30 runs. M5 Pro, Chrome 152, 2026-09-07,
`bench-webllm.html`.

| Model | Variant | Decode ms/token | Speed-up | First token | Same output as stock |
|---|---|---|---|---|---|
| Qwen3 0.6B | stock WebLLM | 7.0 | | 42 ms | |
| | catalog `sg32-burst4` | 4.2 | 1.69x | 42 ms | 14/15 items |
| | catalog `sg32-burst4-flush64` **(default)** | **3.5** | **2.01x** | 40 ms | 14/15 items |
| Qwen3 1.7B | stock WebLLM | 9.8 | | 103 ms | |
| | catalog `sg32-burst4-flush32` | 6.2 | 1.58x | 102 ms | 13/15 items |
| | catalog `sg32-burst1-flush32-lookahead1` **(default)** | **5.8** | **1.68x** | 105 ms | 13/15 items |
| Qwen3 4B | stock WebLLM | 17.7 | | 254 ms | |
| | catalog `sg32-burst4-flush32` | **12.2** | **1.45x** | 252 ms | 15/15 items |
| Qwen3 8B | stock WebLLM | 25.3 | | 457 ms | |
| | catalog `sg32-burst4-flush32` | **20.8** | **1.22x** | 466 ms | 14/15 items |

What the router took from this: every size is faster through the catalog, most
where it matters least (0.6B) and least where it matters most (8B), because the
tricks remove per-token overhead rather than compute. The sibling pairs were
settled here: `flush64` over plain `burst4` at 0.6B, and `lookahead1` over
`burst4-flush32` at 1.7B, where the workbench's +3.9 ms first-token cost did
not show (105 vs 102 ms) and per-token streaming comes back for free. The
losing siblings stay in the catalog and can be forced with
`?webllm=catalog:<variant>`.

The output differences are not the decode tricks: the two 1.7B variants agree
with each other on all 30 runs and differ from stock on the same two items, so
the divergence comes from the subgroup model lib producing slightly different
logits, which flips one near-tie token (0.6B: `\hat{\lambda}` vs `\lambda`;
8B: `k = 0` vs `k` equals zero`). Every divergence reproduced identically in
both rounds. Judged by the 27B (`bench/judge.py`,
`bench/results-qwen3-webllm-2026-09-07.json`), the catalog and stock arms
score identically on every item for every size (0.6B 6/15, 1.7B 8/15, 4B 9/15,
8B 9/15): the flipped tokens change wording, not correctness.

Reproduce: `public/bench-webllm.html?model=Qwen3-1.7B-q4f16_1-MLC&rounds=2`
(arms default to stock plus every catalog variant for the model), then
`python3 bench/judge.py` for the accuracy column.

Everything above is the Qwen3 generation, and its conclusion does not carry to
Qwen3.5. Qwen3.5 4B and 9B have recurrent (space-state) layers. The decode
burst and the lookahead both issue more steps than the CPU has consumed and pop
the surplus positions back at stop time with `vm.builtin.kv_state_popn`; a
recurrent state keeps no per-token history, so the pop fails its own check and
TVM aborts the WASM runtime:

```
rnn_state.cc:366 RNNStateImpObj::PopN
InternalError: Check failed: n <= it->second.available_history_num (1 vs. 0)
```

The pop size tracks the over-fill exactly (`greedyBurst: 4` pops 3, `lookahead:
1` pops 1), so this is not specific to lookahead. It is also not a visible
crash. On 4B the tokens stream out correctly first and the abort lands on the
stop-time rollback, leaving the engine dead — so `pipeline.js`'s
validator-guided repair loop fails into its `catch` and the *un-repaired*
first-pass LaTeX is what the user sees. On 9B the abort arrives mid-generation
and truncates the output. This is why an "optimized" run could score worse than
an unoptimized one: not worse decoding, skipped correction.

Measured warm on M5 Pro / Chrome 152, 2026-09-09, one paste per cell, comparing
the tuned model lib against the stock lib for the same weights:

| Model | stock lib | as shipped before this fix | `burst1 + flush32` |
|---|---|---|---|
| Qwen3.5 0.8B | 41.2 tok/s | 53.3 (`burst5-flush32`) | 113.8 |
| MiniCPM5 2B | 53.6 | 109.5 (`burst1-flush32-lookahead1`) | 81.1 |
| Qwen3.5 4B | 27.9 | **aborts** | **64.2** |
| Qwen3.5 9B | 20.4 | **aborts**, output truncated | **39.6** |

So the subgroup model lib is where the speed is, and the popping knobs were not
paying for themselves on the two models they break: 4B reaches 64.2 tok/s with
`greedyBurst: 1`, against the 64.1 the aborting lookahead variant was credited
with. The two recurrent models now default to `sg32-burst1-flush32`, and
`planEngine` clamps `greedyBurst` to 1 and drops `lookahead` for any model
flagged `recurrentState`, so a hand-picked variant row or a forced
`?webllm=catalog:<variant>` cannot reintroduce the abort. The pop-free knobs
(`batchPass`, `flushEvery`, `bindGroupCache`) are untouched.

MiniCPM5 2B and Qwen3.5 0.8B are not recurrent and keep their variants. The
0.8B number above says its shipped `burst5` default may be leaving a lot on the
table, but that is one paste against a workbench figure measured differently
(190.8 vs 108.9 tok/s), so it is not settled here.

## Loading: getting the weights into the browser

Cold means an empty Cache API and a bypassed HTTP cache, weights streamed from the Hugging Face repo the static build uses; warm means the Cache API. Median of three cold runs for the WebGPU rows, single runs elsewhere. M5 Pro, Chromium 148, one evening, both sides measured within the same two hours.

| Model, runtime | Before: bytes | cold | warm | After: bytes (on the wire) | cold | warm |
|---|---|---|---|---|---|---|
| IntelliTeX, WebGPU int4 | 324 MB | 14.3 s | 1.0 s | 184 MB (182) | **6.8 s** | 1.2 s |
| IntelliTeX, CPU int8 | 263 MB | 12.4 s | 1.0 s | 263 MB (183) | **7.6 s** | 1.4 s |
| Texify, WebGPU int4 | 362 MB | 193 s | not cached: 255 s | 211 MB (182) | **12.1 s** | 3.0 s |
| Texify, CPU int8 | 306 MB | 114 s | 5.2 s | 306 MB (220) | **13.7 s** | 3.0 s |

Three things changed, and one thing was found:

- **The int4 files were mostly fp32.** `MatMulNBits` only rewrites MatMul weights; the token-embedding tables feed a `Gather` and stayed fp32, which made them 53 to 65% of every `_q4` file, so the WebGPU download was *larger* than the CPU one. `scripts/quantize-embeddings.py` stores them as uint8 with a per-row scale and zero point. Outputs: Texify identical on all 18 images; IntelliTeX identical on 14 of 15 text items under deterministic CPU decoding, the exception being the trailing clause of a prose passage the specialist gets wrong either way. For scale, the WebGPU int4 path itself produced different outputs on 4 of the 15 items between two runs of *unchanged* weights.
- **Nothing was compressed on the wire.** Neither `server.js` nor the Hugging Face CDN negotiates `Content-Encoding`. The int8 weights gzip to about 70%, the JSON to about 20%, int4 and fp32 barely at all. `scripts/compress-models.mjs` writes `.gz` siblings where it pays and a manifest; the service worker swaps them in and inflates in flight, keeping the real `Content-Length` so progress stays honest. This is what moves the CPU rows (the int8 files did not change).
- **A 328 MB file did not fit the Cache API.** In this browser `cache.put` fails for bodies above roughly 250 MB, so the old Texify decoder was downloaded again on every visit: the "warm" run was slower than the cold ones. At 170 MB it caches, hence 3.0 s.
- **CDN throughput is per object.** The old Texify objects streamed at about 2 MB/s while everything else came down at 15 to 25 MB/s; the new objects are fast. Measured back to back after the change, the old revision took 47 s for Texify (380 MB at 8.5 MB/s) against 12 s for the new files (182 MB at 17 MB/s), and 18 s for IntelliTeX against 7 s. So the Texify rows overstate what fewer bytes alone buy; the IntelliTeX rows, whose objects were fast on both sides, are the clean signal: about 2x.

Reproduce: `public/bench-load.html?label=x&queue=intellitex,webgpu,q4,hf,cold,1;...` with the local server running (see the file header for the queue syntax), then `curl -s localhost:8000/api/bench > bench/results-load-<date>.json` and `python3 bench/summarize-load.py` on it. The rows behind this table are `bench/results-load-2026-09-03.json`.

Local cold start → first generation (ONNX fallbacks and WebNN per graph), including host loadavg on a shared machine: [docs/load/README.md](load/README.md).

Raw results live in `bench/results-*.json`; a static, crawlable summary is generated into `public/benchmarks.html` by `python3 bench/build-benchmarks-page.py`.
