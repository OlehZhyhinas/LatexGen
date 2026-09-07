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

Raw results live in `bench/results-*.json`; a static, crawlable summary is generated into `public/benchmarks.html` by `python3 bench/build-benchmarks-page.py`.
