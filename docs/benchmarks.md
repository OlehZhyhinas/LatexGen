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
| IntelliTeX | 1.28 s | **0.45 s** | identical outputs; fp16 on GPU corrupted a formula |
| Texify | 8.0 s | **0.81 s** | fp16 on GPU produced garbage; int4 correct |
| Texo | 0.77 s fp32 | 0.78 s | overhead-bound, stays on CPU |

int4 on the CPU was 10x slower than int8 (no fast WASM kernel), so int4 is GPU-only. Graph fusion (O2) produced contrib ops the browser runtime cannot load.

Reproduce: open `public/bench-runtime.html?i=0`.

Raw results live in `bench/results-*.json`; a static, crawlable summary is generated into `public/benchmarks.html` by `python3 bench/build-benchmarks-page.py`.
