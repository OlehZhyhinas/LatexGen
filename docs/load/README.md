# Cold start → first generation

These notes cover a real first visit for each in-browser specialist: a **brand-new regular Chrome profile per row**, empty browser/Cache API state, weights fetched from Hugging Face, graph/session creation, warmup, then one fixture generation. Recipes and benchmark fixtures are served locally; model weights and WebNN constants are remote.

Headline number **T4−T0** = load + internal warmup + first fixture gen. A second gen is recorded so first-token compile/warmup is visible.

## Contended host (read this first)

This machine is shared with other agents. These rows were taken with `os.loadavg()[0]` around **30–37** on **18** logical CPUs, RAM about **97–99%** used, and **~38 Chrome Helper** processes already running before each isolated Chrome. The runner flags a run `contended` when 1-minute loadavg exceeds 60% of CPU count.

**These numbers are not a quiet-machine floor.** Core ML graph compile and ONNX session creation compete for CPU, GPU, and memory with other work. A quiet rerun should be faster, especially on the WebNN `buildMs` columns. Do not treat a later quiet number as a regression of this table, or this table as the best the hardware can do.

How the harness records that:

- `POST /api/bench` stamps `host.loadavg` and `host.memUsedFrac`.
- `bench/run-load-cold.mjs` stamps `runnerHost.before/after` (loadavg, `cpuCount`, Chrome Helper count, `contended`).
- Every row gets a new regular `--user-data-dir`; it is removed after Chrome exits. This prevents HTTP, Cache API, service worker, and model state from leaking between rows without using incognito (which can select TFLite instead of Core ML).

Raw rows: `bench/results-load-cold-hf-2026-09-05.json`, plus the successful Texo ONNX retry in `bench/results-load-cold-hf-texo-retry-2026-09-05.json` after the first request hit Hugging Face HTTP 429. Chrome 152, Apple M5 Pro, 2026-09-05.

## Matrix (one cold run each)

| Path | Runtime | Remote model bytes | Load | Warmup | First gen | Second gen | **T4−T0** | 1m loadavg |
|---|---|---|---|---|---|---|---|---|
| [IntelliTeX ONNX](intellitex-onnx.md) | WebGPU int4 | 184 MB | 9.0 s | 0.34 s | 0.23 s | 0.25 s | **9.5 s** | 35.1 |
| [Texify ONNX](texify-onnx.md) | WebGPU int4 | 211 MB | 62.9 s | 0.40 s | 0.29 s | 0.28 s | **63.6 s** | 35.1→34.0 |
| [Texo ONNX](texo-onnx.md) | CPU fp32 | 76 MB | 15.0 s | 0.45 s | 0.41 s | 0.41 s | **15.9 s** | 34.4→36.7 |
| Texo WebNN ([encoder](texo-encoder.md), [decoder](texo-decoder.md)) | Core ML | 42 MB | 12.2 s | 0.29 s | 0.05 s | 0.04 s | **12.5 s** | 33.5→34.4 |
| Texify WebNN ([encoder](texify-encoder.md), [decoder](texify-decoder.md)) | Core ML | 598 MB | 238.2 s | 2.91 s | 0.18 s | 0.16 s | **241.3 s** | 34.4→29.7 |
| IntelliTeX WebNN (six graphs) | Core ML | 994 MB unique; 1.56 GB transferred | 147.0 s | 0.22 s | 0.34 s | 0.23 s | **147.6 s** | 29.7→34.4 |

Backend for every WebNN row was **`coreml`**, not TFLite.

WebNN per-graph **constantsMs / emitMs / buildMs** come from catalog `loadEntry`. Product IntelliTeX compiles **all three length buckets** before the first `run`. Although decode64/decode128 name the same `intellitex-decode-L32.bin` (~285 MB), the true cold network log contains **three full GETs**. It transferred ~1.56 GB for ~994 MB of unique constants and compiled each decode graph separately (~9 s each).

`downloadMs` is the progress-callback span, not a reliable family fetch total for WebNN. Use per-graph `constantsMs`: Texo 6.73 s total, Texify 218.65 s, IntelliTeX 98.56 s. `buildMs` totals were 5.39 s, 19.51 s, and 48.40 s respectively.

## Reproduce

Dedicated checkout on port 8001 (do not steal the product server):

```bash
PORT=8001 npm start
PORT=8001 node bench/run-load-cold.mjs
python3 bench/summarize-load.py bench/results-load-cold-hf-YYYY-MM-DD.json
```

Single path: `http://127.0.0.1:8001/bench-load.html?label=cold-hf&model=intellitex&device=webnn&dtype=fp16&src=hf&mode=cold&items=0` in a newly created regular Chrome profile. Fixture is `quadratic-formula` (text) or `quadratic` (image).

## GitHub issues

Filed against these numbers (enhancement):

| Issue | Target |
|---|---|
| [#14](https://github.com/OlehZhyhinas/LatexGen/issues/14) | IntelliTeX ONNX |
| [#15](https://github.com/OlehZhyhinas/LatexGen/issues/15) | Texify ONNX |
| [#16](https://github.com/OlehZhyhinas/LatexGen/issues/16) | Texo ONNX |
| [#17](https://github.com/OlehZhyhinas/LatexGen/issues/17) | Texo encoder |
| [#18](https://github.com/OlehZhyhinas/LatexGen/issues/18) | Texo decoder |
| [#19](https://github.com/OlehZhyhinas/LatexGen/issues/19) | Texify encoder |
| [#20](https://github.com/OlehZhyhinas/LatexGen/issues/20) | Texify decoder |
| [#21](https://github.com/OlehZhyhinas/LatexGen/issues/21) | IntelliTeX encoder32 |
| [#22](https://github.com/OlehZhyhinas/LatexGen/issues/22) | IntelliTeX decode32 |
| [#23](https://github.com/OlehZhyhinas/LatexGen/issues/23) | IntelliTeX encoder64 (lazy) |
| [#24](https://github.com/OlehZhyhinas/LatexGen/issues/24) | IntelliTeX decode64 (lazy; shared decode bin) |
| [#25](https://github.com/OlehZhyhinas/LatexGen/issues/25) | IntelliTeX encoder128 (lazy) |
| [#26](https://github.com/OlehZhyhinas/LatexGen/issues/26) | IntelliTeX decode128 (lazy; shared decode bin) |

