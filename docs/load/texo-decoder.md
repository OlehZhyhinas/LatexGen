# Texo WebNN decoder — catalog graph `decoder`

Family `texo-384`. Constants: `texo-decoder-k16.bin` (10.8 MB). 1363 ops, 44 constants. Sixteen greedy decode steps per dispatch.

## Contended measurement

2026-09-05, Hugging Face constants, backend **coreml**, loadavg **33.5→34.4**. Not a quiet-machine floor.

| Graph phase | ms |
|---|---|
| constantsMs | 1760 |
| emitMs | 1.5 |
| **buildMs** | **4832** |

Family T4−T0 **12.5 s**; first image gen is **54 ms** (second **43 ms**).

## What to cut

Core ML `buildGraph` (~4.8 s) for the 1363-op recipe is larger than this graph's 10.8 MB fetch (~1.8 s).

See [encoder](texo-encoder.md).
