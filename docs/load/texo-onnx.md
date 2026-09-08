# Texo ONNX (CPU fp32) — cold → first gen

Unoptimized product fallback: WASM / CPU fp32 (not WebGPU; Texo WebGPU is not the product fallback).

## Contended measurement

2026-09-05, Hugging Face, Chrome 152, new regular profile. **Contended:** loadavg **34.4→36.7**, RAM **97%**. Not a quiet-machine floor. The first attempt hit Hugging Face HTTP 429; this is the successful fresh-profile retry.

| Phase | ms |
|---|---|
| Download | 14180 |
| Session | 483 |
| Load | 14991 |
| Internal warmup | 451 |
| First gen (`quadratic` image) | 412 |
| Second gen | 411 |
| **T4−T0** | **15854** |

Output both gens: `x = { \frac { - b \pm { \sqrt { b ^ { 2 } - 4 a c } } } { 2 a } }`

Remote bytes ~76 MB (encoder 54 MB + decoder 26 MB).

## What to cut

Fetch (~14.2 s) dominates cold start. Warmup + first gen are ~0.4 s each and stay there on the second call.

Reproduce: `bench-load.html?model=texo&device=wasm&dtype=fp32&src=hf&mode=cold&items=0` in a new profile.
