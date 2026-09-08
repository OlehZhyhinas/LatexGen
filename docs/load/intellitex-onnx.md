# IntelliTeX ONNX (WebGPU int4) — cold → first gen

Unoptimized product path: Transformers.js + ONNX Runtime Web, WebGPU int4 (`encoder_model_q4.onnx` + `decoder_model_merged_q4.onnx`).

## Contended measurement

2026-09-05, Hugging Face, Chrome 152, new regular profile. **Contended:** 1-minute loadavg **35.1** on 18 CPUs, RAM **99%**. Not a quiet-machine floor.

| Phase | ms |
|---|---|
| Download | 7851 |
| Session | 572 |
| Load (sum) | 8979 |
| Internal warmup | 335 |
| First gen (`quadratic-formula`) | 230 |
| Second gen | 251 |
| **T4−T0** | **9544** |

Output both gens: `$$x=\frac{-b\pm\sqrt{b^{2}-4ac}}{2a}$$`

Remote model bytes: ~184 MB; network instrumentation saw ~181 MB transferred.

## What to cut

Remote fetch (~7.9 s) dominates. Session, warmup, and generation were all below 0.6 s in this run.

Reproduce: `bench-load.html?model=intellitex&device=webgpu&dtype=q4&src=hf&mode=cold&items=0` in a new profile.
