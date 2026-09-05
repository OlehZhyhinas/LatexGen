# Texify ONNX (WebGPU int4) — cold → first gen

Unoptimized product path: Transformers.js + ONNX Runtime Web, WebGPU int4.

## Contended measurement

2026-09-05, Hugging Face, Chrome 152, new regular profile. **Contended:** loadavg **35.1→34.0**, RAM **99%**. Not a quiet-machine or CDN floor.

| Phase | ms |
|---|---|
| Download | 61704 |
| Session | 458 |
| Load | 62880 |
| Internal warmup | 404 |
| First gen (`quadratic` image) | 287 |
| Second gen | 276 |
| **T4−T0** | **63571** |

Output both gens: `$$x=\frac{-b\pm\sqrt{b^{2}-4ac}}{2a}$$`

Remote model bytes ~211 MB. The ~62 s fetch dominates; the CDN served the large decoder much more slowly than the IntelliTeX objects in this same matrix.

## What to cut

CDN/object delivery of the ~170 MB decoder. Session is already short (~0.46 s).

Reproduce: `bench-load.html?model=texify&device=webgpu&dtype=q4&src=hf&mode=cold&items=0` in a new profile.
