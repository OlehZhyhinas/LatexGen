# Texify WebNN decoder — catalog graph `decoder`

Family `texify-420`. Constants: `texify-decoder.bin` (423.2 MB). 349 ops, 162 constants. One greedy step per dispatch over a cache of 384.

Compiled first in this cold run.

## Contended measurement

2026-09-05, Hugging Face constants, backend **coreml**, loadavg **34.4→29.7**. Not a quiet-machine or CDN floor.

| Graph phase | ms |
|---|---|
| constantsMs | 138592 |
| emitMs | 1.1 |
| **buildMs** | **12177** |

The 423 MB constants fetch took **138.6 s**, then compile took **12.2 s**. Family T4−T0 **241.3 s**; warmup adds 2.91 s.

## What to cut

Remote delivery is the first target; compile is second. The committed manifest now maps local `texify-decoder.bin` to the published `texify-decode.bin`.

See [encoder](texify-encoder.md).
