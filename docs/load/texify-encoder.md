# Texify WebNN encoder — catalog graph `encoder`

Family `texify-420`. Constants: `texify-encoder.bin` (174.4 MB). 1049 ops, 309 constants.

In this run the loader compiled **decoder first**, then encoder.

## Contended measurement

2026-09-05, Hugging Face constants, backend **coreml**, loadavg **34.4→29.7**. Not a quiet-machine or CDN floor.

| Graph phase | ms |
|---|---|
| constantsMs | 80054 |
| emitMs | 1.8 |
| **buildMs** | **7330** |

Family: load 238.2 s, warmup 2.91 s, first 176 ms, second 160 ms, **T4−T0 241.3 s**.

## What to cut

The 174 MB fetch took ~80.1 s; compile took ~7.3 s. Delivery dominates this graph.

See [decoder](texify-decoder.md).
