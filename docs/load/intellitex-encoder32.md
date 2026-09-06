# IntelliTeX WebNN encoder32 — catalog graph `encoder32`

Family `intellitex-t5-220m`. Constants: `intellitex-encoder-L32.bin` (236.1 MB). 553 ops, 149 constants. Used for prompts that fit the 32-token bucket.

`createIntelliTeXWebNN` compiles **encoder32+decode32, then 64, then 128** before the first `run`. A quadratic-formula prompt only needs bucket 32.

## Contended measurement

2026-09-05, Hugging Face constants, backend **coreml**. Family loadavg **29.7→34.4**, RAM **98%**. Not a quiet-machine floor.

| Graph phase | ms |
|---|---|
| constantsMs | 17341 |
| emitMs | 0.7 |
| **buildMs** | **6784** |

Family: load **147.0 s**, warmup 216 ms, first 345 ms, second 225 ms, **T4−T0 147.6 s**.

## What to cut

The 236 MB fetch took ~17.3 s; compile took ~6.8 s. Required for short prompts.

See [decode32](intellitex-decode32.md) and [README](README.md).
