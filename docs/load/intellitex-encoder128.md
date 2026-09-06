# IntelliTeX WebNN encoder128 — catalog graph `encoder128`

Constants: `intellitex-encoder-L128.bin` (236.4 MB). 553 ops. Compiled in the **third** MLContext before first `run`. Unused for the quadratic-formula fixture.

## Contended measurement

2026-09-05, Hugging Face constants, backend **coreml**, contended (family loadavg **29.7→34.4**).

| Graph phase | ms |
|---|---|
| constantsMs | 9634 |
| emitMs | 0.9 |
| **buildMs** | **7386** |

~9.6 s fetch + 7.4 s compile the first short prompt does not need.

## What to cut

Lazy-fetch and lazy-compile with the other unused buckets. This encoder has a distinct constants file.

See [decode128](intellitex-decode128.md), [encoder32](intellitex-encoder32.md).
