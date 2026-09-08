# IntelliTeX WebNN encoder64 — catalog graph `encoder64`

Constants: `intellitex-encoder-L64.bin` (236.2 MB). Same op count as encoder32 (553 / 149). Compiled in the **second** MLContext, before first `run`, even when the prompt fits 32 tokens.

## Contended measurement

2026-09-05, backend **coreml**, contended (loadavg rose during this family).

| Graph phase | ms |
|---|---|
| constantsMs | 18512 |
| emitMs | 1.2 |
| **buildMs** | **7048** |

~18.5 s fetch + 7.0 s compile that the quadratic fixture never needed.

## What to cut

**Lazy-fetch and lazy-compile:** load encoder64/decode64 only when `bucketFor(ids)` is 64. The L64 encoder bin is distinct (~236 MB).

See [decode64](intellitex-decode64.md), [encoder32](intellitex-encoder32.md).
