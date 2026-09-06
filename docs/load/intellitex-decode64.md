# IntelliTeX WebNN decode64 — catalog graph `decode64`

Same recipe shape as decode32 (753 ops, 189 constants). Constants **reuse** `intellitex-decode-L32.bin` (~285 MB). Still a **separate** Core ML compile on a **separate** MLContext.

## Contended measurement

2026-09-05, backend **coreml**, contended host.

| Graph phase | ms |
|---|---|
| constantsMs | 13940 |
| emitMs | 1.0 |
| **buildMs** | **8741** |

This was a second full Hugging Face GET of the shared decode blob, not a cache hit: ~13.9 s fetch plus ~8.7 s compile.

## What to cut

1. Do not compile decode64 until a 64-token prompt appears.
2. Hold the L32 decode constants in memory so this graph does not download 285 MB again.

See [decode32](intellitex-decode32.md), [encoder64](intellitex-encoder64.md).
