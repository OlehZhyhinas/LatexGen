# IntelliTeX WebNN decode128 — catalog graph `decode128`

Same op count as decode32. Constants **reuse** `intellitex-decode-L32.bin` (~285 MB). Separate compile, third MLContext, before first `run`.

## Contended measurement

2026-09-05, backend **coreml**, contended host.

| Graph phase | ms |
|---|---|
| constantsMs | 17925 |
| emitMs | 1.1 |
| **buildMs** | **8852** |

Third full network GET of the shared 285 MB bin (~17.9 s), plus a separate compile (~8.9 s).

Buckets 64+128 cost ~65.1 s of fetch + compile that this short fixture did not need. Family T4−T0 **147.6 s**.

## What to cut

Lazy-fetch/compile decode128 and reuse the in-memory constants from decode32. In this real cold run it **was** a third Hugging Face download.

See [decode32](intellitex-decode32.md), [encoder128](intellitex-encoder128.md).
