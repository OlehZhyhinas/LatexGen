# IntelliTeX WebNN decode32 — catalog graph `decode32`

Family `intellitex-t5-220m`. Constants: `intellitex-decode-L32.bin` (284.8 MB). 753 ops, 189 constants.

decode64 and decode128 **reuse this same bin** (same `constantBytes` 298_684_504) but still compile their own graphs.

## Contended measurement

2026-09-05, backend **coreml**, contended host (see [README](README.md)). First graph in the 32-bucket context.

| Graph phase | ms |
|---|---|
| constantsMs | 21203 |
| emitMs | 1.3 |
| **buildMs** | **9591** |

First fetch of the 285 MB blob took ~21.2 s; compile took ~9.6 s. The true cold network log fetched the full blob three times (once per bucket), transferring ~854 MB for this one unique file.

## What to cut

Cache/share the constants response in memory so buckets 64/128 do not issue two more network GETs; also defer their separate compiles.

See [encoder32](intellitex-encoder32.md), [decode64](intellitex-decode64.md), [decode128](intellitex-decode128.md).
