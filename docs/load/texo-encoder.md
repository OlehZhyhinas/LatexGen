# Texo WebNN encoder — catalog graph `encoder`

Family `texo-384`, entry `coreml-apple-m5-pro-macos26-chrome152`. Constants: `texo-encoder.bin` (33.4 MB). 155 ops, 168 constants.

Product `createTexoWebNN` compiles encoder **and** decoder before first `run`. Encoder `buildMs` is the smaller of the two.

## Contended measurement

2026-09-05, Hugging Face constants, backend **coreml**. Family loadavg **33.5→34.4**. Not a quiet-machine floor.

| Graph phase | ms |
|---|---|
| constantsMs | 4973 |
| emitMs | 0.5 |
| **buildMs** | **558** |

Family wall clock: load 12196 ms, warmup 293 ms, first 54 ms, second 43 ms, **T4−T0 12543 ms**.

## What to cut

The 33 MB remote constants fetch (~5.0 s), then ~0.6 s compile. The encoder is required for every image.

See also [decoder](texo-decoder.md). Family notes: [README](README.md).
