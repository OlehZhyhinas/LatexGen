# LatexGen

Text → LaTeX converter that runs (almost) entirely on the user's machine.
Type math in plain English — *"the integral from 0 to infinity of e to the
minus x squared"* — and get validated, rendered LaTeX back in well under a
second, without the input leaving the browser for most conversions.

## Architecture: a benchmark-driven escalation ladder

Every conversion walks a ladder of increasingly capable (and expensive)
tiers, and only climbs when the previous tier's output fails validation:

1. **Specialist (in-browser, ~0.2–0.8s)** — [IntelliTeX](https://huggingface.co/duanxianpi/IntelliTex)
   (CodeT5+ 220M), int8-quantized ONNX via transformers.js. Handles
   single equations up to ~220 chars. No GPU needed — runs on CPU/WASM, so
   phones and Firefox get this tier too.
2. **General browser LLM (WebGPU)** — WebLLM running a Qwen3-family model.
   A progressive ladder auto-loads Qwen3-0.6B within seconds of page open,
   then hot-swaps to the best model the device can hold (detected from GPU
   budget **and** storage quota). Users can override via a curated picker.
3. **Server escalation** — a local Ollama/MLX backend (self-hosted mode) or
   a Claude API Lambda (AWS mode). Prose/multiline inputs skip tier 2 when
   the loaded browser model is below 4B-class and go straight here.

Every output passes two checks before it's accepted: KaTeX parses every math
segment, and a fidelity pass verifies the numbers/concepts named in the input
actually appear in the LaTeX. A chat-style **Refine** box lets users correct
results conversationally ("change TV to T(V)"); refinements route to the
strongest available model.

### The payoff matrix that shaped the routing

15 items (Wikipedia-referenced), judged by a local 27B (2026-09-01, full
data in `bench/`):

| Approach | Easy | Medium | Hard | Multiline |
|---|---|---|---|---|
| IntelliTeX specialist | **100%** @ 0.8s | **100%** @ 1.8s | 50% @ 1.6s | — |
| Qwen3-1.7B direct | **100%** @ 0.2s | **100%** @ 0.5s | 50% @ 0.5s | 0% |
| Qwen3-4B direct | 75% @ 0.5s | 75% @ 1.5s | **100%** @ 1.1s | **67%** @ 2.7s |
| Qwen3-0.6B direct | 100% @ 0.3s | 25% | 25% | 0% |
| SmolLM2-360M direct | 50% | 0% | 0% | 0% |
| tiny-LLM segmenter + specialist | ≤25% | 0% | 0% | 0% |

Consequences baked into the router: the specialist goes first for single
equations; multiline needs 4B+ or the server; SmolLM and the
segmenter-pipeline idea were dropped (tiny models fail at *instruction
following*, not math). Rerun the benchmark with `public/bench.html` +
`bench/judge.py`.

## What the app does beyond converting

- **Batch lines:** paste several equations, one per line, and each is converted
  by the specialist separately (a structural split on your newlines, not a
  heuristic); the result is one block per line.
- **Copy formats:** LaTeX as-is, display (`\[ \]`), inline (`\( \)`), MathML
  (pastes as a live equation into Word and Google Docs), PNG, or open in
  Overleaf.
- **History and favorites** in this browser only (localStorage), restorable
  with one click.
- **Strict mode** (settings): a second model judges whether the LaTeX says
  what you typed and triggers the repair loop if not. Catches grouping
  errors the heuristic checks cannot.
- **First-visit choice:** Quick (specialists only, ~340 MB) or Full (adds the
  on-device language model); changeable in settings. Downloads are
  prioritized so the specialist is ready before anything else starts.
- **No third-party requests:** every library and font is vendored under
  `public/vendor/` and served from the app's own origin; only model weights
  for the optional WebLLM tier come from HuggingFace. WebLLM runs in a Web
  Worker so the UI never freezes during multi-GB loads.
- **Installable, offline PWA:** `manifest.webmanifest` + `sw.js` cache the app
  shell and vendored libraries; with models already cached it works with the
  network off. Model weights are cached by the libraries themselves, not
  double-cached by the worker.
- **Visual equation editor:** the "✎ Visual editor" toggle on the LaTeX panel
  opens the result in [MathLive](https://cortexjs.io/mathlive/) for WYSIWYG
  editing; the LaTeX, preview, and checks follow every edit.
- **Share links with no server:** "Copy share link" puts the LaTeX in the URL
  fragment (`/#l=…`), which never leaves the browser; opening the link renders
  it.
- **Check LaTeX mode:** paste existing LaTeX to parse, render, and (with an
  on-device model loaded) repair it.
- `public/benchmarks.html` is a static, crawlable page of the benchmark
  results, regenerated with `python3 bench/build-benchmarks-page.py`.

## Tab API: use an open tab from curl, Postman, Shortcuts, Zapier

A browser tab cannot accept connections, so LatexGen turns the problem around:
with **Tab API** enabled (the plug icon in the header), the tab long-polls the
server for jobs and answers them with the models it already has loaded. Each
tab has a stable, unguessable address (128-bit token in localStorage;
*regenerate* revokes it). The API drawer shows the address, live status,
running jobs, and a request log.

Jobs run on the same headless pipeline the buttons use (`public/pipeline.js`),
so the API is one-to-one with the UI — and they run **in the background,
concurrently** (up to 3 at once), without touching what the user sees. ONNX
inference lives in a Web Worker (`onnx-worker.js`), WebLLM in another, so the
page never freezes for either the user or the API.

```bash
B=https://<host>/api/tab/<id>
curl -X POST $B/convert -H 'content-type: application/json' \
  -d '{"text": "the sum from n equals 1 to infinity of 1 over n squared"}'
# {"latex":"$$\\sum_{n=1}^{\\infty}\\frac{1}{n^{2}}$$","ok":true,"issues":[],"model":"IntelliTeX · specialist","ms":412,...}
```

| Endpoint | Body | Mirrors |
|---|---|---|
| `POST /convert` | `{"text"}` — options `format` (`latex`\|`display`\|`inline`\|`mathml`\|`png`), `strict: true` (waits for the second-opinion judge, returned as `judge`), `engine` (`browser`\|`server`) | Convert button, incl. batch lines, escalation, repair |
| `POST /convert` | `{"imageBase64", "mime", "ocr": "auto"\|"texo"\|"texify"}` | Image drop / paste, "read with other model" |
| `POST /convert` or `/check` | `{"latex"}` | Check LaTeX tab (validate + repair) |
| `POST /refine` | `{"latex", "instruction", "original"}` | Refine chat |
| `POST /format` | `{"latex", "format"}` | Copy menu formats |
| `POST /status` | `{}` | Settings drawer: models loaded, runtime, speed, server |
| `POST /history` | `{}` | History drawer |

Every response carries the validator verdict (`ok`, `issues`), which tier
answered (`model`), the routing note, and timing. The relay forwards bytes
only (no inference on the server) and holds each request up to 25s. In-memory
relay in `server.js`; the AWS build needs a small table-backed equivalent.

## In-browser runtime (Level A results)

Every (model × device × dtype) the runtime offers was measured with
`bench-runtime.html` (one config per page load, because a failed session
poisons the ONNX runtime for the rest of the page). Warm, on an M5 Pro:

| Model | CPU int8 | WebGPU int4 | Notes |
|---|---|---|---|
| IntelliTeX (text specialist) | 1.28s | **0.45s** | identical outputs; fp16 on GPU corrupted a formula |
| Texify (image tier 1) | 8.0s | **0.81s** | fp16 on GPU produced garbage; int4 correct |
| Texo (image tier 0) | 0.77s fp32 | 0.78s | overhead-bound; stays on CPU fp32 |

int4 on the CPU is ~10x *slower* than int8 (no fast WASM kernel), so int4 is
GPU-only. The app picks WebGPU int4 when `navigator.gpu` exists, remembers a
failed WebGPU session per model in localStorage, and uses CPU int8 on the next
load. The int4 weights are build artifacts (`scripts/build-model-variants.sh`),
not committed, to stay under the LFS quota.

## Local run (self-hosted, fully offline-capable)

Requires Docker and [Ollama](https://ollama.com) on the host.

```bash
ollama pull qwen3:1.7b            # fast conversion tier
ollama pull qwen3.8:27b-q8_0      # strong tier (refinements + prose) — or any big model
docker compose up --build
# open http://localhost:8013
```

Optional faster tier on Apple Silicon (auto-detected within ~20s):

```bash
scripts/run-mlx.sh
```

Env vars (docker-compose.yml): `OLLAMA_URL`, `OLLAMA_MODEL`,
`OLLAMA_REFINE_MODEL`, `MLX_URL`, `MLX_MODEL`, `MLX_REFINE_MODEL`, `PORT`.
The server probes both backends every ~20s: simple conversions go to MLX
when present (fastest decode on Apple Silicon), refinements and prose
passages to the strongest model. All responses stream as NDJSON; Ollama
models stay pinned (`keep_alive: -1`) with a boot-time warm-up.

To share a local instance temporarily: `cloudflared tunnel --url http://localhost:8013`.

## AWS deployment (static + API escalation)

```bash
AWS_PROFILE=<personal-profile> deploy/deploy.sh
```

Creates: a private S3 bucket (CloudFront-only via Origin Access Control),
a CloudFront distribution (static site + `/api/*` → Lambda), and the
escalation Lambda (`deploy/lambda/index.mjs`). The script refuses to run
against any account other than the one pinned in `EXPECTED_ACCOUNT`.

The Lambda holds the Claude API key **server-side only** — nothing sensitive
ships to the frontend. It deploys with *no key configured*: escalation
returns 503, `/api/health` reports no routes, and the frontend silently runs
browser-only. Enable escalation later with:

```bash
aws lambda update-function-configuration --function-name latexgen-api \
  --environment 'Variables={ANTHROPIC_API_KEY=sk-...}'
```

Idle cost is ~$1–5/month (S3/Lambda free tier; CloudFront egress — mostly
the 264MB specialist download per new user — is the variable part).

## Repo layout

```
public/            frontend (vanilla JS, no build step)
  app.js           UI wiring only: drawers, picker, ladder, Tab API client + drawer
  pipeline.js      headless conversion ladder shared by the UI and the Tab API
  models.js        main-thread client for onnx-worker.js (runtime selection, progress)
  onnx-worker.js   IntelliTeX / Texo / Texify inference off the main thread
  webllm-worker.js WebLLM engine host (Web Worker)
  vendor/          pinned local copies of WebLLM, transformers.js + ONNX runtime, KaTeX, fonts
  benchmarks.html  static benchmark results page (generated)
  sw.js, manifest  PWA: offline app shell, install metadata, icons/
  validator.js     KaTeX syntax + input-fidelity checks
  models/          ONNX weights via git-lfs: IntelliTeX (264MB), Texo (77MB), Texify (305MB)
  bench.html/.js   in-browser text benchmark harness
  bench-images.*   in-browser image-OCR benchmark harness + rendered eval images
server.js          zero-dependency Node proxy (Ollama/MLX routing, NDJSON)
deploy/            AWS: deploy.sh + Lambda escalation proxy
bench/             eval datasets, judges, image renderer, results, benchmarks-page generator
tests/             validator unit tests (npm test); CI in .github/workflows
scripts/           MLX runner, specialist ONNX conversion
```

## Credits

- [IntelliTeX](https://huggingface.co/duanxianpi/IntelliTex) — the tier-0
  specialist, trained on
  [MathBridge](https://huggingface.co/datasets/Kyudan/MathBridge).
- [Texo / FormulaNet](https://github.com/alephpi/Texo) — image OCR tier 0;
  preprocessing ported from [Texo-web](https://github.com/alephpi/Texo-web).
- [Texify](https://github.com/VikParuchuri/texify) via
  [Xenova/texify](https://huggingface.co/Xenova/texify) ONNX — image OCR tier 1.
- [WebLLM](https://github.com/mlc-ai/web-llm),
  [transformers.js](https://github.com/huggingface/transformers.js),
  [KaTeX](https://katex.org).

## Image → LaTeX (fully client-side)

Drop, paste, or pick an image of rendered math. The image is decoded and OCR'd
inside the tab and never leaves the browser (there is no upload path at all).
Two models form a ladder, chosen by an 18-image benchmark
(`bench/render-images.py`, `public/bench-images.html`, `bench/judge-images.py`,
results in `bench/results-images-2026-09.json`):

| model | size | easy | medium | hard | prose+math | tiny/dark | median latency |
|---|---|---|---|---|---|---|---|
| Texo (tier 0) | 77 MB | 100% | 75%* | 80% | 67% | 50%* | 0.7 s |
| Texify (tier 1) | 305 MB | 50%† | 100% | 100% | 100% | 0% | 3.5–4 s |

\* both misses are the KaTeX-only alias `\infin`, which the app now canonicalizes to `\infty` (→ 100%).
† Texify repeats one line to the token cap on very sparse images; the app now collapses exact repeats.

Texo runs first. The app escalates to Texify when Texo's output fails syntax
checks or when it spelled out prose (Texo has no text mode, so words inside
`\mathrm{}` are a reliable signal the image is a passage). A "read image with
the other model" button covers wrong-but-valid readings (e.g. matrices, Texo's
weak spot). Texify is only downloaded when actually needed.
