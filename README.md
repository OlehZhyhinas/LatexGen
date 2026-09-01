# LatexGen

Text → LaTeX converter that runs (almost) entirely on the user's machine.
Type math in plain English — *"the integral from 0 to infinity of e to the
minus x squared"* — and get validated, rendered LaTeX back in well under a
second, without the input leaving the browser for most conversions.

## Architecture: a benchmark-driven escalation ladder

Every conversion walks a ladder of increasingly capable (and expensive)
tiers, and only climbs when the previous tier's output fails validation:

1. **Specialist (in-browser, ~0.2–0.8s)** — [IntelliTeX](https://huggingface.co/duanxianpi/IntelliTex)
   (CodeT5+ 220M, MIT), int8-quantized ONNX via transformers.js. Handles
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
  app.js           routing ladder, model picker, refine chat, streaming
  validator.js     KaTeX syntax + input-fidelity checks
  models/          IntelliTeX int8 ONNX weights (git-lfs, ~264MB)
  bench.html/.js   in-browser benchmark harness
server.js          zero-dependency Node proxy (Ollama/MLX routing, NDJSON)
deploy/            AWS: deploy.sh + Lambda escalation proxy
bench/             eval dataset, LLM judge, 2026-09-01 results
scripts/           MLX runner, specialist ONNX conversion
```

## Licenses / credits

- [IntelliTeX](https://huggingface.co/duanxianpi/IntelliTex) (MIT) — the
  tier-0 specialist, trained on
  [MathBridge](https://huggingface.co/datasets/Kyudan/MathBridge) (MIT).
- [WebLLM](https://github.com/mlc-ai/web-llm) (Apache-2.0),
  [transformers.js](https://github.com/huggingface/transformers.js)
  (Apache-2.0), [KaTeX](https://katex.org) (MIT).
