# Deploying and operating

## Local (self-hosted, offline-capable)

Requires Docker. Optional: [Ollama](https://ollama.com) and/or an OpenAI-compatible local server for the server-model tier.

```bash
ollama pull qwen3:1.7b     # fast tier
ollama pull qwen3:8b       # strong tier: refinements, prose, strict-mode judge
docker compose up --build  # http://localhost:8013
scripts/run-mlx.sh         # optional: MLX fast tier on Apple Silicon
scripts/build-model-variants.sh  # optional: WebGPU int4 weights (not in git)
```

Share a local instance temporarily: `cloudflared tunnel --url http://localhost:8013`.

## GitHub Pages (static, no server)

The on-device app needs nothing but a file server, so it can be published as a
static site. What that costs you is everything backed by a server process: the
server-model tier, the [Tab API](api.md) and the [compute mesh](mesh.md). The
`STATIC_BUILD` flag in `public/config.js` switches those off and hides their UI
rather than leaving them visible and permanently unavailable.

Two things make it more than a file copy:

- **Weights cannot ship with the site.** They are 1.3 GB, past the 1 GB Pages
  limit, and four of them are over GitHub's 100 MB per-file limit, which is why
  they are in Git LFS — and Pages serves LFS *pointer files*, not the contents.
  So the static build excludes `public/models/` and streams the weights from a
  Hugging Face model repo instead. Publish them once:

  ```bash
  brew install hf && hf auth login   # or: pipx install "huggingface_hub[cli]"
  scripts/build-model-variants.sh    # the q4 variants the WebGPU path needs, plus .gz siblings
  scripts/upload-models-hf.sh        # MODEL_REPO=... to use another repo
  ```

  Neither Hugging Face nor `server.js` negotiates `Content-Encoding`, so the
  weights that gain from it (the int8 files, ~30%; the JSON) ship with a
  pre-compressed `.gz` sibling and a per-model `compressed.json` manifest,
  written by `scripts/compress-models.mjs` (the variants script runs it). The
  service worker fetches the `.gz` and inflates it in flight, so nothing else
  in the app knows. Without the manifest the raw files are used, so a deploy
  that skips the step still works, only larger.

  The token needs write scope. Homebrew's Python is externally managed, so
  `pip install` into it fails — use the formula or pipx.

- **A project page is served under a prefix** (`/LatexGen/`), so no asset path
  may be root-absolute. Every path in `public/` is relative to the page or,
  inside workers, resolved against `import.meta.url`. Keep it that way: a
  single `"/vendor/…"` breaks the deployed site while working locally.

Then build and deploy:

```bash
node scripts/build-static.mjs   # -> dist/, about 87 MB
```

`.github/workflows/pages.yml` does exactly that on every push to `main` and
publishes `dist/`. In the repository settings, set Pages' source to *GitHub
Actions*. The repository must be public, or on a paid plan.

What the static build cannot do is set response headers. The CSP is injected
as a `<meta>` tag instead, which costs `frame-ancestors` and HSTS — both
header-only. The self-hosted build keeps them.

## Server model: any provider

The server-model tier speaks to any OpenAI-compatible chat API, or to Ollama.

| Variable | Meaning |
|---|---|
| `OPENAI_BASE_URL` | `https://openrouter.ai/api/v1`, `https://api.openai.com/v1`, Groq, Together, a vLLM box, or local `mlx_lm.server` (default `http://host.docker.internal:8080/v1`) |
| `OPENAI_API_KEY` | bearer key, server-side only |
| `OPENAI_MODEL`, `OPENAI_REFINE_MODEL` | fast and strong model ids |
| `OLLAMA_URL`, `OLLAMA_MODEL`, `OLLAMA_REFINE_MODEL` | self-hosted Ollama |
| `OPENAI_DISABLED=1`, `OLLAMA_DISABLED=1` | turn a backend off |
| `SERVER_KIND` | `local` or `cloud`; auto-detected from the API base URL. Clients use a local server before peers and a cloud one after |

Only conversions the browser could not finish reach the server. With a small model on OpenRouter each one costs well under a cent.

## AWS: App Runner, one container

```bash
AWS_PROFILE=<personal> OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
OPENAI_MODEL=<fast model> OPENAI_REFINE_MODEL=<strong model> \
OPENAI_API_KEY_SECRET_ARN=arn:aws:secretsmanager:...:secret:latexgen/openai \
deploy/aws-apprunner.sh
```

Builds the image, pushes it to ECR, and creates or updates an App Runner service (0.25 vCPU, 0.5 GB, roughly $5 to $10 a month always-on) serving the static site, the server-model proxy and the Tab API and mesh relay. The API key is injected from Secrets Manager at runtime. Keep the service at one instance: the relay is in-memory state. CloudFront can be added in front later to cache model weights; `deploy/deploy.sh` provisions that static path with a stateless Lambda proxy for `/api/convert`, `/api/refine` and `/api/judge` only.

IAM needed by the deploying user: App Runner, ECR, and read on the one secret.

## Production hardening (built in)

- `NODE_ENV=production` disables benchmark endpoints and pages and stops echoing errors.
- Per-IP token buckets: 600/min general, 120/min inference, 1200/min relay. Set `TRUST_PROXY=1` behind a proxy so limits key on the real client IP.
- Payload caps (64 KB text, 6 MB images) with proper 413s, relay caps (2000 tabs, 5000 pending jobs), request and keep-alive timeouts, graceful `SIGTERM` that releases long-polls so tabs reconnect.
- Security headers on every response, including a CSP that allows WebAssembly and WebGPU workers, same-origin vendored assets, HuggingFace weight downloads and the Overleaf form post, and nothing else. `HSTS=1` when TLS terminates in front.
- Structured JSON access logs with ids redacted; client IPs only with `LOG_IP=1`.
- The container runs as the unprivileged `node` user with a healthcheck.

## Tests

`npm test` runs the validator unit tests and integration tests that spawn the server against a mock OpenAI-compatible upstream: streaming proxy, bearer auth, judge, hardening, rate limit, Tab API relay, mesh routing, reciprocity and owner-lane priority.
