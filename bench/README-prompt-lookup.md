# Avenue B: prompt-lookup speculative decoding on the WebLLM text tier

Issue #27, Avenue B. Built end to end and measured paired against a live
baseline on the 120-paragraph arXiv corpus, all four ladder rungs.

## Verdict

Prompt-lookup drafting with **k = 5** cuts time-to-answer on real pastes by
**14–26 %** (ms/char ratio treatment/baseline, median, IQR entirely below
parity) on the 0.8B, 4B and MiniCPM5-2B, and is a wash on the 9B, with the
baseline carrying the catalog's own tuning knobs:

| rung | baseline knobs | ms/char ratio (median, IQR) | tok/s ratio | byte-identical | tokens/pass | draft passes |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | burst5, batchPass, flush32 | **0.818** [0.740–0.907] | 1.22x | 119/120 | 2.96 | 58 % |
| Qwen3.5-4B | burst1, batchPass, flush32 | **0.863** [0.790–0.954] | 1.16x | 116/120 | 2.23 | 40 % |
| Qwen3.5-9B | burst1, batchPass, flush64 | 0.938 [0.870–1.0005] — **rejected**, IQR touches parity | 1.07x | 120/120 | 2.21 | 40 % |
| MiniCPM5-2B | burst1, flush32, lookahead1 | **0.741** [0.651–0.831] | 1.35x | 109/120 | 2.29 | 43 % |

Two things the issue assumed do not hold on this stack, and both are load
bearing:

1. **Output is not byte-identical.** Exact verification produces the target
   model's argmax only in exact arithmetic. Here the k-token `batch_verify`
   pass runs the prefill GEMM kernels, the baseline runs the decode GEMV
   kernels, and their logits differ in the low bits, so near-tie argmaxes
   flip: 1–4 of 120 items on the Qwen3.5 rungs at k = 5, 11 of 120 on
   MiniCPM5-2B, 7 of 120 on the 4B at k = 10. It is the same mechanism that
   retracted Avenue C's exactness claim. The quality question therefore does
   not disappear; it goes to the judge (below).
2. **The verify pass is far from free.** On the shipped libs a (d+1)-token
   `batch_verify` costs 2.3–2.8 decode steps at d = 5 and 4.0–4.2 at d = 10
   on the 4B (≈7 ms per extra verified token against a 20 ms decode step),
   with the same shape on the 0.8B and the 2B. That is why k = 10, the value
   the offline gate assumed, is a wash or a loss (4B, arXiv: 0.905
   [0.819–1.006], IQR touching parity → reject; 4B, pdf-pastes: 1.18
   [1.11–1.26], slower), and why k = 5 is the operating point.

The win is real on prose-heavy pastes and zero-to-negative on maths-heavy
ones: on the rendered `pdf-pastes` development corpus (1.5 tokens/pass) k = 5
is neutral on the 4B (1.02 [0.92–1.09]).

### Judge on the divergent items

Every item whose stripped output differed was judged on both sides with the
`qwen-local` judge, three repeats each, majority verdict
(`bench/judge-prompt-lookup-ab.py`, `bench/results-prompt-lookup-judge.json`).
Identical strings need no judging.

| run | judged | treatment worse | treatment better | both correct | both wrong | sign test |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B, k=5, shipped knobs | 1 | 0 | 0 | 1 | 0 | – |
| Qwen3.5-4B, k=5, shipped knobs | 4 | 0 | 0 | 1 | 3 | – |
| Qwen3.5-4B, k=5, plain baseline | 1 | 0 | 0 | 0 | 1 | – |
| Qwen3.5-4B, k=10, plain baseline | 7 | 0 | 1 | 1 | 5 | p = 1 |
| MiniCPM5-2B, k=5, both baselines | 11 | **2** | 0 | 1 | 8 | p = 0.5 |

On the Qwen3.5 rungs no divergent item got worse (the one discordant pair, at
k = 10, went the other way: the baseline had dropped a closing bracket). On
MiniCPM5-2B two items flip from correct to wrong, in the same direction both
times, on a corpus where the judge's own noise is about two items — not
distinguishable from noise at n = 2, but not nothing either, and it is the
rung with by far the most flips (11/120). All three verdicts were unanimous
on every judged side, so this is not judge jitter on the items themselves.

## What was built

- **Runtime** (`OlehZhyhinas/web-llm-qwen`, branch `prompt-lookup` off
  `qwen-m5`, bundle SHA-256 `6f529d9b…` for the final runs):
  `src/prompt_lookup.ts` (incremental n-gram index, last-match policy,
  nMax→nMin fallback) and a `decodePromptLookupStep` in `llm_chat.ts` behind
  `globalThis.__webllmPromptLookup = { k, nMax, nMin, hybrid? }`. The draft
  is verified with the model lib's `batch_verify` — every shipped lib
  (Qwen3 0.6B–8B, Qwen3.5 0.8B/4B/9B, MiniCPM5-2B) exports it and it returns
  logits at every position — and the compiled `argmax_logits` kernel, which
  turned out to be batch-generic. No model lib was recompiled.
- **Rollback.** Attention-only models pop rejected drafts with
  `kv_state_popn`. All three Qwen3.5 rungs are hybrid (GatedDeltaNet layers
  with an RNN state, the 0.8B included), and TVM's `RNNState::EndForward`
  zeroes the history whenever a forward appends more than one token, so a
  batched verify pass can never be popped at any `max_history_size`. The
  `hybrid: "fork"` strategy forks the live sequence before the pass, keeps
  the fork on full acceptance, and otherwise re-runs the accepted prefix on
  the fork and swaps sequence ids; the caches are created with two sequences.
  Fork→verify→remove on the paged KV cache is exact (the partial last page is
  copied eagerly and single ownership is asserted before any append).
- **A WebGPU runtime bug this uncovered.** TVM's RNN-state fork copies one
  sequence slot to another inside a single storage buffer;
  `copyBufferToBuffer` with identical source and destination is illegal in
  WebGPU and invalidates the whole command buffer, so every fork silently
  produced garbage (8/8 outputs diverged). `scripts/patch-web-runtime-same-buffer-copy.mjs`
  stages such copies through a scratch buffer.
- **Lookahead stop-path fix.** The in-flight speculative step at EOS is now
  awaited once instead of landing as TTFT on the next request.
- **Harness** (`OlehZhyhinas/webnn-workbench`, branch `prompt-lookup`):
  `run-corpus.mjs --ab` runs both sides of every item in the same Chrome
  session and engine, alternating order per item, and records ms/char,
  tok/s, TTFT, byte-identity, the runtime's acceptance statistics and a
  verify-cost table by verify length. `--prompt-lookup name=k[:nMax[:nMin[:fork]]]`
  is also a per-case knob on `run-webllm-interleaved.mjs`, which now reports
  ms/char next to tok/s.
- **Offline gate** (`bench/prompt-lookup-acceptance.py`, this repo): the
  prompt-lookup simulation replayed against the four models' real greedy
  outputs on the arXiv corpus, under the two rollback cost models, with
  per-pass logs kept so any measured cost curve can be applied afterwards.

## Method

Paired, interleaved, within one session: for each of the 120 arXiv items the
same engine generates once with drafting off and once with it on, order
alternating per item, `resetChat` between generations, greedy, `max_tokens`
512, the app's real prompt. The quantity reported is ms per output character
(time to an answer); tok/s is reported alongside and agrees in direction
everywhere. IQR crossing parity = reject. Both sides carry the catalog's
knobs for that rung, so the comparison is against what ships, not against
stock single-step decoding.

k was fixed at 5 on the `pdf-pastes` development corpus (rendered, not the
acceptance corpus): the verify-cost curve there showed C(6) ≈ 2.4 and
C(11) ≈ 4.2 on the 4B, and k = 10 measured slower. Both k = 5 and k = 10 were
then run on arXiv and both are reported. n-gram range 3→2 throughout, the
gate's a-priori setting; the offline grid shows tokens/pass moves by ≤ 0.3
across n in {2,3,4}.

## The cost curve, and what would raise the ceiling

Mean wall time of a verify pass of L = d+1 tokens against the baseline's
decode step (arXiv runs, shipped knobs):

| rung | decode ms/token | C(6) | C(11) |
|---|---|---|---|
| Qwen3.5-0.8B | 6.1 | 2.75 | 3.1 (dev run) |
| Qwen3.5-4B | 22.2 (18.2 without batchPass) | 2.63 (2.29) | 4.02 |
| Qwen3.5-9B | 24.4 | 3.02 | – |
| MiniCPM5-2B | 12.2 (15.1) | 2.54 (2.29) | 3.08 |

The intercept is small (≈3 ms on the 4B); the slope is ≈7 ms per verified
token, about 37 % of a whole decode step per token. A decode step is
weight-bandwidth bound and an 11-token GEMM reads the same weights once, so
in principle C(11) should be ≈1.2; the gap is the generic small-M prefill
schedule in the shipped libs. Retuning those kernels for M ≤ 16 is the lever
that would turn 1.16x into the 1.5–2x the offline arithmetic predicts — and
it is a per-rung model-lib recompile, the #21–#26 compile-cost territory,
not a runtime change. The row-argmax over L rows is included in these
timings and is not separated out (bounded above by ≈1 ms/row).

Offline, the acceptance side was better than the gate predicted:
tokens/pass at k = 10 on the real outputs is 2.3–3.2 (gate: 1.5) because the
arXiv paragraphs copy far more prose than the rendered corpora. Break-even
verify cost at k = 10: 5.4–6.2 decode steps for plain rollback, 3.3–4.6 for
fork + re-run. Measured C(11) sits right at the hybrid break-even, which is
the arithmetic behind the k = 10 result.

## Reproduce

```bash
# offline acceptance on real outputs (this repo)
/Users/oleh/personal/latexgen-text-gates/.venv/bin/python bench/prompt-lookup-acceptance.py

# paired A/B, 4B, shipped knobs (webnn-workbench, branch prompt-lookup; runtime bundle from web-llm-qwen prompt-lookup)
WEBNN_MAX_LOAD=0 node bench/webnn/qwen/run-corpus.mjs \
  --corpus bench/latexgen-snapshot/bench/arxiv-pastes.json \
  --model-slug qwen35-4b --side shipped --model Qwen3.5-4B-q4f16_1-MLC \
  --model-lib /Users/oleh/personal/mlc-llm-webllm084-qwen8/out-qwen35-4b/qwen35-4b-argmax-chunk256-sg32-tr32.wasm \
  --runtime-lib /Users/oleh/personal/web-llm-prompt-lookup/lib/index.js \
  --max-tokens 512 --port 8930 --label qwen35-4b-arxiv-k5-shipped \
  --prompt-lookup 5:3:2:fork --ab --batch-pass --flush-every 32

# judge the divergent pairs (qwen-local)
/Users/oleh/personal/latexgen-text-gates/.venv/bin/python bench/judge-prompt-lookup-ab.py --files <ab result files>
```

Result files: `webnn-workbench` `bench/results/prompt-lookup-ab-*.json`;
this repo `bench/results-prompt-lookup-acceptance.json`,
`bench/results-prompt-lookup-judge.json`.

## Not done, and caveats

- **Qwen3.5-9B does not adopt drafting.** Same acceptance as the 4B (2.21
  tokens/pass, 40 % of passes drafted) but a 6-token verify costs 3.02 decode
  steps there (24.4 ms/token baseline), and the paired ratio's upper quartile
  sits at parity (0.938 [0.870–1.0005]); the rule says reject, so its catalog
  default is unchanged. Output was byte-identical on all 120 items.

- No product wiring: the catalog does not enable the knob. Shipping needs a
  `promptLookup` entry per variant and, for the Qwen3.5 rungs, the fork mode
  set before the engine loads (it changes cache allocation).
- The 0.8B baseline used `--max-history-size 8` so its burst-5 rollback has
  history slots; the catalog variant relies on the same thing implicitly.
- One bundle (`b45b7e5f…`) ran the first arXiv pass and the development runs,
  a second (`6f529d9b…`, formatting-only source change) the shipped-knob
  runs; each result file records its own hash.
- The 4B baseline is slower with batchPass+flush32 than without in these
  runs (22.2 vs 18.2 ms/token); the pairing is within-run so the ratios
  stand, but the catalog's claim for that knob on this path deserves its own
  look.
- Judge noise on this corpus is a couple of items; deltas below ~5 points
  are not distinguishable, as established earlier in the issue.
