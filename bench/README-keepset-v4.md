# Keep-set v4 — merge rank at the optimum, plus a minimal leaf force-add set

Scripts: `bench/build-keepset-v4.py` (build + slot accounting + fertility),
`bench/validate-keepset-v4.py` (validation suite), `bench/compute-keepset-v4-speed.py`
(net speed / MB saved). Held-out text for every fertility number below (unless
stated otherwise): 120 `input`+`reference` strings from `bench/arxiv-pastes.json`
+ 93 `ok` items' `input`+`reference` from `bench/bench-data-extended.json` = 426
non-empty strings, real re-encoding with the rebuilt tokenizer — the same
methodology and the same held-out set `keepset-v3` (`origin/keepset-v3`, landed)
used.

## TL;DR

- **v4 = strategy A at K in {24,576, 32,768, 49,152}, plus a ~1,600-slot
  force-add layer** (single-codepoint accented-Latin/Greek/maths-operator/
  superscript-subscript-digit tokens, LaTeX macro completion, LatexGen's own
  rendered prompt) plus verified merge-ancestor closure over the union.
- **v4 beats strategy A's own fertility at every K, on both tokenizers, both
  in-sample and on a leakage-free disjoint-corpus spot check** — the
  force-add tokens are generically useful enough on held-out text that they
  more than pay for the merge-rank fill slots they displace. Net speed
  improves accordingly: e.g. Qwen3.5-4B at K=32,768 goes from A's 1.077x to
  v4's 1.083x (in-sample), or 1.062x → 1.073x on the disjoint-corpus check.
- **The issue #27 LaTeX-tail claim is confirmed, not retired-as-wrong**: all
  9 named pieces (`\(`, `\[`, `mathrm`, `partial`, `rangle`, `Gamma`, `Psi`,
  `{n`, `}=`) are already present in strategy A at K=32,768 — see "Issue #27"
  below for why an earlier pass of this task's own build script found the
  opposite, and the correction that fixed it.
- **All 6 emitted directories pass every validation check** (v2's five,
  reused via `bench/validate-keepset-v2.py`, plus two new ones): no
  regressions, all candidates.
- **Recommended K = 32,768 for both tokenizers** (matching the brief's own
  design point and `keepset-v3`'s "compromise K" reasoning), 24,576/49,152
  also emitted.
- **Residual risk, stated plainly**: the `Nikolskii`-class corruption is only
  *partly* fixed. See "Residual risk" below.

## What "strategy A" means — corrected mid-task, reconciled against `keepset-v3`

This task's brief describes strategy A, in its "Where this sits" section, as
v1's "plain BPE merge rank" method, and later instructs: "Build strategy A at
K=32,768 yourself with those builders — it is plain merge rank and
deterministic." Read literally and in isolation, an earlier pass of this
task's own `bench/build-keepset-v4.py` (still visible in this branch's git
history) took that to mean a **corpus-free** baseline: byte alphabet +
added/special tokens, padded to K by ascending original token id, with no
corpus, seed, or harness-prompt file read at all.

That version's fertility numbers did not reconcile with `origin/keepset-v3`
(which had landed by the time this task ran — see the brief's own
contingency: "If `origin/keepset-v3` has landed by the time you start, read
its results and reconcile"). `bench/README-keepset-v3.md`'s "Strategy
definitions (exact, not paraphrased)" section defines strategy A precisely
and unambiguously as:

> **A**: `bench/vocab-coverage.py`'s `build_keep_set_for_tokenizer` (its
> default 5 corpus files... unioned with byte alphabet, added/special
> tokens, harness-prompt tokens, KaTeX macro seed, maths-English word seed)
> + `final_keep_set_for_K`. **v1's method, called directly, not
> reimplemented.**

i.e. strategy A *does* read the corpus and seed files — "plain merge rank"
describes its **fill mechanism** (ascending original-token-id padding, the
same proxy every generation of this project has used) as opposed to v2's
targeted-layer-then-restricted-pool fill, not a claim that v1 has zero
corpus dependency. Re-deriving strategy A this way reproduces v3's own
numbers almost exactly: this task's independent K=32,768 spot check
(disjoint-corpus, see "Leakage" below) gives qwen35 A-disjoint inflation
+1.97%, matching v3's own reported A-disjoint number for qwen35 K=32,768
(+1.97%) exactly, and MB-saved figures agree bit-for-bit with v3's `Final
recommendation` table (118.4 / 296.0 / 947.2 / 214.9 MB). **This build script
was corrected to use v3's exact definition** so the two tasks' numbers are
directly comparable, as the brief asks. The corpus-free version is not used
for anything in this README; it is left in git history as a documented
false start, per the brief's own "correct me where I am wrong and say so."

One consequence: **pure strategy A reproduces v1's own originally-named
defect** — the 7 LatexGen-prompt token ids are absent from it at every K
(confirmed empirically below), because v1's training corpus never included
LatexGen's actual client-prompt text (only the separate, narrower
webnn-workbench harness-prompt set). `keepset-v3` found and fixed the same
defect with its own "A-fixed" variant. v4's force-add group 3 (below) closes
the identical gap, plus more (single-codepoint accented/Greek/maths
characters and LaTeX macro completion that A-fixed does not add).

## The force-add set — measured

Force-add groups are computed once per tokenizer, independent of K, then
their cost is measured against strategy A's own final (padded) set at each
K in the sweep.

### Group 1 — single-codepoint non-ASCII tokens

Classified by Unicode General Category + block membership
(`bench/keepset-v2-unicode.py`'s Greek/maths-operator/arrow classifier,
reused verbatim, plus two documented additions: "LATIN"-named Lu/Ll
codepoints outside ASCII for accented Latin letters, and category "No"
codepoints whose Unicode name contains SUPERSCRIPT/SUBSCRIPT for the
digit-only sub/superscript class). Restricted to true single-codepoint
tokens (bare + space-prefixed variants that already exist in the vocab), not
multi-character tokens containing them.

| Tokenizer | accented Latin | Greek | maths operators/relations | superscript/subscript digits | typographic | **total group1 ids** |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5 | 316 | 93 | 64 | 6 | 2 | **481** |
| MiniCPM5-2B | 114 | 59 | 61 | 6 | 2 | **242** |

The brief's own prediction (267/73/34/2/1, qwen35, "missing at K=32,768")
is close in shape but measures a different quantity than the table above
(total group ids in the vocab, not "missing from A at K=32,768" — see the
per-K missing table below for the directly comparable number, which came out
lower than predicted at 236/49/40/1/1 on qwen35 at K=32,768 once strategy A
was corrected to its real, corpus-based definition, since the corpus/seed
layers already happen to cover some of these single codepoints).

### Group 2 — LaTeX macro completion

Tokenizing all 1,241 KaTeX macro/environment/delimiter strings
(`bench/keepset-seed-latex.json`, key `all`; bare + leading-space variants,
`bench/vocab-coverage.py`'s `seed_ids_from_strings`, unmodified):

| Tokenizer | ids (bare only) | ids (bare + leading-space) |
|---|---:|---:|
| Qwen3.5 | 883 | **950** |
| MiniCPM5-2B | 1,099 | **1,250** |

The brief's "94 missing of 1,423" does not reconcile with either the total
id count above or the "missing from strategy A at K=32,768" count measured
below (21 on qwen35, 39 on MiniCPM5-2B) — reported honestly rather than
force-fit. Most of the 1,241 macro strings' tokens are **already** part of
strategy A's own `latex_seed` layer (`build_keep_set_for_tokenizer`'s
`always_keep_ids` unconditionally includes it, corpus-independent), which is
exactly why so little is missing once strategy A is built correctly, and is
consistent with the issue #27 tail turning out to already be present too.

### Group 3 — LatexGen's own prompt

The 7 named Qwen3.5 ids from the brief (80757, 80636, 77019, 65088, 80313,
72452, 94498), plus every token of the rendered chat template + conversion
system prompt + PDF hint + both few-shot pairs, for 2 representative user
texts, both `enable_thinking` shapes, both plain/PDF prompt treatments
(`bench/build-keepset-v2.py`'s `latexgen_prompt_messages`, unmodified — not
its `harness_prompt_ids`, which is the separate webnn-workbench benchmark
prompt set, out of scope for group 3 per the brief).

| Tokenizer | renders | chars | named ids | group3 total |
|---|---:|---:|---:|---:|
| Qwen3.5 | 8 | 10,248 | 7 (all present in the render itself) | **241** |
| MiniCPM5-2B | 8 | 10,248 | 0 (ids are Qwen3.5-specific) | **249** |

### Overlaps and force-add union (pre-closure)

| Tokenizer | g1∩g2 | g1∩g3 | g2∩g3 | all 3 | union (pre-closure) |
|---|---:|---:|---:|---:|---:|
| Qwen3.5 | 1 | 9 | 64 | 0 | **1,598** |
| MiniCPM5-2B | 6 | 11 | 62 | 1 | **1,663** |

### Merge-ancestor closure — verified, not assumed

The brief claims the force-adds "need no closure at all — a single
codepoint's only ancestors are byte tokens, which are always kept, and the
LaTeX macro tokens decompose into short ASCII already inside the top
32,768." **This does not hold exactly**: `compute_merge_ancestor_closure`
over the force-add ∪ strategy-A-natural union finds extra ancestor ids on
both tokenizers.

| Tokenizer | closure extra (beyond force-add ∪ strategy-A-natural) | v4 natural total (post-closure) | new vs. strategy-A-natural |
|---|---:|---:|---:|
| Qwen3.5 | 1,512 | 8,402 | +1,988 |
| MiniCPM5-2B | 1,841 | 9,681 | +2,098 |

However, **almost all of this closure cost turns out to already be free at
the K's actually shipped**: strategy A's own ascending-id padding already
includes the vast majority of these ancestor ids once padded out to
24,576+ (the ancestors tend to be short, common pieces that rank early in
merge order regardless). The build script folds in strategy-A's own closure
gap defensively, then v4's, in a fixed-point pass; at every emitted K on
both tokenizers, **the post-fold closure extra is 0** (see the per-K table
below — `v4_closure_extra_pre_fold`/`post_fold` are both 0 everywhere in
the sweep). So the brief's claim is wrong at the level of the raw force-add
∪ natural-base union (1,512–1,841 extra ids, not 0), but ends up correct in
its practical consequence at the K's this task actually ships (0 extra cost
at K≥24,576) — both facts are reported here rather than picking one.

### Slot accounting per K (cost over strategy A, whole-word tokens displaced)

**Qwen3.5**:

| K | new force-add cost vs. A | — accented Latin | — Greek | — maths ops | — supersub | — typographic | group1 missing | group2 missing | group3 missing | slots displaced from A's fill | of which whole-word |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24,576 | 381 | 253 | 51 | 40 | 1 | 1 | 346 | 22 | 15 | 531 | 243 |
| 32,768 | 359 | 236 | 49 | 40 | 1 | 1 | 327 | 21 | 13 | 450 | 199 |
| 49,152 | 314 | 207 | 48 | 38 | 0 | 0 | 293 | 14 | 9 | 353 | 153 |

**MiniCPM5-2B**:

| K | new force-add cost vs. A | — accented Latin | — Greek | — maths ops | — supersub | — typographic | group1 missing | group2 missing | group3 missing | slots displaced from A's fill | of which whole-word |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24,576 | 190 | 87 | 13 | 23 | 2 | 1 | 126 | 46 | 20 | 458 | 180 |
| 32,768 | 173 | 79 | 12 | 23 | 2 | 1 | 117 | 39 | 19 | 327 | 110 |
| 49,152 | 128 | 59 | 7 | 20 | 1 | 0 | 87 | 34 | 7 | 180 | 72 |

("new force-add cost" and "group N missing" both count against strategy A's
final padded set at that K, so they overlap by construction — group1/2/3's
missing counts are not additive with each other where groups intersect;
"displaced" is the measured set difference between strategy A's own
ascending-rank fill and v4's, at the *same* final K — larger than the raw
new-force-add count because most force-add ids have low enough original
token ids that strategy A's own merge-rank fill would have picked them up
"for free" anyway (only the 314–381 counted above have original ids high
enough to need a genuine, non-incidental force-add), while the fill
mechanism as a whole still has to make room for v4's larger natural base by
walking further down the ascending-id candidate list. Whole-word = a
leading-space-prefixed, ASCII, alphabetic piece of length ≥2; a reasonable
but not exhaustive proxy for "a common-word merge-rank fragment strategy A
would otherwise have kept.")

The force-add cost shrinks with K on both tokenizers (381→314 on qwen35,
190→128 on MiniCPM5-2B) because more of the group1/2/3 ids happen to already
be inside strategy A's own top-K merge-rank window as K grows — consistent
with the brief's framing that this is a cheap, K-largely-independent patch,
not a re-tuned parameter.

## Issue #27's LaTeX tail — confirmed present, not a retired claim

The brief states the tail issue #27 complained about (`\(`, `\[`, `mathrm`,
`partial`, `rangle`, `Gamma`, `Psi`, `{n`, `}=`) is "already in" strategy A
at K=32,768, and asks this to be verified and recorded because it retires a
long-standing claim in that issue. **Verified true, on the corrected
strategy A**: all 9 pieces tokenize to ids that are present in strategy A's
own K=32,768 keep-set on Qwen3.5 (9/9, see
`bench/results-keepset-v4-qwen35.json`'s
`issue_27_tail_claim_verified_against_strategy_A_32768`). An earlier pass of
this build (the corpus-free misreading of strategy A described above) found
the *opposite* — most of these pieces missing — which would have wrongly
un-retired the issue #27 claim; that was a bug in this task's own first
attempt, not a defect in the claim itself, and is corrected here.

## Fertility of v4 vs. strategy A

### In-sample (426-text held-out set — same set used to build strategy A's own corpus layer; see "Leakage" below)

**Qwen3.5** (unpruned fertility = 0.2599 tok/char):

| K | A fertility | v4 fertility | inflation (A vs. unpruned) | inflation (v4 vs. unpruned) | v4 vs. A |
|---:|---:|---:|---:|---:|---:|
| 24,576 | 0.2631 | 0.2599 | +1.22% | +0.00% | **−1.21%** |
| 32,768 | 0.2612 | 0.2599 | +0.50% | +0.00% | **−0.49%** |
| 49,152 | 0.2605 | 0.2599 | +0.22% | +0.00% | **−0.22%** |

**MiniCPM5-2B** (unpruned fertility = 0.2423 tok/char):

| K | A fertility | v4 fertility | inflation (A vs. unpruned) | inflation (v4 vs. unpruned) | v4 vs. A |
|---:|---:|---:|---:|---:|---:|
| 24,576 | 0.2460 | 0.2423 | +1.52% | +0.00% | **−1.49%** |
| 32,768 | 0.2443 | 0.2423 | +0.81% | +0.00% | **−0.80%** |
| 49,152 | 0.2428 | 0.2423 | +0.22% | +0.00% | **−0.21%** |

**v4 does not merely avoid strategy A's fertility cost — it comes out
*better* than A everywhere in-sample**, exactly matching the unpruned
tokenizer's own fertility at every K on both tokenizers. The brief predicted
"v4 costs well under 1% over A" and warned "treat the prediction as suspect
and let the number speak" — the real number is better than predicted (v4
costs strictly *negative*, i.e. an improvement, not just "under 1%"), which
is itself a signal to sanity-check for leakage (see below) rather than take
at face value.

### Leakage — inherited from `keepset-v3`, applies equally here

`keepset-v3`'s README found that strategy A's default corpus
(`bench/vocab-coverage.py`'s `CORPUS_FILES`) includes `arxiv-pastes.json`
and `bench-data-extended.json` as 2 of its 5 training files — **exactly**
the files this task's held-out set is drawn from. Since v4 = strategy A's
base ∪ force-adds, v4 inherits the identical leakage. The near-0%/negative
in-sample inflation above is real but optimistic, for the same reason v3
flagged for its own "A-fixed": it is not evidence the true, held-out cost is
zero.

**Disjoint-corpus spot check at K=32,768** (drop `arxiv` + `bench-extended`
from strategy A's training corpus only — `bench`, `pdf`, `synth-latex`
remain — then re-measure fertility against the *same*, now genuinely
held-out, 426-text set; group1/2/3 are corpus-independent by construction,
so this isolates leakage to strategy A's own base):

| Tokenizer | A-disjoint inflation | v4-disjoint inflation | v4 vs. A (disjoint) |
|---|---:|---:|---:|
| Qwen3.5 | +1.97% | +0.89% | **−1.06%** |
| MiniCPM5-2B | +2.43% | +1.26% | **−1.14%** |

The qwen35 A-disjoint number (+1.97%) matches `keepset-v3`'s own reported
A-disjoint figure for qwen35 K=32,768 (+1.97%) exactly, confirming this
task's re-derivation is consistent with v3's. **The qualitative result
survives leakage**: v4 still beats A by about 1 point of fertility even on
genuinely unseen text, not just in-sample — the force-add tokens are useful
on their own merits, not an artifact of training-corpus overlap. A full
disjoint-corpus sweep across all 3 K's (mirroring v3's complete "Corrected
optimum" table) was not run for this task, for time; the K=32,768 spot
check is reported as a check, not a full re-derivation of the recommended K
(see "Recommended K" below for how this affects that decision).

## Net speed

`ms_per_token(K) = body + head_orig * (K / V)`, `net(K) = ms_total /
(ms_per_token(K) * (1 + fertility_inflation))`. Head-kernel costs verified
against `webnn-workbench` `origin/main` via `git show` (not assumed from the
brief's own figures):

| Model | V | ms_total | head_ms (or range) | hidden | tied | bytes/row |
|---|---:|---:|---:|---:|---|---:|
| Qwen3.5-0.8B | 248,320 | 5.24 | 0.648–0.658 (`docs/qwen35-08-vocab-report.md` Stage 1, `docs/results.md:1690` 190.83 tok/s → 5.24 ms/token) | 1,024 | yes | 576 |
| Qwen3.5-4B | 248,320 | 15.61 | 1.372 (`notes/stage1-profile.md`, cross-checked `docs/qwen35-4b-vocab-report.md:145`) | 2,560 | yes | 1,440 |
| Qwen3.5-9B | 248,320 | 26.70 | range 4.0–4.6% of decode (`docs/qwen35-9b-vocab-report.md` Stage 1; `docs/results.md:2232` 37.45 tok/s → 26.70 ms/token) → 1.068–1.228 | 4,096 | **no** | 4,608 |
| MiniCPM5-2B | 130,560 | 9.74 | 0.6657 (`docs/minicpm5-2b-vocab-report.md` Stage 1) | 2,048 | no | 2,304 |

`bytes_per_row = hidden * 0.5625` at q4f16_1 (4-bit weights + fp16 scales,
group size 32); `MB_saved(K) = (V - K) * bytes_per_row * (2 if untied else
1) / 2^20`. Both formulas and all four rows are identical to
`keepset-v3`'s own verification (`bench/compute-optimum-v3.py`'s `MODELS`
dict) — reused, not re-derived, since v3 already did this verification work
and the brief only asks to re-verify, which independently checking the same
doc lines against `origin/main` confirmed.

### In-sample net speed (v4 vs. strategy A vs. v1@65,536)

| Model | Variant | K | inflation | net(K) | MB saved |
|---|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | v1 | 65,536 | +0.07% | 1.099–1.102x | 100.4 |
| Qwen3.5-0.8B | A | 24,576 | +1.22% | 1.112–1.114x | 122.9 |
| Qwen3.5-0.8B | **v4** | 24,576 | +0.00% | **1.125–1.128x** | 122.9 |
| Qwen3.5-0.8B | A | 32,768 | +0.50% | 1.114–1.117x | 118.4 |
| Qwen3.5-0.8B | **v4** | 32,768 | +0.00% | **1.120–1.123x** | 118.4 |
| Qwen3.5-0.8B | A | 49,152 | +0.22% | 1.107–1.110x | 109.4 |
| Qwen3.5-0.8B | **v4** | 49,152 | +0.00% | **1.110–1.112x** | 109.4 |
| Qwen3.5-4B | v1 | 65,536 | +0.07% | 1.068x | 251.0 |
| Qwen3.5-4B | A | 24,576 | +1.22% | 1.073x | 307.3 |
| Qwen3.5-4B | **v4** | 24,576 | +0.00% | **1.086x** | 307.3 |
| Qwen3.5-4B | A | 32,768 | +0.50% | 1.077x | 296.0 |
| Qwen3.5-4B | **v4** | 32,768 | +0.00% | **1.083x** | 296.0 |
| Qwen3.5-4B | A | 49,152 | +0.22% | 1.074x | 273.5 |
| Qwen3.5-4B | **v4** | 49,152 | +0.00% | **1.076x** | 273.5 |
| Qwen3.5-9B | v1 | 65,536 | +0.07% | 1.028–1.036x | 803.2 |
| Qwen3.5-9B | A | 24,576 | +1.22% | 1.024–1.031x | 983.2 |
| Qwen3.5-9B | **v4** | 24,576 | +0.00% | **1.037–1.044x** | 983.2 |
| Qwen3.5-9B | A | 32,768 | +0.50% | 1.030–1.037x | 947.2 |
| Qwen3.5-9B | **v4** | 32,768 | +0.00% | **1.035–1.042x** | 947.2 |
| Qwen3.5-9B | A | 49,152 | +0.22% | 1.030–1.037x | 875.2 |
| Qwen3.5-9B | **v4** | 49,152 | +0.00% | **1.032–1.040x** | 875.2 |
| MiniCPM5-2B | v1 | 65,536 | +0.09% | 1.034x | 142.9 |
| MiniCPM5-2B | A | 24,576 | +1.52% | 1.043x | 232.9 |
| MiniCPM5-2B | **v4** | 24,576 | +0.00% | **1.059x** | 232.9 |
| MiniCPM5-2B | A | 32,768 | +0.81% | 1.045x | 214.9 |
| MiniCPM5-2B | **v4** | 32,768 | +0.00% | **1.054x** | 214.9 |
| MiniCPM5-2B | A | 49,152 | +0.22% | 1.042x | 178.9 |
| MiniCPM5-2B | **v4** | 49,152 | +0.00% | **1.045x** | 178.9 |

**v4 beats both strategy A and v1@65,536 at every K, on every model**,
while saving strictly more memory than v1@65,536 at every K in the sweep
(109–123 MB more on the 0.8B/4B/9B Qwen3.5 rungs, 36–90 MB more on
MiniCPM5-2B). Two caveats on this table:

1. As with the fertility table above, these in-sample inflation numbers are
   optimistic (leakage). Because inflation floors near 0% at every K here,
   `net(K)` is close to monotonic-decreasing in K on all four models in this
   grid — the smallest K (24,576) looks best everywhere, which is the exact
   in-sample artifact `keepset-v3`'s README warns about ("naively computing
   `net(K)` from the leaky numbers picks the smallest available K... every
   time, which would not generalize"). This table is reported for direct
   comparability with the brief's own in-sample framing, not used alone to
   pick K — see "Recommended K" below.
2. Qwen3.5-9B's head cost is carried as a range per the brief's instruction,
   not collapsed to a point.

### Disjoint-corpus (leakage-free) net speed at K=32,768

| Model | Variant | inflation | net(K) | MB saved |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B | A-disjoint | +1.97% | 1.098–1.101x | 118.4 |
| Qwen3.5-0.8B | **v4-disjoint** | +0.89% | **1.110–1.113x** | 118.4 |
| Qwen3.5-4B | A-disjoint | +1.97% | 1.062x | 296.0 |
| Qwen3.5-4B | **v4-disjoint** | +0.89% | **1.073x** | 296.0 |
| Qwen3.5-9B | A-disjoint | +1.97% | 1.015–1.022x | 947.2 |
| Qwen3.5-9B | **v4-disjoint** | +0.89% | **1.026–1.033x** | 947.2 |
| MiniCPM5-2B | A-disjoint | +2.43% | 1.029x | 214.9 |
| MiniCPM5-2B | **v4-disjoint** | +1.26% | **1.041x** | 214.9 |

Even leakage-free, v4 clears net > 1.0x at K=32,768 on every model
(including Qwen3.5-9B, the tightest case at 1.026–1.033x) and beats strategy
A by roughly a full point of net speed everywhere — the headline "v4 > A"
result is not a leakage artifact, only its exact in-sample magnitude is.

## Recommended K

**K = 32,768 for both tokenizers.** This matches the brief's own design
point ("v4 is therefore: strategy A at K=32,768...") and `keepset-v3`'s own
"compromise K" reasoning (its disjoint-corpus sweep found 0.8B/4B want
K≈24,576, 9B wants a much larger K≈65,536 outside this task's grid, and
32,768 costs the smaller models only 0.02–0.3pp of net relative to their own
true optimum while recovering most of 9B's shortfall). This task's own
K=32,768 disjoint spot check confirms v4 clears net>1.0x there on every
model, including the tightest case (9B). The in-sample table's apparent
preference for K=24,576 on all four models is the same leakage artifact v3
already documented (see caveat 1 above) and is not used to override this
recommendation. All three K's (24,576/32,768/49,152) are still emitted per
the brief, so a future task can re-decide with a fuller leakage-free sweep
if needed.

## Validation results

All checks reused from `bench/validate-keepset-v2.py` (checks 1–5) plus two
new ones added for this task (check 6: every accented-Latin and Greek
single codepoint round-trips; check 7: a set of accented author names
round-trips), run via `bench/validate-keepset-v4.py`.

| Directory | Check 1 (φ/←/↔/↓/× + rare words + LatexGen prompts + 7 named ids) | Check 2 (120 refs + 2,000 dict words) | Check 3 (fertility, v2's own 120-item methodology) | Check 4 (prompt coverage) | Check 6 (932 accented/Greek codepoints) | Check 7 (9 accented author names) | Candidate? |
|---|---|---|---|---|---|---|---|
| qwen35-24576 | all pass, 7/7 named ids | 120/120 + 2000/2000 | +0.00% | 120/120 + 120/120 | 932/932 | all pass | **YES** |
| qwen35-32768 | all pass, 7/7 named ids | 120/120 + 2000/2000 | +0.00% | 120/120 + 120/120 | 932/932 | all pass | **YES** |
| qwen35-49152 | all pass, 7/7 named ids | 120/120 + 2000/2000 | +0.00% | 120/120 + 120/120 | 932/932 | all pass | **YES** |
| minicpm5-2b-24576 | all pass | 120/120 + 2000/2000 | +0.00% | 120/120 + 120/120 | 932/932 | all pass | **YES** |
| minicpm5-2b-32768 | all pass | 120/120 + 2000/2000 | +0.00% | 120/120 + 120/120 | 932/932 | all pass | **YES** |
| minicpm5-2b-49152 | all pass | 120/120 + 2000/2000 | +0.00% | 120/120 + 120/120 | 932/932 | all pass | **YES** |

**All 6 directories are candidates. No regressions found** — full detail in
`bench/results-keepset-v4-validation.json`. (Check 3's "+0.00%" here is v2's
own smaller, in-sample 120-item methodology, consistent with the leakage
finding above, not a separate result; check 1/2/4/6/7 are the checks that
actually gate candidacy.) `len(AutoTokenizer.from_pretrained(dir)) == K`
holds for all 6 directories (see each directory's `coverage.json`
`length_check`).

## Residual risk — stated plainly, not implied solved

Per the brief: the `Nikolskii`-class corruption (e.g. the observed
`Nikolskii` → `Kolskii` failure) is only **partly** addressed by this
change. Force-adding the accented-Latin and Greek single codepoints fixes
the case where a *literal* accented character gets dropped and corrupts
surrounding text. It does **not** fix the separate, still-present failure
mode where an ordinary ASCII subword piece that happens to be rare (like a
fragment of an uncommon surname) is dropped by the K-truncation itself —
that can still corrupt a rare name, and the layer that would close that gap
outright (a broad short-piece/proper-noun coverage layer, structurally
similar to what v2 attempted) is exactly the layer measured, across v2 and
v3, to cost most or all of the speed win this task is trying to preserve.
This trade-off is not resolved here; it is left as a known, named limitation
of v4, same as it was of v1.

## Emitted directories

`bench/keepsets-v4/<label>-<K>/` for both tokenizers at K in
{24,576, 32,768, 49,152}, same file layout as v1/v2/v3 (`keep-idx.json`,
`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`,
`config-patch.json`, `coverage.json`, `provenance.md`, `SHA256SUMS.txt`).
Each directory's `provenance.md` names this recommendation; exact
`keep-idx.json` SHA-256 for all 6 is in
`/Users/oleh/personal/latexgen-task-keepset-v4-RESULT.md`.

## What was not done

- A full disjoint-corpus (leakage-free) fertility/net-speed sweep across all
  3 K's, mirroring `keepset-v3`'s complete "Corrected optimum" table — only
  a K=32,768 spot check was run, for time. The qualitative conclusion (v4 >
  A, both net>1.0x) is confirmed at that K; the exact optimum-K choice
  relies on `keepset-v3`'s own already-published disjoint sweep rather than
  re-deriving it independently for v4's slightly different (force-add-
  augmented) keep-sets at every K.
- Selection D (marginal-token-savings ranking): out of scope for this brief
  entirely (not requested); `keepset-v3`'s Job 2 already found it blocked on
  the `keepset-corpus` sibling's un-pushed frequency tables.
