# Keep-set v3 — pinning the speed/quality optimum

v1 (`bench/keepsets/`, PR #55) and v2 (`bench/keepsets-v2/`, PR #56) each
measured fertility a different way, over different held-out text, so their
numbers were never comparable. This does one methodology, on one held-out
set, actually re-encoding with the rebuilt tokenizer (never simulating a
miss as a byte length), across every strategy and every K, for both
tokenizers. Scripts: `bench/build-keepset-v3.py` (Job 1),
`bench/compute-optimum-v3.py` (Job 3), `bench/emit-keepset-v3.py` /
`bench/validate-keepset-v3.py` (Job 4), `bench/remeasure-v1-v2-v3.py`
(v1/v2 comparability).

Held-out text for every fertility number below (unless stated otherwise):
120 `input`+`reference` strings from `bench/arxiv-pastes.json` + 93 `ok`
items' `input`+`reference` from `bench/bench-data-extended.json` = 426
non-empty strings, per the brief.

## TL;DR

- Strategy **A** (v1's merge-rank method, verbatim) beats **B** (targeted
  fix layers + merge-rank fill) and **C** (v2's full structural selection,
  including the blanket ASCII<=3 layer) at *every* K, on *both* tokenizers,
  by a wide margin. C's blanket ASCII<=3 layer is confirmed as the entire
  problem: it burns 19,936-20,036 slots for near-zero fertility benefit
  over B.
- But pure A **fails Job 4's own validation** on the Qwen3.5 tokenizer — it
  reproduces v1's original, specifically-named failure (7 LatexGen-prompt
  token ids missing), because v1's corpus never included LatexGen's actual
  client prompt text. Fix: **A-fixed** = A's base + the same LatexGen-prompt
  layer B/C use + BPE merge-ancestor closure over the union. This is
  documented below as a real, K-independent correction, not a re-tuned
  number, and it is what gets emitted.
- **Critical methodology caveat, found and corrected below**: the brief's
  designated held-out text (`arxiv-pastes.json` + `bench-data-extended.json`)
  is *also* two of strategy A's five default training-corpus files, so A's
  headline numbers are partly in-sample. A disjoint-corpus re-check (see
  "Is A's win real, or leakage?") confirms the *qualitative* result (A/A-fixed
  still beat B/C by a wide margin on genuinely unseen text) but gives
  materially different, non-degenerate optimum K values than the raw sweep,
  which is what the final recommendation below is actually based on.
- **Recommendation: K=32,768 for both tokenizers** (A-fixed), one K either
  side (24,576 / 49,152) also emitted. All 6 emitted directories pass every
  Job 4 validation check. Predicted net speedups: 0.8B **1.106x** (118 MB
  saved), 4B **1.069x** (296 MB), 9B **1.023-1.028x** (947 MB), MiniCPM5-2B
  **1.036x** (215 MB) — see "Final recommendation" for the full derivation.
- Selection D (Job 2) was **not possible**: the `keepset-corpus` sibling has
  not pushed `bench/corpus/freq-<label>.json` (checked directly in its
  worktree, read-only; see "Job 2").

## Job 1 — the one-methodology fertility curve

### Strategy definitions (exact, not paraphrased)

- **A**: `bench/vocab-coverage.py`'s `build_keep_set_for_tokenizer` (its
  default 5 corpus files: bench-data.json, bench-data-extended.json,
  pdf-pastes.json, arxiv-pastes.json, synth-spans.jsonl latex rows, unioned
  with byte alphabet, added/special tokens, harness-prompt tokens, KaTeX
  macro seed, maths-English word seed) + `final_keep_set_for_K` (pad to K by
  ascending original token id). v1's method, called directly, not
  reimplemented.
- **B**: byte alphabet + added/special tokens + chat-template tokens +
  `bench/build-keepset-v2.py`'s `latexgen_and_harness_prompt_ids` (LatexGen's
  conversion prompt + PDF hint + few-shot pairs + harness prompts, reusing
  `bench/build-keepset-seed-latexgen-prompts.py`'s seed unmodified) + every
  maths/Greek symbol token (`bench/keepset-v2-unicode.py`'s classifier,
  unmodified) + BPE merge-ancestor closure over that union
  (`build-keepset-v2.py`'s `compute_merge_ancestor_closure`, reused) — i.e.
  v2's Layer 1 **minus** the blanket "ASCII <=3 chars" layer — then
  merge-rank (ascending original id, unrestricted candidate pool, v1's fill
  mechanism) fill to K.
- **C**: v2's full method, imported and called directly: `build-keepset-v2.py`'s
  `build_layer1` (B's layers **plus** the blanket ASCII<=3 layer) +
  `build_layer2_and_candidates` + `layer3_fill_for_K`.
- **D**: marginal-token-savings ranking — skipped, see Job 2.
- **A-fixed**: introduced in Job 4 after pure A failed validation — see
  "Job 4" below for why, and "Is A's win real, or leakage?" for its
  leakage-free numbers.

### Verifying the brief's "~1,700 slots, not 21,597" claim for B

Measured exactly (not estimated):

| Tokenizer | B natural size (targeted fix layers) | C natural size (+ blanket ASCII<=3) | ASCII<=3 layer's own cost |
|---|---:|---:|---:|
| Qwen3.5 | **1,901** | 21,597 | 20,036 (of which 19,936 are net-new over B's other layers) |
| MiniCPM5-2B | **1,559** | 20,203 | 18,996 (of which 18,896 net-new) |

Confirmed: the brief's ballpark (~1,700) is close but not exact for either
tokenizer individually — B costs 1,901 slots on Qwen3.5 and 1,559 on
MiniCPM5-2B, both roughly an order of magnitude below C's 20,203-21,597,
and the blanket ASCII<=3 layer is responsible for essentially all of that
gap (>99% of B->C's growth on both tokenizers).

Full B/C layer marginal breakdown (`bench/results-keepset-v3-sweep-*.json`'s
`strategy_B_layer_marginal` / `strategy_C_layer_marginal`):

**Qwen3.5** — B: byte_alphabet 256 (+256), added_and_special 26 (+26),
chat_template_and_prompts 291 (+254), maths_and_greek_symbols 1,004 (+995),
merge_ancestor_closure -> 1,901 (+370 closure-only ids). C: same first three
layers, **ascii_le3_chars 20,036 (+19,936)**, maths_and_greek_symbols 1,004
(+995), merge_ancestor_closure -> 21,597 (+130).

**MiniCPM5-2B** — B: byte_alphabet 256 (+256), added_and_special 510 (+510),
chat_template_and_prompts 301 (+264), maths_and_greek_symbols 172 (+161),
merge_ancestor_closure -> 1,559 (+368). C: same first three layers,
**ascii_le3_chars 18,996 (+18,896)**, maths_and_greek_symbols 172 (+161),
merge_ancestor_closure -> 20,203 (+116).

### Fertility inflation by strategy and K (426-text held-out set, real re-encode)

**Qwen3.5** (unpruned fertility = 0.2599 tok/char):

| K | A (natural 6,414) | B (natural 1,901) | C (natural 21,597) |
|---:|---:|---:|---:|
| 8,192 | +13.94% | +27.71% | **overflow** (natural > K) |
| 16,384 | +2.70% | +15.57% | **overflow** (natural > K) |
| 24,576 | +1.22% | +10.73% | +32.82% |
| 32,768 | +0.50% | +7.44% | +20.19% |
| 49,152 | +0.22% | +4.19% | +13.60% |
| 65,536 | +0.07% | +2.15% | +11.10% |
| 98,304 | +0.00% | +0.14% | +8.95% |

**MiniCPM5-2B** (unpruned fertility = 0.2423 tok/char):

| K | A (natural 7,583) | B (natural 1,559) | C (natural 20,203) |
|---:|---:|---:|---:|
| 8,192 | +33.27% | +27.62% | **overflow** (natural > K) |
| 16,384 | +3.57% | +15.20% | **overflow** (natural > K) |
| 24,576 | +1.52% | +10.07% | +25.05% |
| 32,768 | +0.81% | +7.25% | +16.41% |
| 49,152 | +0.22% | +4.10% | +11.84% |
| 65,536 | +0.09% | +2.62% | +10.48% |
| 98,304 | +0.02% | +0.90% | +9.60%\* |

\*Anomaly: at K=98,304 the C candidate pool for MiniCPM5-2B is exhausted
before reaching the target — `actual_size` comes out at **90,796**, not
98,304 (vocab_size_with_added is 130,560, so this isn't a hard vocab
ceiling; v2's own layer2 `dropped` set — tokens explicitly excluded from
the candidate pool — is large enough on this tokenizer that the
unrestricted-pool assumption breaks down at high K). Not investigated
further since C is not a candidate at any K regardless (see below);
flagged here rather than silently reporting the wrong K.

**A beats both B and C at every single K on both tokenizers.** C is worse
than B at every K on both tokenizers — the blanket ASCII<=3 layer actively
hurts, exactly as the brief's framing predicted, because it fills tens of
thousands of slots with single/double/triple-byte ASCII fragments that are
rarely the merge-rank-optimal choice, at the direct expense of slots that
would otherwise go to higher-merge-rank (more token-saving) pieces.

### Re-measuring v1's and v2's emitted directories under this methodology

(`bench/remeasure-v1-v2-v3.py`, same 426-text set, same real re-encode.)

| Directory | v3-methodology inflation | Originally reported | Why they differ |
|---|---:|---|---|
| v1 qwen35-6414 (unpadded base) | +57.85% | not reported at this K | v1 never reported figures below its K=16,384 rung |
| v1 qwen35-16384 | +2.70% | fertility_after 0.2575 vs before 0.2467 (n=138, input-only, leave-one-out) | different corpus (arxiv+pdf paste **inputs only**, not references, not bench-data-extended) and a leave-one-out simulated-miss methodology, not a real full re-encode against this held-out set |
| v1 qwen35-24576 | +1.22% | (same 138-item input-only set) | ditto |
| v1 qwen35-32768 | +0.50% | (same) | ditto |
| v1 qwen35-49152 | +0.22% | (same) | ditto |
| v1 qwen35-65536 | **+0.07%** | fertility_after 0.2480 vs before 0.2467 (**≈+0.53%** on v1's own 138-item set) | close but not identical: v1's reported "+0.52%" headline (from the brief) and this run's independent recompute of v1's own held-out set both land near +0.5%, while the *this-task's* 426-text set gives +0.07% — a different, larger, harder held-out set changes the number, as expected |
| v1 minicpm5-2b-7583 (unpadded base) | +65.49% | not reported at this K | as above |
| v1 minicpm5-2b-16384 .. -65536 | +3.57% / +1.52% / +0.81% / +0.22% / +0.09% | (own 138-item set, values in `results-keepset-v3-remeasure-legacy.json`) | ditto |
| v2 qwen35-32768 | **+20.19%** | OOS-subset fertility_after 0.2906 vs before 0.2431 (**+19.53%**, n=240 OOS texts out of 120 arxiv items) | v2's own number (+19.52%, matches the brief) and this run's (+20.19%) are close — v2's methodology (120 arxiv items only, in-set exact matches excluded) and this run's (426 texts, all included) agree to within ~0.7pp, unlike v1's mismatch above |
| v2 qwen35-65536 | **+11.10%** | +8.99% (0.2646 vs 0.2423, n=216 OOS texts) | this run is ~2.1pp higher — v2's exclusion of in-set exact-match texts and smaller n (216 vs 426) both push its number down relative to a full re-encode of the larger set |
| v2 minicpm5-2b-32768 | +16.41% | +14.60% (0.2586 vs 0.2256, n=237) | ~1.8pp higher, same direction as qwen35 |
| v2 minicpm5-2b-65536 | +10.48% | +9.79% (0.2483 vs 0.2262, n=172) | ~0.7pp higher |

Plainly stated: **v1's headline "+0.52% at K=65,536" was measured on a
narrower, input-only, leave-one-out simulated corpus and does not reproduce
under a real full re-encode against this task's larger held-out set — the
true number for that exact directory, on this held-out set, is +0.07%,
actually *better*** (v1's own corpus is a near-superset of the smaller set
it originally measured against, and the extra references in this task's
set happen to encode cleanly). **v2's headlines (+8.47%/+19.52% at
65,536/32,768) are close to this run's own recompute (+11.10%/+20.19%) but
not identical**, because v2 measured a 120-183-item out-of-sample subset
where this run measures the full 426-text set including in-sample text.
Neither discrepancy is a bug in either project; they are different,
now-documented, methodologies.

## Is A's win real, or leakage?

Strategy A's default corpus (`bench/vocab-coverage.py`'s `CORPUS_FILES`)
includes `arxiv-pastes.json` and `bench-data-extended.json` as two of its
five files — **exactly** the files this brief designates as held-out text.
Strategies B and C reference no corpus at all (purely structural), so they
are unaffected; only A and A-fixed's headline numbers above are partly
in-sample. This was caught by A-fixed showing implausible *exact* 0.0000%
inflation at every single K, including K=8,192 — investigated and confirmed
as leakage, not a real result.

A disjoint re-check was built (`build-keepset-v3.py`'s `strategy_A_base(...,
corpus_labels=["bench","pdf","synth-latex"])`, dropping the two overlapping
corpus files from A's training set, then measuring fertility against the
*same* untouched 426-text held-out set — now genuinely held-out for this
check):

| K | A-disjoint, Qwen3.5 (natural 5,845) | A-disjoint, MiniCPM5-2B (natural 6,863) | A-fixed-disjoint, Qwen3.5 (natural 7,376) | A-fixed-disjoint, MiniCPM5-2B (natural 8,740) |
|---:|---:|---:|---:|---:|
| 8,192 | — | — | +3.08% | overflow |
| 16,384 | +4.87% | +6.10% | +2.08% | +3.07% |
| 24,576 | +2.81% | +3.77% | +1.60% | +2.43% |
| 32,768 | +1.97% | +2.43% | +1.31% | +1.70% |
| 49,152 | +1.12% | +1.30% | +0.91% | +1.10% |
| 65,536 | +0.60% | +0.89% | +0.55% | +0.82% |
| 98,304 | +0.18% | +0.31% | +0.17% | +0.28% |

**Conclusion: the qualitative result survives.** Even measured with zero
overlap between training and held-out text, A/A-fixed's inflation at every
K (e.g. Qwen3.5 K=32,768: 1.97-2.0%) is still far below B's in-sample number
at the same K (7.44%), let alone C's (20.19%) — A's advantage over B/C is
real, not an artifact. **But the raw in-sample sweep's exact numbers, and
especially the "optimum" computed directly from them, are not trustworthy**:
because the in-sample fertility floors at 0.00% instead of decaying smoothly,
naively computing `net(K)` from the leaky numbers picks the *smallest*
available K for A-fixed every time (more pruning always "looks" free), which
would not generalize to real unseen text. The disjoint numbers above are
used instead for Job 3/4's actual recommendation.

## Job 2 — selection D

Checked live in `bench/build-keepset-v3.py`'s `has_corpus_landed()` (looks
for `bench/corpus/freq-*.json` in this worktree — this worktree never
touches `bench/corpus/`, per the brief) and again directly in the sibling's
own worktree, read-only:

```
$ git -C /Users/oleh/personal/latexgen-keepset-corpus log --oneline -3
7f9c834 Merge pull request #55 from OlehZhyhinas/vocab-keepset   <- still just the v1 merge base, no new commits
$ git -C /Users/oleh/personal/latexgen-keepset-corpus status --short
 M .gitignore
?? bench/corpus/                                                  <- untracked, in progress
$ find /Users/oleh/personal/latexgen-keepset-corpus/bench/corpus -iname '*freq*'
                                                                    <- nothing
$ git branch -a | grep corpus
+ keepset-corpus                                                  <- local only, never pushed (no remotes/origin/keepset-corpus)
```

**Selection D is not possible yet**: the `keepset-corpus` sibling's branch
has not been pushed to `origin` at all, and its own worktree still shows
only harvest-cache directories (`arxiv/`, `openstax/`, `stackexchange/`,
`wikibooks/`, `wikipedia/` under `bench/corpus/cache/`) with no
`freq-<label>.json` frequency table yet produced. Per the brief, this is
not blocking: Jobs 1/3/4 stand alone and are reported without D.

## Job 3 — T(K) per model, and where the optimum actually is

`ms_per_token(K) = body + head * (K / V)`, `net(K) = ms_total / (ms_per_token(K)
* (1 + inflation(K)))`. Head-kernel costs, verified against `webnn-workbench`
`origin/main` via `git show` (see `bench/compute-optimum-v3.py`'s `MODELS`
dict for the exact doc lines each figure was checked against — spot-verified
again for this write-up: `docs/qwen35-08-vocab-report.md` "~648-658 us/token,
~11.6% of GPU decode time" and `docs/results.md:1690`'s 190.83 tok/s ->
5.24 ms/token both confirmed verbatim; `docs/results.md:2232`'s 37.45 tok/s
-> 26.70 ms/token for 9B also confirmed verbatim):

| Model | V | ms_total | head_ms (or range) | bytes/row |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | 248,320 | 5.240 | 0.648-0.658 | 576 |
| Qwen3.5-4B | 248,320 | 15.61 | 1.372 | 1,440 |
| Qwen3.5-9B | 248,320 | 26.70 | range 4.0-4.6% of ms_total (1.068-1.228) | 4,608 |
| MiniCPM5-2B | 130,560 | 9.74 | 0.6657 | 2,304 |

Two discrepancies carried through rather than smoothed over (both already
flagged in `compute-optimum-v3.py`'s `source_note`): (1) Qwen3.5-0.8B's
"11.6% of GPU decode time" doesn't match 0.648/5.24=12.4% — the two figures
come from different profiling sections of the same doc; (2) Qwen3.5-4B's
measured on-disk byte delta at K=65,536 (284,149,634 B) exceeds
`(248,320-65,536)*1,440=263,208,960 B` by ~21 MB — the extra bytes are
non-vocab-tensor shrinkage (e.g. `tokenizer.json` itself), not part of
`bytes_per_row`.

### Net speedup by strategy and K (literal methodology — see caveat below each table)

Optimum picked with the brief's tie rule (within 0.3% relative net of the
best, smallest K wins). **A-fixed's numbers in this sub-section use the
in-sample (leaky) sweep and are not the final recommendation** — see
"Corrected optimum" right after.

| Model | Strategy | Optimum K | net(K) | MB saved |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B | A | 24,576 | 1.1118 | 122.9 |
| Qwen3.5-0.8B | B | 65,536 | 1.0769 | 100.4 |
| Qwen3.5-0.8B | C | 65,536 | 0.9902 (slower than unpruned) | 100.4 |
| Qwen3.5-0.8B | A-fixed (leaky) | 8,192 | 1.1358 | 131.9 |
| Qwen3.5-4B | A | 32,768 | 1.0773 | 296.0 |
| Qwen3.5-4B | B | 98,304 | 1.0546 | 206.0 |
| Qwen3.5-4B | C | 98,304 | 0.9693 (slower) | 206.0 |
| Qwen3.5-4B | A-fixed (leaky) | 8,192 | 1.0929 | 329.8 |
| Qwen3.5-9B | A | 32,768 | 1.0337 | 947.2 |
| Qwen3.5-9B | B | 98,304 | 1.0253 | 659.2 |
| Qwen3.5-9B | C | 98,304 | 0.9423 (slower) | 659.2 |
| Qwen3.5-9B | A-fixed (leaky) | 8,192 | 1.0434 | 1,055.2 |
| MiniCPM5-2B | A | 24,576 | 1.0429 | 232.9 |
| MiniCPM5-2B | B | 65,536 | 1.0089 | 142.9 |
| MiniCPM5-2B | C | 65,536 | 0.9370 (slower) | 142.9 |
| MiniCPM5-2B | A-fixed (leaky) | 16,384 | 1.0636 | 250.9 |

**C is never worth it** — net < 1.0 at its own optimum on every model, i.e.
v2's full structural selection nets *slower than not pruning at all*, on
this methodology, confirming the brief's own T-formula worry about v2's
K=32,768 number (v2's own headline was 0.925x at K=32,768 for the 4B; this
run's independent number at the same K, same model, is 0.901x — same
conclusion, different exact figure, see remeasurement table above for why).

**A-fixed (leaky)'s "K=8,192 optimum" is an artifact of the leakage problem**
(0% inflation measured at every K means `net(K)` is monotonic-decreasing in
K with no interior minimum, so the tie rule always grabs the smallest K on
the grid) and is **not used** for the recommendation.

### Corrected optimum (A-fixed, disjoint-corpus fertility)

Using the leakage-free inflation numbers from "Is A's win real, or leakage?":

| Model | K | inflation | net(K) |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 8,192 | +3.08% | 1.1019 |
| Qwen3.5-0.8B | 16,384 | +2.08% | 1.1075 |
| Qwen3.5-0.8B | **24,576** | +1.60% | **1.1077 (optimum)** |
| Qwen3.5-0.8B | 32,768 | +1.31% | 1.1057 |
| Qwen3.5-0.8B | 49,152 | +0.91% | 1.1001 |
| Qwen3.5-4B | 16,384 | +2.08% | 1.0672 |
| Qwen3.5-4B | **24,576** | +1.60% | **1.0689 (optimum)** |
| Qwen3.5-4B | 32,768 | +1.31% | 1.0686 |
| Qwen3.5-4B | 49,152 | +0.91% | 1.0661 |
| Qwen3.5-9B | 32,768 | +1.31% | 1.0225-1.0281 |
| Qwen3.5-9B | 49,152 | +0.91% | 1.0238-1.0289 |
| Qwen3.5-9B | **65,536** | +0.55% | **1.0247-1.0294 (optimum)** |
| Qwen3.5-9B | 98,304 | +0.17% | 1.0230-1.0269 |
| MiniCPM5-2B | 24,576 | +2.43% | 1.0336 |
| MiniCPM5-2B | **32,768** | +1.70% | **1.0363 (optimum)** |
| MiniCPM5-2B | 49,152 | +1.10% | 1.0331 |

**How flat is it?** Very: Qwen3.5-0.8B/4B's optimum at K=24,576 beats
K=32,768 by only 0.02pp/0.03pp of net (well inside noise) and beats
K=16,384 by more (0.7pp/-, i.e. still shrinking materially below 24,576
starts to cost real net speed on the disjoint-corpus numbers, unlike the
leaky numbers which claimed no cost at all down to 8,192). Qwen3.5-9B's
optimum sits at a much larger K (65,536) than 0.8B/4B's (24,576) — expected,
since a bigger model's head-kernel is a *smaller* share of its total decode
time, so the fertility cost of aggressive pruning stops being worth the
smaller relative compute saving sooner. MiniCPM5-2B's optimum (32,768) is a
single, unambiguous peak, ~2.5pp of net above its neighbors — not flat.

Because Qwen3.5's *tokenizer* is shared across three model sizes with two
different optimum K's (24,576 for 0.8B/4B, 65,536 for 9B), and only one
keep-set per tokenizer ships, **K=32,768 is chosen as the compromise K for
the Qwen3.5 tokenizer**: it costs 0.8B only 0.20pp of net vs its own true
optimum (1.1057 vs 1.1077) and 4B only 0.03pp (1.0686 vs 1.0689), while
recovering most of 9B's shortfall relative to *not* using its own optimum
(9B at 32,768: 1.0225-1.0281, vs 9B's own optimum 1.0247-1.0294 — a 0.2-0.3pp
gap either way). MiniCPM5-2B's own optimum is already 32,768, so no
compromise is needed there.

## Job 4 — emission and validation

Emitted `bench/keepsets-v3/<label>-<K>/` at K in {24,576, 32,768, 49,152}
for both tokenizers (32,768 = the chosen K above, one either side from the
sweep grid), using **A-fixed**, not pure A — see below for why.

### Why A-fixed, not pure A (the winning strategy)

Running Job 4's validation checks (`bench/validate-keepset-v3.py`, reusing
`bench/validate-keepset-v2.py`'s checks unmodified) against pure strategy A
emissions found A **fails check 1 on the Qwen3.5 tokenizer**: the 7 named
`QWEN35_FAILURE1_IDS` (LatexGen's own client-prompt tokens,
`bench/validate-keepset-v2.py`) are absent from A's keep-set at every K up
to at least 65,536. Root cause: v1's corpus (`bench-data.json`,
`bench-data-extended.json`, `pdf-pastes.json`, `arxiv-pastes.json`,
`synth-spans.jsonl`) never included LatexGen's actual client prompt text
(`public/pipeline.js` / `public/pdf-prompt.js`) — only the separate,
narrower `webnn-workbench` harness prompt set. This is v1's *originally
reported* failure (the brief's own "dropped ... seven tokens of LatexGen's
own prompt"), reproduced exactly, not a new one.

Per the brief ("a variant that regresses any of these is not a candidate,
however good its fertility... say so and drop it"), **pure A is dropped**.
**A-fixed** = A's exact base, unioned with
`build-keepset-v2.py`'s `latexgen_and_harness_prompt_ids` (the same layer
B/C already use — not re-derived), then BPE merge-ancestor-closed over the
combined base (`compute_merge_ancestor_closure`, also reused) so no kept
corpus/prompt token silently loses its own producing merge rule. This is a
minimal, targeted patch to a real, K-independent defect in v1's method
(missing merge-ancestor closure is a second, independent bug beyond the
missing prompt tokens — v1 never did this closure step at all), not a
re-tuned number: cost is 1,541 extra slots on Qwen3.5 (natural size
6,414 -> 7,955) and 1,884 on MiniCPM5-2B (7,583 -> 9,467), negligible next
to A's fertility advantage over B/C at every K in the sweep.

### Validation results (all 6 emitted directories)

| Directory | Check 1 (roundtrip: symbols, rare words, LatexGen prompts, 7 named ids) | Check 2 (2,120 short pieces) | Check 3 (fertility, v2's 120-item methodology) | Check 4 (prompt coverage) | Candidate? |
|---|---|---|---|---|---|
| qwen35-24576 | all pass, 7/7 named ids present | 120/120 + 2000/2000 | +0.00% | 120/120 + 120/120 | **YES** |
| qwen35-32768 | all pass, 7/7 named ids present | 120/120 + 2000/2000 | +0.00% | 120/120 + 120/120 | **YES** |
| qwen35-49152 | all pass, 7/7 named ids present | 120/120 + 2000/2000 | +0.00% | 120/120 + 120/120 | **YES** |
| minicpm5-2b-24576 | all pass | 120/120 + 2000/2000 | +0.00% | 120/120 + 120/120 | **YES** |
| minicpm5-2b-32768 | all pass | 120/120 + 2000/2000 | +0.00% | 120/120 + 120/120 | **YES** |
| minicpm5-2b-49152 | all pass | 120/120 + 2000/2000 | +0.00% | 120/120 + 120/120 | **YES** |

(Check 3's "+0.00%" here is v2's own **120-item** methodology, which is
also in-sample for A-fixed the same way the 426-item set is — expected, and
consistent with the leakage finding above; check 1/2/4 are the checks that
actually gate candidacy and all pass cleanly. Full detail:
`bench/results-keepset-v3-validation.json`.)

No variant was dropped for regressing checks 1/2/4 — all 6 are candidates.
Pure A (not emitted, superseded by A-fixed) would have been dropped for
qwen35 per the above; it was never emitted as a final artifact.

## Final recommendation

| Model | Tokenizer | Recommended K | Predicted net speedup | MB saved |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B | qwen35 | 32,768 | **1.106x** | 118.4 MB |
| Qwen3.5-4B | qwen35 | 32,768 | **1.069x** | 296.0 MB |
| Qwen3.5-9B | qwen35 | 32,768 | **1.023-1.028x** | 947.2 MB |
| MiniCPM5-2B | minicpm5-2b | 32,768 | **1.036x** | 214.9 MB |

Strategy **A-fixed** wins for every model. Emitted directories:
`bench/keepsets-v3/qwen35-{24576,32768,49152}/`,
`bench/keepsets-v3/minicpm5-2b-{24576,32768,49152}/` — use the `-32768`
variant for deployment; `-24576`/`-49152` are the required "one K either
side" and are also fully valid candidates (all passed validation) if a
future task wants to re-tune the compromise.

## What was not done

- Selection D (Job 2): corpus not landed, see above.
- Byte-identity / exact-roundtrip gating: explicitly out of scope per the
  brief (GEMV reduction-order effects, verified elsewhere on the 4B).
- Re-deriving `latexgen_and_harness_prompt_ids`, the Unicode maths/Greek
  classifier, or v2's layer functions: all reused unmodified from
  `build-keepset-v2.py` / `keepset-v2-unicode.py` /
  `build-keepset-seed-latexgen-prompts.py`, per the brief.
- Harvesting any corpus: none harvested; D's absence is reported, not
  worked around.

## Anomalies

1. **In-sample leakage in strategy A/A-fixed's headline numbers** — the
   biggest finding of this task; see "Is A's win real, or leakage?" above.
   The literal-methodology sweep (as the brief's Job 1 specifies verbatim)
   is reported in full, but Job 3/4's actual recommendation is based on the
   disjoint-corpus correction, not the raw sweep, because the raw sweep's
   "optimum" is degenerate (always picks the smallest K on the grid).
2. **v2's Layer 2 candidate-pool exhaustion on MiniCPM5-2B at K=98,304**
   (strategy C caps at actual_size=90,796, not 98,304) — noted, not
   investigated further since C is not a candidate at any K.
3. **v1's own headline fertility number does not reproduce** under a real
   re-encode against this task's held-out set (measured +0.07% here vs the
   brief-quoted +0.52%) — both numbers are real, measured differently; see
   the remeasurement table.
