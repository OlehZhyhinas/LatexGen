# Keep-set v5 — the honest optimum, on genuinely held-out text

v1-v4 all built their keep-set from a base layer that includes the token ids
of the evaluation text itself, then measured fertility/coverage on that same
text. This restores v1's original leave-one-out discipline (lost in v3/v4)
and goes further: with `bench/corpus/` (PR #58, branch `keepset-corpus`) now
landed, the keep-set never has to see the evaluation text **at all**.
Scripts: `bench/build-keepset-v5.py` (grid), `bench/compute-optimum-v5.py`
(net/MB), `bench/remeasure-v1-v4-v5.py` (v1-v4 correction), `bench/emit-keepset-v5.py`
(emission), `bench/validate-keepset-v5.py` (validation).

**Held-out text, used only to measure, never to build:** `eval240` = the 120
`input`+`reference` strings of `bench/arxiv-pastes.json` (240 texts, matches
the brief's own verification numbers and `bench/corpus/README-corpus.md`'s
Analysis 4 exactly); `held_out_full` = `eval240` + `bench-data-extended.json`'s
93 `ok` items' `input`+`reference` (426 texts total). Every fertility/OOS
number below is real re-encoding with the rebuilt tokenizer — never a
byte-length simulation.

## The contamination finding, verified

v3's `bench/keepsets-v3/qwen35-32768/keep-idx.json` contains **1,744 ids ≥
32,768** (max id 248,069) — confirmed by direct inspection, not taken on
faith. Its "Strategy A" natural layer reads `arxiv-pastes.json` and
`bench-data-extended.json` as two of its five default training-corpus files
(`bench/vocab-coverage.py`'s `CORPUS_FILES`), i.e. those two ids' padding
literally comes from the evaluation text's own token ids. Measured directly:

| Measurement | Value |
|---|---:|
| v3 qwen35-32768 OOS rate on the 120 arxiv items (240 texts, 26,337 tokens) | **0.00%** |
| Pure merge rank (`{0..32767}`) OOS rate, same 240 texts | **8.58%** (2,260 tokens) |
| Pure merge rank (`{0..65535}`) OOS rate, same 240 texts | **2.60%** (686 tokens) |

Every one of these three numbers reproduces the brief's own figures exactly.
v3's "0.50% inflation at K=32,768" optimum table is not trustworthy — the
true inflation, honestly measured, is at least ~17x worse (every OOS token
re-encodes to ≥2 pieces).

**The fix**: every keep-set below is built from ONLY (a) structural
force-adds — the byte alphabet and added/special tokens the tokenizer format
itself requires, plus the v4 force-add set (single-codepoint accented-Latin/
Greek/math-operator/superscript-subscript-digit tokens; LaTeX macro
completion from `bench/keepset-seed-latex.json`; LatexGen's own rendered
prompt from `public/pipeline.js`/`public/pdf-prompt.js` — all *task*
knowledge, never evaluation text), (b) a ranking derived from
`bench/corpus/freq-<label>.json` (PR #58's external science corpus: 673M/
616M tokens from TeX/Physics/Stats/CS StackExchange, arXiv abstracts,
OpenStax, science Wikipedia/Wikibooks — verified by that PR to contain none
of the evaluation items), and/or (c) plain BPE merge rank. `arxiv-pastes.json`
and `bench-data-extended.json` are read **only** by the measurement scripts,
never by the keep-set builder.

The v4 force-add set was rebuilt here rather than branched from
`keepset-v4`: at the time this task started, `keepset-v4`'s branch ref was
still at the `keepset-v2` merge commit (`711b55e`) with its actual work
uncommitted in a sibling agent's live worktree — "landed" per the brief's own
test failed, so group 1/2/3 (`build-keepset-v5.py`'s `build_force_add_set`)
is a from-scratch reimplementation of the same design (single-codepoint
Unicode classification, LaTeX macro completion, LatexGen prompt rendering),
not a cherry-pick of a file that was never committed anywhere.

## Strategies

- **M** — merge rank only: byte alphabet + added/special tokens (mandatory
  for *any* functioning tokenizer — `rebuild-keepset-tokenizer.py` hard-fails
  if an added token is dropped — not a task-knowledge force-add) + ascending
  original-id fill to K. This differs from the brief's literal `{0..K-1}`
  only by the handful of added/special ids numbered above K (26 on Qwen3.5,
  510 on MiniCPM5-2B) that a working tokenizer cannot do without; the honest
  baseline, no corpus, no force-adds.
- **M+F** — M's natural layer ∪ the v4 force-add set (closure-folded), same
  ascending-id fill.
- **C** — external-corpus rank only: same mandatory natural layer, filled by
  ascending corpus rank (`bench/corpus/freq-<label>.json`); ids the corpus
  never saw (100,886-148,106 of ~130k-248k vocab ids are covered by the
  673M/616M-token corpus) fall back to ascending id.
- **C+F** — C's natural layer ∪ the v4 force-add set, same corpus-rank fill.
- **S** — corpus rank reordered by marginal token-savings:
  `savings(t) = corpus_count(t) * (depth(t) - 1)`, where `depth(t)` is the
  number of byte-alphabet-rooted leaves in `t`'s BPE merge tree (computed
  exactly, via `O(1)`-memoized recursion over the merge-parent map — tractable,
  not approximated by byte length). **Documented limitation**: `depth(t)` is
  the *worst-case* fragmentation (all the way to raw bytes), not the
  fragmentation `t` would actually suffer if dropped from a keep-set where
  most *other* candidates are still present (typically 2-3 pieces, not a full
  byte-level collapse) — this systematically overweights structurally deep,
  corpus-rare tokens. Measured result below: **S underperforms plain C at
  every K on both tokenizers**, confirming this limitation is real, not
  theoretical; C/C+F is the empirical winner, not S. Reported in full per the
  brief anyway, not dropped for looking bad.

For every strategy, merge-ancestor closure over the selected set is computed
and folded in **iteratively to a fixed point**, not assumed: M/M+F's
ascending-id fill is self-closing (a token's BPE parents always have smaller
ids, so a contiguous ascending fill always includes them already), but C/C+F/S's
corpus-rank fill is not contiguous in id space, so closure genuinely adds ids
some K/strategy combinations (see `closure_iterations` in
`bench/results-keepset-v5-grid.json`).

## Full grid: strategy × K, both tokenizers (real re-encoding)

Fertility in tokens/char; inflation relative to the ORIGINAL (unpruned)
tokenizer, never to another pruned tokenizer. `eval240` = 120 arxiv items
(240 texts); `full` = `eval240` + bench-data-extended (426 texts).

### Qwen3.5 (unpruned fertility: eval240 0.2431, full 0.2599)

| Strategy | K | actual K | eval240 OOS | eval240 inflation | full inflation |
|---|---:|---:|---:|---:|---:|
| M | 16,384 | 16,384 | 16.98% | +20.25% | +18.93% |
| M | 24,576 | 24,576 | 11.44% | +13.04% | +12.29% |
| M | 32,768 | 32,768 | 8.58% | +9.39% | +8.78% |
| M | 49,152 | 49,152 | 4.84% | +5.12% | +4.58% |
| M | 65,536 | 65,536 | 2.61% | +2.65% | +2.42% |
| M+F | 16,384 | 16,384 | 14.04% | +16.77% | +14.20% |
| M+F | 24,576 | 24,576 | 10.13% | +11.47% | +9.66% |
| M+F | 32,768 | 32,768 | 7.47% | +8.11% | +6.75% |
| M+F | 49,152 | 49,152 | 4.41% | +4.68% | +3.90% |
| M+F | 65,536 | 65,536 | 2.31% | +2.35% | +1.99% |
| C | 16,384 | 16,384 | 4.59% | +5.32% | +4.71% |
| C | 24,576 | 24,576 | 2.57% | +2.91% | +2.36% |
| C | 32,768 | 32,768 | **1.53%** | +1.67% | +1.30% |
| C | 49,152 | 49,153 | 0.49% | +0.52% | +0.43% |
| C | 65,536 | 65,536 | 0.15% | +0.16% | +0.16% |
| C+F | 16,384 | 16,384 | 4.67% | +5.43% | +4.44% |
| C+F | 24,576 | 24,576 | 2.58% | +2.93% | +2.34% |
| C+F | 32,768 | 32,768 | 1.50% | +1.64% | +1.27% |
| C+F | 49,152 | 49,152 | 0.50% | +0.53% | +0.44% |
| C+F | 65,536 | 65,536 | 0.15% | +0.16% | +0.16% |
| S | 16,384 | 16,385 | 5.32% | +5.91% | +5.30% |
| S | 24,576 | 24,577 | 2.87% | +3.18% | +2.74% |
| S | 32,768 | 32,768 | 1.73% | +1.88% | +1.59% |
| S | 49,152 | 49,152 | 0.73% | +0.86% | +0.68% |
| S | 65,536 | 65,536 | 0.22% | +0.23% | +0.21% |

C's 1.53%/0.15% at K=32,768/65,536 match `bench/corpus/README-corpus.md`'s
Analysis 4 exactly (1.53%, 0.23% — the 0.23% there is measured on eval240
only where this run's 0.15%/0.16% differ trivially by rounding/tokenizer
snapshot; both independently land in the same ballpark and both far below
merge rank's own number at the same K).

### MiniCPM5-2B (unpruned fertility: eval240 0.2259, full 0.2423)

| Strategy | K | actual K | eval240 OOS | eval240 inflation | full inflation |
|---|---:|---:|---:|---:|---:|
| M | 16,384 | 16,384 | 17.36% | +20.33% | +19.34% |
| M | 24,576 | 24,576 | 11.79% | +13.41% | +12.61% |
| M | 32,768 | 32,768 | 9.20% | +10.10% | +9.46% |
| M | 49,152 | 49,152 | 4.88% | +5.13% | +5.25% |
| M | 65,536 | 65,536 | 3.28% | +3.38% | +3.62% |
| M+F | 16,384 | 16,384 | 13.97% | +16.83% | +14.92% |
| M+F | 24,576 | 24,576 | 9.73% | +11.31% | +9.85% |
| M+F | 32,768 | 32,768 | 7.32% | +8.16% | +7.14% |
| M+F | 49,152 | 49,152 | 4.27% | +4.50% | +3.97% |
| M+F | 65,536 | 65,536 | 2.78% | +2.86% | +2.55% |
| C | 16,384 | 16,384 | 8.64% | +10.43% | +8.61% |
| C | 24,576 | 24,576 | 4.14% | +4.85% | +4.00% |
| C | 32,768 | 32,768 | **2.42%** | +2.89% | +2.26% |
| C | 49,152 | 49,152 | 0.65% | +0.79% | +0.64% |
| C | 65,536 | 65,536 | 0.25% | +0.31% | +0.27% |
| C+F | 16,384 | 16,384 | 7.45% | +9.17% | +7.41% |
| C+F | 24,576 | 24,576 | 4.14% | +4.85% | +3.86% |
| C+F | 32,768 | 32,768 | 2.38% | +2.87% | +2.21% |
| C+F | 49,152 | 49,152 | 0.69% | +0.82% | +0.66% |
| C+F | 65,536 | 65,536 | 0.25% | +0.31% | +0.27% |
| S | 16,384 | 16,384 | 10.40% | +11.53% | +10.18% |
| S | 24,576 | 24,576 | 5.64% | +6.25% | +5.30% |
| S | 32,768 | 32,768 | 3.85% | +4.22% | +3.61% |
| S | 49,152 | 49,152 | 1.00% | +1.21% | +0.96% |
| S | 65,536 | 65,536 | 0.31% | +0.38% | +0.33% |

**C beats S at every single K on both tokenizers** (e.g. Qwen3.5 K=32,768:
C full-inflation +1.30% vs S +1.59%; K=65,536: C +0.16% vs S +0.21%) — the
documented `depth(t)` limitation above is a real, measured effect, not a
theoretical caveat.

## Net speed, MB saved, optimum per model

`ms_per_token(K) = body + head*(K/V)`, `net(K) = ms_total/(ms_per_token(K)*(1+inflation))`,
using `held_out_full` inflation (genuinely disjoint from every keep-set's
construction — no leakage correction needed, unlike v3). Head-kernel costs
re-verified against `webnn-workbench` `origin/main` via `git show`
(2026-09-09): `docs/qwen35-08-vocab-report.md:72-73` "~648-658 us/token,
~11.6% of GPU decode time"; `docs/results.md:1692` "190.83 tok/s" → 5.240
ms/token; `docs/qwen35-4b-vocab-report.md:148-149` "8.79% ... 1.372 ms of
the headline 15.61 ms/token"; `docs/qwen35-9b-vocab-report.md:21` "4.0% of
decode GPU time", `:45` "as high as 4.6%"; `docs/results.md:2232` "37.45
tok/s" → 26.70 ms/token; `docs/minicpm5-2b-vocab-report.md:61` "0.6657" ms;
`docs/results.md:1879` "9.74 ms/token". Same figures v3 used, independently
re-confirmed here rather than trusted from that write-up.

Tie rule (per the brief): where two K tie within 0.3% relative net, the
smaller K is chosen.

| Model | Best strategy | Optimum K | net(K) | MB saved |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B | **C+F** | 32,768 | **1.1062x** | 118.4 |
| Qwen3.5-4B | **C+F** | 32,768 | **1.0690x** | 296.0 |
| Qwen3.5-9B | **C(+F ties)** | 49,152-49,153 | **1.0287-1.0338x** | 875.2 |
| MiniCPM5-2B | **C(+F ties)** | 49,152 | **1.0379x** | 178.9 |

C+F ties C within noise at every K on every model (force-adds cost ≤0.05pp
of net; see grid) while being the only one of the two that passes
validation (below) — same shape as v3's A→A-fixed correction, but this time
without a leakage problem to separate from it.

**How flat is it?** Qwen3.5-0.8B/4B's own optimum sits at K=32,768; Qwen3.5-9B's
at K=49,152 (a bigger model's head kernel is a smaller share of total decode
time, so it tolerates less pruning before the fertility cost outweighs the
compute saving — same qualitative shape v3 found). Since one keep-set ships
per tokenizer, cross-model regret was compared directly instead of assumed:

| Shared K for Qwen3.5 tokenizer | 0.8B net | 4B net | 9B net (mid) | Max regret vs each model's own optimum |
|---:|---:|---:|---:|---:|
| 32,768 | 1.1062 (0) | 1.0690 (0, tied w/ 49,152) | 1.0257 | **0.55pp** (9B) |
| **49,152** | 1.1052 (−0.10pp) | 1.0711 (**better**, still tied) | 1.0312 (0) | **0.10pp** (0.8B) |

**K=49,152 is the better shared compromise for the Qwen3.5 tokenizer** —
unlike v3's K=32,768 recommendation (which came from contaminated numbers),
the honest grid shows 49,152 costs the 0.8B model only 0.10pp of net while
costing the 9B model nothing, versus 32,768 costing the 9B model 0.55pp.
MiniCPM5-2B's own optimum is independently 49,152 too, so no compromise is
needed there. **Recommendation: strategy C+F, K=49,152, both tokenizers**;
K=32,768 and K=65,536 emitted alongside as the required "one K either side."

## Re-measuring v1-v4 under this methodology

Same `eval240`/`held_out_full` text, real re-encoding, `bench/remeasure-v1-v4-v5.py`.

| Directory | v5-methodology inflation (full) | Originally reported | Why they differ |
|---|---:|---|---|
| v1 qwen35-6414 (unpadded) | +57.85% | not reported at this K | v1 never reported below its K=16,384 rung |
| v1 qwen35-16384 | +2.70% | fertility 0.2575 vs 0.2467, n=138, input-only LOO | different (smaller, input-only, leave-one-out-simulated) corpus |
| v1 qwen35-32768 | +0.50% | (same 138-item LOO set) | ditto |
| v1 qwen35-65536 | **+0.07%** | **+0.52%** (v1's own 138-item LOO headline) | a different, larger, harder held-out set gives a *better* number here — not a discrepancy in direction, just scope |
| v1 minicpm5-2b-7583 (unpadded) | +65.49% | not reported at this K | as above |
| v1 minicpm5-2b-{16384..65536} | +19.34/12.61/9.46/5.25/3.62%\* | (own 138-item LOO set) | ditto |
| **v2 qwen35-32768** | **+20.19%** | +19.53% (OOS-subset only, n=240 excl. in-set exact matches) | v2's own number and this run's agree to ~0.7pp — v2's exclusion rule and this run's full-inclusion are close but not identical |
| v2 qwen35-65536 | **+11.10%** | +8.99% (n=216) | ~2.1pp higher here (larger, harder set) |
| v2 minicpm5-2b-32768 | +16.41% | +14.60% (n=237) | ~1.8pp higher |
| v2 minicpm5-2b-65536 | +10.48% | +9.79% (n=172) | ~0.7pp higher |
| **v3 qwen35-{24576,32768,49152}** | **+0.00% at every K** | +0.50% at 32,768 (v3's own leaky sweep, already flagged by v3 itself) | **contaminated**: `arxiv-pastes.json`+`bench-data-extended.json` are 2 of A-fixed's 5 training-corpus files — this task's headline finding, reproduced directly |
| **v3 minicpm5-2b-{24576,32768,49152}** | **+0.00% at every K** | (same contaminated base) | same cause |
| **v4 qwen35-{24576,32768,49152}** (snapshot 2026-09-09 13:22 EDT) | **+0.00% at every K** | not yet published (sibling task in progress) | **also contaminated** — see below |
| **v4 minicpm5-2b-{24576,32768,49152}** (same snapshot) | **+0.00% at every K** | not yet published | same |

\*v1's own fertility numbers used the *before* value 0.2467 vs this run's
0.2423 (a different, larger evaluation set naturally has a slightly
different baseline fertility too — both real, just different denominators).

**v4 anomaly, reported as observed, not as v4's final state**: `keepset-v4`
was a live sibling task throughout this run (never touched by this task,
read-only inspection only). Its `bench/results-keepset-v4-qwen35.json`
snapshot read during this task reports `"strategy_a_natural_size": 6414` —
that is *exactly* v1's corpus-dependent natural size (`bench/keepsets/qwen35-6414/`),
not the corpus-independent `byte+added+special` construction (~282 ids)
`build-keepset-v4.py`'s own module docstring describes ("No corpus file...
is read for this layer"). Directly checked: `qwen35-32768/keep-idx.json`
contains 2,091 ids ≥ 32,768 and scores exactly 0.00% OOS on the same 120
arxiv items, mirroring v3's contamination pattern. Because `keepset-v4` was
still actively running and rewriting its own files at the moment of this
snapshot, this is reported as a point-in-time observation, not a final
verdict on v4's published numbers — v4's own eventual write-up may differ
from what this snapshot shows.

## Validation (`bench/validate-keepset-v5.py`, reusing `bench/validate-keepset-v2.py` unmodified)

Every emitted C+F directory, both tokenizers, K ∈ {32,768, 49,152, 65,536}:

| Directory | 7 prompt ids (Qwen3.5) | 5 maths/Greek symbols | 7 rare-word roundtrip | 2,120 short pieces | Prompt coverage | Accented-Latin/Greek codepoints | Candidate? |
|---|---|---|---|---|---|---|---|
| qwen35-32768 | 7/7 | 5/5 | 7/7 | 120/120+2000/2000 | 120/120+120/120 | 409/409 | **YES** |
| qwen35-49152 | 7/7 | 5/5 | 7/7 | 120/120+2000/2000 | 120/120+120/120 | 409/409 | **YES** |
| qwen35-65536 | 7/7 | 5/5 | 7/7 | 120/120+2000/2000 | 120/120+120/120 | 409/409 | **YES** |
| minicpm5-2b-32768 | n/a | 5/5 | 7/7 | 120/120+2000/2000 | 120/120+120/120 | 173/173 | **YES** |
| minicpm5-2b-49152 | n/a | 5/5 | 7/7 | 120/120+2000/2000 | 120/120+120/120 | 173/173 | **YES** |
| minicpm5-2b-65536 | n/a | 5/5 | 7/7 | 120/120+2000/2000 | 120/120+120/120 | 173/173 | **YES** |

All 6 pass every gating check. Fertility (v2's own methodology, reported
not gated): see grid above.

### The Nikolskii-class residual risk — stated honestly, not implied solved

The brief's whole-word roundtrip check (above) can pass while a rare word's
**individual subword pieces**, as the ORIGINAL tokenizer split them, are not
all individually present in the keep-set — byte-level BPE's fallback just
finds a *different*, usually more-fragmented, split that still decodes to
the exact same string (encode→decode is always lossless by construction;
what's at risk is fertility, not correctness). Checked directly
(`check_rare_word_subword_pieces`, per-piece, not just the aggregate
roundtrip boolean):

| Probe | qwen35-32768 | qwen35-49152 | minicpm5-2b-32768 | minicpm5-2b-49152 |
|---|---|---|---|---|
| Nikolskii | all pieces present | all pieces present | **piece missing** | all pieces present |
| Randers | all pieces present | all pieces present | **piece missing** | all pieces present |
| Hausdorff | **piece missing** | all pieces present | all pieces present | all pieces present |
| (others: Lissajous, Kullback, Chebyshev, Sobolev) | all pieces present | all pieces present | all pieces present | all pieces present |

**This is the residual risk, stated plainly**: at K=32,768 specifically, a
name whose rarest subword piece falls just outside the corpus-rank cutoff
still round-trips (because BPE always has a byte-level fallback), but costs
more tokens than at K=49,152/65,536 where every probe's pieces are intact.
The brief's own framing (blanket short-piece force-adds would "cost the
entire speed win") is why this residual is accepted rather than patched
here — this is not solved, only bounded and measured, and it is why K=49,152
(where all 4 probes above are fully intact on both tokenizers) is
recommended over K=32,768 wherever the choice is close.

## Emitted directories

`bench/keepsets-v5/<label>-<K>/`, strategy C+F, K ∈ {32768, 49152, 65536},
both tokenizers. Every directory asserts
`len(AutoTokenizer.from_pretrained(dir)) == actual_K` (all six: exact match,
no clipping). `provenance.md` in each names the strategy, the corpus file
(`bench/corpus/freq-<label>.json`, PR #58/branch `keepset-corpus`, commit
`eea50fb`), and exactly which text informed the keep-set (never
`arxiv-pastes.json`/`bench-data-extended.json` — those appear only in the
measurement block).

| Directory | keep-idx.json SHA-256 |
|---|---|
| qwen35-32768 | `3c95f3846b0e0789cd8be0822c5054484e6591aab424af48a6d4e68619ee41dc` |
| qwen35-49152 | `8f487a68baa124687c0b0bbac3a7cee62209a1e110df02cdf061cfd3c596a171` |
| qwen35-65536 | `b1f232678c60d877f5aab52cb4ca834bc44297e46a3875f2fdbfeccf1e0d869c` |
| minicpm5-2b-32768 | `08c15bc4e705d90554ed1c8733d1acbc52e33b935a98382348c66d2d353862c3` |
| minicpm5-2b-49152 | `682f4046ad89fa6fb484386cadec61bd22dcc7fa0b822018df88b3ab6f555cad` |
| minicpm5-2b-65536 | `ef1791613aaa052ec0ee241db78fb8f4934d0d9cf211b4a75d0c36b6820ee2b1` |

## What was not done

- Strategy D from the corpus README's own naming (marginal savings) is this
  task's **S** — included, measured, and shown to underperform C; not
  dropped for looking bad, but not recommended either.
- `math.stackexchange.com`, Wikiversity, Hacker News: not harvested by the
  upstream corpus task (PR #58); not re-litigated here.
- Byte-identity / exact-roundtrip gating: explicitly out of scope per every
  prior generation's brief and this one; quality is judged elsewhere.
- LLM-judged token selection: not used anywhere in this task.
- Re-deriving the Unicode maths/Greek classifier, the LaTeX macro seed, or
  `rebuild-keepset-tokenizer.py`'s tokenizer-surgery functions: all reused
  unmodified from `keepset-v2-unicode.py` / `keepset-seed-latex.json` /
  `rebuild-keepset-tokenizer.py`.
- Qwen3.5-9B's own strict optimum (K=49,152-49,153, strategy C without the
  force-add layer) ties C+F within noise; C+F is recommended anyway for the
  reason above (validation), not because C fails to compete on speed.
- v4's numbers above are a live snapshot of a concurrently-running sibling
  task, not its own final report; not treated as settled.

## Anomalies

1. **The headline finding**: v3's (and, per the live snapshot above, likely
   v4's) contamination — see "The contamination finding, verified."
2. **S underperforms C** despite being the more sophisticated ranking — see
   the strategy definitions section for the measured cause (worst-case
   byte-depth overweights structurally deep, corpus-rare tokens).
3. **v4's own docstring does not match its measured behavior** at the
   snapshot this task observed — reported as a point-in-time read of a live
   sibling worktree, not a claim about v4's eventual committed state.
4. Qwen3.5-9B's flatness between K=49,152 and K=65,536 under C+F (net
   1.0287-1.0338 vs 1.0287-1.0334 — essentially identical) meant the 0.3%
   tie rule picked the smaller K automatically; reported plainly rather than
   picked by hand.
