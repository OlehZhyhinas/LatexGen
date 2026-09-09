# Layer-3 keep-set corpus: frequency evidence for BPE-merge-rank replacement

This produces the frequency evidence to rank vocabulary-keepset **Layer 3**
(the ~41,577 slots allocated among ~107,670 whole-ASCII-word and
code/punctuation candidates, currently ordered by BPE merge rank — see issue
#27 and `bench/README-text-gates.md`, owned by a sibling task) by frequency
in actual maths/physics/technical prose instead. It does **not** build a
keep-set — that is Layer 3's own consumption task. It produces:

- `harvest_arxiv.py`, `harvest_wikipedia_science.py`,
  `harvest_stackexchange.py`, `harvest_openstax.py`, `harvest_wikibooks.py` —
  resumable, cached, rate-limited harvesters.
- `count_tokens.py` — tokenizes everything harvested with both target
  tokenizers, accumulating per-token-id counts with per-source subtotals.
- `freq-qwen35.json`, `freq-minicpm5-2b.json` — the ranked frequency tables
  (id, count, rank, per-source subtotals in the sibling `*-by-source.json`).
- `analyze.py` and `analysis-qwen35.json` / `analysis-minicpm5-2b.json` — the
  five analyses below.
- `provenance.md` — exact commands, dates, revisions, counts.

Full harvest details (exact commands, dates, category rules, per-site
counts, licences) are in `provenance.md`; this file focuses on the numbers
and what they mean.

## Sources and totals

| Source | Licence | Docs | Qwen3.5 tokens | MiniCPM5-2B tokens |
|---|---|---|---|---|
| Stack Exchange: `tex.stackexchange.com` | CC BY-SA 4.0 | 589,788 posts | 270,638,968 | 244,483,984 |
| Stack Exchange: `physics.stackexchange.com` | CC BY-SA 4.0 | 575,681 posts | 157,505,314 | 145,407,675 |
| Stack Exchange: `stats.stackexchange.com` | CC BY-SA 4.0 | 422,744 posts | 138,026,155 | 125,166,902 |
| arXiv abstracts | arXiv (metadata bulk use permitted) | 214,500 | 50,590,983 | 47,551,429 |
| Stack Exchange: `cs.stackexchange.com` | CC BY-SA 4.0 | 104,644 posts | 27,108,540 | 25,689,786 |
| OpenStax textbooks (17 books) | CC BY 4.0 | 4,347 modules | 13,251,368 | 12,507,210 |
| Wikipedia (science categories) | CC BY-SA 4.0 | 4,861 articles | 10,134,550 | 9,396,331 |
| Wikibooks (science search) | CC BY-SA 4.0 | 2,551 articles | 6,092,960 | 5,777,719 |
| **Total** | | **1,919,116 docs** | **673,348,838** | **615,981,036** |

**Target was ≥5×10⁷ tokens; actual is ~6.7×10⁸ (13.5x over) on Qwen3.5, ~6.2×10⁸
(12.3x) on MiniCPM5-2B.** Given that surplus, `math.stackexchange.com`
(3.63 GB compressed — see "What was not done" in the RESULT file),
Wikiversity, and Hacker News were not harvested; see `provenance.md` for
why each was skippable without shortfall.

Selection rules for what counts as "science" are in the harvester scripts,
not in this document's prose — see each script's module docstring
(`DENY_SUBSTRINGS`/`ALLOW_OVERRIDE_SUBSTRINGS` in
`harvest_wikipedia_science.py` for Wikipedia's biography-of-scientists
carve-out; arXiv and OpenStax need no filtering since they're science-only by
construction; Stack Exchange sites are chosen by subject).

## "Merge rank" == token id

Both tokenizers are byte-level BPE: the first 256 ids are the fixed
byte-alphabet (every raw byte value is individually encodable, so text is
never unencodable), and every id from 256 up is a learned merge, assigned in
the order the merge was learned during training — i.e. **lower id = earlier/
more-frequent merge in Qwen's/MiniCPM's general training mix = higher merge
rank.** Confirmed empirically against
`bench/keepsets/qwen35-65536/keep-idx.json` (untouched by this task): its
65,536 kept ids are exactly `{0..64910}` plus 625 high-numbered
added/special-token ids. "Top-K by merge rank" in the analyses below is
therefore simply the id range `{0, ..., K-1}`.

## Analysis 1 — overlap with merge rank

| Tokenizer | K | Overlap (corpus-top-K ∩ merge-top-K) | Overlap fraction | Corpus-only | Merge-only |
|---|---|---|---|---|---|
| Qwen3.5 | 32,768 | 20,118 | 61.4% | 12,650 | 12,650 |
| Qwen3.5 | 65,536 | 47,241 | 72.1% | 18,295 | 18,295 |
| MiniCPM5-2B | 32,768 | 18,350 | 56.0% | 14,418 | 14,418 |
| MiniCPM5-2B | 65,536 | 42,342 | 64.6% | 23,194 | 23,194 |

**Finding: merge rank was not "already fine."** 27–44% of keep-set slots
would change hands if ranked by this corpus instead of by merge rank, at
both K values, on both tokenizers. This is a real, large disagreement, not
noise — Layer 3 has substantial room to matter. (Reported plainly per the
brief either way; it did not come out as "no gain".)

## Analysis 2 — the contested band (Qwen3.5; MiniCPM5-2B tells the same story)

At K=65,536, 18,295 ids are **promoted** (corpus keeps, merge rank would
drop) and 18,295 are **demoted** (merge rank keeps, corpus would drop) —
symmetric by construction, since both sets have exactly 65,536 members.

**Top 15 of the 50 most significant promoted tokens** (by corpus count —
full 50 in `analysis-qwen35.json`, `analysis_2_contested_band.promoted_top50_by_corpus_count`):

| token | corpus count | merge-rank id |
|---|---|---|
| `tik` | 388,117 | 155,023 |
| `mathrm` | 162,535 | 88,396 |
| `^{-` | 158,528 | 84,203 |
| `ĠLaTeX` | 121,040 | 94,535 |
| `langle` | 92,635 | 68,402 |
| `%\` | 83,893 | 67,257 |
| `subsection` | 81,077 | 66,543 |
| `}}}` | 77,661 | 72,964 |
| `}\\` | 74,141 | 82,585 |
| `rangle` | 73,959 | 68,765 |
| `Declare` | 70,629 | 75,766 |
| `Ġgravitational` | 64,745 | 67,550 |
| `Ġphoton` | 64,350 | 65,620 |
| `Ġ\@` | 59,822 | 71,560 |
| `{n` | 54,418 | 88,216 |

This is exactly the population the brief predicted: LaTeX source markup
(`mathrm`, `langle`/`rangle`, `subsection`, `Declare`[Math...], `\@`, `tik`/
`ĠTik`[Z]) and physics vocabulary (`gravitational`, `photon`, `scattering`,
`curvature`, `Bayesian`, `residuals`, `kinetic`) that merge rank places
beyond 65,536 because it is rare in Qwen's general multilingual mix but
common in LaTeX-heavy science prose.

**The demoted side splits into two very different populations**, which is
itself a finding:

1. **60 of the 18,295 demoted ids (Qwen3.5) are single-byte accented-Latin
   characters from the fixed 256-entry byte alphabet** (`À`, `Á`, `Ø`, `å`,
   `æ`, `ç`, ... — ids 124–255ish). These are always in the vocab by
   tokenizer construction (every raw byte must be individually encodable),
   not a Layer-3 ranking decision — a keep-set builder almost certainly
   cannot drop them regardless of corpus rank (dropping one would make its
   raw byte value unencodable without fallback). Reported here for honesty,
   but this ~0.3% of the demoted set is not real headroom for Layer 3.
2. **The other 18,235 (Qwen3.5) / 23,117 (MiniCPM5-2B) demoted ids (≥256,
   i.e. actual learned merges)** are dominated by general-purpose
   programming-boilerplate and general-web-text tokens with **zero**
   occurrences anywhere in 673M tokens of science text — e.g. (top of the
   list, all count=0): `ĉreturn`, `ĉpublic`, `ĉprivate`, `ĉint`, `ĉstruct`,
   `ĉprintf`, `GridView`, `ViewById`, `.getElement`, `.TabIndex` (C#/Java/
   Android/web-framework boilerplate — irrelevant to LaTeX/math/physics
   conversion), plus isolated general-web tokens like `Ġporno`, `Ġweap`[on].
   Full top 50 in `analysis_2_contested_band.demoted_ge256_top50_by_lowest_corpus_count`.
   **This is the real headroom**: merge rank is spending thousands of the
   64,911 "protected" low-rank slots on general-programming/general-web
   vocabulary with no relevance to this product's domain, while excluding
   physics/LaTeX vocabulary that is actually common in it.

## Analysis 3 — the tokens we know we need (Qwen3.5)

| token | corpus rank | in top 32,768? | in top 65,536? |
|---|---|---|---|
| `Ġoscillator` | 4,636 | **yes** | yes |
| `Ġdamping` | 8,715 | **yes** | yes |
| `Ġconverge` | 5,190 | **yes** | yes |
| `Ġflattened` | 24,682 | **yes** | yes |
| `Ġreorder` | 29,519 | **yes** | yes |
| `Â¨` | 41,996 | no | **yes** |
| `ĠÏĨ` (" φ" as one token) | 10,418 | **yes** | yes |

Individual symbols (encoded standalone, no leading space unless noted):

| symbol | pieces (id: token) | in top 32,768? | in top 65,536? |
|---|---|---|---|
| `φ` | 82099:`ÏĨ` | **yes** | yes |
| `←` | 69416:`âĨĲ` | no | **yes** |
| `↔` | 25098:`âĨ` + 242:`Ķ` | first piece: no | yes |
| `↓` | 76322:`âĨĵ` | no | **yes** |
| `×` | 17044:`ÃĹ` | **yes** | yes |

Proper nouns (encoded with a leading space, as they'd appear mid-sentence),
every subword piece:

| name | pieces | all in top 65,536? | all in top 32,768? |
|---|---|---|---|
| `Nikolskii` | `ĠNik`(rank 32,895) `ols`(3,072) `k`(217) `ii`(2,327) | yes | **no** (`ĠNik` misses) |
| `Randers` | `ĠRand`(24,710) `ers`(1,114) | yes | yes |
| `Lissajous` | `ĠL`(412) `iss`(4,482) `aj`(9,646) `ous`(2,400) | yes | yes |
| `Kullback` | `ĠK`(807) `ull`(3,741) `back`(1,533) | yes | yes |
| `Hausdorff` | `ĠHaus`(16,007) `dor`(15,556) `ff`(1,537) | yes | yes |
| `Chebyshev` | `ĠChe`(14,713) `b`(176) `ys`(2,564) `hev`(16,652) | yes | yes |
| `Sobolev` | `ĠSob`(16,163) `ole`(1,965) `v`(186) | yes | yes |

**Verdict on this analysis:** every single named token/symbol/proper-noun
piece is inside the corpus-rank top 65,536. At the tighter K=32,768, three
pieces fall out: `Ġflattened` and `Ġreorder` stay in, but `Â¨`, `←`, `↔`
(first piece), `↓`, and `ĠNik` (of `Nikolskii`) do not make top 32,768 on
corpus rank alone. **This does not license dropping the structural layers**:
`Ġreorder` is rare in science prose in general but appears in LatexGen's own
prompt text, so Layer 1's force-add of prompt vocabulary stays mandatory
regardless of this corpus's ranking, and the same logic applies to whichever
symbols/diacritics Layer 2 force-adds — corpus rank is evidence for Layer 3's
ASCII-word/punctuation slots, not a replacement for the structural layers
that guarantee specific known-needed pieces survive at K=32,768.

## Analysis 4 — coverage of held-out real input

Held-out set: the 120 `input` + 120 `reference` strings (240 texts,
26,337 Qwen3.5 token occurrences / 24,471 MiniCPM5-2B occurrences) of
`bench/arxiv-pastes.json` — genuinely held out (these are LaTeX-conversion
paste/reference pairs, not present anywhere in the corpus harvested here),
though it is worth noting the held-out set is drawn from arXiv papers and
this corpus's largest single component is Stack Exchange, not arXiv, so it
is not a purely in-domain-for-everything test.

| Tokenizer | K | OOS rate, corpus-rank keep-set | OOS rate, merge-rank keep-set | Reduction |
|---|---|---|---|---|
| Qwen3.5 | 32,768 | **1.53%** (404 tokens) | 8.58% (2,260 tokens) | 5.6x |
| Qwen3.5 | 65,536 | **0.23%** (61 tokens) | 2.60% (686 tokens) | 11.2x |
| MiniCPM5-2B | 32,768 | **2.18%** (533 tokens) | 9.08% (2,221 tokens) | 4.2x |
| MiniCPM5-2B | 65,536 | **0.28%** (68 tokens) | 3.16% (773 tokens) | 11.4x |

**Finding: corpus rank cuts the out-of-set rate on real held-out input by
4–11x at both K values, on both tokenizers.** This is the strongest single
piece of evidence in this report that corpus ranking beats merge rank for
Layer 3's purpose. It is also direct support for the brief's K=32,768
hypothesis: at K=32,768, corpus rank's 1.53%/2.18% OOS rate is well below
merge rank's *own* 65,536 OOS rate (2.60%/3.16%) — i.e. a corpus-ranked
32,768-slot keep-set covers this held-out sample about as well as (Qwen3.5)
or better than (MiniCPM5-2B) a merge-rank-ordered 65,536-slot one does today.

## Analysis 5 — Zipf sanity check

| Tokenizer | Rank | Token | Count (in 673M/616M-token corpus) |
|---|---|---|---|
| Qwen3.5 | 32,768 | `ĠAndre` | 381 |
| Qwen3.5 | 65,536 | `ĠBethlehem` | 53 |
| MiniCPM5-2B | 32,768 | `Focus` | 492 |
| MiniCPM5-2B | 65,536 | `Lor` | 60 |

The 65,536th-ranked token's count (53 / 60) is low but not "low single
digits" — this corpus (673M / 616M tokens, ~26,000x the size of the repo's
own in-domain text) gives a real, if thin, signal all the way out to rank
65,536. It is thinner than the signal at 32,768 (381 / 492, a comfortable
count), which is itself evidence for the brief's framing: **ranking is more
trustworthy at K=32,768 than at K=65,536**, and the tail of Layer 3's
65,536-slot keep-set — regardless of which ranking feeds it — is
necessarily ranking on small counts and is not going to be highly reliable
past roughly rank 40,000–50,000 even with a corpus this size.

## Verdict

- **Corpus rank beats merge rank for Layer 3.** Overlap is only 61–72%
  (Qwen3.5) / 56–65% (MiniCPM5-2B) at the two K values — a large fraction of
  slots would change — and on genuinely held-out real input, corpus rank
  cuts the out-of-set rate by 4–11x at matched K. The contested-band
  breakdown (Analysis 2) explains *why*: merge rank is protecting thousands
  of general-programming/general-web tokens with zero science-corpus usage
  ahead of physics/LaTeX vocabulary that is common in this domain.
- **This supports K=32,768.** Analysis 4's numbers put a corpus-ranked
  32,768-slot keep-set's held-out coverage in the same range as (or better
  than) merge rank's own 65,536-slot coverage today — consistent with the
  brief's claim that better Layer 3 ranking is what makes K=32,768 viable at
  flat quality.
- **It does not replace the structural layers.** Analysis 3 shows specific
  known-needed pieces (`Â¨`, arrow symbols, part of `Nikolskii`) fall outside
  even the corpus-rank top 32,768; those must keep coming from Layers 1/2's
  force-adds, not from ranking.
- **The corpus is adequate for K≤65,536 but its tail is thin.** Analysis 5:
  real but small counts (53–60) at rank 65,536 mean the deepest part of
  Layer 3 is ranking on weak signal even with a 673M-token corpus; that
  weakness is inherent to Zipf's law at this vocabulary depth, not a defect
  in this harvest.
