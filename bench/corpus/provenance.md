# Corpus provenance — Layer 3 keep-set frequency evidence

Harvested 2026-09-09 for the Layer-3 (whole-ASCII-word / code / punctuation)
keep-set ranking task described in `bench/corpus/README-corpus.md`. All
harvesting was done from this repo's `keepset-corpus` worktree on a machine
with ~5–20 GB of free disk at any given time; every harvester is resumable
and every fetch is cached under `bench/corpus/cache/` (gitignored, not
committed — see "Disk" below for the reason and the final on-disk size).

## Tooling

```
git -C /Users/oleh/personal/latexgen worktree add ../latexgen-keepset-corpus -b keepset-corpus origin/text-tier-gates
cd /Users/oleh/personal/latexgen-keepset-corpus
git config user.email 10584307+OlehZhyhinas@users.noreply.github.com
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python tokenizers numpy requests
uv pip install --python .venv/bin/python py7zr   # for Stack Exchange 7z dumps
```

Scripts (this commit): `bench/corpus/harvest_arxiv.py`,
`bench/corpus/harvest_wikipedia_science.py`,
`bench/corpus/harvest_stackexchange.py`, `bench/corpus/harvest_openstax.py`,
`bench/corpus/harvest_wikibooks.py`, `bench/corpus/count_tokens.py`,
`bench/corpus/analyze.py`.

## Tokenizers (pinned, hash-verified against `bench/tokenizer-provenance.json`)

| label | HF repo | revision | vocab size (ids 0..N-1) |
|---|---|---|---|
| `qwen35` | `mlc-ai/Qwen3.5-4B-q4f16_1-MLC` | `44b42469f9e192814bfd90440e3b377d89ba7a13` | 248,070 |
| `minicpm5-2b` | `ozhyhinas/MiniCPM5-2B-q4f16_1-MLC` | `2318f37d9c39277ff01dc64086491028c95d4db4` | 130,560 |

`tokenizer.json` for each was downloaded via `huggingface_hub.hf_hub_download`
pinned to the revision above and its SHA-256 checked against the value
already recorded in `bench/tokenizer-provenance.json` from the earlier
vocab-keepset task — both matched exactly (script output, verbatim):

```
qwen35 mlc-ai/Qwen3.5-4B-q4f16_1-MLC 44b42469f9e192814bfd90440e3b377d89ba7a13 sha256= 5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42 MATCH
minicpm5-2b ozhyhinas/MiniCPM5-2B-q4f16_1-MLC 2318f37d9c39277ff01dc64086491028c95d4db4 sha256= 3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81 MATCH
```

Note: `tokenizer-provenance.json` records the model's *padded* vocab size
(248,320 for Qwen3.5); the tokenizer itself has 248,070 real ids (0..248,069)
-- confirmed by `Tokenizer.get_vocab_size(with_added_tokens=True)` and by the
highest id present in `bench/keepsets/qwen35-65536/keep-idx.json`. This
task's rankings and "merge rank" comparisons use the real 248,070.

**"Merge rank" == token id.** Verified against
`bench/keepsets/qwen35-65536/keep-idx.json` (built by
`bench/rebuild-keepset-tokenizer.py`, not touched by this task): its 65,536
kept ids are exactly `{0..64910}` plus 625 high-numbered added/special-token
ids -- i.e. BPE token ids are assigned in merge order, low id = frequent in
Qwen's training mix = "merge rank". This task therefore treats "top-K by
merge rank" as simply `{0, ..., K-1}`.

## Source A — arXiv (OAI-PMH bulk metadata)

- Endpoint: `http://export.arxiv.org/oai2`, `verb=ListRecords`,
  `metadataPrefix=arXiv`.
- Command:
  `python bench/corpus/harvest_arxiv.py --out bench/corpus/cache/arxiv --from-date 2023-01-01 --until-date 2026-09-09 --max-records 250000 --contact oleh+latexgen-corpus@example.com`
- Rate limiting: 4 s between requests, honors HTTP 503 + `Retry-After`.
  `User-Agent` includes a contact address, per arXiv's OAI-PMH policy.
- Selection rule: no topical filtering — arXiv is science-only by
  construction; every category in the requested datestamp window is kept.
- **Result: 214,500 records** harvested before the run was stopped (see
  "What I did not do" in the RESULT file — a disk-cleanup command deleted the
  in-progress `raw/` cache out from under the running process; the parsed
  `records.jsonl` and the resumable `state.json` were unaffected, but the run
  was not resumed because 214,500 records already gave far more tokens than
  the target — see totals below).
  - OAI datestamp window requested: `from=2023-01-01` to `until=2026-09-09`
    (datestamp = last metadata touch, not paper creation date). The harvest
    covered datestamps up to approximately **2024-01-10** before stopping
    (`state.json.resumption_token` shows the server-side cursor at
    `from=2024-01-10&skip=51`); it is fully resumable from that token.
  - Underlying paper *creation* dates in the harvested set span
    **1992-10-08 to 2024-01-08** (datestamp-window harvesting picks up any
    paper whose metadata was touched in the window, including old papers
    reclassified or corrected recently).
  - Category top-level breakdown (by `arXiv:categories`, top 10):
    cs 153,952; math 69,749; cond-mat 29,054; physics 28,619;
    astro-ph 27,052; eess 17,739; stat 14,299; quant-ph 11,799;
    hep-ph 8,183; hep-th 8,075 (a record can carry more than one category).

## Source B — science-restricted English Wikipedia

- API: `https://en.wikipedia.org/w/api.php`,
  `action=query&list=categorymembers` (BFS) then
  `action=query&prop=extracts&explaintext` (per-title; the API forces
  `exlimit=1` for full-article extracts, so batching only works for
  intro-only extracts, confirmed empirically — see script docstring).
- Command:
  `python bench/corpus/harvest_wikipedia_science.py --out bench/corpus/cache/wikipedia --contact oleh+latexgen-corpus@example.com --max-depth 3 --max-articles 30000 --workers 2 --delay 0.8`
- Root categories: Mathematics, Physics, Chemistry, Biology, Statistics,
  Computer science, Astronomy, Earth science, Engineering. BFS depth ≤ 3.
- Inclusion/exclusion rule (in code, `DENY_SUBSTRINGS` /
  `ALLOW_OVERRIDE_SUBSTRINGS` in `harvest_wikipedia_science.py`): subcategories
  matching general history/politics/sport/geography/entertainment/unrelated-
  biography keywords are excluded from traversal and collection, *unless*
  they also match a scientist/mathematician-biography keyword (e.g.
  "mathematici", "physicist", "nobel laureate"), which overrides the
  exclusion. This is how the harvest keeps scientist biographies (source of
  proper nouns like `Nikolskii`) while dropping generic biography/sport/
  politics categories that happen to be reachable from a science root.
- **Result:** category BFS visited 613 categories and collected 31,945
  candidate article titles before hitting `--max-articles 30000`; extraction
  was rate-limited hard by Wikipedia's anonymous API quota (frequent HTTP
  429s even after dropping to 2 workers / 0.8 s delay) and was stopped by
  hand after **4,861 articles** with nonempty extract text, once total token
  volume from all sources already exceeded the target several times over.
  Fully resumable (`fetched_titles.json` tracks completed titles).

## Source C — plain-language technical prose

### Stack Exchange dumps (CC BY-SA 4.0, official Internet Archive mirror)

- Source: `https://archive.org/download/stackexchange/<site>.7z`
  (`https://archive.org/details/stackexchange`).
- Command:
  `python bench/corpus/harvest_stackexchange.py --out bench/corpus/cache/stackexchange --sites tex.stackexchange.com cs.stackexchange.com stats.stackexchange.com physics.stackexchange.com --contact oleh+latexgen-corpus@example.com`
- Only `Posts.xml` is extracted from each `.7z` (selective extraction via
  `py7zr`); `Comments.xml`/`Votes.xml`/etc. are never downloaded. Question and
  answer `Body` HTML is stripped to plaintext with inline LaTeX left as
  literal text (Stack Exchange stores MathJax source verbatim in the HTML
  body — stripping tags does not touch `$...$`/`\(...\)`).
- Disk-budget rule (in code, `harvest_stackexchange.py` docstring): sites are
  processed one at a time; the raw `.7z` and the decompressed `Posts.xml` are
  deleted immediately after that site's plaintext is written, so peak extra
  disk use is bounded by one site at a time, not the sum of all sites.
- `math.stackexchange.com.7z` (3.63 GB compressed) is **excluded** — see
  "What I did not do".
- Result (no subsetting applied to any of the four sites processed —
  each was small enough in full):

  | site | questions | answers | chars |
  |---|---|---|---|
  | `tex.stackexchange.com` | 259,202 | 330,586 | 802,356,553 |
  | `physics.stackexchange.com` | 234,152 | 341,529 | 620,731,089 |
  | `stats.stackexchange.com` | 213,761 | 208,983 | 491,800,583 |
  | `cs.stackexchange.com` | 48,390 | 56,254 | 102,215,243 |

### OpenStax textbooks (CC BY 4.0), via GitHub CNXML source

- 17 `osbooks-*` repos under `github.com/openstax` (calculus, college
  algebra, prealgebra, algebra 1, contemporary mathematics, university
  physics, college physics, astronomy, chemistry, organic chemistry, biology,
  microbiology, anatomy & physiology, introductory statistics, statistics,
  introduction to Python programming, principles of data science — full list
  and per-book module/char counts in `harvest_openstax.py`'s `BOOKS` list and
  `bench/corpus/cache/openstax/stats.json`).
- Command:
  `python bench/corpus/harvest_openstax.py --out bench/corpus/cache/openstax --contact oleh+latexgen-corpus@example.com`
- Fetched via the GitHub REST tree API (list `*/index.cnxml` paths) plus
  `raw.githubusercontent.com` per file — never `git clone`, which would also
  pull each book's `media/` image directory (100s of MB to >1 GB per book,
  almost entirely image binaries irrelevant to a text corpus).
- Result: 4,347 modules, 17/17 books completed.

### Wikibooks (CC BY-SA, en.wikibooks.org)

- Same MediaWiki API family as Source B
  (`action=query&list=search&srnamespace=0` for title discovery, since
  Wikibooks has no usable subject-category tree — each book instead gets an
  auto-generated per-book `Category:Book:<title>` that is useless for topic
  traversal, confirmed empirically; `action=query&prop=extracts&explaintext`
  for text, same as Source B).
- Command:
  `python bench/corpus/harvest_wikibooks.py --out bench/corpus/cache/wikibooks --site en.wikibooks.org --contact oleh+latexgen-corpus@example.com --max-per-root 300 --workers 2 --delay 0.8`
- Result: 2,551 articles with nonempty extract text (2,581 titles fetched
  total, from the 9 science root search terms).

### Not used

- **Wikiversity**: `harvest_wikibooks.py` supports it (`--site
  en.wikiversity.org`) but it was not run — by the time OpenStax and
  Wikibooks finished, total token volume was already >13x the target and the
  remaining time was spent on tokenization/analysis instead. Documented here
  rather than silently skipped.
- **Hacker News**: the brief makes this optional, "if the above fall short of
  the token target" — they did not, so it was not used.
- **Medium/Substack**: explicitly out of scope per the brief (bulk collection
  disallowed by their terms).

## Disk

Peak `bench/corpus/cache/` size during the harvest was kept under control by
processing Stack Exchange sites sequentially with immediate cleanup (see
above) and by periodically deleting `bench/corpus/cache/arxiv/raw/` (arXiv's
raw OAI-PMH XML batches, redundant with the parsed `records.jsonl` — one such
cleanup, run while the arXiv harvester was still writing, is what stopped
that harvest early; see "What I did not do" in the RESULT file). Final size
of `bench/corpus/cache/` at the end of this task: **2.5 GB**. It is
gitignored (`bench/corpus/cache/` added to `.gitignore`) and not committed.

## Tokenization

- Command:
  ```
  python bench/corpus/count_tokens.py \
    --cache-dir bench/corpus/cache --out-dir bench/corpus \
    --tokenizer qwen35:bench/corpus/cache/tokenizers/qwen35-tokenizer.json \
    --tokenizer minicpm5-2b:bench/corpus/cache/tokenizers/minicpm5-2b-tokenizer.json
  ```
- 1,919,116 documents tokenized with both tokenizers (`add_special_tokens=False`).
- Qwen3.5: **673,348,838 total tokens**, 148,106 distinct ids seen (of 248,070).
- MiniCPM5-2B: **615,981,036 total tokens**, 100,886 distinct ids seen (of 130,560).
- Per-source token totals (Qwen3.5 tokenizer): see
  `bench/corpus/README-corpus.md` and `bench/corpus/freq-qwen35.json`'s
  `source_totals`.

## Analysis

- Command (one per tokenizer):
  ```
  python bench/corpus/analyze.py --freq bench/corpus/freq-qwen35.json \
    --tokenizer bench/corpus/cache/tokenizers/qwen35-tokenizer.json \
    --held-out bench/arxiv-pastes.json --out bench/corpus/analysis-qwen35.json
  python bench/corpus/analyze.py --freq bench/corpus/freq-minicpm5-2b.json \
    --tokenizer bench/corpus/cache/tokenizers/minicpm5-2b-tokenizer.json \
    --held-out bench/arxiv-pastes.json --out bench/corpus/analysis-minicpm5-2b.json
  ```
- Full numbers and interpretation: `bench/corpus/README-corpus.md`.

## Script commit

All of the above scripts were committed on branch `keepset-corpus`; see the
PR against `text-tier-gates` for the exact commit hash.
