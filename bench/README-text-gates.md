# Text gates (issue #27)

This directory contains two offline measurements run with the local Qwen3 tokenizer:

- Gate B: `bench/copy-fraction.py`
- Gate C: `bench/vocab-coverage.py`

## Setup

Run from the repository root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python tokenizers numpy
```

## Build pdf-paste tier

```bash
.venv/bin/python bench/pdf_paste.py --demo
.venv/bin/python bench/build-pdf-pastes.py
```

Pipeline (reference passages with `\(...\)` and `\[...\]` math):
1. Render all passages in one KaTeX HTML document (one page per passage) with sentinel characters around each math block.
2. Print once with headless Chrome to PDF.
3. Extract with `pdftotext` (`-layout` optional), locate sentinel pairs, map extracted spans back to source LaTeX, then remove sentinels.
4. Keep `pdftotext` artifacts as-is (line breaks, stacked fractions, Unicode operators); trim only trailing page whitespace.

KaTeX assets are vendored under `bench/vendor/katex/` at version `0.16.11`.

`copy-fraction.py` now scores synth examples against an explicit `reference` field when present (for `pdf`, this is the source passage rendered to PDF), instead of rebuilding expected output from extracted text. `pdf_paste.py` now checks span completeness against per-formula debris text from the same render pipeline; incomplete spans are greedily extended up to 6 lines, otherwise marked `partial` with `ok: false`. Gate B also reports a normalized-input second pass (`pdf-paste (norm)` and `pdf (norm)` rows) that removes zero-width spaces and wraps before tokenizing input.

Profiles:
- `clean`: current one-page viewer copy behavior.
- `dirty`: per passage, enable a seeded random subset of 2-4 layout knobs (`two_column`, `narrow_justified`, `running_head`, `footnote`, `citations`, `ligatures`, `layout_mode`).
- `two_column` is the main garbling source: pdftotext default mode often interleaves both columns line-by-line, so prose can appear inside a math sentinel pair.
- Sentinel mismatches and partial spans are kept with `ok: false` and `reason`; interleaved spans are kept with `interleaved: true`.

Two-column interleaving example (dirty profile):
- Input excerpt: `A block is pushed across a                        speed comes from v = ds  dt , the ... frictionless table.¹ ...`
- Reference excerpt: `A block is pushed across a frictionless table. ... The speed comes from \( v = \frac{\mathrm{d}s}{\mathrm{d}t} \), ...`

## Run gates

```bash
.venv/bin/python bench/copy-fraction.py
.venv/bin/python bench/vocab-coverage.py
```

Outputs written by the scripts:

- `bench/results-copy-fraction.json`
- `bench/results-vocab-coverage.json`

## Gate B summary table (from run)

Command sequence used:

```bash
.venv/bin/python bench/build-pdf-pastes.py
.venv/bin/python bench/synth-spans.py --seed 0 --n 500 --check
.venv/bin/python bench/copy-fraction.py
```

| group | n | copy@5 mean | copy@5 med | speedup@5 mean | speedup@5 med | copy@10 mean | copy@10 med | speedup@10 mean | speedup@10 med | in_input_token_share mean | in_input_token_share med |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| easy | 15 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.163 | 0.154 |
| easy (norm) | 15 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.163 | 0.154 |
| medium | 12 | 0.067 | 0.018 | 1.088 | 1.019 | 0.075 | 0.018 | 1.105 | 1.019 | 0.121 | 0.071 |
| medium (norm) | 12 | 0.067 | 0.018 | 1.088 | 1.019 | 0.075 | 0.018 | 1.105 | 1.019 | 0.121 | 0.071 |
| hard | 11 | 0.137 | 0.125 | 1.167 | 1.135 | 0.140 | 0.125 | 1.173 | 1.135 | 0.134 | 0.109 |
| hard (norm) | 11 | 0.137 | 0.125 | 1.167 | 1.135 | 0.140 | 0.125 | 1.173 | 1.135 | 0.134 | 0.109 |
| multiline | 5 | 0.395 | 0.409 | 1.652 | 1.692 | 0.427 | 0.445 | 1.751 | 1.803 | 0.563 | 0.560 |
| multiline (norm) | 5 | 0.395 | 0.409 | 1.652 | 1.692 | 0.427 | 0.445 | 1.751 | 1.803 | 0.563 | 0.560 |
| pdf-paste | 16 | 0.289 | 0.308 | 1.420 | 1.435 | 0.308 | 0.332 | 1.464 | 1.485 | 0.561 | 0.573 |
| pdf-paste (norm) | 16 | 0.315 | 0.343 | 1.479 | 1.511 | 0.338 | 0.369 | 1.541 | 1.573 | 0.577 | 0.576 |
| pdf-paste/clean | 9 | 0.302 | 0.309 | 1.445 | 1.444 | 0.324 | 0.333 | 1.495 | 1.500 | 0.548 | 0.556 |
| pdf-paste/clean (norm) | 9 | 0.320 | 0.353 | 1.488 | 1.526 | 0.342 | 0.378 | 1.549 | 1.587 | 0.558 | 0.537 |
| pdf-paste/dirty | 7 | 0.272 | 0.243 | 1.388 | 1.321 | 0.287 | 0.253 | 1.424 | 1.338 | 0.578 | 0.590 |
| pdf-paste/dirty (norm) | 7 | 0.308 | 0.333 | 1.467 | 1.500 | 0.334 | 0.359 | 1.530 | 1.560 | 0.602 | 0.614 |
| overall | 59 | 0.151 | 0.125 | 1.218 | 1.135 | 0.161 | 0.125 | 1.243 | 1.135 | 0.291 | 0.273 |
| overall (norm) | 59 | 0.158 | 0.125 | 1.234 | 1.135 | 0.169 | 0.125 | 1.264 | 1.135 | 0.295 | 0.273 |
| multiline_only | 5 | 0.395 | 0.409 | 1.652 | 1.692 | 0.427 | 0.445 | 1.751 | 1.803 | 0.563 | 0.560 |
| multiline_only (norm) | 5 | 0.395 | 0.409 | 1.652 | 1.692 | 0.427 | 0.445 | 1.751 | 1.803 | 0.563 | 0.560 |

Synthetic spans now mix `pdf`, `unicode`, and `latex` examples.

| slice | group | n | copy@5 mean | copy@5 med | speedup@5 mean | speedup@5 med | copy@10 mean | copy@10 med | speedup@10 mean | speedup@10 med | in_input_token_share mean | in_input_token_share med |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | overall | 418 | 0.453 | 0.425 | 2.082 | 1.694 | 0.487 | 0.452 | 2.407 | 1.767 | 0.719 | 0.701 |
| overall | overall (norm) | 418 | 0.461 | 0.440 | 2.118 | 1.750 | 0.495 | 0.473 | 2.472 | 1.845 | 0.722 | 0.707 |
| dominant_kind | pdf | 143 | 0.273 | 0.250 | 1.404 | 1.333 | 0.286 | 0.263 | 1.441 | 1.333 | 0.537 | 0.533 |
| dominant_kind | pdf (norm) | 143 | 0.285 | 0.269 | 1.438 | 1.351 | 0.300 | 0.279 | 1.486 | 1.357 | 0.541 | 0.533 |
| dominant_kind | unicode | 98 | 0.317 | 0.310 | 1.485 | 1.429 | 0.336 | 0.322 | 1.541 | 1.453 | 0.574 | 0.569 |
| dominant_kind | unicode (norm) | 98 | 0.317 | 0.310 | 1.485 | 1.429 | 0.336 | 0.322 | 1.541 | 1.453 | 0.574 | 0.569 |
| dominant_kind | latex | 96 | 0.646 | 0.662 | 2.874 | 2.909 | 0.699 | 0.714 | 3.473 | 3.483 | 0.922 | 0.938 |
| dominant_kind | latex (norm) | 96 | 0.646 | 0.662 | 2.874 | 2.909 | 0.699 | 0.714 | 3.473 | 3.483 | 0.922 | 0.938 |
| dominant_kind | none | 81 | 0.709 | 0.727 | 3.063 | 3.000 | 0.772 | 0.800 | 3.898 | 3.750 | 0.978 | 1.000 |
| dominant_kind | none (norm) | 81 | 0.725 | 0.733 | 3.189 | 3.000 | 0.792 | 0.800 | 4.152 | 3.750 | 0.984 | 1.000 |
| profile | pdf/clean | 99 | 0.380 | 0.312 | 1.775 | 1.455 | 0.405 | 0.314 | 1.958 | 1.455 | 0.637 | 0.583 |
| profile | pdf/clean (norm) | 99 | 0.391 | 0.324 | 1.846 | 1.455 | 0.419 | 0.338 | 2.102 | 1.488 | 0.639 | 0.583 |
| profile | pdf/dirty | 89 | 0.356 | 0.286 | 1.703 | 1.400 | 0.378 | 0.297 | 1.867 | 1.422 | 0.639 | 0.617 |
| profile | pdf/dirty (norm) | 89 | 0.377 | 0.313 | 1.792 | 1.426 | 0.403 | 0.333 | 2.010 | 1.455 | 0.649 | 0.632 |
| span_char_share | <25% | 211 | 0.505 | 0.486 | 2.200 | 1.850 | 0.544 | 0.510 | 2.584 | 2.000 | 0.774 | 0.784 |
| span_char_share | <25% (norm) | 211 | 0.519 | 0.508 | 2.271 | 1.960 | 0.561 | 0.549 | 2.711 | 2.130 | 0.780 | 0.806 |
| span_char_share | 25-50% | 151 | 0.375 | 0.288 | 1.855 | 1.403 | 0.400 | 0.300 | 2.079 | 1.426 | 0.631 | 0.569 |
| span_char_share | 25-50% (norm) | 151 | 0.376 | 0.293 | 1.856 | 1.411 | 0.401 | 0.302 | 2.081 | 1.429 | 0.631 | 0.567 |
| span_char_share | >50% | 56 | 0.469 | 0.612 | 2.251 | 2.530 | 0.504 | 0.672 | 2.624 | 2.994 | 0.748 | 0.906 |
| span_char_share | >50% (norm) | 56 | 0.467 | 0.612 | 2.249 | 2.530 | 0.502 | 0.672 | 2.623 | 2.994 | 0.747 | 0.906 |

| knob | n_on | copy@10 on | speedup@10 on | n_off | copy@10 off | speedup@10 off |
|---|---:|---:|---:|---:|---:|---:|
| citations | 39 | 0.388 | 1.909 | 50 | 0.370 | 1.834 |
| footnote | 37 | 0.418 | 1.915 | 52 | 0.348 | 1.832 |
| layout_mode | 45 | 0.350 | 1.712 | 44 | 0.406 | 2.025 |
| ligatures | 33 | 0.360 | 1.780 | 56 | 0.388 | 1.918 |
| narrow_justified | 43 | 0.424 | 2.065 | 46 | 0.335 | 1.681 |
| running_head | 35 | 0.379 | 1.969 | 54 | 0.377 | 1.800 |
| two_column | 32 | 0.304 | 1.487 | 57 | 0.419 | 2.080 |

## Gate C summary table (from run)

For padding, we use Qwen byte-level BPE merge rank as a frequency proxy by taking the first `N` regular token ids after the 256 byte tokens.

Command used:

```bash
.venv/bin/python bench/vocab-coverage.py
```

| pad_n | oos_micro_overall | oos_micro_easy | oos_micro_medium | oos_micro_hard | oos_micro_multiline | oos_micro_pdf_paste | fert_pruned_micro_overall | ref_bit_exact_frac |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0889 | 0.0318 | 0.0753 | 0.1781 | 0.1330 | 0.0533 | 0.4683 | 0.6774 |
| 8192 | 0.0437 | 0.0136 | 0.0356 | 0.0982 | 0.0451 | 0.0315 | 0.4242 | 0.6774 |
| 16384 | 0.0313 | 0.0136 | 0.0251 | 0.0662 | 0.0322 | 0.0236 | 0.4082 | 0.6774 |
| 32768 | 0.0178 | 0.0000 | 0.0146 | 0.0297 | 0.0215 | 0.0166 | 0.3852 | 0.7097 |
| 65536 | 0.0073 | 0.0000 | 0.0084 | 0.0137 | 0.0107 | 0.0044 | 0.3655 | 0.8065 |
