# Text gates (issue #27)

This directory contains two offline measurements run with the local Qwen3 tokenizer:

- Gate B: `bench/copy-fraction.py`
- Gate C: `bench/vocab-coverage.py`

## Setup

Run from the repository root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python tokenizers numpy transformers
```

`transformers` is required by `bench/rebuild-keepset-tokenizer.py`'s per-directory
`len(AutoTokenizer)` assertion (see "Keep-set rebuild: extra_special_tokens defect
and fix" below); it is not needed for `copy-fraction.py` or `vocab-coverage.py`
themselves and pulls no GPU/torch dependency for tokenizer-only usage.

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

## Span pipeline vs direct on PDF pastes (2026-09-07)

`bench/pdf_pipeline.py` (ported to `public/pdf-pipeline.js`, parity 530/530) normalizes PDF debris, finds maths spans by rules, sends only the spans to the model and splices the LaTeX back into the untouched prose. `bench/run-pipeline-ollama.py` runs it against the app's direct prompts through Ollama; `bench/judge-rows.py` judges with `qwen-local`; `bench/summarize-pipeline.py` prints the table. Modes: `direct` (app prompts, raw paste), `direct-norm` (normalizer, then app prompts), `pipeline` (one request per span), `pipeline-batch` (all spans in one request), `pipeline-ctx-batch` (spans marked inside the full passage). 30 pdf-paste items, greedy, thinking off, M5 Pro. `qwen3-4b-orig` is the Qwen3 4B Modelfile with thinking disabled; the stock `qwen3:4b` tag leaks reasoning into the output under Ollama and was excluded.

Segmenter alone (`bench/eval-segmenter.py`, character level against gold spans): P 0.97 / R 0.98 overall; pdf-paste/clean 1.00/1.00, synth-pdf/dirty 0.96/0.92, synth-unicode 0.97/0.99.

| model | mode | acc all | s all | acc clean | s clean | acc dirty | s dirty | speedup vs direct |
|---|---|---|---|---|---|---|---|---|
| gemma4:e2b | direct | 30% | 3.48 | 47% | 3.28 | 13% | 3.65 | 1.00x |
| qwen3-4b-orig | direct | 37% | 1.97 | 47% | 1.63 | 27% | 2.60 | 1.00x |
| qwen3-4b-orig | direct-norm | 40% | 1.39 | 40% | 1.34 | 40% | 1.45 | 1.41x |
| qwen3-4b-orig | pipeline | 27% | 0.90 | 27% | 0.95 | 27% | 0.87 | 2.19x |
| qwen3-4b-orig | pipeline-batch | 27% | 0.74 | 27% | 0.71 | 27% | 0.78 | 2.66x |
| qwen3-4b-orig | pipeline-ctx-batch | 27% | 0.82 | 27% | 0.74 | 27% | 0.90 | 2.39x |
| qwen3.5:0.8b | direct | 13% | 1.36 | 13% | 1.20 | 13% | 1.43 | 1.00x |
| qwen3:0.6b | direct | 0% | 0.91 | 0% | 0.99 | 0% | 0.76 | 1.00x |
| qwen3:0.6b | direct-norm | 0% | 0.35 | 0% | 0.28 | 0% | 0.37 | 2.57x |
| qwen3:0.6b | pipeline | 7% | 0.44 | 7% | 0.44 | 7% | 0.44 | 2.07x |
| qwen3:0.6b | pipeline-batch | 0% | 0.39 | 0% | 0.37 | 0% | 0.43 | 2.37x |
| qwen3:0.6b | pipeline-ctx-batch | 0% | 0.79 | 0% | 0.67 | 0% | 0.80 | 1.15x |
| qwen3:1.7b | direct | 0% | 1.42 | 0% | 2.34 | 0% | 1.31 | 1.00x |
| qwen3:1.7b | direct-norm | 3% | 1.02 | 0% | 0.97 | 7% | 1.06 | 1.38x |
| qwen3:1.7b | pipeline | 0% | 0.57 | 0% | 0.57 | 0% | 0.57 | 2.50x |
| qwen3:1.7b | pipeline-batch | 7% | 0.64 | 7% | 0.71 | 7% | 0.59 | 2.21x |
| qwen3:1.7b | pipeline-ctx-batch | 0% | 0.56 | 0% | 0.56 | 0% | 0.56 | 2.54x |

Reading: the pipeline is 2.2–2.7x faster than direct at every size because the model generates only the maths (median 42 vs 86 output tokens) and the prose is returned verbatim, but it is less accurate than direct at 4B (27% vs 37%) and does not make the 0.6B/1.7B usable (7%). The normalizer in front of the unchanged direct prompt is the one change that improves the current path: 37% → 40% overall, 27% → 40% on dirty layouts, 2.0 → 1.4 s. Remaining pipeline failures are formulas that pdftotext scattered beyond what a contiguous span can express (limits, stacked fractions, matrices) and subscripts/exponents the extraction dropped; giving the model the surrounding sentence (`pipeline-ctx-batch`) did not change accuracy. Judge verdicts include a few notation nitpicks (`\|` vs `|`, `k_{B}` vs `k_B`) that count against all approaches equally.

## Gate C corrected: real Qwen3.5 / MiniCPM5-2B tokenizers (2026-09-09)

The Gate C table above was measured against the local Qwen3-0.6B tokenizer as a
family proxy (`bench/vocab-coverage.py:19-21`'s glob). The target models for
Avenue C are **Qwen3.5-0.8B/4B/9B** (vocab 248,320, not 151,936) and
**MiniCPM5-2B** (vocab 130,560); neither tokenizer is Qwen3-0.6B's, and the
old numbers do not transfer. This section replaces them for those two real
tokenizers, widens the corpus, adds two seed lists the repo lacked, adds
harness-prompt coverage, and — new — actually rebuilds each tokenizer in a
compact id space and verifies the rebuild rather than only simulating pruning
on the unpruned tokenizer. Scripts: `bench/vocab-coverage.py` (extended mode,
`--tokenizer`/`--tokenizer-label`/`--corpus`/`--target-size`, legacy
invocation unchanged and still byte-for-byte reproducible),
`bench/build-keepset-seed-latex.py`, `bench/build-keepset-seed-mathsenglish.py`,
`bench/rebuild-keepset-tokenizer.py`, `bench/annotate-keepset-gate.py`.

**Tokenizer findings (Step 1).** Qwen3.5's `tokenizer.json` is byte-identical
(same SHA-256) across all three rungs (0.8B/4B/9B) and identical between the
MLC repo and upstream `Qwen/Qwen3.5-<size>` for every rung — one keep-set
serves all three. MiniCPM5-2B's MLC and upstream tokenizers are also
byte-identical. Both are byte-level BPE with the full 256-token byte alphabet
(not Unigram/SentencePiece), so the same merge-filtering rebuild route applies
to both; MiniCPM5-2B is not unprunable by this route. Two upstream bugs found
along the way, both pre-existing and not introduced by this task:
Qwen3.5's `tokenizer_config.json` declares 7 added tokens (ids 248070-248076,
TTS/audio placeholders) that do not exist in `tokenizer.json`'s own
`added_tokens` at all and are unreachable by `encode()`; and Qwen3.5's
`mlc-chat-config.json` `conv_template.stop_token_ids` (`[151643, 151645]`) are
the old 151,936-vocab Qwen3 ids for `<|endoftext|>`/`<|im_end|>`, which in the
real 248,320-vocab Qwen3.5 tokenizer resolve to an unrelated Korean and Thai
subword — i.e. stop generation was already silently wrong before pruning.
Full detail in `bench/tokenizer-provenance.json` and every
`bench/keepsets/*/provenance.md`.

**Seed lists (Step 2).** `bench/keepset-seed-latex.json`: 1,241 strings — 1,238
extracted programmatically from every `names:[...]` registration array and
standalone backslash-literal in vendored KaTeX 0.16.11's minified bundle, plus
6 hand-added document-level math delimiters (`$ $$ \( \) \[ \]`) that KaTeX's
own parser never sees. `bench/keepset-seed-mathsenglish.json`: 2,563 word
types by document frequency over 255 documents (`arxiv-pastes.json`
reference+meta.reference_prose, `bench-data-extended.json`,
`pdf-pastes.json`). This corpus is small; the list is a floor on
maths-English coverage, not a general English vocabulary, and will miss
common words absent from these specific passages — reported as a limitation,
not papered over. `bench/keepset-harness-prompts.json`: the 6 `FRESH_ITEMS` +
4 `QUALITY_CORPUS` prompt sets from `webnn-workbench@e5671e9`
`bench/webnn/qwen/webllm.js`, every token of every rendered prompt (both
`enable_thinking` variants, both models' chat templates) folded into the
keep-set unconditionally so a prompt token missing from the keep-set can
never masquerade as a pruning bug in the harness itself.

**Both directions, both models, natural size through 64k.** Reference-based
leave-one-out (seeds/harness ids fixed, corpus ids held out per item) and the
four target models' own correct-judged generated outputs:

MiniCPM5-2B tokenizer (natural keep-set 7,583):

| K_target | ref_oos_micro | ref_bit_exact_frac | Qwen3.5-0.8B out oos | MiniCPM5-2B out oos | Qwen3.5-4B out oos |
|---:|---:|---:|---:|---:|---:|
| 7583 (natural) | 0.0125 | 0.7003 | 0.0144 | 0.0161 | 0.0183 |
| 16384 | 0.0087 | 0.7459 | 0.0100 | 0.0077 | 0.0117 |
| 24576 | 0.0066 | 0.7883 | 0.0078 | 0.0059 | 0.0093 |
| 32768 | 0.0053 | 0.8111 | 0.0061 | 0.0053 | 0.0076 |
| 49152 | 0.0038 | 0.8469 | 0.0044 | 0.0039 | 0.0052 |
| 65536 | 0.0029 | 0.8925 | 0.0031 | 0.0028 | 0.0037 |

Qwen3.5 tokenizer, shared across all three rungs (natural keep-set 6,414):

| K_target | ref_oos_micro | ref_bit_exact_frac | Qwen3.5-0.8B out oos | MiniCPM5-2B out oos | Qwen3.5-4B out oos |
|---:|---:|---:|---:|---:|---:|
| 6414 (natural) | 0.0081 | 0.7427 | 0.0110 | 0.0128 | 0.0136 |
| 16384 | 0.0051 | 0.7980 | 0.0065 | 0.0054 | 0.0076 |
| 24576 | 0.0042 | 0.8339 | 0.0056 | 0.0041 | 0.0063 |
| 32768 | 0.0034 | 0.8599 | 0.0039 | 0.0032 | 0.0047 |
| 49152 | 0.0014 | 0.9251 | 0.0021 | 0.0020 | 0.0029 |
| 65536 | 0.0008 | 0.9511 | 0.0010 | 0.0014 | 0.0014 |

Real generated-output OOS tracks the reference-based number closely for both
tokenizers (no material divergence between "what the harness expects" and
"what the models actually produce") — output-side coverage alone looks safe
even at the natural size for either tokenizer.

**Input side is the one that actually decides this** (real held-out
`arxiv-pastes.json`/`pdf-pastes.json` `input` text, leave-one-out keep-set,
`n=138`, 120 arxiv-paste + 18 pdf-paste):

| K_target | MiniCPM5 frac_oos | MiniCPM5 fert_full→sim_pruned | Qwen3.5 frac_oos | Qwen3.5 fert_full→sim_pruned |
|---:|---:|---:|---:|---:|
| natural | 0.6667 | 0.2348 → 0.2640 | 0.6159 | 0.2467 → 0.2780 |
| 16384 | 0.6087 | 0.2348 → 0.2588 | 0.5000 | 0.2467 → 0.2672 |
| 24576 | 0.5435 | 0.2348 → 0.2557 | 0.4275 | 0.2467 → 0.2593 |
| 32768 | 0.5000 | 0.2348 → 0.2521 | 0.3768 | 0.2467 → 0.2558 |
| 49152 | 0.3768 | 0.2348 → 0.2488 | 0.2754 | 0.2467 → 0.2516 |
| 65536 | 0.2826 | 0.2348 → 0.2435 | 0.2246 | 0.2467 → 0.2501 |

Even at 64k, 22-28% of real pasted input still contains at least one
out-of-set token: prefill changes, and nothing downstream is guaranteed for
almost a quarter of real input regardless of size in this sweep. The
recurring misses are not common English or LaTeX macros but Unicode math/
physics symbols (↓ ↑ ≈ ≥ ⟩ ν κ, the `ffi` ligature) and compound domain
tokens (`-price`, `-measure`, ` Sb` as in antimony) — see
`top_miss_tokens` in `bench/results-vocab-coverage-{qwen35,minicpm5-2b}.json`
for the full ranked lists per K.

**Step 5/6 — the rebuild is the real gate, and simulated coverage
understates its cost.** `bench/rebuild-keepset-tokenizer.py` rebuilds each
tokenizer at every target size, remapping `vocab`/`merges`/`added_tokens`/
`post_processor` ids into the compact space (`new_id` = rank of the original
id in ascending sort of the keep-set), fixes the Qwen3.5 `stop_token_ids` bug
in the process, and verifies two things per K: exact roundtrip
(`remap(old_encode(text)) == new_encode(text)` and
`new_decode(new_encode(text)) == text`) for every corpus text whose tokens
were all in the keep-set, and — separately, and this is new relative to the
earlier gate — **real** fertility (re-encoding held-out arxiv/pdf input with
the actual rebuilt tokenizer via a leave-one-out mini-rebuild per item, not
simulating a miss as its byte length):

| K_target | MiniCPM5 merges dropped | MiniCPM5 exact | MiniCPM5 real fertility | Qwen3.5 merges dropped | Qwen3.5 exact | Qwen3.5 real fertility |
|---:|---:|---:|---:|---:|---:|---:|
| natural | 97.0% | 6/614 (1.0%) | 0.2348→0.4067 (+73.2%) | 98.5% | 9/614 (1.5%) | 0.2467→0.4138 (+67.7%) |
| 16384 | 88.3% | 322/614 (52.4%) | 0.2348→0.2491 (+6.1%) | 93.6% | 340/614 (55.4%) | 0.2467→0.2575 (+4.4%) |
| 24576 | 81.8% | 437/614 (71.2%) | 0.2348→0.2425 (+3.3%) | 90.2% | 470/614 (76.5%) | 0.2467→0.2530 (+2.6%) |
| 32768 | 75.4% | 492/614 (80.1%) | 0.2348→0.2397 (+2.1%) | 86.9% | 532/614 (86.6%) | 0.2467→0.2504 (+1.5%) |
| 49152 | 62.8% | 576/614 (93.8%) | 0.2348→0.2372 (+1.0%) | 80.3% | 569/614 (92.7%) | 0.2467→0.2488 (+0.9%) |
| 65536 | 50.1% | 601/614 (97.9%) | 0.2348→0.2363 (+0.6%) | 73.7% | 599/614 (97.6%) | 0.2467→0.2480 (+0.5%) |

Dropping merges is what changes tokenization of ordinary words, and it is a
larger effect than the simulated-OOS numbers above suggest: even for corpus
text whose *final* tokens are all in the keep-set, a dropped intermediate
merge can force the encoder down a different BPE path (more, smaller pieces)
before it reaches those same final tokens, so exactness is not implied by
membership alone. At the natural size, essentially no text roundtrips exactly
(<2%) and real fertility inflates 68-73%; both improve monotonically with K
and only cross into "mostly fine" territory in the last third of the sweep.

**Gate and primary K.** Gate: rebuild-verification exact-match rate ≥95% on
in-set corpus text AND real held-out fertility inflation ≤1%. Only K=65536
clears it for either tokenizer in this sweep (MiniCPM5: 97.9%/+0.6%; Qwen3.5:
97.6%/+0.5%); 49152 comes close but falls short on both models (93.8%/+1.0%
and 92.7%/+0.9%). **Primary K = 65536 for both `minicpm5-2b` and `qwen35`.**
32768 and 49152 are emitted as alternates for the model agents to choose
between size and quality; neither clears the gate as defined here. This gate
says nothing about the input-side risk above, which persists at every K
tested — see "Input side is the one that actually decides this."

Frozen artifacts: `bench/keepsets/{minicpm5-2b,qwen35}-{7583|6414,16384,24576,32768,49152,65536}/`,
each with `keep-idx.json`, `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json`, `config-patch.json`, `coverage.json`,
`provenance.md`, `SHA256SUMS.txt`. `qwen35-*` covers all three Qwen3.5 rungs.

**What this does not do:** it does not decide Avenue C. Output-side coverage
looks safe at any size tested; input-side coverage does not clear a "safe"
bar at any size tested, including the primary K. Whether that is acceptable
is a product call for whoever reads these numbers, not something this gate
resolves by picking a bigger K.

## Keep-set rebuild: `extra_special_tokens` defect and fix (2026-09-09)

**The defect.** Two independent downstream consumers of the frozen
`bench/keepsets/*` artifacts hit the same bug: loading an emitted directory
the way `mlc_llm/interface/gen_config.py:270-284` does --
`transformers.AutoTokenizer.from_pretrained(<dir>)` then `len(tokenizer)` --
came back `K + 3` (e.g. 65539, not 65536) for every `qwen35-*` directory. The
rebuilt `tokenizer_config.json` had kept an `extra_special_tokens` field that
names special-token roles by their literal string
(`audio_bos_token: "<|audio_start|>"`, `audio_eos_token: "<|audio_end|>"`,
`audio_token: "<|audio_pad|>"`, plus four `vision_*`/`image_token`/
`video_token` roles). `rebuild-keepset-tokenizer.py`'s id-based prune only
ever rewrites `added_tokens_decoder` (keyed by id); it never looked at this
field, so it survived byte-for-byte from the unpruned upstream Qwen3.5
config. Three of the seven named strings (the audio/TTS placeholders) fall
outside the K=65536 LaTeX/maths keep-set and are gone from the rebuilt
vocab; `AutoTokenizer` does not error on that, it silently treats each
missing name as a brand-new token and appends it past the end of the
vocabulary -- hence `K + 3`. `gen_config.py` then overwrites
`active_vocab_size` from that inflated `len(hf_tokenizer)`, so a naive
`AutoTokenizer.from_pretrained` + `gen_config` consumer ships a compiled
model with a vocab head three rows too large, with those extra rows' weights
coming from whatever the loader's identity fallback produces for
out-of-range indices -- silently, since nothing anywhere validates tokenizer
length against `vocab_size`. The same latent field, with the same three
missing strings, exists in the original unpruned upstream Qwen3.5
`tokenizer_config.json` too (`transformers.AutoTokenizer.from_pretrained`
against the raw upstream directory also returns 248,323, not 248,320) --
pruning did not introduce this bug, it was just the first thing to make it
bite, because the audio/TTS tokens are common enough to always survive a
248k-token vocab but rare enough in a LaTeX/maths corpus to fall out of a
64k-token keep-set. `bench/keepsets/minicpm5-2b-*` does not have this
particular field, so those directories were never K+3, but they get the new
check too since the underlying class of bug is generic.

**The fix.** `rebuild_tokenizer_config()` in `bench/rebuild-keepset-tokenizer.py`
now computes the surviving token strings (`new tokenizer.json` vocab ∪
`added_tokens`) and drops any `extra_special_tokens` entry whose string is
not in that set, instead of copying the field through unchecked. The same
function additionally checks every other by-string field
(`additional_special_tokens`, `bos_token`, `eos_token`, `pad_token`,
`unk_token`) against the same surviving-strings set and raises loudly
(`SystemExit`) if any of those ever go stale -- they are all required by
`rebuild_tokenizer_json()`'s existing "added/special ids must always
survive" invariant to be real added-token entries, so this should never
trigger, but a silent drop of a load-bearing field is exactly the failure
mode this fix exists to close off, not something to reintroduce quietly for
a different field.

**The new assertion (would have caught this).** For every emitted directory,
the rebuild now loads the tokenizer back from disk two ways -- exactly like
`gen_config.py` (`transformers.AutoTokenizer.from_pretrained(<dir>)`) and via
`tokenizers.Tokenizer.from_file(<dir>/tokenizer.json)` -- and asserts
`len(...) == K` for both, raising `SystemExit` immediately if either
disagrees. Both numbers are recorded per directory in `coverage.json`
(`length_check`) and `provenance.md`. This is a real assertion that runs as
part of every rebuild, not a one-off check: had it existed before, the
extra_special_tokens defect would have failed the rebuild instead of
shipping quietly.

**Per-directory table (post-fix, all 12 keep-sets):**

| directory | K | `extra_special_tokens` entries dropped | `tokenizers.Tokenizer.from_file` len | `transformers.AutoTokenizer.from_pretrained` len |
|---|---:|---:|---:|---:|
| minicpm5-2b-7583 | 7583 | 0 | 7583 | 7583 |
| minicpm5-2b-16384 | 16384 | 0 | 16384 | 16384 |
| minicpm5-2b-24576 | 24576 | 0 | 24576 | 24576 |
| minicpm5-2b-32768 | 32768 | 0 | 32768 | 32768 |
| minicpm5-2b-49152 | 49152 | 0 | 49152 | 49152 |
| minicpm5-2b-65536 | 65536 | 0 | 65536 | 65536 |
| qwen35-6414 | 6414 | 3 | 6414 | 6414 |
| qwen35-16384 | 16384 | 3 | 16384 | 16384 |
| qwen35-24576 | 24576 | 3 | 24576 | 24576 |
| qwen35-32768 | 32768 | 3 | 32768 | 32768 |
| qwen35-49152 | 49152 | 3 | 49152 | 49152 |
| qwen35-65536 | 65536 | 3 | 65536 | 65536 |

All 12 directories were regenerated from the unpruned source tokenizers with
the fixed generator (same command lines as recorded in each
`provenance.md`), every `SHA256SUMS.txt` was regenerated and passes
`shasum -a 256 -c`, and `bench/annotate-keepset-gate.py` was re-run per label
to restore the gate annotation. `tokenizer.json`, `keep-idx.json`,
`special_tokens_map.json`, and `config-patch.json` are byte-identical to the
pre-fix versions in every directory (only `tokenizer_config.json`,
`coverage.json`, `provenance.md`, and `SHA256SUMS.txt` changed) -- the
keep-set contents, BPE merges, roundtrip exactness, and real held-out
fertility numbers reported earlier in this document are unchanged; primary
K=65536 for both tokenizers is unchanged.

**Related check: stale original-id-space ids in `config-patch.json`.**
Checked whether `config-patch.json`'s `stop_token_ids` (the field a
downstream consumer would load and use for generation) carry stale
original-vocab-space ids instead of pruned-space ones, as a sibling agent
found and fixed locally for its own model's `config.json`. In every one of
the 12 `bench/keepsets/*` directories here, `stop_token_ids` are already
correctly remapped through `new_id_map` into the pruned id space (e.g.
`qwen35-65536`: `[65510, 65512]`, both `< 65536`) -- `remap_stop_token_ids()`
already applies `new_id_map` before writing this field, so there was nothing
to fix. The old-space ids visible in `stop_token_ids_source_mismatches` are
intentional documentation of the separate, pre-existing
`mlc-chat-config.json` stop-id bug (see "Tokenizer findings (Step 1)" above)
and are not consumed as ids by anything downstream. No `config.json` file is
emitted by this rebuild at all (only `config-patch.json`); the sibling
agent's `config.json` fix was in its own compiled-model repo, out of scope
here.
