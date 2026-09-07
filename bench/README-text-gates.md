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

## Run

```bash
.venv/bin/python bench/copy-fraction.py
.venv/bin/python bench/vocab-coverage.py
```

Outputs written by the scripts:

- `bench/results-copy-fraction.json`
- `bench/results-vocab-coverage.json`

## Gate B summary table (from run)

| group | n | copy@5 mean | copy@5 med | speedup@5 mean | speedup@5 med | copy@10 mean | copy@10 med | speedup@10 mean | speedup@10 med | in_set_ub mean | in_set_ub med |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| easy | 15 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.163 | 0.154 |
| medium | 12 | 0.067 | 0.018 | 1.088 | 1.019 | 0.075 | 0.018 | 1.105 | 1.019 | 0.121 | 0.071 |
| hard | 11 | 0.137 | 0.125 | 1.167 | 1.135 | 0.140 | 0.125 | 1.173 | 1.135 | 0.134 | 0.109 |
| multiline | 5 | 0.395 | 0.409 | 1.652 | 1.692 | 0.427 | 0.445 | 1.751 | 1.803 | 0.563 | 0.560 |
| overall | 43 | 0.100 | 0.022 | 1.143 | 1.022 | 0.106 | 0.022 | 1.161 | 1.022 | 0.191 | 0.119 |
| multiline_only | 5 | 0.395 | 0.409 | 1.652 | 1.692 | 0.427 | 0.445 | 1.751 | 1.803 | 0.563 | 0.560 |

The bench's three prose items are spoken-form math only; the synthetic set adds unicode and already-LaTeX forms.

| slice | group | n | copy@5 mean | copy@5 med | speedup@5 mean | speedup@5 med | copy@10 mean | copy@10 med | speedup@10 mean | speedup@10 med | in_set_ub mean | in_set_ub med |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | overall | 500 | 0.478 | 0.480 | 2.133 | 1.863 | 0.514 | 0.514 | 2.489 | 2.000 | 0.720 | 0.783 |
| dominant_kind | spoken | 96 | 0.295 | 0.254 | 1.461 | 1.319 | 0.313 | 0.272 | 1.529 | 1.341 | 0.434 | 0.396 |
| dominant_kind | unicode | 88 | 0.354 | 0.347 | 1.603 | 1.510 | 0.373 | 0.354 | 1.682 | 1.525 | 0.624 | 0.647 |
| dominant_kind | latex | 101 | 0.651 | 0.663 | 2.859 | 2.909 | 0.703 | 0.714 | 3.419 | 3.385 | 0.925 | 0.938 |
| dominant_kind | mixed | 137 | 0.405 | 0.395 | 1.756 | 1.625 | 0.432 | 0.418 | 1.881 | 1.700 | 0.672 | 0.681 |
| dominant_kind | none | 78 | 0.749 | 0.750 | 3.283 | 3.000 | 0.821 | 0.818 | 4.446 | 4.000 | 1.000 | 1.000 |
| span_char_share | <25% | 185 | 0.607 | 0.646 | 2.581 | 2.500 | 0.659 | 0.697 | 3.197 | 3.000 | 0.867 | 0.914 |
| span_char_share | 25-50% | 177 | 0.432 | 0.406 | 1.943 | 1.632 | 0.461 | 0.436 | 2.157 | 1.712 | 0.688 | 0.699 |
| span_char_share | >50% | 138 | 0.365 | 0.300 | 1.778 | 1.411 | 0.389 | 0.306 | 1.966 | 1.431 | 0.565 | 0.465 |

## Gate C summary table (from run)

For padding, we use Qwen byte-level BPE merge rank as a frequency proxy by taking the first `N` regular token ids after the 256 byte tokens.

| pad_n | oos_micro_overall | oos_micro_easy | oos_micro_medium | oos_micro_hard | oos_micro_multiline | fert_pruned_micro_overall | ref_bit_exact_frac |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1208 | 0.0813 | 0.0966 | 0.1391 | 0.1281 | 0.5454 | 0.2667 |
| 8192 | 0.0460 | 0.0218 | 0.0324 | 0.0508 | 0.0559 | 0.4757 | 0.4000 |
| 16384 | 0.0218 | 0.0218 | 0.0172 | 0.0231 | 0.0238 | 0.4518 | 0.4000 |
| 32768 | 0.0110 | 0.0060 | 0.0120 | 0.0102 | 0.0121 | 0.4314 | 0.4667 |
| 65536 | 0.0054 | 0.0000 | 0.0089 | 0.0039 | 0.0053 | 0.4220 | 0.6667 |
