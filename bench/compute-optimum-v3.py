#!/usr/bin/env python3
"""Job 3: T(K) per model, per selection strategy, using v3's measured
fertility sweep (bench/results-keepset-v3-sweep-<tokenizer>.json) and
MEASURED head-kernel costs read from webnn-workbench origin/main (via
`git show`, read-only) -- docs/qwen35-08-vocab-report.md,
docs/qwen35-4b-vocab-report.md, docs/qwen35-9b-vocab-report.md,
docs/minicpm5-2b-vocab-report.md, docs/results.md. See
bench/README-keepset-v3.md for the exact quotes/line numbers each figure
below was verified against.

    T = (characters * fertility(S)) * ms_per_token(|S|)
    ms_per_token(K) = body + head * (K / V)          [head scales linearly
                                                        with vocab rows: a
                                                        GEMV dequant+matmul]
    net(K) = ms_per_token(V) / (ms_per_token(K) * (1 + fertility_inflation(K)))

net(K) is the predicted speedup of pruning-to-K relative to not pruning at
all (>1 = faster, <1 = slower), combining both axes the brief's `T` formula
mixes: token-emission speed (via ms_per_token) and how many tokens have to
be emitted per character (via fertility).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "results-keepset-v3-optimum.json"

# =============================================================================
# Measured head-kernel costs (see module docstring for source docs).
# ms_total = the tuned/headline decode ms/token at the model's FULL,
# unpruned vocab V (i.e. ms_per_token(V) = body + head).
# bytes_per_row = the measured on-disk bytes freed per vocabulary row
# dropped (q4f16_1 quantized embedding/lm_head; see README for the
# byte-for-byte reconciliation against each doc's own reported download
# delta at K=65,536).
# =============================================================================
MODELS = {
    "Qwen3.5-0.8B": {
        "tokenizer_label": "qwen35",
        "vocab_size": 248320,
        "ms_total": 5.240,   # 1000/190.83 tok/s, results.md:1690 (Chunk256+K=5+flush32 headline)
        "head_ms": 0.648,    # qwen35-08-vocab-report.md:72-73: "~648-658 us/token, ~11.6% of GPU decode time"
        "head_ms_range": [0.648, 0.658],
        "bytes_per_row": 576,   # tied embed/lm_head; (248320-65536) rows * 576B = 105,283,584B measured delta (report:11, exact)
        "source_note": "head range as directly stated by the doc (648-658us); ms_total from a different section's tuned headline (5.24ms) than the profile capture that produced the 11.6% share (0.648/5.24=12.4%, not 11.6% -- see README for this discrepancy, carried through rather than smoothed over)",
    },
    "Qwen3.5-4B": {
        "tokenizer_label": "qwen35",
        "vocab_size": 248320,
        "ms_total": 15.61,   # qwen35-4b-vocab-report.md: "the headline 15.61 ms/token"
        "head_ms": 1.372,    # qwen35-4b-vocab-report.md: "1.372 ms of the headline 15.61 ms/token" (8.79% share)
        "head_ms_range": [1.372, 1.372],
        "bytes_per_row": 1440,  # tied; (248320-65536)*1440 = 263,208,960B vs measured delta 284,149,634B (report:390) -- delta includes non-vocab-tensor bytes (tokenizer.json shrink etc.), see README
        "source_note": None,
    },
    "Qwen3.5-9B": {
        "tokenizer_label": "qwen35",
        "vocab_size": 248320,
        "ms_total": 26.70,   # 1000/37.45 tok/s, results.md:2232 ("9B chunk256+K=4+flush64" tuned headline -- the shipped roll the vocab report profiles)
        "head_ms": None,     # range only, per the brief's explicit instruction not to pick a point
        "head_ms_range": [26.70 * 0.040, 26.70 * 0.046],  # qwen35-9b-vocab-report.md: "4.0% of decode GPU time"; "as high as 4.6%" -- Stage-1-gate range
        "bytes_per_row": 4608,  # both embed_tokens+lm_head stored on disk despite tie_word_embeddings; qwen35-9b-vocab-report.md:225-230, cross-checked exactly against measured shard bytes and HF API blob listing
        "source_note": "head cost is a range (4.0-4.6% of ms_total), not a point, per the brief; net(K) computed at both ends",
    },
    "MiniCPM5-2B": {
        "tokenizer_label": "minicpm5-2b",
        "vocab_size": 130560,
        "ms_total": 9.74,    # minicpm5-2b-vocab-report.md / results.md:1879 "tuned final observed median is approximately 9.74 ms/token"
        "head_ms": 0.6657,   # minicpm5-2b-vocab-report.md:61 / results.md:2569, measured decode-step profile at the full 130,560 vocab
        "head_ms_range": [0.6657, 0.6657],
        "bytes_per_row": 2304,  # untied; (130560-65536)*2304 = 149,815,296B == measured delta exactly (report:291, exact match)
        "source_note": "head_ms (0.6657) is from the decode-step profile (total 9.91ms/token there); ms_total (9.74) is from the separately-run tuned burst/flush headline -- two different runs, see README",
    },
}


def load_sweep(label):
    return json.loads((ROOT / f"results-keepset-v3-sweep-{label}.json").read_text())


def ms_per_token(model, K, head_ms):
    body = model["ms_total"] - head_ms
    return body + head_ms * (K / model["vocab_size"])


def net(model, K, inflation, head_ms):
    return model["ms_total"] / (ms_per_token(model, K, head_ms) * (1.0 + inflation))


def mb_saved(model, actual_K):
    return (model["vocab_size"] - actual_K) * model["bytes_per_row"] / (2 ** 20)


def main():
    sweeps = {label: load_sweep(label) for label in {m["tokenizer_label"] for m in MODELS.values()}}

    all_results = {}
    for model_name, model in MODELS.items():
        sweep = sweeps[model["tokenizer_label"]]
        model_out = {"by_strategy": {}}
        best_overall = None  # (net, strategy, K) at head_ms point estimate (or range midpoint)

        for strategy, by_k in sweep["by_strategy"].items():
            rows = []
            for k_str, entry in by_k.items():
                if entry.get("overflow"):
                    continue
                actual_K = entry["actual_size"]
                inflation = entry["fertility"]["fertility_inflation_relative"]
                mb = mb_saved(model, actual_K)
                if model["head_ms"] is not None:
                    n = net(model, actual_K, inflation, model["head_ms"])
                    row = {"target_K": int(k_str), "actual_K": actual_K, "inflation": inflation, "mb_saved": mb, "net": n}
                else:
                    lo, hi = model["head_ms_range"]
                    n_lo = net(model, actual_K, inflation, lo)
                    n_hi = net(model, actual_K, inflation, hi)
                    row = {
                        "target_K": int(k_str), "actual_K": actual_K, "inflation": inflation, "mb_saved": mb,
                        "net_range": [min(n_lo, n_hi), max(n_lo, n_hi)],
                        "net": (n_lo + n_hi) / 2.0,
                    }
                rows.append(row)
            rows.sort(key=lambda r: r["target_K"])
            model_out["by_strategy"][strategy] = rows

            if rows:
                best = max(rows, key=lambda r: r["net"])
                # "if two K values are within noise, the smaller K wins" --
                # noise band: within 0.3% relative net of the best, prefer
                # the smallest actual_K among those.
                close = [r for r in rows if r["net"] >= best["net"] * (1 - 0.003)]
                chosen = min(close, key=lambda r: r["actual_K"])
                model_out.setdefault("optimum_by_strategy", {})[strategy] = chosen
                if best_overall is None or chosen["net"] > best_overall[0]:
                    best_overall = (chosen["net"], strategy, chosen)

        if best_overall:
            model_out["overall_best_strategy"] = best_overall[1]
            model_out["overall_optimum"] = best_overall[2]
        all_results[model_name] = model_out

        print(f"\n=== {model_name} (tokenizer={model['tokenizer_label']}, V={model['vocab_size']}, ms_total={model['ms_total']}) ===")
        for strategy, rows in model_out["by_strategy"].items():
            print(f"  strategy {strategy}:")
            for r in rows:
                net_str = f"{r['net']:.4f}" if "net_range" not in r else f"{r['net_range'][0]:.4f}-{r['net_range'][1]:.4f}"
                print(f"    K={r['target_K']:>6} actual={r['actual_K']:>6} inflation={r['inflation']:+.4%} mb_saved={r['mb_saved']:.1f} net={net_str}")
            opt = model_out.get("optimum_by_strategy", {}).get(strategy)
            if opt:
                print(f"    -> optimum: K={opt['actual_K']} net={opt['net']:.4f} mb_saved={opt['mb_saved']:.1f}")
        if best_overall:
            print(f"  BEST STRATEGY: {best_overall[1]} at K={best_overall[2]['actual_K']} net={best_overall[2]['net']:.4f}")

    OUT_PATH.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
