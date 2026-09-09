#!/usr/bin/env python3
"""Job: T(K) / net(K) / MB-saved per model, per strategy, from v5's honest
(genuinely held-out) fertility grid (bench/results-keepset-v5-grid.json).

    ms_per_token(K) = body + head * (K / V)      [head scales linearly with
                                                    vocab rows: a GEMV
                                                    dequant+matmul]
    net(K) = ms_per_token(V) / (ms_per_token(K) * (1 + fertility_inflation(K)))

Head-kernel costs verified against `webnn-workbench` `origin/main` via
`git show` (2026-09-09, this task):
  - docs/qwen35-08-vocab-report.md:72-73 "~648-658 us/token, ~11.6% of GPU
    decode time"; docs/results.md:1692 "190.83 tok/s" (Chunk256+K=5+flush32)
    -> 1000/190.83 = 5.240 ms/token.
  - docs/qwen35-4b-vocab-report.md:148-149 "8.79% ... 1.372 ms of the
    headline 15.61 ms/token".
  - docs/qwen35-9b-vocab-report.md:21 "4.0% of decode GPU time"; :45 "as
    high as 4.6%"; docs/results.md:2232 "37.45 tok/s" (9B chunk256+K=4+
    flush64) -> 1000/37.45 = 26.70 ms/token.
  - docs/minicpm5-2b-vocab-report.md:61 "0.6657" ms (vocab head, decode-step
    profile); docs/results.md:1879 "tuned final observed median is
    approximately 9.74 ms/token".
Same figures v3 used (bench/compute-optimum-v3.py), re-verified here rather
than assumed.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GRID_PATH = ROOT / "results-keepset-v5-grid.json"
OUT_PATH = ROOT / "results-keepset-v5-optimum.json"

MODELS = {
    "Qwen3.5-0.8B": {
        "tokenizer_label": "qwen35",
        "vocab_size": 248320,
        "ms_total": 5.240,
        "head_ms": 0.648,
        "head_ms_range": [0.648, 0.658],
        "bytes_per_row": 576,
    },
    "Qwen3.5-4B": {
        "tokenizer_label": "qwen35",
        "vocab_size": 248320,
        "ms_total": 15.61,
        "head_ms": 1.372,
        "head_ms_range": [1.372, 1.372],
        "bytes_per_row": 1440,
    },
    "Qwen3.5-9B": {
        "tokenizer_label": "qwen35",
        "vocab_size": 248320,
        "ms_total": 26.70,
        "head_ms": None,
        "head_ms_range": [26.70 * 0.040, 26.70 * 0.046],
        "bytes_per_row": 4608,
    },
    "MiniCPM5-2B": {
        "tokenizer_label": "minicpm5-2b",
        "vocab_size": 130560,
        "ms_total": 9.74,
        "head_ms": 0.6657,
        "head_ms_range": [0.6657, 0.6657],
        "bytes_per_row": 2304,
    },
}


def ms_per_token(model, K, head_ms):
    body = model["ms_total"] - head_ms
    return body + head_ms * (K / model["vocab_size"])


def net(model, K, inflation, head_ms):
    return model["ms_total"] / (ms_per_token(model, K, head_ms) * (1.0 + inflation))


def mb_saved(model, actual_K):
    return (model["vocab_size"] - actual_K) * model["bytes_per_row"] / (2 ** 20)


def main():
    grid = json.loads(GRID_PATH.read_text())

    all_results = {}
    for model_name, model in MODELS.items():
        label = model["tokenizer_label"]
        rows_by_strategy = {}
        for row in grid[label]["rows"]:
            if row.get("overflow"):
                continue
            rows_by_strategy.setdefault(row["strategy"], []).append(row)

        model_out = {"by_strategy": {}}
        best_overall = None
        for strategy, rows in rows_by_strategy.items():
            out_rows = []
            for row in sorted(rows, key=lambda r: r["target_K"]):
                actual_K = row["actual_K"]
                inflation = row["held_out_full"]["inflation"]
                mb = mb_saved(model, actual_K)
                if model["head_ms"] is not None:
                    n = net(model, actual_K, inflation, model["head_ms"])
                    r = {"target_K": row["target_K"], "actual_K": actual_K, "inflation": inflation, "mb_saved": mb, "net": n}
                else:
                    lo, hi = model["head_ms_range"]
                    n_lo = net(model, actual_K, inflation, lo)
                    n_hi = net(model, actual_K, inflation, hi)
                    r = {
                        "target_K": row["target_K"], "actual_K": actual_K, "inflation": inflation, "mb_saved": mb,
                        "net_range": [min(n_lo, n_hi), max(n_lo, n_hi)], "net": (n_lo + n_hi) / 2.0,
                    }
                out_rows.append(r)
            model_out["by_strategy"][strategy] = out_rows
            if out_rows:
                best = max(out_rows, key=lambda r: r["net"])
                close = [r for r in out_rows if r["net"] >= best["net"] * (1 - 0.003)]
                chosen = min(close, key=lambda r: r["actual_K"])
                model_out.setdefault("optimum_by_strategy", {})[strategy] = chosen
                if best_overall is None or chosen["net"] > best_overall[0]:
                    best_overall = (chosen["net"], strategy, chosen)
        if best_overall:
            model_out["overall_best_strategy"] = best_overall[1]
            model_out["overall_optimum"] = best_overall[2]
        all_results[model_name] = model_out

        print(f"\n=== {model_name} (tokenizer={label}, V={model['vocab_size']}, ms_total={model['ms_total']}) ===")
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
