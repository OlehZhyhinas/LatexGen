#!/usr/bin/env python3
"""Offline prompt-lookup acceptance simulation on arXiv paste outputs."""
import argparse
import json
from bisect import bisect_left
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

QWEN_TOKENIZER_GLOB = str(
    Path.home()
    / ".cache/huggingface/hub/models--mlc-ai--Qwen3.5-4B-q4f16_1-MLC/snapshots/*/tokenizer.json"
)
MINICPM_TOKENIZER_GLOB = str(
    Path.home()
    / ".cache/huggingface/hub/models--ozhyhinas--MiniCPM5-2B-q4f16_1-MLC/snapshots/*/tokenizer.json"
)

DEFAULT_INPUTS = {
    "qwen35-08b": "/Users/oleh/personal/webnn-workbench-arxiv-quality/bench/results/arxiv-quality-qwen35-08b-shipped.json",
    "qwen35-4b": "/Users/oleh/personal/webnn-workbench-arxiv-quality/bench/results/arxiv-quality-qwen35-4b-shipped.json",
    "qwen35-9b": "/Users/oleh/personal/webnn-workbench-arxiv-quality/bench/results/arxiv-quality-qwen35-9b-shipped.json",
    "minicpm5-2b": "/Users/oleh/personal/webnn-workbench-arxiv-quality/bench/results/arxiv-quality-minicpm5-2b-shipped.json",
}

MODEL_ORDER = ["qwen35-08b", "qwen35-4b", "qwen35-9b", "minicpm5-2b"]
MODEL_COST = {
    "qwen35-08b": "attention_only",
    "qwen35-4b": "hybrid_recurrent",
    "qwen35-9b": "hybrid_recurrent",
    "minicpm5-2b": "attention_only",
}

N_MAX_GRID = [2, 3, 4]
N_MIN_GRID = [1, 2, 3]
K_GRID = [4, 6, 8, 10, 12, 16]
POLICY_GRID = ["last", "first"]
DEFAULT_PARAM = {"n_max": 3, "n_min": 2, "k": 10, "match_policy": "last"}

CONST_CS = [1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]
LINEAR_AS = [0.02, 0.05, 0.1, 0.2]


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen35-08b", default=DEFAULT_INPUTS["qwen35-08b"])
    parser.add_argument("--qwen35-4b", default=DEFAULT_INPUTS["qwen35-4b"])
    parser.add_argument("--qwen35-9b", default=DEFAULT_INPUTS["qwen35-9b"])
    parser.add_argument("--minicpm5-2b", default=DEFAULT_INPUTS["minicpm5-2b"])
    parser.add_argument("--qwen-tokenizer-glob", default=QWEN_TOKENIZER_GLOB)
    parser.add_argument("--minicpm-tokenizer-glob", default=MINICPM_TOKENIZER_GLOB)
    parser.add_argument(
        "--out",
        default=str((here.parent / "bench/results-prompt-lookup-acceptance.json").resolve()),
    )
    return parser.parse_args()


def load_tokenizer(glob_pattern):
    paths = sorted(glob(str(Path(glob_pattern).expanduser())))
    if not paths:
        raise SystemExit(f"tokenizer.json not found: {glob_pattern}")
    return Tokenizer.from_file(paths[-1]), paths[-1]


def tokenize(tok, text):
    return tok.encode(text or "", add_special_tokens=False).ids


def row_is_skipped(row):
    err = row.get("error")
    return bool(err) and err != "None"


def normalize_prompt_text(prompt_messages):
    pieces = []
    for msg in prompt_messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            pieces.append(content)
        elif content is None:
            pieces.append("")
        else:
            pieces.append(str(content))
    return pieces


def build_rows(result_path, tok):
    payload = json.load(open(result_path))
    rows = payload.get("rows", [])
    used = []
    skipped = 0
    abs_len_diffs = []
    ms_per_char = []
    tok_per_s = []

    for row in rows:
        if row_is_skipped(row):
            skipped += 1
            continue

        prompt_contents = normalize_prompt_text(row.get("prompt", []))
        prompt_ids = []
        # We skip chat templating; template control tokens cannot be copied into user-visible output.
        for content in prompt_contents:
            prompt_ids.extend(tokenize(tok, content))

        raw = row.get("raw", "")
        raw = raw if isinstance(raw, str) else str(raw)
        gen_ids = tokenize(tok, raw)

        completion_tokens = row.get("completionTokens")
        if isinstance(completion_tokens, (int, float)):
            diff = abs(len(gen_ids) - int(completion_tokens))
            abs_len_diffs.append(diff)

        ms = row.get("ms")
        output = row.get("output", "")
        output = output if isinstance(output, str) else str(output)
        out_chars = len(output)
        if isinstance(ms, (int, float)) and ms > 0 and out_chars > 0:
            ms_per_char.append(float(ms) / float(out_chars))
        if isinstance(ms, (int, float)) and ms > 0 and isinstance(completion_tokens, (int, float)):
            tok_per_s.append(float(completion_tokens) / (float(ms) / 1000.0))

        used.append(
            {
                "item": str(row.get("item", "")),
                "prompt_ids": prompt_ids,
                "gen_ids": gen_ids,
            }
        )

    mean_abs = float(np.mean(abs_len_diffs)) if abs_len_diffs else 0.0
    max_abs = int(max(abs_len_diffs)) if abs_len_diffs else 0
    median_ms_per_char = float(np.median(np.array(ms_per_char, dtype=float))) if ms_per_char else 0.0
    median_tok_per_s = float(np.median(np.array(tok_per_s, dtype=float))) if tok_per_s else 0.0

    return {
        "rows_total": len(rows),
        "rows_used": used,
        "rows_used_count": len(used),
        "rows_skipped_error": skipped,
        "sanity_mean_abs_len_diff": mean_abs,
        "sanity_max_abs_len_diff": max_abs,
        "median_ms_per_char": median_ms_per_char,
        "median_tok_per_s": median_tok_per_s,
    }


def extend_ngram_index(source_ids, old_len, new_len, index_by_n, n_values):
    for n in n_values:
        if new_len < n:
            continue
        start = max(0, old_len - n + 1)
        end = new_len - n
        table = index_by_n[n]
        for pos in range(start, end + 1):
            table[tuple(source_ids[pos : pos + n])].append(pos)


def simulate_item(prompt_ids, gen_ids, n_min, n_max, k, match_policy, want_log):
    total = len(gen_ids)
    source_ids = list(prompt_ids)
    n_values = list(range(n_min, n_max + 1))
    index_by_n = {n: defaultdict(list) for n in n_values}
    extend_ngram_index(source_ids, 0, len(source_ids), index_by_n, n_values)

    i = 0
    passes = 0
    draft_passes = 0
    nodraft_passes = 0
    sum_j = 0
    full_accept = 0
    sum_d_minus_1 = 0
    partial_count = 0
    partial_j0_count = 0
    partial_sum_j = 0
    j_hist = [0] * 17
    d_hist = [0] * 17
    logs = [] if want_log else None

    while i < total:
        passes += 1
        had_draft = False
        d = 0
        j = 0
        draft = None

        for n in range(n_max, n_min - 1, -1):
            if len(source_ids) < n + 1:
                continue
            query = tuple(source_ids[-n:])
            pos_list = index_by_n[n].get(query)
            if not pos_list:
                continue

            final_pos = len(source_ids) - n
            chosen_pos = None
            if match_policy == "last":
                ix = bisect_left(pos_list, final_pos) - 1
                if ix >= 0:
                    chosen_pos = pos_list[ix]
            else:
                if pos_list[0] < final_pos:
                    chosen_pos = pos_list[0]

            if chosen_pos is None:
                continue

            draft_start = chosen_pos + n
            if draft_start >= len(source_ids):
                continue
            draft_end = min(draft_start + k, len(source_ids))
            if draft_end <= draft_start:
                continue

            draft = source_ids[draft_start:draft_end]
            d = len(draft)
            had_draft = True
            break

        if not had_draft:
            nodraft_passes += 1
            committed = 1
        else:
            draft_passes += 1
            max_check = min(d, total - i)
            while j < max_check and draft[j] == gen_ids[i + j]:
                j += 1
            committed = min(total - i, j + 1)
            sum_j += j
            sum_d_minus_1 += d - 1
            if j == d:
                full_accept += 1
            else:
                partial_count += 1
                partial_sum_j += j
                if j == 0:
                    partial_j0_count += 1
            j_hist[min(j, 16)] += 1
            d_hist[min(d, 16)] += 1

        if want_log:
            logs.append([had_draft, d, j, committed])

        old_len = len(source_ids)
        source_ids.extend(gen_ids[i : i + committed])
        i += committed
        extend_ngram_index(source_ids, old_len, len(source_ids), index_by_n, n_values)

    return {
        "N_tokens": total,
        "N_passes": passes,
        "N_draft_passes": draft_passes,
        "N_nodraft_passes": nodraft_passes,
        "sum_j": sum_j,
        "full_accept_count": full_accept,
        "sum_d_minus_1": sum_d_minus_1,
        "partial_count": partial_count,
        "partial_j0_count": partial_j0_count,
        "partial_sum_j": partial_sum_j,
        "j_hist": j_hist,
        "d_hist": d_hist,
        "log": logs,
    }


def speedup_from_agg(agg, cost_model):
    n_tokens = agg["N_tokens"]
    if n_tokens <= 0:
        return {
            "constant": {f"{c:.2f}": 0.0 for c in CONST_CS},
            "linear": {f"{a:.2f}": 0.0 for a in LINEAR_AS},
        }

    out_const = {}
    out_linear = {}
    n_nodraft = agg["N_nodraft_passes"]
    n_draft = agg["N_draft_passes"]
    partial_count = agg["partial_count"]
    partial_j0_count = agg["partial_j0_count"]
    sum_d_minus_1 = agg["sum_d_minus_1"]
    partial_sum_j = agg["partial_sum_j"]

    for c in CONST_CS:
        if cost_model == "attention_only":
            total_cost = n_nodraft + c * n_draft
        else:
            rerun = c * partial_count + (1.0 - c) * partial_j0_count
            total_cost = n_nodraft + c * n_draft + rerun
        out_const[f"{c:.2f}"] = float(n_tokens / total_cost) if total_cost > 0 else 0.0

    for a in LINEAR_AS:
        if cost_model == "attention_only":
            total_cost = n_nodraft + n_draft + a * sum_d_minus_1
        else:
            rerun = partial_count + a * partial_sum_j
            total_cost = n_nodraft + n_draft + a * sum_d_minus_1 + rerun
        out_linear[f"{a:.2f}"] = float(n_tokens / total_cost) if total_cost > 0 else 0.0

    return {"constant": out_const, "linear": out_linear}


def run_param(rows_used, param, cost_model):
    agg = {
        "N_tokens": 0,
        "N_passes": 0,
        "N_draft_passes": 0,
        "N_nodraft_passes": 0,
        "sum_j": 0,
        "full_accept_count": 0,
        "sum_d_minus_1": 0,
        "partial_count": 0,
        "partial_j0_count": 0,
        "partial_sum_j": 0,
        "j_hist": [0] * 17,
        "d_hist": [0] * 17,
    }
    item_tpp = []
    per_item_logs = {}
    want_log = (
        param["n_max"] == DEFAULT_PARAM["n_max"]
        and param["n_min"] == DEFAULT_PARAM["n_min"]
        and param["k"] == DEFAULT_PARAM["k"]
        and param["match_policy"] == DEFAULT_PARAM["match_policy"]
    )

    for row in rows_used:
        sim = simulate_item(
            row["prompt_ids"],
            row["gen_ids"],
            param["n_min"],
            param["n_max"],
            param["k"],
            param["match_policy"],
            want_log=want_log,
        )
        for key in (
            "N_tokens",
            "N_passes",
            "N_draft_passes",
            "N_nodraft_passes",
            "sum_j",
            "full_accept_count",
            "sum_d_minus_1",
            "partial_count",
            "partial_j0_count",
            "partial_sum_j",
        ):
            agg[key] += sim[key]
        for ix in range(17):
            agg["j_hist"][ix] += sim["j_hist"][ix]
            agg["d_hist"][ix] += sim["d_hist"][ix]
        if sim["N_passes"] > 0:
            item_tpp.append(sim["N_tokens"] / sim["N_passes"])
        else:
            item_tpp.append(0.0)
        if want_log:
            per_item_logs[row["item"]] = sim["log"]

    tpp = float(agg["N_tokens"] / agg["N_passes"]) if agg["N_passes"] > 0 else 0.0
    item_tpp_med = float(np.median(np.array(item_tpp, dtype=float))) if item_tpp else 0.0
    mean_j = float(agg["sum_j"] / agg["N_draft_passes"]) if agg["N_draft_passes"] > 0 else 0.0
    full_rate = (
        float(agg["full_accept_count"] / agg["N_draft_passes"]) if agg["N_draft_passes"] > 0 else 0.0
    )
    c_star = None
    if agg["N_draft_passes"] > 0:
        c_star = float((agg["N_tokens"] - agg["N_nodraft_passes"]) / agg["N_draft_passes"])

    speedups = speedup_from_agg(agg, cost_model)
    out = {
        "n_max": param["n_max"],
        "n_min": param["n_min"],
        "k": param["k"],
        "match_policy": param["match_policy"],
        "N_tokens": agg["N_tokens"],
        "N_passes": agg["N_passes"],
        "N_draft_passes": agg["N_draft_passes"],
        "N_nodraft_passes": agg["N_nodraft_passes"],
        "tokens_per_pass_micro": tpp,
        "tokens_per_pass_item_median": item_tpp_med,
        "mean_j_given_draft": mean_j,
        "full_acceptance_rate": full_rate,
        "hist_j_0_16": agg["j_hist"],
        "hist_d_0_16": agg["d_hist"],
        "break_even_c_star_attention": c_star,
        "speedups": speedups,
    }
    if want_log:
        out["per_item_pass_log"] = per_item_logs
    return out


def make_param_grid():
    grid = []
    for policy in POLICY_GRID:
        for n_max in N_MAX_GRID:
            for n_min in N_MIN_GRID:
                if n_min > n_max:
                    continue
                for k in K_GRID:
                    grid.append(
                        {
                            "n_max": n_max,
                            "n_min": n_min,
                            "k": k,
                            "match_policy": policy,
                        }
                    )
    return grid


def param_key(n_max, n_min, k, policy):
    return f"nmax={n_max}|nmin={n_min}|k={k}|policy={policy}"


def print_default_tables(results_by_model):
    for model in MODEL_ORDER:
        model_blob = results_by_model[model]
        defaults = None
        for row in model_blob["parameter_sets"]:
            if (
                row["n_max"] == DEFAULT_PARAM["n_max"]
                and row["n_min"] == DEFAULT_PARAM["n_min"]
                and row["k"] == DEFAULT_PARAM["k"]
                and row["match_policy"] == DEFAULT_PARAM["match_policy"]
            ):
                defaults = row
                break
        if defaults is None:
            continue
        c_star = defaults["break_even_c_star_attention"]
        c_star_txt = f"{c_star:.3f}" if c_star is not None else "NA"
        c1 = defaults["speedups"]["constant"]["1.00"]
        c12 = defaults["speedups"]["constant"]["1.20"]
        c15 = defaults["speedups"]["constant"]["1.50"]
        c20 = defaults["speedups"]["constant"]["2.00"]
        print(f"\n### {model} (default n_max=3, n_min=2, k=10, policy=last)")
        print("| tokens/pass | full_accept_rate | c* (attention) | speedup c=1.0 | speedup c=1.2 | speedup c=1.5 | speedup c=2.0 |")
        print("|---:|---:|---:|---:|---:|---:|---:|")
        print(
            f"| {defaults['tokens_per_pass_micro']:.3f} | {defaults['full_acceptance_rate']:.3f} | {c_star_txt} | "
            f"{c1:.3f} | {c12:.3f} | {c15:.3f} | {c20:.3f} |"
        )


def print_grid_tpp_table(results_by_model):
    print("\n### Tokens/pass grid (policy=last)")
    print("| model | n_max | n_min | k=4 | k=6 | k=8 | k=10 | k=12 | k=16 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model in MODEL_ORDER:
        rows = results_by_model[model]["parameter_sets"]
        by_key = {
            param_key(r["n_max"], r["n_min"], r["k"], r["match_policy"]): r for r in rows
        }
        for n_max in N_MAX_GRID:
            for n_min in N_MIN_GRID:
                if n_min > n_max:
                    continue
                cells = []
                for k in K_GRID:
                    krow = by_key[param_key(n_max, n_min, k, "last")]
                    cells.append(f"{krow['tokens_per_pass_micro']:.3f}")
                print(
                    f"| {model} | {n_max} | {n_min} | "
                    + " | ".join(cells)
                    + " |"
                )


def print_baseline_reference_table(results_by_model):
    print("\n### Baseline runtime reference")
    print("| model | median ms/char | median tok/s |")
    print("|---|---:|---:|")
    for model in MODEL_ORDER:
        b = results_by_model[model]["baseline_reference"]
        print(f"| {model} | {b['median_ms_per_char']:.6f} | {b['median_tok_per_s']:.3f} |")


def print_policy_materiality_note(results_by_model):
    max_rel = 0.0
    max_abs = 0.0
    max_meta = None
    for model in MODEL_ORDER:
        rows = results_by_model[model]["parameter_sets"]
        by_key = {
            param_key(r["n_max"], r["n_min"], r["k"], r["match_policy"]): r for r in rows
        }
        for n_max in N_MAX_GRID:
            for n_min in N_MIN_GRID:
                if n_min > n_max:
                    continue
                for k in K_GRID:
                    last = by_key[param_key(n_max, n_min, k, "last")]["tokens_per_pass_micro"]
                    first = by_key[param_key(n_max, n_min, k, "first")]["tokens_per_pass_micro"]
                    abs_diff = abs(last - first)
                    rel = abs_diff / last if last > 0 else 0.0
                    if rel > max_rel:
                        max_rel = rel
                        max_abs = abs_diff
                        max_meta = (model, n_max, n_min, k)
    material = max_rel >= 0.02
    verdict = "material" if material else "not material"
    if max_meta is None:
        print("\nfirst vs last: not material (no comparable rows).")
        return
    model, n_max, n_min, k = max_meta
    print(
        "\nfirst vs last: "
        f"{verdict} (max abs diff {max_abs:.3f} tokens/pass, "
        f"{max_rel * 100.0:.2f}% relative at {model}, n_max={n_max}, n_min={n_min}, k={k})."
    )


def main():
    args = parse_args()
    param_grid = make_param_grid()

    qwen_tok, qwen_tok_path = load_tokenizer(args.qwen_tokenizer_glob)
    minicpm_tok, minicpm_tok_path = load_tokenizer(args.minicpm_tokenizer_glob)

    model_inputs = {
        "qwen35-08b": args.qwen35_08b,
        "qwen35-4b": args.qwen35_4b,
        "qwen35-9b": args.qwen35_9b,
        "minicpm5-2b": args.minicpm5_2b,
    }
    model_tokenizers = {
        "qwen35-08b": (qwen_tok, qwen_tok_path),
        "qwen35-4b": (qwen_tok, qwen_tok_path),
        "qwen35-9b": (qwen_tok, qwen_tok_path),
        "minicpm5-2b": (minicpm_tok, minicpm_tok_path),
    }

    out_payload = {
        "config": {
            "n_max_grid": N_MAX_GRID,
            "n_min_grid": N_MIN_GRID,
            "k_grid": K_GRID,
            "match_policies": POLICY_GRID,
            "default_parameter_set": DEFAULT_PARAM,
            "constant_c_grid": CONST_CS,
            "linear_a_grid": LINEAR_AS,
            "cost_models": MODEL_COST,
        },
        "models": {},
    }
    summary_data = {}

    for model in MODEL_ORDER:
        tok, tok_path = model_tokenizers[model]
        built = build_rows(model_inputs[model], tok)
        mean_abs = built["sanity_mean_abs_len_diff"]
        max_abs = built["sanity_max_abs_len_diff"]
        if mean_abs > 3.0:
            print(f"WARNING: {model} mean |len(gen_ids)-completionTokens| = {mean_abs:.3f} (>3)")
        print(
            f"{model}: rows used={built['rows_used_count']} skipped={built['rows_skipped_error']} "
            f"sanity mean_abs={mean_abs:.3f} max_abs={max_abs}"
        )

        param_rows = []
        for param in param_grid:
            param_rows.append(run_param(built["rows_used"], param, MODEL_COST[model]))

        summary_data[model] = {
            "parameter_sets": param_rows,
            "baseline_reference": {
                "median_ms_per_char": built["median_ms_per_char"],
                "median_tok_per_s": built["median_tok_per_s"],
            },
        }
        out_payload["models"][model] = {
            "input_path": model_inputs[model],
            "tokenizer_path": tok_path,
            "cost_model": MODEL_COST[model],
            "rows_total": built["rows_total"],
            "rows_used": built["rows_used_count"],
            "rows_skipped_error": built["rows_skipped_error"],
            "sanity": {
                "mean_abs_len_diff": mean_abs,
                "max_abs_len_diff": max_abs,
                "warning_mean_abs_gt_3": mean_abs > 3.0,
            },
            "baseline_reference": {
                "median_ms_per_char": built["median_ms_per_char"],
                "median_tok_per_s": built["median_tok_per_s"],
            },
            "parameter_sets": param_rows,
        }

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_payload, f, indent=2)

    print(f"\nwrote {out_path}")
    print_default_tables(summary_data)
    print_grid_tpp_table(summary_data)
    print_policy_materiality_note(summary_data)
    print_baseline_reference_table(summary_data)


if __name__ == "__main__":
    main()
