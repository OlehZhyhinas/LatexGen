#!/usr/bin/env python3
"""Gate B: measure prompt-lookup copy fraction on benchmark outputs."""
import argparse
import json
from glob import glob
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
BENCH_PATH = ROOT / "bench-data.json"
RESULTS_PATH = ROOT / "results-2026-09-01.json"
OUT_PATH = ROOT / "results-copy-fraction.json"
TOKENIZER_GLOB = str(
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json"
)


def parse_args():
    parser = argparse.ArgumentParser()
    default_synth = ROOT / "synth-spans.jsonl"
    parser.add_argument(
        "--synth",
        default=str(default_synth) if default_synth.exists() else None,
        help="Optional JSONL synth spans file",
    )
    return parser.parse_args()


def load_tokenizer():
    paths = sorted(glob(TOKENIZER_GLOB))
    if not paths:
        raise SystemExit(f"tokenizer.json not found: {TOKENIZER_GLOB}")
    return Tokenizer.from_file(paths[-1]), paths[-1]


def tokenize(tok, text):
    return tok.encode(text, add_special_tokens=False).ids


def last_occurrence_with_continuation(haystack, needle):
    n = len(needle)
    for i in range(len(haystack) - n, -1, -1):
        if haystack[i : i + n] == needle and i + n < len(haystack):
            return i
    return -1


def simulate_prompt_lookup(input_ids, output_ids, k):
    total = len(output_ids)
    i = 0
    accepted = 0
    steps = 0
    while i < total:
        steps += 1
        context = input_ids + output_ids[:i]
        pos = -1
        n_used = 0
        for n in (3, 2):
            if i < n:
                continue
            needle = output_ids[i - n : i]
            m = last_occurrence_with_continuation(context, needle)
            if m >= 0:
                pos = m
                n_used = n
                break

        accepted_now = 0
        if pos >= 0:
            draft = context[pos + n_used : pos + n_used + k]
            max_check = min(len(draft), total - i)
            while accepted_now < max_check and draft[accepted_now] == output_ids[i + accepted_now]:
                accepted_now += 1
        accepted += accepted_now
        emitted = min(total - i, accepted_now + 1)
        i += emitted
    return accepted, steps


def stats(xs):
    if not xs:
        return {"mean": 0.0, "median": 0.0}
    arr = np.array(xs, dtype=float)
    return {"mean": float(arr.mean()), "median": float(np.median(arr))}


def build_group_stats(rows):
    return {
        "n": len(rows),
        "k5": {
            "copy_fraction": stats([x["k5"]["copy_fraction"] for x in rows]),
            "speedup": stats([x["k5"]["speedup"] for x in rows]),
        },
        "k10": {
            "copy_fraction": stats([x["k10"]["copy_fraction"] for x in rows]),
            "speedup": stats([x["k10"]["speedup"] for x in rows]),
        },
        "upper_bound": stats([x["in_set_upper_bound"] for x in rows]),
    }


def print_pair_table(rows):
    print("\nPer-pair metrics")
    print(
        "| item | source | tier | out_toks | in_set_ub | copy@5 | speedup@5 | copy@10 | speedup@10 |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['item']} | {r['source']} | {r['tier']} | {r['total_tokens']} | "
            f"{r['in_set_upper_bound']:.3f} | {r['k5']['copy_fraction']:.3f} | {r['k5']['speedup']:.3f} | "
            f"{r['k10']['copy_fraction']:.3f} | {r['k10']['speedup']:.3f} |"
        )


def print_summary_table(summary):
    print("\nSummary (means / medians)")
    print("| group | n | copy@5 mean | copy@5 med | speedup@5 mean | speedup@5 med | copy@10 mean | copy@10 med | speedup@10 mean | speedup@10 med | in_set_ub mean | in_set_ub med |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    order = ["easy", "medium", "hard", "multiline", "overall", "multiline_only"]
    for g in order:
        s = summary[g]
        print(
            f"| {g} | {s['n']} | {s['k5']['copy_fraction']['mean']:.3f} | {s['k5']['copy_fraction']['median']:.3f} | "
            f"{s['k5']['speedup']['mean']:.3f} | {s['k5']['speedup']['median']:.3f} | "
            f"{s['k10']['copy_fraction']['mean']:.3f} | {s['k10']['copy_fraction']['median']:.3f} | "
            f"{s['k10']['speedup']['mean']:.3f} | {s['k10']['speedup']['median']:.3f} | "
            f"{s['upper_bound']['mean']:.3f} | {s['upper_bound']['median']:.3f} |"
        )


def build_summary(rows):
    groups = {
        "easy": [r for r in rows if r["tier"] == "easy"],
        "medium": [r for r in rows if r["tier"] == "medium"],
        "hard": [r for r in rows if r["tier"] == "hard"],
        "multiline": [r for r in rows if r["tier"] == "multiline"],
        "overall": rows,
        "multiline_only": [r for r in rows if r["tier"] == "multiline"],
    }
    out = {}
    for name, rs in groups.items():
        out[name] = build_group_stats(rs)
    return out


def print_synth_summary_table(summary):
    print("\nSynth summary (means / medians)")
    print(
        "| slice | group | n | copy@5 mean | copy@5 med | speedup@5 mean | speedup@5 med | copy@10 mean | copy@10 med | speedup@10 mean | speedup@10 med | in_set_ub mean | in_set_ub med |"
    )
    print(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = [("overall", "overall", summary["overall"])]

    preferred_kind_order = ["spoken", "unicode", "latex", "mixed", "none"]
    remaining_kinds = [
        k for k in summary["by_kind"].keys() if k not in set(preferred_kind_order)
    ]
    for k in preferred_kind_order + sorted(remaining_kinds):
        if k in summary["by_kind"]:
            rows.append(("dominant_kind", k, summary["by_kind"][k]))

    for b in ("<25%", "25-50%", ">50%"):
        rows.append(("span_char_share", b, summary["by_span_char_share"][b]))

    for slice_name, group_name, s in rows:
        print(
            f"| {slice_name} | {group_name} | {s['n']} | "
            f"{s['k5']['copy_fraction']['mean']:.3f} | {s['k5']['copy_fraction']['median']:.3f} | "
            f"{s['k5']['speedup']['mean']:.3f} | {s['k5']['speedup']['median']:.3f} | "
            f"{s['k10']['copy_fraction']['mean']:.3f} | {s['k10']['copy_fraction']['median']:.3f} | "
            f"{s['k10']['speedup']['mean']:.3f} | {s['k10']['speedup']['median']:.3f} | "
            f"{s['upper_bound']['mean']:.3f} | {s['upper_bound']['median']:.3f} |"
        )


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def render_expected_output(text, spans):
    out = []
    cursor = 0
    for span in sorted(spans, key=lambda x: x["start"]):
        start = int(span["start"])
        end = int(span["end"])
        if start < cursor or end < start or end > len(text):
            raise ValueError(
                f"invalid span range start={start} end={end} for text length {len(text)}"
            )
        out.append(text[cursor:start])
        out.append(f"\\( {span['latex']} \\)")
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def dominant_kind(spans):
    if not spans:
        return "none"
    kinds = {s.get("kind", "unknown") for s in spans}
    if len(kinds) == 1:
        return next(iter(kinds))
    return "mixed"


def span_char_share(text, spans):
    if not text:
        return 0.0
    total = 0
    for s in spans:
        total += int(s["end"]) - int(s["start"])
    return total / len(text)


def span_share_bucket(share):
    if share < 0.25:
        return "<25%"
    if share <= 0.5:
        return "25-50%"
    return ">50%"


def evaluate_pair(tok, input_text, output_text):
    in_ids = tokenize(tok, input_text)
    out_ids = tokenize(tok, output_text)
    total = len(out_ids)
    if total == 0:
        return None
    in_set = set(in_ids)
    upper = sum(1 for t in out_ids if t in in_set) / total
    acc5, steps5 = simulate_prompt_lookup(in_ids, out_ids, k=5)
    acc10, steps10 = simulate_prompt_lookup(in_ids, out_ids, k=10)
    return {
        "total_tokens": total,
        "in_set_upper_bound": upper,
        "k5": {
            "accepted_tokens": acc5,
            "steps": steps5,
            "copy_fraction": acc5 / total,
            "speedup": total / steps5 if steps5 else 0.0,
        },
        "k10": {
            "accepted_tokens": acc10,
            "steps": steps10,
            "copy_fraction": acc10 / total,
            "speedup": total / steps10 if steps10 else 0.0,
        },
    }


def build_synth_summary(rows):
    by_kind = {}
    by_share = {"<25%": [], "25-50%": [], ">50%": []}
    for r in rows:
        by_kind.setdefault(r["dominant_kind"], []).append(r)
        by_share[r["span_char_share_bucket"]].append(r)

    return {
        "overall": build_group_stats(rows),
        "by_kind": {k: build_group_stats(v) for k, v in by_kind.items()},
        "by_span_char_share": {
            bucket: build_group_stats(rs) for bucket, rs in by_share.items()
        },
    }


def evaluate_synth(tok, synth_path):
    examples = load_jsonl(synth_path)
    rows = []
    for ex in examples:
        text = ex.get("text", "")
        spans = ex.get("spans", [])
        expected = render_expected_output(text, spans)
        share = span_char_share(text, spans)
        metrics = evaluate_pair(tok, text, expected)
        if metrics is None:
            continue
        rows.append(
            {
                "id": ex.get("id", ""),
                "span_count": len(spans),
                "dominant_kind": dominant_kind(spans),
                "span_char_share": share,
                "span_char_share_bucket": span_share_bucket(share),
                **metrics,
            }
        )
    return {
        "path": str(synth_path),
        "n_examples": len(examples),
        "n_scored": len(rows),
        "rows": rows,
        "summary": build_synth_summary(rows),
    }


def main():
    args = parse_args()
    tok, tok_path = load_tokenizer()
    items = json.load(open(BENCH_PATH))
    results = json.load(open(RESULTS_PATH))
    by_id = {x["id"]: x for x in items}

    pairs = []
    for it in items:
        pairs.append(
            {
                "item": it["id"],
                "tier": it["tier"],
                "source": "reference",
                "input": it["input"],
                "output": it["reference"],
            }
        )
    for r in results:
        if r.get("approach", "").startswith("direct:Qwen3-") and r.get("correct") is True:
            it = by_id[r["item"]]
            pairs.append(
                {
                    "item": r["item"],
                    "tier": r["tier"],
                    "source": r["approach"],
                    "input": it["input"],
                    "output": r.get("output", ""),
                }
            )

    rows = []
    for p in pairs:
        metrics = evaluate_pair(tok, p["input"], p["output"])
        if metrics is None:
            continue
        rows.append(
            {
                "item": p["item"],
                "tier": p["tier"],
                "source": p["source"],
                **metrics,
            }
        )

    rows.sort(key=lambda x: (x["item"], x["source"]))
    summary = build_summary(rows)

    print(f"tokenizer: {tok_path}")
    print(f"pairs: {len(rows)}")
    print_pair_table(rows)
    print_summary_table(summary)

    synth = None
    if args.synth:
        synth_path = Path(args.synth).expanduser()
        if not synth_path.is_absolute():
            synth_path = Path.cwd() / synth_path
        if not synth_path.exists():
            raise SystemExit(f"synth JSONL not found: {synth_path}")
        synth = evaluate_synth(tok, synth_path)
        print(f"\nsynth examples: {synth['n_examples']} (scored={synth['n_scored']})")
        print_synth_summary_table(synth["summary"])
    else:
        print("\nsynth examples: skipped (no --synth and default file missing)")

    payload = {
        "tokenizer_path": tok_path,
        "pairs": rows,
        "summary": summary,
        "config": {"n_grams": [3, 2], "draft_tokens": [5, 10]},
        "synth": synth,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
