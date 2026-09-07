#!/usr/bin/env python3
"""Summarize judged pipeline rows as direct-vs-pipeline comparison."""
import argparse
import collections
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parent.parent
PDF_PASTES = ROOT / "bench" / "pdf-pastes.json"
MODE_ORDER = [
    "direct",
    "direct-norm",
    "pipeline",
    "pipeline-batch",
    "pipeline-ctx",
    "pipeline-ctx-batch",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("judged")
    ap.add_argument("--md", action="store_true")
    return ap.parse_args()


def read_json(path):
    with open(path) as f:
        return json.load(f)


def parse_approach(approach):
    if ":" not in approach:
        return None, None
    mode, model = approach.split(":", 1)
    return mode, model


def median_seconds(rows):
    vals = [r["ms"] for r in rows if isinstance(r.get("ms"), (int, float)) and r.get("ms", -1) >= 0]
    if not vals:
        return None
    return statistics.median(vals) / 1000.0


def accuracy(rows):
    if not rows:
        return None
    return sum(1 for r in rows if r.get("correct") is True) / len(rows)


def fmt_pct(v):
    return "—" if v is None else f"{v:.0%}"


def fmt_s(v):
    return "—" if v is None else f"{v:.2f}"


def fmt_speedup(v):
    return "—" if v is None else f"{v:.2f}x"


def mode_sort_key(mode):
    try:
        return (0, MODE_ORDER.index(mode))
    except ValueError:
        return (1, mode)


def main():
    args = parse_args()
    rows = read_json(args.judged)
    profile_by_id = {r["id"]: r.get("profile") for r in read_json(PDF_PASTES)}

    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r.get("tier") != "pdf-paste":
            continue
        mode, model = parse_approach(r.get("approach", ""))
        if not model:
            continue
        row = dict(r)
        row["profile"] = row.get("profile") or profile_by_id.get(row.get("item"))
        grouped[model][mode].append(row)

    lines = []
    for model in sorted(grouped):
        direct_rows = grouped[model].get("direct", [])
        direct_med = median_seconds(direct_rows)
        modes = sorted(grouped[model].keys(), key=mode_sort_key)
        for mode in modes:
            rs = grouped[model].get(mode, [])
            all_acc = accuracy(rs)
            all_med = median_seconds(rs)
            clean = [r for r in rs if r.get("profile") == "clean"]
            dirty = [r for r in rs if r.get("profile") == "dirty"]
            clean_acc = accuracy(clean)
            clean_med = median_seconds(clean)
            dirty_acc = accuracy(dirty)
            dirty_med = median_seconds(dirty)
            speedup = None
            if direct_med and all_med and all_med > 0:
                speedup = direct_med / all_med
            if mode == "direct":
                speedup = 1.0 if all_med is not None else None
            lines.append(
                [
                    model,
                    mode,
                    fmt_pct(all_acc),
                    fmt_s(all_med),
                    fmt_pct(clean_acc),
                    fmt_s(clean_med),
                    fmt_pct(dirty_acc),
                    fmt_s(dirty_med),
                    fmt_speedup(speedup),
                ]
            )

    title = "shipped direct vs pipeline"
    headers = ["model", "mode", "acc all", "s all", "acc clean", "s clean", "acc dirty", "s dirty", "speedup vs direct"]
    if args.md:
        print(f"## {title}")
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join("---" for _ in headers) + "|")
        for row in lines:
            print("| " + " | ".join(row) + " |")
    else:
        print(title)
        widths = [max(len(h), *(len(r[i]) for r in lines)) for i, h in enumerate(headers)]
        print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
        for row in lines:
            print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


if __name__ == "__main__":
    main()
