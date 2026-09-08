#!/usr/bin/env python3
"""Tier-matched summary of a judged text run: each size class compares the
shipped Qwen3 model only against candidates of the same size.

  python3 bench/summarize-text-tiers.py bench/judged-text-tiers-2026-09-07.json [--md]
"""
import json, sys, collections, statistics

CLASSES = [
    ("~1B", ["qwen3:0.6b", "qwen3.5:0.8b", "gemma3:1b", "openbmb/minicpm5:q4_K_M"]),
    ("~2B", ["qwen3:1.7b", "qwen3.5:2b-q4_K_M", "hf.co/openbmb/MiniCPM5-2B-GGUF:q4_K_M", "gemma4:e2b"]),
    ("~4B", ["qwen3-4b-orig", "hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M", "qwen3:4b", "qwen3.5:4b", "gemma3:4b", "gemma4:e4b"]),
]
SHIPPED = {"qwen3:0.6b", "qwen3:1.7b", "qwen3:4b", "hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M", "qwen3-4b-orig"}
NAMES = {"hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M": "qwen3:4b (orig)", "qwen3-4b-orig": "qwen3:4b (orig)", "openbmb/minicpm5:q4_K_M": "minicpm5-1b", "hf.co/openbmb/MiniCPM5-2B-GGUF:q4_K_M": "minicpm5-2b", "qwen3.5:2b-q4_K_M": "qwen3.5:2b"}
TIERS = ["easy", "medium", "hard", "multiline", "pdf-paste", "synth-pdf", "synth-unicode", "synth-latex"]

def main():
    path = sys.argv[1]; md = "--md" in sys.argv
    rows = [r for r in json.load(open(path)) if r.get("tier") in TIERS and r.get("ms", -1) >= 0 or r.get("tier") in TIERS]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[r["approach"].replace("direct:", "")][r["tier"]].append(r)
    tiers = [t for t in TIERS if any(t in by[m] for m in by)]
    sep = " | " if md else "  "
    for cls, models in CLASSES:
        present = [m for m in models if m in by]
        if not present: continue
        head = ["model"] + tiers + ["all", "valid", "med s"]
        print(f"\n### {cls}\n" if md else f"\n== {cls}")
        if md: print("| " + " | ".join(head) + " |"); print("|" + "---|" * len(head))
        else: print(f"{'model':16s}" + "".join(f"{t:>14s}" for t in tiers) + f"{'all':>8s}{'valid':>7s}{'med s':>7s}")
        for m in present:
            cells, tot, ok_n = [], 0, 0
            for t in tiers:
                rs = by[m].get(t, [])
                if not rs: cells.append("—"); continue
                c = sum(bool(r.get("correct")) for r in rs); tot += len(rs); ok_n += c
                cells.append(f"{c}/{len(rs)}")
            allrs = [r for t in tiers for r in by[m].get(t, [])]
            valid = sum(bool(r.get("valid")) for r in allrs) / max(1, len(allrs))
            lat = statistics.median([r["ms"] for r in allrs if r.get("ms", -1) >= 0] or [0]) / 1000
            name = NAMES.get(m, m) + (" (ship)" if m in SHIPPED else "")
            vals = [name] + cells + [f"{ok_n / max(1, tot):.0%}", f"{valid:.0%}", f"{lat:.1f}"]
            if md: print("| " + " | ".join(vals) + " |")
            else: print(f"{name:16s}" + "".join(f"{c:>14s}" for c in cells) + f"{vals[-3]:>8s}{vals[-2]:>7s}{vals[-1]:>7s}")

if __name__ == "__main__":
    main()
