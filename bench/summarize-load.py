#!/usr/bin/env python3
"""Summarize rows from public/bench-load.html (tier "load").

    curl -s http://localhost:8000/api/bench > bench/results-load-YYYY-MM-DD.json
    python3 bench/summarize-load.py bench/results-load-YYYY-MM-DD.json [more.json]

Prints, per (label, model, device, dtype, src, mode): runs, bytes, median load
time (cold: download + session; warm: Cache API read + session), and median
first-inference time. Then, for every label pair that ran the same bench
items, lists outputs that differ, so a weight change can be checked for
equivalence.
"""
import json
import statistics
import sys
from collections import defaultdict

rows = []
for path in sys.argv[1:]:
    with open(path) as f:
        rows += [r for r in json.load(f) if r.get("tier") == "load"]
if not rows:
    sys.exit("no load rows")

key = lambda r: (r.get("label", ""), r["model"], r["device"], r["dtype"], r["src"], r["mode"])
groups = defaultdict(list)
for r in rows:
    groups[key(r)].append(r)

mb = lambda b: f"{b / 2**20:.0f} MB"
med = lambda xs: statistics.median(xs) if xs else 0

print("| label | model | runtime | source | mode | runs | bytes | load (median) | min..max | download | session | first run |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for k in sorted(groups):
    g = [r for r in groups[k] if "loadMs" in r]  # a failed inference still measured the load
    errs = len(groups[k]) - len(g)
    if not g:
        print(f"| {k[0]} | {k[1]} | {k[2]} {k[3]} | {k[4]} | {k[5]} | 0 (+{errs} failed) | | | | | | |")
        continue
    loads = [r["loadMs"] for r in g]
    print(f"| {k[0]} | {k[1]} | {k[2]} {k[3]} | {k[4]} | {k[5]} | {len(g)}{f' (+{errs} failed)' if errs else ''} | {mb(g[0]['totalBytes'])} | "
          f"{med(loads)/1000:.1f} s | {min(loads)/1000:.1f}..{max(loads)/1000:.1f} s | {med([r['downloadMs'] for r in g])/1000:.1f} s | "
          f"{med([r['sessionMs'] for r in g])/1000:.1f} s | {med([r.get('warmupMs', 0) for r in g])/1000:.2f} s |")

# Output equivalence across labels, per model.
by_model = defaultdict(dict)  # model -> label -> {item: output}
for r in rows:
    if r.get("items"):
        by_model[r["model"]].setdefault(r.get("label", ""), {}).update({i["id"]: i["output"] for i in r["items"]})
for model, labels in sorted(by_model.items()):
    names = sorted(labels)
    if len(names) < 2:
        continue
    a, b = names[0], names[-1]
    common = sorted(set(labels[a]) & set(labels[b]))
    diff = [i for i in common if labels[a][i] != labels[b][i]]
    print(f"\n{model}: {len(common)} items compared between '{a}' and '{b}', {len(diff)} differ")
    for i in diff:
        print(f"  {i}\n    {a}: {labels[a][i]}\n    {b}: {labels[b][i]}")
