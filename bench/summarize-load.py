#!/usr/bin/env python3
"""Summarize rows from public/bench-load.html (tier "load").

    curl -s http://localhost:8000/api/bench > bench/results-load-YYYY-MM-DD.json
    python3 bench/summarize-load.py bench/results-load-YYYY-MM-DD.json [more.json]

Prints, per (label, model, device, dtype, src, mode): runs, bytes, median load
time (cold: download + session; warm: Cache API read + session), first/second
generation, and T4-T0 (load + warmup + first gen). WebNN rows also print
per-graph constantsMs/emitMs/buildMs.
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

mb = lambda b: f"{(b or 0) / 2**20:.0f} MB"
med = lambda xs: statistics.median(xs) if xs else 0

print("| label | model | runtime | source | mode | runs | bytes | load | download | session | warmup | first gen | second gen | T4-T0 | loadavg1 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for k in sorted(groups):
    g = [r for r in groups[k] if "loadMs" in r]
    errs = len(groups[k]) - len(g)
    if not g:
        print(f"| {k[0]} | {k[1]} | {k[2]} {k[3]} | {k[4]} | {k[5]} | 0 (+{errs} failed) | | | | | | | | |")
        continue
    loads = [r["loadMs"] for r in g]
    loadavg = med([(r.get("host") or r.get("runnerHost", {}).get("after") or {}).get("loadavg", [0])[0] for r in g])
    t4 = med([r.get("coldToFirstMs", r["loadMs"] + r.get("warmupMs", 0) + r.get("firstGenMs", 0)) for r in g])
    print(f"| {k[0]} | {k[1]} | {k[2]} {k[3]} | {k[4]} | {k[5]} | {len(g)}{f' (+{errs} failed)' if errs else ''} | {mb(g[0].get('totalBytes'))} | "
          f"{med(loads)/1000:.1f} s | {med([r.get('downloadMs', 0) for r in g])/1000:.1f} s | "
          f"{med([r.get('sessionMs', 0) for r in g])/1000:.1f} s | {med([r.get('warmupMs', 0) for r in g])/1000:.2f} s | "
          f"{med([r.get('firstGenMs', 0) for r in g])/1000:.2f} s | {med([r.get('secondGenMs', 0) for r in g])/1000:.2f} s | "
          f"{t4/1000:.1f} s | {loadavg:.2f} |")
    graphs = g[0].get("graphs") or []
    for gr in graphs:
        print(f"|  · {gr['name']} | | | | | | {mb(gr.get('constantBytes'))} | "
              f"{(gr.get('constantsMs') or 0)/1000:.2f} s constants | {(gr.get('emitMs') or 0)/1000:.2f} s emit | {(gr.get('buildMs') or 0)/1000:.2f} s build | | | | | |")

by_model = defaultdict(dict)
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
