#!/usr/bin/env python3
"""Merge arXiv-paste rows with judge verdicts into per-model documents for the
benchmark page's database (docs are prefixed arxiv-, plus arxiv-status).

  python3 bench/push-arxiv-db.py RESULTS JUDGED bench/arxiv-pastes.json OUT_DIR
"""
import json, sys, os, collections, datetime, random
rows_path, judged_path, data_path, out_dir = sys.argv[1:5]
os.makedirs(out_dir, exist_ok=True)
rows = json.load(open(rows_path))
judged = json.load(open(judged_path)) if os.path.exists(judged_path) else []
items = {i["id"]: i for i in json.load(open(data_path))}
verdict = {(r["approach"], r["item"]): r for r in judged if "correct" in r}
KEYS = {"qwen3.5:0.8b": "arxiv-qwen3_5-0_8b", "hf.co/openbmb/MiniCPM5-2B-GGUF:q4_K_M": "arxiv-minicpm5-2b", "qwen3.5:4b": "arxiv-qwen3_5-4b",
        "qwen3.5:0.8b+normpdf": "arxiv-qwen3_5-0_8b-normpdf", "hf.co/openbmb/MiniCPM5-2B-GGUF:q4_K_M+normpdf": "arxiv-minicpm5-2b-normpdf", "qwen3.5:4b+normpdf": "arxiv-qwen3_5-4b-normpdf"}

def group(item):
    m = item.get("meta", {})
    disp = "display" if m.get("n_display", 0) else "inline only"
    ext = "-layout" if "layout" in (m.get("extractor") or "") else "default"
    return f"{disp} / {ext}"

per = collections.defaultdict(list)
for r in rows:
    if r.get("ms", -1) < 0 or r.get("model") not in KEYS or r["item"] not in items: continue
    v = verdict.get((r["approach"], r["item"]))
    per[r["model"]].append({"item": r["item"], "group": group(items[r["item"]]), "ms": r["ms"], "valid": bool(r.get("valid")),
                            "issues": "; ".join(r.get("issues", []))[:200], "out": (r.get("output") or "")[:700],
                            "gated": r.get("gated"), "correct": (None if v is None else bool(v["correct"])), "why": (v or {}).get("why", "")[:120]})
for model, rs in per.items():
    json.dump({"model": model, "updated": datetime.datetime.now().isoformat(timespec="seconds"), "rows": rs},
              open(os.path.join(out_dir, KEYS[model] + ".json"), "w"), ensure_ascii=False)
groups = collections.Counter(group(i) for i in items.values())
rng = random.Random(0)
samples = [{"id": i["id"], "source": i["source"], "input": i["input"][:700], "reference": i["reference"][:700]} for i in rng.sample(list(items.values()), 3)]
json.dump({"updated": datetime.datetime.now().isoformat(timespec="seconds"), "items": len(items), "papers": len({i["id"].split("-p")[0] for i in items}),
           "groups": dict(groups), "samples": samples}, open(os.path.join(out_dir, "arxiv-status.json"), "w"), ensure_ascii=False)
print(f"{sum(len(v) for v in per.values())} rows, {len(per)} model docs, groups {dict(groups)} -> {out_dir}")
