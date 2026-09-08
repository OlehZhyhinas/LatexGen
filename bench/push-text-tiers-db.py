#!/usr/bin/env python3
"""Merge generated rows with judge verdicts into one JSON document per model,
ready to be written into the benchmark page's shared database.

  python3 bench/push-text-tiers-db.py bench/results-text-tiers-2026-09-07.json bench/judged-text-tiers-2026-09-07.json bench/bench-data-extended.json OUT_DIR
Writes OUT_DIR/<key>.json per model and OUT_DIR/status.json.
"""
import json, sys, os, collections, datetime
rows_path, judged_path, data_path, out_dir = sys.argv[1:5]
os.makedirs(out_dir, exist_ok=True)
rows = json.load(open(rows_path))
judged = json.load(open(judged_path)) if os.path.exists(judged_path) else []
items = json.load(open(data_path))
verdict = {(r["approach"], r["item"]): r for r in judged if "correct" in r}
KEYS = {"qwen3:0.6b": "qwen3-0_6b", "qwen3.5:0.8b": "qwen3_5-0_8b", "gemma3:1b": "gemma3-1b", "openbmb/minicpm5:q4_K_M": "minicpm5-1b",
        "qwen3:1.7b": "qwen3-1_7b", "qwen3.5:2b-q4_K_M": "qwen3_5-2b", "hf.co/openbmb/MiniCPM5-2B-GGUF:q4_K_M": "minicpm5-2b", "gemma4:e2b": "gemma4-e2b",
        "qwen3:4b": "qwen3-4b", "hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M": "qwen3-4b", "qwen3-4b-orig": "qwen3-4b", "qwen3.5:4b": "qwen3_5-4b", "gemma3:4b": "gemma3-4b", "gemma4:e4b": "gemma4-e4b",
        "qwen3.5:0.8b+norm": "qwen3_5-0_8b-norm", "hf.co/openbmb/MiniCPM5-2B-GGUF:q4_K_M+norm": "minicpm5-2b-norm", "qwen3.5:4b+norm": "qwen3_5-4b-norm", "qwen3.5:0.8b+normpdf": "qwen3_5-0_8b-normpdf", "hf.co/openbmb/MiniCPM5-2B-GGUF:q4_K_M+normpdf": "minicpm5-2b-normpdf", "qwen3.5:4b+normpdf": "qwen3_5-4b-normpdf"}
per = collections.defaultdict(list)
for r in rows:
    if r.get("ms", -1) < 0 or r.get("model") not in KEYS: continue
    v = verdict.get((r["approach"], r["item"]))
    per[r["model"]].append({"item": r["item"], "tier": r["tier"], "ms": r["ms"], "valid": bool(r.get("valid")),
                            "issues": "; ".join(r.get("issues", []))[:200], "out": (r.get("output") or "")[:700],
                            "correct": (None if v is None else bool(v["correct"])), "why": (v or {}).get("why", "")[:120]})
n_gen = n_judged = 0
for model, rs in per.items():
    n_gen += len(rs); n_judged += sum(r["correct"] is not None for r in rs)
    json.dump({"model": model, "updated": datetime.datetime.now().isoformat(timespec="seconds"), "rows": rs},
              open(os.path.join(out_dir, KEYS[model] + ".json"), "w"), ensure_ascii=False)
json.dump({"updated": datetime.datetime.now().isoformat(timespec="seconds"), "generated": n_gen, "judged": n_judged,
           "models": len(set(KEYS.keys())), "items": len(items), "tiers": dict(collections.Counter(i["tier"] for i in items)),
           "inputs": {i["id"]: i["input"][:600] for i in items if i["id"] in ("mechanics-homework-passage", "euler-formula-passage-pdf-dirty", "riemann-zeta-functional-equation")}},
          open(os.path.join(out_dir, "status.json"), "w"), ensure_ascii=False)
print(f"{n_gen} generated, {n_judged} judged, {len(per)} model docs -> {out_dir}")
