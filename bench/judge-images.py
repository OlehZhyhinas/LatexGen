#!/usr/bin/env python3
"""Judge image-OCR benchmark outputs (Texify vs Texo) against the rendered
references using the local 27B, then print the payoff matrix."""
import json, sys, urllib.request, statistics, collections

BENCH_URL = "http://localhost:8013/api/bench"
OLLAMA = "http://localhost:11434/api/chat"
MODEL = "qwen3.8:27b-q8_0"
ITEMS_PATH = "/Users/oleh/personal/latexgen/public/bench-images.json"

JUDGE_PROMPT = """You judge math OCR: an image of rendered LaTeX was given to a model. You get the REFERENCE LaTeX that produced the image and the CANDIDATE the model read back. Decide if the candidate is mathematically the same expression (notation differences like \\frac vs \\dfrac, \\left/\\right, spacing commands, $$ delimiters, \\mathbf vs plain bold, \\varepsilon vs \\epsilon, or equivalent orderings are all fine). For prose items the candidate must keep the sentence meaning and ALL the math; dropping or mangling any part is wrong. Empty or garbled output is wrong.
Reply with ONLY a JSON object: {"correct": true/false, "reason": "<max 12 words>"}"""

def ask(payload):
    req = urllib.request.Request(OLLAMA, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)

def judge(ref, cand):
    if not cand.strip():
        return False, "empty output"
    body = {
        "model": MODEL, "stream": False, "think": False, "keep_alive": -1,
        "options": {"temperature": 0.0, "num_predict": 120},
        "messages": [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"REFERENCE:\n{ref}\n\nCANDIDATE:\n{cand}"},
        ],
    }
    txt = ask(body)["message"]["content"].strip()
    txt = txt[txt.find("{"):txt.rfind("}") + 1]
    try:
        v = json.loads(txt)
        return bool(v.get("correct")), v.get("reason", "")
    except Exception:
        return False, f"unparseable verdict: {txt[:50]}"

def main():
    rows = json.load(urllib.request.urlopen(BENCH_URL))
    items = {i["id"]: i for i in json.load(open(ITEMS_PATH))}
    loads = {r["approach"]: r["ms"] for r in rows if r["item"] == "__load__"}
    rows = [r for r in rows if r["item"] in items]
    print(f"{len(rows)} rows to judge; load times: {loads}", file=sys.stderr)

    judged = []
    for n, r in enumerate(rows):
        ok, why = judge(items[r["item"]]["reference"], r.get("output", ""))
        judged.append({**r, "correct": ok, "why": why})
        print(f"[{n+1}/{len(rows)}] {r['approach']:8s} {r['item']:20s} {'OK ' if ok else 'BAD'} {why}", file=sys.stderr)

    out = sys.argv[1] if len(sys.argv) > 1 else "judged-images.json"
    with open(out, "w") as f:
        json.dump({"loads": loads, "rows": judged}, f, indent=1)

    tiers = ["easy", "medium", "hard", "mixed", "degraded"]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in judged:
        by[r["approach"]][r["tier"]].append(r)
    print(f"\n{'approach':10s}{'load':>10s}" + "".join(f"{t:>20s}" for t in tiers))
    for ap in sorted(by):
        line = f"{ap:10s}{loads.get(ap, 0)/1000:>9.1f}s"
        for t in tiers:
            rs = by[ap][t]
            if not rs:
                line += f"{'—':>20s}"; continue
            acc = 100 * sum(r["correct"] for r in rs) / len(rs)
            med = statistics.median(r["ms"] for r in rs) / 1000
            line += f"{f'{acc:.0f}% @ {med:.1f}s':>20s}"
        print(line)

if __name__ == "__main__":
    main()
