#!/usr/bin/env python3
"""Judge benchmark outputs against references using the local 27B via Ollama,
then print the payoff matrix."""
import json, sys, urllib.request, statistics, collections

BENCH_URL = "http://localhost:8013/api/bench"
OLLAMA = "http://localhost:11434/api/chat"
MODEL = "qwen3.8:27b-q8_0"

JUDGE_PROMPT = """You judge text-to-LaTeX conversions. Given the original spoken-English input, a REFERENCE LaTeX answer, and a CANDIDATE output, decide if the candidate is mathematically equivalent to the reference (notation differences like \\frac vs \\dfrac, delimiter style, spacing, $$ vs \\[, or equivalent orderings are all fine). For multiline passages, the candidate must also keep the prose meaning and convert ALL the math. An empty candidate, or one that drops or mangles part of the math, is wrong.
Reply with ONLY a JSON object: {"correct": true/false, "reason": "<max 12 words>"}"""

def ask(payload):
    req = urllib.request.Request(OLLAMA, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)

def judge(inp, ref, cand):
    if not cand.strip():
        return False, "empty output"
    body = {
        "model": MODEL, "stream": False, "think": False, "keep_alive": -1,
        "options": {"temperature": 0.0, "num_predict": 120},
        "messages": [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"INPUT:\n{inp}\n\nREFERENCE:\n{ref}\n\nCANDIDATE:\n{cand}"},
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
    items = {i["id"]: i for i in json.load(open("/Users/oleh/personal/latexgen/bench-data.json"))}
    rows = [r for r in rows if r["item"] in items]
    print(f"{len(rows)} rows to judge", file=sys.stderr)

    judged = []
    for n, r in enumerate(rows):
        it = items[r["item"]]
        ok, why = judge(it["input"], it["reference"], r.get("output", ""))
        judged.append({**r, "correct": ok, "why": why})
        print(f"[{n+1}/{len(rows)}] {r['approach']:28s} {r['item']:22s} {'OK ' if ok else 'BAD'} {why}", file=sys.stderr)

    with open(sys.argv[1] if len(sys.argv) > 1 else "judged.json", "w") as f:
        json.dump(judged, f, indent=1)

    # payoff matrix
    tiers = ["easy", "medium", "hard", "multiline"]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in judged:
        by[r["approach"]][r["tier"]].append(r)
    print(f"\n{'approach':30s}" + "".join(f"{t:>22s}" for t in tiers))
    for ap in sorted(by):
        line = f"{ap:30s}"
        for t in tiers:
            rs = by[ap][t]
            if not rs:
                line += f"{'—':>22s}"
            else:
                acc = sum(r["correct"] for r in rs) / len(rs)
                lat = statistics.median([r["ms"] for r in rs if r["ms"] >= 0] or [-1])
                line += f"{f'{acc:.0%} @ {lat/1000:.1f}s':>22s}"
        print(line)

if __name__ == "__main__":
    main()
