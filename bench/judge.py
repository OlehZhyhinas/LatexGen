#!/usr/bin/env python3
"""Judge benchmark outputs against references using the local 27B via Ollama,
then print the payoff matrix."""
import json, sys, urllib.request, statistics, collections

BENCH_URL = "http://localhost:8013/api/bench"
OLLAMA = "http://localhost:11434/api/chat"
MODEL = "qwen-local"

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
    # Rows come from the dev server's collector by default; `--rows FILE`
    # judges a saved run instead (bench/run-ollama-text.mjs writes one).
    args = sys.argv[1:]
    rows_file = None
    if "--rows" in args:
        i = args.index("--rows"); rows_file = args[i + 1]; del args[i:i + 2]
    here = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
    data_file = f"{here}/bench-data.json"
    if "--data" in args:
        i = args.index("--data"); data_file = args[i + 1]; del args[i:i + 2]
    rows = json.load(open(rows_file)) if rows_file else json.load(urllib.request.urlopen(BENCH_URL))
    items = {i["id"]: i for i in json.load(open(data_file))}
    rows = [r for r in rows if r["item"] in items]
    print(f"{len(rows)} rows to judge", file=sys.stderr)

    judged = []
    # Resume: rows that already carry a verdict (from a previous judge pass) are kept as-is.
    prior = {}
    out_path = args[0] if args else "judged.json"
    if __import__("os").path.exists(out_path):
        for r in json.load(open(out_path)):
            if "correct" in r: prior[(r["approach"], r["item"])] = r
    for n, r in enumerate(rows):
        if (r["approach"], r["item"]) in prior:
            judged.append(prior[(r["approach"], r["item"])]); continue
        it = items[r["item"]]
        ok, why = judge(it["input"], it["reference"], r.get("output", ""))
        judged.append({**r, "correct": ok, "why": why})
        print(f"[{n+1}/{len(rows)}] {r['approach']:28s} {r['item']:22s} {'OK ' if ok else 'BAD'} {why}", file=sys.stderr)
        if n % 10 == 0:  # partial verdicts are useful while a long pass runs
            json.dump(judged, open(out_path, "w"), indent=1)

    with open(out_path, "w") as f:
        json.dump(judged, f, indent=1)

    # payoff matrix
    tiers = [t for t in ["easy", "medium", "hard", "multiline", "pdf-paste", "synth-pdf", "synth-unicode", "synth-latex"] if any(r["tier"] == t for r in judged)]
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
