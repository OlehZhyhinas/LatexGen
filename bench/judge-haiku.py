#!/usr/bin/env python3
"""Judge benchmark rows with Claude Haiku through the Claude Code CLI
(`claude -p --model haiku`), in parallel. Same prompt and verdict format as
judge.py, so summaries and pages read either output.

  python3 bench/judge-haiku.py --rows ROWS.json --data bench/bench-data-extended.json OUT.json [--workers 6]

Resume-aware: rows already carrying a verdict in OUT.json are kept.
"""
import json, sys, os, subprocess, collections, statistics, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge import JUDGE_PROMPT  # noqa: E402

def judge_one(inp, ref, cand):
    if not cand.strip():
        return False, "empty output"
    prompt = f"{JUDGE_PROMPT}\n\nINPUT:\n{inp}\n\nREFERENCE:\n{ref}\n\nCANDIDATE:\n{cand}"
    env = {**os.environ, "PATH": os.path.expanduser("~/.local/bin") + ":" + os.environ.get("PATH", "")}
    for attempt in range(3):
        try:
            r = subprocess.run(["claude", "-p", "--model", "haiku", "--output-format", "text", "--tools", ""],
                               input=prompt, capture_output=True, text=True, timeout=120, env=env)
            txt = r.stdout.strip()
            txt = txt[txt.find("{"):txt.rfind("}") + 1]
            v = json.loads(txt)
            return bool(v.get("correct")), str(v.get("reason", ""))[:120]
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:60]}"
    return False, f"judge failed: {last}"

def main():
    args = sys.argv[1:]
    def opt(name, dflt=None):
        if name in args:
            i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
        return dflt
    rows_file = opt("--rows"); data_file = opt("--data", os.path.join(os.path.dirname(__file__), "bench-data.json"))
    workers = int(opt("--workers", "6")); out_path = args[0] if args else "judged.json"
    rows = json.load(open(rows_file))
    items = {i["id"]: i for i in json.load(open(data_file))}
    rows = [r for r in rows if r["item"] in items and r.get("ms", -1) >= 0]
    prior = {}
    if os.path.exists(out_path):
        for r in json.load(open(out_path)):
            if "correct" in r: prior[(r["approach"], r["item"])] = r
    todo = [r for r in rows if (r["approach"], r["item"]) not in prior]
    print(f"{len(rows)} rows, {len(prior)} already judged, {len(todo)} to judge with {workers} workers", file=sys.stderr)
    judged = list(prior.values()); lock = threading.Lock(); n = 0
    def work(r):
        it = items[r["item"]]
        ok, why = judge_one(it["input"], it["reference"], r.get("output", ""))
        return {**r, "correct": ok, "why": why, "judge": "claude-haiku-4-5"}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for f in as_completed(futs):
            res = f.result()
            with lock:
                judged.append(res); n += 1
                print(f"[{n}/{len(todo)}] {res['approach'][:34]:34s} {res['item'][:28]:28s} {'OK ' if res['correct'] else 'BAD'} {res['why']}", file=sys.stderr)
                if n % 5 == 0:
                    json.dump(judged, open(out_path, "w"), indent=1)
    json.dump(judged, open(out_path, "w"), indent=1)
    tiers = [t for t in ["easy", "medium", "hard", "multiline", "pdf-paste", "synth-pdf", "synth-unicode", "synth-latex"] if any(r["tier"] == t for r in judged)]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in judged: by[r["approach"]][r["tier"]].append(r)
    print(f"\n{'approach':40s}" + "".join(f"{t:>14s}" for t in tiers))
    for ap in sorted(by):
        line = f"{ap:40s}"
        for t in tiers:
            rs = by[ap][t]
            line += f"{'—':>14s}" if not rs else f"{sum(r['correct'] for r in rs)}/{len(rs):<3d}".rjust(14)
        print(line)

if __name__ == "__main__":
    main()
