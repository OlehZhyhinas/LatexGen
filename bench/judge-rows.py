#!/usr/bin/env python3
"""Judge result rows against references and print payoff tables."""
import argparse
import collections
import json
import os
import pathlib
import re
import statistics
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
PDF_PASTES = BENCH / "pdf-pastes.json"
SYNTH_SPANS = BENCH / "synth-spans.jsonl"
BENCH_DATA = BENCH / "bench-data.json"

MODEL = "qwen-local"
JUDGE_PROMPT = """You judge text-to-LaTeX conversions. Given the original text pasted from a PDF, a REFERENCE LaTeX answer, and a CANDIDATE output, decide if the candidate is mathematically equivalent to the reference (notation differences like \\frac vs \\dfrac, delimiter style, spacing, $$ vs \\[, or equivalent orderings are all fine). For multiline passages, the candidate must also keep the prose meaning and convert ALL the math. An empty candidate, or one that drops or mangles part of the math, is wrong.
Reply with ONLY a JSON object: {"correct": true/false, "reason": "<max 12 words>"}"""


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--data", default="auto")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", action="store_true")
    return ap.parse_args()


def read_json(path):
    with open(path) as f:
        return json.load(f)


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_item_map(data_mode):
    if data_mode != "auto":
        rows = read_json(data_mode)
        return {r["id"]: r for r in rows}

    out = {}
    for r in read_json(PDF_PASTES):
        out[r["id"]] = {
            "id": r["id"],
            "input": r["input"],
            "reference": r["reference"],
            "tier": "pdf-paste",
            "profile": r.get("profile"),
        }
    tier_map = {"pdf": "synth-pdf", "unicode": "synth-unicode", "latex": "synth-latex"}
    for r in read_jsonl(SYNTH_SPANS):
        tier = tier_map.get(r.get("kind"))
        if not tier:
            continue
        out[r["id"]] = {
            "id": r["id"],
            "input": r["text"],
            "reference": r.get("reference", ""),
            "tier": tier,
            "profile": r.get("profile"),
        }
    for r in read_json(BENCH_DATA):
        out[r["id"]] = {
            "id": r["id"],
            "input": r["input"],
            "reference": r["reference"],
            "tier": r.get("tier", "legacy"),
            "profile": None,
        }
    return out


def ask(payload):
    base = (os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/api/chat")
    req = urllib.request.Request(
        base,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        payload = e.read().decode("utf-8", "ignore")
        raise RuntimeError(f"HTTP {e.code}: {payload[:200]}") from e


def parse_verdict(text):
    a = text.find("{")
    b = text.rfind("}")
    if a < 0 or b < a:
        return False, f"unparseable verdict: {text[:80]}"
    try:
        v = json.loads(text[a : b + 1])
    except Exception:
        return False, f"unparseable verdict: {text[:80]}"
    return bool(v.get("correct")), str(v.get("reason", ""))


def judge_one(inp, ref, cand):
    if not str(cand).strip():
        return False, "empty output"
    body = {
        "model": MODEL,
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": {"temperature": 0.0, "num_predict": 120},
        "messages": [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"INPUT:\n{inp}\n\nREFERENCE:\n{ref}\n\nCANDIDATE:\n{cand}"},
        ],
    }
    txt = (ask(body).get("message") or {}).get("content", "").strip()
    return parse_verdict(txt)


def load_prior(out_path):
    p = pathlib.Path(out_path)
    if not p.exists():
        return {}
    try:
        rows = read_json(p)
    except Exception:
        return {}
    out = {}
    for r in rows:
        if "correct" in r:
            out[(r.get("approach"), r.get("item"))] = r
    return out


def save_rows(path, rows):
    with open(path, "w") as f:
        json.dump(rows, f, indent=1)


def median_secs(rows):
    vals = [r["ms"] for r in rows if isinstance(r.get("ms"), (int, float)) and r.get("ms", -1) >= 0]
    if not vals:
        return None
    return statistics.median(vals) / 1000.0


def stats_cell(rows):
    if not rows:
        return None
    acc = sum(1 for r in rows if r.get("correct") is True) / len(rows)
    med = median_secs(rows)
    if med is None:
        return f"{acc:.0%} @ -"
    return f"{acc:.0%} @ {med:.2f}s"


def pick_rows(rows, col):
    if col == "pdf-paste/clean":
        return [r for r in rows if r.get("tier") == "pdf-paste" and r.get("profile") == "clean"]
    if col == "pdf-paste/dirty":
        return [r for r in rows if r.get("tier") == "pdf-paste" and r.get("profile") == "dirty"]
    return [r for r in rows if r.get("tier") == col]


def columns_present(rows):
    base = [
        "pdf-paste",
        "pdf-paste/clean",
        "pdf-paste/dirty",
        "synth-pdf",
        "synth-unicode",
        "synth-latex",
        "easy",
        "medium",
        "hard",
        "multiline",
    ]
    out = []
    for col in base:
        if pick_rows(rows, col):
            out.append(col)
    return out


def print_matrix(rows, md=False):
    by = collections.defaultdict(list)
    for r in rows:
        by[r.get("approach", "")].append(r)
    cols = columns_present(rows)
    if md:
        print("| approach | " + " | ".join(cols) + " |")
        print("|---|" + "|".join("---" for _ in cols) + "|")
        for ap in sorted(by):
            vals = [stats_cell(pick_rows(by[ap], c)) or "—" for c in cols]
            print("| " + ap + " | " + " | ".join(vals) + " |")
    else:
        print("\n" + "approach".ljust(34) + "".join(c.rjust(20) for c in cols))
        for ap in sorted(by):
            line = ap.ljust(34)
            for c in cols:
                line += (stats_cell(pick_rows(by[ap], c)) or "—").rjust(20)
            print(line)


def print_pdf_cost(rows, md=False):
    pdf_rows = [r for r in rows if r.get("tier") == "pdf-paste"]
    by = collections.defaultdict(list)
    for r in pdf_rows:
        by[r.get("approach", "")].append(r)

    def med_num(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        return statistics.median(vals) if vals else None

    if md:
        print("\n| approach | median evalTokens | median requests |")
        print("|---|---:|---:|")
        for ap in sorted(by):
            rs = by[ap]
            m_eval = med_num([r.get("evalTokens") for r in rs])
            m_req = med_num([r.get("requests", 1) for r in rs])
            eval_txt = f"{m_eval:.1f}" if m_eval is not None else "—"
            req_txt = f"{m_req:.1f}" if m_req is not None else "—"
            print(f"| {ap} | {eval_txt} | {req_txt} |")
    else:
        print("\n" + "approach".ljust(34) + "median evalTokens".rjust(20) + "median requests".rjust(18))
        for ap in sorted(by):
            rs = by[ap]
            m_eval = med_num([r.get("evalTokens") for r in rs])
            m_req = med_num([r.get("requests", 1) for r in rs])
            eval_txt = f"{m_eval:.1f}" if m_eval is not None else "—"
            req_txt = f"{m_req:.1f}" if m_req is not None else "—"
            print(ap.ljust(34) + eval_txt.rjust(20) + req_txt.rjust(18))


def main():
    args = parse_args()
    rows = read_json(args.rows)
    items = load_item_map(args.data)
    prior = load_prior(args.out)
    judged = []
    new_count = 0

    for idx, r in enumerate(rows, 1):
        key = (r.get("approach"), r.get("item"))
        if key in prior:
            judged.append(prior[key])
            continue
        item = items.get(r.get("item"))
        if not item:
            out = dict(r)
            out["correct"] = False
            out["why"] = "missing reference"
            judged.append(out)
            print(f"[{idx}/{len(rows)}] {r.get('approach','')[:28]:28s} {r.get('item','')[:28]:28s} BAD missing reference", file=sys.stderr)
            continue

        ok, why = judge_one(item["input"], item["reference"], r.get("output", ""))
        out = dict(r)
        out["correct"] = ok
        out["why"] = why
        out["tier"] = out.get("tier") or item.get("tier")
        if item.get("profile") is not None:
            out["profile"] = item.get("profile")
        judged.append(out)
        new_count += 1
        print(
            f"[{idx}/{len(rows)}] {out.get('approach','')[:28]:28s} {out.get('item','')[:28]:28s} {'OK ' if ok else 'BAD'} {why}",
            file=sys.stderr,
        )
        if new_count % 10 == 0:
            save_rows(args.out, judged)

    save_rows(args.out, judged)
    print_matrix(judged, md=args.md)
    print_pdf_cost(judged, md=args.md)


if __name__ == "__main__":
    main()
