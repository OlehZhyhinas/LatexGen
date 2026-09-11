#!/usr/bin/env python3
"""Judge divergent prompt-lookup A/B rows with the local Ollama judge model."""

import argparse
import json
import math
import os
import pathlib
import sys
import urllib.error
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "bench" / "results-prompt-lookup-judge.json"
ARXIV_PASTES = pathlib.Path(
    "/Users/oleh/personal/webnn-workbench-prompt-lookup/bench/latexgen-snapshot/bench/arxiv-pastes.json"
)
AB_FILES = [
    pathlib.Path(
        "/Users/oleh/personal/webnn-workbench-prompt-lookup/bench/results/prompt-lookup-ab-qwen35-4b-arxiv-k10.json"
    ),
    pathlib.Path(
        "/Users/oleh/personal/webnn-workbench-prompt-lookup/bench/results/prompt-lookup-ab-qwen35-4b-arxiv-k5.json"
    ),
    pathlib.Path(
        "/Users/oleh/personal/webnn-workbench-prompt-lookup/bench/results/prompt-lookup-ab-minicpm5-2b-arxiv-k5.json"
    ),
]

MODEL = "qwen-local"
# Verbatim from bench/latexgen-snapshot/bench/judge-rows.py.
JUDGE_PROMPT = """You judge text-to-LaTeX conversions. Given the original text pasted from a PDF, a REFERENCE LaTeX answer, and a CANDIDATE output, decide if the candidate is mathematically equivalent to the reference (notation differences like \\frac vs \\dfrac, delimiter style, spacing, $$ vs \\[, or equivalent orderings are all fine). For multiline passages, the candidate must also keep the prose meaning and convert ALL the math. An empty candidate, or one that drops or mangles part of the math, is wrong.
Reply with ONLY a JSON object: {"correct": true/false, "reason": "<max 12 words>"}"""


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--pastes", default=str(ARXIV_PASTES))
    ap.add_argument("--files", nargs="*", default=[str(p) for p in AB_FILES])
    return ap.parse_args()


def read_json(path):
    with open(path) as f:
        return json.load(f)


def ask(payload):
    base = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/api/chat"
    req = urllib.request.Request(
        base,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def parse_verdict_text(text):
    a = text.find("{")
    b = text.rfind("}")
    if a < 0 or b < a:
        return {
            "status": "unparsed",
            "correct": False,
            "reason": f"unparseable verdict: {text[:80]}",
        }
    try:
        v = json.loads(text[a : b + 1])
    except Exception:
        return {
            "status": "unparsed",
            "correct": False,
            "reason": f"unparseable verdict: {text[:80]}",
        }
    correct = bool(v.get("correct"))
    reason = str(v.get("reason", ""))
    return {
        "status": "correct" if correct else "incorrect",
        "correct": correct,
        "reason": reason,
    }


def judge_once(inp, ref, cand):
    if not str(cand).strip():
        return {
            "status": "incorrect",
            "correct": False,
            "reason": "empty output",
            "raw": "",
        }
    # judge-rows.py does not set a seed and uses temperature=0.0; mirrored exactly.
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
    try:
        txt = (ask(body).get("message") or {}).get("content", "").strip()
    except urllib.error.HTTPError as e:
        payload = e.read().decode("utf-8", "ignore")
        msg = f"HTTP {e.code}: {payload[:200]}"
        return {"status": "unparsed", "correct": False, "reason": msg, "raw": msg}
    except Exception as e:  # noqa: BLE001
        msg = f"request error: {e}"
        return {"status": "unparsed", "correct": False, "reason": msg[:200], "raw": msg}

    out = parse_verdict_text(txt)
    out["raw"] = txt
    return out


def mode_reason(repeats, majority_correct):
    target = "correct" if majority_correct else "incorrect"
    counts = {}
    order = []
    for rep in repeats:
        rep_status = "correct" if rep.get("correct") else "incorrect"
        if rep_status != target:
            continue
        reason = str(rep.get("reason", "")).strip()
        if not reason:
            continue
        if reason not in counts:
            counts[reason] = 0
            order.append(reason)
        counts[reason] += 1
    if not counts:
        for rep in repeats:
            reason = str(rep.get("reason", "")).strip()
            if reason:
                return reason
        return ""
    best = max(order, key=lambda r: counts[r])
    return best


def judge_three(inp, ref, cand):
    repeats = [judge_once(inp, ref, cand) for _ in range(3)]
    correct_votes = sum(1 for r in repeats if r.get("correct") is True)
    majority_correct = correct_votes >= 2
    return {
        "repeats": repeats,
        "correctVotes": correct_votes,
        "majorityVerdict": "correct" if majority_correct else "incorrect",
        "majorityCorrect": majority_correct,
        "majorityReason": mode_reason(repeats, majority_correct),
    }


def classify(base_correct, treatment_correct):
    if base_correct and not treatment_correct:
        return "treatment_worse"
    if treatment_correct and not base_correct:
        return "treatment_better"
    if base_correct and treatment_correct:
        return "both_correct"
    return "both_wrong"


def exact_binom_two_sided(k, n):
    if n <= 0:
        return 1.0
    denom = 2**n
    lower = sum(math.comb(n, i) for i in range(0, k + 1)) / denom
    upper = sum(math.comb(n, i) for i in range(k, n + 1)) / denom
    return min(1.0, 2.0 * min(lower, upper))


def summarize_reason(item):
    cls = item["classification"]
    b = item["base"]["majorityReason"].strip()
    t = item["treatment"]["majorityReason"].strip()
    if cls == "treatment_worse":
        return t or b or "treatment judged incorrect"
    if cls == "treatment_better":
        return b or t or "baseline judged incorrect"
    if cls == "both_correct":
        return t or b or "both judged equivalent"
    if b and t and b != t:
        return f"base: {b}; trt: {t}"[:120]
    return b or t or "both judged incorrect"


def process_file(path, item_map):
    data = read_json(path)
    rows = data.get("rows", [])
    judged_items = []
    skipped_identical = 0

    for idx, row in enumerate(rows, 1):
        item_id = row.get("id", "")
        identical = row.get("identicalStripped")
        base_stripped = str((row.get("base") or {}).get("stripped", ""))
        treatment_stripped = str((row.get("treatment") or {}).get("stripped", ""))

        if identical is True:
            skipped_identical += 1
            continue
        if identical is None and base_stripped == treatment_stripped:
            skipped_identical += 1
            continue

        src = item_map.get(item_id)
        if not src:
            base_eval = {
                "repeats": [{"status": "unparsed", "correct": False, "reason": "missing reference", "raw": ""}] * 3,
                "correctVotes": 0,
                "majorityVerdict": "incorrect",
                "majorityCorrect": False,
                "majorityReason": "missing reference",
            }
            treatment_eval = {
                "repeats": [{"status": "unparsed", "correct": False, "reason": "missing reference", "raw": ""}] * 3,
                "correctVotes": 0,
                "majorityVerdict": "incorrect",
                "majorityCorrect": False,
                "majorityReason": "missing reference",
            }
        else:
            base_eval = judge_three(src.get("input", ""), src.get("reference", ""), base_stripped)
            treatment_eval = judge_three(src.get("input", ""), src.get("reference", ""), treatment_stripped)

        item = {
            "id": item_id,
            "base": base_eval,
            "treatment": treatment_eval,
            "classification": classify(base_eval["majorityCorrect"], treatment_eval["majorityCorrect"]),
        }
        item["shortReason"] = summarize_reason(item)
        judged_items.append(item)
        print(
            f"[{idx}/{len(rows)}] {path.name} {item_id} "
            f"base={base_eval['correctVotes']}/3 trt={treatment_eval['correctVotes']}/3 {item['classification']}",
            file=sys.stderr,
        )

    class_counts = {
        "treatment_worse": sum(1 for x in judged_items if x["classification"] == "treatment_worse"),
        "treatment_better": sum(1 for x in judged_items if x["classification"] == "treatment_better"),
        "both_correct": sum(1 for x in judged_items if x["classification"] == "both_correct"),
        "both_wrong": sum(1 for x in judged_items if x["classification"] == "both_wrong"),
    }
    discordant = class_counts["treatment_better"] + class_counts["treatment_worse"]
    p_value = exact_binom_two_sided(class_counts["treatment_better"], discordant)

    return {
        "file": str(path),
        "label": data.get("label", path.stem),
        "modelSlug": data.get("modelSlug"),
        "judgedCount": len(judged_items),
        "skippedIdenticalCount": skipped_identical,
        "itemsJudged": judged_items,
        "totals": {
            **class_counts,
            "discordantPairs": discordant,
            "signTestPValueTwoSided": p_value,
            "baseMajorityCorrect": sum(1 for x in judged_items if x["base"]["majorityCorrect"]),
            "treatmentMajorityCorrect": sum(1 for x in judged_items if x["treatment"]["majorityCorrect"]),
        },
    }


def print_markdown_tables(results):
    for res in results:
        totals = res["totals"]
        print(f"\n### {res['label']}")
        print("| item | base verdict (x/3) | treatment verdict (x/3) | class | reason |")
        print("|---|---|---|---|---|")
        for item in res["itemsJudged"]:
            b = item["base"]
            t = item["treatment"]
            b_cell = f"{b['majorityVerdict']} ({b['correctVotes']}/3)"
            t_cell = f"{t['majorityVerdict']} ({t['correctVotes']}/3)"
            print(f"| {item['id']} | {b_cell} | {t_cell} | {item['classification']} | {item['shortReason']} |")
        totals_class = (
            f"better={totals['treatment_better']}, worse={totals['treatment_worse']}, "
            f"both_correct={totals['both_correct']}, both_wrong={totals['both_wrong']}"
        )
        totals_reason = (
            f"judged={res['judgedCount']}, skipped_identical={res['skippedIdenticalCount']}, "
            f"discordant={totals['discordantPairs']}, p={totals['signTestPValueTwoSided']:.6g}"
        )
        print(f"| totals | - | - | {totals_class} | {totals_reason} |")


def main():
    args = parse_args()
    item_map = {r["id"]: r for r in read_json(args.pastes)}
    results = [process_file(pathlib.Path(p), item_map) for p in args.files]

    out = {
        "judgeModel": MODEL,
        "judgePromptSource": "/Users/oleh/personal/webnn-workbench-prompt-lookup/bench/latexgen-snapshot/bench/judge-rows.py",
        "seedNote": "No seed is set in reference judge-rows.py and temperature is 0.0; requests are repeated unchanged.",
        "files": results,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print_markdown_tables(results)


if __name__ == "__main__":
    main()
