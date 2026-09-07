#!/usr/bin/env python3
"""Gate B: measure prompt-lookup copy fraction on benchmark outputs."""
import argparse
import json
import re
from glob import glob
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
BENCH_PATH = ROOT / "bench-data.json"
PDF_PASTES_PATH = ROOT / "pdf-pastes.json"
RESULTS_PATH = ROOT / "results-2026-09-01.json"
OUT_PATH = ROOT / "results-copy-fraction.json"
TOKENIZER_GLOB = str(
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json"
)


def parse_args():
    parser = argparse.ArgumentParser()
    default_synth = ROOT / "synth-spans.jsonl"
    parser.add_argument(
        "--synth",
        default=str(default_synth) if default_synth.exists() else None,
        help="Optional JSONL synth spans file",
    )
    parser.add_argument(
        "--normalize-input",
        action="store_true",
        help="Use normalized input for the primary pass (normalized comparison rows are always computed)",
    )
    return parser.parse_args()


def load_tokenizer():
    paths = sorted(glob(TOKENIZER_GLOB))
    if not paths:
        raise SystemExit(f"tokenizer.json not found: {TOKENIZER_GLOB}")
    return Tokenizer.from_file(paths[-1]), paths[-1]


def tokenize(tok, text):
    return tok.encode(text, add_special_tokens=False).ids


def normalize_input_text(text):
    text = text.replace("\u200b", "")
    text = re.sub(r"(?<=\S)\n(?=\S)", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text


def last_occurrence_with_continuation(haystack, needle):
    n = len(needle)
    for i in range(len(haystack) - n, -1, -1):
        if haystack[i : i + n] == needle and i + n < len(haystack):
            return i
    return -1


def simulate_prompt_lookup(input_ids, output_ids, k):
    total = len(output_ids)
    i = 0
    accepted = 0
    steps = 0
    while i < total:
        steps += 1
        context = input_ids + output_ids[:i]
        pos = -1
        n_used = 0
        for n in (3, 2):
            if len(context) < n:
                continue
            needle = context[-n:]
            m = last_occurrence_with_continuation(context, needle)
            if m >= 0:
                pos = m
                n_used = n
                break

        accepted_now = 0
        if pos >= 0:
            draft = context[pos + n_used : pos + n_used + k]
            max_check = min(len(draft), total - i)
            while accepted_now < max_check and draft[accepted_now] == output_ids[i + accepted_now]:
                accepted_now += 1
        accepted += accepted_now
        emitted = min(total - i, accepted_now + 1)
        i += emitted
    return accepted, steps


def stats(xs):
    if not xs:
        return {"mean": 0.0, "median": 0.0}
    arr = np.array(xs, dtype=float)
    return {"mean": float(arr.mean()), "median": float(np.median(arr))}


def build_group_stats(rows):
    return {
        "n": len(rows),
        "k5": {
            "copy_fraction": stats([x["k5"]["copy_fraction"] for x in rows]),
            "speedup": stats([x["k5"]["speedup"] for x in rows]),
        },
        "k10": {
            "copy_fraction": stats([x["k10"]["copy_fraction"] for x in rows]),
            "speedup": stats([x["k10"]["speedup"] for x in rows]),
        },
        "in_input_token_share": stats([x["in_input_token_share"] for x in rows]),
    }


def print_pair_table(rows):
    print("\nPer-pair metrics")
    print(
        "| item | source | tier | out_toks | in_input_token_share | copy@5 | speedup@5 | copy@10 | speedup@10 |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['item']} | {r['source']} | {r['tier']} | {r['total_tokens']} | "
            f"{r['in_input_token_share']:.3f} | {r['k5']['copy_fraction']:.3f} | {r['k5']['speedup']:.3f} | "
            f"{r['k10']['copy_fraction']:.3f} | {r['k10']['speedup']:.3f} |"
        )


def print_summary_table(summary_raw, summary_norm):
    print("\nSummary (means / medians)")
    print("| group | n | copy@5 mean | copy@5 med | speedup@5 mean | speedup@5 med | copy@10 mean | copy@10 med | speedup@10 mean | speedup@10 med | in_input_token_share mean | in_input_token_share med |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    order = [
        "easy",
        "medium",
        "hard",
        "multiline",
        "pdf-paste",
        "pdf-paste/clean",
        "pdf-paste/dirty",
        "overall",
        "multiline_only",
    ]
    extras = [k for k in summary_raw.keys() if k not in set(order)]
    order.extend(sorted(extras))

    def _print_row(group_name, s):
        print(
            f"| {group_name} | {s['n']} | {s['k5']['copy_fraction']['mean']:.3f} | {s['k5']['copy_fraction']['median']:.3f} | "
            f"{s['k5']['speedup']['mean']:.3f} | {s['k5']['speedup']['median']:.3f} | "
            f"{s['k10']['copy_fraction']['mean']:.3f} | {s['k10']['copy_fraction']['median']:.3f} | "
            f"{s['k10']['speedup']['mean']:.3f} | {s['k10']['speedup']['median']:.3f} | "
            f"{s['in_input_token_share']['mean']:.3f} | {s['in_input_token_share']['median']:.3f} |"
        )

    for g in order:
        if g not in summary_raw:
            continue
        _print_row(g, summary_raw[g])
        if g in summary_norm:
            _print_row(f"{g} (norm)", summary_norm[g])


def build_summary(rows):
    tiers = sorted({r["tier"] for r in rows})
    groups = {tier: [r for r in rows if r["tier"] == tier] for tier in tiers}
    pdf_rows = [r for r in rows if r["tier"] == "pdf-paste"]
    for profile in ("clean", "dirty"):
        prof_rows = [r for r in pdf_rows if r.get("profile") == profile]
        if prof_rows:
            groups[f"pdf-paste/{profile}"] = prof_rows
    groups["overall"] = rows
    if "multiline" in groups:
        groups["multiline_only"] = groups["multiline"]
    out = {}
    for name, rs in groups.items():
        out[name] = build_group_stats(rs)
    return out


def print_synth_summary_table(summary_raw, summary_norm):
    print("\nSynth summary (means / medians)")
    print(
        "| slice | group | n | copy@5 mean | copy@5 med | speedup@5 mean | speedup@5 med | copy@10 mean | copy@10 med | speedup@10 mean | speedup@10 med | in_input_token_share mean | in_input_token_share med |"
    )
    print(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = [("overall", "overall", summary_raw["overall"])]

    preferred_kind_order = ["pdf", "unicode", "latex", "mixed", "none"]
    remaining_kinds = [
        k for k in summary_raw["by_kind"].keys() if k not in set(preferred_kind_order)
    ]
    for k in preferred_kind_order + sorted(remaining_kinds):
        if k in summary_raw["by_kind"]:
            rows.append(("dominant_kind", k, summary_raw["by_kind"][k]))

    for profile in ("clean", "dirty"):
        key = f"pdf/{profile}"
        if key in summary_raw["by_profile"]:
            rows.append(("profile", key, summary_raw["by_profile"][key]))

    for b in ("<25%", "25-50%", ">50%"):
        rows.append(("span_char_share", b, summary_raw["by_span_char_share"][b]))

    def _print(slice_name, group_name, s):
        print(
            f"| {slice_name} | {group_name} | {s['n']} | "
            f"{s['k5']['copy_fraction']['mean']:.3f} | {s['k5']['copy_fraction']['median']:.3f} | "
            f"{s['k5']['speedup']['mean']:.3f} | {s['k5']['speedup']['median']:.3f} | "
            f"{s['k10']['copy_fraction']['mean']:.3f} | {s['k10']['copy_fraction']['median']:.3f} | "
            f"{s['k10']['speedup']['mean']:.3f} | {s['k10']['speedup']['median']:.3f} | "
            f"{s['in_input_token_share']['mean']:.3f} | {s['in_input_token_share']['median']:.3f} |"
        )
    for slice_name, group_name, s in rows:
        _print(slice_name, group_name, s)
        norm_s = None
        if slice_name == "overall":
            norm_s = summary_norm.get("overall")
        elif slice_name == "dominant_kind":
            norm_s = summary_norm.get("by_kind", {}).get(group_name)
        elif slice_name == "profile":
            norm_s = summary_norm.get("by_profile", {}).get(group_name)
        elif slice_name == "span_char_share":
            norm_s = summary_norm.get("by_span_char_share", {}).get(group_name)
        if norm_s:
            _print(slice_name, f"{group_name} (norm)", norm_s)


def print_synth_dirty_knob_table(knob_rows):
    print("\nSynth dirty profile by knob (means)")
    print("| knob | n_on | copy@10 on | speedup@10 on | n_off | copy@10 off | speedup@10 off |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in knob_rows:
        print(
            f"| {row['knob']} | {row['n_on']} | {row['copy10_on']:.3f} | {row['speedup10_on']:.3f} | "
            f"{row['n_off']} | {row['copy10_off']:.3f} | {row['speedup10_off']:.3f} |"
        )


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def render_expected_output(text, spans):
    out = []
    cursor = 0
    for span in sorted(spans, key=lambda x: x["start"]):
        start = int(span["start"])
        end = int(span["end"])
        if start < cursor or end < start or end > len(text):
            raise ValueError(
                f"invalid span range start={start} end={end} for text length {len(text)}"
            )
        out.append(text[cursor:start])
        out.append(f"\\( {span['latex']} \\)")
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def dominant_kind(spans):
    if not spans:
        return "none"
    kinds = {s.get("kind", "unknown") for s in spans}
    if len(kinds) == 1:
        return next(iter(kinds))
    return "mixed"


def span_char_share(text, spans):
    if not text:
        return 0.0
    total = 0
    for s in spans:
        total += int(s["end"]) - int(s["start"])
    return total / len(text)


def span_share_bucket(share):
    if share < 0.25:
        return "<25%"
    if share <= 0.5:
        return "25-50%"
    return ">50%"


def evaluate_pair(tok, input_text, output_text, *, normalize_input=False):
    if normalize_input:
        input_text = normalize_input_text(input_text)
    in_ids = tokenize(tok, input_text)
    out_ids = tokenize(tok, output_text)
    total = len(out_ids)
    if total == 0:
        return None
    in_set = set(in_ids)
    upper = sum(1 for t in out_ids if t in in_set) / total
    acc5, steps5 = simulate_prompt_lookup(in_ids, out_ids, k=5)
    acc10, steps10 = simulate_prompt_lookup(in_ids, out_ids, k=10)
    return {
        "total_tokens": total,
        "in_input_token_share": upper,
        "k5": {
            "accepted_tokens": acc5,
            "steps": steps5,
            "copy_fraction": acc5 / total,
            "speedup": total / steps5 if steps5 else 0.0,
        },
        "k10": {
            "accepted_tokens": acc10,
            "steps": steps10,
            "copy_fraction": acc10 / total,
            "speedup": total / steps10 if steps10 else 0.0,
        },
    }


def build_synth_summary(rows):
    by_kind = {}
    by_share = {"<25%": [], "25-50%": [], ">50%": []}
    for r in rows:
        by_kind.setdefault(r["dominant_kind"], []).append(r)
        by_share[r["span_char_share_bucket"]].append(r)

    by_profile = {}
    for profile in ("clean", "dirty"):
        prof_rows = [
            r
            for r in rows
            if r.get("kind") == "pdf" and r.get("profile") == profile
        ]
        if prof_rows:
            by_profile[f"pdf/{profile}"] = build_group_stats(prof_rows)

    dirty_pdf_rows = [r for r in rows if r.get("kind") == "pdf" and r.get("profile") == "dirty"]
    knob_names = sorted({k for r in dirty_pdf_rows for k in r.get("knobs", [])})
    knob_table = []
    for knob in knob_names:
        on = [r for r in dirty_pdf_rows if knob in r.get("knobs", [])]
        off = [r for r in dirty_pdf_rows if knob not in r.get("knobs", [])]
        on_copy = [r["k10"]["copy_fraction"] for r in on]
        on_speed = [r["k10"]["speedup"] for r in on]
        off_copy = [r["k10"]["copy_fraction"] for r in off]
        off_speed = [r["k10"]["speedup"] for r in off]
        knob_table.append(
            {
                "knob": knob,
                "n_on": len(on),
                "copy10_on": float(np.mean(on_copy)) if on_copy else 0.0,
                "speedup10_on": float(np.mean(on_speed)) if on_speed else 0.0,
                "n_off": len(off),
                "copy10_off": float(np.mean(off_copy)) if off_copy else 0.0,
                "speedup10_off": float(np.mean(off_speed)) if off_speed else 0.0,
            }
        )

    return {
        "overall": build_group_stats(rows),
        "by_kind": {k: build_group_stats(v) for k, v in by_kind.items()},
        "by_profile": by_profile,
        "by_span_char_share": {
            bucket: build_group_stats(rs) for bucket, rs in by_share.items()
        },
        "dirty_knob_table": knob_table,
    }


def evaluate_synth(tok, synth_path, *, primary_normalize=False):
    examples = load_jsonl(synth_path)
    rows_raw = []
    rows_norm = []
    skipped_bad = 0
    for ex in examples:
        if ex.get("ok") is False:
            skipped_bad += 1
            continue
        text = ex.get("text", "")
        spans = ex.get("spans", [])
        expected = ex.get("reference") or render_expected_output(text, spans)
        share = span_char_share(text, spans)
        metrics_raw = evaluate_pair(tok, text, expected, normalize_input=False)
        metrics_norm = evaluate_pair(tok, text, expected, normalize_input=True)
        if metrics_raw is None or metrics_norm is None:
            continue
        base = {
            "id": ex.get("id", ""),
            "kind": ex.get("kind", dominant_kind(spans)),
            "profile": ex.get("profile"),
            "knobs": ex.get("knobs", []),
            "span_count": len(spans),
            "dominant_kind": dominant_kind(spans),
            "span_char_share": share,
            "span_char_share_bucket": span_share_bucket(share),
        }
        rows_raw.append({**base, **metrics_raw})
        rows_norm.append({**base, **metrics_norm})

    rows_raw_sorted = sorted(rows_raw, key=lambda x: x["id"])
    rows_norm_sorted = sorted(rows_norm, key=lambda x: x["id"])
    rows_primary = rows_norm_sorted if primary_normalize else rows_raw_sorted
    summary_raw = build_synth_summary(rows_raw_sorted)
    summary_norm = build_synth_summary(rows_norm_sorted)
    return {
        "path": str(synth_path),
        "n_examples": len(examples),
        "n_skipped_bad": skipped_bad,
        "n_scored": len(rows_primary),
        "rows": rows_primary,
        "rows_raw": rows_raw_sorted,
        "rows_norm": rows_norm_sorted,
        "summary": summary_norm if primary_normalize else summary_raw,
        "summary_raw": summary_raw,
        "summary_norm": summary_norm,
    }


def main():
    args = parse_args()
    tok, tok_path = load_tokenizer()
    bench_items = json.load(open(BENCH_PATH))
    pdf_items = []
    if PDF_PASTES_PATH.exists():
        pdf_items = json.load(open(PDF_PASTES_PATH))
    all_items = bench_items + pdf_items
    skipped_items = sum(1 for it in all_items if it.get("ok") is False)
    items = [it for it in all_items if it.get("ok", True)]
    results = json.load(open(RESULTS_PATH))
    by_id = {x["id"]: x for x in items}

    pairs = []
    for it in items:
        pairs.append(
            {
                "item": it["id"],
                "tier": it["tier"],
                "source": "reference",
                "input": it["input"],
                "output": it["reference"],
                "profile": it.get("profile"),
                "knobs": it.get("knobs", []),
            }
        )
    for r in results:
        if r.get("approach", "").startswith("direct:Qwen3-") and r.get("correct") is True:
            if r["item"] not in by_id:
                continue
            it = by_id[r["item"]]
            pairs.append(
                {
                    "item": r["item"],
                    "tier": r["tier"],
                    "source": r["approach"],
                    "input": it["input"],
                    "output": r.get("output", ""),
                    "profile": it.get("profile"),
                    "knobs": it.get("knobs", []),
                }
            )

    rows_raw = []
    rows_norm = []
    for p in pairs:
        metrics_raw = evaluate_pair(tok, p["input"], p["output"], normalize_input=False)
        metrics_norm = evaluate_pair(tok, p["input"], p["output"], normalize_input=True)
        if metrics_raw is None or metrics_norm is None:
            continue
        base = {
            "item": p["item"],
            "tier": p["tier"],
            "source": p["source"],
            "profile": p.get("profile"),
            "knobs": p.get("knobs", []),
        }
        rows_raw.append({**base, **metrics_raw})
        rows_norm.append({**base, **metrics_norm})

    rows_raw_sorted = sorted(rows_raw, key=lambda x: (x["item"], x["source"]))
    rows_norm_sorted = sorted(rows_norm, key=lambda x: (x["item"], x["source"]))
    rows = rows_norm_sorted if args.normalize_input else rows_raw_sorted
    summary_raw = build_summary(rows_raw_sorted)
    summary_norm = build_summary(rows_norm_sorted)

    print(f"tokenizer: {tok_path}")
    print(f"pairs: {len(rows)}")
    print(f"items skipped (ok=false): {skipped_items}")
    print_pair_table(rows)
    print_summary_table(summary_raw, summary_norm)

    synth = None
    if args.synth:
        synth_path = Path(args.synth).expanduser()
        if not synth_path.is_absolute():
            synth_path = Path.cwd() / synth_path
        if not synth_path.exists():
            raise SystemExit(f"synth JSONL not found: {synth_path}")
        synth = evaluate_synth(tok, synth_path, primary_normalize=args.normalize_input)
        print(
            f"\nsynth examples: {synth['n_examples']} "
            f"(scored={synth['n_scored']}, skipped ok=false={synth['n_skipped_bad']})"
        )
        print_synth_summary_table(synth["summary_raw"], synth["summary_norm"])
        print_synth_dirty_knob_table(synth["summary"]["dirty_knob_table"])
    else:
        print("\nsynth examples: skipped (no --synth and default file missing)")

    payload = {
        "tokenizer_path": tok_path,
        "pairs": rows,
        "pairs_raw": rows_raw_sorted,
        "pairs_norm": rows_norm_sorted,
        "summary": summary_norm if args.normalize_input else summary_raw,
        "summary_raw": summary_raw,
        "summary_norm": summary_norm,
        "config": {"n_grams": [3, 2], "draft_tokens": [5, 10]},
        "synth": synth,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
