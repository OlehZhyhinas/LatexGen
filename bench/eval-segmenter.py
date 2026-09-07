#!/usr/bin/env python3
"""Offline eval for bench/pdf_pipeline.py."""
import difflib
import json
import pathlib
import re

import pdf_pipeline

ROOT = pathlib.Path(__file__).resolve().parent
PDF_PASTES = ROOT / "pdf-pastes.json"
SYNTH = ROOT / "synth-spans.jsonl"
OUT_JSON = ROOT / "results-segmenter.json"

MATH_DELIM_RE = re.compile(r"\\\((.*?)\\\)|\\\[(.*?)\\\]", re.DOTALL)


def _collapse_ws(s):
    return re.sub(r"\s+", " ", s).strip()


def _extract_reference_prose(reference):
    return _collapse_ws(MATH_DELIM_RE.sub(" ", reference))


def _span_mask(text, spans):
    mask = [False] * len(text)
    for s, e in spans:
        s = max(0, min(len(text), int(s)))
        e = max(0, min(len(text), int(e)))
        for i in range(s, e):
            mask[i] = True
    return mask


def _mapped_gold_spans(omap, gold_spans):
    out = []
    for sp in gold_spans:
        start = int(sp["start"])
        end = int(sp["end"])
        inside = [False] * len(omap)
        for i, oi in enumerate(omap):
            if start <= oi < end:
                inside[i] = True
        i = 0
        while i < len(inside):
            if not inside[i]:
                i += 1
                continue
            j = i + 1
            while j < len(inside):
                if inside[j]:
                    j += 1
                    continue
                if omap[j] == -1 and j + 1 < len(inside) and inside[j + 1]:
                    j += 1
                    continue
                break
            out.append((i, j))
            i = j
    out.sort()
    merged = []
    for s, e in out:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(a, b) for a, b in merged]


def _chars_in_spans(text, spans):
    out = set()
    for s, e in spans:
        for i in range(s, e):
            if i < 0 or i >= len(text):
                continue
            if text[i].isspace():
                continue
            out.add(i)
    return out


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f


def _iou(a, b):
    s1, e1 = a
    s2, e2 = b
    inter = max(0, min(e1, e2) - max(s1, s2))
    if inter <= 0:
        return 0.0
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


def _span_match_iou(gold, pred, thr=0.5):
    cand = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            v = _iou(g, p)
            if v >= thr:
                cand.append((v, gi, pi))
    cand.sort(reverse=True)
    used_g = set()
    used_p = set()
    hit = 0
    for _, gi, pi in cand:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        hit += 1
    return hit


def _source_key(row):
    if row["set"] == "pdf-pastes":
        return f"pdf-paste/{row.get('profile') or '-'}"
    kind = row.get("kind")
    profile = row.get("profile")
    if kind == "pdf":
        return f"synth-pdf/{profile or '-'}"
    if kind == "unicode":
        return "synth-unicode"
    return f"synth-{kind or 'unknown'}"


def _init_bucket():
    return {
        "n": 0,
        "char_tp": 0,
        "char_fp": 0,
        "char_fn": 0,
        "exact_tp": 0,
        "exact_fp": 0,
        "exact_fn": 0,
        "iou_tp": 0,
        "iou_fp": 0,
        "iou_fn": 0,
        "prose_exact": 0,
        "prose_ratio_sum": 0.0,
    }


def _load_rows():
    rows = []
    for item in json.loads(PDF_PASTES.read_text()):
        if not item.get("ok", True):
            continue
        rows.append(
            {
                "set": "pdf-pastes",
                "id": item["id"],
                "input": item["input"],
                "reference": item["reference"],
                "spans": item["spans"],
                "kind": "pdf",
                "profile": item.get("profile"),
            }
        )
    with SYNTH.open() as f:
        for line in f:
            ex = json.loads(line)
            if not ex.get("ok", True):
                continue
            if ex.get("kind") not in {"pdf", "unicode"}:
                continue
            rows.append(
                {
                    "set": "synth",
                    "id": ex["id"],
                    "input": ex["text"],
                    "reference": ex.get("reference", ex["text"]),
                    "spans": ex.get("spans", []),
                    "kind": ex.get("kind"),
                    "profile": ex.get("profile"),
                }
            )
    return rows


def _span_text(text, spans):
    chunks = []
    for s, e in spans:
        frag = _collapse_ws(text[s:e])
        if len(frag) > 64:
            frag = frag[:61] + "..."
        chunks.append(f"[{s}:{e}] {frag}")
    return chunks


def main():
    rows = _load_rows()
    buckets = {"overall": _init_bucket()}
    worst = []
    details = []

    for row in rows:
        clean, omap = pdf_pipeline.normalize(row["input"])
        gold = _mapped_gold_spans(omap, row["spans"])
        pred = pdf_pipeline.segment(clean)

        gold_chars = _chars_in_spans(clean, gold)
        pred_chars = _chars_in_spans(clean, pred)
        tp = len(gold_chars & pred_chars)
        fp = len(pred_chars - gold_chars)
        fn = len(gold_chars - pred_chars)
        c_p, c_r, c_f = _prf(tp, fp, fn)

        exact_hit = 0
        pred_left = pred[:]
        for g in gold:
            if g in pred_left:
                exact_hit += 1
                pred_left.remove(g)
        e_tp = exact_hit
        e_fp = len(pred) - exact_hit
        e_fn = len(gold) - exact_hit

        i_tp = _span_match_iou(gold, pred, thr=0.5)
        i_fp = len(pred) - i_tp
        i_fn = len(gold) - i_tp

        gmask = _span_mask(clean, gold)
        clean_prose = "".join(ch for i, ch in enumerate(clean) if not gmask[i])
        clean_prose = _collapse_ws(clean_prose)
        ref_prose = _extract_reference_prose(row["reference"])
        prose_exact = clean_prose == ref_prose
        prose_ratio = difflib.SequenceMatcher(None, clean_prose, ref_prose).ratio()

        key = _source_key(row)
        if key not in buckets:
            buckets[key] = _init_bucket()
        for bkey in ("overall", key):
            b = buckets[bkey]
            b["n"] += 1
            b["char_tp"] += tp
            b["char_fp"] += fp
            b["char_fn"] += fn
            b["exact_tp"] += e_tp
            b["exact_fp"] += e_fp
            b["exact_fn"] += e_fn
            b["iou_tp"] += i_tp
            b["iou_fp"] += i_fp
            b["iou_fn"] += i_fn
            b["prose_exact"] += int(prose_exact)
            b["prose_ratio_sum"] += prose_ratio

        worst.append(
            {
                "id": row["id"],
                "source": key,
                "char_f1": c_f,
                "clean": clean,
                "gold": _span_text(clean, gold),
                "pred": _span_text(clean, pred),
            }
        )
        details.append(
            {
                "id": row["id"],
                "source": key,
                "char_p": c_p,
                "char_r": c_r,
                "char_f1": c_f,
                "gold_n": len(gold),
                "pred_n": len(pred),
                "gold_spans": gold,
                "pred_spans": pred,
                "prose_exact": prose_exact,
                "prose_ratio": prose_ratio,
            }
        )

    metric_rows = {}
    prose_rows = {}
    for key, b in buckets.items():
        cp, cr, cf = _prf(b["char_tp"], b["char_fp"], b["char_fn"])
        ep, er, ef = _prf(b["exact_tp"], b["exact_fp"], b["exact_fn"])
        ip, ir, iof = _prf(b["iou_tp"], b["iou_fp"], b["iou_fn"])
        metric_rows[key] = {
            "n": b["n"],
            "char_precision": cp,
            "char_recall": cr,
            "char_f1": cf,
            "span_exact_precision": ep,
            "span_exact_recall": er,
            "span_exact_f1": ef,
            "span_iou_precision": ip,
            "span_iou_recall": ir,
            "span_iou_f1": iof,
        }
        prose_rows[key] = {
            "n": b["n"],
            "exact_prose_match_rate": (b["prose_exact"] / b["n"]) if b["n"] else 0.0,
            "mean_prose_ratio": (b["prose_ratio_sum"] / b["n"]) if b["n"] else 0.0,
        }

    ordered = [
        "pdf-paste/clean",
        "pdf-paste/dirty",
        "synth-pdf/clean",
        "synth-pdf/dirty",
        "synth-unicode",
        "overall",
    ]
    print("Segmenter metrics")
    print("| source | n | char_P | char_R | char_F1 | exact_F1 | iou_F1 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for key in ordered:
        row = metric_rows.get(key)
        if not row:
            continue
        print(
            f"| {key} | {row['n']} | {row['char_precision']:.3f} | "
            f"{row['char_recall']:.3f} | {row['char_f1']:.3f} | "
            f"{row['span_exact_f1']:.3f} | {row['span_iou_f1']:.3f} |"
        )

    print("\nProse match")
    print("| source | n | exact_rate | mean_ratio |")
    print("|---|---:|---:|---:|")
    for key in ordered:
        row = prose_rows.get(key)
        if not row:
            continue
        print(
            f"| {key} | {row['n']} | {row['exact_prose_match_rate']:.3f} | "
            f"{row['mean_prose_ratio']:.3f} |"
        )

    worst.sort(key=lambda x: x["char_f1"])
    print("\nWorst 10 by char-F1")
    for item in worst[:10]:
        print(f"\n== {item['id']} ({item['source']}) f1={item['char_f1']:.3f}")
        clean = item["clean"]
        if len(clean) > 240:
            clean = clean[:240] + "..."
        print(f"clean: {clean}")
        print("gold:")
        for line in item["gold"]:
            print(f"  {line}")
        print("pred:")
        for line in item["pred"]:
            print(f"  {line}")

    payload = {
        "metrics": metric_rows,
        "prose": prose_rows,
        "worst10": worst[:10],
        "details": details,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
