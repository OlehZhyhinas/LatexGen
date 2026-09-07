#!/usr/bin/env python3
"""Gate C: measure vocabulary keep-set coverage and leave-one-item-out risk."""
import argparse
import collections
import json
from glob import glob
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
BENCH_PATH = ROOT / "bench-data.json"
PDF_PASTES_PATH = ROOT / "pdf-pastes.json"
RESULTS_PATH = ROOT / "results-2026-09-01.json"
OUT_PATH = ROOT / "results-vocab-coverage.json"
TOKENIZER_GLOB = str(
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json"
)
ORIG_VOCAB_SIZE = 151_936
PAD_MULTIPLE = 4_096
DEFAULT_PAD_VALUES = [0, 8192, 16384, 32768, 65536]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pad",
        type=int,
        action="append",
        default=None,
        metavar="N",
        help="append first N regular token ids to leave-one-out keep-set (repeatable)",
    )
    return ap.parse_args()


def resolve_pad_values(values):
    if values is None:
        return DEFAULT_PAD_VALUES
    out = []
    seen = set()
    for n in values:
        if n < 0:
            raise SystemExit(f"--pad must be >= 0, got {n}")
        if n not in seen:
            seen.add(n)
            out.append(n)
    if not out:
        raise SystemExit("no pad values resolved")
    return out


def load_tokenizer():
    paths = sorted(glob(TOKENIZER_GLOB))
    if not paths:
        raise SystemExit(f"tokenizer.json not found: {TOKENIZER_GLOB}")
    return Tokenizer.from_file(paths[-1]), paths[-1]


def tokenize(tok, text):
    return tok.encode(text, add_special_tokens=False).ids


def gpt2_bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def get_byte_token_ids(tok):
    vocab = tok.get_vocab()
    out = set()
    for ch in gpt2_bytes_to_unicode().values():
        tid = vocab.get(ch)
        if tid is not None:
            out.add(tid)
    if len(out) != 256:
        raise SystemExit(f"expected 256 byte tokens, got {len(out)}")
    return out


def get_regular_token_ids(tok, byte_ids, added_ids):
    regular = []
    vocab_size = tok.get_vocab_size()
    for tid in range(256, vocab_size):
        if tid in byte_ids or tid in added_ids:
            continue
        if tok.id_to_token(tid) is None:
            continue
        regular.append(tid)
    return regular


def stats(xs):
    if not xs:
        return {"mean": 0.0, "median": 0.0}
    arr = np.array(xs, dtype=float)
    return {"mean": float(arr.mean()), "median": float(np.median(arr))}


def fmt_bytes(n):
    return f"{n / 2**20:.2f} MiB"


def token_display(tok, tid):
    piece = tok.decode([tid], skip_special_tokens=False)
    if piece == "":
        piece = tok.id_to_token(tid) or ""
    return piece.encode("unicode_escape").decode("ascii")


def build_sequences(tok, items, results):
    seqs = []
    by_item = collections.defaultdict(list)
    for it in items:
        rec = {"item": it["id"], "kind": "input", "source": "input", "text": it["input"]}
        rec["ids"] = tokenize(tok, rec["text"])
        seqs.append(rec)
        by_item[it["id"]].append(rec)

        rec = {"item": it["id"], "kind": "reference", "source": "reference", "text": it["reference"]}
        rec["ids"] = tokenize(tok, rec["text"])
        seqs.append(rec)
        by_item[it["id"]].append(rec)

    for r in results:
        rec = {
            "item": r["item"],
            "kind": "output",
            "source": r["approach"],
            "text": r.get("output", ""),
        }
        rec["ids"] = tokenize(tok, rec["text"])
        seqs.append(rec)
        by_item[r["item"]].append(rec)
    return seqs, by_item


def aggregate_loo(rows):
    def group_rows(name):
        if name == "overall":
            return rows
        return [r for r in rows if r["tier"] == name]

    agg = {}
    for name in ["easy", "medium", "hard", "multiline", "pdf-paste", "overall"]:
        rs = group_rows(name)
        total_eval = sum(r["eval_tokens"] for r in rs)
        total_chars = sum(r["eval_chars"] for r in rs)
        total_oos = sum(r["out_of_set_tokens"] for r in rs)
        total_sim = sum(r["sim_tokens_pruned"] for r in rs)
        agg[name] = {
            "n_items": len(rs),
            "oos_rate_micro": (total_oos / total_eval) if total_eval else 0.0,
            "oos_rate": stats([r["out_of_set_rate"] for r in rs]),
            "fertility_full_micro": (total_eval / total_chars) if total_chars else 0.0,
            "fertility_pruned_sim_micro": (total_sim / total_chars) if total_chars else 0.0,
            "fertility_full": stats([r["fertility_full"] for r in rs]),
            "fertility_pruned_sim": stats([r["fertility_pruned_sim"] for r in rs]),
        }
    return agg


def eval_leave_one_out(
    tok,
    items,
    by_item,
    all_counts,
    byte_ids,
    added_ids,
    special_ids,
    regular_ids,
    pad_n,
):
    # Assumption: in Qwen byte-level BPE, merge rank roughly follows corpus frequency,
    # and token ids track that merge order after the first 256 byte tokens.
    head_ids = set(regular_ids[:pad_n])
    byte_len_cache = {}

    def token_byte_len(tid):
        if tid in byte_len_cache:
            return byte_len_cache[tid]
        piece = tok.decode([tid], skip_special_tokens=False)
        if piece == "":
            piece = tok.id_to_token(tid) or ""
        n = len(piece.encode("utf-8"))
        if n == 0:
            n = 1
        byte_len_cache[tid] = n
        return n

    rows = []
    oos_counts = collections.Counter()
    ref_exact_items = []

    for it in items:
        held = it["id"]
        held_counts = collections.Counter()
        for s in by_item[held]:
            held_counts.update(s["ids"])
        train_counts = all_counts - held_counts
        train_seen = set(train_counts.keys())
        loo_keep = train_seen | byte_ids | added_ids | special_ids | head_ids

        ref_seq = next(s for s in by_item[held] if s["kind"] == "reference")
        ref_oos = 0
        for tid in ref_seq["ids"]:
            if tid not in loo_keep:
                ref_oos += 1
        ref_exact = (ref_oos == 0)
        if ref_exact:
            ref_exact_items.append(held)

        eval_seqs = [s for s in by_item[held] if s["kind"] in {"reference", "output"}]
        eval_tokens = 0
        eval_chars = 0
        oos = 0
        sim_tokens = 0
        oos_ids = set()
        for s in eval_seqs:
            eval_chars += len(s["text"])
            eval_tokens += len(s["ids"])
            for tid in s["ids"]:
                if tid in loo_keep:
                    sim_tokens += 1
                else:
                    oos += 1
                    oos_ids.add(tid)
                    oos_counts[tid] += 1
                    sim_tokens += token_byte_len(tid)

        out_rate = (oos / eval_tokens) if eval_tokens else 0.0
        fert_full = (eval_tokens / eval_chars) if eval_chars else 0.0
        fert_sim = (sim_tokens / eval_chars) if eval_chars else 0.0
        rows.append(
            {
                "item": held,
                "tier": it["tier"],
                "eval_sequences": len(eval_seqs),
                "eval_chars": eval_chars,
                "eval_tokens": eval_tokens,
                "out_of_set_tokens": oos,
                "out_of_set_rate": out_rate,
                "fertility_full": fert_full,
                "fertility_pruned_sim": fert_sim,
                "sim_tokens_pruned": sim_tokens,
                "reference_oos_tokens": ref_oos,
                "reference_bit_exact": ref_exact,
                "out_of_set_ids": sorted(oos_ids),
                "out_of_set_tokens_decoded": [token_display(tok, tid) for tid in sorted(oos_ids)],
            }
        )

    rows.sort(key=lambda r: r["item"])
    summary = aggregate_loo(rows)
    top_oos = []
    for tid, c in oos_counts.most_common(30):
        top_oos.append({"id": tid, "decoded": token_display(tok, tid), "count": c})

    return {
        "pad_n": pad_n,
        "head_ids_size": len(head_ids),
        "head_ids_sample": sorted(head_ids)[:16],
        "reference_bit_exact_fraction": len(ref_exact_items) / len(items) if items else 0.0,
        "reference_bit_exact_items": ref_exact_items,
        "per_item": rows,
        "summary": summary,
        "oos_top_tokens": top_oos,
    }


def main():
    args = parse_args()
    pad_values = resolve_pad_values(args.pad)
    tok, tok_path = load_tokenizer()
    bench_items = json.load(open(BENCH_PATH))
    pdf_items = []
    if PDF_PASTES_PATH.exists():
        pdf_items = json.load(open(PDF_PASTES_PATH))
    all_items = bench_items + pdf_items
    skipped_items = sum(1 for it in all_items if it.get("ok") is False)
    items = [it for it in all_items if it.get("ok", True)]
    item_ids = {it["id"] for it in items}
    raw_results = json.load(open(RESULTS_PATH))
    skipped_results_item = sum(1 for r in raw_results if r.get("item") not in item_ids)
    skipped_results_filter = sum(
        1
        for r in raw_results
        if r.get("item") in item_ids
        and (not r.get("approach", "").startswith("direct:Qwen3-") or r.get("correct") is not True)
    )
    results = [
        r
        for r in raw_results
        if r.get("item") in item_ids
        and r.get("approach", "").startswith("direct:Qwen3-")
        and r.get("correct") is True
    ]

    all_seqs, by_item = build_sequences(tok, items, results)
    all_counts = collections.Counter()
    total_chars = 0
    total_tokens = 0
    for s in all_seqs:
        all_counts.update(s["ids"])
        total_chars += len(s["text"])
        total_tokens += len(s["ids"])

    byte_ids = get_byte_token_ids(tok)
    added_decoder = tok.get_added_tokens_decoder()
    added_ids = set(added_decoder.keys())
    special_ids = {tid for tid, t in added_decoder.items() if getattr(t, "special", False)}
    regular_ids = get_regular_token_ids(tok, byte_ids, added_ids)

    seen_ids = set(all_counts.keys())
    base_keep_ids = seen_ids | byte_ids | added_ids | special_ids
    keep_size = len(base_keep_ids)
    padded_keep = ((keep_size + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
    keep_size_by_pad = {}
    padded_keep_by_pad = {}
    for n in pad_values:
        keep_n = len(base_keep_ids | set(regular_ids[:n]))
        keep_size_by_pad[str(n)] = keep_n
        padded_keep_by_pad[str(n)] = ((keep_n + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE

    fertility_full = total_tokens / total_chars if total_chars else 0.0

    loo_by_pad = {}
    for n in pad_values:
        loo_by_pad[str(n)] = eval_leave_one_out(
            tok=tok,
            items=items,
            by_item=by_item,
            all_counts=all_counts,
            byte_ids=byte_ids,
            added_ids=added_ids,
            special_ids=special_ids,
            regular_ids=regular_ids,
            pad_n=n,
        )

    dims = [
        ("Qwen3-0.6B", 1024),
        ("Qwen3-1.7B", 2048),
        ("Qwen3-4B", 2560),
    ]
    memory = []
    for n in pad_values:
        keep_n = keep_size_by_pad[str(n)]
        padded_n = padded_keep_by_pad[str(n)]
        for name, h in dims:
            row = {
                "pad_n": n,
                "keep_size": keep_n,
                "padded_keep_size": padded_n,
                "model": name,
                "hidden": h,
            }
            for dtype, bpp in [("fp16", 2.0), ("int4", 0.5)]:
                orig = int(ORIG_VOCAB_SIZE * h * bpp)
                pruned = int(padded_n * h * bpp)
                row[dtype] = {
                    "orig_bytes": orig,
                    "pruned_bytes": pruned,
                    "saved_bytes": orig - pruned,
                }
            memory.append(row)

    print(f"tokenizer: {tok_path}")
    print(f"corpus texts: {len(all_seqs)}")
    print(f"items skipped (ok=false): {skipped_items}")
    print(f"outputs skipped (item filtered out): {skipped_results_item}")
    print(f"outputs skipped (approach/correct filter): {skipped_results_filter}")
    print(f"tokenizer vocab size: {tok.get_vocab_size()}")
    print(f"token ids seen in corpus: {len(seen_ids)}")
    print(f"base keep-set size (seen + 256 bytes + specials/added): {keep_size}")
    print(f"base keep-set padded to {PAD_MULTIPLE}: {padded_keep}")
    print(f"regular ids available for head padding: {len(regular_ids)}")
    print(f"fertility full corpus (tokens/char): {fertility_full:.4f}")

    print("\nLeave-one-item-out by head padding")
    print("| pad_n | oos_micro_overall | oos_micro_easy | oos_micro_medium | oos_micro_hard | oos_micro_multiline | oos_micro_pdf_paste | fert_pruned_micro_overall | ref_bit_exact_frac |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n in pad_values:
        block = loo_by_pad[str(n)]
        s = block["summary"]
        print(
            f"| {n} | {s['overall']['oos_rate_micro']:.4f} | {s['easy']['oos_rate_micro']:.4f} | "
            f"{s['medium']['oos_rate_micro']:.4f} | {s['hard']['oos_rate_micro']:.4f} | "
            f"{s['multiline']['oos_rate_micro']:.4f} | {s['pdf-paste']['oos_rate_micro']:.4f} | "
            f"{s['overall']['fertility_pruned_sim_micro']:.4f} | "
            f"{block['reference_bit_exact_fraction']:.4f} |"
        )

    for n in pad_values:
        block = loo_by_pad[str(n)]
        pairs = [f"{x['decoded']} ({x['count']})" for x in block["oos_top_tokens"]]
        joined = ", ".join(pairs) if pairs else "(none)"
        print(f"\nN={n} top-30 out-of-set decoded tokens: {joined}")

    print("\nEmbedding bytes (tied vocab matrix)")
    print("| pad_n | keep_size | padded_keep | model | hidden | fp16 orig | fp16 pruned | fp16 saved | int4 orig | int4 pruned | int4 saved |")
    print("|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in memory:
        print(
            f"| {r['pad_n']} | {r['keep_size']} | {r['padded_keep_size']} | {r['model']} | {r['hidden']} | "
            f"{fmt_bytes(r['fp16']['orig_bytes'])} | {fmt_bytes(r['fp16']['pruned_bytes'])} | "
            f"{fmt_bytes(r['fp16']['saved_bytes'])} | {fmt_bytes(r['int4']['orig_bytes'])} | "
            f"{fmt_bytes(r['int4']['pruned_bytes'])} | {fmt_bytes(r['int4']['saved_bytes'])} |"
        )

    payload = {
        "tokenizer_path": tok_path,
        "config": {
            "original_vocab_size_for_memory": ORIG_VOCAB_SIZE,
            "pad_multiple": PAD_MULTIPLE,
            "pad_values": pad_values,
            "head_frequency_proxy": "Use first N regular token ids (after 256 byte tokens) as merge-rank proxy for top-N corpus frequency.",
        },
        "corpus": {
            "num_texts": len(all_seqs),
            "total_chars": total_chars,
            "total_tokens": total_tokens,
            "fertility_tokens_per_char": fertility_full,
            "seen_token_ids": len(seen_ids),
            "token_frequencies": dict(all_counts),
        },
        "keep_set": {
            "size": keep_size,
            "padded_size": padded_keep,
            "size_by_pad": keep_size_by_pad,
            "padded_size_by_pad": padded_keep_by_pad,
            "byte_token_ids": sorted(byte_ids),
            "added_token_ids": sorted(added_ids),
            "special_token_ids": sorted(special_ids),
            "regular_token_ids_count": len(regular_ids),
        },
        "leave_one_out": {
            "by_pad": loo_by_pad,
        },
        "embedding_memory": memory,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
