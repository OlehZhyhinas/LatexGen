#!/usr/bin/env python3
"""Gate C: measure vocabulary keep-set coverage and leave-one-item-out risk.

Two modes, selected by whether --tokenizer-label is given:

- Legacy mode (no --tokenizer-label): unchanged from the original script.
  Same default tokenizer glob (local Qwen3-0.6B cache), same two corpora
  (bench-data.json + pdf-pastes.json), same --pad semantics, same
  results-vocab-coverage.json output path, same numbers. This exists so the
  original Gate C invocation and its recorded numbers stay reproducible.

- Extended mode (--tokenizer-label given): the real target tokenizers
  (Qwen3.5 / MiniCPM5), the widened corpus set from bench-data.json,
  bench-data-extended.json, pdf-pastes.json, arxiv-pastes.json and the
  latex-kind rows of synth-spans.jsonl (each selectable via --corpus, union
  by default), judged model outputs from judged-arxiv-2026-09-07.json /
  judged-text-tiers-2026-09-07.json (only where correct), the KaTeX macro
  seed, the maths-English word seed, and every harness-prompt token
  (bench/keepset-harness-prompts.json) folded into the keep-set
  unconditionally. Target sizes are the natural (no-padding) size plus
  --target-size values (default 16384/24576/32768/49152/65536), never
  rounded. Writes bench/results-vocab-coverage-<label>.json, never touching
  the legacy output file.
"""
import argparse
import collections
import json
from glob import glob
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent

# ---- legacy defaults (unchanged) -------------------------------------------
BENCH_PATH = ROOT / "bench-data.json"
PDF_PASTES_PATH = ROOT / "pdf-pastes.json"
RESULTS_PATH = ROOT / "results-2026-09-01.json"
OUT_PATH = ROOT / "results-vocab-coverage.json"
LEGACY_TOKENIZER_GLOB = str(
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json"
)
ORIG_VOCAB_SIZE = 151_936
PAD_MULTIPLE = 4_096
DEFAULT_PAD_VALUES = [0, 8192, 16384, 32768, 65536]

# ---- extended-mode defaults -------------------------------------------------
EXTENDED_TARGET_SIZES = [16_384, 24_576, 32_768, 49_152, 65_536]
CORPUS_FILES = {
    "bench": ("bench-data.json", "json"),
    "bench-extended": ("bench-data-extended.json", "json"),
    "pdf": ("pdf-pastes.json", "json"),
    "arxiv": ("arxiv-pastes.json", "json"),
    "synth-latex": ("synth-spans.jsonl", "jsonl-latex"),
}
JUDGED_FILES = ["judged-arxiv-2026-09-07.json", "judged-text-tiers-2026-09-07.json"]
HARNESS_PROMPTS_PATH = ROOT / "keepset-harness-prompts.json"
LATEX_SEED_PATH = ROOT / "keepset-seed-latex.json"
MATHSENGLISH_SEED_PATH = ROOT / "keepset-seed-mathsenglish.json"

# judged "model" field -> canonical target-model name. Only these four are
# in scope for the model-agent handoff; anything else (qwen3:*, gemma*,
# qwen3.5:2b-q4_K_M, ...) is measured-but-out-of-scope comparison data from
# the same judged files and is not attributed to a target model.
TARGET_MODEL_ALIASES = {
    "qwen3.5:0.8b": "Qwen3.5-0.8B",
    "qwen3.5:4b": "Qwen3.5-4B",
    "hf.co/openbmb/MiniCPM5-2B-GGUF:q4_K_M": "MiniCPM5-2B",
    "openbmb/minicpm5:q4_K_M": "MiniCPM5-2B",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pad",
        type=int,
        action="append",
        default=None,
        metavar="N",
        help="[legacy mode only] append first N regular token ids to leave-one-out keep-set (repeatable)",
    )
    ap.add_argument("--tokenizer", default=None, metavar="PATH", help="path to a tokenizer.json (extended mode)")
    ap.add_argument("--tokenizer-label", default=None, metavar="LABEL", help="label used in output filenames; switches to extended mode")
    ap.add_argument(
        "--corpus",
        action="append",
        choices=sorted(CORPUS_FILES.keys()),
        default=None,
        help="[extended mode] corpus to include (repeatable); union of all if omitted",
    )
    ap.add_argument(
        "--target-size",
        type=int,
        action="append",
        default=None,
        metavar="K",
        help="[extended mode] target keep-set size (repeatable); default 16384/24576/32768/49152/65536, plus the natural no-padding size always",
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


def load_tokenizer(path=None):
    if path:
        return Tokenizer.from_file(path), path
    paths = sorted(glob(LEGACY_TOKENIZER_GLOB))
    if not paths:
        raise SystemExit(f"tokenizer.json not found: {LEGACY_TOKENIZER_GLOB}")
    return Tokenizer.from_file(paths[-1]), paths[-1]


def tokenize(tok, text):
    return tok.encode(text, add_special_tokens=False).ids


def gpt2_bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("\xa1"), ord("\xac") + 1))
    bs += list(range(ord("\xae"), ord("\xff") + 1))
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


# =============================================================================
# Legacy mode (verbatim behavior from the original script)
# =============================================================================

def build_sequences_legacy(tok, items, results):
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


def aggregate_loo_legacy(rows):
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


def eval_leave_one_out_legacy(tok, items, by_item, all_counts, byte_ids, added_ids, special_ids, regular_ids, pad_n):
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
    summary = aggregate_loo_legacy(rows)
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


def run_legacy(args):
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

    all_seqs, by_item = build_sequences_legacy(tok, items, results)
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
        loo_by_pad[str(n)] = eval_leave_one_out_legacy(
            tok=tok, items=items, by_item=by_item, all_counts=all_counts,
            byte_ids=byte_ids, added_ids=added_ids, special_ids=special_ids,
            regular_ids=regular_ids, pad_n=n,
        )

    dims = [("Qwen3-0.6B", 1024), ("Qwen3-1.7B", 2048), ("Qwen3-4B", 2560)]
    memory = []
    for n in pad_values:
        keep_n = keep_size_by_pad[str(n)]
        padded_n = padded_keep_by_pad[str(n)]
        for name, h in dims:
            row = {"pad_n": n, "keep_size": keep_n, "padded_keep_size": padded_n, "model": name, "hidden": h}
            for dtype, bpp in [("fp16", 2.0), ("int4", 0.5)]:
                orig = int(ORIG_VOCAB_SIZE * h * bpp)
                pruned = int(padded_n * h * bpp)
                row[dtype] = {"orig_bytes": orig, "pruned_bytes": pruned, "saved_bytes": orig - pruned}
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
        "leave_one_out": {"by_pad": loo_by_pad},
        "embedding_memory": memory,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {OUT_PATH}")


# =============================================================================
# Extended mode
# =============================================================================

def load_corpus_items(selected):
    merged = {}  # id -> item dict
    conflicts = []
    per_corpus_counts = {}
    for label in selected:
        fname, kind = CORPUS_FILES[label]
        path = ROOT / fname
        if not path.exists():
            print(f"warning: corpus file missing, skipping: {path}")
            continue
        raw_items = []
        if kind == "json":
            raw_items = json.load(open(path))
        elif kind == "jsonl-latex":
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    if d.get("kind") != "latex":
                        continue
                    raw_items.append(
                        {
                            "id": d["id"],
                            "tier": "synth-latex",
                            "input": d.get("text", ""),
                            "reference": d.get("reference", ""),
                            "ok": d.get("ok", True),
                        }
                    )
        n_used = 0
        n_skipped_notok = 0
        for it in raw_items:
            if it.get("ok") is False:
                n_skipped_notok += 1
                continue
            iid = it["id"]
            rec = {
                "id": iid,
                "tier": it.get("tier", label),
                "input": it.get("input", ""),
                "reference": it.get("reference", ""),
                "corpus": label,
            }
            if iid in merged:
                prev = merged[iid]
                if prev["input"] == rec["input"] and prev["reference"] == rec["reference"]:
                    pass  # exact duplicate across corpora (e.g. bench-extended re-includes bench items); keep first
                else:
                    conflicts.append(iid)
                continue
            merged[iid] = rec
            n_used += 1
        per_corpus_counts[label] = {"raw": len(raw_items), "used": n_used, "skipped_not_ok": n_skipped_notok}
    return merged, per_corpus_counts, conflicts


def load_judged_outputs(item_ids):
    outputs = collections.defaultdict(list)  # item id -> list of output records
    per_file_counts = {}
    for fname in JUDGED_FILES:
        path = ROOT / fname
        if not path.exists():
            print(f"warning: judged file missing, skipping: {path}")
            continue
        rows = json.load(open(path))
        n_total = len(rows)
        n_correct_in_scope = 0
        for r in rows:
            item = r.get("item")
            if item not in item_ids:
                continue
            if r.get("correct") is not True:
                continue
            model = r.get("model", "")
            target_model = TARGET_MODEL_ALIASES.get(model)
            outputs[item].append(
                {
                    "item": item,
                    "model": model,
                    "target_model": target_model,
                    "approach": r.get("approach", ""),
                    "text": r.get("output", ""),
                }
            )
            n_correct_in_scope += 1
        per_file_counts[fname] = {"n_rows": n_total, "n_correct_in_scope": n_correct_in_scope}
    return outputs, per_file_counts


def build_sequences_extended(tok, items, outputs_by_item):
    seqs = []
    by_item = collections.defaultdict(list)
    for iid, it in items.items():
        rec = {"item": iid, "kind": "input", "source": "input", "text": it["input"], "target_model": None}
        rec["ids"] = tokenize(tok, rec["text"])
        seqs.append(rec)
        by_item[iid].append(rec)

        rec = {"item": iid, "kind": "reference", "source": "reference", "text": it["reference"], "target_model": None}
        rec["ids"] = tokenize(tok, rec["text"])
        seqs.append(rec)
        by_item[iid].append(rec)

        for o in outputs_by_item.get(iid, []):
            rec = {
                "item": iid,
                "kind": "output",
                "source": f"{o['approach']}:{o['model']}",
                "text": o["text"],
                "target_model": o["target_model"],
            }
            rec["ids"] = tokenize(tok, rec["text"])
            seqs.append(rec)
            by_item[iid].append(rec)
    return seqs, by_item


def render_chatml(conv, messages, think_open_only):
    """Render messages the way MLC's simple ChatML-style conv_template does
    (system_template / roles / role_content_sep / seps), matching the
    structure of both the Qwen3.5 and MiniCPM5 mlc-chat-config.json files
    inspected for this task (conv_template.name == "qwen2" / "minicpm5",
    both plain ChatML). Appends the generation-prompt assistant preamble
    with the <think> block in both the enable_thinking=true and
    enable_thinking=false shapes the harness's `extra_body` can select, so
    every control token either shape can emit is included."""
    out = []
    msgs = list(messages)
    if msgs and msgs[0]["role"] == "system":
        sys_msg = msgs[0]["content"]
        msgs = msgs[1:]
    else:
        sys_msg = conv.get("system_message", "")
    out.append(conv["system_template"].replace("{system_message}", sys_msg))
    for m in msgs:
        role_tag = conv["roles"][m["role"]]
        out.append(role_tag + conv["role_content_sep"] + m["content"] + conv["seps"][0])
    out.append(conv["roles"]["assistant"] + conv["role_content_sep"])
    if think_open_only:
        out.append("<think>\n")
    else:
        out.append("<think>\n\n</think>\n\n")
    return "".join(out)


def harness_prompt_ids(tok, conv):
    data = json.load(open(HARNESS_PROMPTS_PATH))
    all_ids = set()
    n_prompts = 0
    n_chars = 0
    for group in ("fresh_items_prompts", "quality_corpus_prompts"):
        for entry in data[group]:
            for think_variant in (True, False):
                text = render_chatml(conv, entry["messages"], think_variant)
                ids = tokenize(tok, text)
                all_ids.update(ids)
                n_chars += len(text)
            n_prompts += 1
    return all_ids, n_prompts, n_chars


def seed_ids_from_strings(tok, strings, try_leading_space=False):
    all_ids = set()
    for s in strings:
        all_ids.update(tokenize(tok, s))
        if try_leading_space:
            all_ids.update(tokenize(tok, " " + s))
    return all_ids


def compute_keep_layers(tok, all_counts, conv):
    """Returns dict of layer_name -> id set, in the union order the brief
    specifies, plus the raw seed string counts for provenance."""
    byte_ids = get_byte_token_ids(tok)
    added_decoder = tok.get_added_tokens_decoder()
    added_ids = set(added_decoder.keys())
    special_ids = {tid for tid, t in added_decoder.items() if getattr(t, "special", False)}

    corpus_ids = set(all_counts.keys())

    if conv is not None:
        harness_ids, n_harness_prompts, n_harness_chars = harness_prompt_ids(tok, conv)
    else:
        harness_ids, n_harness_prompts, n_harness_chars = set(), 0, 0

    latex_seed = json.load(open(LATEX_SEED_PATH))
    latex_strings = latex_seed["all"]
    latex_ids = seed_ids_from_strings(tok, latex_strings, try_leading_space=False)

    me_seed = json.load(open(MATHSENGLISH_SEED_PATH))
    me_words = me_seed["words"]
    me_ids = seed_ids_from_strings(tok, me_words, try_leading_space=True)

    layers = [
        ("corpus", corpus_ids),
        ("byte", byte_ids),
        ("added_special", added_ids | special_ids),
        ("harness_prompts", harness_ids),
        ("latex_seed", latex_ids),
        ("mathsenglish_seed", me_ids),
    ]
    # The union of every non-corpus layer: byte/added/special tokens always
    # exist in the tokenizer regardless of corpus, and harness/latex/maths-
    # English seed ids are frozen deployment choices, not corpus evidence.
    # This must NOT be computed as "base_keep_ids minus corpus-seen ids",
    # because a token can legitimately belong to both a fixed layer (e.g. a
    # special token) AND appear literally in some corpus item's text; that
    # id must still always survive leave-one-out even when the ONE item
    # holding it in the corpus is the one being held out. Subtracting
    # all_counts.keys() from base_keep_ids incorrectly drops such ids for
    # that fold, which is the bug this comment guards against.
    always_keep_ids = byte_ids | added_ids | special_ids | harness_ids | latex_ids | me_ids
    meta = {
        "n_latex_seed_strings": len(latex_strings),
        "n_mathsenglish_seed_words": len(me_words),
        "n_harness_prompts": n_harness_prompts,
        "n_harness_chars": n_harness_chars,
        "byte_ids": byte_ids,
        "added_ids": added_ids,
        "special_ids": special_ids,
        "always_keep_ids": always_keep_ids,
    }
    return layers, meta


def union_with_marginal(layers):
    """Union all layers in order, recording how many NEW ids each layer adds
    beyond the ones before it (so seed vs. padding contribution is visible)."""
    seen = set()
    marginal = []
    for name, ids in layers:
        new = ids - seen
        marginal.append({"layer": name, "layer_size": len(ids), "new_ids": len(new)})
        seen |= ids
    return seen, marginal


def precompute_train_seen(items, by_item, all_counts):
    """Per held-out item, the set of token ids that still occur in some
    OTHER item's input/reference/outputs. Independent of target K, so this
    is computed once and reused across every target size instead of
    redoing an O(vocab) Counter subtraction per (item, K) pair."""
    train_seen_by_item = {}
    for iid in items:
        held_counts = collections.Counter()
        for s in by_item[iid]:
            held_counts.update(s["ids"])
        train_counts = all_counts - held_counts
        train_seen_by_item[iid] = set(train_counts.keys())
    return train_seen_by_item


def eval_leave_one_out_extended(tok, items, by_item, train_seen_by_item, fixed_extra_ids, padding_ids, target_models):
    """fixed_extra_ids: byte/added/special/harness/latex/mathsenglish ids,
    never removed per-item. padding_ids: the fixed frequency-padding set for
    this target K (also never removed per-item; it is a deployment choice,
    not corpus evidence). Only corpus token ids are subject to
    leave-one-out."""
    rows = []
    per_model_rows = collections.defaultdict(list)
    oos_counts = collections.Counter()
    ref_exact_items = []
    byte_len_cache = {}

    def token_byte_len(tid):
        if tid in byte_len_cache:
            return byte_len_cache[tid]
        piece = tok.decode([tid], skip_special_tokens=False)
        if piece == "":
            piece = tok.id_to_token(tid) or ""
        n = len(piece.encode("utf-8")) or 1
        byte_len_cache[tid] = n
        return n

    for iid, it in items.items():
        loo_keep = train_seen_by_item[iid] | fixed_extra_ids | padding_ids

        ref_seq = next(s for s in by_item[iid] if s["kind"] == "reference")
        ref_oos = sum(1 for tid in ref_seq["ids"] if tid not in loo_keep)
        ref_len = len(ref_seq["ids"])
        ref_exact = (ref_oos == 0)
        if ref_exact:
            ref_exact_items.append(iid)

        # Input side, same leave-one-out keep-set: this item's own `input`
        # text (real pdftotext output for arxiv-paste/pdf-paste tiers)
        # tested against a keep-set that was built WITHOUT this item's
        # contribution -- a genuine held-out check. Checking it against the
        # non-LOO deployed keep-set would be circular (that keep-set was
        # built including this very input's tokens) and always show 0% OOS.
        input_seq = next(s for s in by_item[iid] if s["kind"] == "input")
        input_oos_ids = [tid for tid in input_seq["ids"] if tid not in loo_keep]
        input_sim_pruned_tokens = sum(
            (1 if tid in loo_keep else token_byte_len(tid)) for tid in input_seq["ids"]
        )

        all_output_seqs = [s for s in by_item[iid] if s["kind"] == "output"]
        eval_seqs = [ref_seq] + all_output_seqs
        eval_tokens = 0
        eval_chars = 0
        oos = 0
        oos_ids = set()
        for s in eval_seqs:
            eval_chars += len(s["text"])
            eval_tokens += len(s["ids"])
            for tid in s["ids"]:
                if tid in loo_keep:
                    continue
                oos += 1
                oos_ids.add(tid)
                oos_counts[tid] += 1

        rows.append(
            {
                "item": iid,
                "tier": it["tier"],
                "corpus": it["corpus"],
                "eval_sequences": len(eval_seqs),
                "eval_chars": eval_chars,
                "eval_tokens": eval_tokens,
                "out_of_set_tokens": oos,
                "out_of_set_rate": (oos / eval_tokens) if eval_tokens else 0.0,
                "reference_tokens": ref_len,
                "reference_oos_tokens": ref_oos,
                "reference_bit_exact": ref_exact,
                "out_of_set_ids": sorted(oos_ids),
                "input_chars": len(it["input"]),
                "input_tokens": len(input_seq["ids"]),
                "input_oos_tokens": len(input_oos_ids),
                "input_has_oos": len(input_oos_ids) > 0,
                "input_oos_ids": input_oos_ids,
                "input_sim_pruned_tokens": input_sim_pruned_tokens,
            }
        )

        for s in all_output_seqs:
            tm = s.get("target_model")
            if tm is None or tm not in target_models:
                continue
            s_oos = sum(1 for tid in s["ids"] if tid not in loo_keep)
            per_model_rows[tm].append(
                {
                    "item": iid,
                    "tier": it["tier"],
                    "tokens": len(s["ids"]),
                    "out_of_set_tokens": s_oos,
                }
            )

    rows.sort(key=lambda r: r["item"])

    def agg(rs):
        total_eval = sum(r["eval_tokens"] for r in rs)
        total_oos = sum(r["out_of_set_tokens"] for r in rs)
        total_ref = sum(r["reference_tokens"] for r in rs)
        total_ref_oos = sum(r["reference_oos_tokens"] for r in rs)
        n_exact = sum(1 for r in rs if r["reference_bit_exact"])
        return {
            "n_items": len(rs),
            "oos_rate_micro_all_eval": (total_oos / total_eval) if total_eval else 0.0,
            "reference_oos_rate_micro": (total_ref_oos / total_ref) if total_ref else 0.0,
            "reference_bit_exact_fraction": (n_exact / len(rs)) if rs else 0.0,
        }

    tiers = sorted({r["tier"] for r in rows}) + ["overall"]
    summary = {}
    for t in tiers:
        rs = rows if t == "overall" else [r for r in rows if r["tier"] == t]
        summary[t] = agg(rs)

    # Input-side (held-out, leave-one-out) report: real pdftotext input text
    # from the arxiv-paste and pdf-paste tiers only (the tiers that are
    # actual extracted-document text; the others are already-clean spoken
    # or synthetic strings, not "real pastes").
    input_oos_id_counts = collections.Counter()
    for r in rows:
        if r["tier"] not in ("arxiv-paste", "pdf-paste"):
            continue
        input_oos_id_counts.update(r["input_oos_ids"])

    def input_agg(rs):
        n = len(rs)
        n_with_oos = sum(1 for r in rs if r["input_has_oos"])
        total_chars = sum(r["input_chars"] for r in rs)
        total_tokens = sum(r["input_tokens"] for r in rs)
        total_sim_pruned = sum(r["input_sim_pruned_tokens"] for r in rs)
        return {
            "n_inputs": n,
            "fraction_with_oos_token": (n_with_oos / n) if n else 0.0,
            "fertility_full_tokens_per_char": (total_tokens / total_chars) if total_chars else 0.0,
            "fertility_sim_pruned_tokens_per_char": (total_sim_pruned / total_chars) if total_chars else 0.0,
        }

    input_side_tiers = [t for t in ("arxiv-paste", "pdf-paste") if any(r["tier"] == t for r in rows)]
    input_side_summary = {}
    all_input_rows = [r for r in rows if r["tier"] in ("arxiv-paste", "pdf-paste")]
    for t in input_side_tiers:
        input_side_summary[t] = input_agg([r for r in rows if r["tier"] == t])
    if all_input_rows:
        input_side_summary["overall"] = input_agg(all_input_rows)
    input_side_top_misses = [
        {"id": tid, "decoded": token_display(tok, tid), "count": c}
        for tid, c in input_oos_id_counts.most_common(40)
    ]

    model_summary = {}
    for tm, rs in per_model_rows.items():
        total_tok = sum(r["tokens"] for r in rs)
        total_oos = sum(r["out_of_set_tokens"] for r in rs)
        model_summary[tm] = {
            "n_outputs": len(rs),
            "oos_rate_micro": (total_oos / total_tok) if total_tok else 0.0,
            "total_tokens": total_tok,
            "total_oos_tokens": total_oos,
        }

    top_oos = [{"id": tid, "decoded": token_display(tok, tid), "count": c} for tid, c in oos_counts.most_common(40)]

    return {
        "reference_bit_exact_fraction": len(ref_exact_items) / len(items) if items else 0.0,
        "reference_bit_exact_items": ref_exact_items,
        "summary_by_tier": summary,
        "target_model_output_summary": model_summary,
        "oos_top_tokens": top_oos,
        "input_side_held_out": {
            "by_tier": input_side_summary,
            "top_miss_tokens": input_side_top_misses,
        },
        "per_item": rows,
    }


def compute_padding_for_target(base_keep_ids, regular_ids, k_natural, K):
    """Walk regular_ids in ascending-id order (documented merge-rank
    frequency proxy) and add each one to the padding set -- including ones
    that happen to already be in base_keep_ids -- until the union reaches
    K. Shared between vocab-coverage.py's measurement pass and
    rebuild-tokenizer.py's Step 5 rebuild, so both compute the *same*
    padding set for a given (tokenizer, corpus, target K)."""
    if K <= k_natural:
        return set(), (K < k_natural)
    pad_ids = set()
    new_count = 0
    for tid in regular_ids:
        if tid in pad_ids:
            continue
        pad_ids.add(tid)
        if tid not in base_keep_ids:
            new_count += 1
        if k_natural + new_count >= K:
            break
    return pad_ids, False


def load_conv_template(tokenizer_path):
    tok_dir = Path(tokenizer_path).parent
    conv_path = tok_dir / "mlc-chat-config.json"
    if conv_path.exists():
        return json.load(open(conv_path))["conv_template"]
    return None


def build_keep_set_for_tokenizer(tokenizer_path, corpus_labels):
    """One-shot, importable entry point used by both vocab-coverage.py
    (measurement) and rebuild-tokenizer.py (Step 5): loads the tokenizer,
    the selected corpora, judged-correct outputs, and all seed/harness
    layers, and returns everything needed to build a keep-set for any
    target K deterministically. Guarantees the rebuild step prunes exactly
    the keep-set that was measured, not a re-derived approximation of it."""
    tok, tok_path = load_tokenizer(tokenizer_path)
    items, per_corpus_counts, conflicts = load_corpus_items(corpus_labels)
    item_ids = set(items.keys())
    outputs_by_item, judged_file_counts = load_judged_outputs(item_ids)
    all_seqs, by_item = build_sequences_extended(tok, items, outputs_by_item)
    all_counts = collections.Counter()
    for s in all_seqs:
        all_counts.update(s["ids"])
    conv = load_conv_template(tok_path)
    layers, meta = compute_keep_layers(tok, all_counts, conv)
    base_keep_ids, marginal = union_with_marginal(layers)
    k_natural = len(base_keep_ids)
    byte_ids = meta["byte_ids"]
    added_ids = meta["added_ids"]
    regular_ids = get_regular_token_ids(tok, byte_ids, added_ids)
    return {
        "tok": tok,
        "tok_path": tok_path,
        "items": items,
        "by_item": by_item,
        "all_counts": all_counts,
        "base_keep_ids": base_keep_ids,
        "always_keep_ids": meta["always_keep_ids"],
        "marginal": marginal,
        "k_natural": k_natural,
        "regular_ids": regular_ids,
        "meta": meta,
        "conv": conv,
        "per_corpus_counts": per_corpus_counts,
        "conflicts": conflicts,
        "judged_file_counts": judged_file_counts,
    }


def final_keep_set_for_K(built, K):
    pad_ids, clipped = compute_padding_for_target(built["base_keep_ids"], built["regular_ids"], built["k_natural"], K)
    return built["base_keep_ids"] | pad_ids, pad_ids, clipped


def run_extended(args):
    if not args.tokenizer:
        raise SystemExit("--tokenizer is required in extended mode (with --tokenizer-label)")
    label = args.tokenizer_label
    out_path = ROOT / f"results-vocab-coverage-{label}.json"

    selected_corpora = args.corpus if args.corpus else sorted(CORPUS_FILES.keys())
    built = build_keep_set_for_tokenizer(args.tokenizer, selected_corpora)
    tok, tok_path = built["tok"], built["tok_path"]
    items, by_item, all_counts = built["items"], built["by_item"], built["all_counts"]
    per_corpus_counts, conflicts, judged_file_counts = built["per_corpus_counts"], built["conflicts"], built["judged_file_counts"]
    base_keep_ids, marginal, k_natural = built["base_keep_ids"], built["marginal"], built["k_natural"]
    regular_ids, meta, conv = built["regular_ids"], built["meta"], built["conv"]
    if conv is None:
        print(f"warning: no mlc-chat-config.json next to {tok_path}; harness-prompt contribution will be empty")

    target_sizes = sorted(set(args.target_size if args.target_size else EXTENDED_TARGET_SIZES) | {k_natural})

    non_corpus_extra_ids = built["always_keep_ids"]  # byte/added/special/harness/seeds, never removed per-item
    train_seen_by_item = precompute_train_seen(items, by_item, all_counts)

    by_size = {}
    final_keep_by_size = {}
    for K in target_sizes:
        final_keep, pad_ids, clipped = final_keep_set_for_K(built, K)
        final_keep_by_size[K] = final_keep
        by_size[str(K)] = {
            "target_size": K,
            "actual_size": len(final_keep),
            "clipped_to_natural": clipped,
            "padding_ids_walked": len(pad_ids),
            "padding_new_ids": len(final_keep) - k_natural,
        }
        loo = eval_leave_one_out_extended(
            tok, items, by_item, train_seen_by_item,
            fixed_extra_ids=non_corpus_extra_ids,
            padding_ids=pad_ids,
            target_models=set(TARGET_MODEL_ALIASES.values()),
        )
        by_size[str(K)]["leave_one_out"] = loo

    print(f"tokenizer: {tok_path} (label={label})")
    print(f"vocab size (with added): {tok.get_vocab_size(with_added_tokens=True)}")
    print(f"corpora selected: {selected_corpora}")
    for c, counts in per_corpus_counts.items():
        print(f"  {c}: raw={counts['raw']} used={counts['used']} skipped_not_ok={counts['skipped_not_ok']}")
    if conflicts:
        print(f"id conflicts across corpora (kept first occurrence): {len(conflicts)}")
    for fname, c in judged_file_counts.items():
        print(f"judged file {fname}: {c['n_rows']} rows, {c['n_correct_in_scope']} correct+in-scope")
    print(f"items total: {len(items)}")
    print(f"\nkeep-set layer contributions (marginal = new ids beyond prior layers):")
    for m in marginal:
        print(f"  {m['layer']}: layer_size={m['layer_size']} new_ids={m['new_ids']}")
    print(f"natural (no-padding) keep-set size: {k_natural}")
    print(f"\nTarget sizes: {target_sizes}")
    print("\n| K_target | K_actual | clipped | padding_new_ids | ref_oos_micro | ref_bit_exact_frac | all_eval_oos_micro |")
    print("|---:|---:|---|---:|---:|---:|---:|")
    for K in target_sizes:
        e = by_size[str(K)]
        loo = e["leave_one_out"]
        ov = loo["summary_by_tier"]["overall"]
        print(
            f"| {K} | {e['actual_size']} | {e['clipped_to_natural']} | {e['padding_new_ids']} | "
            f"{ov['reference_oos_rate_micro']:.4f} | {ov['reference_bit_exact_fraction']:.4f} | "
            f"{ov['oos_rate_micro_all_eval']:.4f} |"
        )

    print("\nTarget-model output OOS rate (real generated text, correct-judged rows only):")
    for K in target_sizes:
        loo = by_size[str(K)]["leave_one_out"]
        print(f"  K={K}:")
        for tm, s in loo["target_model_output_summary"].items():
            print(f"    {tm}: n={s['n_outputs']} oos_micro={s['oos_rate_micro']:.4f}")

    print("\nInput-side (real held-out arxiv/pdf `input` text, leave-one-out keep-set), fraction with >=1 OOS token, sim fertility:")
    for K in target_sizes:
        print(f"  K={K}:")
        for tier, r in by_size[str(K)]["leave_one_out"]["input_side_held_out"]["by_tier"].items():
            print(
                f"    {tier}: n={r['n_inputs']} frac_oos={r['fraction_with_oos_token']:.4f} "
                f"fert_full={r['fertility_full_tokens_per_char']:.4f} fert_sim_pruned={r['fertility_sim_pruned_tokens_per_char']:.4f}"
            )

    payload = {
        "tokenizer_path": tok_path,
        "tokenizer_label": label,
        "vocab_size_with_added": tok.get_vocab_size(with_added_tokens=True),
        "corpora_selected": selected_corpora,
        "per_corpus_counts": per_corpus_counts,
        "id_conflicts": conflicts,
        "judged_file_counts": judged_file_counts,
        "n_items": len(items),
        "keep_set_layers_marginal": marginal,
        "natural_keep_set_size": k_natural,
        "target_sizes": target_sizes,
        "by_target_size": by_size,
        "seed_meta": {
            "n_latex_seed_strings": meta["n_latex_seed_strings"],
            "n_mathsenglish_seed_words": meta["n_mathsenglish_seed_words"],
            "n_harness_prompts": meta["n_harness_prompts"],
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out_path}")


def main():
    args = parse_args()
    if args.tokenizer_label:
        run_extended(args)
    else:
        run_legacy(args)


if __name__ == "__main__":
    main()
