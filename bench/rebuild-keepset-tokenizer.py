#!/usr/bin/env python3
"""Step 5/6: rebuild a tokenizer in the compact keep-set id space and emit
the frozen bench/keepsets/<label>-<K>/ artifact directory.

Uses the exact same keep-set construction as bench/vocab-coverage.py
(imported directly, since a rebuild that used a re-derived approximation of
the keep-set instead of the one that was actually measured would defeat the
point of measuring it first).

new_id = rank of the original id in the ascending sort of the keep-set.
This is the deterministic remap the model agents will apply to embedding
and lm_head rows, so `keep-idx.json`'s order *is* the contract.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEEPSETS_ROOT = ROOT / "keepsets"

# Files whose SHA-256 + item/row count belong in every provenance.md,
# independent of which tokenizer/K is being emitted -- the corpus and seed
# inputs are the same for every rebuild.
PROVENANCE_INPUT_FILES = [
    ("bench-data.json", "json"),
    ("bench-data-extended.json", "json"),
    ("pdf-pastes.json", "json"),
    ("arxiv-pastes.json", "json"),
    ("synth-spans.jsonl", "jsonl"),
    ("judged-arxiv-2026-09-07.json", "json"),
    ("judged-text-tiers-2026-09-07.json", "json"),
    ("keepset-harness-prompts.json", "json-dict"),
    ("keepset-seed-latex.json", "json-dict"),
    ("keepset-seed-mathsenglish.json", "json-dict"),
]


def git_head_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "UNKNOWN (git rev-parse HEAD failed; run after committing the scripts)"


def describe_input_file(fname):
    path = ROOT / fname
    if not path.exists():
        return {"file": fname, "present": False}
    h = sha256_file(path)
    n = None
    if fname.endswith(".jsonl"):
        n = sum(1 for _ in open(path))
    else:
        d = json.loads(path.read_text())
        if isinstance(d, list):
            n = len(d)
        elif isinstance(d, dict):
            n = None  # seed/harness files: count reported by their own provenance below
    return {"file": fname, "present": True, "sha256": h, "n_rows_or_items": n}

_spec = importlib.util.spec_from_file_location("vocab_coverage", ROOT / "vocab-coverage.py")
vocab_coverage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vocab_coverage)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="path to the original tokenizer.json (mlc-chat-config.json must live alongside it)")
    ap.add_argument("--tokenizer-label", required=True, help="label used for the bench/keepsets/<label>-<K>/ directory name")
    ap.add_argument("--corpus", action="append", default=None, help="corpus to include (repeatable); union of all if omitted")
    ap.add_argument("--target-size", type=int, action="append", required=True, help="target K (repeatable)")
    return ap.parse_args()


def sha256_file(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def split_merge(m):
    if isinstance(m, str):
        a, b = m.split(" ", 1)
        return a, b
    return m[0], m[1]


def join_merge(a, b, as_list):
    return [a, b] if as_list else f"{a} {b}"


def rebuild_tokenizer_json(raw, keep_ids, new_id_map):
    vocab = raw["model"]["vocab"]

    new_vocab = {tok: new_id_map[old_id] for tok, old_id in vocab.items() if old_id in keep_ids}

    # Not every kept id has a model.vocab entry: some tokenizers (MiniCPM5)
    # carry pure "added token" ids (control tokens, `<unused_token_N>`
    # placeholders) above the regular BPE vocab range that only ever appear
    # in the top-level `added_tokens` list, never in model.vocab. Those are
    # handled by the added_tokens rebuild below, so the coverage check here
    # is against vocab-entries UNION added-token ids, not vocab alone.
    added_ids = {e["id"] for e in raw["added_tokens"]}
    covered = set(vocab.values()) | added_ids
    missing = sorted(i for i in keep_ids if i not in covered)
    if missing:
        raise SystemExit(
            f"BUG: {len(missing)} kept ids have no vocab string and no added-token entry at all "
            f"(sample: {missing[:10]})"
        )

    old_merges = raw["model"]["merges"]
    merges_as_list = bool(old_merges) and isinstance(old_merges[0], list)
    new_merges = []
    n_dropped = 0
    for m in old_merges:
        a, b = split_merge(m)
        merged = a + b
        a_id, b_id, m_id = vocab.get(a), vocab.get(b), vocab.get(merged)
        if a_id is None or b_id is None or m_id is None:
            n_dropped += 1
            continue
        if a_id in keep_ids and b_id in keep_ids and m_id in keep_ids:
            new_merges.append(m)
        else:
            n_dropped += 1

    new_added_tokens = []
    for entry in raw["added_tokens"]:
        old_id = entry["id"]
        if old_id not in keep_ids:
            raise SystemExit(f"BUG: added token {entry['content']!r} (id {old_id}) was dropped from the keep-set; added/special ids must always survive")
        e = dict(entry)
        e["id"] = new_id_map[old_id]
        new_added_tokens.append(e)
    new_added_tokens.sort(key=lambda e: e["id"])

    new_post_processor = json.loads(json.dumps(raw.get("post_processor")))  # deep copy
    if new_post_processor and new_post_processor.get("type") == "TemplateProcessing":
        for _, spec in new_post_processor.get("special_tokens", {}).items():
            spec["ids"] = [new_id_map[i] for i in spec["ids"]]

    new_raw = dict(raw)
    new_raw["added_tokens"] = new_added_tokens
    new_raw["post_processor"] = new_post_processor
    new_raw["model"] = dict(raw["model"])
    new_raw["model"]["vocab"] = new_vocab
    new_raw["model"]["merges"] = new_merges

    return new_raw, len(old_merges), len(new_merges), n_dropped


def rebuild_tokenizer_config(raw_cfg, keep_ids, new_id_map):
    new_cfg = json.loads(json.dumps(raw_cfg))
    old_atd = raw_cfg.get("added_tokens_decoder", {})
    new_atd = {}
    n_phantom_dropped = 0
    for old_id_str, spec in old_atd.items():
        old_id = int(old_id_str)
        if old_id not in keep_ids:
            # e.g. Qwen3.5's tokenizer_config.json declares 7 audio/tts
            # added tokens (ids 248070-248076) that do not exist in
            # tokenizer.json's own added_tokens list at all -- a pre-existing
            # mismatch between the two files (see provenance.md). Such a
            # ghost entry can never be produced by tok.encode() anyway, so
            # dropping it here is not a new loss, it is fixing the mismatch
            # so len(hf_tokenizer) == K exactly, per the task's requirement.
            n_phantom_dropped += 1
            continue
        s = dict(spec)
        new_atd[str(new_id_map[old_id])] = s
        new_cfg.setdefault("_id_remap_note", None)
    new_cfg["added_tokens_decoder"] = new_atd
    return new_cfg, n_phantom_dropped


def build_special_tokens_map(tok_cfg):
    out = {}
    for key in ("bos_token", "eos_token", "pad_token", "unk_token", "additional_special_tokens"):
        v = tok_cfg.get(key)
        if v is not None:
            out[key] = v
    return out


def remap_stop_token_ids(conv, vocab_str_to_old_id, new_id_map, keep_ids):
    """MLC's conv_template carries both `stop_str` (text) and
    `stop_token_ids` (old-space ids). Remap the latter through new_id_map --
    but if an original stop id doesn't correspond to the token `stop_str`
    says it should (a real mismatch found in Qwen3.5's shipped
    mlc-chat-config.json: stop_token_ids [151643, 151645] are old Qwen3
    151936-vocab ids that land on unrelated tokens in the actual 248320-vocab
    Qwen3.5 tokenizer, not <|endoftext|>/<|im_end|>), use the id that
    `stop_str` actually names instead of propagating the bug forward, and
    report the mismatch."""
    old_ids = conv.get("stop_token_ids", [])
    stop_strs = conv.get("stop_str", [])
    resolved_old_ids = []
    mismatches = []
    for i, old_id in enumerate(old_ids):
        want_str = stop_strs[i] if i < len(stop_strs) else None
        actual_str_for_id = None
        # reverse lookup: which vocab string currently has this old id
        # (vocab_str_to_old_id is str->id; invert lazily per call, small)
        for s, oid in vocab_str_to_old_id.items():
            if oid == old_id:
                actual_str_for_id = s
                break
        if want_str is not None and actual_str_for_id != want_str:
            correct_old_id = vocab_str_to_old_id.get(want_str)
            mismatches.append(
                {
                    "index": i,
                    "declared_stop_token_id": old_id,
                    "stop_str": want_str,
                    "token_at_declared_id": actual_str_for_id,
                    "corrected_old_id": correct_old_id,
                }
            )
            resolved_old_ids.append(correct_old_id if correct_old_id is not None else old_id)
        else:
            resolved_old_ids.append(old_id)

    new_ids = []
    for old_id in resolved_old_ids:
        if old_id is None:
            continue
        if old_id not in keep_ids:
            # stop tokens are always-special/added ids, which are always in
            # the keep-set by construction; this branch should not trigger,
            # but fail loudly rather than silently drop a stop id.
            raise SystemExit(f"BUG: resolved stop_token_id {old_id} is not in the keep-set")
        new_ids.append(new_id_map[old_id])
    return new_ids, mismatches


def verify_and_measure_fertility(old_tok, new_tok, new_id_map, keep_ids, texts):
    """Per text: if every old-space token id is in the keep-set, assert
    remap(old_encode) == new_encode exactly and new_decode(new_encode) ==
    text. Otherwise, measure the REAL fertility delta by actually
    re-encoding with the rebuilt tokenizer (not simulated)."""
    n_exact_total = 0
    n_exact_pass = 0
    failures = []
    fertility_rows = []
    for text in texts:
        if not text:
            continue
        old_ids = old_tok.encode(text, add_special_tokens=False).ids
        all_in = all(i in keep_ids for i in old_ids)
        new_ids = new_tok.encode(text, add_special_tokens=False).ids
        if all_in:
            n_exact_total += 1
            expected = [new_id_map[i] for i in old_ids]
            decoded = new_tok.decode(new_ids, skip_special_tokens=False)
            if expected == new_ids and decoded == text:
                n_exact_pass += 1
            else:
                failures.append(
                    {
                        "text": text[:200],
                        "expected_ids": expected[:40],
                        "got_ids": new_ids[:40],
                        "decoded": decoded[:200],
                        "roundtrip_ok": decoded == text,
                        "ids_ok": expected == new_ids,
                    }
                )
        else:
            fertility_rows.append(
                {
                    "chars": len(text),
                    "old_tokens": len(old_ids),
                    "new_tokens": len(new_ids),
                }
            )
    total_chars = sum(r["chars"] for r in fertility_rows)
    total_old = sum(r["old_tokens"] for r in fertility_rows)
    total_new = sum(r["new_tokens"] for r in fertility_rows)
    return {
        "exact_match_total": n_exact_total,
        "exact_match_passed": n_exact_pass,
        "exact_match_failures": failures,
        "oos_texts_n": len(fertility_rows),
        "oos_texts_fertility_before": (total_old / total_chars) if total_chars else 0.0,
        "oos_texts_fertility_after_real": (total_new / total_chars) if total_chars else 0.0,
    }


def real_held_out_fertility(built, raw, final_keep, pad_ids, tiers=("arxiv-paste", "pdf-paste")):
    """The REAL (not simulated) input-side fertility delta: for each
    arxiv-paste/pdf-paste item, rebuild a miniature vocab+merges (leave-one-
    out: this item's own corpus contribution removed, exactly like
    vocab-coverage.py's LOO keep-set for the same target K) and actually
    re-encode that item's `input` text with it, rather than approximating a
    miss as its byte length. This is the genuine held-out measurement the
    task brief asks for; the coverage.json fertility numbers elsewhere in
    this repo (from vocab-coverage.py) are the simulated upper bound."""
    from tokenizers import Tokenizer

    items, by_item, all_counts = built["items"], built["by_item"], built["all_counts"]
    non_corpus_extra_ids = built["always_keep_ids"]
    train_seen_by_item = vocab_coverage.precompute_train_seen(items, by_item, all_counts)
    old_tok = built["tok"]

    rows = []
    for iid, it in items.items():
        if it["tier"] not in tiers:
            continue
        loo_keep = train_seen_by_item[iid] | non_corpus_extra_ids | pad_ids
        loo_idx = sorted(loo_keep)
        loo_id_map = {old: i for i, old in enumerate(loo_idx)}
        new_raw, _, _, _ = rebuild_tokenizer_json(raw, loo_keep, loo_id_map)
        loo_tok = Tokenizer.from_str(json.dumps(new_raw))
        text = it["input"]
        old_ids = old_tok.encode(text, add_special_tokens=False).ids
        new_ids = loo_tok.encode(text, add_special_tokens=False).ids
        rows.append(
            {
                "item": iid,
                "tier": it["tier"],
                "chars": len(text),
                "tokens_before": len(old_ids),
                "tokens_after_real": len(new_ids),
            }
        )
    total_chars = sum(r["chars"] for r in rows)
    total_before = sum(r["tokens_before"] for r in rows)
    total_after = sum(r["tokens_after_real"] for r in rows)
    by_tier = {}
    for t in tiers:
        rs = [r for r in rows if r["tier"] == t]
        tc = sum(r["chars"] for r in rs)
        tb = sum(r["tokens_before"] for r in rs)
        ta = sum(r["tokens_after_real"] for r in rs)
        by_tier[t] = {
            "n": len(rs),
            "fertility_before": (tb / tc) if tc else 0.0,
            "fertility_after_real": (ta / tc) if tc else 0.0,
        }
    return {
        "n_items": len(rows),
        "fertility_before_tokens_per_char": (total_before / total_chars) if total_chars else 0.0,
        "fertility_after_real_tokens_per_char": (total_after / total_chars) if total_chars else 0.0,
        "by_tier": by_tier,
        "per_item": rows,
    }


def build_provenance_md(args, K, actual_K, coverage, tok_prov_entry, hashes, script_sha, exact_cmd):
    lines = []
    lines.append(f"# Provenance: {args.tokenizer_label}-{actual_K}")
    lines.append("")
    lines.append(f"Generated by `bench/rebuild-keepset-tokenizer.py` at script commit `{script_sha}`.")
    lines.append("")
    lines.append("## Exact command line")
    lines.append("")
    lines.append("```")
    lines.append(exact_cmd)
    lines.append("```")
    lines.append("")
    lines.append("## Tokenizer source")
    lines.append("")
    lines.append(json.dumps(tok_prov_entry, indent=2))
    lines.append("")
    lines.append("## Corpus and seed inputs (this rebuild's `--corpus` selection: "
                 f"{args.corpus if args.corpus else 'all (default)'})")
    lines.append("")
    lines.append("| file | sha256 | n_rows/items |")
    lines.append("|---|---|---:|")
    for fname, _ in PROVENANCE_INPUT_FILES:
        d = describe_input_file(fname)
        if not d.get("present"):
            lines.append(f"| {fname} | (missing) | - |")
            continue
        n = d["n_rows_or_items"]
        lines.append(f"| {fname} | `{d['sha256']}` | {n if n is not None else 'see provenance in file'} |")
    lines.append("")
    lines.append("## Seed list counts")
    lines.append("")
    try:
        latex_seed = json.load(open(ROOT / "keepset-seed-latex.json"))
        lines.append(f"- LaTeX macro/environment/delimiter seed: {latex_seed['provenance']['total_count']} strings "
                      f"(KaTeX {latex_seed['provenance']['katex_version']}, "
                      f"{latex_seed['provenance']['programmatic_count']} extracted programmatically + "
                      f"{latex_seed['provenance']['hand_added_count']} hand-added document-level delimiters)")
    except Exception as e:
        lines.append(f"- LaTeX seed: (error reading provenance: {e})")
    try:
        me_seed = json.load(open(ROOT / "keepset-seed-mathsenglish.json"))
        p = me_seed["provenance"]
        lines.append(f"- Maths-English word seed: {p['n_word_types']} word types by document frequency over "
                      f"{p['n_documents_total']} documents across {len(p['sources'])} corpus files. {p['limitation']}")
    except Exception as e:
        lines.append(f"- Maths-English seed: (error reading provenance: {e})")
    try:
        harness = json.load(open(ROOT / "keepset-harness-prompts.json"))
        p = harness["provenance"]
        lines.append(f"- Harness prompts: {len(harness['fresh_items_prompts'])} FRESH_ITEMS + "
                      f"{len(harness['quality_corpus_prompts'])} QUALITY_CORPUS items, source "
                      f"`{p['source_repo']}@{p['source_commit']}` file `{p['source_file']}` "
                      f"(last touched at `{p['source_file_last_commit']}`)")
    except Exception as e:
        lines.append(f"- Harness prompts: (error reading provenance: {e})")
    lines.append("")
    lines.append("## Measured rates (this K)")
    lines.append("")
    lines.append(f"- target K: {K}, actual size achieved: {actual_K}, clipped to natural: {coverage['clipped_to_natural']}")
    lines.append(f"- natural (no-padding) keep-set size for this tokenizer: {coverage['natural_keep_set_size']}")
    m = coverage["merges"]
    lines.append(f"- BPE merges: {m['before']} -> {m['after']} (dropped {m['dropped']}, {m['dropped_fraction']:.4f})")
    lines.append(f"- tokenizer_config.json phantom added-tokens dropped: {coverage['tokenizer_config_phantom_added_tokens_dropped']}")
    v = coverage["verify_all_corpus_texts"]
    lines.append(f"- rebuild verification (all corpus texts, in-set-only subset): {v['exact_match_passed']}/{v['exact_match_total']} exact "
                 f"({(v['exact_match_passed']/v['exact_match_total'] if v['exact_match_total'] else 0):.4f}), "
                 f"{len(v['exact_match_failures'])} failures")
    iv = coverage["verify_input_side_arxiv_pdf"]
    lines.append(f"- rebuild verification (arxiv/pdf input subset, in-set-only): {iv['exact_match_passed']}/{iv['exact_match_total']} exact")
    rf = coverage["real_held_out_fertility_arxiv_pdf"]
    lines.append(f"- REAL held-out fertility (arxiv/pdf input, leave-one-out rebuild, n={rf['n_items']}): "
                 f"before={rf['fertility_before_tokens_per_char']:.4f} tok/char, "
                 f"after={rf['fertility_after_real_tokens_per_char']:.4f} tok/char "
                 f"({(rf['fertility_after_real_tokens_per_char']/rf['fertility_before_tokens_per_char'] - 1 if rf['fertility_before_tokens_per_char'] else 0):+.2%} relative)")
    if coverage["stop_token_id_mismatches_found"]:
        lines.append(f"- stop_token_id mismatches found and corrected: {json.dumps(coverage['stop_token_id_mismatches_found'])}")
    lines.append("")
    lines.append("## Emitted file hashes")
    lines.append("")
    lines.append("| file | sha256 |")
    lines.append("|---|---|")
    for f, h in hashes.items():
        lines.append(f"| {f} | `{h}` |")
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    from tokenizers import Tokenizer

    corpus_labels = args.corpus if args.corpus else sorted(vocab_coverage.CORPUS_FILES.keys())
    built = vocab_coverage.build_keep_set_for_tokenizer(args.tokenizer, corpus_labels)
    tok, tok_path = built["tok"], built["tok_path"]
    items = built["items"]

    tok_dir = Path(tok_path).parent
    raw = json.loads(Path(tok_path).read_text())
    tok_cfg_path = tok_dir / "tokenizer_config.json"
    raw_cfg = json.loads(tok_cfg_path.read_text()) if tok_cfg_path.exists() else {}
    mlc_cfg_path = tok_dir / "mlc-chat-config.json"
    mlc_cfg = json.loads(mlc_cfg_path.read_text()) if mlc_cfg_path.exists() else {}

    vocab_str_to_old_id = raw["model"]["vocab"]
    # added tokens (e.g. <|im_start|>) live outside model.vocab for some
    # tokenizers (Qwen) but inside it for others (MiniCPM); build one
    # complete string->old_id map covering both so stop-token resolution
    # works regardless of scheme.
    full_str_to_old_id = dict(vocab_str_to_old_id)
    for e in raw["added_tokens"]:
        full_str_to_old_id[e["content"]] = e["id"]

    for K in args.target_size:
        final_keep, pad_ids, clipped = vocab_coverage.final_keep_set_for_K(built, K)
        keep_idx = sorted(final_keep)
        new_id_map = {old: i for i, old in enumerate(keep_idx)}
        actual_K = len(keep_idx)

        new_raw, n_merges_before, n_merges_after, n_merges_dropped = rebuild_tokenizer_json(raw, final_keep, new_id_map)
        new_cfg, n_phantom_dropped = rebuild_tokenizer_config(raw_cfg, final_keep, new_id_map) if raw_cfg else ({}, 0)
        special_tokens_map = build_special_tokens_map(raw_cfg) if raw_cfg else {}

        new_conv_stop_ids, stop_id_mismatches = ([], [])
        if mlc_cfg:
            new_conv_stop_ids, stop_id_mismatches = remap_stop_token_ids(
                mlc_cfg["conv_template"], full_str_to_old_id, new_id_map, final_keep
            )

        out_dir = KEEPSETS_ROOT / f"{args.tokenizer_label}-{actual_K}"
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "keep-idx.json").write_text(json.dumps(keep_idx) + "\n")
        (out_dir / "tokenizer.json").write_text(json.dumps(new_raw, ensure_ascii=False))
        if raw_cfg:
            (out_dir / "tokenizer_config.json").write_text(json.dumps(new_cfg, indent=2, ensure_ascii=False) + "\n")
        (out_dir / "special_tokens_map.json").write_text(json.dumps(special_tokens_map, indent=2, ensure_ascii=False) + "\n")

        config_patch = {
            "vocab_size": actual_K,
            "active_vocab_size": actual_K,
            "stop_token_ids": new_conv_stop_ids,
            "stop_token_ids_source_mismatches": stop_id_mismatches,
        }
        (out_dir / "config-patch.json").write_text(json.dumps(config_patch, indent=2) + "\n")

        # Verify by loading the actual rebuilt tokenizer.json back through
        # the tokenizers library (not just the dict we built in memory).
        new_tok = Tokenizer.from_str(json.dumps(new_raw))
        if new_tok.get_vocab_size(with_added_tokens=True) != actual_K:
            print(
                f"ANOMALY: rebuilt tokenizer reports "
                f"{new_tok.get_vocab_size(with_added_tokens=True)} entries, expected {actual_K}"
            )

        texts = []
        for iid, it in items.items():
            texts.append(it["input"])
            texts.append(it["reference"])
        verify = verify_and_measure_fertility(tok, new_tok, new_id_map, final_keep, texts)

        input_texts = [it["input"] for it in items.values() if it["tier"] in ("arxiv-paste", "pdf-paste")]
        input_verify = verify_and_measure_fertility(tok, new_tok, new_id_map, final_keep, input_texts)

        real_fertility = real_held_out_fertility(built, raw, final_keep, pad_ids)

        coverage = {
            "tokenizer_label": args.tokenizer_label,
            "tokenizer_path": tok_path,
            "corpora": corpus_labels,
            "target_size": K,
            "actual_size": actual_K,
            "clipped_to_natural": clipped,
            "natural_keep_set_size": built["k_natural"],
            "merges": {
                "before": n_merges_before,
                "after": n_merges_after,
                "dropped": n_merges_dropped,
                "dropped_fraction": (n_merges_dropped / n_merges_before) if n_merges_before else 0.0,
            },
            "tokenizer_config_phantom_added_tokens_dropped": n_phantom_dropped,
            "verify_all_corpus_texts": verify,
            "verify_input_side_arxiv_pdf": input_verify,
            "real_held_out_fertility_arxiv_pdf": real_fertility,
            "stop_token_id_mismatches_found": stop_id_mismatches,
        }
        (out_dir / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")

        files = ["keep-idx.json", "tokenizer.json", "special_tokens_map.json", "config-patch.json", "coverage.json"]
        if raw_cfg:
            files.append("tokenizer_config.json")
        hashes = {f: sha256_file(out_dir / f) for f in files}

        tok_prov = json.loads((ROOT / "tokenizer-provenance.json").read_text())
        tok_prov_entry = tok_prov.get(args.tokenizer_label) or {
            "note": f"no manual entry for label {args.tokenizer_label!r} in bench/tokenizer-provenance.json",
            "tokenizer_path": tok_path,
        }
        script_sha = git_head_sha()
        exact_cmd = (
            f"python3 bench/rebuild-keepset-tokenizer.py --tokenizer {args.tokenizer} "
            f"--tokenizer-label {args.tokenizer_label} "
            + " ".join(f"--corpus {c}" for c in (args.corpus or []))
            + " ".join(f"--target-size {t}" for t in args.target_size)
        )
        provenance_md = build_provenance_md(args, K, actual_K, coverage, tok_prov_entry, hashes, script_sha, exact_cmd)
        (out_dir / "provenance.md").write_text(provenance_md)
        hashes["provenance.md"] = sha256_file(out_dir / "provenance.md")
        (out_dir / "SHA256SUMS.txt").write_text("\n".join(f"{h}  {f}" for f, h in hashes.items()) + "\n")

        print(f"=== {args.tokenizer_label} K={K} -> actual={actual_K} ===")
        print(f"  merges: {n_merges_before} -> {n_merges_after} (dropped {n_merges_dropped}, {coverage['merges']['dropped_fraction']:.4f})")
        print(f"  tokenizer_config phantom added-tokens dropped: {n_phantom_dropped}")
        print(f"  verify (all corpus texts, in-set only): {verify['exact_match_passed']}/{verify['exact_match_total']} passed, {len(verify['exact_match_failures'])} failures")
        print(f"  verify (arxiv/pdf input, in-set only): {input_verify['exact_match_passed']}/{input_verify['exact_match_total']} passed")
        print(
            f"  REAL held-out fertility (arxiv/pdf input, leave-one-out rebuild, n={real_fertility['n_items']}): "
            f"before={real_fertility['fertility_before_tokens_per_char']:.4f} "
            f"after={real_fertility['fertility_after_real_tokens_per_char']:.4f}"
        )
        if stop_id_mismatches:
            print(f"  stop_token_id mismatches found and corrected: {stop_id_mismatches}")
        for f, h in hashes.items():
            print(f"  {h}  {f}")
        print(f"  wrote {out_dir}")


if __name__ == "__main__":
    main()
