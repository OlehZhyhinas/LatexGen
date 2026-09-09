#!/usr/bin/env python3
"""Emit bench/keepsets-v5/<label>-<K>/ for the winning strategy (C+F) at its
optimum K and one K either side, for both tokenizers. Reuses
build-keepset-v5.py's context/keep-set construction (never re-derives it)
and rebuild-keepset-tokenizer.py's tokenizer-surgery functions (same as
every prior generation), so the emitted artifact is exactly what Job 1's
grid measured, not a re-derived approximation of it.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEEPSETS_V5_ROOT = ROOT / "keepsets-v5"

import importlib.util


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v5 = _load_module("build_keepset_v5", "build-keepset-v5.py")
rk = v5.rk


def sha256_file(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


EMIT_PLAN = {
    "qwen35": [32768, 49152, 65536],
    "minicpm5-2b": [32768, 49152, 65536],
}
STRATEGY = "C+F"


def emit_one(label, K, ctx):
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    final_keep, meta = v5.build_keepset(STRATEGY, label, K, ctx)
    actual_K = len(final_keep)
    keep_idx = sorted(final_keep)
    new_id_map = {old: i for i, old in enumerate(keep_idx)}

    raw = ctx["raw"]
    tok_path = ctx["tok_path"]
    tok_dir = Path(tok_path).parent
    raw_cfg_path = tok_dir / "tokenizer_config.json"
    raw_cfg = json.loads(raw_cfg_path.read_text()) if raw_cfg_path.exists() else {}
    mlc_cfg_path = tok_dir / "mlc-chat-config.json"
    mlc_cfg = json.loads(mlc_cfg_path.read_text()) if mlc_cfg_path.exists() else {}

    new_raw, n_before, n_after, n_dropped = rk.rebuild_tokenizer_json(raw, final_keep, new_id_map)
    surviving_strings = set(new_raw["model"]["vocab"].keys()) | {e["content"] for e in new_raw["added_tokens"]}
    new_cfg, n_phantom_dropped, n_extra_special_dropped = (
        rk.rebuild_tokenizer_config(raw_cfg, final_keep, new_id_map, surviving_strings) if raw_cfg else ({}, 0, 0)
    )
    special_tokens_map = rk.build_special_tokens_map(raw_cfg) if raw_cfg else {}

    full_str_to_old_id = dict(raw["model"]["vocab"])
    for e in raw["added_tokens"]:
        full_str_to_old_id[e["content"]] = e["id"]
    new_conv_stop_ids, stop_id_mismatches = ([], [])
    if mlc_cfg:
        new_conv_stop_ids, stop_id_mismatches = rk.remap_stop_token_ids(
            mlc_cfg["conv_template"], full_str_to_old_id, new_id_map, final_keep
        )

    out_dir = KEEPSETS_V5_ROOT / f"{label}-{actual_K}"
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

    new_tok = Tokenizer.from_str(json.dumps(new_raw))
    file_tok_len = Tokenizer.from_file(str(out_dir / "tokenizer.json")).get_vocab_size(with_added_tokens=True)
    hf_tok_len = len(AutoTokenizer.from_pretrained(str(out_dir)))
    length_check = {
        "expected_K": actual_K,
        "tokenizers_from_file_len": file_tok_len,
        "hf_autotokenizer_from_pretrained_len": hf_tok_len,
    }
    if file_tok_len != actual_K or hf_tok_len != actual_K:
        raise SystemExit(
            f"BUG: {out_dir} length check failed -- expected K={actual_K}, got "
            f"tokenizers.Tokenizer.from_file={file_tok_len}, transformers.AutoTokenizer.from_pretrained={hf_tok_len}"
        )

    texts = v5.load_texts()
    m240_before = v5.fertility_and_oos(ctx["tok"], final_keep, texts["eval240"])
    m240_after = v5.real_fertility_after(new_tok, texts["eval240"])
    mfull_before = v5.fertility_and_oos(ctx["tok"], final_keep, texts["held_out_full"])
    mfull_after = v5.real_fertility_after(new_tok, texts["held_out_full"])

    coverage = {
        "tokenizer_label": label,
        "tokenizer_path": tok_path,
        "keepset_version": "v5",
        "strategy": STRATEGY,
        "target_size": K,
        "actual_size": actual_K,
        "natural_size": meta["natural_size"],
        "closure_iterations": meta.get("closure_iterations"),
        "length_check": length_check,
        "merges": {
            "before": n_before, "after": n_after, "dropped": n_dropped,
            "dropped_fraction": (n_dropped / n_before) if n_before else 0.0,
        },
        "tokenizer_config_phantom_added_tokens_dropped": n_phantom_dropped,
        "tokenizer_config_extra_special_tokens_dropped": n_extra_special_dropped,
        "stop_token_id_mismatches_found": stop_id_mismatches,
        "held_out_fertility": {
            "eval240": {
                "n_texts": m240_before["n_texts"], "total_chars": m240_before["total_chars"],
                "oos_rate": m240_before["oos_rate"], "oos_tokens": m240_before["oos_tokens"],
                "fertility_before": m240_before["fertility_before"], "fertility_after": m240_after["fertility_after"],
                "inflation": (m240_after["fertility_after"] / m240_before["fertility_before"] - 1.0) if m240_before["fertility_before"] else 0.0,
            },
            "held_out_full": {
                "n_texts": mfull_before["n_texts"], "total_chars": mfull_before["total_chars"],
                "oos_rate": mfull_before["oos_rate"], "oos_tokens": mfull_before["oos_tokens"],
                "fertility_before": mfull_before["fertility_before"], "fertility_after": mfull_after["fertility_after"],
                "inflation": (mfull_after["fertility_after"] / mfull_before["fertility_before"] - 1.0) if mfull_before["fertility_before"] else 0.0,
            },
        },
        "force_add_accounting": ctx["F_accounting"],
        "corpus_meta": ctx["corpus_meta"],
        "text_that_informed_this_keepset": (
            "Structural force-adds (byte alphabet, added/special tokens, single-codepoint "
            "accented-Latin/Greek/math-operator/superscript-subscript-digit tokens, LaTeX macro "
            "completion from bench/keepset-seed-latex.json, LatexGen's own rendered prompt from "
            "public/pipeline.js + public/pdf-prompt.js) plus a ranking derived from "
            f"bench/corpus/freq-{label}.json (external science corpus: TeX/Physics/Stats/CS "
            "StackExchange, arXiv abstracts, OpenStax, science Wikipedia/Wikibooks -- PR #58, "
            "branch keepset-corpus). NONE of bench/arxiv-pastes.json or bench-data-extended.json "
            "was read to build this keep-set; those files are read ONLY in this coverage.json's "
            "held_out_fertility block, to MEASURE the result, never to build it."
        ),
    }
    (out_dir / "coverage.json").write_text(json.dumps(coverage, indent=2, default=str) + "\n")

    script_sha = rk.git_head_sha()
    lines = []
    lines.append(f"# Provenance: keep-set v5 -- {label}-{actual_K}")
    lines.append("")
    lines.append(f"Generated by `bench/emit-keepset-v5.py` (keep-set construction from "
                 f"`bench/build-keepset-v5.py`) at script commit `{script_sha}`.")
    lines.append("")
    lines.append("## Exact command line")
    lines.append("")
    lines.append("```")
    lines.append(f"python3 bench/emit-keepset-v5.py   # emits {label}-{K} among the fixed EMIT_PLAN")
    lines.append("```")
    lines.append("")
    lines.append("## Strategy: C+F")
    lines.append("")
    lines.append(
        "Corpus rank (bench/corpus/freq-" + label + ".json, PR #58) as the Layer-3 fill order, "
        "plus the v4 force-add set (single-codepoint accented-Latin/Greek/math-operator/"
        "superscript-subscript-digit tokens; LaTeX macro completion; LatexGen's own rendered "
        "prompt + the 7 named ids), plus the mandatory byte-alphabet/added/special-token layer. "
        "BPE merge-ancestor closure over the union is computed and folded in iteratively (not "
        "assumed) since corpus rank, unlike ascending-id merge-rank fill, is not automatically "
        "closure-safe."
    )
    lines.append("")
    lines.append("## What text informed this keep-set (the brief's separation requirement)")
    lines.append("")
    lines.append(coverage["text_that_informed_this_keepset"])
    lines.append("")
    lines.append("## Force-add accounting (this tokenizer, independent of K)")
    lines.append("")
    lines.append("```")
    lines.append(json.dumps(ctx["F_accounting"], indent=2, default=str))
    lines.append("```")
    lines.append("")
    lines.append("## Measured rates (this K, held-out -- never used to build this keep-set)")
    lines.append("")
    lines.append(f"- target K: {K}, actual size achieved: {actual_K}, natural (pre-corpus-fill) size: {meta['natural_size']}")
    m = coverage["merges"]
    lines.append(f"- BPE merges: {m['before']} -> {m['after']} (dropped {m['dropped']}, {m['dropped_fraction']:.4f})")
    lines.append(
        f"- length check: expected K={actual_K}, tokenizers.Tokenizer.from_file={file_tok_len}, "
        f"transformers.AutoTokenizer.from_pretrained={hf_tok_len}"
    )
    e = coverage["held_out_fertility"]["eval240"]
    f = coverage["held_out_fertility"]["held_out_full"]
    lines.append(
        f"- eval240 (120 arxiv-pastes.json input+reference pairs, n={e['n_texts']} texts): "
        f"OOS rate {e['oos_rate']:.4%}, fertility before={e['fertility_before']:.4f} "
        f"after={e['fertility_after']:.4f} tok/char (inflation {e['inflation']:+.4%})"
    )
    lines.append(
        f"- held_out_full (+ bench-data-extended.json, n={f['n_texts']} texts): "
        f"OOS rate {f['oos_rate']:.4%}, fertility before={f['fertility_before']:.4f} "
        f"after={f['fertility_after']:.4f} tok/char (inflation {f['inflation']:+.4%})"
    )
    if stop_id_mismatches:
        lines.append(f"- stop_token_id mismatches found and corrected: {json.dumps(stop_id_mismatches)}")
    lines.append("")
    lines.append(
        "See `SHA256SUMS.txt` in this directory for the authoritative, current SHA-256 of every "
        "emitted file (including this one)."
    )
    lines.append("")
    (out_dir / "provenance.md").write_text("\n".join(lines) + "\n")

    files = ["keep-idx.json", "tokenizer.json", "special_tokens_map.json", "config-patch.json", "coverage.json", "provenance.md"]
    if raw_cfg:
        files.append("tokenizer_config.json")
    hashes = {f: sha256_file(out_dir / f) for f in files}
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(f"{h}  {f}" for f, h in hashes.items()) + "\n")

    print(f"=== v5 {label} K={K} -> actual={actual_K} ===")
    print(f"  length check: K={actual_K}, tokenizers={file_tok_len}, transformers={hf_tok_len}")
    print(f"  eval240: oos={e['oos_rate']:.4%} inflation={e['inflation']:+.4%}")
    print(f"  held_out_full: oos={f['oos_rate']:.4%} inflation={f['inflation']:+.4%}")
    for fn, h in hashes.items():
        print(f"  {h}  {fn}")
    print(f"  wrote {out_dir}")
    return {"out_dir": str(out_dir), "actual_K": actual_K, "hashes": hashes}


def main():
    results = {}
    for label, ks in EMIT_PLAN.items():
        print(f"\n=== building context: {label} ===")
        ctx = v5.build_context(label)
        results[label] = {}
        for K in ks:
            r = emit_one(label, K, ctx)
            results[label][str(K)] = r
    out_path = ROOT / "results-keepset-v5-emit.json"
    out_path.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
