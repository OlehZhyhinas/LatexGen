#!/usr/bin/env python3
"""Keep-set v2: structural, layered selection (not merge-rank-dominated
frequency padding). See bench/README-text-gates.md's "Keep-set v2" section
for the full design rationale.

Extends v1's machinery (imported, not rewritten):
  - bench/vocab-coverage.py: load_tokenizer, get_byte_token_ids,
    get_regular_token_ids, load_conv_template, render_chatml, tokenize,
    harness_prompt_ids, token_display.
  - bench/rebuild-keepset-tokenizer.py: rebuild_tokenizer_json,
    rebuild_tokenizer_config, build_special_tokens_map,
    remap_stop_token_ids, verify_and_measure_fertility, sha256_file,
    sha256_bytes, git_head_sha, describe_input_file,
    PROVENANCE_INPUT_FILES, split_merge.

The keep-set itself is built entirely differently from v1 (no corpus
frequency, no leave-one-out): three mandatory layers computed from Unicode
properties, the tokenizer's own chat template, LatexGen's and the harness's
actual prompts, and BPE merge-ancestor closure (Layer 1); a wholesale
Unicode-script drop (Layer 2); and residual ascending-token-id (merge-rank)
fill (Layer 3). Every layer's slot cost is recorded for the accounting
table the brief requires.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEEPSETS_V2_ROOT = ROOT / "keepsets-v2"
ARXIV_PASTES_PATH = ROOT / "arxiv-pastes.json"
HARNESS_PROMPTS_PATH = ROOT / "keepset-harness-prompts.json"
LATEXGEN_PROMPTS_PATH = ROOT / "keepset-seed-latexgen-prompts.json"
SYSTEM_WORDLIST_PATH = Path("/usr/share/dict/words")

TARGET_SIZES = [32_768, 65_536]

RARE_WORD_PROBES = [
    "Nikolskii", "Randers", "Lissajous", "Kullback", "Hausdorff", "Chebyshev", "Sobolev",
]
MATHS_SYMBOL_PROBES = ["\u03c6", "\u2190", "\u2194", "\u2193", "\u00d7"]  # phi, <-, <->, down-arrow, x

# v1's natural (no-padding) keep-set sizes, for the accounting table's
# "vs v1" column (see bench/keepsets/{label}-65536/provenance.md).
V1_NATURAL_SIZE = {"qwen35": 6414, "minicpm5-2b": 7583}


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vc = _load_module("vocab_coverage_v2dep", "vocab-coverage.py")
rk = _load_module("rebuild_keepset_tokenizer_v2dep", "rebuild-keepset-tokenizer.py")
ku = _load_module("keepset_v2_unicode", "keepset-v2-unicode.py")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tokenizer-label", required=True)
    ap.add_argument("--target-size", type=int, action="append", default=None)
    return ap.parse_args()


# =============================================================================
# Layer 1: structural, must-keep
# =============================================================================

def latexgen_prompt_messages(seed, sample_text):
    system_prompt = seed["system_prompt"]
    convert_user_prefix = seed["convert_user_prefix"]
    pdf_hint = seed["pdf_hint"]
    fewshot = seed["pdf_fewshot"]

    def convert_user(text):
        return convert_user_prefix + text

    plain_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": convert_user(sample_text)},
    ]
    pdf_messages = [{"role": "system", "content": system_prompt + "\n\n" + pdf_hint}]
    for pair in fewshot:
        pdf_messages.append({"role": "user", "content": convert_user(pair["user"])})
        pdf_messages.append({"role": "assistant", "content": pair["assistant"]})
    pdf_messages.append({"role": "user", "content": convert_user(sample_text)})
    return [("plain", plain_messages), ("pdf", pdf_messages)]


def latexgen_and_harness_prompt_ids(tok, conv):
    """Every token id produced by rendering (a) LatexGen's own conversion
    prompt, PDF hint and few-shot pairs (public/pipeline.js +
    public/pdf-prompt.js, via keepset-seed-latexgen-prompts.json) and (b) the
    webnn-workbench harness's FRESH_ITEMS + QUALITY_CORPUS prompts (v1's
    keepset-harness-prompts.json, reused unchanged) through the tokenizer's
    own chat template, both enable_thinking shapes. This is Layer 1's fix
    for failure 1 and doubles as the 'chat template' token coverage (role
    tags, seps, think-block control tokens) since every render goes through
    render_chatml."""
    all_ids = set()
    n_renders = 0
    n_chars = 0
    if conv is None:
        return all_ids, n_renders, n_chars

    seed = json.load(open(LATEXGEN_PROMPTS_PATH))
    sample_texts = [
        "the integral from zero to infinity of e to the minus x squared dx",
        "the quadratic formula for a x squared plus b x plus c equals zero",
    ]
    for sample_text in sample_texts:
        for _, messages in latexgen_prompt_messages(seed, sample_text):
            for think_variant in (True, False):
                text = vc.render_chatml(conv, messages, think_variant)
                ids = vc.tokenize(tok, text)
                all_ids.update(ids)
                n_chars += len(text)
                n_renders += 1

    harness_ids, n_harness_prompts, n_harness_chars = vc.harness_prompt_ids(tok, conv)
    all_ids.update(harness_ids)
    n_renders += n_harness_prompts
    n_chars += n_harness_chars
    return all_ids, n_renders, n_chars


def compute_merge_map(raw):
    merge_map = {}
    for m in raw["model"]["merges"]:
        a, b = rk.split_merge(m)
        merge_map[a + b] = (a, b)
    return merge_map


def compute_merge_ancestor_closure(raw, merge_map, seed_ids):
    """For every kept id, keep both parts of the merge that produced it,
    transitively, until fixpoint. Applied as its own accounted layer, not a
    hard invariant baked silently into the others."""
    vocab = raw["model"]["vocab"]
    id_to_str = {v: k for k, v in vocab.items()}
    for e in raw.get("added_tokens", []):
        id_to_str.setdefault(e["id"], e["content"])

    keep = set(seed_ids)
    frontier = list(seed_ids)
    while frontier:
        nxt = []
        for tid in frontier:
            s = id_to_str.get(tid)
            if s is None:
                continue
            parents = merge_map.get(s)
            if not parents:
                continue
            for p in parents:
                pid = vocab.get(p)
                if pid is not None and pid not in keep:
                    keep.add(pid)
                    nxt.append(pid)
        frontier = nxt
    return keep


def decode_all_regular_tokens(tok, byte_ids, added_ids):
    """Decode every non-byte, non-added token id once; reused for Layer 1's
    ASCII<=3/maths classification and Layer 2's ASCII-letter classification,
    instead of decoding twice."""
    vocab_size = tok.get_vocab_size(with_added_tokens=True)
    text_by_id = {}
    for tid in range(vocab_size):
        if tid in byte_ids or tid in added_ids:
            continue
        txt = tok.decode([tid], skip_special_tokens=False)
        if txt == "":
            continue
        text_by_id[tid] = txt
    return text_by_id


def build_layer1(tok, tok_path, raw):
    byte_ids = vc.get_byte_token_ids(tok)
    added_decoder = tok.get_added_tokens_decoder()
    added_ids = set(added_decoder.keys())
    special_ids = {tid for tid, t in added_decoder.items() if getattr(t, "special", False)}

    conv = vc.load_conv_template(tok_path)
    prompt_ids, n_prompt_renders, n_prompt_chars = latexgen_and_harness_prompt_ids(tok, conv)

    text_by_id = decode_all_regular_tokens(tok, byte_ids, added_ids)
    ascii3_ids = {tid for tid, txt in text_by_id.items() if ku.is_short_ascii_token_text(txt)}
    maths_ids = {tid for tid, txt in text_by_id.items() if ku.is_maths_or_greek_text(txt)}

    layers_pre_closure = [
        ("byte_alphabet", byte_ids),
        ("added_and_special", added_ids | special_ids),
        ("chat_template_and_prompts", prompt_ids),
        ("ascii_le3_chars", ascii3_ids),
        ("maths_and_greek_symbols", maths_ids),
    ]
    marginal = []
    seen = set()
    for name, ids in layers_pre_closure:
        new = ids - seen
        marginal.append({"layer": name, "layer_size": len(ids), "new_ids": len(new)})
        seen |= ids
    pre_closure_ids = seen

    merge_map = compute_merge_map(raw)
    closure_ids = compute_merge_ancestor_closure(raw, merge_map, pre_closure_ids)
    closure_extra = closure_ids - pre_closure_ids
    marginal.append({"layer": "merge_ancestor_closure", "layer_size": len(closure_ids), "new_ids": len(closure_extra)})

    meta = {
        "byte_ids": byte_ids,
        "added_ids": added_ids,
        "special_ids": special_ids,
        "conv_template_present": conv is not None,
        "n_prompt_renders": n_prompt_renders,
        "n_prompt_chars": n_prompt_chars,
        "text_by_id": text_by_id,
    }
    return closure_ids, marginal, meta


# =============================================================================
# Layer 2: wholesale drop / Layer 3: residual merge-rank fill
# =============================================================================

def build_layer2_and_candidates(tok, layer1_ids, byte_ids, added_ids, text_by_id):
    dropped = set()
    candidates = []
    for tid, txt in text_by_id.items():
        if tid in layer1_ids:
            continue
        if ku.has_ascii_letter(txt):
            candidates.append(tid)
        else:
            dropped.add(tid)
    candidates.sort()
    return dropped, candidates


def layer3_fill_for_K(layer1_ids, candidates, K):
    natural = len(layer1_ids)
    if natural > K:
        return set(), 0, True, natural
    budget = K - natural
    chosen = candidates[:budget]
    return set(chosen), len(chosen), False, natural


# =============================================================================
# Emission (reuses rebuild-keepset-tokenizer.py's tokenizer-surgery functions)
# =============================================================================

def emit_for_K(args, raw, raw_cfg, mlc_cfg, tok, tok_path, layer1_ids, layer3_ids, K, layer_marginal, layer2_dropped_count, items_for_texts):
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    final_keep = layer1_ids | layer3_ids
    keep_idx = sorted(final_keep)
    new_id_map = {old: i for i, old in enumerate(keep_idx)}
    actual_K = len(keep_idx)

    new_raw, n_merges_before, n_merges_after, n_merges_dropped = rk.rebuild_tokenizer_json(raw, final_keep, new_id_map)
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

    out_dir = KEEPSETS_V2_ROOT / f"{args.tokenizer_label}-{actual_K}"
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

    texts = []
    for it in items_for_texts:
        texts.append(it["input"])
        texts.append(it["reference"])
    verify_all = rk.verify_and_measure_fertility(tok, new_tok, new_id_map, final_keep, texts)

    coverage = {
        "tokenizer_label": args.tokenizer_label,
        "tokenizer_path": tok_path,
        "keepset_version": "v2",
        "target_size": K,
        "actual_size": actual_K,
        "clipped_to_natural": actual_K < K,
        "layer1_natural_size": len(layer1_ids),
        "layer2_dropped_count": layer2_dropped_count,
        "layer3_admitted_count": len(layer3_ids),
        "layer_accounting_marginal": layer_marginal,
        "merges": {
            "before": n_merges_before,
            "after": n_merges_after,
            "dropped": n_merges_dropped,
            "dropped_fraction": (n_merges_dropped / n_merges_before) if n_merges_before else 0.0,
        },
        "tokenizer_config_phantom_added_tokens_dropped": n_phantom_dropped,
        "tokenizer_config_extra_special_tokens_dropped": n_extra_special_dropped,
        "length_check": length_check,
        "verify_all_corpus_texts": verify_all,
        "stop_token_id_mismatches_found": stop_id_mismatches,
    }
    (out_dir / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")

    script_sha = rk.git_head_sha()
    lines = []
    lines.append(f"# Provenance: keep-set v2 -- {args.tokenizer_label}-{actual_K}")
    lines.append("")
    lines.append(f"Generated by `bench/build-keepset-v2.py` at script commit `{script_sha}`.")
    lines.append("")
    lines.append("## Exact command line")
    lines.append("")
    lines.append("```")
    lines.append(f"python3 bench/build-keepset-v2.py --tokenizer {args.tokenizer} --tokenizer-label {args.tokenizer_label} --target-size {K}")
    lines.append("```")
    lines.append("")
    lines.append("## Selection method")
    lines.append("")
    lines.append(
        "Structural layered selection (not corpus-frequency merge-rank padding like v1): "
        "Layer 1 (structural must-keep: byte alphabet, added/special tokens, chat-template + "
        "LatexGen-prompt + harness-prompt tokens, all ASCII tokens <=3 characters, all maths/Greek "
        "symbol tokens, BPE merge-ancestor closure over all of the above) is deterministic and "
        "corpus-independent. Layer 2 (wholesale drop: no ASCII letter and not a Layer-1 maths "
        "symbol) is diagnostic/exclusionary only. Layer 3 (residual candidates -- ASCII-letter-"
        "bearing tokens not already kept -- ranked by ascending original token id, the same "
        "BPE-merge-rank proxy v1 used) fills the remaining budget up to K. See "
        "`bench/README-text-gates.md`'s \"Keep-set v2\" section for the full layer accounting."
    )
    lines.append("")
    lines.append("## Layer accounting (marginal = new ids beyond prior layers)")
    lines.append("")
    lines.append("| layer | layer_size | new_ids |")
    lines.append("|---|---:|---:|")
    for m in layer_marginal:
        lines.append(f"| {m['layer']} | {m['layer_size']} | {m['new_ids']} |")
    lines.append("")
    lines.append(f"- Layer 1 natural (no-padding) size: {len(layer1_ids)}")
    lines.append(f"- Layer 2 dropped wholesale: {layer2_dropped_count}")
    lines.append(f"- Layer 3 admitted at K={K}: {len(layer3_ids)}")
    lines.append(f"- target K: {K}, actual size achieved: {actual_K}, clipped to natural: {actual_K < K}")
    m = coverage["merges"]
    lines.append(f"- BPE merges: {m['before']} -> {m['after']} (dropped {m['dropped']}, {m['dropped_fraction']:.4f})")
    lines.append(f"- tokenizer_config.json phantom added-tokens dropped: {n_phantom_dropped}")
    lines.append(f"- tokenizer_config.json stale extra_special_tokens dropped: {n_extra_special_dropped}")
    lc = length_check
    lines.append(
        f"- length check: expected K={lc['expected_K']}, tokenizers.Tokenizer.from_file={lc['tokenizers_from_file_len']}, "
        f"transformers.AutoTokenizer.from_pretrained={lc['hf_autotokenizer_from_pretrained_len']}"
    )
    v = verify_all
    lines.append(
        f"- rebuild verification (all corpus texts, in-set-only subset): {v['exact_match_passed']}/{v['exact_match_total']} exact, "
        f"{len(v['exact_match_failures'])} failures; OOS-subset real fertility before={v['oos_texts_fertility_before']:.4f} "
        f"after={v['oos_texts_fertility_after_real']:.4f} tok/char (n={v['oos_texts_n']})"
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
    hashes = {f: rk.sha256_file(out_dir / f) for f in files}
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(f"{h}  {f}" for f, h in hashes.items()) + "\n")

    print(f"=== v2 {args.tokenizer_label} K={K} -> actual={actual_K} ===")
    print(f"  layer1_natural={len(layer1_ids)} layer2_dropped={layer2_dropped_count} layer3_admitted={len(layer3_ids)}")
    print(f"  merges: {n_merges_before} -> {n_merges_after} (dropped {n_merges_dropped})")
    print(f"  length check: K={actual_K}, tokenizers={file_tok_len}, transformers={hf_tok_len}")
    print(f"  verify (all corpus texts, in-set only): {v['exact_match_passed']}/{v['exact_match_total']} passed")
    for f, h in hashes.items():
        print(f"  {h}  {f}")
    print(f"  wrote {out_dir}")
    return {"out_dir": str(out_dir), "actual_K": actual_K, "hashes": hashes, "coverage": coverage}


def main():
    args = parse_args()
    target_sizes = args.target_size if args.target_size else TARGET_SIZES

    tok, tok_path = vc.load_tokenizer(args.tokenizer)
    tok_dir = Path(tok_path).parent
    raw = json.loads(Path(tok_path).read_text())
    tok_cfg_path = tok_dir / "tokenizer_config.json"
    raw_cfg = json.loads(tok_cfg_path.read_text()) if tok_cfg_path.exists() else {}
    mlc_cfg_path = tok_dir / "mlc-chat-config.json"
    mlc_cfg = json.loads(mlc_cfg_path.read_text()) if mlc_cfg_path.exists() else {}

    layer1_ids, layer_marginal, meta = build_layer1(tok, tok_path, raw)
    dropped, candidates = build_layer2_and_candidates(
        tok, layer1_ids, meta["byte_ids"], meta["added_ids"], meta["text_by_id"]
    )

    print(f"tokenizer: {tok_path} (label={args.tokenizer_label})")
    print(f"vocab size (with added): {tok.get_vocab_size(with_added_tokens=True)}")
    print("layer accounting (marginal = new ids beyond prior layers):")
    for m in layer_marginal:
        print(f"  {m['layer']}: layer_size={m['layer_size']} new_ids={m['new_ids']}")
    print(f"Layer 1 natural (no-padding) size: {len(layer1_ids)}")
    print(f"Layer 2 dropped wholesale: {len(dropped)}")
    print(f"Layer 3 candidate pool: {len(candidates)}")

    items = json.load(open(ARXIV_PASTES_PATH))
    items = [it for it in items if it.get("ok", True)]

    results = []
    layer_summary = {
        "tokenizer_label": args.tokenizer_label,
        "tokenizer_path": tok_path,
        "vocab_size_with_added": tok.get_vocab_size(with_added_tokens=True),
        "layer_accounting_marginal": layer_marginal,
        "layer1_natural_size": len(layer1_ids),
        "layer2_dropped_count": len(dropped),
        "layer3_candidate_pool_size": len(candidates),
        "by_target_size": {},
    }
    for K in target_sizes:
        layer3_ids, n_admitted, overflow, natural = layer3_fill_for_K(layer1_ids, candidates, K)
        if overflow:
            print(
                f"ANOMALY: Layer 1+closure natural size {natural} exceeds target K={K}; "
                f"this K cannot be emitted without truncating a mandatory layer. "
                f"Smallest viable K for this tokenizer is >= {natural} (next practical size: "
                f"{((natural + 1023) // 1024) * 1024})."
            )
            layer_summary["by_target_size"][str(K)] = {
                "target_size": K,
                "overflow": True,
                "layer1_natural_size": natural,
                "minimum_viable_K": natural,
            }
            continue
        r = emit_for_K(args, raw, raw_cfg, mlc_cfg, tok, tok_path, layer1_ids, layer3_ids, K, layer_marginal, len(dropped), items)
        r["layer3_admitted"] = n_admitted
        r["layer1_natural_size"] = len(layer1_ids)
        r["layer2_dropped_count"] = len(dropped)
        results.append(r)
        layer_summary["by_target_size"][str(K)] = {
            "target_size": K,
            "overflow": False,
            "actual_size": r["actual_K"],
            "layer1_natural_size": len(layer1_ids),
            "layer2_dropped_count": len(dropped),
            "layer3_admitted_count": n_admitted,
            "keep_idx_sha256": r["hashes"]["keep-idx.json"],
        }

    out_summary_path = ROOT / f"results-keepset-v2-layers-{args.tokenizer_label}.json"
    out_summary_path.write_text(json.dumps(layer_summary, indent=2) + "\n")
    print(f"\nwrote {out_summary_path}")


if __name__ == "__main__":
    main()
