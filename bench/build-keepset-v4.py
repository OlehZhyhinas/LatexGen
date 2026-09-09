#!/usr/bin/env python3
"""Keep-set v4: strategy A at K in {24576, 32768, 49152}, plus a minimal,
named force-add set that restores the specific failure classes strategy A
is measured to drop. See bench/README-keepset-v4.md for the full design
rationale and the measured tables this script's output feeds.

Strategy A, exactly as `keepset-v3` (PR pending against this same base,
landed at `origin/keepset-v3` -- see `bench/README-keepset-v3.md`'s "Strategy
definitions (exact, not paraphrased)") defines and verifies it: **v1's own
method, called directly, not reimplemented** --
`bench/vocab-coverage.py`'s `build_keep_set_for_tokenizer` (its default 5
corpus files -- bench-data.json, bench-data-extended.json, pdf-pastes.json,
arxiv-pastes.json, synth-spans.jsonl -- unioned with the byte alphabet,
added/special tokens, the webnn-workbench harness-prompt tokens, the KaTeX
macro seed, and the maths-English word seed) then `final_keep_set_for_K`
(pad to K by ascending original token id -- the "plain merge rank" fill
mechanism the brief's text describes, as opposed to v2's targeted-layer-then
restricted-pool fill). This task's brief describes strategy A in its "Where
this sits" section as "plain merge rank" and this task's build script
originally (first pass, superseded, see git history on this branch) took
that too literally and built a corpus-free byte+special+ascending-fill
baseline instead. That version's fertility numbers did not reconcile with
`origin/keepset-v3`'s landed sweep (its inflation was many points higher
than v3's "A" at every K), which sent us back to `keepset-v3`'s own README
for the precise, already-verified definition -- see
bench/README-keepset-v4.md's "What 'strategy A' means, corrected" section
for the full reconciliation trail. This version uses v3's exact definition
so the two tasks' numbers are directly comparable, as the brief asks.

This means pure strategy A is expected to reproduce v1's own originally-named
failure -- it drops the 7 LatexGen-prompt token ids on Qwen3.5 (v3's Job 4
found and reported the identical defect) -- which is exactly the gap force-add
group 3 below exists to close.

Force-add groups (computed once per tokenizer, independent of K; see the
brief's "The force-add set" section):
  1. Single-codepoint non-ASCII tokens (bare + space-prefixed variants),
     classified by Unicode General Category / block membership (reusing
     bench/keepset-v2-unicode.py's Greek/maths-operator/arrow blocks and
     adding two new, documented rules: "LATIN"-named Lu/Ll codepoints
     outside ASCII, for accented Latin letters; category "No" codepoints
     whose Unicode name contains SUPERSCRIPT/SUBSCRIPT, for the digit-only
     sub/superscript class the maths/Greek classifier does not cover).
  2. LaTeX macro completion: every token id that tokenizing each of KaTeX's
     1,241 macro/environment/delimiter strings (bench/keepset-seed-latex.json)
     produces, bare and space-prefixed (reusing bench/vocab-coverage.py's
     seed_ids_from_strings, unmodified).
  3. LatexGen's own prompt: the 7 named Qwen3.5 ids from the brief, plus
     every token of the rendered chat template + conversion system prompt +
     PDF hint + both few-shot pairs for two representative user texts, both
     enable_thinking shapes, both plain/PDF prompt treatments (reusing
     bench/build-keepset-v2.py's latexgen_prompt_messages, unmodified --
     deliberately NOT reusing its harness_prompt_ids, since the brief scopes
     group 3 to LatexGen's own prompt only, not the webnn-workbench harness's
     separate benchmark prompts).
  4. Byte alphabet and added/special tokens: already strategy A's natural
     layer; nothing to add.

Merge-ancestor closure is computed over the union and VERIFIED, not
assumed (the brief is explicit that this must not be assumed): closure over
strategy A's own final (padded) set is also checked, and any extra ancestor
ids closure pulls in for either set are folded back in and the padding
recomputed so the emitted directory's actual size is still exactly K.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import unicodedata as ud
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEEPSETS_V4_ROOT = ROOT / "keepsets-v4"
ARXIV_PASTES_PATH = ROOT / "arxiv-pastes.json"
BENCH_EXTENDED_PATH = ROOT / "bench-data-extended.json"
LATEX_SEED_PATH = ROOT / "keepset-seed-latex.json"
LATEXGEN_PROMPTS_PATH = ROOT / "keepset-seed-latexgen-prompts.json"

TARGET_SIZES = [24576, 32768, 49152]

TOKENIZER_PATHS = {
    "qwen35": "/Users/oleh/.cache/huggingface/hub/models--mlc-ai--Qwen3.5-0.8B-q4f16_1-MLC/snapshots/0ec138972555613c1d7812a821778ad0398c8790/tokenizer.json",
    "minicpm5-2b": "/Users/oleh/.cache/huggingface/hub/models--ozhyhinas--MiniCPM5-2B-q4f16_1-MLC/snapshots/2318f37d9c39277ff01dc64086491028c95d4db4/tokenizer.json",
}

# The 7 ids named in the brief, in the Qwen3.5 tokenizer's id space
# specifically (see bench/keepset-seed-latexgen-prompts.json's provenance
# note: these are exactly the tokens keep-set v1 was missing).
NAMED_QWEN35_PROMPT_IDS = {80757, 80636, 77019, 65088, 80313, 72452, 94498}

RARE_WORD_PROBES = [
    "Nikolskii", "Randers", "Lissajous", "Kullback", "Hausdorff", "Chebyshev", "Sobolev",
]
ACCENTED_AUTHOR_NAME_PROBES = [
    "L\u00f3pez-Monsalvo", "A\u00e7\u0131kg\u00f6z", "Erd\u0151s", "Schr\u00f6dinger",
    "M\u00fcller", "Poincar\u00e9", "Rios-Ram\u00edrez", "\u00c7etin", "Bia\u0142ynicki-Birula",
]
MATHS_SYMBOL_PROBES = ["\u03c6", "\u2190", "\u2194", "\u2193", "\u00d7"]


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vc = _load_module("vocab_coverage_v4dep", "vocab-coverage.py")
rk = _load_module("rebuild_keepset_tokenizer_v4dep", "rebuild-keepset-tokenizer.py")
ku = _load_module("keepset_v2_unicode_v4dep", "keepset-v2-unicode.py")
bkv2 = _load_module("build_keepset_v2_v4dep", "build-keepset-v2.py")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-label", required=True, choices=sorted(TOKENIZER_PATHS.keys()))
    ap.add_argument("--target-size", type=int, action="append", default=None)
    return ap.parse_args()


# =============================================================================
# Held-out text (v3/v4's one methodology: real re-encoding, never simulated)
# =============================================================================

def load_held_out_texts():
    texts = []
    n_arxiv = n_bench_ext = 0
    arxiv_items = json.load(open(ARXIV_PASTES_PATH))
    for it in arxiv_items:
        if not it.get("ok", True):
            continue
        texts.append(it["input"])
        texts.append(it["reference"])
        n_arxiv += 1
    bench_ext_items = json.load(open(BENCH_EXTENDED_PATH))
    for it in bench_ext_items:
        if not it.get("ok", True):
            continue
        texts.append(it.get("input", ""))
        texts.append(it.get("reference", ""))
        n_bench_ext += 1
    return texts, n_arxiv, n_bench_ext


def fertility_over_texts(tok, texts):
    total_chars = 0
    total_tokens = 0
    for t in texts:
        if not t:
            continue
        total_chars += len(t)
        total_tokens += len(tok.encode(t, add_special_tokens=False).ids)
    fert = (total_tokens / total_chars) if total_chars else 0.0
    return fert, total_tokens, total_chars


# =============================================================================
# Force-add group 1: single-codepoint non-ASCII tokens, Unicode-classified
# =============================================================================

def classify_force_add_char(ch):
    """Classification for group 1. Documented, Unicode-property-derived
    (General Category + the same block constants bench/keepset-v2-unicode.py
    already uses for its maths/Greek classifier), with two additions of the
    same "block-membership correction" style that module already uses:
    - "LATIN"-named Lu/Ll codepoints outside ASCII = accented Latin letters
      (covers Latin-1 Supplement, Latin Extended-A/B/Additional letters
      alike: A-with-circumflex, c-with-caron, o-with-horn, O-with-diaeresis,
      i-with-dot-below, s-with-caron, i-with-diaeresis, o-with-stroke,
      I-with-dot-above, o-with-horn, O-with-acute, ...).
    - category "No" (Number, other) codepoints whose Unicode name contains
      SUPERSCRIPT or SUBSCRIPT = the digit-only sub/superscript class (e.g.
      U+00B3 SUPERSCRIPT THREE, U+2070 SUPERSCRIPT ZERO). This deliberately
      excludes the letter/sign sub/superscripts in the same block
      (SUPERSCRIPT LATIN SMALL LETTER N is category Lm, SUPERSCRIPT PLUS
      SIGN is category Sm) which keepset-v2-unicode.py's own
      SUPERSCRIPT_SUBSCRIPT_BLOCK sweeps in wholesale for its broader maths
      definition -- here the class is scoped to just the digits the brief
      names ("superscript/subscript digits").
    Everything else (Greek letters, maths operators/relations/arrows,
    primes, math alphanumerics, combining accents, the degree sign) reuses
    bench/keepset-v2-unicode.py's is_maths_or_greek_char verbatim, then is
    sub-classified for the accounting table only (which block it came from),
    not re-derived.
    """
    cp = ord(ch)
    if cp < 128:
        return None
    cat = ud.category(ch)
    name = ud.name(ch, "")
    if name.startswith("LATIN") and cat in ("Lu", "Ll"):
        return "accented_latin_letter"
    if cat == "No" and ("SUPERSCRIPT" in name or "SUBSCRIPT" in name):
        return "superscript_subscript_digit"
    if ku.is_maths_or_greek_char(ch):
        lo, hi = ku.GREEK_BLOCK
        if lo <= cp <= hi:
            return "greek_letter"
        if cp == 0x00B0:
            return "typographic"
        return "math_operator_relation"
    return None


def compute_group1(tok, byte_ids, added_ids):
    text_by_id = bkv2.decode_all_regular_tokens(tok, byte_ids, added_ids)
    classified = {}
    for tid, txt in text_by_id.items():
        has_space = txt.startswith(" ")
        content = txt[1:] if has_space else txt
        if len(content) != 1:
            continue
        cls = classify_force_add_char(content)
        if cls:
            classified[tid] = {"char": content, "class": cls, "leading_space": has_space}
    return classified


# =============================================================================
# Force-add group 2: LaTeX macro completion
# =============================================================================

def compute_group2(tok):
    seed = json.load(open(LATEX_SEED_PATH))
    strings = seed["all"]
    ids_bare = vc.seed_ids_from_strings(tok, strings, try_leading_space=False)
    ids_all = vc.seed_ids_from_strings(tok, strings, try_leading_space=True)
    return ids_all, {
        "n_macro_strings": len(strings),
        "n_ids_bare_only": len(ids_bare),
        "n_ids_bare_or_leadspace": len(ids_all),
    }


# =============================================================================
# Force-add group 3: LatexGen's own prompt (system + PDF hint + few-shot),
# rendered through the chat template. NOT the webnn-workbench harness's
# separate benchmark prompts (that stays out of scope per the brief).
# =============================================================================

SAMPLE_TEXTS = [
    "the integral from zero to infinity of e to the minus x squared dx",
    "the quadratic formula for a x squared plus b x plus c equals zero",
]


def compute_group3(tok, conv, tokenizer_label):
    ids = set()
    n_renders = 0
    n_chars = 0
    if conv is not None:
        lg_seed = json.load(open(LATEXGEN_PROMPTS_PATH))
        for sample_text in SAMPLE_TEXTS:
            for _, messages in bkv2.latexgen_prompt_messages(lg_seed, sample_text):
                for think_variant in (True, False):
                    text = vc.render_chatml(conv, messages, think_variant)
                    new_ids = vc.tokenize(tok, text)
                    ids.update(new_ids)
                    n_chars += len(text)
                    n_renders += 1
    named_ids = NAMED_QWEN35_PROMPT_IDS if tokenizer_label == "qwen35" else set()
    named_missing_from_render = named_ids - ids
    ids |= named_ids
    return ids, {
        "n_renders": n_renders,
        "n_chars": n_chars,
        "n_named_ids": len(named_ids),
        "named_ids_missing_from_render": sorted(named_missing_from_render),
        "conv_template_present": conv is not None,
    }


# =============================================================================
# Merge-ancestor closure verification (do NOT assume; measure)
# =============================================================================

def verify_closure(raw, merge_map, ids):
    closure = bkv2.compute_merge_ancestor_closure(raw, merge_map, ids)
    extra = closure - ids
    return closure, extra


# =============================================================================
# Emission (reuses rebuild-keepset-tokenizer.py's tokenizer-surgery functions,
# same pattern as build-keepset-v2.py's emit_for_K)
# =============================================================================

def emit_v4_dir(tokenizer_label, raw, raw_cfg, mlc_cfg, tok, tok_path, final_keep, K,
                 accounting, held_out_texts, fertility_A):
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

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

    out_dir = KEEPSETS_V4_ROOT / f"{tokenizer_label}-{actual_K}"
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

    fert_v4, tok_v4, chars_v4 = fertility_over_texts(new_tok, held_out_texts)

    verify_all = rk.verify_and_measure_fertility(tok, new_tok, new_id_map, final_keep, held_out_texts)

    coverage = {
        "tokenizer_label": tokenizer_label,
        "tokenizer_path": tok_path,
        "keepset_version": "v4",
        "target_size": K,
        "actual_size": actual_K,
        "length_check": length_check,
        "merges": {
            "before": n_merges_before,
            "after": n_merges_after,
            "dropped": n_merges_dropped,
            "dropped_fraction": (n_merges_dropped / n_merges_before) if n_merges_before else 0.0,
        },
        "tokenizer_config_phantom_added_tokens_dropped": n_phantom_dropped,
        "tokenizer_config_extra_special_tokens_dropped": n_extra_special_dropped,
        "stop_token_id_mismatches_found": stop_id_mismatches,
        "held_out_fertility": {
            "n_texts": len([t for t in held_out_texts if t]),
            "total_chars": chars_v4,
            "total_tokens_v4": tok_v4,
            "fertility_v4": fert_v4,
            "fertility_strategy_A": fertility_A,
            "fertility_inflation_v4_vs_A": ((fert_v4 / fertility_A) - 1.0) if fertility_A else 0.0,
        },
        "verify_all_held_out_texts": verify_all,
        "force_add_accounting": accounting,
    }
    (out_dir / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")

    script_sha = rk.git_head_sha()
    lines = []
    lines.append(f"# Provenance: keep-set v4 -- {tokenizer_label}-{actual_K}")
    lines.append("")
    lines.append(f"Generated by `bench/build-keepset-v4.py` at script commit `{script_sha}`.")
    lines.append("")
    lines.append("## Exact command line")
    lines.append("")
    lines.append("```")
    lines.append(f"python3 bench/build-keepset-v4.py --tokenizer-label {tokenizer_label} --target-size {K}")
    lines.append("```")
    lines.append("")
    lines.append("## Selection method")
    lines.append("")
    lines.append(
        "Strategy A (plain BPE merge rank: byte alphabet + added/special tokens + ascending-"
        "original-token-id fill to K, no corpus/seed dependency) plus a minimal, named force-add "
        "layer: single-codepoint accented-Latin/Greek/maths-operator/superscript-subscript-digit "
        "tokens (bare + space-prefixed variants that already exist in the vocab), LaTeX macro "
        "completion (bench/keepset-seed-latex.json, bare + leading-space), and LatexGen's own "
        "rendered prompt (system prompt, PDF hint, both few-shot pairs, chat template, both "
        "enable_thinking shapes, both plain/PDF treatments) plus the 7 named ids from the brief. "
        "Merge-ancestor closure over the union was measured, not assumed; see "
        "`force_add_accounting.closure` in `coverage.json` for the exact count and whether it was "
        "already free (already covered by strategy A's own ascending-id padding at this K)."
    )
    lines.append("")
    lines.append("## Force-add accounting (this tokenizer, independent of K)")
    lines.append("")
    lines.append("```")
    lines.append(json.dumps(accounting, indent=2, default=str))
    lines.append("```")
    lines.append("")
    lines.append("## Measured rates (this K)")
    lines.append("")
    lines.append(f"- target K: {K}, actual size achieved: {actual_K}")
    m = coverage["merges"]
    lines.append(f"- BPE merges: {m['before']} -> {m['after']} (dropped {m['dropped']}, {m['dropped_fraction']:.4f})")
    lines.append(f"- tokenizer_config.json phantom added-tokens dropped: {n_phantom_dropped}")
    lines.append(f"- tokenizer_config.json stale extra_special_tokens dropped: {n_extra_special_dropped}")
    lines.append(
        f"- length check: expected K={actual_K}, tokenizers.Tokenizer.from_file={file_tok_len}, "
        f"transformers.AutoTokenizer.from_pretrained={hf_tok_len}"
    )
    hf = coverage["held_out_fertility"]
    lines.append(
        f"- held-out fertility (real re-encoding, n={hf['n_texts']} texts, {hf['total_chars']} chars): "
        f"strategy A={hf['fertility_strategy_A']:.4f}, v4={hf['fertility_v4']:.4f} tok/char "
        f"({hf['fertility_inflation_v4_vs_A']:+.2%} vs strategy A at this K)"
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

    print(f"=== v4 {tokenizer_label} K={K} -> actual={actual_K} ===")
    print(f"  merges: {n_merges_before} -> {n_merges_after} (dropped {n_merges_dropped})")
    print(f"  length check: K={actual_K}, tokenizers={file_tok_len}, transformers={hf_tok_len}")
    print(f"  held-out fertility: A={hf['fertility_strategy_A']:.4f} v4={hf['fertility_v4']:.4f} ({hf['fertility_inflation_v4_vs_A']:+.2%})")
    for f, h in hashes.items():
        print(f"  {h}  {f}")
    print(f"  wrote {out_dir}")
    return {"out_dir": str(out_dir), "actual_K": actual_K, "hashes": hashes, "coverage": coverage}


def main():
    args = parse_args()
    target_sizes = args.target_size if args.target_size else TARGET_SIZES
    tokenizer_label = args.tokenizer_label
    tok_path_arg = TOKENIZER_PATHS[tokenizer_label]

    held_out_texts, n_arxiv, n_bench_ext = load_held_out_texts()
    print(f"held-out text: {n_arxiv} arxiv-pastes items + {n_bench_ext} bench-data-extended items "
          f"-> {len(held_out_texts)} texts")

    # Strategy A, exactly as keepset-v3 defines and verifies it: v1's own
    # build_keep_set_for_tokenizer (default 5 corpus files + byte/added/
    # special/harness-prompt/latex-seed/mathsenglish-seed layers), called
    # directly -- not reimplemented, not stripped down to a corpus-free
    # approximation. See module docstring for the reconciliation trail.
    built = vc.build_keep_set_for_tokenizer(tok_path_arg, sorted(vc.CORPUS_FILES.keys()))
    tok, tok_path = built["tok"], built["tok_path"]
    raw = json.loads(Path(tok_path).read_text())
    tok_dir = Path(tok_path).parent
    raw_cfg = json.loads((tok_dir / "tokenizer_config.json").read_text()) if (tok_dir / "tokenizer_config.json").exists() else {}
    mlc_cfg = json.loads((tok_dir / "mlc-chat-config.json").read_text()) if (tok_dir / "mlc-chat-config.json").exists() else {}
    conv = built["conv"]

    byte_ids = built["meta"]["byte_ids"]
    added_ids = built["meta"]["added_ids"]
    regular_ids = built["regular_ids"]
    natural_A = built["base_keep_ids"]
    print(f"tokenizer: {tok_path} (label={tokenizer_label})")
    print(f"vocab size (with added): {tok.get_vocab_size(with_added_tokens=True)}")
    print(f"strategy A natural (v1 method: byte+added+special+harness+latex-seed+mathsenglish-seed+corpus) size: {len(natural_A)}")

    # ---- force-add groups ----
    group1 = compute_group1(tok, byte_ids, added_ids)
    group1_ids = set(group1.keys())
    group1_by_class = collections.Counter(v["class"] for v in group1.values())

    group2_ids, group2_meta = compute_group2(tok)
    group3_ids, group3_meta = compute_group3(tok, conv, tokenizer_label)

    force_ids_pre = group1_ids | group2_ids | group3_ids
    overlaps = {
        "group1_and_group2": len(group1_ids & group2_ids),
        "group1_and_group3": len(group1_ids & group3_ids),
        "group2_and_group3": len(group2_ids & group3_ids),
        "all_three": len(group1_ids & group2_ids & group3_ids),
    }
    v4_natural_pre = natural_A | force_ids_pre

    merge_map = bkv2.compute_merge_map(raw)
    closure_pre, closure_extra_pre = verify_closure(raw, merge_map, v4_natural_pre)
    v4_natural = v4_natural_pre | closure_extra_pre

    print(f"group1 (single-codepoint): {len(group1_ids)} ids {dict(group1_by_class)}")
    print(f"group2 (latex macro completion): {len(group2_ids)} ids ({group2_meta})")
    print(f"group3 (latexgen prompt + chat template): {len(group3_ids)} ids ({group3_meta})")
    print(f"force-add union (pre-closure): {len(force_ids_pre)}, overlaps={overlaps}")
    print(f"merge-ancestor closure extra (beyond force-add + strategy-A-natural): {len(closure_extra_pre)}")

    accounting = {
        "tokenizer_label": tokenizer_label,
        "strategy_a_natural_size": len(natural_A),
        "group1_single_codepoint": {
            "total": len(group1_ids),
            "by_class": dict(group1_by_class),
            "sample": [
                {"id": tid, "char": v["char"], "class": v["class"], "leading_space": v["leading_space"]}
                for tid, v in list(group1.items())[:20]
            ],
        },
        "group2_latex_macro_completion": {"total": len(group2_ids), **group2_meta},
        "group3_latexgen_prompt_and_chat_template": {"total": len(group3_ids), **group3_meta},
        "overlaps": overlaps,
        "force_add_union_pre_closure": len(force_ids_pre),
        "closure": {
            "extra_ancestor_ids": len(closure_extra_pre),
            "sample_extra_ids": sorted(closure_extra_pre)[:20],
        },
        "v4_natural_total_post_closure": len(v4_natural),
        "v4_natural_new_vs_strategy_a": len(v4_natural) - len(natural_A),
        "by_target_size": {},
    }

    # ---- Issue #27 LaTeX-tail claim, verified against the REAL strategy A
    # at K=32,768 (not the corpus-free approximation the brief's prose could
    # be misread as) ----
    if 32768 in target_sizes:
        padA32, _ = vc.compute_padding_for_target(natural_A, regular_ids, len(natural_A), 32768)
        finalA32 = natural_A | padA32
        issue27_pieces = ["\\(", "\\[", "mathrm", "partial", "rangle", "Gamma", "Psi", "{n", "}="]
        issue27_report = {}
        for piece in issue27_pieces:
            ids = tok.encode(piece, add_special_tokens=False).ids
            missing = [i for i in ids if i not in finalA32]
            issue27_report[piece] = {"ids": ids, "all_present_in_strategy_A_32768": len(missing) == 0, "missing": missing}
        accounting["issue_27_tail_claim_verified_against_strategy_A_32768"] = issue27_report
        n_fully_present = sum(1 for v in issue27_report.values() if v["all_present_in_strategy_A_32768"])
        print(f"issue #27 tail claim check (strategy A @ 32768): {n_fully_present}/{len(issue27_pieces)} pieces fully present")

    # ---- Leakage sanity check (keepset-v3 found strategy A's default
    # corpus includes 2 of this task's own held-out files -- see
    # bench/README-keepset-v3.md's "Is A's win real, or leakage?"). Reuse
    # the same disjoint-corpus recipe (drop arxiv + bench-extended from A's
    # training set) at K=32,768 only, as a spot check, not a full re-sweep. ----
    if 32768 in target_sizes:
        disjoint_labels = sorted(set(vc.CORPUS_FILES.keys()) - {"arxiv", "bench-extended"})
        built_disjoint = vc.build_keep_set_for_tokenizer(tok_path_arg, disjoint_labels)
        finalA32_disjoint, _, _ = vc.final_keep_set_for_K(built_disjoint, 32768)
        from tokenizers import Tokenizer as _TokCls
        new_id_map_disjoint = {old: i for i, old in enumerate(sorted(finalA32_disjoint))}
        new_raw_disjoint, *_ = rk.rebuild_tokenizer_json(raw, finalA32_disjoint, new_id_map_disjoint)
        rebuilt_disjoint = _TokCls.from_str(json.dumps(new_raw_disjoint))
        fert_disjoint, _, _ = fertility_over_texts(rebuilt_disjoint, held_out_texts)
        fert_unpruned, _, _ = fertility_over_texts(tok, held_out_texts)
        # v4-disjoint: force-add groups 1/2/3 are corpus-independent (Unicode
        # classification + static seed files, no corpus item ever touches
        # them), so the *only* leakage source in v4 is strategy A's own
        # corpus-derived base. Union the disjoint A base with the same
        # force-add groups + closure, and re-measure -- this gives a
        # leakage-free v4 number for direct comparison to v3's "A-fixed-disjoint".
        v4_disjoint_natural_pre = built_disjoint["base_keep_ids"] | force_ids_pre
        _, closure_extra_disjoint = verify_closure(raw, merge_map, v4_disjoint_natural_pre)
        v4_disjoint_natural = v4_disjoint_natural_pre | closure_extra_disjoint
        pad4_disjoint, _ = vc.compute_padding_for_target(
            v4_disjoint_natural, built_disjoint["regular_ids"], len(v4_disjoint_natural), 32768
        )
        final4_disjoint = v4_disjoint_natural | pad4_disjoint
        new_id_map_4disjoint = {old: i for i, old in enumerate(sorted(final4_disjoint))}
        new_raw_4disjoint, *_ = rk.rebuild_tokenizer_json(raw, final4_disjoint, new_id_map_4disjoint)
        rebuilt_4disjoint = _TokCls.from_str(json.dumps(new_raw_4disjoint))
        fert_4disjoint, _, _ = fertility_over_texts(rebuilt_4disjoint, held_out_texts)

        accounting["leakage_sanity_check_strategy_A_32768"] = {
            "note": "A's default corpus includes 2 of this task's held-out files (arxiv-pastes.json, "
                    "bench-data-extended.json), per keepset-v3's finding. This drops those 2 from A's "
                    "training corpus only (disjoint_labels below) and re-measures fertility against the "
                    "SAME unmodified held-out text, as a spot check at K=32,768. v4-disjoint additionally "
                    "unions in the (corpus-independent) force-add groups + closure on top of the disjoint "
                    "A base, so it isolates the force-add layer's true, leakage-free cost.",
            "disjoint_corpus_labels": disjoint_labels,
            "disjoint_natural_size": built_disjoint["k_natural"],
            "fertility_unpruned": fert_unpruned,
            "fertility_A_disjoint_32768": fert_disjoint,
            "inflation_A_disjoint_32768": (fert_disjoint / fert_unpruned - 1.0) if fert_unpruned else 0.0,
            "fertility_v4_disjoint_32768": fert_4disjoint,
            "inflation_v4_disjoint_32768": (fert_4disjoint / fert_unpruned - 1.0) if fert_unpruned else 0.0,
            "v4_vs_A_cost_disjoint_32768": (fert_4disjoint / fert_disjoint - 1.0) if fert_disjoint else 0.0,
        }
        print(f"leakage sanity check: disjoint-corpus A@32768 fertility={fert_disjoint:.4f} "
              f"(inflation {(fert_disjoint/fert_unpruned-1.0):+.2%}) vs unpruned={fert_unpruned:.4f}")
        print(f"leakage sanity check: disjoint-corpus v4@32768 fertility={fert_4disjoint:.4f} "
              f"(inflation {(fert_4disjoint/fert_unpruned-1.0):+.2%}, "
              f"v4-vs-A cost {(fert_4disjoint/fert_disjoint-1.0):+.2%})")

    for K in target_sizes:
        # Strategy A at this K
        padA, clippedA = vc.compute_padding_for_target(natural_A, regular_ids, len(natural_A), K)
        finalA = natural_A | padA
        closureA, closure_extra_A = verify_closure(raw, merge_map, finalA)

        # v4 at this K (fold in any A-side closure gap defensively, then check
        # v4's own closure and fold that in too, one fixed-point pass)
        v4_natural_k = v4_natural | closure_extra_A
        pad4, clipped4 = vc.compute_padding_for_target(v4_natural_k, regular_ids, len(v4_natural_k), K)
        final4 = v4_natural_k | pad4
        closure4, closure_extra_4 = verify_closure(raw, merge_map, final4)
        if closure_extra_4:
            v4_natural_k2 = v4_natural_k | closure_extra_4
            pad4, clipped4 = vc.compute_padding_for_target(v4_natural_k2, regular_ids, len(v4_natural_k2), K)
            final4 = v4_natural_k2 | pad4
            closure4b, closure_extra_4b = verify_closure(raw, merge_map, final4)
        else:
            closure_extra_4b = set()

        new_force_missing_from_A = force_ids_pre - finalA  # cost over strategy A at this K
        new_force_missing_from_A_by_class = collections.Counter(
            group1[t]["class"] for t in (new_force_missing_from_A & group1_ids)
        )
        # per-group missing breakdown (a token can count in more than one
        # group if the groups overlap; overlaps are reported separately above)
        group_missing_breakdown = {
            "group1_single_codepoint": len(group1_ids - finalA),
            "group2_latex_macro_completion": len(group2_ids - finalA),
            "group3_latexgen_prompt_and_chat_template": len(group3_ids - finalA),
        }

        # Slot accounting: whole-word tokens displaced from strategy A's own
        # ascending-id fill at this K by v4's force-add layer (the fill is a
        # fixed-size window into the same ascending-rank candidate pool, so
        # every extra id v4 keeps for a force-add reason costs exactly one
        # fewer merge-rank fill slot at the same K).
        displaced_ids = padA - pad4
        displaced_classified = []
        for did in sorted(displaced_ids):
            txt = tok.id_to_token(did) or ""
            piece = bkv2.decode_all_regular_tokens(tok, byte_ids, added_ids).get(did, txt)
            stripped = piece[1:] if piece.startswith(" ") else piece
            is_whole_word = (
                piece.startswith(" ")
                and len(stripped) >= 2
                and stripped.isalpha()
                and stripped.isascii()
            )
            displaced_classified.append(is_whole_word)
        n_displaced_whole_words = sum(displaced_classified)

        # measure REAL strategy-A fertility via rebuilt (id-remapped) tokenizer
        from tokenizers import Tokenizer
        new_id_map_A = {old: i for i, old in enumerate(sorted(finalA))}
        new_raw_A, *_ = rk.rebuild_tokenizer_json(raw, finalA, new_id_map_A)
        rebuilt_tok_A = Tokenizer.from_str(json.dumps(new_raw_A))
        fertA_real, tokA_real, charsA_real = fertility_over_texts(rebuilt_tok_A, held_out_texts)

        print(f"\n--- K={K} ---")
        print(f"  strategy A: size={len(finalA)} closure_extra={len(closure_extra_A)} fertility={fertA_real:.4f}")
        print(f"  force-add new-vs-A at this K: {len(new_force_missing_from_A)} {dict(new_force_missing_from_A_by_class)}")
        print(f"  per-group missing from A at this K: {group_missing_breakdown}")
        print(f"  v4 pre-fold closure_extra={len(closure_extra_4)} (post-fold extra={len(closure_extra_4b)})")
        print(f"  displaced merge-rank fill slots: {len(displaced_ids)} (whole-word among them: {n_displaced_whole_words})")

        r = emit_v4_dir(tokenizer_label, raw, raw_cfg, mlc_cfg, tok, tok_path, final4, K,
                         accounting, held_out_texts, fertA_real)
        accounting["by_target_size"][str(K)] = {
            "target_size": K,
            "strategy_a_size": len(finalA),
            "strategy_a_closure_extra": len(closure_extra_A),
            "strategy_a_fertility": fertA_real,
            "v4_actual_size": r["actual_K"],
            "v4_closure_extra_pre_fold": len(closure_extra_4),
            "v4_closure_extra_post_fold": len(closure_extra_4b),
            "new_force_add_cost_vs_strategy_a": len(new_force_missing_from_A),
            "new_force_add_cost_vs_strategy_a_by_class": dict(new_force_missing_from_A_by_class),
            "group_missing_breakdown": group_missing_breakdown,
            "displaced_merge_rank_fill_slots": len(displaced_ids),
            "displaced_whole_word_tokens": n_displaced_whole_words,
            "v4_fertility": r["coverage"]["held_out_fertility"]["fertility_v4"],
            "keep_idx_sha256": r["hashes"]["keep-idx.json"],
        }

    out_summary_path = ROOT / f"results-keepset-v4-{tokenizer_label}.json"
    out_summary_path.write_text(json.dumps(accounting, indent=2, default=str) + "\n")
    print(f"\nwrote {out_summary_path}")


if __name__ == "__main__":
    main()
