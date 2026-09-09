#!/usr/bin/env python3
"""Keep-set v3: one fertility methodology, applied to every selection
strategy (A/B/C, D if the corpus sibling has landed), at every target K, on
the same held-out text -- Job 1 of latexgen-task-keepset-v3.md.

Strategies (see the brief for the full rationale):
  A: merge-rank selection, v1's method verbatim (bench/vocab-coverage.py's
     build_keep_set_for_tokenizer + final_keep_set_for_K -- corpus +
     seed-string layers, then ascending-original-token-id fill to K).
  B: the targeted structural fix layers only (byte alphabet, added/special
     tokens, chat-template + LatexGen-prompt + harness-prompt tokens, all
     maths/Greek symbol tokens, BPE merge-ancestor closure over those --
     i.e. v2's Layer 1 MINUS the blanket "ASCII <=3 chars" layer), then
     plain merge-rank (ascending original token id, v1's fill mechanism,
     unrestricted candidate pool) fill to K. Built from the same reused
     primitives as C (bench/keepset-v2-unicode.py's classifiers,
     bench/build-keepset-seed-latexgen-prompts.py's seed via
     bench/build-keepset-v2.py's latexgen_and_harness_prompt_ids), just
     unioned differently -- neither seed script nor classifier module is
     re-derived.
  C: v2's full structural selection verbatim (bench/build-keepset-v2.py's
     build_layer1 + build_layer2_and_candidates + layer3_fill_for_K,
     imported and called directly, not reimplemented).
  D: marginal-token-savings ranking (Job 2) -- only if
     bench/corpus/freq-<label>.json exists; see has_corpus_landed().

Fertility is measured by ACTUALLY rebuilding the tokenizer for the
resulting keep-set and re-encoding held-out text with it (never simulated
as a missing token's byte length).
"""
import argparse
import collections
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEEPSETS_V3_ROOT = ROOT / "keepsets-v3"
ARXIV_PASTES_PATH = ROOT / "arxiv-pastes.json"
BENCH_EXTENDED_PATH = ROOT / "bench-data-extended.json"
CORPUS_DIR = ROOT / "corpus"

SWEEP_KS = [8192, 16384, 24576, 32768, 49152, 65536, 98304]

TOKENIZER_PATHS = {
    "qwen35": "/Users/oleh/.cache/huggingface/hub/models--mlc-ai--Qwen3.5-0.8B-q4f16_1-MLC/snapshots/0ec138972555613c1d7812a821778ad0398c8790/tokenizer.json",
    "minicpm5-2b": "/Users/oleh/.cache/huggingface/hub/models--ozhyhinas--MiniCPM5-2B-q4f16_1-MLC/snapshots/2318f37d9c39277ff01dc64086491028c95d4db4/tokenizer.json",
}


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vc = _load_module("vocab_coverage_v3dep", "vocab-coverage.py")
rk = _load_module("rebuild_keepset_tokenizer_v3dep", "rebuild-keepset-tokenizer.py")
ku = _load_module("keepset_v2_unicode_v3dep", "keepset-v2-unicode.py")
kv2 = _load_module("build_keepset_v2_v3dep", "build-keepset-v2.py")


def has_corpus_landed():
    """Job 2 gate: does bench/corpus/freq-<label>.json exist (sibling
    keepset-corpus branch merged/pushed its frequency tables)? Checked
    locally (this worktree does not touch bench/corpus/, per the brief) and
    does not block anything else if absent."""
    if not CORPUS_DIR.exists():
        return False, []
    found = sorted(p.name for p in CORPUS_DIR.glob("freq-*.json"))
    return len(found) > 0, found


# =============================================================================
# Held-out text (same for every strategy, every K, every tokenizer)
# =============================================================================

def held_out_texts():
    texts = []
    n_arxiv = 0
    n_extended = 0
    arxiv_items = json.load(open(ARXIV_PASTES_PATH))
    for it in arxiv_items:
        if not it.get("ok", True):
            continue
        texts.append(it["input"])
        texts.append(it["reference"])
        n_arxiv += 1
    if BENCH_EXTENDED_PATH.exists():
        ext_items = json.load(open(BENCH_EXTENDED_PATH))
        for it in ext_items:
            if not it.get("ok", True):
                continue
            texts.append(it["input"])
            texts.append(it["reference"])
            n_extended += 1
    return texts, {"n_arxiv_pastes_items": n_arxiv, "n_bench_extended_items": n_extended}


def measure_fertility(old_tok, new_tok, texts):
    """Real re-encode-and-count fertility, before (original tokenizer) vs
    after (rebuilt, pruned tokenizer), on the exact same text list."""
    total_chars = 0
    total_old = 0
    total_new = 0
    n_texts = 0
    for t in texts:
        if not t:
            continue
        n_texts += 1
        total_chars += len(t)
        total_old += len(old_tok.encode(t, add_special_tokens=False).ids)
        total_new += len(new_tok.encode(t, add_special_tokens=False).ids)
    fert_before = (total_old / total_chars) if total_chars else 0.0
    fert_after = (total_new / total_chars) if total_chars else 0.0
    inflation = (fert_after / fert_before - 1.0) if fert_before else 0.0
    return {
        "n_texts": n_texts,
        "total_chars": total_chars,
        "total_tokens_before": total_old,
        "total_tokens_after": total_new,
        "fertility_before_tokens_per_char": fert_before,
        "fertility_after_tokens_per_char": fert_after,
        "fertility_inflation_relative": inflation,
    }


def rebuild_and_measure(raw, keep_ids, old_tok, texts):
    from tokenizers import Tokenizer

    keep_idx = sorted(keep_ids)
    new_id_map = {old: i for i, old in enumerate(keep_idx)}
    new_raw, n_merges_before, n_merges_after, n_merges_dropped = rk.rebuild_tokenizer_json(raw, keep_ids, new_id_map)
    new_tok = Tokenizer.from_str(json.dumps(new_raw))
    fert = measure_fertility(old_tok, new_tok, texts)
    fert["merges_before"] = n_merges_before
    fert["merges_after"] = n_merges_after
    fert["merges_dropped"] = n_merges_dropped
    return fert, new_raw, keep_idx, new_id_map


# =============================================================================
# Strategy A: v1's method verbatim
# =============================================================================

def strategy_A_base(tokenizer_path, corpus_labels=None):
    built = vc.build_keep_set_for_tokenizer(tokenizer_path, corpus_labels or sorted(vc.CORPUS_FILES.keys()))
    return built


# Corpus labels that do NOT overlap with this task's held-out text
# (arxiv-pastes.json, bench-data-extended.json). Used for the leakage-check
# in bench/README-keepset-v3.md's "Is strategy A's win real, or leakage?"
# section: v1's method (unlike B/C, which reference no corpus at all) folds
# in arxiv-pastes.json + bench-data-extended.json as two of its five
# default corpus files -- exactly the files the brief designates as this
# task's held-out text -- so the main A/B/C/D sweep's numbers for A are
# partly in-sample. This alternate corpus selection answers "does A still
# win on text genuinely absent from its own construction?".
DISJOINT_CORPUS_LABELS = ["bench", "pdf", "synth-latex"]


def strategy_A_for_K(built, K):
    final_keep, pad_ids, clipped = vc.final_keep_set_for_K(built, K)
    return final_keep, built["k_natural"], clipped


# =============================================================================
# Strategy B: targeted fix layers only (v2's Layer 1 minus the ASCII<=3
# blanket layer), merge-rank (ascending id, unrestricted pool) fill to K.
# Reuses kv2's (build-keepset-v2.py's) helper functions directly; only the
# union of which layers feed the merge-ancestor closure differs from v2.
# =============================================================================

def strategy_B_base(tok, tok_path, raw):
    byte_ids = vc.get_byte_token_ids(tok)
    added_decoder = tok.get_added_tokens_decoder()
    added_ids = set(added_decoder.keys())
    special_ids = {tid for tid, t in added_decoder.items() if getattr(t, "special", False)}

    conv = vc.load_conv_template(tok_path)
    prompt_ids, n_prompt_renders, n_prompt_chars = kv2.latexgen_and_harness_prompt_ids(tok, conv)

    text_by_id = kv2.decode_all_regular_tokens(tok, byte_ids, added_ids)
    maths_ids = {tid for tid, txt in text_by_id.items() if ku.is_maths_or_greek_text(txt)}

    layers_pre_closure = [
        ("byte_alphabet", byte_ids),
        ("added_and_special", added_ids | special_ids),
        ("chat_template_and_prompts", prompt_ids),
        ("maths_and_greek_symbols", maths_ids),
    ]
    marginal = []
    seen = set()
    for name, ids in layers_pre_closure:
        new = ids - seen
        marginal.append({"layer": name, "layer_size": len(ids), "new_ids": len(new)})
        seen |= ids
    pre_closure_ids = seen

    merge_map = kv2.compute_merge_map(raw)
    closure_ids = kv2.compute_merge_ancestor_closure(raw, merge_map, pre_closure_ids)
    closure_extra = closure_ids - pre_closure_ids
    marginal.append({"layer": "merge_ancestor_closure", "layer_size": len(closure_ids), "new_ids": len(closure_extra)})

    regular_ids = vc.get_regular_token_ids(tok, byte_ids, added_ids)  # ascending, excludes byte/added
    candidates = [tid for tid in regular_ids if tid not in closure_ids]

    return {
        "base_ids": closure_ids,
        "natural_size": len(closure_ids),
        "marginal": marginal,
        "candidates": candidates,  # ascending original id = merge-rank order, unrestricted pool
    }


def strategy_B_for_K(base, K):
    natural = base["natural_size"]
    if natural >= K:
        return set(base["base_ids"]), natural, True
    budget = K - natural
    fill = base["candidates"][:budget]
    return base["base_ids"] | set(fill), natural, False


# =============================================================================
# Strategy C: v2's full method verbatim (imported, not reimplemented)
# =============================================================================

def strategy_C_base(tok, tok_path, raw):
    layer1_ids, layer_marginal, meta = kv2.build_layer1(tok, tok_path, raw)
    dropped, candidates = kv2.build_layer2_and_candidates(
        tok, layer1_ids, meta["byte_ids"], meta["added_ids"], meta["text_by_id"]
    )
    return {
        "layer1_ids": layer1_ids,
        "natural_size": len(layer1_ids),
        "marginal": layer_marginal,
        "layer2_dropped_count": len(dropped),
        "candidates": candidates,
    }


def strategy_C_for_K(base, K):
    layer3_ids, n_admitted, overflow, natural = kv2.layer3_fill_for_K(base["layer1_ids"], base["candidates"], K)
    if overflow:
        return set(base["layer1_ids"]), natural, True
    return base["layer1_ids"] | layer3_ids, natural, False


# =============================================================================
# Strategy D: marginal-token-savings ranking (Job 2). NOT IMPLEMENTED here --
# see has_corpus_landed(). The brief is explicit that D is conditional on
# the keepset-corpus sibling's bench/corpus/freq-<label>.json having
# landed; as of this run it has not (checked live below, and again just
# before the PR was opened -- see bench/README-keepset-v3.md), so D is
# skipped rather than approximated. If it lands later, D would rank the
# strategy-B residual candidate pool by
# savings(t) = count(t) * (pieces_if_dropped(t) - 1) (count from the
# frequency table; pieces_if_dropped(t) from an EXACT re-tokenization of
# t's decoded string with t's own merge deleted from the tokenizer, not a
# depth/byte-length approximation) instead of B's plain merge-rank fill.
# =============================================================================

# =============================================================================
# Strategy A-fixed: strategy A (v1 verbatim) plus the one targeted-fix layer
# Job 4's validation shows A actually needs -- see bench/README-keepset-v3.md
# "Job 4 validation" section. Running the brief's own 5 checks against pure
# strategy-A emissions found A reproduces v1's ORIGINAL, specifically-named
# failure 1 (the 7 LatexGen-own-prompt token ids): v1's corpus (bench-data*,
# pdf-pastes, arxiv-pastes, synth-spans) never included LatexGen's actual
# client prompt text (public/pipeline.js / public/pdf-prompt.js) -- that
# corpus gap is exactly what build-keepset-seed-latexgen-prompts.py and v2's
# Layer 1 were built to close, and it is a real, K-independent hole in v1's
# method, not a fertility tradeoff. Per the brief ("a variant that regresses
# any of these is not a candidate, however good its fertility -- say so and
# drop it"), pure A is DROPPED as a candidate and replaced by this minimal
# patch: A's exact base plus the LatexGen-prompt token layer (reusing the
# same kv2.latexgen_and_harness_prompt_ids used by B/C), merge-ancestor-
# closed, then filled to K exactly like A. Cost: ~250-300 extra guaranteed
# slots out of tens of thousands -- negligible next to A's fertility
# advantage over B/C, and it is the layer, not a re-tuned number.
# =============================================================================

def strategy_A_fixed_base(tokenizer_path):
    tok, tok_path = vc.load_tokenizer(tokenizer_path)
    raw = json.loads(Path(tok_path).read_text())
    built = strategy_A_base(tokenizer_path)
    conv = vc.load_conv_template(tok_path)
    prompt_ids, n_prompt_renders, n_prompt_chars = kv2.latexgen_and_harness_prompt_ids(tok, conv)
    pre_closure = built["base_keep_ids"] | prompt_ids
    merge_map = kv2.compute_merge_map(raw)
    fixed_base_ids = kv2.compute_merge_ancestor_closure(raw, merge_map, pre_closure)
    added_by_fix = fixed_base_ids - built["base_keep_ids"]
    built2 = dict(built)
    built2["base_keep_ids"] = fixed_base_ids
    built2["k_natural"] = len(fixed_base_ids)
    built2["n_ids_added_by_prompt_fix"] = len(added_by_fix)
    built2["always_keep_ids"] = built["always_keep_ids"] | prompt_ids
    return built2


def strategy_A_fixed_for_K(built2, K):
    final_keep, pad_ids, clipped = vc.final_keep_set_for_K(built2, K)
    return final_keep, built2["k_natural"], clipped


STRATEGY_BASE_BUILDERS = {"A": strategy_A_base, "B": strategy_B_base, "C": strategy_C_base,
                           "A-fixed": strategy_A_fixed_base}
STRATEGY_FOR_K = {"A": strategy_A_for_K, "B": strategy_B_for_K, "C": strategy_C_for_K,
                   "A-fixed": strategy_A_fixed_for_K}


def run_sweep(label, strategies, ks):
    tok, tok_path = vc.load_tokenizer(TOKENIZER_PATHS[label])
    raw = json.loads(Path(tok_path).read_text())
    texts, held_out_meta = held_out_texts()

    corpus_landed, freq_files = has_corpus_landed()
    if "D" in strategies and not corpus_landed:
        print("D skipped: bench/corpus/freq-<label>.json not found (keepset-corpus sibling has not landed it yet).")
    strategies = [s for s in strategies if s != "D" or corpus_landed]

    bases = {}
    if "A" in strategies:
        bases["A"] = strategy_A_base(tok_path)
    if "B" in strategies:
        bases["B"] = strategy_B_base(tok, tok_path, raw)
    if "C" in strategies:
        bases["C"] = strategy_C_base(tok, tok_path, raw)
    if "A-fixed" in strategies:
        bases["A-fixed"] = strategy_A_fixed_base(tok_path)

    results = {
        "tokenizer_label": label,
        "tokenizer_path": tok_path,
        "vocab_size_with_added": tok.get_vocab_size(with_added_tokens=True),
        "held_out_meta": held_out_meta,
        "n_held_out_texts": len([t for t in texts if t]),
        "corpus_landed_for_D": corpus_landed,
        "corpus_freq_files": freq_files,
        "strategy_natural_sizes": {s: bases[s]["k_natural"] if s in ("A", "A-fixed") else bases[s]["natural_size"] for s in bases},
        "strategy_B_layer_marginal": bases["B"]["marginal"] if "B" in bases else None,
        "strategy_C_layer_marginal": bases["C"]["marginal"] if "C" in bases else None,
        "by_strategy": {},
    }

    for s in strategies:
        if s not in bases:
            continue
        by_k = {}
        for K in ks:
            if s == "A":
                final_keep, natural, clipped = strategy_A_for_K(bases["A"], K)
            elif s == "B":
                final_keep, natural, clipped = strategy_B_for_K(bases["B"], K)
            elif s == "C":
                final_keep, natural, clipped = strategy_C_for_K(bases["C"], K)
            elif s == "A-fixed":
                final_keep, natural, clipped = strategy_A_fixed_for_K(bases["A-fixed"], K)
            else:
                continue
            if clipped:
                by_k[str(K)] = {
                    "target_size": K,
                    "overflow": True,
                    "natural_size": natural,
                    "minimum_viable_K": natural,
                }
                print(f"ANOMALY: {label} strategy {s} K={K}: natural size {natural} exceeds target; cannot emit.")
                continue
            fert, new_raw, keep_idx, new_id_map = rebuild_and_measure(raw, final_keep, tok, texts)
            actual_K = len(keep_idx)
            by_k[str(K)] = {
                "target_size": K,
                "actual_size": actual_K,
                "overflow": False,
                "natural_size": natural,
                "padding_admitted": actual_K - natural,
                "fertility": fert,
            }
            print(
                f"{label} strategy {s} K={K} -> actual={actual_K} natural={natural} "
                f"fertility_before={fert['fertility_before_tokens_per_char']:.4f} "
                f"after={fert['fertility_after_tokens_per_char']:.4f} "
                f"inflation={fert['fertility_inflation_relative']:+.4%}"
            )
        results["by_strategy"][s] = by_k

    out_path = ROOT / f"results-keepset-v3-sweep-{label}.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out_path}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-label", required=True, choices=sorted(TOKENIZER_PATHS.keys()))
    ap.add_argument("--strategy", action="append", default=None, choices=["A", "B", "C", "D", "A-fixed"])
    ap.add_argument("--target-size", type=int, action="append", default=None)
    args = ap.parse_args()

    strategies = args.strategy if args.strategy else ["A", "B", "C", "D", "A-fixed"]
    ks = args.target_size if args.target_size else SWEEP_KS
    run_sweep(args.tokenizer_label, strategies, ks)


if __name__ == "__main__":
    main()
