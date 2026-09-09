#!/usr/bin/env python3
"""Keep-set v5: the honest optimum, on genuinely held-out text.

Every prior generation (v1-v4) built its keep-set from a base layer that
includes the token ids of the evaluation text itself (v1/A: the 5 default
training corpus files include arxiv-pastes.json / bench-data-extended.json
directly; v3/A-fixed and v4 pad by merge rank starting from a natural set,
which is contamination-free for the *natural* layer but v3's own remeasurement
already showed the in-sample sweep is unusable for optimum-picking). This
script builds every keep-set from ONLY:

  (a) structural force-adds (byte alphabet, added/special tokens -- required
      by the tokenizer format itself, not evaluation-derived; plus the v4
      force-add set: single-codepoint accented-Latin/Greek/math-operator/
      superscript-subscript-digit tokens, LaTeX macro completion via
      bench/keepset-seed-latex.json, and LatexGen's own rendered prompt --
      all task knowledge, never text from arxiv-pastes.json or
      bench-data-extended.json),
  (b) a ranking derived from the external science corpus
      (bench/corpus/freq-<label>.json, PR #58, harvested from TeX/Physics/
      Stats/CS StackExchange, arXiv abstracts, OpenStax, science Wikipedia/
      Wikibooks -- verified to contain none of the evaluation items), and/or
  (c) plain BPE merge rank (ascending original token id).

Nothing else. `bench/arxiv-pastes.json` (120 input/reference pairs) and
`bench/bench-data-extended.json` are read ONLY for measurement (real
re-encoding with the rebuilt tokenizer), never folded into any keep-set.

Strategies (see module-level STRATEGIES for the exact definitions):
  M     - merge rank only (the honest baseline)
  M+F   - merge rank + v4 force-add set
  C     - external-corpus rank only
  C+F   - corpus rank + v4 force-add set
  S     - corpus rank reordered by marginal token-savings
          savings(t) = corpus_count(t) * (depth(t) - 1), where depth(t) is
          the number of byte-alphabet-rooted merge-tree leaves under t (see
          `token_depth` below for the exact, tractable definition and its
          documented limitation).
"""
import argparse
import collections
import functools
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARXIV_PASTES_PATH = ROOT / "arxiv-pastes.json"
BENCH_EXTENDED_PATH = ROOT / "bench-data-extended.json"
LATEX_SEED_PATH = ROOT / "keepset-seed-latex.json"
LATEXGEN_PROMPTS_PATH = ROOT / "keepset-seed-latexgen-prompts.json"
CORPUS_DIR = ROOT / "corpus"

TARGET_SIZES = [16384, 24576, 32768, 49152, 65536]
STRATEGIES = ["M", "M+F", "C", "C+F", "S"]

TOKENIZER_PATHS = {
    "qwen35": "/Users/oleh/.cache/huggingface/hub/models--mlc-ai--Qwen3.5-0.8B-q4f16_1-MLC/snapshots/0ec138972555613c1d7812a821778ad0398c8790/tokenizer.json",
    "minicpm5-2b": "/Users/oleh/.cache/huggingface/hub/models--ozhyhinas--MiniCPM5-2B-q4f16_1-MLC/snapshots/2318f37d9c39277ff01dc64086491028c95d4db4/tokenizer.json",
}
CORPUS_FREQ_PATHS = {
    "qwen35": CORPUS_DIR / "freq-qwen35.json",
    "minicpm5-2b": CORPUS_DIR / "freq-minicpm5-2b.json",
}

# The 7 ids named in the brief, in the Qwen3.5 tokenizer's id space (see
# bench/keepset-seed-latexgen-prompts.json's provenance note).
NAMED_QWEN35_PROMPT_IDS = {80757, 80636, 77019, 65088, 80313, 72452, 94498}


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vc = _load_module("vocab_coverage_v5dep", "vocab-coverage.py")
rk = _load_module("rebuild_keepset_tokenizer_v5dep", "rebuild-keepset-tokenizer.py")
ku = _load_module("keepset_v2_unicode_v5dep", "keepset-v2-unicode.py")
bkv2 = _load_module("build_keepset_v2_v5dep", "build-keepset-v2.py")


# =============================================================================
# Held-out text: NEVER used to build any keep-set. Two views:
#  - "eval240": exactly the 120 arxiv-pastes.json input+reference strings
#    (240 texts) -- matches the brief's own contamination-verification
#    numbers and bench/corpus/README-corpus.md's Analysis 4 exactly.
#  - "held_out_full": eval240 + bench-data-extended.json's `ok` items'
#    input+reference (matches v3/v4's broader fertility-grid set).
# =============================================================================

def load_texts():
    eval240 = []
    arxiv_items = json.load(open(ARXIV_PASTES_PATH))
    n_arxiv = 0
    for it in arxiv_items:
        if not it.get("ok", True):
            continue
        eval240.append(it["input"])
        eval240.append(it["reference"])
        n_arxiv += 1

    extra = []
    n_bench_ext = 0
    bench_ext_items = json.load(open(BENCH_EXTENDED_PATH))
    for it in bench_ext_items:
        if not it.get("ok", True):
            continue
        extra.append(it.get("input", ""))
        extra.append(it.get("reference", ""))
        n_bench_ext += 1

    held_out_full = eval240 + extra
    return {
        "eval240": eval240,
        "held_out_full": held_out_full,
        "n_arxiv": n_arxiv,
        "n_bench_ext": n_bench_ext,
    }


def fertility_and_oos(tok, keep_ids, texts):
    """Real re-encoding (never simulated) with the ORIGINAL tokenizer:
    fertility before pruning, and out-of-set rate against `keep_ids`
    (the fraction of original-space tokens that would need >=2 pieces
    after pruning to keep_ids)."""
    total_chars = 0
    total_tokens = 0
    total_oos = 0
    for t in texts:
        if not t:
            continue
        ids = tok.encode(t, add_special_tokens=False).ids
        total_chars += len(t)
        total_tokens += len(ids)
        total_oos += sum(1 for i in ids if i not in keep_ids)
    return {
        "n_texts": len([t for t in texts if t]),
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "oos_tokens": total_oos,
        "oos_rate": (total_oos / total_tokens) if total_tokens else 0.0,
        "fertility_before": (total_tokens / total_chars) if total_chars else 0.0,
    }


def real_fertility_after(new_tok, texts):
    total_chars = 0
    total_tokens = 0
    for t in texts:
        if not t:
            continue
        ids = new_tok.encode(t, add_special_tokens=False).ids
        total_chars += len(t)
        total_tokens += len(ids)
    return {
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "fertility_after": (total_tokens / total_chars) if total_chars else 0.0,
    }


# =============================================================================
# Force-add set F (the v4 force-add set, rebuilt here since keepset-v4 has
# not landed any commits yet -- see bench/README-keepset-v5.md). Logic
# ported from the (uncommitted, in-progress) bench/build-keepset-v4.py in
# the sibling keepset-v4 worktree, read-only, not copied verbatim as a
# file (that worktree's files are not committed anywhere to cherry-pick
# from) -- reimplemented against this script's own module set.
# =============================================================================

def classify_force_add_char(ch):
    import unicodedata as ud
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


def compute_force_add_group1(tok, byte_ids, added_ids):
    text_by_id = bkv2.decode_all_regular_tokens(tok, byte_ids, added_ids)
    classified = {}
    for tid, txt in text_by_id.items():
        has_space = txt.startswith(" ")
        content = txt[1:] if has_space else txt
        if len(content) != 1:
            continue
        cls = classify_force_add_char(content)
        if cls:
            classified[tid] = cls
    return classified


def compute_force_add_group2(tok):
    seed = json.load(open(LATEX_SEED_PATH))
    strings = seed["all"]
    ids_all = vc.seed_ids_from_strings(tok, strings, try_leading_space=True)
    return ids_all, {"n_macro_strings": len(strings)}


SAMPLE_TEXTS = [
    "the integral from zero to infinity of e to the minus x squared dx",
    "the quadratic formula for a x squared plus b x plus c equals zero",
]


def compute_force_add_group3(tok, conv, tokenizer_label):
    ids = set()
    n_renders = 0
    if conv is not None:
        lg_seed = json.load(open(LATEXGEN_PROMPTS_PATH))
        for sample_text in SAMPLE_TEXTS:
            for _, messages in bkv2.latexgen_prompt_messages(lg_seed, sample_text):
                for think_variant in (True, False):
                    text = vc.render_chatml(conv, messages, think_variant)
                    ids.update(vc.tokenize(tok, text))
                    n_renders += 1
    named_ids = NAMED_QWEN35_PROMPT_IDS if tokenizer_label == "qwen35" else set()
    ids |= named_ids
    return ids, {"n_renders": n_renders, "n_named_ids": len(named_ids)}


def build_force_add_set(tok, byte_ids, added_ids, conv, tokenizer_label):
    group1 = compute_force_add_group1(tok, byte_ids, added_ids)
    group1_ids = set(group1.keys())
    group1_by_class = collections.Counter(group1.values())
    group2_ids, group2_meta = compute_force_add_group2(tok)
    group3_ids, group3_meta = compute_force_add_group3(tok, conv, tokenizer_label)
    union = group1_ids | group2_ids | group3_ids
    accounting = {
        "group1_single_codepoint": {"total": len(group1_ids), "by_class": dict(group1_by_class)},
        "group2_latex_macro_completion": {"total": len(group2_ids), **group2_meta},
        "group3_latexgen_prompt_and_chat_template": {"total": len(group3_ids), **group3_meta},
        "union_pre_closure": len(union),
    }
    return union, accounting


# =============================================================================
# Merge-tree depth (for strategy S's marginal-savings ranking)
# =============================================================================

def build_depth_fn(merge_map, id_to_str, str_to_id, byte_ids):
    memo = {}

    def depth(tid):
        if tid in memo:
            return memo[tid]
        if tid in byte_ids:
            memo[tid] = 1
            return 1
        s = id_to_str.get(tid)
        parents = merge_map.get(s) if s is not None else None
        if not parents:
            memo[tid] = 1  # added/special/leaf-without-a-recorded-merge: atomic
            return 1
        a, b = parents
        a_id, b_id = str_to_id.get(a), str_to_id.get(b)
        if a_id is None or b_id is None:
            memo[tid] = 1
            return 1
        memo[tid] = -1  # cycle guard (should never trigger for a DAG merge tree)
        d = depth(a_id) + depth(b_id)
        memo[tid] = d
        return d

    return depth


# =============================================================================
# Candidate ranking orders
# =============================================================================

def build_rank_orders(regular_ids, corpus_rank_by_id, depth_by_id, corpus_count_by_id):
    order_M = sorted(regular_ids)  # ascending id == merge rank
    order_C = sorted(
        regular_ids,
        key=lambda tid: (corpus_rank_by_id.get(tid, float("inf")), tid),
    )
    order_S = sorted(
        regular_ids,
        key=lambda tid: (
            -(corpus_count_by_id.get(tid, 0) * (depth_by_id.get(tid, 1) - 1)),
            corpus_rank_by_id.get(tid, float("inf")),
            tid,
        ),
    )
    return {"M": order_M, "C": order_C, "S": order_S}


# =============================================================================
# Fill-to-K with iterative merge-ancestor-closure fold (candidates ranked by
# an arbitrary (non-ascending-id) order are not automatically closure-safe
# the way M/M+F's ascending-id fill is -- see bench/README-keepset-v5.md).
# =============================================================================

def fill_to_K(natural, order, K, raw, merge_map):
    mandatory = set(natural)
    order_list = list(order)
    for _iteration in range(6):
        selected = set()
        new_count = 0
        budget = K - len(mandatory)
        if budget <= 0:
            break
        for tid in order_list:
            if tid in mandatory or tid in selected:
                continue
            selected.add(tid)
            new_count += 1
            if new_count >= budget:
                break
        candidate_final = mandatory | selected
        closure = bkv2.compute_merge_ancestor_closure(raw, merge_map, candidate_final)
        extra = closure - candidate_final
        if not extra:
            return candidate_final, _iteration
        mandatory = mandatory | extra
    # Fixed point not reached in 6 iterations (never observed in practice for
    # this tokenizer/corpus combination) -- return the best available,
    # closure-safe by construction (mandatory always folds extras in).
    final = mandatory | selected
    final = bkv2.compute_merge_ancestor_closure(raw, merge_map, final)
    return final, 6


def build_keepset(strategy, tokenizer_label, K, ctx):
    """Returns (final_keep_ids, meta) for one (strategy, tokenizer, K)."""
    natural_base = ctx["natural_base"]
    F_ids_closed = ctx["F_ids_closed"]
    raw = ctx["raw"]
    merge_map = ctx["merge_map"]
    orders = ctx["orders"]

    has_F = "+F" in strategy
    base_strategy = strategy.replace("+F", "")
    natural = natural_base | F_ids_closed if has_F else natural_base
    order = orders[base_strategy]

    if len(natural) >= K:
        return natural, {"overflow": True, "natural_size": len(natural)}

    final_keep, n_iter = fill_to_K(natural, order, K, raw, merge_map)
    return final_keep, {"overflow": False, "natural_size": len(natural), "closure_iterations": n_iter}


# =============================================================================
# Main
# =============================================================================

def load_corpus_rank(tokenizer_label):
    path = CORPUS_FREQ_PATHS[tokenizer_label]
    data = json.load(open(path))
    rank_by_id = {}
    count_by_id = {}
    for row in data["ranked"]:
        rank_by_id[row["id"]] = row["rank"]
        count_by_id[row["id"]] = row["count"]
    return rank_by_id, count_by_id, data


def build_context(tokenizer_label):
    tok_path = TOKENIZER_PATHS[tokenizer_label]
    tok, _ = vc.load_tokenizer(tok_path)
    raw = json.loads(Path(tok_path).read_text())
    conv = vc.load_conv_template(tok_path)

    byte_ids = vc.get_byte_token_ids(tok)
    added_decoder = tok.get_added_tokens_decoder()
    added_ids = set(added_decoder.keys())
    special_ids = {tid for tid, t in added_decoder.items() if getattr(t, "special", False)}
    regular_ids = vc.get_regular_token_ids(tok, byte_ids, added_ids)
    natural_base = byte_ids | added_ids | special_ids

    merge_map = bkv2.compute_merge_map(raw)
    vocab = raw["model"]["vocab"]
    id_to_str = {v: k for k, v in vocab.items()}
    for e in raw.get("added_tokens", []):
        id_to_str.setdefault(e["id"], e["content"])
    str_to_id = dict(vocab)
    for e in raw.get("added_tokens", []):
        str_to_id.setdefault(e["content"], e["id"])

    F_ids_pre, F_accounting = build_force_add_set(tok, byte_ids, added_ids, conv, tokenizer_label)
    closure_of_natural_and_F = bkv2.compute_merge_ancestor_closure(raw, merge_map, natural_base | F_ids_pre)
    F_ids_closed = closure_of_natural_and_F - natural_base
    F_accounting["closure_extra_beyond_natural_and_pre_closure_union"] = len(
        closure_of_natural_and_F - (natural_base | F_ids_pre)
    )

    corpus_rank_by_id, corpus_count_by_id, corpus_meta = load_corpus_rank(tokenizer_label)

    depth_fn = build_depth_fn(merge_map, id_to_str, str_to_id, byte_ids)
    depth_by_id = {tid: depth_fn(tid) for tid in regular_ids}

    orders = build_rank_orders(regular_ids, corpus_rank_by_id, depth_by_id, corpus_count_by_id)

    return {
        "tokenizer_label": tokenizer_label,
        "tok": tok,
        "tok_path": tok_path,
        "raw": raw,
        "conv": conv,
        "byte_ids": byte_ids,
        "added_ids": added_ids,
        "special_ids": special_ids,
        "regular_ids": regular_ids,
        "natural_base": natural_base,
        "merge_map": merge_map,
        "id_to_str": id_to_str,
        "str_to_id": str_to_id,
        "F_ids_closed": F_ids_closed,
        "F_accounting": F_accounting,
        "corpus_rank_by_id": corpus_rank_by_id,
        "corpus_count_by_id": corpus_count_by_id,
        "corpus_meta": {
            "total_tokens": corpus_meta["total_tokens"],
            "total_docs": corpus_meta["total_docs"],
            "n_distinct_ids_seen": corpus_meta["n_distinct_ids_seen"],
            "vocab_size": corpus_meta["vocab_size"],
        },
        "depth_by_id": depth_by_id,
        "orders": orders,
    }


def rebuild_in_memory_tokenizer(raw, final_keep):
    from tokenizers import Tokenizer

    keep_idx = sorted(final_keep)
    new_id_map = {old: i for i, old in enumerate(keep_idx)}
    new_raw, n_before, n_after, n_dropped = rk.rebuild_tokenizer_json(raw, final_keep, new_id_map)
    new_tok = Tokenizer.from_str(json.dumps(new_raw))
    return new_tok, new_raw, keep_idx, new_id_map, {"before": n_before, "after": n_after, "dropped": n_dropped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-label", action="append", default=None, choices=sorted(TOKENIZER_PATHS.keys()))
    ap.add_argument("--target-size", type=int, action="append", default=None)
    ap.add_argument("--strategy", action="append", default=None, choices=STRATEGIES)
    args = ap.parse_args()

    labels = args.tokenizer_label or sorted(TOKENIZER_PATHS.keys())
    target_sizes = args.target_size or TARGET_SIZES
    strategies = args.strategy or STRATEGIES

    texts = load_texts()
    print(f"held-out text: eval240 n={len(texts['eval240'])} (arxiv items={texts['n_arxiv']}), "
          f"held_out_full n={len(texts['held_out_full'])} (+bench-extended items={texts['n_bench_ext']})")

    all_results = {}
    for label in labels:
        print(f"\n=== building context: {label} ===")
        ctx = build_context(label)
        print(f"  natural_base (byte+added+special): {len(ctx['natural_base'])}")
        print(f"  force-add F (pre-closure): {ctx['F_accounting']['union_pre_closure']}, "
              f"closed size: {len(ctx['F_ids_closed'])}")
        print(f"  corpus: {ctx['corpus_meta']['n_distinct_ids_seen']} distinct ids seen of "
              f"{ctx['corpus_meta']['vocab_size']} vocab, {ctx['corpus_meta']['total_tokens']} tokens")

        old_tok = ctx["tok"]
        rows = []
        for strategy in strategies:
            for K in target_sizes:
                final_keep, meta = build_keepset(strategy, label, K, ctx)
                if meta.get("overflow"):
                    print(f"  {strategy} K={K}: OVERFLOW (natural {meta['natural_size']} > K)")
                    rows.append({"strategy": strategy, "target_K": K, "overflow": True, **meta})
                    continue
                actual_K = len(final_keep)
                new_tok, new_raw, keep_idx, new_id_map, merges_info = rebuild_in_memory_tokenizer(ctx["raw"], final_keep)

                m_240 = fertility_and_oos(old_tok, final_keep, texts["eval240"])
                m_240_after = real_fertility_after(new_tok, texts["eval240"])
                m_full = fertility_and_oos(old_tok, final_keep, texts["held_out_full"])
                m_full_after = real_fertility_after(new_tok, texts["held_out_full"])

                inflation_240 = (m_240_after["fertility_after"] / m_240["fertility_before"] - 1.0) if m_240["fertility_before"] else 0.0
                inflation_full = (m_full_after["fertility_after"] / m_full["fertility_before"] - 1.0) if m_full["fertility_before"] else 0.0

                row = {
                    "strategy": strategy,
                    "target_K": K,
                    "actual_K": actual_K,
                    "overflow": False,
                    "natural_size": meta["natural_size"],
                    "closure_iterations": meta.get("closure_iterations"),
                    "eval240": {
                        "oos_rate": m_240["oos_rate"],
                        "oos_tokens": m_240["oos_tokens"],
                        "total_tokens": m_240["total_tokens"],
                        "fertility_before": m_240["fertility_before"],
                        "fertility_after": m_240_after["fertility_after"],
                        "inflation": inflation_240,
                    },
                    "held_out_full": {
                        "oos_rate": m_full["oos_rate"],
                        "oos_tokens": m_full["oos_tokens"],
                        "total_tokens": m_full["total_tokens"],
                        "fertility_before": m_full["fertility_before"],
                        "fertility_after": m_full_after["fertility_after"],
                        "inflation": inflation_full,
                    },
                }
                rows.append(row)
                print(
                    f"  {strategy:5s} K={K:6d} actual={actual_K:6d} "
                    f"eval240_oos={m_240['oos_rate']:.4%} eval240_inflation={inflation_240:+.4%} "
                    f"full_inflation={inflation_full:+.4%}"
                )

        all_results[label] = {
            "natural_base_size": len(ctx["natural_base"]),
            "force_add_accounting": ctx["F_accounting"],
            "force_add_closed_size": len(ctx["F_ids_closed"]),
            "corpus_meta": ctx["corpus_meta"],
            "rows": rows,
        }

    out_path = ROOT / "results-keepset-v5-grid.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
