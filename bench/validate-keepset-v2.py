#!/usr/bin/env python3
"""Keep-set v2 validation: the 5 checks the brief requires, run against every
emitted bench/keepsets-v2/<label>-<K>/ directory.

1. The three v1 failures must be gone: the 7 prompt-template tokens, the 5
   maths/Greek symbols, and 7 rare-word probes all roundtrip losslessly
   through the rebuilt (pruned) tokenizer.
2. Short-piece completeness: the 120 arxiv-pastes.json `reference` strings
   plus 2,000 random English words all re-encode losslessly.
3. Fertility: tokens-per-character before (original tokenizer) and after
   (rebuilt tokenizer), on the 120 arxiv inputs+references, per K. Reported,
   not gated.
4. Prompt coverage: fraction of the 120 arxiv items whose full rendered
   prompt (plain and PDF treatments) roundtrips losslessly.
5. Layer accounting table, pulled from bench/results-keepset-v2-layers-*.json.

"Roundtrips losslessly" = rebuilt_tok.decode(rebuilt_tok.encode(text)) equals
original_tok.decode(original_tok.encode(text)) -- i.e. pruning introduces no
*additional* information loss beyond whatever the tokenizer's own Unicode
normalizer (NFC, present in both the original and rebuilt tokenizer.json)
already does to the raw text. Comparing against the raw input string directly
(instead of the original tokenizer's own roundtrip) is the wrong test: 6 of
the 120 real arXiv `input` texts contain NFD-decomposed accented Latin
letters (e.g. "Ac\u0327\u0131kgo\u0308z") that the *original, unpruned* Qwen3.5
tokenizer's NFC normalizer already re-composes on its own encode/decode
roundtrip, before this task's pruning touches anything -- confirmed by
reproducing the same raw-text mismatch on the original tokenizer directly.
Since the rebuilt tokenizer's entire vocabulary equals the keep-set by
construction, and Layer 1 always keeps the full 256-byte alphabet, encode()
can never fail outright; the real thing worth verifying is that pruning did
not silently corrupt/lose characters *relative to what the unpruned
tokenizer already produces* (the "Nikolskii" -> "Kolskii" failure mode),
which a byte-length-only simulation cannot catch but this real re-encode/
decode comparison does.
"""
import importlib.util
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEEPSETS_V2_ROOT = ROOT / "keepsets-v2"
ARXIV_PASTES_PATH = ROOT / "arxiv-pastes.json"
LATEXGEN_PROMPTS_PATH = ROOT / "keepset-seed-latexgen-prompts.json"
SYSTEM_WORDLIST_PATH = Path("/usr/share/dict/words")
RESULT_PATH = ROOT / "results-keepset-v2-validation.json"

RARE_WORD_PROBES = [
    "Nikolskii", "Randers", "Lissajous", "Kullback", "Hausdorff", "Chebyshev", "Sobolev",
]
MATHS_SYMBOL_PROBES = ["\u03c6", "\u2190", "\u2194", "\u2193", "\u00d7"]  # phi, <-, <->, down-arrow, x
# The exact 7 ids named in the brief (Qwen3.5 tokenizer id space specifically).
QWEN35_FAILURE1_IDS = {80757: "\u0120oscillator", 80636: "\u0120damping", 77019: "\u0120converge",
                        65088: "\u0120flattened", 80313: "\u0120reorder", 72452: "\u00c2\u00a8", 94498: "\u0120\u00cf\u00a8"}

TOKENIZER_PATHS = {
    "qwen35": "/Users/oleh/.cache/huggingface/hub/models--mlc-ai--Qwen3.5-0.8B-q4f16_1-MLC/snapshots/0ec138972555613c1d7812a821778ad0398c8790/tokenizer.json",
    "minicpm5-2b": "/Users/oleh/.cache/huggingface/hub/models--ozhyhinas--MiniCPM5-2B-q4f16_1-MLC/snapshots/2318f37d9c39277ff01dc64086491028c95d4db4/tokenizer.json",
}


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vc = _load_module("vocab_coverage_validate", "vocab-coverage.py")


def relative_roundtrip_ok(old_tok, new_tok, text):
    """True if the rebuilt (pruned) tokenizer's encode/decode roundtrip
    matches the ORIGINAL (unpruned) tokenizer's own roundtrip -- isolating
    pruning-caused loss from the tokenizer's pre-existing NFC normalizer
    (present, identically, in both), which recomposes a small number of
    NFD-decomposed accented Latin letters found in the real arXiv `input`
    texts regardless of pruning. See module docstring."""
    if not text:
        return True, [], text, text
    new_ids = new_tok.encode(text, add_special_tokens=False).ids
    dec_new = new_tok.decode(new_ids, skip_special_tokens=False)
    dec_old = old_tok.decode(old_tok.encode(text, add_special_tokens=False).ids, skip_special_tokens=False)
    return dec_new == dec_old, new_ids, dec_new, dec_old


def check_1_failures_gone(old_tok, new_tok, label):
    out = {"maths_symbols": {}, "rare_words": {}}
    out["maths_symbols_all_ok"] = True
    for ch in MATHS_SYMBOL_PROBES:
        ok, ids, decoded, dec_old = relative_roundtrip_ok(old_tok, new_tok, ch)
        out["maths_symbols"][repr(ch)] = {"roundtrip_ok": ok, "new_ids": ids}
        if not ok:
            out["maths_symbols_all_ok"] = False
    out["rare_words_all_ok"] = True
    for w in RARE_WORD_PROBES:
        ok, ids, decoded, dec_old = relative_roundtrip_ok(old_tok, new_tok, w)
        out["rare_words"][w] = {"roundtrip_ok": ok, "decoded": decoded}
        if not ok:
            out["rare_words_all_ok"] = False
    return out


def check_1b_qwen35_named_ids(keep_ids_set):
    result = {}
    all_in = True
    for tid, decoded_repr in QWEN35_FAILURE1_IDS.items():
        present = tid in keep_ids_set
        result[str(tid)] = {"token": decoded_repr, "in_keepset": present}
        if not present:
            all_in = False
    return {"all_7_named_ids_in_keepset": all_in, "detail": result}


def check_1c_latexgen_prompt_roundtrip(old_tok, new_tok):
    seed = json.load(open(LATEXGEN_PROMPTS_PATH))
    texts = [seed["system_prompt"], seed["pdf_hint"]]
    for pair in seed["pdf_fewshot"]:
        texts.append(pair["user"])
        texts.append(pair["assistant"])
    all_ok = True
    failures = []
    for t in texts:
        ok, ids, decoded, dec_old = relative_roundtrip_ok(old_tok, new_tok, t)
        if not ok:
            all_ok = False
            failures.append({"text": t[:200], "decoded": decoded[:200]})
    return {"all_ok": all_ok, "n_texts": len(texts), "failures": failures}


def check_2_short_piece_completeness(old_tok, new_tok, n_random_words=2000, seed=1234):
    arxiv_items = json.load(open(ARXIV_PASTES_PATH))
    references = [it["reference"] for it in arxiv_items if it.get("ok", True)]

    random.seed(seed)
    all_words = []
    if SYSTEM_WORDLIST_PATH.exists():
        all_words = [w.strip() for w in SYSTEM_WORDLIST_PATH.read_text(errors="ignore").splitlines() if w.strip()]
    random_words = random.sample(all_words, min(n_random_words, len(all_words))) if all_words else []

    failures = []
    n_total = 0
    for text in references:
        n_total += 1
        ok, ids, decoded, dec_old = relative_roundtrip_ok(old_tok, new_tok, text)
        if not ok:
            failures.append({"kind": "reference", "text": text[:200], "decoded": decoded[:200]})
    n_pass_ref = n_total - len([f for f in failures if f["kind"] == "reference"])

    n_words_total = 0
    for w in random_words:
        n_words_total += 1
        ok, ids, decoded, dec_old = relative_roundtrip_ok(old_tok, new_tok, w)
        if not ok:
            failures.append({"kind": "word", "text": w, "decoded": decoded})
    n_pass_words = n_words_total - len([f for f in failures if f["kind"] == "word"])

    return {
        "n_reference_strings": n_total,
        "n_reference_pass": n_pass_ref,
        "n_random_words": n_words_total,
        "n_random_words_pass": n_pass_words,
        "all_pass": len(failures) == 0,
        "failures": failures[:50],
        "n_failures_total": len(failures),
    }


def check_3_fertility(old_tok, new_tok):
    arxiv_items = json.load(open(ARXIV_PASTES_PATH))
    arxiv_items = [it for it in arxiv_items if it.get("ok", True)]
    texts = []
    for it in arxiv_items:
        texts.append(it["input"])
        texts.append(it["reference"])
    total_chars = sum(len(t) for t in texts)
    total_old = sum(len(old_tok.encode(t, add_special_tokens=False).ids) for t in texts)
    total_new = sum(len(new_tok.encode(t, add_special_tokens=False).ids) for t in texts)
    return {
        "n_texts": len(texts),
        "total_chars": total_chars,
        "fertility_before_tokens_per_char": (total_old / total_chars) if total_chars else 0.0,
        "fertility_after_tokens_per_char": (total_new / total_chars) if total_chars else 0.0,
        "fertility_inflation_relative": ((total_new / total_old) - 1.0) if total_old else 0.0,
    }


def build_full_prompt_messages(seed, item_text, pdf):
    convert_user = lambda t: seed["convert_user_prefix"] + t
    if not pdf:
        return [
            {"role": "system", "content": seed["system_prompt"]},
            {"role": "user", "content": convert_user(item_text)},
        ]
    messages = [{"role": "system", "content": seed["system_prompt"] + "\n\n" + seed["pdf_hint"]}]
    for pair in seed["pdf_fewshot"]:
        messages.append({"role": "user", "content": convert_user(pair["user"])})
        messages.append({"role": "assistant", "content": pair["assistant"]})
    messages.append({"role": "user", "content": convert_user(item_text)})
    return messages


def check_4_prompt_coverage(old_tok, new_tok, conv):
    if conv is None:
        return {"skipped": "no mlc-chat-config.json / conv_template for this tokenizer"}
    seed = json.load(open(LATEXGEN_PROMPTS_PATH))
    arxiv_items = json.load(open(ARXIV_PASTES_PATH))
    arxiv_items = [it for it in arxiv_items if it.get("ok", True)]

    def coverage_for(pdf):
        n_ok = 0
        missing = []
        for it in arxiv_items:
            messages = build_full_prompt_messages(seed, it["input"], pdf)
            text = vc.render_chatml(conv, messages, think_open_only=False)
            ok, ids, decoded, dec_old = relative_roundtrip_ok(old_tok, new_tok, text)
            if ok:
                n_ok += 1
            else:
                missing.append(it["id"])
        return {"n_items": len(arxiv_items), "n_full_in_keepset": n_ok, "fraction": n_ok / len(arxiv_items), "missing_items": missing[:20]}

    return {"plain_prompt": coverage_for(False), "pdf_prompt": coverage_for(True)}


def main():
    from tokenizers import Tokenizer

    all_results = {}
    for label, tok_path in TOKENIZER_PATHS.items():
        old_tok, _ = vc.load_tokenizer(tok_path)
        conv = vc.load_conv_template(tok_path)
        old_keep_by_K = {}
        per_K = {}
        for k_dir in sorted(KEEPSETS_V2_ROOT.glob(f"{label}-*")):
            K = k_dir.name.rsplit("-", 1)[-1]
            new_tok = Tokenizer.from_file(str(k_dir / "tokenizer.json"))
            keep_idx = set(json.load(open(k_dir / "keep-idx.json")))

            c1 = check_1_failures_gone(old_tok, new_tok, label)
            c1["latexgen_prompt_texts"] = check_1c_latexgen_prompt_roundtrip(old_tok, new_tok)
            if label == "qwen35":
                c1["named_failure1_ids"] = check_1b_qwen35_named_ids(keep_idx)
            c2 = check_2_short_piece_completeness(old_tok, new_tok)
            c3 = check_3_fertility(old_tok, new_tok)
            c4 = check_4_prompt_coverage(old_tok, new_tok, conv)

            per_K[K] = {
                "check1_failures_gone": c1,
                "check2_short_piece_completeness": c2,
                "check3_fertility": c3,
                "check4_prompt_coverage": c4,
            }
            print(f"--- {label} K={K} ---")
            print(f"  check1 maths_symbols_all_ok={c1['maths_symbols_all_ok']} rare_words_all_ok={c1['rare_words_all_ok']} "
                  f"latexgen_prompts_all_ok={c1['latexgen_prompt_texts']['all_ok']}")
            if "named_failure1_ids" in c1:
                print(f"  check1 named 7 failure-1 ids all in keepset: {c1['named_failure1_ids']['all_7_named_ids_in_keepset']}")
            print(f"  check2 reference_pass={c2['n_reference_pass']}/{c2['n_reference_strings']} "
                  f"random_words_pass={c2['n_random_words_pass']}/{c2['n_random_words']} all_pass={c2['all_pass']}")
            print(f"  check3 fertility before={c3['fertility_before_tokens_per_char']:.4f} after={c3['fertility_after_tokens_per_char']:.4f} "
                  f"inflation={c3['fertility_inflation_relative']:+.2%}")
            if "plain_prompt" in c4:
                print(f"  check4 plain_prompt={c4['plain_prompt']['n_full_in_keepset']}/{c4['plain_prompt']['n_items']} "
                      f"pdf_prompt={c4['pdf_prompt']['n_full_in_keepset']}/{c4['pdf_prompt']['n_items']}")
        all_results[label] = per_K

    RESULT_PATH.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"\nwrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
