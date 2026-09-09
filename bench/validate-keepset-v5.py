#!/usr/bin/env python3
"""Keep-set v5 validation: reuses bench/validate-keepset-v2.py's checks
unmodified (imported, not reimplemented) against bench/keepsets-v5/, plus
the residual-risk probe the brief calls out by name (Nikolskii-class:
confirm whether the rare-word probes' individual SUBWORD PIECES, not just
the whole word's roundtrip, are actually present in the keep-set)."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEEPSETS_V5_ROOT = ROOT / "keepsets-v5"
RESULT_PATH = ROOT / "results-keepset-v5-validation.json"


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v2val = _load_module("validate_keepset_v2_dep", "validate-keepset-v2.py")
vc = v2val.vc


def check_rare_word_subword_pieces(tok, keep_ids):
    """The brief's residual-risk note: a whole-word roundtrip check can pass
    even though pruning silently swapped one of the word's SUBWORD PIECES
    for a different piece with the same decoded output by coincidence (not
    possible for byte-level BPE in practice, since decode is injective per
    id, but the more real risk is the *opposite* direction the brief names:
    a dropped ASCII subword elsewhere in the vocab corrupting a DIFFERENT
    rare name that happens to share that piece). Report, per rare-word
    probe, exactly which of its ORIGINAL-tokenizer subword-piece ids are and
    are not in this keep-set, not just the aggregate roundtrip boolean."""
    out = {}
    for w in v2val.RARE_WORD_PROBES:
        ids = tok.encode(w, add_special_tokens=False).ids
        pieces = [{"id": i, "token": tok.id_to_token(i), "in_keepset": i in keep_ids} for i in ids]
        out[w] = {"pieces": pieces, "all_pieces_in_keepset": all(p["in_keepset"] for p in pieces)}
    return out


def main():
    from tokenizers import Tokenizer

    all_results = {}
    for label, tok_path in v2val.TOKENIZER_PATHS.items():
        old_tok, _ = vc.load_tokenizer(tok_path)
        conv = vc.load_conv_template(tok_path)
        per_K = {}
        for k_dir in sorted(KEEPSETS_V5_ROOT.glob(f"{label}-*")):
            K = k_dir.name.rsplit("-", 1)[-1]
            new_tok = Tokenizer.from_file(str(k_dir / "tokenizer.json"))
            keep_idx = set(json.load(open(k_dir / "keep-idx.json")))

            c1 = v2val.check_1_failures_gone(old_tok, new_tok, label)
            c1["latexgen_prompt_texts"] = v2val.check_1c_latexgen_prompt_roundtrip(old_tok, new_tok)
            if label == "qwen35":
                c1["named_failure1_ids"] = v2val.check_1b_qwen35_named_ids(keep_idx)
            c2 = v2val.check_2_short_piece_completeness(old_tok, new_tok)
            c3 = v2val.check_3_fertility(old_tok, new_tok)
            c4 = v2val.check_4_prompt_coverage(old_tok, new_tok, conv)
            c5_subword = check_rare_word_subword_pieces(old_tok, keep_idx)

            # Accented-Latin / Greek single-codepoint roundtrip (brief's
            # "every accented-Latin and Greek single codepoint round-trips"),
            # derived the same way build-keepset-v5.py's force-add group 1
            # does (Unicode-property scan over the ORIGINAL tokenizer's
            # single-codepoint tokens), not a hand list.
            import unicodedata as ud
            byte_ids = vc.get_byte_token_ids(old_tok)
            added_ids = set(old_tok.get_added_tokens_decoder().keys())
            probe_chars = []
            for tid in range(old_tok.get_vocab_size(with_added_tokens=True)):
                if tid in byte_ids or tid in added_ids:
                    continue
                txt = old_tok.decode([tid], skip_special_tokens=False)
                content = txt[1:] if txt.startswith(" ") else txt
                if len(content) != 1:
                    continue
                cp = ord(content)
                if cp < 128:
                    continue
                cat = ud.category(content)
                name = ud.name(content, "")
                is_accented_latin = name.startswith("LATIN") and cat in ("Lu", "Ll")
                is_greek = False
                try:
                    ku = _load_module("ku_dep_v5", "keepset-v2-unicode.py")
                    if ku.is_maths_or_greek_char(content):
                        lo, hi = ku.GREEK_BLOCK
                        is_greek = lo <= cp <= hi
                except Exception:
                    pass
                if is_accented_latin or is_greek:
                    probe_chars.append(content)
            n_ok = 0
            failures = []
            for ch in probe_chars:
                ok, ids, decoded, dec_old = v2val.relative_roundtrip_ok(old_tok, new_tok, ch)
                if ok:
                    n_ok += 1
                else:
                    failures.append(ch)
            c6 = {"n_probes": len(probe_chars), "n_pass": n_ok, "all_pass": len(failures) == 0, "failures": failures[:20]}

            per_K[K] = {
                "check1_failures_gone": c1,
                "check2_short_piece_completeness": c2,
                "check3_fertility": c3,
                "check4_prompt_coverage": c4,
                "check5_rare_word_subword_pieces": c5_subword,
                "check6_accented_latin_and_greek_single_codepoint_roundtrip": c6,
            }
            candidate = (
                c1["maths_symbols_all_ok"] and c1["rare_words_all_ok"] and c1["latexgen_prompt_texts"]["all_ok"]
                and c2["all_pass"] and c6["all_pass"]
                and (c1.get("named_failure1_ids", {}).get("all_7_named_ids_in_keepset", True))
                and (c4.get("plain_prompt", {}).get("fraction", 1.0) == 1.0)
                and (c4.get("pdf_prompt", {}).get("fraction", 1.0) == 1.0)
            )
            per_K[K]["candidate"] = candidate

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
            print(f"  check5 rare-word all-pieces-present: " +
                  ", ".join(f"{w}={v['all_pieces_in_keepset']}" for w, v in c5_subword.items()))
            print(f"  check6 accented-latin/greek single-codepoint: {n_ok}/{len(probe_chars)} pass")
            print(f"  CANDIDATE: {candidate}")
        all_results[label] = per_K

    RESULT_PATH.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"\nwrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
