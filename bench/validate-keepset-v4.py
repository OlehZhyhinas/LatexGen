#!/usr/bin/env python3
"""Keep-set v4 validation: v2's five checks (reused from
bench/validate-keepset-v2.py, unmodified functions), run against every
emitted bench/keepsets-v4/<label>-<K>/ directory, PLUS the two checks the
v4 brief adds on top:

6. Every accented-Latin and Greek SINGLE codepoint (the group-1 force-add
   universe) round-trips.
7. `L\u00f3pez-Monsalvo` and a set of other accented author names survive
   (round-trip against the original tokenizer's own roundtrip, same
   "relative roundtrip" methodology as checks 1/2/4 -- see
   validate-keepset-v2.py's module docstring for why that is the right test
   and a raw-string comparison is not).

A variant that regresses any check is not a candidate -- flagged plainly
in the printed summary and in results-keepset-v4-validation.json.
"""
import importlib.util
import json
import random
import unicodedata as ud
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEEPSETS_V4_ROOT = ROOT / "keepsets-v4"
RESULT_PATH = ROOT / "results-keepset-v4-validation.json"

RARE_WORD_PROBES = [
    "Nikolskii", "Randers", "Lissajous", "Kullback", "Hausdorff", "Chebyshev", "Sobolev",
]
ACCENTED_AUTHOR_NAME_PROBES = [
    "L\u00f3pez-Monsalvo", "A\u00e7\u0131kg\u00f6z", "Erd\u0151s", "Schr\u00f6dinger",
    "M\u00fcller", "Poincar\u00e9", "Rios-Ram\u00edrez", "\u00c7etin", "Bia\u0142ynicki-Birula",
]

TOKENIZER_PATHS = {
    "qwen35": "/Users/oleh/.cache/huggingface/hub/models--mlc-ai--Qwen3.5-0.8B-q4f16_1-MLC/snapshots/0ec138972555613c1d7812a821778ad0398c8790/tokenizer.json",
    "minicpm5-2b": "/Users/oleh/.cache/huggingface/hub/models--ozhyhinas--MiniCPM5-2B-q4f16_1-MLC/snapshots/2318f37d9c39277ff01dc64086491028c95d4db4/tokenizer.json",
}


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vc = _load_module("vocab_coverage_validate_v4", "vocab-coverage.py")
v2check = _load_module("validate_keepset_v2_dep", "validate-keepset-v2.py")
bkv2 = _load_module("build_keepset_v2_validate_v4", "build-keepset-v2.py")


def is_accented_latin_or_greek(ch):
    cp = ord(ch)
    if cp < 128:
        return False
    cat = ud.category(ch)
    name = ud.name(ch, "")
    if name.startswith("LATIN") and cat in ("Lu", "Ll"):
        return True
    lo, hi = (0x0370, 0x03FF)  # Greek and Coptic block
    if lo <= cp <= hi and cat in ("Lu", "Ll"):
        if "WITH TONOS" in name or "WITH DIALYTIKA" in name:
            return False
        return True
    return False


def all_accented_latin_and_greek_codepoints():
    """The same universe group 1 of build-keepset-v4.py's force-add set
    draws single-codepoint tokens from -- every codepoint in the BMP that
    classifies as an accented Latin letter or a Greek letter (unaccented),
    used here as an independent probe set (built from Unicode properties
    directly, not from build-keepset-v4.py's token-derived candidate list),
    so this check is not circular."""
    out = []
    for cp in range(0x80, 0x2100):
        ch = chr(cp)
        if is_accented_latin_or_greek(ch):
            out.append(ch)
    return out


def check_6_accented_and_greek_codepoints(old_tok, new_tok):
    chars = all_accented_latin_and_greek_codepoints()
    failures = []
    for ch in chars:
        ok, ids, decoded, dec_old = v2check.relative_roundtrip_ok(old_tok, new_tok, ch)
        if not ok:
            failures.append({"char": ch, "codepoint": f"U+{ord(ch):04X}", "name": ud.name(ch, ""), "decoded": decoded})
    return {
        "n_codepoints_probed": len(chars),
        "n_pass": len(chars) - len(failures),
        "all_pass": len(failures) == 0,
        "failures": failures[:50],
        "n_failures_total": len(failures),
    }


def check_7_accented_author_names(old_tok, new_tok):
    results = {}
    all_ok = True
    for name in ACCENTED_AUTHOR_NAME_PROBES:
        ok, ids, decoded, dec_old = v2check.relative_roundtrip_ok(old_tok, new_tok, name)
        results[name] = {"roundtrip_ok": ok, "decoded": decoded}
        if not ok:
            all_ok = False
    return {"all_ok": all_ok, "detail": results}


def main():
    from tokenizers import Tokenizer

    all_results = {}
    any_regression = False
    for label, tok_path in TOKENIZER_PATHS.items():
        old_tok, _ = vc.load_tokenizer(tok_path)
        conv = vc.load_conv_template(tok_path)
        per_K = {}
        for k_dir in sorted(KEEPSETS_V4_ROOT.glob(f"{label}-*")):
            K = k_dir.name.rsplit("-", 1)[-1]
            new_tok = Tokenizer.from_file(str(k_dir / "tokenizer.json"))
            keep_idx = set(json.load(open(k_dir / "keep-idx.json")))

            c1 = v2check.check_1_failures_gone(old_tok, new_tok, label)
            c1["latexgen_prompt_texts"] = v2check.check_1c_latexgen_prompt_roundtrip(old_tok, new_tok)
            if label == "qwen35":
                c1["named_failure1_ids"] = v2check.check_1b_qwen35_named_ids(keep_idx)
            c2 = v2check.check_2_short_piece_completeness(old_tok, new_tok)
            c3 = v2check.check_3_fertility(old_tok, new_tok)
            c4 = v2check.check_4_prompt_coverage(old_tok, new_tok, conv)
            c6 = check_6_accented_and_greek_codepoints(old_tok, new_tok)
            c7 = check_7_accented_author_names(old_tok, new_tok)

            regressions = []
            if not c1["maths_symbols_all_ok"]:
                regressions.append("check1_maths_symbols")
            if not c1["rare_words_all_ok"]:
                regressions.append("check1_rare_words")
            if not c1["latexgen_prompt_texts"]["all_ok"]:
                regressions.append("check1_latexgen_prompt_texts")
            if label == "qwen35" and not c1["named_failure1_ids"]["all_7_named_ids_in_keepset"]:
                regressions.append("check1_named_failure1_ids")
            if not c2["all_pass"]:
                regressions.append("check2_short_piece_completeness")
            if "plain_prompt" in c4 and c4["plain_prompt"]["fraction"] < 1.0:
                regressions.append("check4_plain_prompt_coverage")
            if "pdf_prompt" in c4 and c4["pdf_prompt"]["fraction"] < 1.0:
                regressions.append("check4_pdf_prompt_coverage")
            if not c6["all_pass"]:
                regressions.append("check6_accented_greek_codepoints")
            if not c7["all_ok"]:
                regressions.append("check7_accented_author_names")

            per_K[K] = {
                "check1_failures_gone": c1,
                "check2_short_piece_completeness": c2,
                "check3_fertility": c3,
                "check4_prompt_coverage": c4,
                "check6_accented_greek_codepoints": c6,
                "check7_accented_author_names": c7,
                "regressions": regressions,
                "is_candidate": len(regressions) == 0,
            }
            if regressions:
                any_regression = True
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
            print(f"  check6 accented/greek codepoints: {c6['n_pass']}/{c6['n_codepoints_probed']} all_pass={c6['all_pass']}")
            print(f"  check7 accented author names all_ok={c7['all_ok']}")
            print(f"  CANDIDATE: {len(regressions) == 0} (regressions={regressions})")
        all_results[label] = per_K

    RESULT_PATH.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"\nwrote {RESULT_PATH}")
    print(f"\nANY REGRESSION ACROSS ALL EMITTED DIRECTORIES: {any_regression}")


if __name__ == "__main__":
    main()
