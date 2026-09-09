#!/usr/bin/env python3
"""Job 4 validation: re-run v2's five checks (bench/validate-keepset-v2.py,
imported and reused, not reimplemented) against every emitted
bench/keepsets-v3/<label>-<K>/ directory:

1. The 7 prompt-template tokens' failure-1 ids (Qwen3.5) + the 5 maths/Greek
   symbol probes + the 7 rare-word probes all roundtrip losslessly.
2. Short-piece completeness: 120 arxiv-pastes.json references + 2,000
   random English words.
3. Fertility (v2's own 120-item methodology, reported alongside v3's own
   426-item number already in coverage.json, for direct comparability with
   v1/v2's historical numbers).
4. Prompt coverage (plain + pdf treatments), 120/120 target.
5. Layer/selection accounting -- pulled from this directory's own
   coverage.json (strategy A carries no discrete "layer" accounting the way
   v2's structural layers do, so this is reported as natural size + padding
   admitted instead).

A variant that regresses any of checks 1/2/4 is flagged NOT A CANDIDATE,
regardless of its fertility number, per the brief.
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEEPSETS_V3_ROOT = ROOT / "keepsets-v3"
RESULT_PATH = ROOT / "results-keepset-v3-validation.json"


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v2val = _load_module("validate_keepset_v2_dep", "validate-keepset-v2.py")
vc = v2val.vc


def main():
    from tokenizers import Tokenizer

    all_results = {}
    for label, tok_path in v2val.TOKENIZER_PATHS.items():
        old_tok, _ = vc.load_tokenizer(tok_path)
        conv = vc.load_conv_template(tok_path)
        per_K = {}
        for k_dir in sorted(KEEPSETS_V3_ROOT.glob(f"{label}-*")):
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

            failure1_ok = c1.get("named_failure1_ids", {}).get("all_7_named_ids_in_keepset", True)
            is_candidate = (
                c1["maths_symbols_all_ok"]
                and c1["rare_words_all_ok"]
                and c1["latexgen_prompt_texts"]["all_ok"]
                and failure1_ok
                and c2["all_pass"]
                and (c4.get("plain_prompt", {}).get("fraction") == 1.0)
                and (c4.get("pdf_prompt", {}).get("fraction") == 1.0)
            )

            coverage = json.load(open(k_dir / "coverage.json"))
            per_K[K] = {
                "check1_failures_gone": c1,
                "check2_short_piece_completeness": c2,
                "check3_fertility_v2_methodology_120_items": c3,
                "check4_prompt_coverage": c4,
                "check5_selection_accounting": {
                    "selection_strategy": coverage.get("selection_strategy"),
                    "natural_keep_set_size": coverage.get("natural_keep_set_size"),
                    "target_size": coverage.get("target_size"),
                    "actual_size": coverage.get("actual_size"),
                    "padding_admitted": coverage.get("actual_size", 0) - coverage.get("natural_keep_set_size", 0),
                },
                "v3_methodology_fertility_426_items": coverage.get("v3_methodology_fertility"),
                "IS_CANDIDATE": is_candidate,
            }
            print(f"--- {label} K={K} ---")
            print(f"  check1 maths_symbols_all_ok={c1['maths_symbols_all_ok']} rare_words_all_ok={c1['rare_words_all_ok']} "
                  f"latexgen_prompts_all_ok={c1['latexgen_prompt_texts']['all_ok']}")
            if "named_failure1_ids" in c1:
                print(f"  check1 named 7 failure-1 ids all in keepset: {c1['named_failure1_ids']['all_7_named_ids_in_keepset']}")
            print(f"  check2 reference_pass={c2['n_reference_pass']}/{c2['n_reference_strings']} "
                  f"random_words_pass={c2['n_random_words_pass']}/{c2['n_random_words']} all_pass={c2['all_pass']}")
            print(f"  check3 fertility(v2's 120-item methodology) before={c3['fertility_before_tokens_per_char']:.4f} "
                  f"after={c3['fertility_after_tokens_per_char']:.4f} inflation={c3['fertility_inflation_relative']:+.2%}")
            if "plain_prompt" in c4:
                print(f"  check4 plain_prompt={c4['plain_prompt']['n_full_in_keepset']}/{c4['plain_prompt']['n_items']} "
                      f"pdf_prompt={c4['pdf_prompt']['n_full_in_keepset']}/{c4['pdf_prompt']['n_items']}")
            print(f"  IS_CANDIDATE: {is_candidate}")
        all_results[label] = per_K

    RESULT_PATH.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"\nwrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
