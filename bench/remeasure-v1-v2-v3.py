#!/usr/bin/env python3
"""Job 1's other half: re-measure v1's (bench/keepsets/) and v2's
(bench/keepsets-v2/) already-emitted tokenizer.json directories under the
SAME one methodology as the v3 sweep (bench/build-keepset-v3.py) -- the 120
arxiv-pastes.json input+reference strings plus bench-data-extended.json's
ok items, real re-encode, tokens-per-character -- so the historical numbers
become comparable to v3's, and any difference from what v1/v2 originally
reported is stated plainly (their own methodology used a narrower held-out
set: v1 measured leave-one-out simulated fertility over a different corpus
union; v2's coverage.json measured only the 120 arxiv-pastes.json items,
not bench-data-extended.json's 93 additional items).
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEEPSETS_V1_ROOT = ROOT / "keepsets"
KEEPSETS_V2_ROOT = ROOT / "keepsets-v2"
OUT_PATH = ROOT / "results-keepset-v3-remeasure-legacy.json"


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v3 = _load_module("build_keepset_v3_dep", "build-keepset-v3.py")


def main():
    from tokenizers import Tokenizer

    texts, held_out_meta = v3.held_out_texts()
    print(f"held-out text: {held_out_meta}, {len([t for t in texts if t])} non-empty strings")

    results = {"held_out_meta": held_out_meta, "v1": {}, "v2": {}}

    for label, tok_path in v3.TOKENIZER_PATHS.items():
        old_tok, _ = v3.vc.load_tokenizer(tok_path)

        for root_name, root, out_key in [("v1", KEEPSETS_V1_ROOT, "v1"), ("v2", KEEPSETS_V2_ROOT, "v2")]:
            for d in sorted(root.glob(f"{label}-*")):
                tok_json = d / "tokenizer.json"
                keep_idx_path = d / "keep-idx.json"
                if not tok_json.exists():
                    continue
                new_tok = Tokenizer.from_file(str(tok_json))
                fert = v3.measure_fertility(old_tok, new_tok, texts)
                actual_K = len(json.load(open(keep_idx_path))) if keep_idx_path.exists() else new_tok.get_vocab_size(with_added_tokens=True)

                orig_report = None
                cov_path = d / "coverage.json"
                if cov_path.exists():
                    cov = json.load(open(cov_path))
                    if root_name == "v1":
                        rf = cov.get("real_held_out_fertility_arxiv_pdf", {})
                        orig_report = {
                            "methodology": "v1 own: leave-one-out real re-encode, arxiv-paste+pdf-paste INPUT text only (not references), n=" + str(rf.get("n_items")),
                            "fertility_before": rf.get("fertility_before_tokens_per_char"),
                            "fertility_after_real": rf.get("fertility_after_real_tokens_per_char"),
                        }
                    else:
                        vv = cov.get("verify_all_corpus_texts", {})
                        orig_report = {
                            "methodology": "v2 own: coverage.json's verify_all_corpus_texts OOS-subset real re-encode, arxiv-pastes.json input+reference only (n=120 items -> " + str(vv.get("oos_texts_n")) + " OOS-subset texts, in-set exact-match texts excluded from this fertility figure)",
                            "fertility_before": vv.get("oos_texts_fertility_before"),
                            "fertility_after_real": vv.get("oos_texts_fertility_after_real"),
                        }

                results[out_key][d.name] = {
                    "dir": str(d.relative_to(ROOT)),
                    "actual_K": actual_K,
                    "remeasured_under_v3_methodology": fert,
                    "originally_reported": orig_report,
                }
                print(
                    f"{root_name} {d.name}: v3-methodology fertility before={fert['fertility_before_tokens_per_char']:.4f} "
                    f"after={fert['fertility_after_tokens_per_char']:.4f} inflation={fert['fertility_inflation_relative']:+.4%} "
                    f"(n={fert['n_texts']} texts) | originally reported: {orig_report}"
                )

    OUT_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
