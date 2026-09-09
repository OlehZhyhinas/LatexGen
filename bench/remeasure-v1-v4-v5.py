#!/usr/bin/env python3
"""Re-measure v1/v2/v3/v4's emitted keep-set directories under v5's
methodology: real re-encoding with the ORIGINAL (unpruned) tokenizer against
the SAME held-out text v5 uses everywhere else (eval240 = the 120
arxiv-pastes.json input+reference strings; held_out_full = eval240 +
bench-data-extended.json's `ok` items). Read-only against the sibling
worktrees (never writes there); v4's directories may be incomplete (a
sibling `keepset-v4` task was still running when this was captured -- see
bench/README-keepset-v5.md for which directories were available and when).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARXIV_PASTES_PATH = ROOT / "arxiv-pastes.json"
BENCH_EXTENDED_PATH = ROOT / "bench-data-extended.json"

TOKENIZER_PATHS = {
    "qwen35": "/Users/oleh/.cache/huggingface/hub/models--mlc-ai--Qwen3.5-0.8B-q4f16_1-MLC/snapshots/0ec138972555613c1d7812a821778ad0398c8790/tokenizer.json",
    "minicpm5-2b": "/Users/oleh/.cache/huggingface/hub/models--ozhyhinas--MiniCPM5-2B-q4f16_1-MLC/snapshots/2318f37d9c39277ff01dc64086491028c95d4db4/tokenizer.json",
}

DIRECTORIES = {
    "v1": ROOT / "keepsets",
    "v2": Path("/Users/oleh/personal/latexgen-keepset-v2/bench/keepsets-v2"),
    "v3": Path("/Users/oleh/personal/latexgen-keepset-v3/bench/keepsets-v3"),
    "v4": Path("/Users/oleh/personal/latexgen-keepset-v4/bench/keepsets-v4"),
}

# What each generation ITSELF reported, for the "originally reported" column
# (transcribed from each generation's own README/brief; see
# bench/README-keepset-v5.md for exact citations).
ORIGINALLY_REPORTED = {
    ("v1", "qwen35", 16384): "fertility_after 0.2575 vs before 0.2467 (n=138, input-only, leave-one-out)",
    ("v1", "qwen35", 24576): "(same 138-item input-only LOO set)",
    ("v1", "qwen35", 32768): "(same 138-item input-only LOO set)",
    ("v1", "qwen35", 49152): "(same 138-item input-only LOO set)",
    ("v1", "qwen35", 65536): "+0.52% (own 138-item LOO set, brief-quoted headline)",
    ("v1", "minicpm5-2b", 16384): "(own 138-item input-only LOO set)",
    ("v1", "minicpm5-2b", 24576): "(own 138-item input-only LOO set)",
    ("v1", "minicpm5-2b", 32768): "(own 138-item input-only LOO set)",
    ("v1", "minicpm5-2b", 49152): "(own 138-item input-only LOO set)",
    ("v1", "minicpm5-2b", 65536): "(own 138-item input-only LOO set)",
    ("v2", "qwen35", 32768): "+19.53% (OOS-subset only, n=240 out of 120 arxiv items, exact in-set matches excluded)",
    ("v2", "qwen35", 65536): "+8.99% (OOS-subset only, n=216)",
    ("v2", "minicpm5-2b", 32768): "+14.60% (OOS-subset only, n=237)",
    ("v2", "minicpm5-2b", 65536): "+9.79% (OOS-subset only, n=172)",
    ("v3", "qwen35", 24576): "+0.00% (v2's 120-item in-set methodology; A-fixed base includes arxiv-pastes.json+bench-data-extended.json as 2 of its 5 corpus files -- in-sample by construction)",
    ("v3", "qwen35", 32768): "+0.00% (same in-sample cause)",
    ("v3", "qwen35", 49152): "+0.00% (same in-sample cause)",
    ("v3", "minicpm5-2b", 24576): "+0.00% (same in-sample cause)",
    ("v3", "minicpm5-2b", 32768): "+0.00% (same in-sample cause)",
    ("v3", "minicpm5-2b", 49152): "+0.00% (same in-sample cause)",
}


def load_texts():
    eval240 = []
    for it in json.load(open(ARXIV_PASTES_PATH)):
        if not it.get("ok", True):
            continue
        eval240.append(it["input"])
        eval240.append(it["reference"])
    extra = []
    for it in json.load(open(BENCH_EXTENDED_PATH)):
        if not it.get("ok", True):
            continue
        extra.append(it.get("input", ""))
        extra.append(it.get("reference", ""))
    return eval240, eval240 + extra


def measure(old_tok, new_tok, keep_ids, texts):
    total_chars = 0
    total_before = 0
    total_after = 0
    total_oos = 0
    for t in texts:
        if not t:
            continue
        old_ids = old_tok.encode(t, add_special_tokens=False).ids
        new_ids = new_tok.encode(t, add_special_tokens=False).ids
        total_chars += len(t)
        total_before += len(old_ids)
        total_after += len(new_ids)
        total_oos += sum(1 for i in old_ids if i not in keep_ids)
    fert_before = (total_before / total_chars) if total_chars else 0.0
    fert_after = (total_after / total_chars) if total_chars else 0.0
    return {
        "n_texts": len([t for t in texts if t]),
        "fertility_before": fert_before,
        "fertility_after": fert_after,
        "inflation": (fert_after / fert_before - 1.0) if fert_before else 0.0,
        "oos_rate": (total_oos / total_before) if total_before else 0.0,
    }


def main():
    from tokenizers import Tokenizer

    eval240, held_out_full = load_texts()
    old_toks = {label: Tokenizer.from_file(path) for label, path in TOKENIZER_PATHS.items()}

    results = {}
    for gen, root in DIRECTORIES.items():
        if not root.exists():
            print(f"{gen}: {root} does not exist, skipping")
            continue
        gen_results = {}
        for d in sorted(root.glob("*-*")):
            name = d.name
            # split "qwen35-32768" / "minicpm5-2b-32768"
            try:
                label, k_str = name.rsplit("-", 1)
                K = int(k_str)
            except ValueError:
                continue
            if label not in TOKENIZER_PATHS:
                continue
            tok_path = d / "tokenizer.json"
            keep_path = d / "keep-idx.json"
            if not tok_path.exists() or not keep_path.exists():
                print(f"  {gen}/{name}: missing tokenizer.json or keep-idx.json, skipping (possibly still being written)")
                continue
            try:
                new_tok = Tokenizer.from_file(str(tok_path))
                keep_ids = set(json.load(open(keep_path)))
            except Exception as e:
                print(f"  {gen}/{name}: failed to load ({e}), skipping")
                continue
            old_tok = old_toks[label]
            m240 = measure(old_tok, new_tok, keep_ids, eval240)
            mfull = measure(old_tok, new_tok, keep_ids, held_out_full)
            reported = ORIGINALLY_REPORTED.get((gen, label, K), "not reported at this K / not found in that generation's write-up")
            row = {
                "directory": str(d),
                "tokenizer_label": label,
                "K": K,
                "actual_size": len(keep_ids),
                "eval240": m240,
                "held_out_full": mfull,
                "originally_reported": reported,
            }
            gen_results[name] = row
            print(f"{gen}/{name}: actual_size={len(keep_ids)} eval240_inflation={m240['inflation']:+.4%} "
                  f"eval240_oos={m240['oos_rate']:.4%} full_inflation={mfull['inflation']:+.4%}")
        results[gen] = gen_results

    out_path = ROOT / "results-keepset-v5-remeasure-v1-v4.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
