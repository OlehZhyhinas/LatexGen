#!/usr/bin/env python3
"""Keep-set v4 -- net speed table.

    ms_per_token(K) = body + head_orig * (K / V_orig)
    net = ms_total_baseline / (ms_per_token(K) * (1 + fertility_inflation(K)))

Head-kernel costs, verified against `webnn-workbench` `origin/main` docs via
`git show` (not assumed from the v3/v4 brief's own figures -- see
bench/README-keepset-v4.md's "Head-cost verification" section for the exact
source lines):

- Qwen3.5-0.8B: head 0.648-0.658 ms/token (~11.6% of GPU decode time,
  `docs/qwen35-08-vocab-report.md` Stage 1), paired against the model's own
  tuned headline total 5.24 ms/token (190.83 tok/s,
  `docs/qwen35-08-m5-webgpu-report.md`). Brief's "~0.648 of 5.24": confirmed.
- Qwen3.5-4B: head 1.372 ms/token (`notes/stage1-profile.md`, cross-checked
  against `docs/qwen35-4b-vocab-report.md` line 145), total 15.61 ms/token
  (64.06 tok/s headline, `docs/qwen35-4b-m5-webgpu-report.md`). Brief's
  "1.372 of 15.61": confirmed exactly.
- Qwen3.5-9B: head share 4.03%-4.6% of decode GPU time
  (`docs/qwen35-9b-vocab-report.md` Stage 1; carried through as a range per
  the brief's own instruction, not collapsed to a point), total 26.70
  ms/token (37.45 tok/s tuned headline, `docs/qwen35-9b-m5-webgpu-report.md`).
  Head absolute = share * total = 1.068-1.228 ms/token. Brief's "~1.10 of
  26.70": within this range, confirmed as a reasonable point pick, but this
  script carries the full range through instead.
- MiniCPM5-2B: head 0.6657 ms/token (`docs/minicpm5-2b-vocab-report.md`
  Stage 1), total 9.74 ms/token (tuned median,
  `docs/minicpm5-2b-m5-webgpu-report.md`). Brief's "0.666 of 9.74": confirmed.

Hidden sizes and tied/untied status, verified the same way:
- Qwen3.5-0.8B: hidden=1024, tied (`docs/qwen35-08-vocab-report.md`).
- Qwen3.5-4B: hidden=2560 (derived from `q_weight` shape (K,320) uint32 =
  320*4 bytes/row = 1280 bytes -> hidden/8=320 -> hidden=2560,
  `docs/qwen35-4b-vocab-report.md` line 91), tied.
- Qwen3.5-9B: hidden=4096 (derived from `q_weight`+`q_scale` = 2048+256
  bytes/row per matrix -> hidden/8=2048/4=512? see note below), UNTIED
  (`docs/qwen35-9b-vocab-report.md` line 225-227: 4608 bytes/row "across
  both matrices" -- the only one of the four Qwen3.5/MiniCPM5 rungs in this
  project that does not tie embeddings).
- MiniCPM5-2B: hidden=2048, untied (`docs/minicpm5-2b-vocab-report.md`).

Row bytes at q4f16_1 (4-bit weights, group size 32, fp16 scales):
    bytes_per_row = hidden * 0.5 (q_weight, 8 nibbles/uint32) +
                    hidden / 32 * 2 (q_scale, fp16) = hidden * 0.5625
Verified against MiniCPM5-2B's own reported shapes (256*4 + 64*2 = 1152 =
2048*0.5625) and Qwen3.5-4B's (320*4 + 80*2 = 1440 = 2560*0.5625).
MB_saved(K) = (V_orig - K) * bytes_per_row * (2 if untied else 1) / 2**20
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEEPSETS_ROOT = ROOT / "keepsets"

TARGET_SIZES = [24576, 32768, 49152]
V1_K = 65536

MODELS = {
    "Qwen3.5-0.8B": {
        "tokenizer_label": "qwen35",
        "V": 248320,
        "hidden": 1024,
        "tied": True,
        "head_ms": (0.648, 0.658),
        "total_ms": 5.24,
    },
    "Qwen3.5-4B": {
        "tokenizer_label": "qwen35",
        "V": 248320,
        "hidden": 2560,
        "tied": True,
        "head_ms": (1.372, 1.372),
        "total_ms": 15.61,
    },
    "Qwen3.5-9B": {
        "tokenizer_label": "qwen35",
        "V": 248320,
        "hidden": 4096,
        "tied": False,
        "head_ms": (26.70 * 0.040, 26.70 * 0.046),
        "total_ms": 26.70,
    },
    "MiniCPM5-2B": {
        "tokenizer_label": "minicpm5-2b",
        "V": 130560,
        "hidden": 2048,
        "tied": False,
        "head_ms": (0.6657, 0.6657),
        "total_ms": 9.74,
    },
}

TOKENIZER_PATHS = {
    "qwen35": "/Users/oleh/.cache/huggingface/hub/models--mlc-ai--Qwen3.5-0.8B-q4f16_1-MLC/snapshots/0ec138972555613c1d7812a821778ad0398c8790/tokenizer.json",
    "minicpm5-2b": "/Users/oleh/.cache/huggingface/hub/models--ozhyhinas--MiniCPM5-2B-q4f16_1-MLC/snapshots/2318f37d9c39277ff01dc64086491028c95d4db4/tokenizer.json",
}


def _load_module(name, filename):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vc = _load_module("vocab_coverage_speed_v4", "vocab-coverage.py")
b4 = _load_module("build_keepset_v4_speed_dep", "build-keepset-v4.py")


def bytes_per_row(hidden):
    return hidden * 0.5625


def mb_saved(V, K, hidden, tied):
    rows_dropped = V - K
    factor = 1 if tied else 2
    return rows_dropped * bytes_per_row(hidden) * factor / (2 ** 20)


def ms_per_token(body, head_orig, K, V):
    return body + head_orig * (K / V)


def net_speed(ms_total_baseline, ms_per_tok_pruned, fertility_inflation):
    return ms_total_baseline / (ms_per_tok_pruned * (1 + fertility_inflation))


def main():
    held_out_texts, n_arxiv, n_bench_ext = b4.load_held_out_texts()

    fertility_before = {}
    fertility_v1_65536 = {}
    fertility_A = {}
    fertility_v4 = {}

    for label, tok_path in TOKENIZER_PATHS.items():
        tok, _ = vc.load_tokenizer(tok_path)
        fert0, _, _ = b4.fertility_over_texts(tok, held_out_texts)
        fertility_before[label] = fert0

        v1_dir = KEEPSETS_ROOT / f"{label}-{V1_K}"
        if (v1_dir / "tokenizer.json").exists():
            from tokenizers import Tokenizer
            v1_tok = Tokenizer.from_file(str(v1_dir / "tokenizer.json"))
            fert1, _, _ = b4.fertility_over_texts(v1_tok, held_out_texts)
            fertility_v1_65536[label] = fert1
        else:
            fertility_v1_65536[label] = None

        results = json.load(open(ROOT / f"results-keepset-v4-{label}.json"))
        fertility_A[label] = {}
        fertility_v4[label] = {}
        for K in TARGET_SIZES:
            row = results["by_target_size"][str(K)]
            fertility_A[label][K] = row["strategy_a_fertility"]
            fertility_v4[label][K] = row["v4_fertility"]

    print("Fertility before (unpruned tokenizer, held-out text, real re-encoding):")
    for label, f in fertility_before.items():
        print(f"  {label}: {f:.4f}")
    print("v1 shipped K=65536 fertility (same held-out text):")
    for label, f in fertility_v1_65536.items():
        print(f"  {label}: {f}")

    all_rows = []
    for model_name, cfg in MODELS.items():
        label = cfg["tokenizer_label"]
        V = cfg["V"]
        hidden = cfg["hidden"]
        tied = cfg["tied"]
        head_lo, head_hi = cfg["head_ms"]
        total = cfg["total_ms"]
        body_lo = total - head_hi
        body_hi = total - head_lo
        fert0 = fertility_before[label]

        print(f"\n=== {model_name} (tokenizer={label}, V={V}, hidden={hidden}, tied={tied}) ===")
        print(f"  head={head_lo:.3f}-{head_hi:.3f} ms/token, total={total:.2f} ms/token, body={body_lo:.3f}-{body_hi:.3f} ms/token")

        # v1 @ 65536
        if fertility_v1_65536[label] is not None:
            infl_v1 = (fertility_v1_65536[label] / fert0) - 1.0
            mpt_lo = ms_per_token(body_lo, head_lo, V1_K, V)
            mpt_hi = ms_per_token(body_hi, head_hi, V1_K, V)
            net_lo = net_speed(total, mpt_hi, infl_v1)
            net_hi = net_speed(total, mpt_lo, infl_v1)
            mb_v1 = mb_saved(V, V1_K, hidden, tied)
            print(f"  v1@65536: fertility_inflation={infl_v1:+.2%}, ms_per_token={mpt_lo:.3f}-{mpt_hi:.3f}, "
                  f"net={net_lo:.3f}-{net_hi:.3f}x, MB_saved={mb_v1:.1f}")
            all_rows.append({"model": model_name, "variant": "v1", "K": V1_K,
                              "fertility_inflation": infl_v1, "net_lo": net_lo, "net_hi": net_hi,
                              "mb_saved": mb_v1})

        for K in TARGET_SIZES:
            for variant, fert_table in (("A", fertility_A), ("v4", fertility_v4)):
                fert = fert_table[label][K]
                infl = (fert / fert0) - 1.0
                mpt_lo = ms_per_token(body_lo, head_lo, K, V)
                mpt_hi = ms_per_token(body_hi, head_hi, K, V)
                net_lo = net_speed(total, mpt_hi, infl)
                net_hi = net_speed(total, mpt_lo, infl)
                mb = mb_saved(V, K, hidden, tied)
                print(f"  {variant}@{K}: fertility={fert:.4f} inflation={infl:+.2%} "
                      f"ms_per_token={mpt_lo:.3f}-{mpt_hi:.3f} net={net_lo:.3f}-{net_hi:.3f}x MB_saved={mb:.1f}")
                all_rows.append({"model": model_name, "variant": variant, "K": K,
                                  "fertility": fert, "fertility_inflation": infl,
                                  "net_lo": net_lo, "net_hi": net_hi, "mb_saved": mb})

    # ---- Leakage-free (disjoint-corpus) net speed at K=32,768, mirroring
    # keepset-v3's "Corrected optimum" section -- strategy A's default corpus
    # includes 2 of this task's held-out files, so the in-sample numbers
    # above are optimistic; this uses the disjoint-corpus inflation numbers
    # bench/build-keepset-v4.py already measured and stored. ----
    print("\n=== Leakage-free (disjoint-corpus) net speed at K=32,768 ===")
    disjoint_rows = []
    for model_name, cfg in MODELS.items():
        label = cfg["tokenizer_label"]
        results = json.load(open(ROOT / f"results-keepset-v4-{label}.json"))
        leak = results.get("leakage_sanity_check_strategy_A_32768")
        if not leak:
            continue
        V, hidden, tied = cfg["V"], cfg["hidden"], cfg["tied"]
        head_lo, head_hi = cfg["head_ms"]
        total = cfg["total_ms"]
        body_lo, body_hi = total - head_hi, total - head_lo
        mpt_lo = ms_per_token(body_lo, head_lo, 32768, V)
        mpt_hi = ms_per_token(body_hi, head_hi, 32768, V)
        mb = mb_saved(V, 32768, hidden, tied)
        for variant, infl_key in (("A-disjoint", "inflation_A_disjoint_32768"), ("v4-disjoint", "inflation_v4_disjoint_32768")):
            infl = leak[infl_key]
            net_lo = net_speed(total, mpt_hi, infl)
            net_hi = net_speed(total, mpt_lo, infl)
            print(f"  {model_name} {variant}@32768: inflation={infl:+.2%} net={net_lo:.3f}-{net_hi:.3f}x MB_saved={mb:.1f}")
            disjoint_rows.append({"model": model_name, "variant": variant, "K": 32768,
                                   "fertility_inflation": infl, "net_lo": net_lo, "net_hi": net_hi, "mb_saved": mb})

    out = {
        "fertility_before": fertility_before,
        "fertility_v1_65536": fertility_v1_65536,
        "held_out": {"n_arxiv": n_arxiv, "n_bench_ext": n_bench_ext, "n_texts": len(held_out_texts)},
        "models": MODELS,
        "rows": all_rows,
        "disjoint_rows_32768": disjoint_rows,
    }
    out_path = ROOT / "results-keepset-v4-speed.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
