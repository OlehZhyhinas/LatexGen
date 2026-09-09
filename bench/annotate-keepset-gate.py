#!/usr/bin/env python3
"""Post-process bench/keepsets/<label>-<K>/{coverage.json,provenance.md}:
apply this task's exactness/fertility gate across the already-rebuilt K
sweep for a tokenizer label, mark which K's clear it, and designate the
smallest clearing K as primary. Run once per tokenizer label after
rebuild-keepset-tokenizer.py has emitted every K in the sweep.

Gate (decided from the measured rebuild-verification numbers across the
mandated 16k/24k/32k/48k/64k sweep, not tuned to any target): exact-match
roundtrip rate over "all corpus texts, in-set only" >= 0.95, AND real
(re-encoded, leave-one-out) held-out fertility inflation on arxiv/pdf input
<= 1%. Both are rebuild-time facts, not the simulated coverage numbers from
vocab-coverage.py; below this, either a majority-scale share of nominally
in-set text silently retokenizes differently (bit-exactness gate), or the
prefill cost on real unseen input is no longer small (fertility gate).
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEEPSETS_ROOT = ROOT / "keepsets"

GATE_MIN_EXACT_RATE = 0.95
GATE_MAX_FERTILITY_INFLATION = 0.01


def sha256_file(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-label", required=True)
    args = ap.parse_args()

    dirs = sorted(
        (d for d in KEEPSETS_ROOT.glob(f"{args.tokenizer_label}-*") if (d / "coverage.json").exists()),
        key=lambda d: int(d.name.rsplit("-", 1)[1]),
    )
    if not dirs:
        raise SystemExit(f"no bench/keepsets/{args.tokenizer_label}-* dirs found")

    rows = []
    for d in dirs:
        K = int(d.name.rsplit("-", 1)[1])
        cov = json.loads((d / "coverage.json").read_text())
        v = cov["verify_all_corpus_texts"]
        rf = cov["real_held_out_fertility_arxiv_pdf"]
        exact_rate = (v["exact_match_passed"] / v["exact_match_total"]) if v["exact_match_total"] else 0.0
        fert_before = rf["fertility_before_tokens_per_char"]
        fert_after = rf["fertility_after_real_tokens_per_char"]
        fert_inflation = (fert_after / fert_before - 1) if fert_before else 0.0
        clears = exact_rate >= GATE_MIN_EXACT_RATE and fert_inflation <= GATE_MAX_FERTILITY_INFLATION
        rows.append(
            {
                "dir": d,
                "K": K,
                "actual_size": cov["actual_size"],
                "exact_rate": exact_rate,
                "fertility_inflation": fert_inflation,
                "clears_gate": clears,
            }
        )

    primary_K = next((r["K"] for r in rows if r["clears_gate"]), None)

    print(f"=== gate decision for {args.tokenizer_label} ===")
    print(f"gate: exact_rate >= {GATE_MIN_EXACT_RATE:.2f} AND fertility_inflation <= {GATE_MAX_FERTILITY_INFLATION:.2%}")
    for r in rows:
        tag = "PRIMARY" if r["K"] == primary_K else ("clears" if r["clears_gate"] else "does not clear")
        print(f"  K={r['K']:>6} actual={r['actual_size']:>6}  exact={r['exact_rate']:.4f}  fert_inflation={r['fertility_inflation']:+.4f}  [{tag}]")
    if primary_K is None:
        print("  WARNING: no K in this sweep clears the gate; no primary designated.")

    for r in rows:
        d = r["dir"]
        cov_path = d / "coverage.json"
        cov = json.loads(cov_path.read_text())
        cov["gate"] = {
            "min_exact_rate": GATE_MIN_EXACT_RATE,
            "max_fertility_inflation": GATE_MAX_FERTILITY_INFLATION,
            "exact_rate": r["exact_rate"],
            "fertility_inflation": r["fertility_inflation"],
            "clears_gate": r["clears_gate"],
            "is_primary": (r["K"] == primary_K),
            "primary_K_for_this_tokenizer": primary_K,
            "all_K_in_sweep": [x["K"] for x in rows],
        }
        cov_path.write_text(json.dumps(cov, indent=2) + "\n")

        prov_path = d / "provenance.md"
        text = prov_path.read_text()
        marker = "## Gate decision"
        if marker in text:
            text = text.split(marker)[0].rstrip() + "\n"
        status = "PRIMARY (smallest K in the sweep that clears the gate)" if r["K"] == primary_K else (
            "clears the gate but is not primary (a larger, more conservative alternate)" if r["clears_gate"]
            else "does NOT clear the gate -- emitted as a cost/quality comparison point only, not for deployment"
        )
        gate_lines = [
            "",
            marker,
            "",
            f"Gate: exact-match rebuild-verification rate >= {GATE_MIN_EXACT_RATE:.0%} AND real held-out "
            f"fertility inflation <= {GATE_MAX_FERTILITY_INFLATION:.0%}, evaluated across the full "
            f"{[x['K'] for x in rows]} sweep for this tokenizer.",
            "",
            f"This K ({r['K']} -> actual {r['actual_size']}): exact_rate={r['exact_rate']:.4f}, "
            f"fertility_inflation={r['fertility_inflation']:+.4f} -> **{status}**",
            "",
            "| K | actual | exact_rate | fertility_inflation | verdict |",
            "|---:|---:|---:|---:|---|",
        ]
        for x in rows:
            v = "PRIMARY" if x["K"] == primary_K else ("clears" if x["clears_gate"] else "does not clear")
            gate_lines.append(f"| {x['K']} | {x['actual_size']} | {x['exact_rate']:.4f} | {x['fertility_inflation']:+.4f} | {v} |")
        gate_lines.append("")
        text = text.rstrip() + "\n" + "\n".join(gate_lines)
        prov_path.write_text(text)

        # coverage.json and provenance.md changed; re-hash and rewrite SHA256SUMS.txt
        files = [p.name for p in d.iterdir() if p.name != "SHA256SUMS.txt"]
        hashes = {f: sha256_file(d / f) for f in sorted(files)}
        (d / "SHA256SUMS.txt").write_text("\n".join(f"{h}  {f}" for f, h in hashes.items()) + "\n")

    print(f"\nprimary K for {args.tokenizer_label}: {primary_K}")


if __name__ == "__main__":
    main()
