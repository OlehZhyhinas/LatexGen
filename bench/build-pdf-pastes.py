#!/usr/bin/env python3
"""Build the pdf-paste benchmark tier."""
import argparse
import importlib.util
import json
from pathlib import Path

import pdf_paste

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "pdf-pastes.json"
BENCH_DATA_PATH = ROOT / "bench-data.json"


def load_formula_table():
    path = ROOT / "synth-spans.py"
    spec = importlib.util.spec_from_file_location("synth_spans_table", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    table = {}
    for row in mod.FORMULAS:
        if len(row) == 5:
            fid, latex, _spoken, _unicode, source = row
        else:
            fid, latex, _unicode, source = row
        table[fid] = {"latex": latex, "source": source}
    return table


def load_multiline_references():
    items = json.loads(BENCH_DATA_PATH.read_text())
    out = []
    for item in items:
        if item.get("tier") == "multiline":
            out.append(
                {
                    "id": f"{item['id']}-pdf",
                    "reference": item["reference"],
                    "source": item["source"],
                }
            )
    return out


def make_new_references(f):
    rows = []
    rows.append(
        {
            "id": "probability-mixture-pdf",
            "reference": (
                f"In the 2024 cohort, we estimated posterior odds with \\( {f['bayes-theorem']['latex']} \\). "
                "The same report also tracks calibration drift over monthly windows. "
                f"For normalization we keep \\[ {f['normal-distribution-pdf']['latex']} \\] as the baseline density."
            ),
            "formula_ids": ["bayes-theorem", "normal-distribution-pdf"],
        }
    )
    rows.append(
        {
            "id": "series-and-limit-pdf",
            "reference": (
                "Two identities appear in the margin notes and are reused in later proofs. "
                f"First, \\( {f['geometric-series-sum']['latex']} \\). "
                f"Second, \\[ {f['limit-definition-of-e']['latex']} \\] when the compounding step is refined."
            ),
            "formula_ids": ["geometric-series-sum", "limit-definition-of-e"],
        }
    )
    rows.append(
        {
            "id": "determinant-appendix-pdf",
            "reference": (
                "Appendix B computes a local Jacobian before the nonlinear update. "
                f"The determinant identity is written as \\[ {f['determinant-2x2']['latex']} \\]. "
                "This section then compares signs across neighboring cells."
            ),
            "formula_ids": ["determinant-2x2"],
        }
    )
    rows.append(
        {
            "id": "maxwell-note-pdf",
            "reference": (
                "A short electrodynamics note summarizes the field equations in one block. "
                f"We copy the pair directly as \\[ {f['maxwell-aligned-pair']['latex']} \\]. "
                "The follow-up sentence comments on boundary conditions in vacuum."
            ),
            "formula_ids": ["maxwell-aligned-pair"],
        }
    )
    rows.append(
        {
            "id": "kinematics-lab-pdf",
            "reference": (
                "On 2026-03-18, the lab sheet used 10 milliliters of dye and a 2 hour run. "
                f"The velocity estimate was \\( {f['derivative-definition']['latex']} \\), then energy was computed with \\( {f['kinetic-energy']['latex']} \\). "
                "The prose numbers are part of setup notes, not symbolic inputs."
            ),
            "formula_ids": ["derivative-definition", "kinetic-energy"],
        }
    )
    rows.append(
        {
            "id": "waves-lecture-pdf",
            "reference": (
                "The lecture transitions from local oscillation to spectral analysis. "
                f"It first states \\( {f['de-moivres-formula']['latex']} \\). "
                f"The transform convention on the next line is \\[ {f['fourier-transform']['latex']} \\]."
            ),
            "formula_ids": ["de-moivres-formula", "fourier-transform"],
        }
    )
    rows.append(
        {
            "id": "calculus-derivatives-pdf",
            "reference": (
                f"For the chain of substitutions we repeatedly invoke \\( {f['chain-rule']['latex']} \\) "
                f"and \\( {f['product-rule']['latex']} \\). "
                f"At the endpoint check, the slope estimate is \\( {f['mean-value-theorem']['latex']} \\)."
            ),
            "formula_ids": ["chain-rule", "product-rule", "mean-value-theorem"],
        }
    )
    rows.append(
        {
            "id": "physics-constants-pdf",
            "reference": (
                f"The constants table starts from \\( {f['mass-energy-equivalence']['latex']} \\) and compares it with \\( {f['planck-einstein-relation']['latex']} \\). "
                f"A later paragraph rewrites field strength as \\( {f['coulombs-law']['latex']} \\)."
            ),
            "formula_ids": [
                "mass-energy-equivalence",
                "planck-einstein-relation",
                "coulombs-law",
            ],
        }
    )
    rows.append(
        {
            "id": "thermo-note-pdf",
            "reference": (
                "A compact thermodynamics summary keeps the ideal gas relation explicit: "
                f"\\( {f['ideal-gas-law']['latex']} \\). "
                f"It then states entropy as \\( {f['entropy-boltzmann']['latex']} \\) before discussing units."
            ),
            "formula_ids": ["ideal-gas-law", "entropy-boltzmann"],
        }
    )
    rows.append(
        {
            "id": "triangle-geometry-pdf",
            "reference": (
                "The geometry section alternates between metric and angle viewpoints. "
                f"The side relation is \\( {f['law-of-cosines']['latex']} \\), while the norm bound is \\( {f['triangle-inequality']['latex']} \\). "
                "These are used to bound rounding error in a later estimate."
            ),
            "formula_ids": ["law-of-cosines", "triangle-inequality"],
        }
    )
    rows.append(
        {
            "id": "probability-gradient-pdf",
            "reference": (
                f"To compare two classifiers, the notes write the logistic link \\( {f['logistic-function']['latex']} \\). "
                f"An uncertainty correction adds \\( {f['gaussian-integral']['latex']} \\) in the derivation."
            ),
            "formula_ids": ["logistic-function", "gaussian-integral"],
        }
    )
    rows.append(
        {
            "id": "financial-model-pdf",
            "reference": (
                "In the December 2025 worksheet, a nominal rate of 7 percent is compounded monthly. "
                f"The growth rule is \\( {f['compound-interest']['latex']} \\), and the reserve estimate includes \\( {f['newtons-law-of-gravitation']['latex']} \\) as a toy stress term. "
                "The date and percentage belong to prose metadata."
            ),
            "formula_ids": ["compound-interest", "newtons-law-of-gravitation"],
        }
    )
    return rows


def with_sources(rows, formula_table):
    out = []
    for row in rows:
        sources = sorted({formula_table[fid]["source"] for fid in row["formula_ids"]})
        out.append(
            {
                "id": row["id"],
                "reference": row["reference"],
                "source": ", ".join(sources),
            }
        )
    return out


def build_items(records, layout=False, seed=0):
    passages = [r["reference"] for r in records]
    clean, clean_stats = pdf_paste.pdf_paste(passages, profile="clean", layout=layout, seed=seed)
    dirty, dirty_stats = pdf_paste.pdf_paste(passages, profile="dirty", layout=layout, seed=seed)

    items = []
    extracted_rows = []
    for rec, ex in zip(records, clean):
        ex_id = f"{rec['id']}-clean"
        item = {
            "id": ex_id,
            "tier": "pdf-paste",
            "profile": "clean",
            "knobs": [],
            "input": ex["text"],
            "reference": rec["reference"],
            "spans": ex["spans"],
            "source": rec["source"],
            "ok": ex["ok"],
            "reason": ex["reason"],
        }
        items.append(item)
        extracted_rows.append(item)

    for rec, ex in zip(records, dirty):
        ex_id = f"{rec['id']}-dirty"
        item = {
            "id": ex_id,
            "tier": "pdf-paste",
            "profile": "dirty",
            "knobs": ex["knobs"],
            "input": ex["text"],
            "reference": rec["reference"],
            "spans": ex["spans"],
            "source": rec["source"],
            "ok": ex["ok"],
            "reason": ex["reason"],
        }
        items.append(item)
        extracted_rows.append(item)
    return items, extracted_rows, clean_stats, dirty_stats


def print_table(items):
    print("| id | profile | knobs | chars | spans | ok |")
    print("|---|---|---|---:|---:|---:|")
    for item in items:
        knob_txt = ",".join(item["knobs"]) if item["knobs"] else "-"
        print(
            f"| {item['id']} | {item['profile']} | {knob_txt} | {len(item['input'])} | "
            f"{len(item['spans'])} | {'yes' if item['ok'] else 'no'} |"
        )


def print_dirty_drop_rates(dirty_stats):
    print("\nDirty sentinel mismatch drop rate by knob")
    print("| knob | total | failed | drop_rate | ext_spans | ext_rate | part_spans | part_rate |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for knob, row in dirty_stats["drop_rate_by_knob"].items():
        print(
            f"| {knob} | {row['total']} | {row['failed']} | {row['drop_rate']:.3f} | "
            f"{row.get('extended_spans', 0)} | {row.get('extended_rate', 0.0):.3f} | "
            f"{row.get('partial_spans', 0)} | {row.get('partial_rate', 0.0):.3f} |"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    formulas = load_formula_table()
    base = load_multiline_references()
    new_rows = with_sources(make_new_references(formulas), formulas)
    records = base + new_rows

    items, extracted, clean_stats, dirty_stats = build_items(records, layout=args.layout, seed=args.seed)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {out_path} ({len(items)} items)")
    print(
        f"clean: failed={clean_stats['failed']}/{clean_stats['total']} "
        f"dirty: failed={dirty_stats['failed']}/{dirty_stats['total']}"
    )
    clean_comp = clean_stats.get("span_completeness", {})
    dirty_comp = dirty_stats.get("span_completeness", {})
    print(
        "span completeness clean: "
        f"extended={clean_comp.get('extended_spans', 0)}/{clean_comp.get('total_spans', 0)} "
        f"({clean_comp.get('extended_rate', 0.0):.3f}) "
        f"partial={clean_comp.get('partial_spans', 0)}/{clean_comp.get('total_spans', 0)} "
        f"({clean_comp.get('partial_rate', 0.0):.3f})"
    )
    print(
        "span completeness dirty: "
        f"extended={dirty_comp.get('extended_spans', 0)}/{dirty_comp.get('total_spans', 0)} "
        f"({dirty_comp.get('extended_rate', 0.0):.3f}) "
        f"partial={dirty_comp.get('partial_spans', 0)}/{dirty_comp.get('total_spans', 0)} "
        f"({dirty_comp.get('partial_rate', 0.0):.3f})"
    )
    print_table(extracted)
    print_dirty_drop_rates(dirty_stats)


if __name__ == "__main__":
    main()
