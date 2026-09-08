#!/usr/bin/env python3
import json, re
from pathlib import Path

import pdf_pipeline as p

BRACKET_RE = re.compile(r"\[[^\[\]\n]+\]")
LEGACY_CITE_RE = re.compile(r"\[(?:\d+\s*(?:,\s*\d+\s*)*)\]")


def cset(text):
    return {re.sub(r"\s+", "", g) for g in BRACKET_RE.findall(text or "")}


def count_losses(rows, legacy=False):
    n = 0
    for row in rows:
        ref, inp = row.get("reference", ""), row.get("input", "")
        clean, _ = p.normalize(inp)
        if legacy: clean = LEGACY_CITE_RE.sub("", clean)
        want, before, after = cset(ref), cset(inp), cset(clean)
        if want and any(g in before and g not in after for g in want): n += 1
    return n


if __name__ == "__main__":
    rows = json.loads(Path("bench/arxiv-pastes.json").read_text())
    print(f"before={count_losses(rows, legacy=True)}")
    print(f"after={count_losses(rows, legacy=False)}")
