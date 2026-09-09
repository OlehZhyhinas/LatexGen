#!/usr/bin/env python3
"""Build bench/keepset-seed-mathsenglish.json: an English-word-type seed list
derived from the maths-adjacent corpora already in this repo (not invented,
not a general English wordlist -- see the "small corpus" caveat in the
provenance block and in the README/PR).

Sources (document = one corpus item; a word type is counted once per
document it appears in, exact surface string, case preserved):

- bench/arxiv-pastes.json (120 items): `reference` (prose with LaTeX spans
  and `(ref)` citation placeholders) and `meta.reference_prose` (the same
  paragraph with math spans stripped to pure prose) -- both fields named
  explicitly in the task brief.
- bench/bench-data-extended.json (105 items): `input` (the natural-language
  description of the formula -- spoken form for the original 15, PDF-style
  extracted prose for the pdf-paste/synth-pdf/synth-unicode tiers it adds)
  and `reference` (mostly LaTeX, included for completeness though it
  contributes few English words).
- bench/pdf-pastes.json (30 items): `input` and `reference`, same reasoning.

Word types are extracted with a plain ASCII-letter regex (optionally with an
internal apostrophe, e.g. "Euler's"); no stemming, no casing normalization,
no filtering beyond "appeared in this text" -- the point is to recover
exactly the vocabulary already present in the benchmark's own maths-English
text, not to synthesize new coverage.
"""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "keepset-seed-mathsenglish.json"

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def words_in(text):
    return set(WORD_RE.findall(text or ""))


def main():
    doc_sources = []  # (corpus_label, doc_id, set_of_words)

    arxiv = json.load(open(ROOT / "arxiv-pastes.json"))
    for it in arxiv:
        ws = words_in(it.get("reference", "")) | words_in(it.get("meta", {}).get("reference_prose", ""))
        doc_sources.append(("arxiv-pastes.reference+meta.reference_prose", it["id"], ws))

    bde = json.load(open(ROOT / "bench-data-extended.json"))
    for it in bde:
        ws = words_in(it.get("input", "")) | words_in(it.get("reference", ""))
        doc_sources.append(("bench-data-extended.input+reference", it["id"], ws))

    pdf = json.load(open(ROOT / "pdf-pastes.json"))
    for it in pdf:
        ws = words_in(it.get("input", "")) | words_in(it.get("reference", ""))
        doc_sources.append(("pdf-pastes.input+reference", it["id"], ws))

    df = Counter()
    per_source_docs = Counter()
    for source, doc_id, ws in doc_sources:
        per_source_docs[source] += 1
        for w in ws:
            df[w] += 1

    words_sorted = sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))

    payload = {
        "provenance": {
            "method": (
                "Document frequency over word types (ASCII letters, "
                "optional internal apostrophe) extracted from the fields "
                "named below. One document = one corpus item; a word's "
                "document frequency is the number of distinct items it "
                "appears in (not raw token count). No word was hand-added, "
                "no casing/stemming normalization was applied."
            ),
            "sources": [
                {
                    "corpus": "bench/arxiv-pastes.json",
                    "fields": ["reference", "meta.reference_prose"],
                    "n_items": per_source_docs["arxiv-pastes.reference+meta.reference_prose"],
                },
                {
                    "corpus": "bench/bench-data-extended.json",
                    "fields": ["input", "reference"],
                    "n_items": per_source_docs["bench-data-extended.input+reference"],
                },
                {
                    "corpus": "bench/pdf-pastes.json",
                    "fields": ["input", "reference"],
                    "n_items": per_source_docs["pdf-pastes.input+reference"],
                },
            ],
            "n_documents_total": len(doc_sources),
            "n_word_types": len(df),
            "limitation": (
                "This corpus is small (255 items total, mostly short "
                "single-formula passages plus 120 arXiv paragraphs). The "
                "resulting word list is a floor on maths-English coverage, "
                "not a general English vocabulary -- it will miss common "
                "English words that happen not to occur in these specific "
                "benchmark passages. Reported as a limitation, not papered "
                "over: see the PR/README for how this affects the gate."
            ),
        },
        "word_document_frequency": [{"word": w, "df": c} for w, c in words_sorted],
        "words": [w for w, _ in words_sorted],
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"documents scanned: {len(doc_sources)}")
    print(f"distinct word types: {len(df)}")
    print(f"top 20 by document frequency: {[w for w, _ in words_sorted[:20]]}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
