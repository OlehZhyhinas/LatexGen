#!/usr/bin/env python3
"""Tokenize the harvested corpus with both target tokenizers and accumulate
per-token-id counts, with per-source subtotals kept separable.

Reads every cached harvest output under --cache-dir:
  arxiv/records.jsonl            (field: "abstract", plus "title")
  wikipedia/articles.jsonl       (field: "text")
  wikibooks/articles.jsonl       (field: "text")
  wikiversity/articles.jsonl     (field: "text")
  stackexchange/posts.jsonl      (field: "text", "site")
  openstax/modules.jsonl         (field: "text", "repo")

For each of the two tokenizers (Qwen3.5, MiniCPM5-2B; both loaded from the
pinned, hash-verified tokenizer.json files downloaded by this task -- see
bench/corpus/README-corpus.md), every document is encoded and per-token-id
counts are accumulated:
  - a global count vector (length = vocab size)
  - one count vector per source-group (arxiv / wikipedia / wikibooks /
    wikiversity / stackexchange:<site> / openstax), so each source's
    contribution to any given token's count can be recovered later.

Output: bench/corpus/freq-<label>.json
  {
    "tokenizer_label": "...",
    "vocab_size": N,
    "total_tokens": N,
    "total_docs": N,
    "source_totals": {"arxiv": n_tokens, ...},
    "ranked": [{"id": ..., "token": ..., "count": ..., "rank": ...}, ...],
    "by_source_top": ... (not stored per-id here; see freq-<label>-by-source.json)
  }

A companion bench/corpus/freq-<label>-by-source.json stores, per source, the
count contributed to every *nonzero* token id (sparse), so the ranking can be
audited per source without re-tokenizing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def iter_docs(cache_dir: Path):
    """Yield (source_group, text) for every cached document."""
    arxiv_path = cache_dir / "arxiv" / "records.jsonl"
    if arxiv_path.exists():
        with arxiv_path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                text = (r.get("title", "") + "\n" + r.get("abstract", "")).strip()
                if text:
                    yield "arxiv", text

    wiki_path = cache_dir / "wikipedia" / "articles.jsonl"
    if wiki_path.exists():
        with wiki_path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("text"):
                    yield "wikipedia", r["text"]

    for wname, group in (("wikibooks", "wikibooks"), ("wikiversity", "wikiversity")):
        p = cache_dir / wname / "articles.jsonl"
        if p.exists():
            with p.open(encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("text"):
                        yield group, r["text"]

    se_path = cache_dir / "stackexchange" / "posts.jsonl"
    if se_path.exists():
        with se_path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("text"):
                    yield f"stackexchange:{r.get('site', 'unknown')}", r["text"]

    os_path = cache_dir / "openstax" / "modules.jsonl"
    if os_path.exists():
        with os_path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("text"):
                    yield "openstax", r["text"]

    hn_path = cache_dir / "hackernews" / "items.jsonl"
    if hn_path.exists():
        with hn_path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("text"):
                    yield "hackernews", r["text"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--tokenizer",
        action="append",
        required=True,
        metavar="LABEL:PATH",
        help="e.g. qwen35:bench/corpus/cache/tokenizers/qwen35-tokenizer.json",
    )
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--report-every", type=int, default=200_000)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = []
    for spec in args.tokenizer:
        label, path = spec.split(":", 1)
        tok = Tokenizer.from_file(path)
        vocab_size = tok.get_vocab_size(with_added_tokens=True)
        specs.append(
            {
                "label": label,
                "path": path,
                "tok": tok,
                "vocab_size": vocab_size,
                "global_counts": np.zeros(vocab_size, dtype=np.int64),
                "source_counts": {},  # source -> np.int64 array (lazy)
                "source_totals": {},  # source -> int
                "total_tokens": 0,
                "total_docs": 0,
            }
        )

    def get_source_arr(spec, source):
        arr = spec["source_counts"].get(source)
        if arr is None:
            arr = np.zeros(spec["vocab_size"], dtype=np.int64)
            spec["source_counts"][source] = arr
        return arr

    batch_texts: list[str] = []
    batch_sources: list[str] = []
    n_docs = 0

    def flush_batch():
        if not batch_texts:
            return
        for spec in specs:
            encodings = spec["tok"].encode_batch(batch_texts, add_special_tokens=False)
            for enc, source in zip(encodings, batch_sources):
                ids = enc.ids
                if not ids:
                    continue
                arr = np.asarray(ids, dtype=np.int64)
                np.add.at(spec["global_counts"], arr, 1)
                np.add.at(get_source_arr(spec, source), arr, 1)
                spec["total_tokens"] += len(ids)
                spec["source_totals"][source] = spec["source_totals"].get(source, 0) + len(ids)
        batch_texts.clear()
        batch_sources.clear()

    for source, text in iter_docs(cache_dir):
        batch_texts.append(text)
        batch_sources.append(source)
        n_docs += 1
        if len(batch_texts) >= args.batch_size:
            flush_batch()
        if n_docs % args.report_every == 0:
            print(f"[count_tokens] processed {n_docs} docs...", file=sys.stderr)
    flush_batch()

    for spec in specs:
        spec["total_docs"] = n_docs

    for spec in specs:
        label = spec["label"]
        counts = spec["global_counts"]
        vocab = spec["tok"].get_vocab(with_added_tokens=True)
        id_to_token = {v: k for k, v in vocab.items()}

        nonzero_ids = np.nonzero(counts)[0]
        order = nonzero_ids[np.argsort(-counts[nonzero_ids], kind="stable")]

        ranked = []
        for rank, tid in enumerate(order, start=1):
            ranked.append(
                {
                    "id": int(tid),
                    "token": id_to_token.get(int(tid), ""),
                    "count": int(counts[tid]),
                    "rank": rank,
                }
            )

        out_path = out_dir / f"freq-{label}.json"
        out_path.write_text(
            json.dumps(
                {
                    "tokenizer_label": label,
                    "tokenizer_path": spec["path"],
                    "vocab_size": spec["vocab_size"],
                    "total_tokens": spec["total_tokens"],
                    "total_docs": spec["total_docs"],
                    "n_distinct_ids_seen": int(len(nonzero_ids)),
                    "source_totals": spec["source_totals"],
                    "ranked": ranked,
                },
                ensure_ascii=False,
            )
        )
        print(f"[count_tokens] wrote {out_path} ({len(ranked)} distinct ids, {spec['total_tokens']} total tokens)")

        # Sparse per-source breakdown, for auditing "what contributed this
        # token's count" without re-tokenizing.
        by_source = {}
        for source, arr in spec["source_counts"].items():
            nz = np.nonzero(arr)[0]
            by_source[source] = {int(i): int(arr[i]) for i in nz}
        by_source_path = out_dir / f"freq-{label}-by-source.json"
        by_source_path.write_text(json.dumps(by_source, ensure_ascii=False))
        print(f"[count_tokens] wrote {by_source_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
