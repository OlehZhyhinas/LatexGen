#!/usr/bin/env python3
"""The five analyses the brief asks for, computed from freq-<label>.json.

1. Overlap between corpus-frequency top-K and merge-rank top-K (K=32768,
   65536). Merge rank == token id (BPE ids are assigned in the order merges
   were learned; verified empirically against bench/keepsets/*/keep-idx.json
   -- see README-corpus.md), so "merge-rank top-K" is simply {0, ..., K-1}.
2. The contested band: tokens corpus-rank promotes into top-65536 that
   merge-rank excludes, and vice versa. Top 50 each way.
3. Corpus rank (and top-65536/32768 membership) of the specific tokens named
   in the brief.
4. Held-out coverage on bench/arxiv-pastes.json input+reference strings:
   per-token out-of-set rate for a corpus-rank keep-set vs a merge-rank
   keep-set, at K=32768 and 65536.
5. Zipf sanity check: count of the 32768th- and 65536th-ranked token.

Usage:
    python bench/corpus/analyze.py \\
        --freq bench/corpus/freq-qwen35.json \\
        --tokenizer bench/corpus/cache/tokenizers/qwen35-tokenizer.json \\
        --held-out bench/arxiv-pastes.json \\
        --out bench/corpus/analysis-qwen35.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Tokens named explicitly in the brief's "tokens we know we need" analysis.
KNOWN_WHOLE_TOKENS = ["Ġoscillator", "Ġdamping", "Ġconverge", "Ġflattened", "Ġreorder", "Â¨", "ĠÏĨ"]
KNOWN_SYMBOLS = ["φ", "←", "↔", "↓", "×"]
KNOWN_PROPER_NOUNS = ["Nikolskii", "Randers", "Lissajous", "Kullback", "Hausdorff", "Chebyshev", "Sobolev"]


def load_freq(path: Path):
    data = json.loads(path.read_text())
    ranked = data["ranked"]
    id_to_rank = {}
    id_to_count = {}
    for r in ranked:
        id_to_rank[r["id"]] = r["rank"]
        id_to_count[r["id"]] = r["count"]
    return data, id_to_rank, id_to_count


def corpus_topk_ids(ranked, k):
    return {r["id"] for r in ranked if r["rank"] <= k}


def merge_rank_topk_ids(k):
    return set(range(k))


def analysis_1_overlap(ranked, ks):
    out = {}
    for k in ks:
        corpus_set = corpus_topk_ids(ranked, k)
        merge_set = merge_rank_topk_ids(k)
        inter = corpus_set & merge_set
        out[str(k)] = {
            "overlap_count": len(inter),
            "overlap_fraction_of_k": len(inter) / k,
            "corpus_only": len(corpus_set - merge_set),
            "merge_only": len(merge_set - corpus_set),
        }
    return out


def analysis_2_contested_band(ranked, id_to_token, id_to_count, k=65536, top_n=50):
    corpus_set = corpus_topk_ids(ranked, k)
    merge_set = merge_rank_topk_ids(k)

    promoted_ids = corpus_set - merge_set  # corpus keeps, merge rank would drop
    demoted_ids = merge_set - corpus_set  # merge rank keeps, corpus would drop

    promoted = sorted(
        ({"id": i, "token": id_to_token.get(i, ""), "corpus_count": id_to_count.get(i, 0), "merge_rank_id": i} for i in promoted_ids),
        key=lambda r: -r["corpus_count"],
    )[:top_n]

    demoted = sorted(
        ({"id": i, "token": id_to_token.get(i, ""), "corpus_count": id_to_count.get(i, 0), "merge_rank_id": i} for i in demoted_ids),
        key=lambda r: r["merge_rank_id"],
    )[:top_n]

    # ids 0-255 are the byte-level BPE base alphabet (verified empirically:
    # id 256 is the first learned merge, "ĠĠ"). Every one of those 256 ids is
    # always present in the vocab as a matter of tokenizer construction, not
    # a Layer-3 ranking decision, so a "demoted" byte-alphabet id is not
    # actually contested territory for whole-ASCII-word/code/punctuation
    # ranking. Report the demoted list a second time restricted to ids>=256
    # so the two effects don't get conflated.
    demoted_ge256_ids = {i for i in demoted_ids if i >= 256}
    demoted_ge256 = sorted(
        ({"id": i, "token": id_to_token.get(i, ""), "corpus_count": id_to_count.get(i, 0), "merge_rank_id": i} for i in demoted_ge256_ids),
        key=lambda r: (r["corpus_count"], r["merge_rank_id"]),
    )[:top_n]

    return {
        "k": k,
        "n_promoted_total": len(promoted_ids),
        "n_demoted_total": len(demoted_ids),
        "n_demoted_in_byte_alphabet_ids_lt_256": len(demoted_ids) - len(demoted_ge256_ids),
        "n_demoted_ge_256": len(demoted_ge256_ids),
        "promoted_top50_by_corpus_count": promoted,
        "demoted_top50_by_lowest_merge_rank_id": demoted,
        "demoted_ge256_top50_by_lowest_corpus_count": demoted_ge256,
    }


def encode_token_pieces(tok, word, leading_space=True):
    text = (" " if leading_space else "") + word
    enc = tok.encode(text, add_special_tokens=False)
    return list(zip(enc.ids, enc.tokens))


def analysis_3_known_tokens(tok, id_to_rank, id_to_count, ks):
    out = {"whole_tokens": {}, "symbols": {}, "proper_nouns": {}}

    for tstr in KNOWN_WHOLE_TOKENS:
        tid = tok.token_to_id(tstr)
        if tid is None:
            out["whole_tokens"][tstr] = {"found_as_single_token": False}
            continue
        rank = id_to_rank.get(tid)
        out["whole_tokens"][tstr] = {
            "found_as_single_token": True,
            "id": tid,
            "corpus_rank": rank,
            "corpus_count": id_to_count.get(tid, 0),
            **{f"in_top_{k}_by_corpus_rank": (rank is not None and rank <= k) for k in ks},
        }

    for sym in KNOWN_SYMBOLS:
        pieces = encode_token_pieces(tok, sym, leading_space=False)
        detail = []
        for tid, tokstr in pieces:
            rank = id_to_rank.get(tid)
            detail.append(
                {
                    "id": tid,
                    "token": tokstr,
                    "corpus_rank": rank,
                    "corpus_count": id_to_count.get(tid, 0),
                    **{f"in_top_{k}_by_corpus_rank": (rank is not None and rank <= k) for k in ks},
                }
            )
        out["symbols"][sym] = detail

    for name in KNOWN_PROPER_NOUNS:
        pieces = encode_token_pieces(tok, name, leading_space=True)
        detail = []
        for tid, tokstr in pieces:
            rank = id_to_rank.get(tid)
            detail.append(
                {
                    "id": tid,
                    "token": tokstr,
                    "corpus_rank": rank,
                    "corpus_count": id_to_count.get(tid, 0),
                    **{f"in_top_{k}_by_corpus_rank": (rank is not None and rank <= k) for k in ks},
                }
            )
        out["proper_nouns"][name] = detail

    return out


def analysis_4_heldout_coverage(tok, held_out_path, ranked, ks):
    items = json.loads(Path(held_out_path).read_text())
    texts = []
    for it in items:
        if it.get("input"):
            texts.append(it["input"])
        if it.get("reference"):
            texts.append(it["reference"])

    out = {"n_texts": len(texts), "by_k": {}}
    total_occurrences = 0
    all_ids = []
    for t in texts:
        enc = tok.encode(t, add_special_tokens=False)
        all_ids.extend(enc.ids)
        total_occurrences += len(enc.ids)
    out["total_token_occurrences"] = total_occurrences

    for k in ks:
        corpus_set = corpus_topk_ids(ranked, k)
        merge_set = merge_rank_topk_ids(k)
        oos_corpus = sum(1 for i in all_ids if i not in corpus_set)
        oos_merge = sum(1 for i in all_ids if i not in merge_set)
        out["by_k"][str(k)] = {
            "out_of_set_rate_corpus_rank_keepset": oos_corpus / total_occurrences if total_occurrences else None,
            "out_of_set_rate_merge_rank_keepset": oos_merge / total_occurrences if total_occurrences else None,
            "out_of_set_count_corpus_rank_keepset": oos_corpus,
            "out_of_set_count_merge_rank_keepset": oos_merge,
        }
    return out


def analysis_5_zipf(ranked, ks):
    by_rank = {r["rank"]: r for r in ranked}
    out = {}
    for k in ks:
        r = by_rank.get(k)
        out[str(k)] = {
            "token": r["token"] if r else None,
            "id": r["id"] if r else None,
            "count": r["count"] if r else 0,
            "note": "no token reached this rank in the corpus" if r is None else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--held-out", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    freq_data, id_to_rank, id_to_count = load_freq(Path(args.freq))
    ranked = freq_data["ranked"]
    tok = Tokenizer.from_file(args.tokenizer)
    vocab = tok.get_vocab(with_added_tokens=True)
    id_to_token = {v: k for k, v in vocab.items()}

    ks = [32768, 65536]
    result = {
        "tokenizer_label": freq_data["tokenizer_label"],
        "vocab_size": freq_data["vocab_size"],
        "total_tokens": freq_data["total_tokens"],
        "total_docs": freq_data["total_docs"],
        "source_totals": freq_data["source_totals"],
        "analysis_1_overlap_with_merge_rank": analysis_1_overlap(ranked, ks),
        "analysis_2_contested_band": analysis_2_contested_band(ranked, id_to_token, id_to_count),
        "analysis_3_known_tokens": analysis_3_known_tokens(tok, id_to_rank, id_to_count, ks),
        "analysis_4_heldout_coverage": analysis_4_heldout_coverage(tok, args.held_out, ranked, ks),
        "analysis_5_zipf_sanity": analysis_5_zipf(ranked, ks),
    }

    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[analyze] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
