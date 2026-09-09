#!/usr/bin/env python3
"""Harvest plaintext (with inline LaTeX preserved) from Stack Exchange dumps.

Part of Source C of the Layer-3 keep-set corpus (see
bench/corpus/README-corpus.md). Stack Exchange publishes official bulk data
dumps under CC-BY-SA on the Internet Archive
(https://archive.org/details/stackexchange) specifically so this kind of
bulk, text-only reuse does not require scraping. This script downloads only
that dump, never a live site.

Disk-budget rule (in code, not in my head): this machine had ~8GB of free
disk for this whole task. Stack Exchange's *.7z dumps are downloaded and
processed **one site at a time**, and the raw .7z plus the extracted
Posts.xml are deleted immediately after that site's plaintext is written out,
so peak extra disk usage is bounded to roughly one site's
(compressed + decompressed Posts.xml) size, not the sum of all sites.

`math.stackexchange.com.7z` is 3.63 GB compressed; empirically Posts.xml
decompresses to about 1.4x the archive size, so processing it the same way
would peak at ~8.6 GB -- more than the free disk on this machine. It is
excluded by default (see DEFAULT_SITES) and documented as not attempted
rather than silently skipped. Pass --sites explicitly (with enough free disk)
to include it.

Only `Posts.xml` is extracted from each dump (selective extraction via
py7zr); Comments.xml/Votes.xml/etc. are never downloaded or decompressed.
Question (PostTypeId=1) and Answer (PostTypeId=2) bodies are HTML-stripped to
plaintext with inline LaTeX left untouched (Stack Exchange stores MathJax
source as literal dollar-delimited or backslash-paren-delimited text inside
the HTML body -- stripping tags does not touch it).

For a site named on --subset-sites, only the top --subset-top-n questions by
Score are kept, plus every answer to those questions (documented subsetting
rule, per the brief).

Usage:
    python bench/corpus/harvest_stackexchange.py \\
        --out bench/corpus/cache/stackexchange \\
        --sites tex.stackexchange.com cs.stackexchange.com \\
                stats.stackexchange.com physics.stackexchange.com
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

ARCHIVE_BASE = "https://archive.org/download/stackexchange"

# math.stackexchange.com.7z (3.63 GB) is intentionally excluded here: see
# module docstring. tex.stackexchange.com is first because the brief singles
# it out as the closest text in existence to this product's input.
DEFAULT_SITES = [
    "tex.stackexchange.com",
    "cs.stackexchange.com",
    "stats.stackexchange.com",
    "physics.stackexchange.com",
]


class _HTMLToText(HTMLParser):
    """Minimal HTML->text: drop tags, keep text nodes (incl. inline LaTeX),
    put newlines around block elements so words don't run together."""

    BLOCK_TAGS = {"p", "div", "li", "br", "pre", "blockquote", "h1", "h2", "h3", "ul", "ol"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)

    def gettext(self) -> str:
        return re.sub(r"\n{2,}", "\n", "".join(self.parts)).strip()


def html_to_text(body_html: str) -> str:
    if not body_html:
        return ""
    parser = _HTMLToText()
    try:
        parser.feed(body_html)
    except Exception:
        return html.unescape(re.sub("<[^>]+>", " ", body_html))
    return parser.gettext()


def download(url: str, dest: Path, user_agent: str) -> None:
    if dest.exists():
        print(f"[se] {dest.name} already downloaded, skipping fetch")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as resp, tmp.open("wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        chunk = resp.read(1 << 20)
        while chunk:
            f.write(chunk)
            read += len(chunk)
            chunk = resp.read(1 << 20)
        print(f"[se] downloaded {dest.name}: {read} bytes (expected {total}) in {time.time()-t0:.1f}s")
    tmp.rename(dest)


def extract_posts_xml(archive_path: Path, extract_dir: Path) -> Path:
    import py7zr

    extract_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        z.extract(path=str(extract_dir), targets=["Posts.xml"])
    return extract_dir / "Posts.xml"


def parse_posts(
    posts_xml_path: Path,
    site: str,
    out_f,
    subset_top_n: int | None,
) -> dict:
    """Stream-parse Posts.xml. Returns per-site subtotal stats."""
    stats = {"site": site, "n_questions": 0, "n_answers": 0, "chars": 0, "subset_applied": False}

    if subset_top_n is not None:
        # Pass 1: collect (id, score, post_type, parent_id) only -- cheap.
        rows = []
        for _, elem in ET.iterparse(str(posts_xml_path), events=("end",)):
            if elem.tag != "row":
                continue
            pt = elem.get("PostTypeId")
            if pt in ("1", "2"):
                rows.append(
                    (
                        int(elem.get("Id")),
                        int(elem.get("Score", "0")),
                        pt,
                        elem.get("ParentId"),
                    )
                )
            elem.clear()
        question_scores = sorted(
            ((score, pid) for pid, score, pt, _ in rows if pt == "1"), reverse=True
        )
        keep_question_ids = {pid for _, pid in question_scores[:subset_top_n]}
        keep_ids = set(keep_question_ids)
        for pid, score, pt, parent_id in rows:
            if pt == "2" and parent_id is not None and int(parent_id) in keep_question_ids:
                keep_ids.add(pid)
        stats["subset_applied"] = True
        stats["subset_top_n_questions"] = subset_top_n
        stats["subset_kept_question_ids"] = len(keep_question_ids)
        stats["subset_kept_total_posts"] = len(keep_ids)
        del rows
    else:
        keep_ids = None

    # Pass 2 (or only pass, if no subsetting): extract text.
    for _, elem in ET.iterparse(str(posts_xml_path), events=("end",)):
        if elem.tag != "row":
            continue
        pt = elem.get("PostTypeId")
        if pt not in ("1", "2"):
            elem.clear()
            continue
        pid = int(elem.get("Id"))
        if keep_ids is not None and pid not in keep_ids:
            elem.clear()
            continue
        title = elem.get("Title", "")
        body_html = elem.get("Body", "")
        text = ((title + "\n") if title else "") + html_to_text(body_html)
        if text.strip():
            out_f.write(
                json.dumps(
                    {
                        "site": site,
                        "id": pid,
                        "post_type": "question" if pt == "1" else "answer",
                        "score": int(elem.get("Score", "0")),
                        "text": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stats["chars"] += len(text)
            if pt == "1":
                stats["n_questions"] += 1
            else:
                stats["n_answers"] += 1
        elem.clear()

    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sites", nargs="+", default=DEFAULT_SITES)
    ap.add_argument("--contact", required=True)
    ap.add_argument(
        "--subset-site",
        action="append",
        default=[],
        metavar="SITE:TOP_N",
        help="apply top-N-questions-by-score subsetting to a site, e.g. math.stackexchange.com:50000",
    )
    args = ap.parse_args()

    subset_map = {}
    for spec in args.subset_site:
        site, n = spec.split(":")
        subset_map[site] = int(n)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = out_dir / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    posts_out_path = out_dir / "posts.jsonl"
    stats_path = out_dir / "stats.json"

    user_agent = (
        "latexgen-keepset-corpus/1.0 (Stack Exchange CC-BY-SA dump processing "
        f"for BPE frequency ranking; contact: {args.contact})"
    )

    all_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    done_sites = set(all_stats.keys())

    with posts_out_path.open("a", encoding="utf-8") as out_f:
        for site in args.sites:
            if site in done_sites:
                print(f"[se] {site} already processed, skipping")
                continue

            archive_path = scratch_dir / f"{site}.7z"
            url = f"{ARCHIVE_BASE}/{site}.7z"
            print(f"[se] === {site} ===")
            download(url, archive_path, user_agent)

            extract_dir = scratch_dir / f"{site}-extract"
            posts_xml = extract_posts_xml(archive_path, extract_dir)
            print(f"[se] extracted {posts_xml} ({posts_xml.stat().st_size} bytes)")

            # Free the compressed archive before parsing -- this is the disk
            # budget rule: never hold (compressed + decompressed) at once
            # longer than necessary.
            archive_path.unlink()

            stats = parse_posts(posts_xml, site, out_f, subset_map.get(site))
            out_f.flush()
            print(f"[se] {site}: {stats}")

            posts_xml.unlink()
            try:
                extract_dir.rmdir()
            except OSError:
                pass

            all_stats[site] = stats
            stats_path.write_text(json.dumps(all_stats, indent=2))

    try:
        scratch_dir.rmdir()
    except OSError:
        pass

    print(f"[se] done. sites processed: {list(all_stats.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
