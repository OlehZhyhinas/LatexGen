#!/usr/bin/env python3
"""Harvest OpenStax textbook prose (CNXML source), CC-licensed, via GitHub.

Part of Source C of the Layer-3 keep-set corpus (see
bench/corpus/README-corpus.md). OpenStax textbooks are openly licensed
(commonly CC BY 4.0; some CC BY-NC-SA -- the exact licence is recorded per
book below and in the provenance output) and their source is published as
CNXML in GitHub repos under github.com/openstax (the "osbooks-*" repos).

Disk-budget rule (in code): a full `git clone` of an osbooks-* repo also
pulls its `media/` directory of images, which is much larger than the text
and irrelevant to this task -- several of these repos are 100-700+ MB of
git-repo size, mostly images. Instead this script uses the GitHub REST API
to (a) list the repo's file tree and (b) fetch only `*/index.cnxml` files
individually via raw.githubusercontent.com, never cloning the repo or
touching `media/`. This keeps disk usage to a few MB of XML per book.

Text is extracted as all text content under CNXML's <content> element
(paragraphs, captions, list items, notes, titles, and the individual
mi/mn/mo tokens of any embedded MathML) -- this is a deliberately generic
extraction that keeps prose and drops only markup, matching the brief's
"plain-language technical prose ... the register users actually paste out of
PDFs".

Usage:
    python bench/corpus/harvest_openstax.py \\
        --out bench/corpus/cache/openstax --contact "you@example.com"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

GITHUB_API = "https://api.github.com"
RAW_BASE = "https://raw.githubusercontent.com/openstax"
CNXML_NS = "{http://cnx.rice.edu/cnxml}"

# Science-root books selected from github.com/openstax's public osbooks-*
# repos (enumerated via the GitHub search API on 2026-09-09), restricted to
# English-language math/physics/chemistry/biology/statistics/CS textbooks
# per the brief's science roots. Licence as declared in each repo's
# LICENSE/META-INF (recorded in provenance.md; OpenStax books are CC BY 4.0
# unless noted).
BOOKS = [
    ("osbooks-calculus-bundle", "main", "CC BY 4.0"),
    ("osbooks-college-algebra-bundle", "main", "CC BY 4.0"),
    ("osbooks-prealgebra-bundle", "main", "CC BY 4.0"),
    ("osbooks-algebra-1", "main", "CC BY 4.0"),
    ("osbooks-contemporary-mathematics", "main", "CC BY 4.0"),
    ("osbooks-university-physics-bundle", "main", "CC BY 4.0"),
    ("osbooks-college-physics-bundle", "main", "CC BY 4.0"),
    ("osbooks-astronomy", "main", "CC BY 4.0"),
    ("osbooks-chemistry-bundle", "main", "CC BY 4.0"),
    ("osbooks-organic-chemistry", "main", "CC BY 4.0"),
    ("osbooks-biology-bundle", "main", "CC BY 4.0"),
    ("osbooks-microbiology", "main", "CC BY 4.0"),
    ("osbooks-anatomy-physiology", "main", "CC BY 4.0"),
    ("osbooks-introductory-statistics-bundle", "main", "CC BY 4.0"),
    ("osbooks-statistics", "main", "CC BY 4.0"),
    ("osbooks-introduction-python-programming", "main", "CC BY 4.0"),
    ("osbooks-principles-data-science", "main", "CC BY 4.0"),
]


def http_get(url: str, user_agent: str, as_json: bool, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/vnd.github+json"})
    while True:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                return json.loads(body) if as_json else body.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                wait = int(e.headers.get("Retry-After", "60") or "60")
                print(f"[openstax] HTTP {e.code} (rate limit?), waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            if e.code == 404:
                raise
            print(f"[openstax] HTTP {e.code}, retrying in 15s", file=sys.stderr)
            time.sleep(15)
            continue
        except urllib.error.URLError as e:
            print(f"[openstax] network error {e!r}, retrying in 15s", file=sys.stderr)
            time.sleep(15)
            continue


def list_cnxml_paths(repo: str, branch: str, user_agent: str) -> list[str]:
    url = f"{GITHUB_API}/repos/openstax/{repo}/git/trees/{branch}?recursive=1"
    data = http_get(url, user_agent, as_json=True)
    return [t["path"] for t in data.get("tree", []) if t["path"].endswith("index.cnxml")]


def cnxml_to_text(xml_text: str) -> str:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return ""
    content = root.find(f"{CNXML_NS}content")
    if content is None:
        content = root
    text = " ".join(t.strip() for t in content.itertext() if t and t.strip())
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--contact", required=True)
    ap.add_argument("--delay", type=float, default=0.3)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "modules.jsonl"
    state_path = out_dir / "state.json"
    stats_path = out_dir / "stats.json"

    user_agent = f"latexgen-keepset-corpus/1.0 (OpenStax CNXML harvest via GitHub API for BPE frequency ranking; contact: {args.contact})"

    state = json.loads(state_path.read_text()) if state_path.exists() else {"fetched_paths": []}
    fetched = set(state["fetched_paths"])
    all_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    with out_path.open("a", encoding="utf-8") as out_f:
        for repo, branch, licence in BOOKS:
            if repo in all_stats:
                print(f"[openstax] {repo} already done, skipping")
                continue
            print(f"[openstax] === {repo} ===")
            try:
                paths = list_cnxml_paths(repo, branch, user_agent)
            except urllib.error.HTTPError as e:
                print(f"[openstax] could not list {repo}: {e}", file=sys.stderr)
                continue
            book_chars = 0
            book_modules = 0
            for path in paths:
                key = f"{repo}/{path}"
                if key in fetched:
                    continue
                url = f"{RAW_BASE}/{repo}/{branch}/{path}"
                try:
                    xml_text = http_get(url, user_agent, as_json=False)
                except urllib.error.HTTPError as e:
                    print(f"[openstax] skip {key}: {e}", file=sys.stderr)
                    fetched.add(key)
                    continue
                text = cnxml_to_text(xml_text)
                if text:
                    out_f.write(
                        json.dumps(
                            {"repo": repo, "path": path, "licence": licence, "text": text},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    book_chars += len(text)
                    book_modules += 1
                fetched.add(key)
                time.sleep(args.delay)
            out_f.flush()
            state["fetched_paths"] = sorted(fetched)
            state_path.write_text(json.dumps(state))
            all_stats[repo] = {"licence": licence, "modules": book_modules, "chars": book_chars}
            stats_path.write_text(json.dumps(all_stats, indent=2))
            print(f"[openstax] {repo}: {book_modules} modules, {book_chars} chars")

    print(f"[openstax] done. books: {list(all_stats.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
