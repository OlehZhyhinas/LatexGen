#!/usr/bin/env python3
"""Harvest science/maths Wikibooks and Wikiversity text.

Part of Source C of the Layer-3 keep-set corpus (see
bench/corpus/README-corpus.md), via "the same MediaWiki API route as Source
B" (action=query&prop=extracts&explaintext), pointed at en.wikibooks.org and
en.wikiversity.org.

Selection rule (in code): unlike English Wikipedia, Wikibooks/Wikiversity do
not maintain a clean subject-category tree (each book instead gets its own
auto-generated per-book "Book:<title>" category, which is useless for topic
traversal -- checked empirically 2026-09-09). Title discovery therefore uses
action=query&list=search restricted to the main namespace (ns=0, i.e. actual
book/lesson pages, not talk/user/template pages), one search per science root
term, paginated, deduplicated across roots. This is still the same MediaWiki
API family as Source B, just using search instead of category BFS because
category BFS does not have adequate signal on these two wikis.

Cached, resumable, rate-limited exactly like harvest_wikipedia_science.py.

Usage:
    python bench/corpus/harvest_wikibooks.py \\
        --out bench/corpus/cache/wikibooks --site en.wikibooks.org \\
        --contact "you@example.com" --max-per-root 400
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT_TERMS = [
    "Mathematics",
    "Physics",
    "Chemistry",
    "Biology",
    "Statistics",
    "Computer science",
    "Astronomy",
    "Earth science",
    "Engineering",
]


def api_get(api_url: str, params: dict, user_agent: str, timeout: int = 60) -> dict:
    params = {**params, "format": "json"}
    url = api_url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    while True:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                print(f"[wikibooks] HTTP {e.code}, retrying in 20s", file=sys.stderr)
                time.sleep(20)
                continue
            raise
        except urllib.error.URLError as e:
            print(f"[wikibooks] network error {e!r}, retrying in 20s", file=sys.stderr)
            time.sleep(20)
            continue


def search_titles(api_url: str, term: str, max_results: int, user_agent: str, delay: float):
    offset = 0
    seen = 0
    while seen < max_results:
        data = api_get(
            api_url,
            {
                "action": "query",
                "list": "search",
                "srsearch": term,
                "srnamespace": "0",
                "srlimit": "50",
                "sroffset": str(offset),
            },
            user_agent,
        )
        results = data.get("query", {}).get("search", [])
        if not results:
            break
        for r in results:
            yield r["title"]
        seen += len(results)
        time.sleep(delay)
        cont = data.get("continue", {}).get("sroffset")
        if cont is None:
            break
        offset = cont


def fetch_one_extract(api_url: str, title: str, user_agent: str) -> tuple[str, str]:
    data = api_get(
        api_url,
        {"action": "query", "prop": "extracts", "explaintext": "1", "redirects": "1", "titles": title},
        user_agent,
    )
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        return page.get("title", title), page.get("extract", "") or ""
    return title, ""


def load_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--site", required=True, help="e.g. en.wikibooks.org or en.wikiversity.org")
    ap.add_argument("--contact", required=True)
    ap.add_argument("--max-per-root", type=int, default=400)
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    api_url = f"https://{args.site}/w/api.php"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    titles_path = out_dir / "titles.json"
    articles_path = out_dir / "articles.jsonl"
    fetched_path = out_dir / "fetched_titles.json"

    user_agent = f"latexgen-keepset-corpus/1.0 (science search+extract harvest of {args.site} for BPE frequency ranking; contact: {args.contact})"

    titles = load_json(titles_path, None)
    if titles is None:
        seen = set()
        titles = []
        for term in ROOT_TERMS:
            print(f"[wikibooks] searching {args.site} for {term!r}")
            for t in search_titles(api_url, term, args.max_per_root, user_agent, args.delay):
                if t not in seen:
                    seen.add(t)
                    titles.append(t)
        save_json(titles_path, titles)
    print(f"[wikibooks] {len(titles)} unique titles from {args.site}")

    fetched = set(load_json(fetched_path, []))
    todo = [t for t in titles if t not in fetched]
    print(f"[wikibooks] {len(fetched)} already fetched, {len(todo)} to go")

    import concurrent.futures
    import threading

    write_lock = threading.Lock()
    articles_f = articles_path.open("a", encoding="utf-8")
    done_count = 0

    def worker(title):
        nonlocal done_count
        try:
            got_title, extract = fetch_one_extract(api_url, title, user_agent)
        except Exception as e:
            print(f"[wikibooks] error fetching {title!r}: {e}", file=sys.stderr)
            got_title, extract = title, ""
        time.sleep(args.delay)
        with write_lock:
            if extract:
                articles_f.write(
                    json.dumps({"site": args.site, "title": got_title, "text": extract}, ensure_ascii=False) + "\n"
                )
                articles_f.flush()
            fetched.add(title)
            done_count += 1
            if done_count % 100 == 0:
                save_json(fetched_path, sorted(fetched))
                print(f"[wikibooks] extracts {done_count}/{len(todo)}")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(worker, todo))
    finally:
        save_json(fetched_path, sorted(fetched))
        articles_f.close()

    print(f"[wikibooks] done: {len(fetched)} titles fetched for {args.site}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
