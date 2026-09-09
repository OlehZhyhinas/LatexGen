#!/usr/bin/env python3
"""Harvest science-restricted English Wikipedia article text.

Source B of the Layer-3 keep-set corpus (see bench/corpus/README-corpus.md).

This deliberately does NOT use a Wikipedia dump or "random article" sampling
of general Wikipedia (Oleh, 2026-09-09: general Wikipedia's vocabulary is
dominated by biography/geography/sport/history/politics, which would crowd
out the technical terms this task is trying to rank). Instead it:

  1. Starts from a fixed list of science category roots.
  2. Performs a depth-limited breadth-first traversal of *subcategories*
     using action=query&list=categorymembers.
  3. At every visited category (any depth up to --max-depth), collects
     member *articles* (ns=0).
  4. Applies an explicit keyword allow/deny list (below, in code) to decide
     whether to traverse into / collect from a subcategory, so a reader can
     see exactly what "scientific Wikipedia" meant here.
  5. Fetches article plaintext via action=query&prop=extracts&explaintext,
     batched, and caches every fetch to disk so the harvest is resumable and
     need never be repeated.

Selection rule (in code):
  - DENY_SUBSTRINGS: a subcategory whose (lowercased) title contains any of
    these substrings is neither traversed into nor used as a source of
    articles -- this is how general history/politics/sport/geography/
    entertainment/unrelated-biography is excluded.
  - ALLOW_OVERRIDE_SUBSTRINGS: if a subcategory title also contains one of
    these, the deny is *overridden* -- this is how biographies of
    scientists/mathematicians (who otherwise live under generic
    "people"/"births"/"deaths"-style categories) are explicitly kept. Oleh
    2026-09-09: proper-noun subwords like `Nikolskii` come from exactly this
    population, and a dropped subword (emitted as `Kolskii`) was an observed
    production failure.

Usage:
    python bench/corpus/harvest_wikipedia_science.py \\
        --out bench/corpus/cache/wikipedia \\
        --contact "you@example.com" \\
        --max-depth 3 --max-articles 60000
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

API = "https://en.wikipedia.org/w/api.php"
DEFAULT_DELAY_S = 0.6

ROOT_CATEGORIES = [
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

# Substrings (checked against the lowercased category title with the
# "Category:" prefix stripped) that mark a subcategory as off-topic general
# Wikipedia content: history/politics/sport/geography/entertainment/
# unrelated biography, per the brief's exclusion list.
DENY_SUBSTRINGS = [
    "footballer", "cricketer", "olympic", "politician", "election",
    "military", "warfare", "battles of", "battle of", "wars of",
    "geography of", "cities in", "towns in", "villages in", "airports",
    "discography", "album", "singles by", "songs by", "television",
    "film", "actor", "actress", "sportsp", "association football",
    "national parks", "constituencies", "administrative division",
    "populated places", "county,", "diplomat", "ambassador",
    "people from", "births", "deaths", "alumni of", "murder",
    "terrorism", "hurricane", "census", "monarch", "president of",
    "prime minister", "cabinet", "cuisine", "recipes", "diocese",
    "saints", "fictional", "video game", "novels", "poetry",
    "paintings", "sculptures", "beauty pageant", "reality television",
    "wrestlers", "basketball", "rugby", "cycling", "tennis players",
]

# If a category title contains one of these, it is kept even if it also
# matched DENY_SUBSTRINGS above (e.g. "Mathematicians who died in..." would
# hit "deaths" but must stay because "mathematician" overrides it).
ALLOW_OVERRIDE_SUBSTRINGS = [
    "mathematici", "physicist", "chemist", "biologist", "statistician",
    "astronomer", "engineer", "computer scientist", "scientist",
    "nobel laureate", "fields medal", "turing award", "academician",
    "geologist", "cosmologist", "ecologist", "botanist", "zoologist",
]


def is_denied(title: str) -> bool:
    t = title.lower()
    if any(a in t for a in ALLOW_OVERRIDE_SUBSTRINGS):
        return False
    return any(d in t for d in DENY_SUBSTRINGS)


def api_get(params: dict, user_agent: str, timeout: int = 60) -> dict:
    params = {**params, "format": "json"}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    while True:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                print(f"[wiki] HTTP {e.code}, retrying in 20s", file=sys.stderr)
                time.sleep(20)
                continue
            raise
        except urllib.error.URLError as e:
            print(f"[wiki] network error {e!r}, retrying in 20s", file=sys.stderr)
            time.sleep(20)
            continue


def category_members(cat_title: str, user_agent: str, delay: float):
    """Yield (title, ns) for every member of a category, paginated."""
    cmcontinue = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{cat_title}",
            "cmlimit": "500",
            "cmprop": "title|ns",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = api_get(params, user_agent)
        time.sleep(delay)
        members = data.get("query", {}).get("categorymembers", [])
        for m in members:
            yield m["title"], m["ns"]
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
        cmcontinue = cont


def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text())
    return default


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2))


def crawl_categories(out_dir: Path, user_agent: str, delay: float, max_depth: int, max_articles: int):
    """Phase 1: BFS the category graph, writing article titles to titles.jsonl."""
    state_path = out_dir / "crawl_state.json"
    titles_path = out_dir / "titles.jsonl"
    state = load_json(
        state_path,
        {
            "visited_categories": [],
            "frontier": [[c, 0] for c in ROOT_CATEGORIES],
            "collected_titles": [],
            "done": False,
        },
    )
    visited = set(state["visited_categories"])
    collected = set(state["collected_titles"])
    frontier = state["frontier"]

    if state.get("done"):
        print(f"[wiki] category crawl already done: {len(collected)} article titles")
        return

    titles_f = titles_path.open("a", encoding="utf-8")
    try:
        while frontier and len(collected) < max_articles:
            cat_title, depth = frontier.pop(0)
            key = cat_title.strip()
            if key in visited:
                continue
            visited.add(key)

            if is_denied(key) and depth > 0:
                # Root categories are never denied; subcats are filtered.
                continue

            print(f"[wiki] visiting Category:{key} (depth={depth}, collected={len(collected)})")
            try:
                for title, ns in category_members(key, user_agent, delay):
                    if ns == 0:  # article
                        if title not in collected:
                            collected.add(title)
                            titles_f.write(json.dumps({"title": title, "via_category": key, "depth": depth}) + "\n")
                    elif ns == 14 and depth < max_depth:  # subcategory
                        sub = title.split(":", 1)[1] if ":" in title else title
                        if sub not in visited and not is_denied(sub):
                            frontier.append([sub, depth + 1])
            except Exception as e:
                print(f"[wiki] error on Category:{key}: {e}", file=sys.stderr)

            titles_f.flush()
            state["visited_categories"] = sorted(visited)
            state["collected_titles"] = sorted(collected)
            state["frontier"] = frontier
            save_json(state_path, state)

            if len(collected) >= max_articles:
                print(f"[wiki] hit --max-articles={max_articles}, stopping category crawl")
                break

        if not frontier:
            state["done"] = True
            save_json(state_path, state)
    finally:
        titles_f.close()

    print(f"[wiki] category crawl: {len(visited)} categories visited, {len(collected)} article titles collected")


def fetch_one_extract(title: str, user_agent: str) -> tuple[str, str]:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "titles": title,
    }
    data = api_get(params, user_agent)
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        return page.get("title", title), page.get("extract", "") or ""
    return title, ""


def fetch_extracts(out_dir: Path, user_agent: str, delay: float, workers: int = 4):
    """Phase 2: fetch full plaintext for every collected title, cached.

    The MediaWiki API only allows batching prop=extracts across multiple
    titles when the extract is intro-only (exintro); a *full*-article
    explaintext request is always forced to exlimit=1 server-side (confirmed
    empirically -- the API returns a warning to that effect). We want full
    article bodies for vocabulary coverage, not just intros, so this fetches
    one title per HTTP request, parallelized across a small thread pool to
    keep wall-clock time reasonable while still being polite (each worker
    sleeps --delay between its own requests).
    """
    import concurrent.futures
    import threading

    titles_path = out_dir / "titles.jsonl"
    articles_path = out_dir / "articles.jsonl"
    fetched_path = out_dir / "fetched_titles.json"

    all_titles = []
    seen_t = set()
    with titles_path.open(encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)["title"]
            if t not in seen_t:
                seen_t.add(t)
                all_titles.append(t)

    fetched = set(load_json(fetched_path, []))
    todo = [t for t in all_titles if t not in fetched]
    print(f"[wiki] {len(all_titles)} titles total, {len(fetched)} already fetched, {len(todo)} to go")

    write_lock = threading.Lock()
    articles_f = articles_path.open("a", encoding="utf-8")
    done_count = 0

    def worker(title):
        nonlocal done_count
        try:
            got_title, extract = fetch_one_extract(title, user_agent)
        except Exception as e:
            print(f"[wiki] error fetching {title!r}: {e}", file=sys.stderr)
            got_title, extract = title, ""
        time.sleep(delay)
        with write_lock:
            if extract:
                articles_f.write(json.dumps({"title": got_title, "text": extract}, ensure_ascii=False) + "\n")
                articles_f.flush()
            fetched.add(title)
            done_count += 1
            if done_count % 200 == 0:
                save_json(fetched_path, sorted(fetched))
                print(f"[wiki] extracts {done_count}/{len(todo)}")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(worker, todo))
    finally:
        save_json(fetched_path, sorted(fetched))
        articles_f.close()

    print(f"[wiki] extracts fetch complete: {len(fetched)} titles fetched")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--contact", required=True)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--max-articles", type=int, default=60_000)
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY_S)
    ap.add_argument("--workers", type=int, default=4, help="concurrent extract-fetch workers")
    ap.add_argument("--skip-crawl", action="store_true", help="skip phase 1, only fetch extracts")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    user_agent = (
        f"latexgen-keepset-corpus/1.0 (science-category Wikipedia extract harvest "
        f"for BPE frequency ranking; contact: {args.contact})"
    )

    if not args.skip_crawl:
        crawl_categories(out_dir, user_agent, args.delay, args.max_depth, args.max_articles)
    fetch_extracts(out_dir, user_agent, args.delay, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
