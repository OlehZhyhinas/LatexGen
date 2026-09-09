#!/usr/bin/env python3
"""Harvest arXiv abstract metadata via the sanctioned OAI-PMH bulk route.

Source A of the Layer-3 keep-set corpus (see bench/corpus/README-corpus.md).

Endpoint: http://export.arxiv.org/oai2 (OAI-PMH, ListRecords, metadataPrefix
"arXiv"). This is the bulk *metadata* harvesting route arXiv publishes for
this purpose: https://info.arxiv.org/help/oa/index.html. It is NOT the
requester-pays S3 full-text bulk export, and it does NOT scrape article pages
-- both are explicitly out of scope for this task.

Design:
  - Resumable: the OAI resumptionToken and progress are written to
    `<out>/state.json` after every batch, so a killed/restarted run resumes
    from the last resumption token instead of re-fetching.
  - Cached: every raw OAI-PMH XML response is saved to `<out>/raw/`, and
    parsed records are appended to `<out>/records.jsonl`, so the harvest
    never needs to be repeated to be reprocessed.
  - Rate-limited: a fixed delay between requests plus honoring HTTP 503 +
    Retry-After, per arXiv's OAI-PMH usage policy.
  - Selection rule (in code, not "in your head"): arXiv is a science-only
    repository by construction, so no topical filtering is applied -- every
    OAI set (math, physics, cs, q-bio, q-fin, stat, eess, econ, ...) that
    falls in the requested [from, until) date window is eligible. We record
    the categories string per record for later auditing.

Usage:
    python bench/corpus/harvest_arxiv.py \\
        --out bench/corpus/cache/arxiv \\
        --from-date 2023-01-01 --until-date 2026-09-09 \\
        --max-records 200000 \\
        --contact "you@example.com"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

OAI_BASE = "http://export.arxiv.org/oai2"
NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arxiv": "http://arxiv.org/OAI/arXiv/",
}
DEFAULT_DELAY_S = 4.0  # polite delay between OAI-PMH requests


def fetch(params: dict, user_agent: str, timeout: int = 90) -> bytes:
    url = OAI_BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    while True:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 503:
                retry_after = int(e.headers.get("Retry-After", "30") or "30")
                print(f"[arxiv] 503 Retry-After={retry_after}s", file=sys.stderr)
                time.sleep(retry_after)
                continue
            raise
        except urllib.error.URLError as e:
            print(f"[arxiv] network error {e!r}, retrying in 30s", file=sys.stderr)
            time.sleep(30)
            continue


def parse_records(xml_bytes: bytes):
    """Yield (record_dict or None-if-deleted) plus the resumptionToken (or None)."""
    root = ET.fromstring(xml_bytes)
    list_records = root.find("oai:ListRecords", NS)
    if list_records is None:
        # Could be an OAI error (e.g. noRecordsMatch)
        err = root.find("oai:error", NS)
        if err is not None:
            raise RuntimeError(f"OAI error: {err.get('code')}: {err.text}")
        return [], None

    out = []
    for rec in list_records.findall("oai:record", NS):
        header = rec.find("oai:header", NS)
        if header is not None and header.get("status") == "deleted":
            continue
        meta = rec.find("oai:metadata", NS)
        if meta is None:
            continue
        av = meta.find("arxiv:arXiv", NS)
        if av is None:
            continue

        def gettext(tag):
            el = av.find(f"arxiv:{tag}", NS)
            return (el.text or "").strip() if el is not None and el.text else ""

        arxiv_id = gettext("id")
        categories = gettext("categories")
        created = gettext("created")
        title = " ".join(gettext("title").split())
        abstract = gettext("abstract").strip()
        if not abstract:
            continue
        out.append(
            {
                "id": arxiv_id,
                "categories": categories,
                "created": created,
                "title": title,
                "abstract": abstract,
            }
        )

    token_el = list_records.find("oai:resumptionToken", NS)
    resumption_token = None
    if token_el is not None and token_el.text and token_el.text.strip():
        resumption_token = token_el.text.strip()
    cursor = token_el.get("cursor") if token_el is not None else None
    complete_list_size = token_el.get("completeListSize") if token_el is not None else None
    return out, {
        "resumption_token": resumption_token,
        "cursor": cursor,
        "complete_list_size": complete_list_size,
    }


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {
        "resumption_token": None,
        "batch_num": 0,
        "records_fetched": 0,
        "done": False,
        "from_date": None,
        "until_date": None,
    }


def save_state(state_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="cache directory")
    ap.add_argument("--from-date", default=None, help="OAI 'from' date, YYYY-MM-DD")
    ap.add_argument("--until-date", default=None, help="OAI 'until' date, YYYY-MM-DD")
    ap.add_argument("--max-records", type=int, default=200_000)
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY_S)
    ap.add_argument(
        "--contact",
        required=True,
        help="contact email/URL included in the User-Agent, per arXiv OAI-PMH policy",
    )
    ap.add_argument("--max-batches", type=int, default=None, help="stop after N batches this run (for testing)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    records_path = out_dir / "records.jsonl"

    user_agent = f"latexgen-keepset-corpus/1.0 (bulk OAI-PMH abstract harvest for BPE frequency ranking; contact: {args.contact})"

    state = load_state(state_path)
    if state["from_date"] is None:
        state["from_date"] = args.from_date
        state["until_date"] = args.until_date

    if state.get("done"):
        print(f"[arxiv] already marked done at {state['records_fetched']} records; nothing to do")
        return 0

    batches_this_run = 0
    with records_path.open("a", encoding="utf-8") as records_f:
        while state["records_fetched"] < args.max_records:
            if args.max_batches is not None and batches_this_run >= args.max_batches:
                print(f"[arxiv] hit --max-batches={args.max_batches} for this run, stopping (resumable)")
                break

            if state["resumption_token"]:
                params = {"verb": "ListRecords", "resumptionToken": state["resumption_token"]}
            else:
                params = {"verb": "ListRecords", "metadataPrefix": "arXiv"}
                if state["from_date"]:
                    params["from"] = state["from_date"]
                if state["until_date"]:
                    params["until"] = state["until_date"]

            t0 = time.time()
            body = fetch(params, user_agent)
            batch_num = state["batch_num"]
            (raw_dir / f"batch-{batch_num:06d}.xml").write_bytes(body)

            try:
                recs, token_info = parse_records(body)
            except RuntimeError as e:
                print(f"[arxiv] {e}", file=sys.stderr)
                state["done"] = True
                save_state(state_path, state)
                return 0

            for r in recs:
                records_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            records_f.flush()

            state["batch_num"] = batch_num + 1
            state["records_fetched"] += len(recs)
            state["resumption_token"] = token_info["resumption_token"] if token_info else None
            state["complete_list_size"] = token_info.get("complete_list_size") if token_info else None
            save_state(state_path, state)

            elapsed = time.time() - t0
            print(
                f"[arxiv] batch={batch_num} +{len(recs)} recs "
                f"total={state['records_fetched']} cursor={token_info.get('cursor') if token_info else None} "
                f"/{state.get('complete_list_size')} ({elapsed:.1f}s)"
            )
            batches_this_run += 1

            if not state["resumption_token"]:
                print("[arxiv] no resumptionToken returned; harvest complete")
                state["done"] = True
                save_state(state_path, state)
                break

            time.sleep(max(0.0, args.delay - elapsed))

    print(f"[arxiv] stopped at {state['records_fetched']} records (done={state.get('done', False)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
