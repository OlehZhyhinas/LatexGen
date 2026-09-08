#!/usr/bin/env python3
"""Build arXiv PDF-paste benchmark items with TeX-derived references."""
from __future__ import annotations

import argparse
import collections
import dataclasses
import difflib
import gzip
import io
import json
import math
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / "arxiv" / "cache"
PDFTOTEXT = "/opt/homebrew/bin/pdftotext"
SUMMARY_DEFAULT = ROOT / "arxiv-pastes-summary.md"

CATEGORIES = [
    "math.AP",
    "math.PR",
    "math.CO",
    "math.NA",
    "physics.class-ph",
    "cond-mat.stat-mech",
    "quant-ph",
    "cs.LG",
    "cs.IT",
    "stat.ML",
    "econ.TH",
    "q-bio.PE",
    "astro-ph.GA",
    "hep-th",
]

NEW_STYLE_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")
WORD_RE = re.compile(r"[A-Za-z0-9]+")
ALPHA4_RE = re.compile(r"[a-z]{4,}")

MATH_ENVS = {
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
    "multline",
    "multline*",
}
SKIP_ENVS = {
    "figure",
    "figure*",
    "table",
    "table*",
    "tabular",
    "tabular*",
    "algorithm",
    "algorithm*",
    "algorithmic",
    "lstlisting",
    "verbatim",
    "thebibliography",
    "abstract",
}

CITE_RE = re.compile(r"\\cite[a-zA-Z*]*\s*(?:\[[^\]]*\]\s*){0,2}\{[^{}]*\}")
REF_RE = re.compile(r"\\(?:ref|eqref|autoref|[cC]ref\*?)\s*\{[^{}]*\}")
LABEL_RE = re.compile(r"\\label\s*\{[^{}]*\}")
BEGIN_RE = re.compile(r"\\begin\{([^{}]+)\}")
INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^{}]+)\}")

GREEK_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")
MATH_HINT_RE = re.compile(r"[=+\-−/^∑∫√∂∇≤≥≈∞]")
ALNUM_ADJ_RE = re.compile(r"(?:[A-Za-z]\d|\d[A-Za-z])")
BACKSLASH_CMD_RE = re.compile(r"\\[A-Za-z@]+|\\.")

DROP_ENV_NAMES = {
    "proof",
    "theorem",
    "lemma",
    "remark",
    "definition",
    "proposition",
    "corollary",
    "example",
    "enumerate",
    "itemize",
}
DROP_ENV_LINE_RE = re.compile(
    r"^\s*\\(?:begin|end)\{([A-Za-z@]+)\*?\}(?:\[[^\]]*\])?\s*$",
    flags=re.MULTILINE,
)
DROP_ENV_CMD_RE = re.compile(
    r"\\(?:begin|end)\{((?:proof|theorem|lemma|remark|definition|proposition|corollary|example|enumerate|itemize)\*?)\}(?:\[[^\]]*\])?",
    flags=re.IGNORECASE,
)
SECTION_START_RE = re.compile(r"^\s*\\(?:sub)?section\*?\b")
SPACING_CMD_RE = re.compile(
    r"\\(?:noindent|indent|newline|linebreak|pagebreak|smallskip|medskip|bigskip)\b"
)
VSPACE_CMD_RE = re.compile(r"\\(?:vspace|hspace)\*?\s*\{[^{}]*\}")
FOOTNOTEMARK_RE = re.compile(r"\\footnotemark(?:\s*\[[^\]]*\])?")

GREEK_CMD_NAMES = {
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "varepsilon",
    "zeta",
    "eta",
    "theta",
    "vartheta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "pi",
    "varpi",
    "rho",
    "varrho",
    "sigma",
    "varsigma",
    "tau",
    "upsilon",
    "phi",
    "varphi",
    "chi",
    "psi",
    "omega",
    "Gamma",
    "Delta",
    "Theta",
    "Lambda",
    "Xi",
    "Pi",
    "Sigma",
    "Upsilon",
    "Phi",
    "Psi",
    "Omega",
}


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def is_escaped(text: str, idx: int) -> bool:
    n = 0
    j = idx - 1
    while j >= 0 and text[j] == "\\":
        n += 1
        j -= 1
    return (n % 2) == 1


def strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        cut = len(line)
        for i, ch in enumerate(line):
            if ch == "%" and not is_escaped(line, i):
                cut = i
                break
        out.append(line[:cut])
    return "\n".join(out)


def safe_extract_tar_bytes(blob: bytes, dest: Path) -> bool:
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tf:
            root = dest.resolve()
            for member in tf.getmembers():
                target = (dest / member.name).resolve()
                if not str(target).startswith(str(root) + "/") and target != root:
                    continue
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    src = tf.extractfile(member)
                    if src is None:
                        continue
                    target.write_bytes(src.read())
            return True
    except tarfile.TarError:
        return False


def maybe_decode_text(blob: bytes) -> str | None:
    for enc in ("utf-8", "latin-1"):
        try:
            txt = blob.decode(enc)
        except UnicodeDecodeError:
            continue
        if "\\begin{document}" in txt or "\\documentclass" in txt:
            return txt
    return None


class ArxivClient:
    def __init__(self, min_interval: float = 3.0):
        self.min_interval = min_interval
        self.last_request: dict[str, float] = {}
        self.hosts = {"export.arxiv.org", "arxiv.org"}

    def _wait(self, host: str) -> None:
        if host not in self.hosts:
            return
        now = time.time()
        last = self.last_request.get(host)
        if last is not None:
            dt = now - last
            if dt < self.min_interval:
                time.sleep(self.min_interval - dt)
        self.last_request[host] = time.time()

    def fetch_bytes(self, url: str, cache_path: Path | None = None, max_attempts: int = 7) -> bytes:
        if cache_path and cache_path.exists():
            return cache_path.read_bytes()
        host = urllib.parse.urlparse(url).netloc
        err = None
        for attempt in range(max_attempts):
            self._wait(host)
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "latexgen-text-gates/1.0"},
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = r.read()
                break
            except urllib.error.HTTPError as ex:
                err = ex
                if ex.code not in {429, 500, 502, 503, 504}:
                    raise
                retry_after = 0
                if ex.headers is not None:
                    try:
                        retry_after = int(ex.headers.get("Retry-After", "0") or "0")
                    except ValueError:
                        retry_after = 0
                sleep_s = max(12.0, retry_after, self.min_interval * (attempt + 2))
                time.sleep(sleep_s)
                continue
            except urllib.error.URLError as ex:
                err = ex
                time.sleep(max(8.0, self.min_interval * (attempt + 2)))
                continue
        else:
            if err is not None:
                raise err
            raise RuntimeError(f"fetch failed for {url}")
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(data)
        return data


def parse_arxiv_feed(xml_bytes: bytes, fallback_category: str) -> list[dict]:
    out = []
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(xml_bytes)
    for ent in root.findall("atom:entry", ns):
        id_text = (ent.findtext("atom:id", default="", namespaces=ns) or "").strip()
        if not id_text:
            continue
        if "/abs/" in id_text:
            aid = id_text.rsplit("/abs/", 1)[-1]
        else:
            aid = id_text.split("/")[-1]
        aid = aid.strip()
        aid_no_ver = re.sub(r"v\d+$", "", aid)
        if not NEW_STYLE_ID_RE.fullmatch(aid_no_ver):
            continue

        cat = fallback_category
        pc = ent.find("arxiv:primary_category", ns)
        if pc is not None and pc.get("term"):
            cat = pc.get("term")

        lic = "unknown"
        for link in ent.findall("atom:link", ns):
            if (link.get("title") or "").strip().lower() == "license":
                href = (link.get("href") or "").strip()
                if href:
                    lic = href
                    break
        arxiv_lic = (ent.findtext("arxiv:license", default="", namespaces=ns) or "").strip()
        if arxiv_lic:
            lic = arxiv_lic

        out.append(
            {
                "id": aid_no_ver,
                "id_versioned": aid,
                "category": cat,
                "license": lic,
            }
        )
    return out


def parse_recent_list_ids(html_text: str) -> list[str]:
    ids = re.findall(r"/abs/(\d{4}\.\d{4,5})(?:v\d+)?", html_text)
    out = []
    seen = set()
    for aid in ids:
        if aid in seen:
            continue
        seen.add(aid)
        out.append(aid)
    return out


def query_category_entries(client: ArxivClient, rng: random.Random, cat: str) -> tuple[list[dict], int]:
    queries = 0
    start = rng.randint(0, 400)
    url = (
        "https://export.arxiv.org/api/query?"
        f"search_query=cat:{urllib.parse.quote(cat)}&sortBy=submittedDate&"
        f"sortOrder=descending&start={start}&max_results=25"
    )
    try:
        feed = client.fetch_bytes(url, max_attempts=1)
        queries += 1
        rows = parse_arxiv_feed(feed, fallback_category=cat)
        if rows:
            return rows, queries
    except Exception:
        pass
    return [], queries


def collect_candidate_entries(client: ArxivClient, rng: random.Random, target_ids: int) -> tuple[list[dict], int]:
    per_cat = max(4, math.ceil(target_ids / len(CATEGORIES)))
    entries_by_cat: dict[str, list[dict]] = {}
    queries = 0

    categories = CATEGORIES[:]
    rng.shuffle(categories)

    for cat in categories:
        rows, q = query_category_entries(client, rng, cat)
        queries += q
        rng.shuffle(rows)
        dedup = []
        seen_local = set()
        for row in rows:
            if row["id"] in seen_local:
                continue
            seen_local.add(row["id"])
            dedup.append(row)
        entries_by_cat[cat] = dedup[: max(per_cat, 8)]

    entries: list[dict] = []
    seen_global: set[str] = set()
    added = True
    while added and len(entries) < target_ids:
        added = False
        for cat in categories:
            bucket = entries_by_cat.get(cat) or []
            if not bucket:
                continue
            row = bucket.pop(0)
            if row["id"] in seen_global:
                continue
            seen_global.add(row["id"])
            entries.append(row)
            added = True
            if len(entries) >= target_ids:
                break

    if len(entries) < target_ids:
        for cat in categories:
            rows, q = query_category_entries(client, rng, cat)
            queries += q
            for row in rows:
                if len(entries) >= target_ids:
                    break
                if row["id"] in seen_global:
                    continue
                seen_global.add(row["id"])
                entries.append(row)
            if len(entries) >= target_ids:
                break

    return entries, queries


def load_cached_entries(rng: random.Random, limit: int) -> list[dict]:
    out = []
    for p in CACHE_ROOT.glob("*/entry.json"):
        try:
            row = json.loads(p.read_text())
        except Exception:
            continue
        aid = (row.get("id") or p.parent.name).strip()
        if not NEW_STYLE_ID_RE.fullmatch(aid):
            continue
        out.append(
            {
                "id": aid,
                "id_versioned": row.get("id_versioned", aid),
                "category": row.get("category", "unknown"),
                "license": row.get("license", "unknown"),
            }
        )
    rng.shuffle(out)
    return out[:limit]


def merge_entries_unique(primary: list[dict], secondary: list[dict], max_items: int) -> list[dict]:
    out = []
    seen = set()
    for row in primary + secondary:
        aid = row["id"]
        if aid in seen:
            continue
        seen.add(aid)
        out.append(row)
        if len(out) >= max_items:
            break
    return out


def ensure_source_tree(src_blob_path: Path, source_dir: Path) -> bool:
    marker = source_dir / ".ready"
    if marker.exists():
        return True
    if source_dir.exists():
        shutil.rmtree(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)

    raw = src_blob_path.read_bytes()
    decoded_candidates: list[bytes] = []
    if raw.startswith(b"\x1f\x8b"):
        try:
            decoded_candidates.append(gzip.decompress(raw))
        except OSError:
            pass
    decoded_candidates.append(raw)

    for blob in decoded_candidates:
        if safe_extract_tar_bytes(blob, source_dir):
            marker.write_text("ok\n")
            return True

    for blob in decoded_candidates:
        txt = maybe_decode_text(blob)
        if txt is None:
            continue
        (source_dir / "main.tex").write_text(txt)
        marker.write_text("ok\n")
        return True

    return False


def load_tex_files(source_dir: Path) -> dict[Path, str]:
    out = {}
    for p in source_dir.rglob("*.tex"):
        try:
            raw = p.read_text()
        except UnicodeDecodeError:
            raw = p.read_text(encoding="latin-1", errors="ignore")
        out[p.resolve()] = strip_comments(raw)
    return out


def load_style_files(source_dir: Path) -> dict[Path, str]:
    out = {}
    for p in source_dir.rglob("*.sty"):
        try:
            raw = p.read_text()
        except UnicodeDecodeError:
            raw = p.read_text(encoding="latin-1", errors="ignore")
        out[p.resolve()] = strip_comments(raw)
    return out


def choose_main_tex(tex_map: dict[Path, str]) -> Path | None:
    cands = []
    for p, txt in tex_map.items():
        if "\\begin{document}" in txt:
            cands.append((p, txt))
    if not cands:
        return None

    def score(row: tuple[Path, str]) -> tuple[int, int, int, int]:
        p, txt = row
        return (
            1 if "\\documentclass" in txt else 0,
            1 if p.name.lower() in {"main.tex", "ms.tex", "paper.tex"} else 0,
            -len(p.parts),
            len(txt),
        )

    cands.sort(key=score, reverse=True)
    return cands[0][0]


def resolve_include(name: str, base_dir: Path, tex_map: dict[Path, str]) -> Path | None:
    n = name.strip()
    if not n:
        return None
    candidates = []
    p = Path(n)
    if p.suffix:
        candidates.append((base_dir / p).resolve())
    else:
        candidates.append((base_dir / p).with_suffix(".tex").resolve())
        candidates.append((base_dir / p).resolve())
    for c in candidates:
        if c in tex_map:
            return c
    target_name = p.name if p.suffix else p.name + ".tex"
    fallback = [k for k in tex_map if k.name == target_name]
    if len(fallback) == 1:
        return fallback[0]
    return None


def expand_inputs(path: Path, tex_map: dict[Path, str], stack: set[Path] | None = None) -> str:
    if stack is None:
        stack = set()
    if path in stack:
        return ""
    stack.add(path)
    txt = tex_map.get(path, "")
    out = []
    cur = 0
    for m in INPUT_RE.finditer(txt):
        out.append(txt[cur : m.start()])
        cur = m.end()
        target = resolve_include(m.group(1), path.parent, tex_map)
        if target is not None:
            out.append("\n")
            out.append(expand_inputs(target, tex_map, stack=stack))
            out.append("\n")
    out.append(txt[cur:])
    stack.remove(path)
    return "".join(out)


def between_document(text: str) -> str | None:
    m1 = re.search(r"\\begin\{document\}", text)
    if not m1:
        return None
    m2 = re.search(r"\\end\{document\}", text[m1.end() :])
    if not m2:
        return text[m1.end() :]
    return text[m1.end() : m1.end() + m2.start()]


def remove_skip_regions(text: str) -> str:
    out = text
    for env in SKIP_ENVS:
        pat = re.compile(
            rf"\\begin\{{{re.escape(env)}\}}.*?\\end\{{{re.escape(env)}\}}",
            re.DOTALL,
        )
        out = pat.sub("\n\n", out)
    out = re.sub(r"\\section\*\s*\{[^{}]*\}", "\n\n", out)
    out = re.sub(r"\\caption(?:\[[^\]]*\])?\s*\{[^{}]*\}", "", out)
    return out


def read_balanced_group(text: str, brace_open_idx: int) -> tuple[str | None, int]:
    if brace_open_idx >= len(text) or text[brace_open_idx] != "{":
        return None, brace_open_idx
    depth = 0
    i = brace_open_idx
    start = i + 1
    while i < len(text):
        ch = text[i]
        if ch == "{" and not is_escaped(text, i):
            depth += 1
        elif ch == "}" and not is_escaped(text, i):
            depth -= 1
            if depth == 0:
                return text[start:i], i + 1
        i += 1
    return None, brace_open_idx


def unwrap_simple_command(text: str, cmd: str) -> str:
    token = "\\" + cmd
    out = []
    i = 0
    n = len(text)
    while i < n:
        j = text.find(token, i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        k = j + len(token)
        while k < n and text[k].isspace():
            k += 1
        if k < n and text[k] == "{":
            inner, nxt = read_balanced_group(text, k)
            if inner is not None:
                out.append(inner)
                i = nxt
                continue
        out.append(text[j : j + len(token)])
        i = j + len(token)
    return "".join(out)


def extract_custom_macros(preamble: str) -> set[str]:
    out = set()
    for m in re.finditer(r"\\newcommand\*?\s*(?:\{\\([A-Za-z@]+)\}|\\([A-Za-z@]+))", preamble):
        out.add(m.group(1) or m.group(2))
    for m in re.finditer(r"\\def\s*\\([A-Za-z@]+)\b", preamble):
        out.add(m.group(1))
    for m in re.finditer(r"\\DeclareMathOperator\*?\s*\{\\([A-Za-z@]+)\}", preamble):
        out.add(m.group(1))
    return out


@dataclasses.dataclass
class MathFrag:
    start: int
    end: int
    kind: str  # inline | display
    mode: str  # dollar | ddollar | paren | brack | env
    inner: str
    raw: str


def find_closing_inline_dollar(text: str, start: int) -> int:
    i = start
    while True:
        j = text.find("$", i)
        if j < 0:
            return -1
        if is_escaped(text, j):
            i = j + 1
            continue
        if j + 1 < len(text) and text[j + 1] == "$":
            i = j + 2
            continue
        return j


def find_closing_double_dollar(text: str, start: int) -> int:
    i = start
    while True:
        j = text.find("$$", i)
        if j < 0:
            return -1
        if is_escaped(text, j):
            i = j + 2
            continue
        return j


def parse_math_fragments(text: str) -> list[MathFrag] | None:
    out: list[MathFrag] = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("\\begin{", i):
            j = text.find("}", i + 7)
            if j > 0:
                env = text[i + 7 : j]
                if env in MATH_ENVS:
                    end_tok = f"\\end{{{env}}}"
                    k = text.find(end_tok, j + 1)
                    if k < 0:
                        return None
                    inner = text[j + 1 : k]
                    raw = text[i : k + len(end_tok)]
                    out.append(MathFrag(i, k + len(end_tok), "display", "env", inner, raw))
                    i = k + len(end_tok)
                    continue
        if text.startswith("\\(", i):
            j = text.find("\\)", i + 2)
            if j < 0:
                return None
            inner = text[i + 2 : j]
            raw = text[i : j + 2]
            out.append(MathFrag(i, j + 2, "inline", "paren", inner, raw))
            i = j + 2
            continue
        if text.startswith("\\[", i):
            j = text.find("\\]", i + 2)
            if j < 0:
                return None
            inner = text[i + 2 : j]
            raw = text[i : j + 2]
            out.append(MathFrag(i, j + 2, "display", "brack", inner, raw))
            i = j + 2
            continue
        if text[i] == "$" and not is_escaped(text, i):
            if i + 1 < n and text[i + 1] == "$":
                j = find_closing_double_dollar(text, i + 2)
                if j < 0:
                    return None
                inner = text[i + 2 : j]
                raw = text[i : j + 2]
                out.append(MathFrag(i, j + 2, "display", "ddollar", inner, raw))
                i = j + 2
                continue
            j = find_closing_inline_dollar(text, i + 1)
            if j < 0:
                return None
            inner = text[i + 1 : j]
            raw = text[i : j + 1]
            out.append(MathFrag(i, j + 1, "inline", "dollar", inner, raw))
            i = j + 1
            continue
        i += 1
    return out


def strip_drop_environment_lines(text: str) -> str:
    out_lines = []
    for line in text.splitlines():
        m = DROP_ENV_LINE_RE.match(line)
        if m is not None and m.group(1).lower() in DROP_ENV_NAMES:
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def normalize_paragraph_commands(text: str) -> str:
    out = text
    out = DROP_ENV_CMD_RE.sub(" ", out)
    out = re.sub(r"\\(?:sub)?paragraph\*?\s*\{([^{}]*)\}\s*", r"\1 ", out)
    out = re.sub(r"\\href\s*\{[^{}]*\}\s*\{([^{}]*)\}", r"\1", out)
    out = re.sub(r"\\url\s*\{([^{}]*)\}", r"\1", out)
    out = FOOTNOTEMARK_RE.sub("", out)
    out = SPACING_CMD_RE.sub(" ", out)
    out = VSPACE_CMD_RE.sub(" ", out)
    for cmd in ("textsc", "textrm", "underline", "texttt"):
        prev = None
        cur = out
        while prev != cur:
            prev = cur
            cur = unwrap_simple_command(cur, cmd)
        out = cur
    return out


def strip_math_content(text: str) -> str:
    frags = parse_math_fragments(text)
    if frags is None:
        return text
    out = []
    cur = 0
    for frag in frags:
        out.append(text[cur : frag.start])
        out.append(" ")
        cur = frag.end
    out.append(text[cur:])
    return "".join(out)


def has_backslash_command_outside_math(text: str) -> bool:
    outside = strip_math_content(text)
    return BACKSLASH_CMD_RE.search(outside) is not None


def math_fragment_has_real_signal(inner: str) -> bool:
    s = inner.strip()
    if not s:
        return False
    s = re.sub(r"\\left|\\right", " ", s)
    s = re.sub(r"\s+", "", s)
    if not s:
        return False

    if re.fullmatch(r"[+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?", s):
        return False
    if re.fullmatch(r"[A-Za-z]", s):
        return False
    if re.fullmatch(r"[A-Za-z]_(?:\d+|\{\d+\})", s):
        return False

    if re.search(r"[=<>≤≥+\-*/^_]", s):
        return True
    if re.search(
        r"\\(?:frac|sum|int|prod|lim|partial|nabla|sqrt|mathbb|mathcal|hat|bar|vec|infty|to|le|ge|neq|approx|sim|in|subset|cdot|times)\b",
        s,
    ):
        return True
    if any(re.search(rf"\\{name}\b", s) for name in GREEK_CMD_NAMES):
        return True
    if re.search(r"[A-Za-z]\s*\([^\)]*\)", inner):
        return True
    return False


def paragraph_has_real_math(frags: list[MathFrag]) -> bool:
    return any(math_fragment_has_real_signal(frag.inner) for frag in frags)


def apply_prose_transforms(text: str) -> str:
    out = text
    out = strip_drop_environment_lines(out)
    out = normalize_paragraph_commands(out)
    out = out.replace("~", " ")
    out = CITE_RE.sub("[ref]", out)
    out = REF_RE.sub("(ref)", out)
    out = LABEL_RE.sub("", out)
    for cmd in ("emph", "textit", "textbf", "text"):
        prev = None
        cur = out
        while prev != cur:
            prev = cur
            cur = unwrap_simple_command(cur, cmd)
        out = cur
    out = out.replace(r"\%", "%")
    out = out.replace(r"\&", "&")
    out = out.replace(r"\,", " ")
    out = re.sub(r"\\\\(?:\[[^\]]*\])?", "\n", out)
    out = out.replace("---", "—")
    out = out.replace("--", "–")
    out = out.replace(r"\ldots", "…")
    out = out.replace(r"\dots", "…")
    out = re.sub(r"``([^`]*)''", r"“\1”", out)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r" *\n *", "\n", out)
    return out.strip()


def normalize_math_fragment(frag: MathFrag) -> str:
    inner = frag.inner
    if frag.mode == "env":
        inner = LABEL_RE.sub("", inner)
    inner = inner.strip()
    if frag.kind == "inline":
        return f"\\( {inner} \\)"
    return f"\\[ {inner} \\]"


def build_reference_from_tex(paragraph: str, frags: list[MathFrag]) -> str:
    out = []
    cur = 0
    for frag in frags:
        prose = paragraph[cur : frag.start]
        out.append(apply_prose_transforms(prose))
        out.append(normalize_math_fragment(frag))
        cur = frag.end
    out.append(apply_prose_transforms(paragraph[cur:]))
    text = " ".join(part for part in out if part)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def cleanup_candidate_paragraph(paragraph: str) -> str | None:
    out = strip_drop_environment_lines(paragraph).strip()
    if not out:
        return None
    if SECTION_START_RE.match(out):
        return None
    out = normalize_paragraph_commands(out)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r" *\n *", "\n", out)
    out = out.strip()
    if not out:
        return None
    return out


def reference_without_math(reference: str) -> str:
    text = re.sub(r"\\\((.*?)\\\)", " ", reference, flags=re.DOTALL)
    text = re.sub(r"\\\[(.*?)\\\]", " ", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def reference_cleanup_ok(reference: str) -> bool:
    if not reference.strip():
        return False
    if SECTION_START_RE.match(reference):
        return False
    if re.search(r"\\(?:sub)?paragraph\*?\s*\{", reference):
        return False
    if re.search(r"\\(?:sub)?section\*?\s*\{", reference):
        return False
    if re.search(r"\\(?:auto|[cC])ref\*?\s*\{", reference):
        return False
    if has_backslash_command_outside_math(reference):
        return False
    return True


def sentence_count(text: str) -> int:
    return len(re.findall(r"[.!?](?=\s|$)", text))


def paragraph_uses_custom_macros(frags: list[MathFrag], custom_macros: set[str]) -> bool:
    if not custom_macros:
        return False
    for frag in frags:
        for cmd in re.findall(r"\\([A-Za-z@]+)", frag.inner):
            if cmd in custom_macros:
                return True
    return False


def detect_candidate_paragraphs(expanded_doc: str, style_map: dict[Path, str] | None = None) -> tuple[list[dict], set[str]]:
    body = between_document(expanded_doc)
    if body is None:
        return [], set()

    docclass_idx = expanded_doc.find("\\begin{document}")
    preamble = expanded_doc[:docclass_idx] if docclass_idx >= 0 else expanded_doc
    custom_macros = extract_custom_macros(preamble)
    if style_map:
        for sty_txt in style_map.values():
            custom_macros.update(extract_custom_macros(sty_txt))

    body = remove_skip_regions(body)
    paras = [p.strip() for p in re.split(r"\n\s*\n+", body) if p.strip()]
    rows = []
    for idx, para in enumerate(paras):
        if "\\footnote" in para or "\\includegraphics" in para or "\\item" in para:
            continue
        cleaned_para = cleanup_candidate_paragraph(para)
        if cleaned_para is None:
            continue
        frags = parse_math_fragments(cleaned_para)
        if frags is None:
            continue
        n_math = len(frags)
        if n_math < 1 or n_math > 4:
            continue
        n_display = sum(1 for x in frags if x.kind == "display")
        if n_display > 2:
            continue
        n_inline = n_math - n_display
        if paragraph_uses_custom_macros(frags, custom_macros):
            continue
        if not paragraph_has_real_math(frags):
            continue
        reference = build_reference_from_tex(cleaned_para, frags)
        if not reference_cleanup_ok(reference):
            continue
        reference_prose = reference_without_math(reference)
        if not (150 <= len(reference_prose) <= 700):
            continue
        if sentence_count(reference_prose) < 2:
            continue
        pref = (2.0 if n_inline > 0 else 0.0) + (1.0 - 0.4 * n_display)
        rows.append(
            {
                "idx": idx,
                "paragraph": cleaned_para,
                "reference": reference,
                "reference_prose": reference_prose,
                "n_inline": n_inline,
                "n_display": n_display,
                "pref": pref,
            }
        )
    return rows, custom_macros


def run_pdftotext(pdf_path: Path, out_path: Path, layout: bool) -> str:
    if out_path.exists():
        return out_path.read_text()
    cmd = [PDFTOTEXT]
    if layout:
        cmd.append("-layout")
    cmd.extend([str(pdf_path), "-"])
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    out_path.write_text(proc.stdout)
    return proc.stdout


def tokenize_words(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0).lower(), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def alpha4_counter(text: str) -> collections.Counter:
    return collections.Counter(ALPHA4_RE.findall(text.lower()))


def bag_overlap_score(region: str, ref_prose: str) -> float:
    a = alpha4_counter(region)
    b = alpha4_counter(ref_prose)
    den = sum(b.values())
    if den == 0:
        return 0.0
    inter = 0
    for k, v in b.items():
        inter += min(v, a.get(k, 0))
    return inter / den


def anchor_score(words: list[str], i: int, anchor: list[str]) -> float:
    if i < 0 or i + len(anchor) > len(words):
        return 0.0
    s1 = " ".join(words[i : i + len(anchor)])
    s2 = " ".join(anchor)
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def candidate_anchor_positions(words: list[str], anchor: list[str], top_k: int = 48) -> list[tuple[float, int]]:
    if not words or not anchor:
        return []
    limit = len(words) - len(anchor) + 1
    if limit <= 0:
        return []
    positions = [i for i in range(limit) if words[i] == anchor[0]]
    if not positions:
        step = max(1, limit // 240)
        positions = list(range(0, limit, step))
    scored = [(anchor_score(words, i, anchor), i) for i in positions]
    scored.sort(reverse=True)
    top = scored[:top_k]
    if not top:
        top = [(0.0, 0)]
    return top


def extend_sentence_punct(text: str, end_char: int) -> int:
    i = end_char
    if i < len(text) and text[i] in ".!?":
        i += 1
        while i < len(text) and text[i] in "\"')]}":
            i += 1
    return i


def align_region(pdf_text: str, reference_prose: str) -> dict | None:
    toks = tokenize_words(pdf_text)
    if len(toks) < 12:
        return None
    words = [w for w, _, _ in toks]
    ref_words = [w.lower() for w in WORD_RE.findall(reference_prose)]
    if len(ref_words) < 12:
        return None

    anchor_n = 6
    if len(ref_words) < anchor_n * 2:
        return None
    start_anchor = ref_words[:anchor_n]
    end_anchor = ref_words[-anchor_n:]

    starts = candidate_anchor_positions(words, start_anchor, top_k=56)
    ends = candidate_anchor_positions(words, end_anchor, top_k=72)
    best_start_score = max((sc for sc, _ in starts), default=0.0)
    best_end_score = max((sc for sc, _ in ends), default=0.0)
    best = None
    max_span_words = max(anchor_n * 2, int(math.ceil(len(ref_words) * 2.5)))
    for s_sc, s in starts:
        for e_sc, e in ends:
            last = e + anchor_n - 1
            if last >= len(toks) or e <= s:
                continue
            span_words = last - s + 1
            if span_words > max_span_words:
                continue
            start_char = toks[s][1]
            end_char = extend_sentence_punct(pdf_text, toks[last][2])
            region = pdf_text[start_char:end_char].strip()
            if not region:
                continue
            overlap = bag_overlap_score(region, reference_prose)
            word_ratio = span_words / max(1, len(ref_words))
            if word_ratio <= 0 or word_ratio > 2.5:
                continue
            score = overlap + 0.6 * s_sc + 0.6 * e_sc - 0.08 * abs(word_ratio - 1.0)
            if best is None or score > best["score"]:
                best = {
                    "region": region,
                    "overlap": overlap,
                    "ratio": word_ratio,
                    "start_score": s_sc,
                    "end_score": e_sc,
                    "best_start_score": best_start_score,
                    "best_end_score": best_end_score,
                    "span_words": span_words,
                    "score": score,
                }

    if best is None:
        return None
    if best["best_start_score"] < 0.85 or best["best_end_score"] < 0.85:
        return None
    if best["start_score"] < 0.85 or best["end_score"] < 0.85:
        return None
    if best["overlap"] < 0.75:
        return None
    if best["ratio"] > 2.5:
        return None
    return best


def two_column_heuristic(text: str) -> bool:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 70:
        return False
    short = sum(1 for ln in lines if len(ln) <= 58)
    short_ratio = short / len(lines)
    broken = 0
    pairs = 0
    for a, b in zip(lines, lines[1:]):
        if len(a) <= 75 and len(b) <= 75:
            pairs += 1
            if re.search(r"[A-Za-z0-9,;:]$", a) and re.match(r"^[a-z]", b):
                broken += 1
    broken_ratio = (broken / pairs) if pairs else 0.0
    return short_ratio >= 0.58 and broken_ratio >= 0.33


def choose_spread(cands: list[dict], k: int = 3) -> list[dict]:
    if len(cands) <= k:
        return sorted(cands, key=lambda x: x["idx"])
    ordered = sorted(cands, key=lambda x: x["idx"])
    lo = ordered[0]["idx"]
    hi = ordered[-1]["idx"]
    targets = [0.14, 0.5, 0.86][:k]
    picked = []
    used = set()
    for t in targets:
        goal = lo + t * (hi - lo)
        best = None
        for j, row in enumerate(ordered):
            if j in used:
                continue
            dist = abs(row["idx"] - goal)
            plain_overlap = (row.get("plain_hit") or {}).get("overlap", 0.0)
            layout_overlap = (row.get("layout_hit") or {}).get("overlap", 0.0)
            score = row["pref"] + max(float(plain_overlap), float(layout_overlap))
            cand = (dist, -score, j, row)
            if best is None or cand < best:
                best = cand
        if best is not None:
            used.add(best[2])
            picked.append(best[3])
    return sorted(picked, key=lambda x: x["idx"])


def sanitize_item_id(aid: str) -> str:
    return aid.replace("/", "-")


def load_existing_id_maps(out_path: Path) -> tuple[dict[tuple[str, str, str], str], dict[tuple[str, str], list[str]], set[str]]:
    by_exact: dict[tuple[str, str, str], str] = {}
    by_ref: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    used_ids: set[str] = set()
    if not out_path.exists():
        return by_exact, by_ref, used_ids
    try:
        rows = json.loads(out_path.read_text())
    except Exception:
        return by_exact, by_ref, used_ids
    if not isinstance(rows, list):
        return by_exact, by_ref, used_ids
    for it in rows:
        if not isinstance(it, dict):
            continue
        item_id = (it.get("id") or "").strip()
        if not item_id:
            continue
        meta = it.get("meta") if isinstance(it.get("meta"), dict) else {}
        paper_id = (meta.get("paper_id") or "").strip()
        ref = (it.get("reference") or "").strip()
        inp = (it.get("input") or "").strip()
        if paper_id and ref and inp:
            by_exact[(paper_id, ref, inp)] = item_id
        if paper_id and ref:
            by_ref[(paper_id, ref)].append(item_id)
        used_ids.add(item_id)
    return by_exact, by_ref, used_ids


def allocate_item_id(
    aid: str,
    ref: str,
    inp: str,
    used_ids: set[str],
    exact_map: dict[tuple[str, str, str], str],
    ref_map: dict[tuple[str, str], list[str]],
) -> str:
    key_exact = (aid, ref, inp)
    existing = exact_map.get(key_exact)
    if existing and existing not in used_ids:
        used_ids.add(existing)
        return existing

    key_ref = (aid, ref)
    candidates = [x for x in ref_map.get(key_ref, []) if x not in used_ids]
    if len(candidates) == 1:
        used_ids.add(candidates[0])
        return candidates[0]

    n = 1
    prefix = f"arxiv-{sanitize_item_id(aid)}-p"
    while f"{prefix}{n}" in used_ids:
        n += 1
    item_id = f"{prefix}{n}"
    used_ids.add(item_id)
    return item_id


def input_has_math_signal(inp: str) -> bool:
    return bool(MATH_HINT_RE.search(inp) or GREEK_RE.search(inp) or ALNUM_ADJ_RE.search(inp))


def reference_delimiters_ok(ref: str) -> bool:
    if "\\(" not in ref and "\\[" not in ref:
        return False
    if ref.count("\\(") != ref.count("\\)"):
        return False
    if ref.count("\\[") != ref.count("\\]"):
        return False
    return True


def paper_cache_dir(aid: str) -> Path:
    return CACHE_ROOT / aid


def load_or_write_entry_json(pdir: Path, entry: dict) -> dict:
    p = pdir / "entry.json"
    if p.exists():
        try:
            old = json.loads(p.read_text())
        except json.JSONDecodeError:
            old = {}
        old.update(entry)
        entry = old
    p.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n")
    return entry


def prepare_paper_assets(client: ArxivClient, entry: dict) -> tuple[dict, Path, Path, dict]:
    aid = entry["id"]
    pdir = paper_cache_dir(aid)
    pdir.mkdir(parents=True, exist_ok=True)
    entry = load_or_write_entry_json(pdir, entry)

    pdf_path = pdir / "paper.pdf"
    src_blob = pdir / "source.bin"
    downloaded = {"pdf": False, "source": False}

    if not pdf_path.exists():
        client.fetch_bytes(f"https://export.arxiv.org/pdf/{aid}", cache_path=pdf_path)
        downloaded["pdf"] = True
    if not src_blob.exists():
        client.fetch_bytes(f"https://export.arxiv.org/e-print/{aid}", cache_path=src_blob)
        downloaded["source"] = True

    return entry, pdf_path, src_blob, downloaded


def ensure_text_extractions(pdf_path: Path, pdir: Path) -> tuple[str, str]:
    plain_path = pdir / "pdftotext.txt"
    layout_path = pdir / "pdftotext-layout.txt"
    plain = run_pdftotext(pdf_path, plain_path, layout=False)
    layout = run_pdftotext(pdf_path, layout_path, layout=True)
    return plain, layout


def build_items(
    seed: int,
    target: int,
    out_path: Path,
    summary_path: Path,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    client = ArxivClient(min_interval=3.0)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    input_seen: set[str] = set()
    desired_layout = int(round(target * 0.25))
    layout_count = 0
    existing_exact, existing_ref, _prior_used_ids = load_existing_id_maps(out_path)
    used_item_ids: set[str] = set()

    stats = {
        "queries": 0,
        "papers_considered": 0,
        "papers_with_items": 0,
        "downloaded_pdf": 0,
        "downloaded_source": 0,
        "skipped": collections.Counter(),
    }

    cached = load_cached_entries(rng, limit=400)
    candidate_entries = cached[:]
    pos = 0
    while len(items) < target:
        if pos >= len(candidate_entries):
            extra, q2 = collect_candidate_entries(client, rng, target_ids=40)
            stats["queries"] += q2
            candidate_entries = merge_entries_unique(candidate_entries, extra, max_items=400)
            if pos >= len(candidate_entries):
                break

        entry = candidate_entries[pos]
        pos += 1
        aid = entry["id"]
        pdir = paper_cache_dir(aid)
        stats["papers_considered"] += 1

        try:
            entry2, pdf_path, src_blob_path, dl = prepare_paper_assets(client, entry)
            if dl["pdf"]:
                stats["downloaded_pdf"] += 1
            if dl["source"]:
                stats["downloaded_source"] += 1
            license_value = entry2.get("license", "unknown")

            source_dir = pdir / "source"
            if not ensure_source_tree(src_blob_path, source_dir):
                stats["skipped"]["not_latex_source"] += 1
                eprint(f"{aid} cat={entry2.get('category','?')} license={license_value} kept=0")
                continue

            tex_map = load_tex_files(source_dir)
            if not tex_map:
                stats["skipped"]["no_tex_files"] += 1
                eprint(f"{aid} cat={entry2.get('category','?')} license={license_value} kept=0")
                continue
            style_map = load_style_files(source_dir)
            main_tex = choose_main_tex(tex_map)
            if main_tex is None:
                stats["skipped"]["no_document_tex"] += 1
                eprint(f"{aid} cat={entry2.get('category','?')} license={license_value} kept=0")
                continue
            expanded = expand_inputs(main_tex, tex_map)
            candidates, _custom_macros = detect_candidate_paragraphs(expanded, style_map=style_map)
            if not candidates:
                stats["skipped"]["no_candidate_paragraph"] += 1
                eprint(f"{aid} cat={entry2.get('category','?')} license={license_value} kept=0")
                continue

            pdf_plain, pdf_layout = ensure_text_extractions(pdf_path, pdir)
            two_col_plain = two_column_heuristic(pdf_plain)
            two_col_layout = two_column_heuristic(pdf_layout)

            aligned_rows = []
            for cand in candidates:
                plain_hit = align_region(pdf_plain, cand["reference_prose"])
                layout_hit = align_region(pdf_layout, cand["reference_prose"])
                if plain_hit is None and layout_hit is None:
                    continue

                row = {
                    **cand,
                    "plain_hit": plain_hit,
                    "layout_hit": layout_hit,
                }
                aligned_rows.append(row)

            if not aligned_rows:
                stats["skipped"]["no_aligned_paragraph"] += 1
                eprint(f"{aid} cat={entry2.get('category','?')} license={license_value} kept=0")
                continue

            selected = choose_spread(aligned_rows, k=3)
            kept_here = 0
            for row in selected:
                if len(items) >= target:
                    break

                remaining_slots = target - len(items)
                remaining_layout = desired_layout - layout_count
                plain_hit = row.get("plain_hit")
                layout_hit = row.get("layout_hit")
                choose_layout = False
                if remaining_layout <= 0:
                    if plain_hit is None:
                        continue
                    choose_layout = False
                elif remaining_layout >= remaining_slots:
                    if layout_hit is None:
                        continue
                    choose_layout = True
                elif layout_hit is not None and plain_hit is not None:
                    p_layout = remaining_layout / max(1, remaining_slots)
                    choose_layout = rng.random() < p_layout
                elif layout_hit is not None:
                    choose_layout = True
                elif plain_hit is not None:
                    choose_layout = False
                else:
                    continue

                chosen = layout_hit if choose_layout else plain_hit
                if chosen is None:
                    continue
                inp = (chosen.get("region") or "").strip()
                if not inp:
                    continue
                if inp in input_seen:
                    continue
                if not input_has_math_signal(inp):
                    continue
                if not reference_delimiters_ok(row["reference"]):
                    continue

                extractor = "pdftotext -layout" if choose_layout else "pdftotext"
                two_col = two_col_layout if choose_layout else two_col_plain
                if choose_layout:
                    layout_count += 1
                kept_here += 1
                item_id = allocate_item_id(
                    aid=aid,
                    ref=row["reference"],
                    inp=inp,
                    used_ids=used_item_ids,
                    exact_map=existing_exact,
                    ref_map=existing_ref,
                )
                item = {
                    "id": item_id,
                    "tier": "arxiv-paste",
                    "input": inp,
                    "reference": row["reference"],
                    "source": f"https://arxiv.org/abs/{aid}",
                    "meta": {
                        "paper_id": aid,
                        "category": entry2.get("category", "unknown"),
                        "license": entry2.get("license", "unknown"),
                        "extractor": extractor,
                        "two_column": bool(two_col),
                        "n_inline": row["n_inline"],
                        "n_display": row["n_display"],
                        "chars": len(inp),
                        "overlap": round(float(chosen["overlap"]), 4),
                        "reference_prose": row["reference_prose"],
                        "has_custom_macros": False,
                    },
                }
                items.append(item)
                input_seen.add(inp)

            if kept_here == 0:
                stats["skipped"]["none_selected_after_filters"] += 1
            else:
                stats["papers_with_items"] += 1
            eprint(f"{aid} cat={entry2.get('category','?')} license={license_value} kept={kept_here}")
        except subprocess.CalledProcessError:
            stats["skipped"]["pdftotext_error"] += 1
            eprint(f"{aid} cat={entry.get('category','?')} license={entry.get('license','unknown')} kept=0")
        except Exception:
            stats["skipped"]["paper_error"] += 1
            eprint(f"{aid} cat={entry.get('category','?')} license={entry.get('license','unknown')} kept=0")

    sanity_check_items(items, min_count=min(100, target))
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n")
    write_summary(summary_path, items, stats, seed=seed)
    return items, stats


def sanity_check_items(items: list[dict], min_count: int = 1) -> None:
    if not items:
        raise RuntimeError("no items built")
    if len(items) < min_count:
        raise RuntimeError(f"built only {len(items)} items; need at least {min_count}")
    seen_inputs = set()
    for it in items:
        inp = (it.get("input") or "").strip()
        if not inp:
            raise RuntimeError(f"{it.get('id')} has empty input")
        if not input_has_math_signal(inp):
            raise RuntimeError(f"{it.get('id')} input lacks math signal")
        ref = it.get("reference", "")
        if not reference_delimiters_ok(ref):
            raise RuntimeError(f"{it.get('id')} reference delimiters invalid")
        if has_backslash_command_outside_math(ref):
            raise RuntimeError(f"{it.get('id')} has command outside math after cleanup")
        ref_prose = (it.get("meta", {}) or {}).get("reference_prose") or reference_without_math(ref)
        aligned = align_region(inp, ref_prose)
        if aligned is None or aligned.get("region", "").strip() != inp:
            raise RuntimeError(f"{it.get('id')} input not tightly anchored to reference prose")
        if inp in seen_inputs:
            raise RuntimeError(f"duplicate input across items: {it.get('id')}")
        seen_inputs.add(inp)


def write_summary(summary_path: Path, items: list[dict], stats: dict, seed: int) -> None:
    by_cat = collections.Counter(it["meta"].get("category", "unknown") for it in items)
    by_disp = collections.Counter(it["meta"].get("n_display", 0) for it in items)
    by_ext = collections.Counter(it["meta"].get("extractor", "unknown") for it in items)
    mean_chars = sum(it["meta"].get("chars", 0) for it in items) / len(items)
    custom_count = sum(1 for it in items if it["meta"].get("has_custom_macros"))
    rng = random.Random(seed + 17)
    sample_n = min(3, len(items))
    examples = rng.sample(items, k=sample_n)

    lines = []
    lines.append("# arXiv paste benchmark summary")
    lines.append("")
    lines.append(f"- items: {len(items)}")
    lines.append(f"- mean chars: {mean_chars:.1f}")
    lines.append(f"- custom-macro math items: {custom_count}")
    lines.append(f"- api queries: {stats['queries']}")
    lines.append(f"- papers considered: {stats['papers_considered']}")
    lines.append(f"- papers with items: {stats['papers_with_items']}")
    lines.append(f"- downloaded pdf: {stats['downloaded_pdf']}")
    lines.append(f"- downloaded source: {stats['downloaded_source']}")
    lines.append("")
    lines.append("## by category")
    for cat, n in sorted(by_cat.items()):
        lines.append(f"- {cat}: {n}")
    lines.append("")
    lines.append("## by n_display")
    for n_disp, n in sorted(by_disp.items()):
        lines.append(f"- {n_disp}: {n}")
    lines.append("")
    lines.append("## by extractor")
    for ext, n in sorted(by_ext.items()):
        lines.append(f"- {ext}: {n}")
    lines.append("")
    lines.append("## skipped papers")
    skipped = stats["skipped"]
    if skipped:
        for reason, n in sorted(skipped.items()):
            lines.append(f"- {reason}: {n}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## examples")
    for it in examples:
        lines.append("")
        lines.append(f"### {it['id']}")
        lines.append(f"source: {it['source']}")
        lines.append("")
        lines.append("input:")
        lines.append("")
        lines.append("```")
        lines.append(it["input"])
        lines.append("```")
        lines.append("")
        lines.append("reference:")
        lines.append("")
        lines.append("```")
        lines.append(it["reference"])
        lines.append("```")
    summary_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", type=int, default=120)
    ap.add_argument("--out", default=str(ROOT / "arxiv-pastes.json"))
    ap.add_argument("--summary", default=str(SUMMARY_DEFAULT))
    args = ap.parse_args()

    out_path = Path(args.out)
    summary_path = Path(args.summary)
    items, _stats = build_items(seed=args.seed, target=args.target, out_path=out_path, summary_path=summary_path)
    print(f"wrote {out_path} ({len(items)} items)")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
