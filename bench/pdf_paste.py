#!/usr/bin/env python3
"""Render LaTeX passages to PDF and recover copy-paste text with math spans."""
import argparse
import collections
import html
import json
import pathlib
import random
import re
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PDFTOTEXT = "/opt/homebrew/bin/pdftotext"
KATEX_DIR = ROOT / "vendor" / "katex"
KATEX_CSS = KATEX_DIR / "katex.min.css"
KATEX_JS = KATEX_DIR / "katex.min.js"
KATEX_AUTORENDER_JS = KATEX_DIR / "contrib" / "auto-render.min.js"

# Primary sentinels tested with pdftotext. Fallbacks are here in case a local
# toolchain drops the primary pair.
SENTINEL_CANDIDATES = [("⦃", "⦄"), ("⟦", "⟧"), ("⟪", "⟫")]
DIRTY_KNOBS = (
    "two_column",
    "narrow_justified",
    "running_head",
    "footnote",
    "citations",
    "ligatures",
    "layout_mode",
)
MATH_RE = re.compile(r"\\\((.+?)\\\)|\\\[(.+?)\\\]", re.DOTALL)
SENTENCE_END_RE = re.compile(r"[^.!?]+[.!?]")
RAW_LATEX_RE = re.compile(r"\\(?:frac|sum)\b")
PROSE_RUN_RE = re.compile(r"\b[a-z]{4,}(?: [a-z]{4,}){2,}\b")

HTML_TEMPLATE_CLEAN = """<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="{katex_css_url}">
<script src="{katex_js_url}"></script>
<script src="{katex_autorender_js_url}"></script>
<style>
 @page {{ margin: 0.95in; }}
 body {{
   margin: 0;
   color: #111;
   background: white;
   font-family: Georgia, 'Times New Roman', serif;
   font-size: 11.5pt;
   line-height: 1.35;
 }}
 .page {{
   min-height: 9in;
   page-break-after: always;
 }}
 .page:last-child {{
   page-break-after: auto;
 }}
 .passage {{
   white-space: pre-wrap;
 }}
</style></head><body>
{body}
<script>renderMathInElement(document.body, {{delimiters:[
  {{left:'\\\\[',right:'\\\\]',display:true}},{{left:'\\\\(',right:'\\\\)',display:false}}], throwOnError:false}});</script>
</body></html>"""

HTML_TEMPLATE_DIRTY = """<!doctype html><html lang="{lang}"><head><meta charset="utf-8">
<link rel="stylesheet" href="{katex_css_url}">
<script src="{katex_js_url}"></script>
<script src="{katex_autorender_js_url}"></script>
<style>
 @page {{ size: {page_size}; margin: {page_margin}; }}
 body {{
   margin: 0;
   color: #111;
   background: white;
   font-family: {font_family};
   font-size: 11.5pt;
   line-height: 1.35;
   {body_extra}
 }}
 .wrap {{
   white-space: pre-wrap;
   {passage_extra}
 }}
 .running-head {{
   position: fixed;
   top: 0.1in;
   left: 0;
   right: 0;
   text-align: center;
   font-size: 8.5pt;
   letter-spacing: 0.04em;
   font-variant: small-caps;
   color: #444;
 }}
 .running-foot {{
   position: fixed;
   bottom: 0.1in;
   left: 0;
   right: 0;
   text-align: center;
   font-size: 8.5pt;
   color: #444;
 }}
 .running-foot::after {{
   content: counter(page);
 }}
 .footnote-block {{
   position: fixed;
   left: 0;
   right: 0;
   bottom: 0.45in;
   border-top: 0.6px solid #888;
   padding-top: 0.09in;
   font-size: 8.4pt;
   line-height: 1.25;
   color: #222;
 }}
</style></head><body>
{head_html}
<div class="wrap">{body}</div>
{foot_html}
{footnum_html}
<script>renderMathInElement(document.body, {{delimiters:[
  {{left:'\\\\[',right:'\\\\]',display:true}},{{left:'\\\\(',right:'\\\\)',display:false}}], throwOnError:false}});</script>
</body></html>"""


def _non_ws_chars(text):
    return [ch for ch in text if not ch.isspace() and ch != "\u200b"]


def _katex_asset_urls():
    required = [KATEX_CSS, KATEX_JS, KATEX_AUTORENDER_JS]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing KaTeX assets: {', '.join(missing)}")
    return {
        "katex_css_url": KATEX_CSS.resolve().as_uri(),
        "katex_js_url": KATEX_JS.resolve().as_uri(),
        "katex_autorender_js_url": KATEX_AUTORENDER_JS.resolve().as_uri(),
    }


def _char_coverage(span_text, debris_counter, debris_total):
    if debris_total == 0:
        return 1.0
    span_counter = collections.Counter(_non_ws_chars(span_text))
    return _char_coverage_from_counter(span_counter, debris_counter, debris_total)


def _char_coverage_from_counter(span_counter, debris_counter, debris_total):
    if debris_total == 0:
        return 1.0
    covered = 0
    for ch, need in debris_counter.items():
        covered += min(span_counter.get(ch, 0), need)
    return covered / debris_total


def _next_line_end(text, start):
    if start >= len(text):
        return start
    nl = text.find("\n", start)
    if nl < 0:
        return len(text)
    return nl + 1


def _prose_sentence_ends(text):
    masked = list(text)
    for m in MATH_RE.finditer(text):
        s, e = m.span()
        for i in range(s, e):
            masked[i] = " "
    masked_text = "".join(masked)
    return [m.end() for m in SENTENCE_END_RE.finditer(masked_text)]


def _annotate_passage(passage, sent_l, sent_r):
    fragments = []
    out = []
    cursor = 0
    for m in MATH_RE.finditer(passage):
        start, end = m.span()
        if start > cursor:
            out.append(html.escape(passage[cursor:start]))
        if m.group(1) is not None:
            inner = m.group(1)
            math_text = f"\\({inner}\\)"
        else:
            inner = m.group(2)
            math_text = f"\\[{inner}\\]"
        out.append(html.escape(f"{sent_l}{math_text}{sent_r}"))
        fragments.append(inner.strip())
        cursor = end
    if cursor < len(passage):
        out.append(html.escape(passage[cursor:]))
    return "".join(out), fragments


def _apply_citations(text):
    markers = ["[12]", "[3, 7]"]
    out = []
    cursor = 0
    used = 0
    for end in _prose_sentence_ends(text):
        chunk = text[cursor:end]
        cursor = end
        if used < len(markers):
            chunk = f"{chunk}{markers[used]}"
            used += 1
        out.append(chunk)
    out.append(text[cursor:])
    return "".join(out)


def _apply_footnote(text):
    ends = _prose_sentence_ends(text)
    if not ends:
        return text + " ¹", "1. Footnote note: in print/PDF, this line is separated from body flow."
    end = ends[0]
    marked = text[:end] + "¹" + text[end:]
    note = "1. Footnote note: in print/PDF, this line is separated from body flow."
    return marked, note


def _dirty_page_style(knobs):
    page_size = "6.7in 9.3in"
    page_margin = "0.7in"
    passage_bits = []
    body_bits = []
    font_family = "Georgia, 'Times New Roman', serif"

    if "two_column" in knobs:
        page_size = "5.7in 8.0in"
        page_margin = "0.42in"
        passage_bits.append("column-count: 2;")
        passage_bits.append("column-gap: 1.5em;")
    if "narrow_justified" in knobs:
        passage_bits.append("max-width: 34em;")
        passage_bits.append("text-align: justify;")
        passage_bits.append("hyphens: auto;")
    if "running_head" in knobs:
        # Short pages force at least one page break through moderate passages.
        page_size = page_size.split()[0] + " 3.7in"
        page_margin = "0.65in 0.45in 0.75in 0.45in"
        body_bits.append("padding-top: 0.16in;")
        body_bits.append("padding-bottom: 0.2in;")
    if "footnote" in knobs:
        body_bits.append("padding-bottom: 1.2in;")
    if "ligatures" in knobs:
        # On this macOS + system font stack (Hoefler/Times), pdftotext has so far
        # emitted plain "fi"/"fl" sequences rather than U+FB01/U+FB02 ligature codepoints.
        font_family = '"Hoefler Text", "Times New Roman", serif'
        passage_bits.append('font-feature-settings: "liga" 1;')
        passage_bits.append("font-variant-ligatures: common-ligatures;")

    return {
        "page_size": page_size,
        "page_margin": page_margin,
        "body_extra": " ".join(body_bits),
        "passage_extra": " ".join(passage_bits),
        "font_family": font_family,
    }


def _build_clean_html(passages, sent_l, sent_r):
    pages = []
    expected = []
    assets = _katex_asset_urls()
    for passage in passages:
        rendered, fragments = _annotate_passage(passage, sent_l, sent_r)
        pages.append(f'<section class="page"><div class="passage">{rendered}</div></section>')
        expected.append(fragments)
    return HTML_TEMPLATE_CLEAN.format(body="\n".join(pages), **assets), expected


def _build_dirty_html(passage, sent_l, sent_r, knobs):
    mod = passage
    footnote_note = None
    if "citations" in knobs:
        mod = _apply_citations(mod)
    if "footnote" in knobs:
        mod, footnote_note = _apply_footnote(mod)
    rendered, expected = _annotate_passage(mod, sent_l, sent_r)
    style = _dirty_page_style(knobs)
    assets = _katex_asset_urls()

    head_html = '<div class="running-head">Proceedings Note - Sample Author</div>' if "running_head" in knobs else ""
    footnum_html = '<div class="running-foot"></div>' if "running_head" in knobs else ""
    foot_html = f'<div class="footnote-block">{html.escape(footnote_note)}</div>' if footnote_note else ""

    html_text = HTML_TEMPLATE_DIRTY.format(
        lang="en",
        page_size=style["page_size"],
        page_margin=style["page_margin"],
        font_family=style["font_family"],
        body_extra=style["body_extra"],
        passage_extra=style["passage_extra"],
        head_html=head_html,
        foot_html=foot_html,
        footnum_html=footnum_html,
        body=rendered,
        **assets,
    )
    return html_text, expected


def _latex_from_passage(passage):
    out = []
    for m in MATH_RE.finditer(passage):
        out.append((m.group(1) if m.group(1) is not None else m.group(2)).strip())
    return out


def _build_formula_debris_html(formulas):
    pages = []
    assets = _katex_asset_urls()
    for latex in formulas:
        payload = html.escape(f"\\( {latex} \\)")
        pages.append(f'<section class="page"><div class="passage">{payload}</div></section>')
    return HTML_TEMPLATE_CLEAN.format(body="\n".join(pages), **assets)


def _cache_key(latex, layout):
    return (latex, bool(layout))


def _build_debris_cache(formulas, *, layout, chrome, pdftotext, cache):
    pending = []
    seen = set()
    for latex in formulas:
        if not latex:
            continue
        key = _cache_key(latex, layout)
        if key in cache or key in seen:
            continue
        seen.add(key)
        pending.append(latex)
    if not pending:
        return

    with tempfile.TemporaryDirectory() as td:
        tdir = pathlib.Path(td)
        source = tdir / "debris.html"
        pdf = tdir / "debris.pdf"
        source.write_text(_build_formula_debris_html(pending))
        _run_chrome_to_pdf(source, pdf, chrome=chrome)
        out = _run_pdftotext(pdf, pdftotext=pdftotext, layout=layout)
        pages = out.split("\f")
        if pages and not pages[-1].strip():
            pages = pages[:-1]
        for i, latex in enumerate(pending):
            page = pages[i] if i < len(pages) else ""
            chars = _non_ws_chars(page)
            cache[_cache_key(latex, layout)] = {
                "counter": collections.Counter(chars),
                "total": len(chars),
            }


def _looks_interleaved_math(frag):
    if not frag:
        return False
    if re.search(r"[.!?]", frag):
        return True
    words = re.findall(r"[A-Za-z]{5,}", frag)
    return len(words) >= 1 and len(frag.splitlines()) >= 2


def _has_prose_run(text):
    compact = re.sub(r"\s+", " ", text).strip()
    return bool(PROSE_RUN_RE.search(compact))


def _line_qualifies_for_extension(added_line, span_counter, debris_counter):
    chars = _non_ws_chars(added_line)
    if not chars:
        return False, collections.Counter()
    if _has_prose_run(added_line):
        return False, collections.Counter()
    line_counter = collections.Counter(chars)
    covered = 0
    for ch, n in line_counter.items():
        covered += min(n, debris_counter.get(ch, 0))
    if (covered / len(chars)) < 0.7:
        return False, line_counter
    adds_missing = any(
        line_counter[ch] > 0 and span_counter.get(ch, 0) < debris_counter.get(ch, 0)
        for ch in line_counter
    )
    return adds_missing, line_counter


def _render_assertions(page_texts, context):
    for i, page_text in enumerate(page_texts):
        m = RAW_LATEX_RE.search(page_text)
        if m:
            raise RuntimeError(
                f"KaTeX render failed ({context}, page={i}): raw command {m.group(0)} in extracted text"
            )


def _sentinel_failure_score(rec):
    expected = rec.get("expected", 0)
    left = rec.get("left", 0)
    right = rec.get("right", 0)
    found = rec.get("found", 0)
    partial = rec.get("partial_spans", 0)
    return (
        abs(expected - found) + abs(expected - left) + abs(expected - right),
        partial,
    )


def _extract_text(text, expected_latex, sent_l, sent_r, knobs, debris_cache, layout):
    page_text = text.replace("\f", "\n").strip()
    start_count = page_text.count(sent_l)
    end_count = page_text.count(sent_r)
    out = []
    spans = []
    out_len = 0
    cursor = 0
    found = 0
    reason = ""

    while cursor < len(page_text):
        left = page_text.find(sent_l, cursor)
        if left < 0:
            tail = page_text[cursor:]
            out.append(tail)
            out_len += len(tail)
            break

        head = page_text[cursor:left]
        out.append(head)
        out_len += len(head)
        right = page_text.find(sent_r, left + len(sent_l))
        if right < 0:
            reason = "unmatched_left_sentinel"
            tail = page_text[left + len(sent_l) :]
            out.append(tail)
            out_len += len(tail)
            break

        frag = page_text[left + len(sent_l) : right]
        span_start = out_len
        out.append(frag)
        out_len += len(frag)
        latex = expected_latex[found] if found < len(expected_latex) else ""
        span = {
            "start": span_start,
            "end": out_len,
            "latex": latex,
            "kind": "pdf",
            "_after_right": right + len(sent_r),
        }
        if "two_column" in knobs and _looks_interleaved_math(frag):
            span["interleaved"] = True
        spans.append(span)
        found += 1
        cursor = right + len(sent_r)

    expected = len(expected_latex)
    ok = (
        start_count == end_count
        and start_count == expected
        and found == expected
        and not reason
    )
    if not ok and not reason:
        reason = f"sentinel_count expected={expected} left={start_count} right={end_count} found={found}"

    output_text = "".join(out)

    partial_spans = 0
    extended_spans = 0
    for span_idx, span in enumerate(spans):
        latex = span.get("latex", "").strip()
        debris = debris_cache.get(_cache_key(latex, layout))
        if not latex or debris is None:
            continue

        start = span["start"]
        base_end = span["end"]
        span_counter = collections.Counter(_non_ws_chars(output_text[start:base_end]))
        coverage = _char_coverage_from_counter(
            span_counter,
            debris_counter=debris["counter"],
            debris_total=debris["total"],
        )
        if coverage >= 0.8:
            continue

        after_right = span["_after_right"]
        best_end = base_end
        best_cov = coverage
        cursor_after = after_right
        for _ in range(6):
            line_end = _next_line_end(page_text, cursor_after)
            if line_end <= cursor_after:
                break
            added_line = page_text[cursor_after:line_end]
            if sent_l in added_line:
                break
            ok_line, line_counter = _line_qualifies_for_extension(
                added_line,
                span_counter=span_counter,
                debris_counter=debris["counter"],
            )
            if not ok_line:
                break
            for ch, n in line_counter.items():
                span_counter[ch] += n
            extra_len = len(page_text[after_right:line_end])
            best_end = base_end + extra_len
            best_cov = _char_coverage_from_counter(
                span_counter,
                debris_counter=debris["counter"],
                debris_total=debris["total"],
            )
            cursor_after = line_end
            if best_cov >= 0.8:
                break

        if best_end > base_end and best_cov >= 0.8:
            span["end"] = best_end
            span["extended"] = True
            extended_spans += 1
        else:
            span["partial"] = True
            partial_spans += 1

    if partial_spans and (ok or not reason):
        ok = False
        reason = "partial span"

    interleaved_bad = 0
    for span_idx, span in enumerate(spans):
        assigned = span.get("latex", "").strip()
        if not assigned:
            continue
        piece = output_text[span["start"] : span["end"]]
        piece_counter = collections.Counter(_non_ws_chars(piece))
        assigned_debris = debris_cache.get(_cache_key(assigned, layout))
        if assigned_debris is None:
            continue
        assigned_cov = _char_coverage_from_counter(
            piece_counter,
            assigned_debris["counter"],
            assigned_debris["total"],
        )
        best_idx = span_idx
        best_cov = assigned_cov
        for j, cand_latex in enumerate(expected_latex):
            cand_latex = (cand_latex or "").strip()
            cand = debris_cache.get(_cache_key(cand_latex, layout))
            if cand is None:
                continue
            cov = _char_coverage_from_counter(piece_counter, cand["counter"], cand["total"])
            if cov > best_cov + 1e-9:
                best_cov = cov
                best_idx = j
        best_latex = (expected_latex[best_idx].strip() if best_idx < len(expected_latex) else "")
        if best_latex and best_latex != assigned and best_cov > assigned_cov + 1e-9:
            span["interleaved"] = True
            interleaved_bad += 1

    if interleaved_bad:
        ok = False
        if not reason or reason == "partial span":
            reason = "interleaved order"

    prose_bad = 0
    for span in spans:
        latex = span.get("latex", "")
        if "\\text" in latex:
            continue
        piece = output_text[span["start"] : span["end"]]
        if _has_prose_run(piece):
            span["prose_like"] = True
            span["partial"] = True
            prose_bad += 1

    if prose_bad:
        ok = False
        if not reason or reason in {"partial span", "interleaved order"}:
            reason = "prose span"
    partial_spans = sum(1 for sp in spans if sp.get("partial"))

    for span in spans:
        span.pop("_after_right", None)

    return {
        "text": output_text,
        "spans": spans,
        "ok": ok,
        "reason": reason,
        "expected": expected,
        "left": start_count,
        "right": end_count,
        "found": found,
        "extended_spans": extended_spans,
        "partial_spans": partial_spans,
        "interleaved_spans": interleaved_bad,
        "prose_spans": prose_bad,
    }


def _run_chrome_to_pdf(source, pdf, chrome):
    subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--virtual-time-budget=9000",
            f"--print-to-pdf={pdf}",
            f"file://{source}",
        ],
        check=True,
        capture_output=True,
    )


def _run_pdftotext(pdf, pdftotext, layout):
    cmd = [pdftotext]
    if layout:
        cmd.append("-layout")
    cmd.extend([str(pdf), "-"])
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def _choose_dirty_knobs(rng):
    n = rng.randint(2, 4)
    return sorted(rng.sample(DIRTY_KNOBS, k=n))


def _clean_batch(passages, layout, chrome, pdftotext):
    debris_cache = {}
    formulas = []
    for passage in passages:
        formulas.extend(_latex_from_passage(passage))
    _build_debris_cache(
        formulas,
        layout=layout,
        chrome=chrome,
        pdftotext=pdftotext,
        cache=debris_cache,
    )

    with tempfile.TemporaryDirectory() as td:
        tdir = pathlib.Path(td)
        source = tdir / "pages.html"
        pdf = tdir / "pages.pdf"
        best_results = None
        best_stats = None
        best_score = None
        for sent_l, sent_r in SENTINEL_CANDIDATES:
            html_text, expected = _build_clean_html(passages, sent_l, sent_r)
            source.write_text(html_text)
            _run_chrome_to_pdf(source, pdf, chrome=chrome)
            out = _run_pdftotext(pdf, pdftotext=pdftotext, layout=layout)
            pages = out.split("\f")
            if pages and not pages[-1].strip():
                pages = pages[:-1]
            _render_assertions(pages, context="clean")

            results = []
            failed = 0
            span_total = 0
            span_extended = 0
            span_partial = 0
            for i in range(len(passages)):
                page = pages[i] if i < len(pages) else ""
                rec = _extract_text(
                    page.strip(),
                    expected[i],
                    sent_l,
                    sent_r,
                    knobs=(),
                    debris_cache=debris_cache,
                    layout=layout,
                )
                rec["profile"] = "clean"
                rec["knobs"] = []
                if not rec["ok"]:
                    failed += 1
                span_total += len(rec["spans"])
                span_extended += rec.get("extended_spans", 0)
                span_partial += rec.get("partial_spans", 0)
                results.append(rec)

            stats = {
                "profile": "clean",
                "sentinel": [sent_l, sent_r],
                "failed": failed,
                "total": len(results),
                "drop_rate_by_knob": {},
                "span_completeness": {
                    "total_spans": span_total,
                    "extended_spans": span_extended,
                    "partial_spans": span_partial,
                    "extended_rate": (span_extended / span_total) if span_total else 0.0,
                    "partial_rate": (span_partial / span_total) if span_total else 0.0,
                },
            }
            score = (failed, span_partial)
            if best_score is None or score < best_score:
                best_score = score
                best_results = results
                best_stats = stats
            if failed == 0:
                return results, stats

        return best_results, best_stats


def _dirty_one(passage, knobs, chrome, pdftotext, debris_cache):
    use_layout = "layout_mode" in knobs
    _build_debris_cache(
        _latex_from_passage(passage),
        layout=use_layout,
        chrome=chrome,
        pdftotext=pdftotext,
        cache=debris_cache,
    )
    with tempfile.TemporaryDirectory() as td:
        tdir = pathlib.Path(td)
        source = tdir / "page.html"
        pdf = tdir / "page.pdf"
        best = None
        best_score = None
        for sent_l, sent_r in SENTINEL_CANDIDATES:
            html_text, expected = _build_dirty_html(passage, sent_l, sent_r, knobs=knobs)
            source.write_text(html_text)
            _run_chrome_to_pdf(source, pdf, chrome=chrome)
            out = _run_pdftotext(pdf, pdftotext=pdftotext, layout=use_layout)
            _render_assertions([out], context="dirty")
            rec = _extract_text(
                out,
                expected,
                sent_l,
                sent_r,
                knobs=knobs,
                debris_cache=debris_cache,
                layout=use_layout,
            )
            rec["profile"] = "dirty"
            rec["knobs"] = knobs
            score = _sentinel_failure_score(rec)
            if best_score is None or score < best_score:
                best_score = score
                best = (rec, (sent_l, sent_r))
            if rec["ok"]:
                return rec, (sent_l, sent_r)
        return best


def _dirty_batch(passages, seed, chrome, pdftotext, knob_rows=None):
    if knob_rows is None:
        rng = random.Random(seed)
        knob_rows = [_choose_dirty_knobs(rng) for _ in passages]
    else:
        knob_rows = [list(x) for x in knob_rows]
    if len(knob_rows) != len(passages):
        raise ValueError(
            f"knob_rows length mismatch: got {len(knob_rows)} for {len(passages)} passages"
        )
    debris_cache = {}
    formulas = []
    for passage in passages:
        formulas.extend(_latex_from_passage(passage))
    _build_debris_cache(
        formulas,
        layout=False,
        chrome=chrome,
        pdftotext=pdftotext,
        cache=debris_cache,
    )

    results = []
    failed = 0
    sentinel_counts = collections.Counter()
    knob_total = {k: 0 for k in DIRTY_KNOBS}
    knob_failed = {k: 0 for k in DIRTY_KNOBS}
    knob_span_total = {k: 0 for k in DIRTY_KNOBS}
    knob_span_extended = {k: 0 for k in DIRTY_KNOBS}
    knob_span_partial = {k: 0 for k in DIRTY_KNOBS}
    span_total = 0
    span_extended = 0
    span_partial = 0

    for passage, knobs in zip(passages, knob_rows):
        rec, sent = _dirty_one(
            passage,
            knobs=knobs,
            chrome=chrome,
            pdftotext=pdftotext,
            debris_cache=debris_cache,
        )
        sentinel_counts[sent] += 1
        rec_span_total = len(rec["spans"])
        rec_span_extended = rec.get("extended_spans", 0)
        rec_span_partial = rec.get("partial_spans", 0)
        span_total += rec_span_total
        span_extended += rec_span_extended
        span_partial += rec_span_partial
        for knob in knobs:
            knob_total[knob] += 1
            knob_span_total[knob] += rec_span_total
            knob_span_extended[knob] += rec_span_extended
            knob_span_partial[knob] += rec_span_partial
            if not rec["ok"]:
                knob_failed[knob] += 1
        if not rec["ok"]:
            failed += 1
        results.append(rec)

    drop_rate = {}
    for knob in DIRTY_KNOBS:
        total = knob_total[knob]
        bad = knob_failed[knob]
        drop_rate[knob] = {
            "total": total,
            "failed": bad,
            "drop_rate": (bad / total) if total else 0.0,
            "span_total": knob_span_total[knob],
            "extended_spans": knob_span_extended[knob],
            "partial_spans": knob_span_partial[knob],
            "extended_rate": (
                knob_span_extended[knob] / knob_span_total[knob]
            )
            if knob_span_total[knob]
            else 0.0,
            "partial_rate": (
                knob_span_partial[knob] / knob_span_total[knob]
            )
            if knob_span_total[knob]
            else 0.0,
        }
    stats = {
        "profile": "dirty",
        "sentinel": list(sentinel_counts.most_common(1)[0][0]) if sentinel_counts else None,
        "sentinel_counts": [
            {"sentinel": [pair[0], pair[1]], "count": count}
            for pair, count in sentinel_counts.most_common()
        ],
        "failed": failed,
        "total": len(results),
        "drop_rate_by_knob": drop_rate,
        "span_completeness": {
            "total_spans": span_total,
            "extended_spans": span_extended,
            "partial_spans": span_partial,
            "extended_rate": (span_extended / span_total) if span_total else 0.0,
            "partial_rate": (span_partial / span_total) if span_total else 0.0,
        },
    }
    return results, stats


def pdf_paste(
    passages,
    profile="clean",
    layout=False,
    seed=0,
    chrome=CHROME,
    pdftotext=PDFTOTEXT,
    knob_rows=None,
):
    if profile not in {"clean", "dirty"}:
        raise ValueError(f"unknown profile: {profile}")
    if not passages:
        return [], {
            "profile": profile,
            "sentinel": None,
            "failed": 0,
            "total": 0,
            "drop_rate_by_knob": {},
            "span_completeness": {
                "total_spans": 0,
                "extended_spans": 0,
                "partial_spans": 0,
                "extended_rate": 0.0,
                "partial_rate": 0.0,
            },
        }

    if profile == "clean":
        return _clean_batch(passages, layout=layout, chrome=chrome, pdftotext=pdftotext)
    return _dirty_batch(
        passages,
        seed=seed,
        chrome=chrome,
        pdftotext=pdftotext,
        knob_rows=knob_rows,
    )


def _load_multiline_refs(path):
    items = json.loads(path.read_text())
    out = []
    for it in items:
        if it.get("tier") == "multiline":
            out.append({"id": it["id"], "reference": it["reference"]})
    return out


def _side_by_side(left, right, width=60):
    left_lines = left.splitlines() or [""]
    right_lines = right.splitlines() or [""]
    rows = []
    n = max(len(left_lines), len(right_lines))
    for i in range(n):
        l = left_lines[i] if i < len(left_lines) else ""
        r = right_lines[i] if i < len(right_lines) else ""
        rows.append(f"{l[:width]:<{width}} | {r}")
    return "\n".join(rows)


def run_demo(profile="clean", layout=False, seed=0):
    refs = _load_multiline_refs(ROOT / "bench-data.json")
    passages = [x["reference"] for x in refs]
    results, stats = pdf_paste(passages, profile=profile, layout=layout, seed=seed)
    sentinel = stats["sentinel"] or ["?", "?"]
    print(
        f"demo passages={len(refs)} profile={profile} sentinel={sentinel[0]}{sentinel[1]} failed={stats['failed']}"
    )
    for row, rec in zip(refs, results):
        knobs = ",".join(rec["knobs"]) if rec.get("knobs") else "-"
        print(f"\n== {row['id']} ==")
        print("input".ljust(60) + " | reference")
        print("-" * 60 + "-+-" + "-" * 60)
        print(_side_by_side(rec["text"], row["reference"]))
        print(
            f"spans={len(rec['spans'])} ok={rec['ok']} profile={rec['profile']} "
            f"knobs={knobs} reason={rec['reason'] or '-'}"
        )
    comp = stats.get("span_completeness", {})
    print(
        "\nSpan completeness: "
        f"extended={comp.get('extended_spans', 0)}/{comp.get('total_spans', 0)} "
        f"({comp.get('extended_rate', 0.0):.3f}) "
        f"partial={comp.get('partial_spans', 0)}/{comp.get('total_spans', 0)} "
        f"({comp.get('partial_rate', 0.0):.3f})"
    )
    if profile == "dirty":
        print("\nDirty drop rate by knob")
        print("| knob | total | failed | drop_rate | ext_spans | ext_rate | part_spans | part_rate |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for knob, row in stats["drop_rate_by_knob"].items():
            print(
                f"| {knob} | {row['total']} | {row['failed']} | {row['drop_rate']:.3f} | "
                f"{row.get('extended_spans', 0)} | {row.get('extended_rate', 0.0):.3f} | "
                f"{row.get('partial_spans', 0)} | {row.get('partial_rate', 0.0):.3f} |"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--layout", action="store_true")
    ap.add_argument("--profile", choices=["clean", "dirty"], default="clean")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.demo:
        run_demo(profile=args.profile, layout=args.layout, seed=args.seed)
        return
    ap.error("No action requested. Use --demo.")


if __name__ == "__main__":
    main()
