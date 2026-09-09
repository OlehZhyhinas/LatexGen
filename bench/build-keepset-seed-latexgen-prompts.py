#!/usr/bin/env python3
"""Extract the prompts LatexGen's own client actually sends (not the harness's
webnn-workbench proxy prompts, which keepset-harness-prompts.json already
covers) from `public/pipeline.js` and `public/pdf-prompt.js`, and write
`bench/keepset-seed-latexgen-prompts.json`.

This is the fix for keep-set v2 failure 1's root cause: v1 only folded in the
webnn-workbench harness's SYSTEM/CONVERT_USER strings, which do not contain
the PDF_HINT or PDF_FEWSHOT text -- exactly where the 7 missing prompt-
template tokens ("oscillator", "damping", "converge", "flattened", "reorder",
the diaeresis-above byte, and literal Greek phi) actually live.

Extraction is regex-based against the exact literal syntax of these two
files (not a general JS parser) because both are small, stable, single-
purpose modules whose relevant declarations are simple template/string
literals -- the same tradeoff `build-keepset-seed-latex.py` already makes
against vendored KaTeX source in this repo. Every extracted string is
asserted against known substrings from the source below, so a future edit
that silently changes these declarations' syntax fails loudly here instead
of quietly shipping a stale seed.
"""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
PIPELINE_JS = REPO_ROOT / "public" / "pipeline.js"
PDF_PROMPT_JS = REPO_ROOT / "public" / "pdf-prompt.js"
OUT_PATH = ROOT / "keepset-seed-latexgen-prompts.json"


def git_last_commit(path):
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "log", "-1", "--format=%H", "--", str(path.relative_to(REPO_ROOT))],
        text=True,
    ).strip()


def unescape_js(s):
    """Unescape the small set of backslash escapes actually used in these
    two files' string/template literals: \\\\ -> \\, \\n -> newline,
    \\" -> ", \\` -> `. Order matters: double-backslash first."""
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt == "\\":
                out.append("\\")
                i += 2
                continue
            if nxt == "n":
                out.append("\n")
                i += 2
                continue
            if nxt == '"':
                out.append('"')
                i += 2
                continue
            if nxt == "`":
                out.append("`")
                i += 2
                continue
            if nxt == "(" or nxt == "[" or nxt == "]" or nxt == ")":
                # backslash is literal LaTeX-delimiter content in this
                # source (e.g. \\[ ... \\] in the backtick template), not a
                # recognized JS escape -- keep both characters.
                out.append(c)
                i += 1
                continue
        out.append(c)
        i += 1
    return "".join(out)


def extract_backtick_const(src, name):
    m = re.search(re.escape(f"const {name} = `") + r"([\s\S]*?)`;", src)
    if not m:
        raise SystemExit(f"BUG: could not find backtick const {name!r}")
    return unescape_js(m.group(1))


def extract_dquote_export_const(src, name):
    m = re.search(re.escape(f'export const {name} = "') + r'([\s\S]*?)";', src)
    if not m:
        raise SystemExit(f"BUG: could not find double-quoted export const {name!r}")
    return unescape_js(m.group(1))


def extract_convert_user_prefix(src):
    m = re.search(r"const CONVERT_USER = \(text\) => `([\s\S]*?)\$\{text\}`;", src)
    if not m:
        raise SystemExit("BUG: could not find CONVERT_USER template")
    return unescape_js(m.group(1))


def extract_pdf_fewshot(src):
    m = re.search(r"export const PDF_FEWSHOT = (\[[\s\S]*?\n\]);", src)
    if not m:
        raise SystemExit("BUG: could not find PDF_FEWSHOT array literal")
    raw = m.group(1)
    data = json.loads(raw)  # the literal is valid JSON (double-quoted keys/strings)
    for pair in data:
        if set(pair.keys()) != {"user", "assistant"}:
            raise SystemExit(f"BUG: unexpected PDF_FEWSHOT pair shape: {sorted(pair.keys())}")
    return data


def main():
    pipeline_src = PIPELINE_JS.read_text()
    pdf_prompt_src = PDF_PROMPT_JS.read_text()

    system_prompt = extract_backtick_const(pipeline_src, "SYSTEM_PROMPT")
    convert_user_prefix = extract_convert_user_prefix(pipeline_src)
    pdf_hint = extract_dquote_export_const(pdf_prompt_src, "PDF_HINT")
    pdf_fewshot = extract_pdf_fewshot(pdf_prompt_src)

    # Explicit correctness assertions: the whole point of this seed is that
    # these specific words/characters (the ones keep-set v1 was missing) are
    # actually present in what got extracted, not just that the regex found
    # *some* string.
    must_contain_system = ["Convert the text to LaTeX", "amsmath"]
    for s in must_contain_system:
        if s not in system_prompt:
            raise SystemExit(f"BUG: SYSTEM_PROMPT missing expected substring {s!r}")
    must_contain_hint = ["flattened", "reorder", "superscripts/subscripts"]
    for s in must_contain_hint:
        if s not in pdf_hint:
            raise SystemExit(f"BUG: PDF_HINT missing expected substring {s!r}")
    fewshot_all_user = "\n".join(p["user"] for p in pdf_fewshot)
    fewshot_all_assistant = "\n".join(p["assistant"] for p in pdf_fewshot)
    must_contain_fewshot = ["oscillator", "damping", "converge", "\u03c6"]  # literal phi (U+03C6)
    for s in must_contain_fewshot:
        if s not in fewshot_all_user and s not in fewshot_all_assistant:
            raise SystemExit(f"BUG: PDF_FEWSHOT missing expected substring {s!r}")
    if "x\u00a8" not in fewshot_all_user and "\u00a8" not in fewshot_all_user:
        raise SystemExit("BUG: PDF_FEWSHOT user text missing expected diaeresis-above character (U+00A8)")
    if len(pdf_fewshot) != 2:
        raise SystemExit(f"BUG: expected 2 PDF_FEWSHOT pairs, found {len(pdf_fewshot)}")

    payload = {
        "provenance": {
            "source_repo_relative_files": ["public/pipeline.js", "public/pdf-prompt.js"],
            "source_file_last_commits": {
                "public/pipeline.js": git_last_commit(PIPELINE_JS),
                "public/pdf-prompt.js": git_last_commit(PDF_PROMPT_JS),
            },
            "extraction_method": (
                "Regex-extracted verbatim from the two source files' literal SYSTEM_PROMPT "
                "(backtick template), CONVERT_USER (backtick template with one ${text} interpolation "
                "point, captured as a fixed prefix string), PDF_HINT (double-quoted string), and "
                "PDF_FEWSHOT (JSON array literal, parsed directly with json.loads since the literal's "
                "syntax is valid JSON) declarations. Not hand-transcribed: re-running this script "
                "against an edited pipeline.js/pdf-prompt.js regenerates this file from the new text."
            ),
            "note": (
                "This is LatexGen's OWN conversion prompt (what the app actually sends), distinct from "
                "keepset-harness-prompts.json which covers the webnn-workbench harness's separate proxy "
                "SYSTEM/CONVERT_USER/FRESH_ITEMS/QUALITY_CORPUS strings. v1 only had the harness file, "
                "which does not contain PDF_HINT or PDF_FEWSHOT text -- exactly where the 7 tokens keep-set "
                "v1 was missing (oscillator, damping, converge, flattened, reorder, a diaeresis-above byte, "
                "and literal Greek phi) actually occur."
            ),
        },
        "system_prompt": system_prompt,
        "convert_user_prefix": convert_user_prefix,
        "pdf_hint": pdf_hint,
        "pdf_fewshot": pdf_fewshot,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_PATH}")
    print(f"  system_prompt: {len(system_prompt)} chars")
    print(f"  convert_user_prefix: {convert_user_prefix!r}")
    print(f"  pdf_hint: {len(pdf_hint)} chars")
    print(f"  pdf_fewshot: {len(pdf_fewshot)} pairs")


if __name__ == "__main__":
    main()
