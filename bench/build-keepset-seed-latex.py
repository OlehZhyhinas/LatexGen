#!/usr/bin/env python3
r"""Build bench/keepset-seed-latex.json: LaTeX macro/environment/delimiter seed
list extracted programmatically from the vendored KaTeX bundle.

Extraction method (no hand-typed macro names):

1. Parse every `names:[...]` array literal in `bench/vendor/katex/katex.min.js`.
   KaTeX registers every function and environment it supports through calls of
   the shape `defineFunction({..., names: ["\\frac", ...], ...})` /
   `defineEnvironment({..., names: ["pmatrix", ...], ...})`. The minifier
   renames the call target and the object keys survive only where they are
   used as property names (`names:`), but the string literals inside the
   array -- the actual command/environment spellings -- are left untouched
   because they are semantically significant (they are looked up by the
   parser at runtime). We regex out every `names:[ ... ]` array and JSON-parse
   each double-quoted string literal inside it with a JS-string-literal-aware
   pattern (`"(?:[^"\\]|\\.)*"`) so backslash/quote escapes decode correctly.
2. Separately, collect every standalone backslash-command string literal
   anywhere in the bundle (`"\\command"`), which also catches macros defined
   via `defineMacro("\\ne", ...)`-style calls whose call target the minifier
   renamed away, and which are not necessarily inside a `names:` array.
3. Union both sets, then partition into: environments (names with no leading
   backslash), functions/macros (leading backslash), and manually-appended
   delimiter forms that KaTeX recognizes but are not spelled as `\\command`
   tokens in its source (`$`, `$$`, literal `\\(`, `\\)`, `\\[`, `\\]` are already
   caught by step 1/2; `\left`/`\right` likewise). We only *add* delimiter
   pairs by hand where they are true multi-character parser syntax rather
   than a command name (`$$`), record that explicitly, and keep the count of
   hand-added entries separate from the programmatically extracted count.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KATEX_JS = ROOT / "vendor" / "katex" / "katex.min.js"
KATEX_VERSION = "0.16.11"  # from README.md badge / package pinned in this vendor dir
OUT_PATH = ROOT / "keepset-seed-latex.json"

JS_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')
NAMES_ARRAY = re.compile(r"names:\[(.*?)\]", re.S)


_JS_ESCAPES = {
    "n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v",
    "0": "\0", "\\": "\\", '"': '"', "'": "'",
}


def js_string_literal_to_str(lit):
    # JS double-quoted string escapes are a superset of JSON's (also allow
    # \xNN hex and a bare-backslash-before-any-char passthrough that JSON's
    # decoder rejects), so decode by hand rather than via json.loads.
    body = lit[1:-1]
    out = []
    i = 0
    while i < len(body):
        c = body[i]
        if c == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            if nxt == "x" and i + 3 < len(body):
                out.append(chr(int(body[i + 2:i + 4], 16)))
                i += 4
                continue
            if nxt == "u" and i + 5 < len(body):
                out.append(chr(int(body[i + 2:i + 6], 16)))
                i += 6
                continue
            out.append(_JS_ESCAPES.get(nxt, nxt))
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def extract_names_arrays(src):
    names = set()
    n_arrays = 0
    for m in NAMES_ARRAY.finditer(src):
        n_arrays += 1
        body = m.group(1)
        for lit in JS_STRING.findall(body):
            names.add(js_string_literal_to_str(lit))
    return names, n_arrays


def extract_backslash_literals(src):
    names = set()
    for lit in JS_STRING.findall(src):
        s = js_string_literal_to_str(lit)
        if len(s) >= 2 and s[0] == "\\" and s[1:2].isalpha():
            names.add(s)
        elif s in ("\\'", "\\`", "\\^", "\\~", "\\=", "\\.", '\\"', "\\c", "\\r"):
            names.add(s)
    return names


# Math-mode delimiter syntax LatexGen's own prompts/output rely on ($, $$,
# \(, \), \[, \]) that is *not* KaTeX command/environment syntax at all: KaTeX
# renderToString() takes raw TeX and never sees these wrappers (they are
# stripped by whatever places the math into a document -- LatexGen's own
# SYSTEM prompt says "Pure math: \[ ... \]. Prose with math: wrap math in
# \( ... \)."). Checked directly against the bundle: KaTeX registers `\(`
# and `$` as an inline math-mode toggler (`names:["\\(","$"]`, type
# "styling") for the sole purpose of accepting nested `$...$`/`\(...\)`
# inside already-parsed math, but never registers `\)`, `\[`, `\]`, or `$$`
# as commands -- `\)`/`$`'s closing counterpart is consumed via
# `parser.expect()`, not dispatched as a function, and `\[`/`\]`/`$$` do not
# appear anywhere in the bundle outside of unrelated tokenizer regexes. So
# all six forms are added here by hand; `\left`/`\right`/`\middle` are true
# KaTeX macros and are already covered by the programmatic extraction above.
HAND_ADDED_DELIMITERS = ["$", "$$", "\\(", "\\)", "\\[", "\\]"]


def main():
    src = KATEX_JS.read_text(encoding="utf-8")
    names_from_arrays, n_arrays = extract_names_arrays(src)
    names_from_literals = extract_backslash_literals(src)

    all_names = names_from_arrays | names_from_literals
    programmatic_count = len(all_names)

    for d in HAND_ADDED_DELIMITERS:
        all_names.add(d)

    environments = sorted(n for n in all_names if not n.startswith("\\") and n not in ("$", "$$"))
    delimiters = sorted(n for n in all_names if n in ("$", "$$", "\\(", "\\)", "\\[", "\\]", "\\{", "\\}", "\\left", "\\right", "\\middle"))
    functions_macros = sorted(n for n in all_names if n.startswith("\\") and n not in delimiters)

    payload = {
        "provenance": {
            "katex_version": KATEX_VERSION,
            "source_file": "bench/vendor/katex/katex.min.js",
            "extraction_method": (
                "Regex over every `names:[...]` array literal in the minified "
                "KaTeX bundle (KaTeX's own defineFunction/defineEnvironment "
                "registration tables), unioned with every standalone "
                "backslash-command string literal in the bundle (catches "
                "defineMacro-registered aliases whose call site was renamed "
                "by the minifier). JS string literals decoded by hand "
                "(escape syntax is a superset of JSON's). No macro or "
                "environment name was hand-typed; only the six math-mode "
                "delimiter wrappers noted below were added by hand, "
                "because they are LatexGen/document syntax that KaTeX's "
                "renderToString never sees, not KaTeX commands."
            ),
            "names_arrays_found": n_arrays,
            "programmatic_count": programmatic_count,
            "hand_added_delimiters": HAND_ADDED_DELIMITERS,
            "hand_added_count": len(HAND_ADDED_DELIMITERS),
            "total_count": len(all_names),
        },
        "counts": {
            "environments": len(environments),
            "delimiters": len(delimiters),
            "functions_macros": len(functions_macros),
            "total": len(all_names),
        },
        "environments": environments,
        "delimiters": delimiters,
        "functions_macros": functions_macros,
        "all": sorted(all_names),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"names arrays found: {n_arrays}")
    print(f"programmatic macro/env/delimiter names: {programmatic_count}")
    print(f"hand-added delimiter forms: {HAND_ADDED_DELIMITERS}")
    print(f"total seed entries: {len(all_names)}")
    print(f"  environments: {len(environments)}")
    print(f"  delimiters: {len(delimiters)}")
    print(f"  functions_macros: {len(functions_macros)}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
