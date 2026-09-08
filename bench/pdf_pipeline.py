#!/usr/bin/env python3
"""Deterministic PDF-paste cleanup + rule-based math span segmentation."""
import re, unicodedata

ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}
OPERATORS = set("=+*/^_±×⋅·≤≥≠≈→∂∇∈∉⊂∪∩∮⌈⌊|-−∣∞")
CONNECTIVES = {"for", "where", "and", ",", "."}
SPAN_EDGE_WORDS = {
    "for", "any", "all", "where", "with", "and", "then", "if", "so", "as", "such", "that",
    "which", "when", "first", "second", "third", "real", "number", "the", "a", "an", "of", "in", "is", "are",
}
SUPERS, SUBS = "⁰¹²³⁴⁵⁶⁷⁸⁹", "₀₁₂₃₄₅₆₇₈₉"
FOOTNOTE_LINE_RE = re.compile(rf"^(?:\d{{1,2}}\.\s+\S|[{re.escape(SUPERS)}](?:[\).])?\s*\S)")
GREEK_RE = re.compile(r"[α-ωΑ-ΩπΠμσΣΔΩθΘλΛγΓβΒϵεϕφψΨηΗξΞρΡτΤυΥχΧζΖ]")
MIX_RE = re.compile(r"(?=.*[A-Za-z])(?=.*\d)")
DATE_RE, UNIT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$"), re.compile(r"^\d+(?:\.\d+)?\s+[A-Za-z]{1,8}s?$")
YEAR_RE = re.compile(r"^(18|19|20)\d{2}$")
MATH_WORDS = {"sin", "cos", "tan", "log", "ln", "lim", "det", "exp", "max", "min"}
LOWER_WORD_RE = re.compile(r"^[a-z]{4,}$")
PROSE_WORD_RE = re.compile(r"^[a-z]{3,}$")
PROSE_MATH_NAMES = {
    "sin", "cos", "tan", "ln", "log", "lim", "exp", "max", "min", "det",
    "arg", "sup", "inf", "mod", "gcd", "dx", "dy", "dt", "ds", "dz",
}
ORPHAN_SYMBOLS = {"∞", "∫", "∑", "∏", "√"}
HYPHEN_CHARS = {"-", "\u2010", "\u2011"}
BRIDGE_TOKENS = {"(", ")", ","}
SHORT_PROSE = {
    "in", "on", "of", "to", "at", "by", "as", "an", "is", "we", "if", "so", "it", "its",
    "the", "a", "or", "not", "was", "find", "show", "know", "use", "given", "recall",
    "this", "that", "with", "from", "over", "then", "here", "note", "well",
}


def _tok_core(tok): return tok.strip(".,;:!?\"`")
def _token_rows(line): return [m.group(0) for m in re.finditer(r"\S+", line)]


def _remove_zero_width(text):
    chars, omap = [], []
    for i, ch in enumerate(text):
        if ch == "\r" or ch in ZERO_WIDTH or unicodedata.category(ch) == "Cf": continue
        chars.append(ch); omap.append(i)
    return chars, omap


def _split_lines(chars, omap):
    out, cc, cm, page = [], [], [], 0
    for ch, oi in zip(chars, omap):
        if ch in "\n\f":
            out.append({"text": "".join(cc), "map": cm[:], "page": page}); cc, cm = [], []
            if ch == "\f": page += 1
        else:
            cc.append(ch); cm.append(oi)
    out.append({"text": "".join(cc), "map": cm[:], "page": page})
    return out


def _line_key(s):
    t = re.sub(r"\s+", " ", s).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", t) if t else ""


def _strip_piece(text, mp):
    i, j = 0, len(text)
    while i < j and text[i].isspace(): i += 1
    while j > i and text[j - 1].isspace(): j -= 1
    return text[i:j], mp[i:j]


def _drop_running_heads(lines):
    counts = {}
    for row in lines:
        t, key = row["text"], _line_key(row["text"])
        if key and len(t.strip()) <= 90 and sum(ch.isalpha() for ch in t) >= 8 and len(key) >= 10:
            counts[key] = counts.get(key, 0) + 1
    repeat, out = {k for k, v in counts.items() if v >= 2}, []
    for row in lines:
        text, key = row["text"].strip(), _line_key(row["text"].strip())
        if key in repeat: continue
        out.append(row)
    return out


def _is_page_number(lines, i):
    s = lines[i]["text"].strip()
    if not re.fullmatch(r"\d{1,3}", s): return False
    prev_s = lines[i - 1]["text"].strip() if i > 0 else ""
    next_s = lines[i + 1]["text"].strip() if i + 1 < len(lines) else ""
    if not prev_s and not next_s: return True
    if (i == 0 or not prev_s) and next_s and len(re.findall(r"[A-Za-z]{4,}", next_s)) >= 3 and int(s) <= 20: return True
    if (i == len(lines) - 1 or not next_s) and prev_s and len(re.findall(r"[A-Za-z]{4,}", prev_s)) >= 3 and int(s) <= 20: return True
    return False


def _is_footnote_line(s):
    return bool(FOOTNOTE_LINE_RE.match(s))


def _drop_footers(lines):
    out = []
    i = 0
    while i < len(lines):
        page = lines[i].get("page", 0)
        j = i
        while j < len(lines) and lines[j].get("page", 0) == page: j += 1
        page_rows = lines[i:j]
        keep = [True] * len(page_rows)

        for k, row in enumerate(page_rows):
            s = row["text"].strip()
            if _is_page_number(page_rows, k):
                if k == 0 and _is_orphan_paragraph(s) and len(_token_rows(s)) <= 2:
                    continue
                keep[k] = False

        t = len(page_rows) - 1
        while t >= 0 and (not keep[t] or not page_rows[t]["text"].strip()): t -= 1
        end = t
        while t >= 0 and keep[t] and _is_footnote_line(page_rows[t]["text"].strip()): t -= 1
        start = t + 1
        if start <= end:
            has_body = any(
                keep[k] and page_rows[k]["text"].strip() and not _is_footnote_line(page_rows[k]["text"].strip())
                for k in range(start)
            )
            if has_body:
                for k in range(start, end + 1): keep[k] = False

        for k, row in enumerate(page_rows):
            if keep[k]: out.append(row)
        i = j
    return out


def _split_two_col_row(text, mp):
    m = None
    for hit in re.finditer(r" {8,}", text):
        if m is None or (hit.end() - hit.start()) > (m.end() - m.start()): m = hit
    if m is None: return None
    ltxt, lmp = _strip_piece(text[:m.start()], mp[:m.start()])
    rtxt, rmp = _strip_piece(text[m.end():], mp[m.end():])
    return ((ltxt, lmp), (rtxt, rmp)) if ltxt and rtxt else None


def _deinterleave(lines):
    split_rows = [(i, _split_two_col_row(r["text"], r["map"])) for i, r in enumerate(lines)]
    split_rows = [(i, sp) for i, sp in split_rows if sp is not None]
    if len(split_rows) < 3: return lines
    left, right, used = [], [], set()
    for i, (lp, rp) in split_rows:
        page = lines[i].get("page", 0)
        used.add(i); left.append({"text": lp[0], "map": lp[1], "page": page}); right.append({"text": rp[0], "map": rp[1], "page": page})
    if len(" ".join(x["text"] for x in left)) < 20 or len(" ".join(x["text"] for x in right)) < 20: return lines
    out = [r for i, r in enumerate(lines) if i not in used]
    return out + [{"text": "", "map": []}] + left + [{"text": "", "map": []}] + right


def _line_mathy_ratio(line):
    toks = _token_rows(line)
    if not toks: return 0.0
    n = 0
    for tok in toks:
        core = _tok_core(tok)
        if DATE_RE.fullmatch(core): continue
        if any(c in core for c in OPERATORS) or GREEK_RE.search(core) or any(c in core for c in SUPERS + SUBS): n += 1; continue
        if "/" in core or "^" in core or "_" in core or MIX_RE.search(core): n += 1; continue
        if re.fullmatch(r"\d+(?:\.\d+)?", core) or core.lower() in MATH_WORDS: n += 1
    return n / len(toks)


def _looks_residue_line(line):
    s = line.strip()
    if not s: return False
    toks = _token_rows(s)
    if len(toks) > 3 or _line_mathy_ratio(s) >= 0.5: return False
    return s[0] in ",;:)]" or s.lower() in {"the", "and", "or", "for", "where", ", the"}


def _hyphen_join_mode(prev_text, next_text):
    if not prev_text or not next_text: return None
    prev, nxt = prev_text.rstrip(), next_text.lstrip()
    if not prev or not nxt or prev[-1] not in HYPHEN_CHARS: return None
    i = len(prev) - 2
    while i >= 0 and prev[i].isspace(): i -= 1
    if i < 0 or not prev[i].isalpha(): return None
    j = i
    while j >= 0 and prev[j].isalpha(): j -= 1
    prefix = prev[j + 1:i + 1]
    first = nxt[0]
    if first.islower(): return "keep" if len(prefix) == 1 else "drop"
    if first.isupper() or first.isdigit(): return "keep"
    return None


def _pop_trailing_hyphenation(out_t, out_m):
    while out_t and out_t[-1].isspace(): out_t.pop(); out_m.pop()
    if out_t and out_t[-1] in HYPHEN_CHARS: out_t.pop(); out_m.pop()


def _join_rows(lines):
    if not lines: return "", []
    out_t, out_m = list(lines[0]["text"]), lines[0]["map"][:]
    for row in lines[1:]:
        txt, mp = row["text"], row["map"]
        mode = _hyphen_join_mode("".join(out_t), txt)
        if mode == "drop":
            _pop_trailing_hyphenation(out_t, out_m)
        elif mode == "keep":
            while out_t and out_t[-1].isspace(): out_t.pop(); out_m.pop()
        else:
            out_t.append(" "); out_m.append(-1)
        out_t.extend(txt); out_m.extend(mp)
    return "".join(out_t), out_m


def _join_prose(lines):
    return _join_rows(lines)


def _join_math(lines):
    main, tail = [], []
    for row in lines:
        (tail if _looks_residue_line(row["text"]) else main).append(row)
    return _join_rows(main + tail)


def _collapse_spaces(text, omap):
    out_t, out_m, prev_space = [], [], False
    for ch, oi in zip(text, omap):
        ch = " " if ch == "\t" else ch
        if ch == "\n":
            while out_t and out_t[-1] == " ": out_t.pop(); out_m.pop()
            out_t.append("\n"); out_m.append(oi); prev_space = False; continue
        if ch.isspace():
            if prev_space: continue
            out_t.append(" "); out_m.append(oi); prev_space = True
        else:
            out_t.append(ch); out_m.append(oi); prev_space = False
    text2, map2, nl = [], [], 0
    for ch, oi in zip(out_t, out_m):
        nl = nl + 1 if ch == "\n" else 0
        if nl > 2: continue
        text2.append(ch); map2.append(oi)
    a, b = 0, len(text2)
    while a < b and text2[a].isspace(): a += 1
    while b > a and text2[b - 1].isspace(): b -= 1
    return "".join(text2[a:b]), map2[a:b]


def normalize(text: str) -> tuple[str, list[int]]:
    chars, omap = _remove_zero_width(text)
    lines = _deinterleave(_drop_footers(_drop_running_heads(_split_lines(chars, omap))))
    blocks, cur = [], []
    for row in lines:
        if row["text"].strip():
            t, m = _strip_piece(row["text"], row["map"]); cur.append({"text": t, "map": m})
        else:
            if cur: blocks.append(cur); cur = []
            blocks.append([])
    if cur: blocks.append(cur)
    first_non_empty = next((i for i, block in enumerate(blocks) if block), None)
    if first_non_empty is not None:
        block = blocks[first_non_empty]
        if (
            len(block) >= 2
            and _is_orphan_paragraph(block[0]["text"])
            and len(_token_rows(block[0]["text"])) <= 2
            and not _hyphen_join_mode(block[0]["text"], block[1]["text"])
        ):
            blocks = blocks[:first_non_empty] + [[block[0]], [], block[1:]] + blocks[first_non_empty + 1:]
    out_t, out_m, wrote = [], [], False
    for block in blocks:
        if not block:
            if wrote: out_t.extend(["\n", "\n"]); out_m.extend([-1, -1])
            wrote = False
            continue
        tok_n = sum(len(_token_rows(r["text"])) for r in block)
        ratios = [_line_mathy_ratio(r["text"]) for r in block]
        tok_m = sum(int(round(rt * len(_token_rows(r["text"])))) for rt, r in zip(ratios, block))
        short = sum(1 for r in block if len(_token_rows(r["text"])) <= 2)
        has_op = any(re.search(r"[=+*/^_≤≥≠≈∫∑∏√|]", r["text"]) for r in block)
        is_math = (tok_n > 0 and (tok_m / tok_n) >= 0.6) or (has_op and short >= 1 and len(block) >= 2)
        bt, bm = _join_math(block) if is_math else _join_prose(block)
        if out_t and out_t[-1] != "\n": out_t.append("\n"); out_m.append(-1)
        out_t.extend(bt); out_m.extend(bm); wrote = True
    return _collapse_spaces("".join(out_t), out_m)


def _tokenize(text):
    out, starts, line_i = [], [0], 0
    for i, ch in enumerate(text):
        if ch == "\n": starts.append(i + 1)
    for m in re.finditer(r"\S+", text):
        s, e = m.span()
        while line_i + 1 < len(starts) and s >= starts[line_i + 1]: line_i += 1
        out.append({"t": m.group(0), "s": s, "e": e, "line": line_i})
    return out


def _has_op(tok):
    if not tok: return False
    c = _tok_core(tok)
    if _is_prose_hyphen(c): return False
    return any(ch in c for ch in OPERATORS) or "/" in c or "^" in c or "_" in c or bool(re.fullmatch(r"\(?[A-Za-z0-9]+\)?'", c))


def _is_prose_hyphen(core):
    m = re.fullmatch(r"([a-z]+)-([a-z]+)", core)
    if not m: return False
    a, b = m.group(1), m.group(2)
    if len(a) >= 2 and len(b) >= 2: return True
    if (len(a) == 1 and len(b) >= 2) or (len(b) == 1 and len(a) >= 2): return True
    return False


def _raw_mathy(tok):
    core, low = _tok_core(tok), _tok_core(tok).lower()
    if not core or DATE_RE.fullmatch(core) or low in {"for", "where", "and", "or"}: return False, False
    if _is_prose_hyphen(core): return False, False
    if any(c in core for c in OPERATORS) or GREEK_RE.search(core) or any(c in core for c in SUPERS + SUBS): return True, True
    if "/" in core or MIX_RE.search(core) or re.fullmatch(r"[A-Za-z0-9]{1,3}-[A-Za-z0-9]{1,3}", core): return True, True
    if re.match(r"^d[A-Za-z]$", core) or re.fullmatch(r"[A-Za-z]{1,6}\([^)]*\)", core): return True, True
    if "'" in core and not low.endswith("'s") and re.fullmatch(r"[\(\)A-Za-z]{1,8}'", core): return True, True
    if low in MATH_WORDS or re.fullmatch(r"[A-Za-z]{2,3}", core) or re.fullmatch(r"\d+(?:\.\d+)?", core): return True, False
    if len(core) == 1 and core.isalpha(): return True, False
    if "(" in core and ")" in core and re.search(r"[=+/*^_]", core): return True, True
    return False, False


def _mathy_token(tok, prev_tok, next_tok):
    core, (mathy, strong) = _tok_core(tok), _raw_mathy(tok)
    if not mathy: return False, False
    prev, nxt = (_tok_core(prev_tok).lower() if prev_tok else ""), (_tok_core(next_tok).lower() if next_tok else "")
    if len(core) == 1 and core in {"a", "A", "I"} and next_tok and re.fullmatch(r"[a-z]{2,}", _tok_core(next_tok)): return False, False
    if len(core) == 1 and core.isalpha():
        near = _has_op(prev_tok) or _has_op(next_tok) or (len(prev) == 1 and prev.isalpha()) or (len(nxt) == 1 and nxt.isalpha())
        if not near: return False, False
    if re.fullmatch(r"\d+(?:\.\d+)?", core):
        near_math = _has_op(prev_tok) or _has_op(next_tok)
        near_var = bool(re.fullmatch(r"[A-Za-z]{1,2}", prev)) or bool(re.fullmatch(r"[A-Za-z]{1,2}", nxt))
        if prev in MATH_WORDS or nxt in MATH_WORDS: near_var = True
        if not near_math and not near_var: return False, False
    if re.fullmatch(r"[A-Za-z]{2,3}", core):
        if core.lower() in MATH_WORDS: return True, strong or _has_op(prev_tok) or _has_op(next_tok)
        near_math = _has_op(prev_tok) or _has_op(next_tok)
        near_num = bool(re.fullmatch(r"\d+(?:\.\d+)?", prev)) or bool(re.fullmatch(r"\d+(?:\.\d+)?", nxt))
        if (not near_math and not near_num) or core.lower() in SHORT_PROSE: return False, False
    if YEAR_RE.fullmatch(core): return False, False
    return True, strong


def _span_trim(text, s, e):
    while e > s and text[e - 1] in ".,;:!?": e -= 1
    while s < e and text[s].isspace(): s += 1
    while e > s and text[e - 1].isspace(): e -= 1
    return s, e


def _trim_connective_edges(text, s, e):
    s, e = _span_trim(text, s, e)
    if s >= e: return s, e

    while True:
        toks = list(re.finditer(r"\S+", text[s:e]))
        if not toks: return s, e
        first = _tok_core(toks[0].group(0)).lower()
        if first not in SPAN_EDGE_WORDS: break
        ns = s + toks[0].end()
        ns, _ = _span_trim(text, ns, e)
        rem = list(re.finditer(r"\S+", text[ns:e]))
        if not rem: break
        if _has_op(rem[0].group(0)): break
        s = ns

    while True:
        toks = list(re.finditer(r"\S+", text[s:e]))
        if not toks: return s, e
        last = _tok_core(toks[-1].group(0)).lower()
        if last not in SPAN_EDGE_WORDS: break
        ne = s + toks[-1].start()
        _, ne = _span_trim(text, s, ne)
        rem = list(re.finditer(r"\S+", text[s:ne]))
        if not rem: break
        if _has_op(rem[-1].group(0)): break
        e = ne

    return _span_trim(text, s, e)


def _reject_span(text, s, e):
    piece = re.sub(r"\s+", " ", text[s:e]).strip()
    if not piece or piece in {"-", "−", "+", "=", "*", "/", "|", "∣"}: return True
    if YEAR_RE.fullmatch(piece): return True
    if UNIT_RE.fullmatch(piece) and "/" not in piece: return True
    if re.fullmatch(r"\d{1,3}", piece):
        left = text[max(0, s - 32):s].lower()
        return True if ("room" in left or "chapter" in left) else True
    if re.fullmatch(r"\d+(?:\.\d+)?", piece): return True
    if re.fullmatch(r"[A-Za-z' ]+", piece):
        words = [w for w in piece.replace("'", " ").split() if w]
        if words and all(w.lower() not in MATH_WORDS for w in words): return True
    if piece in {"x", "t", "n", "c", "e", "A"}:
        left = text[max(0, s - 32):s].lower()
        if any(w in left for w in ("variable", "time", "letter", "name")): return True
    return False


def _can_extend(tok):
    core = _tok_core(tok)
    if not core: return False
    low = core.lower()
    if low in SHORT_PROSE and low not in MATH_WORDS: return False
    if any(c in core for c in OPERATORS) or GREEK_RE.search(core) or any(c in core for c in SUPERS + SUBS): return True
    if re.fullmatch(r"\d+(?:\.\d+)?", core) or re.fullmatch(r"[A-Za-z]{1,3}", core): return True
    if "/" in core or "^" in core or "_" in core: return True
    if any(x in core for x in "()[]"): return bool(re.search(r"[A-Za-z0-9]", core))
    return False


def _collapse_inline(s):
    return re.sub(r"\s+", " ", s).strip()


def _is_strong_mathy_token(tok):
    core = _tok_core(tok)
    if not core or DATE_RE.fullmatch(core) or _is_prose_hyphen(core):
        return False
    if "=" in core or any(ch in core for ch in ORPHAN_SYMBOLS):
        return True
    if GREEK_RE.search(core):
        return True
    return _has_op(core)


def _is_orphan_like_token(tok):
    core = _tok_core(tok)
    if not core:
        core = tok.strip()
    if not core:
        return False
    if re.fullmatch(r"\d+", core):
        return True
    if GREEK_RE.fullmatch(core):
        return True
    if len(core) == 1 and (core in ORPHAN_SYMBOLS or core in OPERATORS):
        return True
    return len(core) <= 3


def _is_orphan_paragraph(par_text):
    toks = _token_rows(par_text)
    if not toks:
        return False
    if any(LOWER_WORD_RE.fullmatch(_tok_core(tok) or "") for tok in toks):
        return False
    return all(_is_orphan_like_token(tok) for tok in toks)


def _is_prose_word_token(tok):
    core = (_tok_core(tok) or "").lower()
    if not PROSE_WORD_RE.fullmatch(core):
        return False
    return core not in PROSE_MATH_NAMES


def _max_prose_word_run(par_text):
    run = 0
    best = 0
    for tok in _token_rows(par_text):
        if _is_prose_word_token(tok):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _paragraph_slices(text):
    if not text:
        return []
    out = []
    i = 0
    n = len(text)
    while i < n:
        j = text.find("\n\n", i)
        if j < 0:
            j = n
        out.append((i, j))
        i = j + 2
    return out


def _gap_is_orphan_bridge(gap_text):
    toks = _token_rows(gap_text)
    if not toks:
        return False
    for tok in toks:
        if _is_prose_word_token(tok):
            return False
        core = _tok_core(tok)
        if core in BRIDGE_TOKENS:
            continue
        if _is_orphan_like_token(tok):
            continue
        return False
    return True


def _has_prose_word_in_range(text, s, e):
    if s >= e:
        return False
    for tok in _token_rows(text[s:e]):
        if _is_prose_word_token(tok):
            return True
    return False


def _split_span_on_prose_runs(text, s, e):
    toks = []
    for m in re.finditer(r"\S+", text[s:e]):
        toks.append((s + m.start(), s + m.end(), m.group(0)))
    if not toks:
        return [(s, e)]

    drops = []
    run_start = None
    run_end = None
    run_len = 0
    for ts, te, tok in toks:
        if _is_prose_word_token(tok):
            if run_start is None:
                run_start = ts
            run_end = te
            run_len += 1
            continue
        if run_len >= 2:
            drops.append((run_start, run_end))
        run_start = None
        run_end = None
        run_len = 0
    if run_len >= 2:
        drops.append((run_start, run_end))
    if not drops:
        return [(s, e)]

    out = []
    cur = s
    for ds, de in drops:
        if cur < ds:
            out.append((cur, ds))
        cur = max(cur, de)
    if cur < e:
        out.append((cur, e))
    return out


def _is_short_orphan_fragment(text):
    toks = _token_rows(text)
    if not toks or len(toks) > 2:
        return False
    return all(_is_orphan_like_token(tok) for tok in toks)


def _merge_span_objs(spans):
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: (x["s"], x["e"]))
    out = [dict(spans[0])]
    for cur in spans[1:]:
        prev = out[-1]
        if cur["s"] <= prev["e"]:
            prev["e"] = max(prev["e"], cur["e"])
            prev["display"] = prev["display"] or cur["display"]
        else:
            out.append(dict(cur))
    return out


def _unwrap_env_once(s, name):
    m = re.fullmatch(rf"\\begin\{{{name}\}}([\s\S]*?)\\end\{{{name}\}}", s.strip())
    if not m: return s, False
    body = m.group(1).strip()
    tail_dropped = re.sub(r"(?:\\\\\s*)+$", "", body).strip()
    if "\\\\" in tail_dropped: return s, False
    return tail_dropped, True


def clean_span_latex(s):
    t = str(s or "")
    t = re.sub(r"^\s*latex\s*:\s*", "", t, flags=re.I)
    m = re.fullmatch(r"\s*```(?:latex|tex)?\s*([\s\S]*?)\s*```\s*", t, flags=re.I)
    if m: t = m.group(1)
    t = t.strip().strip("`").strip()

    while True:
        prev = t
        t = re.sub(r"^\s*\\boxed\s*\{([\s\S]*)\}\s*$", r"\1", t).strip()
        t = re.sub(r"^\s*\\\(\s*([\s\S]*?)\s*\\\)\s*$", r"\1", t).strip()
        t = re.sub(r"^\s*\\\[\s*([\s\S]*?)\s*\\\]\s*$", r"\1", t).strip()
        t = re.sub(r"^\s*\$\$\s*([\s\S]*?)\s*\$\$\s*$", r"\1", t).strip()
        t = re.sub(r"^\s*\$\s*([\s\S]*?)\s*\$\s*$", r"\1", t).strip()
        if t.startswith("12730") and t.endswith("12730") and len(t) > 10:
            t = t[5:-5].strip()
        if t.startswith("ㆺ") and t.endswith("ㆺ") and len(t) > 2:
            t = t[1:-1].strip()
        t2, ok = _unwrap_env_once(t, "aligned")
        if ok: t = t2
        t2, ok = _unwrap_env_once(t, "equation")
        if ok: t = t2
        t = re.sub(r"(?:\\\\\s*)+$", "", t).strip()
        if t == prev: break
    return _collapse_inline(t)


def _has_math_operator_text(text):
    t = _tok_core(_collapse_inline(text))
    if not t: return False
    if _is_prose_hyphen(t): return False
    return any(ch in t for ch in OPERATORS) or any(ch in t for ch in "/^_")


def _segment_with_display(clean_text: str) -> tuple[list[tuple[int, int]], list[bool]]:
    toks = _tokenize(clean_text)
    if not toks:
        return [], []
    mathy, strong = [], []
    for i, row in enumerate(toks):
        prv, nxt = (toks[i - 1]["t"] if i else ""), (toks[i + 1]["t"] if i + 1 < len(toks) else "")
        m, s = _mathy_token(row["t"], prv, nxt); mathy.append(m); strong.append(s)
    for i, row in enumerate(toks):
        if mathy[i]: continue
        core = _tok_core(row["t"])
        weak = bool(re.fullmatch(r"\d+(?:\.\d+)?", core))
        weak = weak or (bool(re.fullmatch(r"[A-Za-z]{1,2}", core)) and core.lower() not in SHORT_PROSE)
        weak = weak or core.lower() in MATH_WORDS or core in {"′", "'"}
        if weak and ((i > 0 and mathy[i - 1]) or (i + 1 < len(toks) and mathy[i + 1])): mathy[i] = True

    lines, bounds, pos = clean_text.split("\n"), [], 0
    for line in lines:
        bounds.append((pos, pos + len(line))); pos += len(line) + 1
    dmask = [False] * len(lines)
    for i, line in enumerate(lines):
        ratio = _line_mathy_ratio(line)
        prose = [w for w in re.findall(r"[A-Za-z]{4,}", line) if w.lower() not in MATH_WORDS]
        if ratio >= 0.6 and len(_token_rows(line)) >= 2 and len(prose) <= 1: dmask[i] = True
    for i in range(1, len(lines) - 1):
        if not dmask[i]: continue
        for j in (i - 1, i + 1):
            if re.fullmatch(r"[A-Za-z]|\d{1,3}|[A-Za-z]{1,3}\d{0,2}", lines[j].strip()): dmask[j] = True

    span_objs, i = [], 0
    while i < len(dmask):
        if not dmask[i]: i += 1; continue
        j = i + 1
        while j < len(dmask) and dmask[j]: j += 1
        s, e = _trim_connective_edges(clean_text, bounds[i][0], bounds[j - 1][1])
        if s < e and not _reject_span(clean_text, s, e): span_objs.append({"s": s, "e": e, "display": False})
        i = j

    runs, i = [], 0
    while i < len(toks):
        if not mathy[i]: i += 1; continue
        j = i + 1
        while j < len(toks) and mathy[j]: j += 1
        runs.append((i, j - 1)); i = j
    merged, i = [], 0
    while i < len(runs):
        a, b = runs[i]
        while i + 1 < len(runs):
            c, d = runs[i + 1]; gap = toks[b + 1:c]
            if len(gap) > 2 or not gap: break
            if any(_tok_core(g["t"]).lower() not in CONNECTIVES and _tok_core(g["t"]).lower() != "=" for g in gap): break
            if not any(strong[k] for k in range(a, b + 1)) or not any(strong[k] for k in range(c, d + 1)): break
            b = d; i += 1
        merged.append((a, b)); i += 1
    for a, b in merged:
        while a > 0 and _can_extend(toks[a - 1]["t"]) and _tok_core(toks[a - 1]["t"]) != ".": a -= 1
        while b + 1 < len(toks) and _can_extend(toks[b + 1]["t"]) and _tok_core(toks[b + 1]["t"]) != ".": b += 1
        s, e = _trim_connective_edges(clean_text, toks[a]["s"], toks[b]["e"])
        if s < e and not _reject_span(clean_text, s, e): span_objs.append({"s": s, "e": e, "display": False})

    paragraph_ranges = _paragraph_slices(clean_text)
    paragraphs = []
    tok_i = 0
    for ps, pe in paragraph_ranges:
        while tok_i < len(toks) and toks[tok_i]["e"] <= ps:
            tok_i += 1
        tok_j = tok_i
        tok_ids = []
        while tok_j < len(toks) and toks[tok_j]["s"] < pe:
            tok_ids.append(tok_j)
            tok_j += 1
        tok_i = tok_j
        p_text = clean_text[ps:pe]
        starts_math = bool(tok_ids and mathy[tok_ids[0]])
        has_math = any(mathy[k] for k in tok_ids)
        has_strong_math = any(_is_strong_mathy_token(toks[k]["t"]) for k in tok_ids)
        is_orphan = _is_orphan_paragraph(p_text)
        has_sentence_prose_break = bool(re.search(r"[.!?]\s+[A-Z][a-z]{2,}", p_text))
        is_math_paragraph = (
            has_strong_math
            and _max_prose_word_run(p_text) < 2
            and not has_sentence_prose_break
            and (starts_math or "\n" in p_text)
        )
        paragraphs.append({
            "s": ps,
            "e": pe,
            "starts_math": starts_math,
            "has_math": has_math,
            "is_orphan": is_orphan,
            "is_math_paragraph": is_math_paragraph,
        })
        if is_math_paragraph:
            s, e = _trim_connective_edges(clean_text, ps, pe)
            if s < e and not _reject_span(clean_text, s, e):
                span_objs.append({"s": s, "e": e, "display": True})

    span_objs = _merge_span_objs(span_objs)

    def paragraph_idx_for_pos(pos):
        if pos < 0:
            return None
        for idx, par in enumerate(paragraphs):
            if par["s"] <= pos < par["e"]:
                return idx
        return None

    bridged = []
    i = 0
    while i < len(span_objs):
        cur = dict(span_objs[i])
        while i + 1 < len(span_objs):
            nxt = span_objs[i + 1]
            cur_par = paragraph_idx_for_pos(cur["s"])
            nxt_par = paragraph_idx_for_pos(nxt["s"])
            if cur_par is None or nxt_par is None or cur_par != nxt_par:
                break
            gap_text = clean_text[cur["e"]:nxt["s"]]
            if "\n\n" in gap_text or not _gap_is_orphan_bridge(gap_text):
                break
            cur["e"] = max(cur["e"], nxt["e"])
            cur["display"] = True or cur["display"] or nxt["display"]
            i += 1
        bridged.append(cur)
        i += 1
    span_objs = _merge_span_objs(bridged)

    def build_spans_by_par():
        out = [[] for _ in paragraphs]
        for si, sp in enumerate(span_objs):
            for pi, par in enumerate(paragraphs):
                if sp["s"] < par["e"] and sp["e"] > par["s"]:
                    out[pi].append(si)
        return out

    for pi, par in enumerate(paragraphs):
        if (not par["is_orphan"]) or par["is_math_paragraph"]:
            continue
        spans_by_par = build_spans_by_par()
        prev_i = pi - 1 if pi > 0 else None
        next_i = pi + 1 if pi + 1 < len(paragraphs) else None

        def boundary_span(target_i, side):
            if target_i is None or not spans_by_par[target_i]:
                return None
            tpar = paragraphs[target_i]
            if side == "next":
                si = min(spans_by_par[target_i], key=lambda k: span_objs[k]["s"])
                if _has_prose_word_in_range(clean_text, tpar["s"], span_objs[si]["s"]):
                    return None
                return si
            si = max(spans_by_par[target_i], key=lambda k: span_objs[k]["e"])
            if _has_prose_word_in_range(clean_text, span_objs[si]["e"], tpar["e"]):
                return None
            return si

        target_side = None
        target_i = None
        if pi == 0 and boundary_span(next_i, "next") is not None:
            target_side, target_i = "next", next_i
        else:
            prev_si = boundary_span(prev_i, "prev")
            next_si = boundary_span(next_i, "next")
            prev_has = prev_si is not None
            next_has = next_si is not None
            if prev_has and next_has:
                if paragraphs[next_i]["starts_math"]:
                    target_side, target_i = "next", next_i
                else:
                    target_side, target_i = "prev", prev_i
            elif next_has:
                target_side, target_i = "next", next_i
            elif prev_has:
                target_side, target_i = "prev", prev_i

        if target_i is None:
            continue

        if target_side == "next":
            si = boundary_span(target_i, "next")
            if si is None:
                continue
            span_objs[si]["s"] = min(span_objs[si]["s"], par["s"])
        else:
            si = boundary_span(target_i, "prev")
            if si is None:
                continue
            span_objs[si]["e"] = max(span_objs[si]["e"], par["e"])
        span_objs[si]["display"] = True

    span_objs = _merge_span_objs(span_objs)
    trimmed = []
    for sp in span_objs:
        for ss, ee in _split_span_on_prose_runs(clean_text, sp["s"], sp["e"]):
            s, e = _trim_connective_edges(clean_text, ss, ee)
            if s >= e or _reject_span(clean_text, s, e):
                continue
            if _is_short_orphan_fragment(clean_text[s:e]):
                continue
            if not trimmed or s > trimmed[-1]["e"]:
                trimmed.append({"s": s, "e": e, "display": sp["display"]})
            else:
                trimmed[-1]["e"] = max(trimmed[-1]["e"], e)
                trimmed[-1]["display"] = trimmed[-1]["display"] or sp["display"]

    spans = [(row["s"], row["e"]) for row in trimmed]
    display = [bool(row["display"]) for row in trimmed]
    return spans, display


def segment(clean_text: str) -> list[tuple[int, int]]:
    spans, _ = _segment_with_display(clean_text)
    return spans


def splice(clean_text: str, spans: list[tuple[int, int]], latex: list[str], display_flags: list[bool] | None = None) -> str:
    if len(spans) != len(latex): raise ValueError("spans/latex length mismatch")
    if display_flags is not None and len(display_flags) != len(spans):
        raise ValueError("display_flags/spans length mismatch")
    out, cur = [], 0
    for idx, ((s, e), lx) in enumerate(zip(spans, latex)):
        span_text = clean_text[s:e]
        out.append(clean_text[cur:s])
        if _collapse_inline(lx) == _collapse_inline(span_text) and not _has_math_operator_text(span_text):
            out.append(span_text); cur = e; continue
        left = clean_text.rfind("\n", 0, s) + 1
        right = clean_text.find("\n", e)
        if right < 0: right = len(clean_text)
        line_only = clean_text[left:right].strip() == span_text.strip()
        force_display = bool(display_flags[idx]) if display_flags is not None else False
        is_display = force_display or line_only or "\n" in clean_text[s:e]
        out.append(f"\\[ {lx} \\]" if is_display else f"\\( {lx} \\)")
        cur = e
    out.append(clean_text[cur:])
    return "".join(out)


def run(text: str, convert) -> dict:
    clean, _ = normalize(text)
    spans, display_flags = _segment_with_display(clean)
    span_texts = [clean[s:e] for s, e in spans]
    model_span_texts = [_collapse_inline(s) for s in span_texts]
    latex = [clean_span_latex(s) for s in list(convert(model_span_texts))]
    latex = [lx if lx else span_texts[i] for i, lx in enumerate(latex)]
    return {"clean": clean, "spans": spans, "span_texts": span_texts, "latex": latex, "output": splice(clean, spans, latex, display_flags)}
