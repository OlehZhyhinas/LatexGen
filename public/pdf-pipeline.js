const ZERO_WIDTH = new Set(["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"]);
const BRACKET_CITE_RE = /\[(?:\d+\s*(?:,\s*\d+\s*)*)\]/gu;
const OPERATORS = new Set("=+*/^_±×⋅·≤≥≠≈→∂∇∈∉⊂∪∩∮⌈⌊|-−∣∞");
const CONNECTIVES = new Set(["for", "where", "and", ",", "."]);
const SPAN_EDGE_WORDS = new Set([
  "for", "any", "all", "where", "with", "and", "then", "if", "so", "as", "such", "that",
  "which", "when", "first", "second", "third", "real", "number", "the", "a", "an", "of", "in", "is", "are",
]);
const SUPERS = "⁰¹²³⁴⁵⁶⁷⁸⁹";
const SUBS = "₀₁₂₃₄₅₆₇₈₉";
const GREEK_RE = /[α-ωΑ-ΩπΠμσΣΔΩθΘλΛγΓβΒϵεϕφψΨηΗξΞρΡτΤυΥχΧζΖ]/u;
const MIX_RE = /(?=.*[A-Za-z])(?=.*\d)/u;
const DATE_RE = /^\d{4}-\d{2}-\d{2}$/u;
const UNIT_RE = /^\d+(?:\.\d+)?\s+[A-Za-z]{1,8}s?$/u;
const YEAR_RE = /^(18|19|20)\d{2}$/u;
const MATH_WORDS = new Set(["sin", "cos", "tan", "log", "ln", "lim", "det", "exp", "max", "min"]);
const LOWER_WORD_RE = /^[a-z]{4,}$/u;
const PROSE_WORD_RE = /^[a-z]{3,}$/u;
const PROSE_MATH_NAMES = new Set([
  "sin", "cos", "tan", "ln", "log", "lim", "exp", "max", "min", "det",
  "arg", "sup", "inf", "mod", "gcd", "dx", "dy", "dt", "ds", "dz",
]);
const ORPHAN_SYMBOLS = new Set(["∞", "∫", "∑", "∏", "√"]);
const BRIDGE_TOKENS = new Set(["(", ")", ","]);
const SHORT_PROSE = new Set([
  "in", "on", "of", "to", "at", "by", "as", "an", "is", "we", "if", "so", "it", "its",
  "the", "a", "or", "not", "was", "find", "show", "know", "use", "given", "recall",
  "this", "that", "with", "from", "over", "then", "here", "note", "well",
]);

export const SPAN_SYSTEM_PROMPT = "Convert this fragment of mathematics, copied from a PDF, to LaTeX. The fragment may have lost superscripts, fraction bars and spacing: `x2` means x^2, `12 mv 2` means \\frac{1}{2} m v^2, `eix` means e^{ix}, a number above a number is a fraction, `∑ n=1 ∞` are limits. Output only the LaTeX body, no delimiters, no words, no commentary. Never use \\boxed, \\text, or environments such as aligned or equation. Output one bare expression, exactly the mathematics in the fragment, nothing added and nothing dropped.";

const hasOwn = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);

function pyRound(n) {
  const lo = Math.floor(n);
  const frac = n - lo;
  if (frac < 0.5) return lo;
  if (frac > 0.5) return lo + 1;
  return lo % 2 === 0 ? lo : lo + 1;
}

function fullMatch(re, s) {
  const m = s.match(re);
  return !!m && m[0] === s;
}

function _tokCore(tok) {
  return tok.replace(/^[.,;:!?"`]+|[.,;:!?"`]+$/gu, "");
}

function _tokenRows(line) {
  return line.match(/\S+/gu) || [];
}

function _removeZeroWidth(text) {
  const chars = [];
  const omap = [];
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === "\r" || ZERO_WIDTH.has(ch) || /\p{Cf}/u.test(ch)) continue;
    chars.push(ch);
    omap.push(i);
  }
  return [chars, omap];
}

function _splitLines(chars, omap) {
  const out = [];
  let cc = [];
  let cm = [];
  for (let i = 0; i < chars.length; i++) {
    const ch = chars[i];
    const oi = omap[i];
    if (ch === "\n" || ch === "\f") {
      out.push({ text: cc.join(""), map: cm.slice() });
      cc = [];
      cm = [];
    } else {
      cc.push(ch);
      cm.push(oi);
    }
  }
  out.push({ text: cc.join(""), map: cm.slice() });
  return out;
}

function _lineKey(s) {
  const t = s.replace(/\s+/gu, " ").trim().toLowerCase();
  return t ? t.replace(/[^a-z0-9]+/gu, "") : "";
}

function _stripPiece(text, mp) {
  let i = 0;
  let j = text.length;
  while (i < j && /\s/u.test(text[i])) i++;
  while (j > i && /\s/u.test(text[j - 1])) j--;
  return [text.slice(i, j), mp.slice(i, j)];
}

function _dropRunningHeads(lines) {
  const counts = {};
  for (const row of lines) {
    const t = row.text;
    const key = _lineKey(row.text);
    const alphaCount = (t.match(/\p{L}/gu) || []).length;
    if (key && t.trim().length <= 90 && alphaCount >= 8 && key.length >= 10) {
      counts[key] = (counts[key] || 0) + 1;
    }
  }
  const repeat = new Set(Object.keys(counts).filter((k) => counts[k] >= 2));
  const out = [];
  for (const row of lines) {
    const text = row.text.trim();
    const key = _lineKey(text);
    const toks = text.split(/\s+/u).filter(Boolean);
    const oneLetter = toks.filter((t) => t.length === 1 && /^\p{L}$/u.test(t)).length;
    if (repeat.has(key)) continue;
    if (text.includes("-") && toks.length >= 4 && oneLetter >= Math.floor(toks.length / 2)) continue;
    if (fullMatch(/^[A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)* - [A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*$/u, text)) continue;
    out.push(row);
  }
  return out;
}

function _isPageNumber(lines, i) {
  const s = lines[i].text.trim();
  if (!fullMatch(/^\d{1,3}$/u, s)) return false;
  const prevS = i > 0 ? lines[i - 1].text.trim() : "";
  const nextS = i + 1 < lines.length ? lines[i + 1].text.trim() : "";
  if (!prevS && !nextS) return true;
  if ((i === 0 || !prevS) && nextS && (nextS.match(/[A-Za-z]{4,}/g) || []).length >= 3 && Number(s) <= 20) return true;
  if ((i === lines.length - 1 || !nextS) && prevS && (prevS.match(/[A-Za-z]{4,}/g) || []).length >= 3 && Number(s) <= 20) return true;
  return false;
}

function _dropFooters(lines) {
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    const row = lines[i];
    const s = row.text.trim();
    if (/^\d{1,2}\.\s+\S/u.test(s)) continue;
    if (_isPageNumber(lines, i)) {
      if (i === 0 && _isOrphanParagraph(s) && _tokenRows(s).length <= 2) {
        out.push(row);
        continue;
      }
      continue;
    }
    out.push(row);
  }
  return out;
}

function _splitTwoColRow(text, mp) {
  let m = null;
  for (const hit of text.matchAll(/ {8,}/gu)) {
    if (!m || hit[0].length > m[0].length) m = hit;
  }
  if (!m) return null;
  const start = m.index;
  const end = m.index + m[0].length;
  const [ltxt, lmp] = _stripPiece(text.slice(0, start), mp.slice(0, start));
  const [rtxt, rmp] = _stripPiece(text.slice(end), mp.slice(end));
  return ltxt && rtxt ? [[ltxt, lmp], [rtxt, rmp]] : null;
}

function _deinterleave(lines) {
  let splitRows = lines.map((r, i) => [i, _splitTwoColRow(r.text, r.map)]);
  splitRows = splitRows.filter((x) => x[1] !== null);
  if (splitRows.length < 3) return lines;
  const left = [];
  const right = [];
  const used = new Set();
  for (const [i, sp] of splitRows) {
    used.add(i);
    left.push({ text: sp[0][0], map: sp[0][1] });
    right.push({ text: sp[1][0], map: sp[1][1] });
  }
  if (left.map((x) => x.text).join(" ").length < 20 || right.map((x) => x.text).join(" ").length < 20) return lines;
  const out = [];
  for (let i = 0; i < lines.length; i++) if (!used.has(i)) out.push(lines[i]);
  return out.concat([{ text: "", map: [] }], left, [{ text: "", map: [] }], right);
}

function _removeCitations(text, omap) {
  const keep = Array(text.length).fill(true);
  for (const m of text.matchAll(BRACKET_CITE_RE)) {
    const a = m.index;
    const b = m.index + m[0].length;
    for (let i = a; i < b; i++) keep[i] = false;
  }
  let i = 0;
  while (i < text.length) {
    if (SUPERS.includes(text[i])) {
      let j = i;
      while (j < text.length && SUPERS.includes(text[j])) j++;
      let k = i - 1;
      while (k >= 0 && " .,)".includes(text[k])) k--;
      while (k >= 0 && /\p{L}/u.test(text[k])) k--;
      const wordLen = i - (k + 1);
      const nextCh = j < text.length ? text[j] : "";
      if (wordLen >= 3 && (!nextCh || /\s/u.test(nextCh) || ".,;:!?[]".includes(nextCh))) {
        for (let p = i; p < j; p++) keep[p] = false;
      }
      i = j;
    } else {
      i++;
    }
  }
  const outT = [];
  const outM = [];
  for (let p = 0; p < text.length; p++) {
    if (!keep[p]) continue;
    outT.push(text[p]);
    outM.push(omap[p]);
  }
  return [outT.join(""), outM];
}

function _lineMathyRatio(line) {
  const toks = _tokenRows(line);
  if (!toks.length) return 0;
  let n = 0;
  for (const tok of toks) {
    const core = _tokCore(tok);
    if (DATE_RE.test(core)) continue;
    if ([...core].some((c) => OPERATORS.has(c)) || GREEK_RE.test(core) || [...core].some((c) => SUPERS.includes(c) || SUBS.includes(c))) {
      n++;
      continue;
    }
    if (core.includes("/") || core.includes("^") || core.includes("_") || MIX_RE.test(core)) {
      n++;
      continue;
    }
    if (fullMatch(/^\d+(?:\.\d+)?$/u, core) || MATH_WORDS.has(core.toLowerCase())) n++;
  }
  return n / toks.length;
}

function _looksResidueLine(line) {
  const s = line.trim();
  if (!s) return false;
  const toks = _tokenRows(s);
  if (toks.length > 3 || _lineMathyRatio(s) >= 0.5) return false;
  return ",;:)]".includes(s[0]) || new Set(["the", "and", "or", "for", "where", ", the"]).has(s.toLowerCase());
}

function _joinProse(lines) {
  if (!lines.length) return ["", []];
  const outT = [...lines[0].text];
  const outM = lines[0].map.slice();
  for (let i = 1; i < lines.length; i++) {
    const txt = lines[i].text;
    const mp = lines[i].map;
    if (outT.length && outT[outT.length - 1] === "-" && txt && /[a-z]/u.test(txt[0])) {
      outT.pop();
      outM.pop();
    } else {
      outT.push(" ");
      outM.push(-1);
    }
    outT.push(...txt);
    outM.push(...mp);
  }
  return [outT.join(""), outM];
}

function _joinMath(lines) {
  const main = [];
  const tail = [];
  for (const row of lines) (_looksResidueLine(row.text) ? tail : main).push(row);
  const outT = [];
  const outM = [];
  for (const row of main.concat(tail)) {
    if (outT.length) {
      outT.push(" ");
      outM.push(-1);
    }
    outT.push(...row.text);
    outM.push(...row.map);
  }
  return [outT.join(""), outM];
}

function _collapseSpaces(text, omap) {
  const outT = [];
  const outM = [];
  let prevSpace = false;
  for (let i = 0; i < text.length; i++) {
    const oi = omap[i];
    let ch = text[i];
    if (ch === "\t") ch = " ";
    if (ch === "\n") {
      while (outT.length && outT[outT.length - 1] === " ") {
        outT.pop();
        outM.pop();
      }
      outT.push("\n");
      outM.push(oi);
      prevSpace = false;
      continue;
    }
    if (/\s/u.test(ch)) {
      if (prevSpace) continue;
      outT.push(" ");
      outM.push(oi);
      prevSpace = true;
    } else {
      outT.push(ch);
      outM.push(oi);
      prevSpace = false;
    }
  }
  const text2 = [];
  const map2 = [];
  let nl = 0;
  for (let i = 0; i < outT.length; i++) {
    const ch = outT[i];
    const oi = outM[i];
    nl = ch === "\n" ? nl + 1 : 0;
    if (nl > 2) continue;
    text2.push(ch);
    map2.push(oi);
  }
  let a = 0;
  let b = text2.length;
  while (a < b && /\s/u.test(text2[a])) a++;
  while (b > a && /\s/u.test(text2[b - 1])) b--;
  return [text2.slice(a, b).join(""), map2.slice(a, b)];
}

export function normalize(text) {
  const [chars, omap] = _removeZeroWidth(text);
  const lines = _deinterleave(_dropFooters(_dropRunningHeads(_splitLines(chars, omap))));
  const blocks = [];
  let cur = [];
  for (const row of lines) {
    if (row.text.trim()) {
      const [t, m] = _stripPiece(row.text, row.map);
      cur.push({ text: t, map: m });
    } else {
      if (cur.length) {
        blocks.push(cur);
        cur = [];
      }
      blocks.push([]);
    }
  }
  if (cur.length) blocks.push(cur);
  const firstNonEmpty = blocks.findIndex((block) => block.length > 0);
  if (firstNonEmpty >= 0) {
    const block = blocks[firstNonEmpty];
    if (block.length >= 2 && _isOrphanParagraph(block[0].text) && _tokenRows(block[0].text).length <= 2) {
      blocks.splice(firstNonEmpty, 1, [block[0]], [], block.slice(1));
    }
  }
  const outT = [];
  const outM = [];
  let wrote = false;
  for (const block of blocks) {
    if (!block.length) {
      if (wrote) {
        outT.push("\n", "\n");
        outM.push(-1, -1);
      }
      wrote = false;
      continue;
    }
    const tokN = block.reduce((acc, r) => acc + _tokenRows(r.text).length, 0);
    const ratios = block.map((r) => _lineMathyRatio(r.text));
    const tokM = block.reduce((acc, r, i) => acc + pyRound(ratios[i] * _tokenRows(r.text).length), 0);
    const short = block.filter((r) => _tokenRows(r.text).length <= 2).length;
    const hasOp = block.some((r) => /[=+*/^_≤≥≠≈∫∑∏√|]/u.test(r.text));
    const isMath = (tokN > 0 && (tokM / tokN) >= 0.6) || (hasOp && short >= 1 && block.length >= 2);
    const [bt, bm] = isMath ? _joinMath(block) : _joinProse(block);
    if (outT.length && outT[outT.length - 1] !== "\n") {
      outT.push("\n");
      outM.push(-1);
    }
    outT.push(...bt);
    outM.push(...bm);
    wrote = true;
  }
  const [clean, cmap] = _removeCitations(outT.join(""), outM);
  return _collapseSpaces(clean, cmap);
}

function _tokenize(text) {
  const out = [];
  const starts = [0];
  let lineI = 0;
  for (let i = 0; i < text.length; i++) if (text[i] === "\n") starts.push(i + 1);
  for (const m of text.matchAll(/\S+/gu)) {
    const s = m.index;
    const e = m.index + m[0].length;
    while (lineI + 1 < starts.length && s >= starts[lineI + 1]) lineI++;
    out.push({ t: m[0], s, e, line: lineI });
  }
  return out;
}

function _hasOp(tok) {
  if (!tok) return false;
  const c = _tokCore(tok);
  if (!c) return false;
  if (_isProseHyphen(c)) return false;
  return [...c].some((ch) => OPERATORS.has(ch)) || c.includes("/") || c.includes("^") || c.includes("_") || fullMatch(/^\(?[A-Za-z0-9]+\)?'$/u, c);
}

function _isProseHyphen(core) {
  const m = core.match(/^([a-z]+)-([a-z]+)$/u);
  if (!m) return false;
  const a = m[1];
  const b = m[2];
  if (a.length >= 2 && b.length >= 2) return true;
  if ((a.length === 1 && b.length >= 2) || (b.length === 1 && a.length >= 2)) return true;
  return false;
}

function _rawMathy(tok) {
  const core = _tokCore(tok);
  const low = core.toLowerCase();
  if (!core || DATE_RE.test(core) || new Set(["for", "where", "and", "or"]).has(low)) return [false, false];
  if (_isProseHyphen(core)) return [false, false];
  if ([...core].some((c) => OPERATORS.has(c)) || GREEK_RE.test(core) || [...core].some((c) => SUPERS.includes(c) || SUBS.includes(c))) return [true, true];
  if (core.includes("/") || MIX_RE.test(core) || fullMatch(/^[A-Za-z0-9]{1,3}-[A-Za-z0-9]{1,3}$/u, core)) return [true, true];
  if (/^d[A-Za-z]$/u.test(core) || fullMatch(/^[A-Za-z]{1,6}\([^)]*\)$/u, core)) return [true, true];
  if (core.includes("'") && !low.endsWith("'s") && fullMatch(/^[()A-Za-z]{1,8}'$/u, core)) return [true, true];
  if (MATH_WORDS.has(low) || fullMatch(/^[A-Za-z]{2,3}$/u, core) || fullMatch(/^\d+(?:\.\d+)?$/u, core)) return [true, false];
  if (core.length === 1 && /^\p{L}$/u.test(core)) return [true, false];
  if (core.includes("(") && core.includes(")") && /[=+/*^_]/u.test(core)) return [true, true];
  return [false, false];
}

function _mathyToken(tok, prevTok, nextTok) {
  const core = _tokCore(tok);
  const [mathy, strong] = _rawMathy(tok);
  if (!mathy) return [false, false];
  const prev = prevTok ? _tokCore(prevTok).toLowerCase() : "";
  const nxt = nextTok ? _tokCore(nextTok).toLowerCase() : "";
  if (core.length === 1 && new Set(["a", "A", "I"]).has(core) && nextTok && fullMatch(/^[a-z]{2,}$/u, _tokCore(nextTok))) return [false, false];
  if (core.length === 1 && /^\p{L}$/u.test(core)) {
    const near = _hasOp(prevTok) || _hasOp(nextTok) || (prev.length === 1 && /^\p{L}$/u.test(prev)) || (nxt.length === 1 && /^\p{L}$/u.test(nxt));
    if (!near) return [false, false];
  }
  if (fullMatch(/^\d+(?:\.\d+)?$/u, core)) {
    const nearMath = _hasOp(prevTok) || _hasOp(nextTok);
    let nearVar = fullMatch(/^[A-Za-z]{1,2}$/u, prev) || fullMatch(/^[A-Za-z]{1,2}$/u, nxt);
    if (MATH_WORDS.has(prev) || MATH_WORDS.has(nxt)) nearVar = true;
    if (!nearMath && !nearVar) return [false, false];
  }
  if (fullMatch(/^[A-Za-z]{2,3}$/u, core)) {
    if (MATH_WORDS.has(core.toLowerCase())) return [true, strong || _hasOp(prevTok) || _hasOp(nextTok)];
    const nearMath = _hasOp(prevTok) || _hasOp(nextTok);
    const nearNum = fullMatch(/^\d+(?:\.\d+)?$/u, prev) || fullMatch(/^\d+(?:\.\d+)?$/u, nxt);
    if ((!nearMath && !nearNum) || SHORT_PROSE.has(core.toLowerCase())) return [false, false];
  }
  if (YEAR_RE.test(core)) return [false, false];
  return [true, strong];
}

function _spanTrim(text, s, e) {
  while (e > s && ".,;:!?".includes(text[e - 1])) e--;
  while (s < e && /\s/u.test(text[s])) s++;
  while (e > s && /\s/u.test(text[e - 1])) e--;
  return [s, e];
}

function _trimConnectiveEdges(text, s, e) {
  [s, e] = _spanTrim(text, s, e);
  if (s >= e) return [s, e];

  for (;;) {
    const toks = Array.from(text.slice(s, e).matchAll(/\S+/gu));
    if (!toks.length) return [s, e];
    const first = _tokCore(toks[0][0]).toLowerCase();
    if (!SPAN_EDGE_WORDS.has(first)) break;
    let ns = s + toks[0].index + toks[0][0].length;
    [ns] = _spanTrim(text, ns, e);
    const rem = Array.from(text.slice(ns, e).matchAll(/\S+/gu));
    if (!rem.length) break;
    if (_hasOp(rem[0][0])) break;
    s = ns;
  }

  for (;;) {
    const toks = Array.from(text.slice(s, e).matchAll(/\S+/gu));
    if (!toks.length) return [s, e];
    const last = _tokCore(toks[toks.length - 1][0]).toLowerCase();
    if (!SPAN_EDGE_WORDS.has(last)) break;
    let ne = s + toks[toks.length - 1].index;
    [, ne] = _spanTrim(text, s, ne);
    const rem = Array.from(text.slice(s, ne).matchAll(/\S+/gu));
    if (!rem.length) break;
    if (_hasOp(rem[rem.length - 1][0])) break;
    e = ne;
  }

  return _spanTrim(text, s, e);
}

function _rejectSpan(text, s, e) {
  const piece = text.slice(s, e).replace(/\s+/gu, " ").trim();
  if (!piece || new Set(["-", "−", "+", "=", "*", "/", "|", "∣"]).has(piece)) return true;
  if (YEAR_RE.test(piece)) return true;
  if (UNIT_RE.test(piece) && !piece.includes("/")) return true;
  if (fullMatch(/^\d{1,3}$/u, piece)) {
    const left = text.slice(Math.max(0, s - 32), s).toLowerCase();
    return left.includes("room") || left.includes("chapter") ? true : true;
  }
  if (fullMatch(/^\d+(?:\.\d+)?$/u, piece)) return true;
  if (fullMatch(/^[A-Za-z' ]+$/u, piece)) {
    const words = piece.replace(/'/gu, " ").split(/\s+/u).filter(Boolean);
    if (words.length && words.every((w) => !MATH_WORDS.has(w.toLowerCase()))) return true;
  }
  if (new Set(["x", "t", "n", "c", "e", "A"]).has(piece)) {
    const left = text.slice(Math.max(0, s - 32), s).toLowerCase();
    if (["variable", "time", "letter", "name"].some((w) => left.includes(w))) return true;
  }
  return false;
}

function _canExtend(tok) {
  const core = _tokCore(tok);
  if (!core) return false;
  const low = core.toLowerCase();
  if (SHORT_PROSE.has(low) && !MATH_WORDS.has(low)) return false;
  if ([...core].some((c) => OPERATORS.has(c)) || GREEK_RE.test(core) || [...core].some((c) => SUPERS.includes(c) || SUBS.includes(c))) return true;
  if (fullMatch(/^\d+(?:\.\d+)?$/u, core) || fullMatch(/^[A-Za-z]{1,3}$/u, core)) return true;
  if (core.includes("/") || core.includes("^") || core.includes("_")) return true;
  if (core.includes("(") || core.includes(")") || core.includes("[") || core.includes("]")) return /[A-Za-z0-9]/u.test(core);
  return false;
}

function _collapseInline(s) {
  return String(s || "").replace(/\s+/gu, " ").trim();
}

function _isStrongMathyToken(tok) {
  const core = _tokCore(tok);
  if (!core || DATE_RE.test(core) || _isProseHyphen(core)) return false;
  if (core.includes("=") || [...core].some((ch) => ORPHAN_SYMBOLS.has(ch))) return true;
  if (GREEK_RE.test(core)) return true;
  return _hasOp(core);
}

function _isOrphanLikeToken(tok) {
  let core = _tokCore(tok);
  if (!core) core = tok.trim();
  if (!core) return false;
  if (fullMatch(/^\d+$/u, core)) return true;
  if (fullMatch(GREEK_RE, core)) return true;
  if (core.length === 1 && (ORPHAN_SYMBOLS.has(core) || OPERATORS.has(core))) return true;
  return core.length <= 3;
}

function _isOrphanParagraph(parText) {
  const toks = _tokenRows(parText);
  if (!toks.length) return false;
  if (toks.some((tok) => LOWER_WORD_RE.test(_tokCore(tok) || ""))) return false;
  return toks.every((tok) => _isOrphanLikeToken(tok));
}

function _isProseWordToken(tok) {
  const core = (_tokCore(tok) || "").toLowerCase();
  if (!PROSE_WORD_RE.test(core)) return false;
  return !PROSE_MATH_NAMES.has(core);
}

function _maxProseWordRun(parText) {
  let run = 0;
  let best = 0;
  for (const tok of _tokenRows(parText)) {
    if (_isProseWordToken(tok)) {
      run++;
      best = Math.max(best, run);
    } else {
      run = 0;
    }
  }
  return best;
}

function _paragraphSlices(text) {
  if (!text) return [];
  const out = [];
  let i = 0;
  while (i < text.length) {
    let j = text.indexOf("\n\n", i);
    if (j < 0) j = text.length;
    out.push([i, j]);
    i = j + 2;
  }
  return out;
}

function _gapIsOrphanBridge(gapText) {
  const toks = _tokenRows(gapText);
  if (!toks.length) return false;
  for (const tok of toks) {
    if (_isProseWordToken(tok)) return false;
    const core = _tokCore(tok);
    if (BRIDGE_TOKENS.has(core)) continue;
    if (_isOrphanLikeToken(tok)) continue;
    return false;
  }
  return true;
}

function _hasProseWordInRange(text, s, e) {
  if (s >= e) return false;
  for (const tok of _tokenRows(text.slice(s, e))) {
    if (_isProseWordToken(tok)) return true;
  }
  return false;
}

function _splitSpanOnProseRuns(text, s, e) {
  const toks = Array.from(text.slice(s, e).matchAll(/\S+/gu)).map((m) => ({
    s: s + m.index,
    e: s + m.index + m[0].length,
    t: m[0],
  }));
  if (!toks.length) return [[s, e]];

  const drops = [];
  let runStart = null;
  let runEnd = null;
  let runLen = 0;
  for (const tok of toks) {
    if (_isProseWordToken(tok.t)) {
      if (runStart === null) runStart = tok.s;
      runEnd = tok.e;
      runLen++;
      continue;
    }
    if (runLen >= 2) drops.push([runStart, runEnd]);
    runStart = null;
    runEnd = null;
    runLen = 0;
  }
  if (runLen >= 2) drops.push([runStart, runEnd]);
  if (!drops.length) return [[s, e]];

  const out = [];
  let cur = s;
  for (const [ds, de] of drops) {
    if (cur < ds) out.push([cur, ds]);
    cur = Math.max(cur, de);
  }
  if (cur < e) out.push([cur, e]);
  return out;
}

function _isShortOrphanFragment(text) {
  const toks = _tokenRows(text);
  if (!toks.length || toks.length > 2) return false;
  return toks.every((tok) => _isOrphanLikeToken(tok));
}

function _mergeSpanObjs(spans) {
  if (!spans.length) return [];
  const sorted = [...spans].sort((a, b) => (a.s - b.s) || (a.e - b.e));
  const out = [{ ...sorted[0] }];
  for (let i = 1; i < sorted.length; i++) {
    const cur = sorted[i];
    const prev = out[out.length - 1];
    if (cur.s <= prev.e) {
      prev.e = Math.max(prev.e, cur.e);
      prev.display = prev.display || cur.display;
    } else {
      out.push({ ...cur });
    }
  }
  return out;
}

function _unwrapEnvOnce(s, name) {
  const re = new RegExp(`^\\\\begin\\{${name}\\}([\\s\\S]*?)\\\\end\\{${name}\\}$`, "u");
  const m = s.trim().match(re);
  if (!m) return [s, false];
  const body = m[1].trim();
  const tailDropped = body.replace(/(?:\\\\\s*)+$/gu, "").trim();
  if (tailDropped.includes("\\\\")) return [s, false];
  return [tailDropped, true];
}

export function cleanSpanLatex(s) {
  let t = String(s || "");
  t = t.replace(/^\s*latex\s*:\s*/iu, "");
  const fenced = t.match(/^\s*```(?:latex|tex)?\s*([\s\S]*?)\s*```\s*$/iu);
  if (fenced) t = fenced[1];
  t = t.trim().replace(/^`+|`+$/gu, "").trim();

  for (;;) {
    const prev = t;
    t = t.replace(/^\s*\\boxed\s*\{([\s\S]*)\}\s*$/u, "$1").trim();
    t = t.replace(/^\s*\\\(\s*([\s\S]*?)\s*\\\)\s*$/u, "$1").trim();
    t = t.replace(/^\s*\\\[\s*([\s\S]*?)\s*\\\]\s*$/u, "$1").trim();
    t = t.replace(/^\s*\$\$\s*([\s\S]*?)\s*\$\$\s*$/u, "$1").trim();
    t = t.replace(/^\s*\$\s*([\s\S]*?)\s*\$\s*$/u, "$1").trim();
    if (t.startsWith("12730") && t.endsWith("12730") && t.length > 10) t = t.slice(5, -5).trim();
    if (t.startsWith("ㆺ") && t.endsWith("ㆺ") && t.length > 2) t = t.slice(1, -1).trim();
    let ok;
    [t, ok] = _unwrapEnvOnce(t, "aligned");
    if (!ok) [t, ok] = _unwrapEnvOnce(t, "equation");
    t = t.replace(/(?:\\\\\s*)+$/gu, "").trim();
    if (t === prev) break;
  }
  return _collapseInline(t);
}

function _hasMathOperatorText(text) {
  const t = _tokCore(_collapseInline(text));
  if (!t) return false;
  if (_isProseHyphen(t)) return false;
  return [...t].some((ch) => OPERATORS.has(ch)) || [...t].some((ch) => "/^_".includes(ch));
}

function _segmentWithDisplay(cleanText) {
  const toks = _tokenize(cleanText);
  if (!toks.length) return { spans: [], displayFlags: [] };
  const mathy = [];
  const strong = [];
  for (let i = 0; i < toks.length; i++) {
    const prv = i ? toks[i - 1].t : "";
    const nxt = i + 1 < toks.length ? toks[i + 1].t : "";
    const [m, s] = _mathyToken(toks[i].t, prv, nxt);
    mathy.push(m);
    strong.push(s);
  }
  for (let i = 0; i < toks.length; i++) {
    if (mathy[i]) continue;
    const core = _tokCore(toks[i].t);
    let weak = fullMatch(/^\d+(?:\.\d+)?$/u, core);
    weak = weak || (fullMatch(/^[A-Za-z]{1,2}$/u, core) && !SHORT_PROSE.has(core.toLowerCase()));
    weak = weak || MATH_WORDS.has(core.toLowerCase()) || new Set(["′", "'"]).has(core);
    if (weak && ((i > 0 && mathy[i - 1]) || (i + 1 < toks.length && mathy[i + 1]))) mathy[i] = true;
  }

  const lines = cleanText.split("\n");
  const bounds = [];
  let pos = 0;
  for (const line of lines) {
    bounds.push([pos, pos + line.length]);
    pos += line.length + 1;
  }
  const dmask = Array(lines.length).fill(false);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const ratio = _lineMathyRatio(line);
    const prose = (line.match(/[A-Za-z]{4,}/gu) || []).filter((w) => !MATH_WORDS.has(w.toLowerCase()));
    if (ratio >= 0.6 && _tokenRows(line).length >= 2 && prose.length <= 1) dmask[i] = true;
  }
  for (let i = 1; i < lines.length - 1; i++) {
    if (!dmask[i]) continue;
    for (const j of [i - 1, i + 1]) {
      if (fullMatch(/^(?:[A-Za-z]|\d{1,3}|[A-Za-z]{1,3}\d{0,2})$/u, lines[j].trim())) dmask[j] = true;
    }
  }

  const spanObjs = [];
  let i = 0;
  while (i < dmask.length) {
    if (!dmask[i]) {
      i++;
      continue;
    }
    let j = i + 1;
    while (j < dmask.length && dmask[j]) j++;
    let [s, e] = _trimConnectiveEdges(cleanText, bounds[i][0], bounds[j - 1][1]);
    if (s < e && !_rejectSpan(cleanText, s, e)) spanObjs.push({ s, e, display: false });
    i = j;
  }

  const runs = [];
  i = 0;
  while (i < toks.length) {
    if (!mathy[i]) {
      i++;
      continue;
    }
    let j = i + 1;
    while (j < toks.length && mathy[j]) j++;
    runs.push([i, j - 1]);
    i = j;
  }
  const merged = [];
  i = 0;
  while (i < runs.length) {
    const a = runs[i][0];
    let b = runs[i][1];
    while (i + 1 < runs.length) {
      const c = runs[i + 1][0];
      const d = runs[i + 1][1];
      const gap = toks.slice(b + 1, c);
      if (gap.length > 2 || !gap.length) break;
      if (gap.some((g) => {
        const t = _tokCore(g.t).toLowerCase();
        return !CONNECTIVES.has(t) && t !== "=";
      })) break;
      let leftStrong = false;
      let rightStrong = false;
      for (let k = a; k <= b; k++) if (strong[k]) { leftStrong = true; break; }
      for (let k = c; k <= d; k++) if (strong[k]) { rightStrong = true; break; }
      if (!leftStrong || !rightStrong) break;
      b = d;
      i++;
    }
    merged.push([a, b]);
    i++;
  }
  for (const m of merged) {
    let a = m[0];
    let b = m[1];
    while (a > 0 && _canExtend(toks[a - 1].t) && _tokCore(toks[a - 1].t) !== ".") a--;
    while (b + 1 < toks.length && _canExtend(toks[b + 1].t) && _tokCore(toks[b + 1].t) !== ".") b++;
    let [s, e] = _trimConnectiveEdges(cleanText, toks[a].s, toks[b].e);
    if (s < e && !_rejectSpan(cleanText, s, e)) spanObjs.push({ s, e, display: false });
  }

  const paragraphRanges = _paragraphSlices(cleanText);
  const paragraphs = [];
  let tokI = 0;
  for (const [ps, pe] of paragraphRanges) {
    while (tokI < toks.length && toks[tokI].e <= ps) tokI++;
    let tokJ = tokI;
    const tokIds = [];
    while (tokJ < toks.length && toks[tokJ].s < pe) {
      tokIds.push(tokJ);
      tokJ++;
    }
    tokI = tokJ;
    const pText = cleanText.slice(ps, pe);
    const startsMath = !!(tokIds.length && mathy[tokIds[0]]);
    const hasMath = tokIds.some((k) => mathy[k]);
    const hasStrongMath = tokIds.some((k) => _isStrongMathyToken(toks[k].t));
    const isOrphan = _isOrphanParagraph(pText);
    const hasSentenceProseBreak = /[.!?]\s+[A-Z][a-z]{2,}/u.test(pText);
    const isMathParagraph = hasStrongMath && _maxProseWordRun(pText) < 2 && !hasSentenceProseBreak && (startsMath || pText.includes("\n"));
    paragraphs.push({ s: ps, e: pe, startsMath, hasMath, isOrphan, isMathParagraph });
    if (isMathParagraph) {
      const [s, e] = _trimConnectiveEdges(cleanText, ps, pe);
      if (s < e && !_rejectSpan(cleanText, s, e)) {
        spanObjs.push({ s, e, display: true });
      }
    }
  }

  let mergedSpans = _mergeSpanObjs(spanObjs);

  const paragraphIdxForPos = (pos) => {
    if (pos < 0) return null;
    for (let p = 0; p < paragraphs.length; p++) {
      if (paragraphs[p].s <= pos && pos < paragraphs[p].e) return p;
    }
    return null;
  };

  const bridged = [];
  i = 0;
  while (i < mergedSpans.length) {
    const cur = { ...mergedSpans[i] };
    while (i + 1 < mergedSpans.length) {
      const nxt = mergedSpans[i + 1];
      const curPar = paragraphIdxForPos(cur.s);
      const nxtPar = paragraphIdxForPos(nxt.s);
      if (curPar === null || nxtPar === null || curPar !== nxtPar) break;
      const gapText = cleanText.slice(cur.e, nxt.s);
      if (gapText.includes("\n\n") || !_gapIsOrphanBridge(gapText)) break;
      cur.e = Math.max(cur.e, nxt.e);
      cur.display = true;
      i++;
    }
    bridged.push(cur);
    i++;
  }
  mergedSpans = _mergeSpanObjs(bridged);

  const buildSpansByPar = () => {
    const out = Array.from({ length: paragraphs.length }, () => []);
    for (let si = 0; si < mergedSpans.length; si++) {
      const sp = mergedSpans[si];
      for (let pi = 0; pi < paragraphs.length; pi++) {
        const par = paragraphs[pi];
        if (sp.s < par.e && sp.e > par.s) out[pi].push(si);
      }
    }
    return out;
  };

  for (let pi = 0; pi < paragraphs.length; pi++) {
    const par = paragraphs[pi];
    if (!par.isOrphan || par.isMathParagraph) continue;
    const spansByPar = buildSpansByPar();
    const prevI = pi > 0 ? pi - 1 : null;
    const nextI = pi + 1 < paragraphs.length ? pi + 1 : null;

    const boundarySpan = (targetI, side) => {
      if (targetI === null || spansByPar[targetI].length === 0) return null;
      const tpar = paragraphs[targetI];
      if (side === "next") {
        const si = spansByPar[targetI].reduce((best, cur) => (mergedSpans[cur].s < mergedSpans[best].s ? cur : best), spansByPar[targetI][0]);
        if (_hasProseWordInRange(cleanText, tpar.s, mergedSpans[si].s)) return null;
        return si;
      }
      const si = spansByPar[targetI].reduce((best, cur) => (mergedSpans[cur].e > mergedSpans[best].e ? cur : best), spansByPar[targetI][0]);
      if (_hasProseWordInRange(cleanText, mergedSpans[si].e, tpar.e)) return null;
      return si;
    };

    let targetSide = null;
    let targetI = null;
    if (pi === 0 && boundarySpan(nextI, "next") !== null) {
      targetSide = "next";
      targetI = nextI;
    } else {
      const prevSi = boundarySpan(prevI, "prev");
      const nextSi = boundarySpan(nextI, "next");
      const prevHas = prevSi !== null;
      const nextHas = nextSi !== null;
      if (prevHas && nextHas) {
        if (paragraphs[nextI].startsMath) {
          targetSide = "next";
          targetI = nextI;
        } else {
          targetSide = "prev";
          targetI = prevI;
        }
      } else if (nextHas) {
        targetSide = "next";
        targetI = nextI;
      } else if (prevHas) {
        targetSide = "prev";
        targetI = prevI;
      }
    }
    if (targetI === null) continue;

    if (targetSide === "next") {
      const si = boundarySpan(targetI, "next");
      if (si === null) continue;
      mergedSpans[si].s = Math.min(mergedSpans[si].s, par.s);
      mergedSpans[si].display = true;
    } else {
      const si = boundarySpan(targetI, "prev");
      if (si === null) continue;
      mergedSpans[si].e = Math.max(mergedSpans[si].e, par.e);
      mergedSpans[si].display = true;
    }
  }

  mergedSpans = _mergeSpanObjs(mergedSpans);
  const trimmed = [];
  for (const sp of mergedSpans) {
    for (const [ss, ee] of _splitSpanOnProseRuns(cleanText, sp.s, sp.e)) {
      const [s, e] = _trimConnectiveEdges(cleanText, ss, ee);
      if (s >= e || _rejectSpan(cleanText, s, e)) continue;
      if (_isShortOrphanFragment(cleanText.slice(s, e))) continue;
      if (!trimmed.length || s > trimmed[trimmed.length - 1].e) {
        trimmed.push({ s, e, display: sp.display });
      } else {
        trimmed[trimmed.length - 1].e = Math.max(trimmed[trimmed.length - 1].e, e);
        trimmed[trimmed.length - 1].display = trimmed[trimmed.length - 1].display || sp.display;
      }
    }
  }

  return {
    spans: trimmed.map((row) => [row.s, row.e]),
    displayFlags: trimmed.map((row) => !!row.display),
  };
}

export function segment(cleanText) {
  return _segmentWithDisplay(cleanText).spans;
}

export function splice(cleanText, spans, latex, displayFlags = null) {
  if (spans.length !== latex.length) throw new Error("spans/latex length mismatch");
  if (displayFlags !== null && displayFlags.length !== spans.length) {
    throw new Error("displayFlags/spans length mismatch");
  }
  const out = [];
  let cur = 0;
  for (let i = 0; i < spans.length; i++) {
    const s = spans[i][0];
    const e = spans[i][1];
    const lx = latex[i];
    const spanText = cleanText.slice(s, e);
    out.push(cleanText.slice(cur, s));
    if (_collapseInline(lx) === _collapseInline(spanText) && !_hasMathOperatorText(spanText)) {
      out.push(spanText);
      cur = e;
      continue;
    }
    const left = cleanText.lastIndexOf("\n", s - 1) + 1;
    let right = cleanText.indexOf("\n", e);
    if (right < 0) right = cleanText.length;
    const lineOnly = cleanText.slice(left, right).trim() === spanText.trim();
    const forceDisplay = displayFlags !== null ? !!displayFlags[i] : false;
    const isDisplay = forceDisplay || lineOnly || spanText.includes("\n");
    out.push(isDisplay ? `\\[ ${lx} \\]` : `\\( ${lx} \\)`);
    cur = e;
  }
  out.push(cleanText.slice(cur));
  return out.join("");
}

function _sentenceLooksProse(sentence) {
  const toks = _tokenRows(sentence);
  if (toks.length < 4) return false;
  const words = (sentence.match(/[A-Za-z]{3,}/gu) || []).filter((w) => !MATH_WORDS.has(w.toLowerCase()));
  if (words.length < 2) return false;
  return _lineMathyRatio(sentence) < 0.6;
}

export function looksLikePdfPaste(text) {
  if (typeof text !== "string" || !text.trim()) return false;
  const [clean] = normalize(text);
  if (!clean) return false;
  const spans = segment(clean);
  if (!spans.length) return false;
  const sentences = clean.split(/[.!?]+/u).map((s) => s.trim()).filter(Boolean);
  const proseSentences = sentences.filter(_sentenceLooksProse);
  if (proseSentences.length < 2) return false;
  const proseWords = (clean.match(/[A-Za-z]{3,}/gu) || []).filter((w) => !MATH_WORDS.has(w.toLowerCase()));
  if (!proseWords.length) return false;
  if (_lineMathyRatio(clean.replace(/\n/gu, " ")) >= 0.9) return false;
  return true;
}

export async function run(text, convert) {
  const [clean] = normalize(text);
  const seg = _segmentWithDisplay(clean);
  const spans = seg.spans;
  const displayFlags = seg.displayFlags;
  const spanTexts = spans.map(([s, e]) => clean.slice(s, e));
  const modelSpanTexts = spanTexts.map((s) => _collapseInline(s));
  const latexRaw = Array.from(await convert(modelSpanTexts));
  const latex = latexRaw.map((s, i) => {
    const cleaned = cleanSpanLatex(s);
    return cleaned || spanTexts[i];
  });
  return { clean, spans, spanTexts, latex, output: splice(clean, spans, latex, displayFlags) };
}
