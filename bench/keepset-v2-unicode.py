#!/usr/bin/env python3
"""Unicode-property-based classifiers for keep-set v2's Layer 1 (maths/Greek
symbol tokens) and Layer 2 (ASCII-letter presence). Derived from Unicode
General Category and block ranges -- not a hand-typed symbol list, per the
brief's requirement -- with one deliberate, documented exception: a short
named-exclusion list for the handful of non-mathematical codepoints that
happen to sit inside otherwise-mathematical Unicode blocks (see
LETTERLIKE_SYMBOLS_NON_MATH_EXCEPTIONS and GREEK_NON_LETTER_PUNCTUATION
below). Those are block-membership corrections, not a symbol allowlist.

Maths usage vs natural-language usage, separated:
  - Greek and Coptic block (U+0370-03FF): the *unaccented* letters (used as
    the standard maths/physics variable alphabet: alpha, beta, phi, omega,
    ...) are classified as maths. Modern Greek orthography adds acute-accent
    ("tonos") and dialytika diacritics to indicate stress/diaeresis in actual
    Greek-language text (\u03ac \u03ad \u03ae \u03af \u03cc \u03cd \u03ce,
    \u03ca \u03cb, and their capitals) -- those precomposed codepoints are
    excluded by name (every one of them has "WITH TONOS" or "WITH DIALYTIKA"
    in its Unicode name), so plain Greek prose gets filtered out while the
    unaccented maths alphabet does not. A few Greek *punctuation* codepoints
    in the same block (the Greek question mark, ano teleia, tonos-as-a-
    standalone-accent, numeral sign) are letters of neither language nor
    maths and are excluded by category/name.
  - CJK/other-script punctuation (ideographic comma/full stop, fullwidth
    punctuation, etc.) is never in any of the ranges below, so it is not
    reachable by this classifier at all -- it falls through to Layer 2's
    wholesale drop, which is what the brief asks for.
"""
import unicodedata as ud

# Unicode General Category "Sm" (Math_Symbol) alone covers the large
# majority of operators/relations/arrows/set-ops used in maths notation,
# in any block: +-*/=<> (ASCII, already covered by the byte-alphabet and
# ASCII<=3-char layers), plus non-ASCII ones like x, /, +/-, not-sign,
# for-all/exists, summation/integral/product, <=/>=/!=, subset/superset,
# and most arrows (checked explicitly below: LEFTWARDS/RIGHTWARDS/UP/DOWN/
# LEFT RIGHT ARROW are all category Sm).

# Arrow blocks: not everything in these ranges is category Sm (a few dingbat-
# style arrows in the "Miscellaneous Symbols and Arrows" block are category
# So and mixed with non-mathematical dingbats), so only the three arrow-
# specific blocks are swept wholesale; the general dingbat block is not.
ARROW_BLOCKS = [
    (0x2190, 0x21FF),  # Arrows
    (0x27F0, 0x27FF),  # Supplemental Arrows-A
    (0x2900, 0x297F),  # Supplemental Arrows-B
]

MATH_OPERATOR_BLOCKS = [
    (0x2200, 0x22FF),  # Mathematical Operators
    (0x2A00, 0x2AFF),  # Supplemental Mathematical Operators
    (0x27C0, 0x27EF),  # Miscellaneous Mathematical Symbols-A
    (0x2980, 0x29FF),  # Miscellaneous Mathematical Symbols-B
]

GREEK_BLOCK = (0x0370, 0x03FF)
GREEK_NON_LETTER_PUNCTUATION_NAMES = {
    "GREEK QUESTION MARK",
    "GREEK ANO TELEIA",
    "GREEK TONOS",
    "GREEK NUMERAL SIGN",
    "GREEK LOWER NUMERAL SIGN",
}

LETTERLIKE_SYMBOLS_BLOCK = (0x2100, 0x214F)
# Block-membership corrections: these five codepoints sit in the Letterlike
# Symbols block for historical Unicode-layout reasons but are not maths
# notation (telephone/service-mark/trademark/numero/facsimile signs).
LETTERLIKE_SYMBOLS_NON_MATH_EXCEPTIONS = {
    "TELEPHONE SIGN",
    "SERVICE MARK",
    "TRADE MARK SIGN",
    "NUMERO SIGN",
    "FACSIMILE SIGN",
}

SUPERSCRIPT_SUBSCRIPT_BLOCK = (0x2070, 0x209F)  # sub/superscript digits+letters
MATH_ALPHANUMERIC_BLOCK = (0x1D400, 0x1D7FF)  # bold/italic/script/blackboard math letters/digits
PRIME_CODEPOINTS = set(range(0x2032, 0x2038)) | {0x2057}  # ' '' ''' '''' + quadruple prime
COMBINING_DIACRITIC_BLOCK = (0x0300, 0x036F)  # math accents on variables: dot/hat/bar/vec above/below
LATIN1_MATH_SYMBOLS = {0x00D7, 0x00F7, 0x00B1, 0x00B0}  # x / +/- / degree (category So, not Sm)


def _in_any(cp, blocks):
    return any(lo <= cp <= hi for lo, hi in blocks)


def is_maths_or_greek_char(ch):
    """True if `ch` is a maths-notation or maths-alphabet character (Greek
    letters used as maths variables, operators, relations, arrows,
    integrals, set symbols, primes, sub/superscript digits, math-alphanumeric
    letters, or a combining diacritic used as a maths accent). False for
    Greek-language accented letters, Greek punctuation, CJK/other-script
    punctuation, and plain ASCII (ASCII maths operators are handled by the
    byte-alphabet/<=3-char ASCII layers, not here)."""
    cp = ord(ch)
    if cp < 128:
        return False
    cat = ud.category(ch)
    if cat == "Sm":
        return True
    if _in_any(cp, ARROW_BLOCKS) or _in_any(cp, MATH_OPERATOR_BLOCKS):
        return True
    lo, hi = GREEK_BLOCK
    if lo <= cp <= hi:
        name = ud.name(ch, "")
        if name in GREEK_NON_LETTER_PUNCTUATION_NAMES:
            return False
        if "WITH TONOS" in name or "WITH DIALYTIKA" in name:
            return False  # modern Greek-language orthography, not maths
        return cat in ("Ll", "Lu")
    lo, hi = LETTERLIKE_SYMBOLS_BLOCK
    if lo <= cp <= hi:
        return ud.name(ch, "") not in LETTERLIKE_SYMBOLS_NON_MATH_EXCEPTIONS
    lo, hi = SUPERSCRIPT_SUBSCRIPT_BLOCK
    if lo <= cp <= hi:
        return True
    lo, hi = MATH_ALPHANUMERIC_BLOCK
    if lo <= cp <= hi:
        return True
    if cp in PRIME_CODEPOINTS:
        return True
    lo, hi = COMBINING_DIACRITIC_BLOCK
    if lo <= cp <= hi:
        return True
    if cp in LATIN1_MATH_SYMBOLS:
        return True
    return False


def is_maths_or_greek_text(text):
    """A token's decoded text qualifies for the maths/Greek symbol layer if,
    stripped of ASCII whitespace (the GPT-2 leading-space marker decodes to
    a literal space), it is non-empty and every character is either ASCII
    whitespace or a maths/Greek character. This intentionally excludes any
    token that mixes a maths symbol with ASCII letters/digits (those are
    Layer 3 candidates, ranked by merge rank like any other token) and any
    token that mixes maths symbols with non-maths non-ASCII text (e.g. a
    CJK character next to an arrow) -- pure maths-symbol tokens only."""
    stripped = text.strip(" \t")
    if not stripped:
        return False
    return all(is_maths_or_greek_char(ch) for ch in stripped)


def is_ascii_letter(ch):
    o = ord(ch)
    return (0x41 <= o <= 0x5A) or (0x61 <= o <= 0x7A)


def has_ascii_letter(text):
    return any(is_ascii_letter(ch) for ch in text)


def is_ascii_text(text):
    return all(ord(ch) < 128 for ch in text)


def is_short_ascii_token_text(text):
    """Layer 1's '<=3 characters' ASCII-token rule. Length is measured on
    the token's *content*, stripped of leading/trailing whitespace: the
    GPT-2 leading-space marker (a literal space once decoded) is a
    word-boundary marker, not content, so " the" (4 decoded characters) and
    "the" (3) are the same 3-character word for this rule -- otherwise every
    space-prefixed token would need one fewer content character to qualify
    than its non-prefixed twin, which is not what "ASCII tokens of <=3
    characters" means. Whole-whitespace tokens (e.g. a run of indent spaces)
    strip to empty and are excluded here; they are ordinary Layer 3
    candidates (or dropped) like any other punctuation-only token, since
    the byte-alphabet layer already guarantees a bare space is always kept."""
    if not is_ascii_text(text):
        return False
    content = text.strip(" \t\r\n")
    return 0 < len(content) <= 3
