#!/usr/bin/env python3
import pdf_pipeline


def _one(clean, needle):
    i = clean.find(needle)
    assert i >= 0, (needle, clean)
    return (i, i + len(needle))


def _spans_text(text, spans):
    return [text[s:e] for s, e in spans]


def test_normalize():
    clean, omap = pdf_pipeline.normalize("ab\u200bcd")
    assert clean == "abcd"
    assert len(clean) == len(omap)

    clean, _ = pdf_pipeline.normalize("deriv-\native")
    assert clean == "derivative"

    clean, _ = pdf_pipeline.normalize("first line\nsecond line")
    assert clean == "first line second line"

    clean, _ = pdf_pipeline.normalize("p1 line\n\np2 line")
    assert clean == "p1 line\n\np2 line"

    clean, _ = pdf_pipeline.normalize("Euler.[12] Next.[3, 7]")
    assert clean == "Euler.[12] Next.[3, 7]"

    clean, _ = pdf_pipeline.normalize("table¹ text")
    assert clean == "table¹ text"

    clean, _ = pdf_pipeline.normalize("[0, 1]V and [−1, 1]V")
    assert clean == "[0, 1]V and [−1, 1]V"

    clean, _ = pdf_pipeline.normalize("Amir et al. [1] proved")
    assert clean == "Amir et al. [1] proved"

    clean, _ = pdf_pipeline.normalize("Refs. [5, 6, 20];")
    assert clean == "Refs. [5, 6, 20];"

    clean, _ = pdf_pipeline.normalize("new [17–19].")
    assert clean == "new [17–19]."

    clean, _ = pdf_pipeline.normalize("Body line\n\n1. Footnote note: extra")
    assert clean == "Body line"

    clean, _ = pdf_pipeline.normalize("Body line\n\n¹ Footnote note: extra")
    assert clean == "Body line"

    clean, _ = pdf_pipeline.normalize("Body\n\n0\n\nTail")
    assert clean == "Body\n\nTail"

    raw = "Proceedings Note - Sample Author\nText one\nProceedings Note - Sample Author\nText two"
    clean, _ = pdf_pipeline.normalize(raw)
    assert "Proceedings Note - Sample Author" not in clean
    assert "Text one" in clean and "Text two" in clean

    raw = (
        "left one        right one\n"
        "left two        right two\n"
        "left three        right three\n"
    )
    clean, _ = pdf_pipeline.normalize(raw)
    assert "left one left two left three" in clean
    assert "right one right two right three" in clean

    raw = "v = ds\n, the\ndt"
    clean, _ = pdf_pipeline.normalize(raw)
    assert clean == "v = ds dt , the"

    clean, _ = pdf_pipeline.normalize("a   b\t\tc")
    assert clean == "a b c"

    raw = "1\nTo compare two classifiers, the notes write the logistic link σ(x) = 1+e −x ."
    clean, _ = pdf_pipeline.normalize(raw)
    assert clean.startswith("1\n\nTo compare two classifiers")


def test_segment_positive():
    text = "Euler wrote eix = cos x + i sin x in class."
    spans = pdf_pipeline.segment(text)
    assert _spans_text(text, spans) == ["eix = cos x + i sin x"]

    text = "Use x2 + y2 = z2 and q1/q2 in the ratio."
    spans = pdf_pipeline.segment(text)
    assert _spans_text(text, spans) == ["x2 + y2 = z2 and q1/q2"]

    text = "Given dy/dx = dy/du * du/dx, continue."
    spans = pdf_pipeline.segment(text)
    assert _spans_text(text, spans) == ["dy/dx = dy/du * du/dx"]

    text = "The line is\nx + y = z\nfor all."
    spans = pdf_pipeline.segment(text)
    assert any("x + y = z" in piece for piece in _spans_text(text, spans))

    text = "A bound uses |a + b| <= |a| + |b| in proofs."
    spans = pdf_pipeline.segment(text)
    assert _spans_text(text, spans) == ["|a + b| <= |a| + |b|"]

    text = "For sigma, use σ(x) = 1/(1+e^-x)."
    spans = pdf_pipeline.segment(text)
    assert _spans_text(text, spans) == ["σ(x) = 1/(1+e^-x)"]

    text = "Euler wrote eix = cos x + i sin x for any real number."
    spans = pdf_pipeline.segment(text)
    assert _spans_text(text, spans) == ["eix = cos x + i sin x"]


def test_segment_orphan_and_display_examples():
    text = (
        "... First, ∞\n\n"
        "a ∑k=0 ark = 1−r . Second,\n\n"
        "1 e = lim (1 + ) n→∞ n\n\n"
        "when the compounding step is refined.\n\n"
        "n"
    )
    spans = pdf_pipeline.segment(text)
    assert _spans_text(text, spans) == [
        "∞\n\na ∑k=0 ark = 1−r",
        "1 e = lim (1 + ) n→∞ n",
    ]

    text = (
        "... the Basel problem: ∞\n\n"
        "1 π2 ∑ 2 = n 6 n=1\n\n"
        "Second, the Gaussian integral:\n\n"
        "∫\n\n"
        "∞\n\n"
        "e−x dx =\n\n"
        "π\n\n"
        "−∞\n\n"
        "Both answers ..."
    )
    spans = pdf_pipeline.segment(text)
    assert _spans_text(text, spans) == [
        "∞\n\n1 π2 ∑ 2 = n 6 n=1",
        "∫\n\n∞\n\ne−x dx =\n\nπ\n\n−∞",
    ]

    text = (
        "To compare two classifiers, the notes write the logistic link σ(x) = 1+e −x . "
        "An uncertainty 2 ∞ correction adds ∫−∞ e−x dx =\n\n"
        "π in the derivation."
    )
    spans = pdf_pipeline.segment(text)
    assert _spans_text(text, spans) == [
        "σ(x) = 1+e −x",
        "∫−∞ e−x dx =\n\nπ",
    ]

    text = (
        "... The determinant identity is written as\n\n"
        "det (\n\n"
        "a b ) = ad − bc c d\n\n"
        ". This section ..."
    )
    spans = pdf_pipeline.segment(text)
    assert _spans_text(text, spans) == ["det (\n\na b ) = ad − bc c d"]


def test_segment_negatives():
    assert pdf_pipeline.segment("The variable x was measured five times.") == []
    assert pdf_pipeline.segment("At time t the sample had cooled.") == []
    assert pdf_pipeline.segment("5 apples were left on the table.") == []
    assert pdf_pipeline.segment("In 1905 the result was announced.") == []
    assert pdf_pipeline.segment("The room number was 214 next door.") == []
    assert pdf_pipeline.segment("On 2026-03-18 we wrote notes.") == []
    assert pdf_pipeline.segment("The right-hand side is immediate.") == []
    assert pdf_pipeline.segment("The x-axis marks time and n-th term is prose.") == []
    assert pdf_pipeline.segment("A two-column layout appears in print.") == []


def test_clean_span_latex():
    got = pdf_pipeline.clean_span_latex("```latex\nLaTeX: \\boxed{\\( x + y \\\\ \\)}\n```")
    assert got == "x + y"

    got = pdf_pipeline.clean_span_latex("$$ \\begin{equation} z^2 \\\\ \\end{equation} $$")
    assert got == "z^2"


def test_splice_and_run():
    clean = "Force is F = ma in text."
    sp = [_one(clean, "F = ma")]
    out = pdf_pipeline.splice(clean, sp, ["\\mathbf{F}=m\\mathbf{a}"])
    assert out == "Force is \\( \\mathbf{F}=m\\mathbf{a} \\) in text."

    clean = "Before\nx + y = z\nAfter"
    sp = [_one(clean, "x + y = z")]
    out = pdf_pipeline.splice(clean, sp, ["x+y=z"])
    assert out == "Before\n\\[ x+y=z \\]\nAfter"

    clean = "The right-hand side is immediate."
    sp = [_one(clean, "right-hand")]
    out = pdf_pipeline.splice(clean, sp, ["right-hand"])
    assert out == "The right-hand side is immediate."

    def conv(parts):
        return [p.upper() for p in parts]

    rec = pdf_pipeline.run("Euler wrote eix = cos x + i sin x.", conv)
    assert rec["span_texts"] == ["eix = cos x + i sin x"]
    assert rec["latex"] == ["EIX = COS X + I SIN X"]
    assert "\\(" in rec["output"]

    rec = pdf_pipeline.run("Euler wrote eix = cos x + i sin x.", lambda parts: ["$$ $$"])
    assert rec["latex"] == ["eix = cos x + i sin x"]


if __name__ == "__main__":
    test_normalize()
    test_segment_positive()
    test_segment_orphan_and_display_examples()
    test_segment_negatives()
    test_splice_and_run()
    print("ok")
