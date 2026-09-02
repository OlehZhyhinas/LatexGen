#!/usr/bin/env python3
"""Render the image-OCR benchmark set: LaTeX references -> PNGs via KaTeX in
headless Chrome. Writes public/bench-images/<id>.png and
public/bench-images.json (id, tier, reference, kind)."""
import json, os, subprocess, sys, tempfile, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "public" / "bench-images"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# kind: "eq" = single display equation; "text" = prose with inline math
ITEMS = [
    # easy
    ("pythagorean", "easy", "eq", r"x^2 + y^2 = r^2"),
    ("quadratic", "easy", "eq", r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}"),
    ("euler", "easy", "eq", r"e^{i\pi} + 1 = 0"),
    ("circle-area", "easy", "eq", r"A = \pi r^2"),
    # medium
    ("gaussian", "medium", "eq", r"\int_{-\infty}^{\infty} e^{-x^2}\,dx = \sqrt{\pi}"),
    ("basel", "medium", "eq", r"\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}"),
    ("binomial", "medium", "eq", r"(x+y)^n = \sum_{k=0}^{n} \binom{n}{k} x^{k} y^{n-k}"),
    ("derivative", "medium", "eq", r"f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}"),
    # hard
    ("det2x2", "hard", "eq", r"\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc"),
    ("schrodinger", "hard", "eq",
     r"i\hbar \frac{\partial}{\partial t}\Psi(\mathbf{r},t) = \left[-\frac{\hbar^2}{2m}\nabla^2 + V(\mathbf{r},t)\right]\Psi(\mathbf{r},t)"),
    ("cauchy", "hard", "eq", r"f(a) = \frac{1}{2\pi i} \oint_\gamma \frac{f(z)}{z-a}\,dz"),
    ("zeta", "hard", "eq", r"\zeta(s) = 2^s \pi^{s-1} \sin\left(\frac{\pi s}{2}\right) \Gamma(1-s)\, \zeta(1-s)"),
    ("maxwell", "hard", "eq",
     r"\begin{aligned} \nabla \cdot \mathbf{E} &= \frac{\rho}{\varepsilon_0} \\ \nabla \times \mathbf{B} &= \mu_0 \mathbf{J} + \mu_0 \varepsilon_0 \frac{\partial \mathbf{E}}{\partial t} \end{aligned}"),
    # mixed prose + inline math (Texify's home turf; single-equation OCRs should struggle)
    ("newton-text", "mixed", "text",
     r"Newton's second law states that \(F = ma\). The kinetic energy of the body is \(K = \tfrac{1}{2} m v^2\)."),
    ("epsilon-delta-text", "mixed", "text",
     r"For all \(\varepsilon > 0\) there exists \(\delta > 0\) such that \(|x - a| < \delta\) implies \(|f(x) - L| < \varepsilon\)."),
    ("gaussian-text", "mixed", "text",
     r"The Gaussian integral \(\int_{-\infty}^{\infty} e^{-x^2}\,dx = \sqrt{\pi}\) follows from polar coordinates, where \(dA = r\,dr\,d\theta\)."),
    # degraded: same math, hostile rendering (tiny font; dark mode)
    ("quadratic-tiny", "degraded", "eq-tiny", r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}"),
    ("basel-dark", "degraded", "eq-dark", r"\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}"),
]

TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<style>
 body {{ margin:0; background:{bg}; color:{fg}; font-family: Georgia, 'Times New Roman', serif; }}
 #box {{ display:inline-block; padding:24px 32px; font-size:{fs}px; max-width:{maxw}px; line-height:1.5; }}
</style></head><body><div id="box">{body}</div>
<script>renderMathInElement(document.getElementById('box'), {{delimiters:[
  {{left:'\\\\[',right:'\\\\]',display:true}},{{left:'\\\\(',right:'\\\\)',display:false}}], throwOnError:false}});</script>
</body></html>"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = []
    with tempfile.TemporaryDirectory() as td:
        for id_, tier, kind, ref in ITEMS:
            if kind == "text":
                body, fs, maxw, bg, fg = ref, 20, 720, "white", "black"
            else:
                body = r"\[" + ref + r"\]"
                fs = 12 if kind == "eq-tiny" else 30
                maxw = 900
                bg, fg = ("#111", "#eee") if kind == "eq-dark" else ("white", "black")
            html = TEMPLATE.format(body=body, fs=fs, maxw=maxw, bg=bg, fg=fg)
            src = pathlib.Path(td) / f"{id_}.html"
            src.write_text(html)
            png = OUT / f"{id_}.png"
            subprocess.run([
                CHROME, "--headless=new", "--hide-scrollbars", "--disable-gpu",
                f"--window-size={maxw + 80},{'260' if kind != 'text' else '220'}",
                "--virtual-time-budget=4000",
                f"--screenshot={png}", f"file://{src}",
            ], check=True, capture_output=True)
            meta.append({"id": id_, "tier": tier, "kind": kind, "reference": ref, "image": f"/bench-images/{id_}.png"})
            print(f"rendered {id_:22s} {png.stat().st_size // 1024:4d} KB", file=sys.stderr)
    (ROOT / "public" / "bench-images.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(meta)} items -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
