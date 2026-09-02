#!/usr/bin/env python3
"""Generate public/benchmarks.html from the judged benchmark results in bench/.
Static HTML so the page is crawlable and needs no JS."""
import json, pathlib, statistics, collections, html

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEXT = json.load(open(ROOT / "bench" / "results-2026-09-01.json"))
IMG = json.load(open(ROOT / "bench" / "results-images-2026-09.json"))

def matrix(rows, tiers, key="approach"):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[r[key]][r["tier"]].append(r)
    out = []
    for ap in sorted(by, key=lambda a: -sum(r["correct"] for t in by[a] for r in by[a][t])):
        cells = []
        for t in tiers:
            rs = by[ap][t]
            if not rs:
                cells.append(("—", None)); continue
            acc = 100 * sum(r["correct"] for r in rs) / len(rs)
            med = statistics.median(r["ms"] for r in rs) / 1000
            cells.append((f"{acc:.0f}% <span class=lat>@ {med:.1f}s</span>", acc))
        out.append((ap, cells))
    return out

NAMES = {
    "specialist": "IntelliTeX specialist (220M, in-browser CPU)",
    "direct:Qwen3-1.7B": "Qwen3 1.7B (WebGPU)", "direct:Qwen3-4B": "Qwen3 4B (WebGPU)",
    "direct:Qwen3-0.6B": "Qwen3 0.6B (WebGPU)", "direct:Llama-3.2-3B": "Llama 3.2 3B (WebGPU)",
    "direct:SmolLM2-360M": "SmolLM2 360M (WebGPU)",
    "pipeline:Qwen3-0.6B+spec": "Qwen3 0.6B segmenter → specialist", "pipeline:SmolLM2-360M+spec": "SmolLM2 segmenter → specialist",
    "texo": "Texo (77 MB)", "texify": "Texify (305 MB)",
}

def table(rows, tiers, caption):
    h = [f"<table><caption>{caption}</caption><thead><tr><th>approach</th>" + "".join(f"<th>{t}</th>" for t in tiers) + "</tr></thead><tbody>"]
    for ap, cells in matrix(rows, tiers):
        h.append(f"<tr><th scope=row>{html.escape(NAMES.get(ap, ap))}</th>")
        for txt, acc in cells:
            cls = "" if acc is None else (" class=good" if acc >= 75 else " class=bad" if acc < 50 else "")
            h.append(f"<td{cls}>{txt}</td>")
        h.append("</tr>")
    h.append("</tbody></table>")
    return "\n".join(h)

text_rows = [r for r in TEXT if "correct" in r]
img_rows = IMG["rows"]
n_text_items = len({r["item"] for r in text_rows})
n_img_items = len({r["item"] for r in img_rows})

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>LatexGen benchmarks — which on-device models actually convert math to LaTeX</title>
<meta name="description" content="Accuracy and latency of in-browser models on text-to-LaTeX and image-to-LaTeX, judged against Wikipedia-referenced equations. The data behind LatexGen's routing." />
<link rel="stylesheet" href="/vendor/fonts/geist/geist.css" />
<link rel="stylesheet" href="/style.css" />
<style>
  .prose {{ max-width: 860px; margin: 0 auto; padding: 32px 16px 64px; }}
  .prose h1 {{ font-size: 28px; margin: 0 0 8px; }}
  .prose h2 {{ font-size: 20px; margin: 40px 0 8px; }}
  .prose p, .prose li {{ font-size: 15px; color: var(--text); }}
  .prose .lead {{ font-size: 17px; color: var(--muted); }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 14px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-md); overflow: hidden; }}
  caption {{ text-align: left; font-weight: 600; padding: 12px 14px; background: var(--surface-2); }}
  th, td {{ padding: 10px 14px; text-align: left; border-top: 1px solid var(--border); white-space: nowrap; }}
  th[scope=row] {{ font-weight: 500; white-space: normal; }}
  td.good {{ color: var(--success); font-weight: 600; }}
  td.bad {{ color: var(--muted); }}
  .lat {{ color: var(--muted); font-weight: 400; font-size: 12px; }}
  .wrap-x {{ overflow-x: auto; }}
</style>
</head>
<body>
<header class="topbar">
  <a class="brand" href="/"><span class="logo-mark" aria-hidden="true">Σ</span><span class="wordmark">LatexGen</span></a>
  <nav class="nav"><a href="/#how">How it works</a><a href="/benchmarks.html">Benchmarks</a><a href="https://github.com/OlehZhyhinas/LatexGen" target="_blank" rel="noopener">GitHub</a></nav>
</header>
<main class="prose">
  <h1>Benchmarks</h1>
  <p class="lead">Every routing decision in LatexGen comes from these numbers, not from model marketing. Each cell is accuracy on the tier (median latency), judged by a local 27B model against reference LaTeX taken from Wikipedia.</p>

  <h2>Text → LaTeX ({n_text_items} items)</h2>
  <p>Inputs are plain-English math as a student would type it. Tiers: <b>easy</b> (quadratic formula, Euler's identity…), <b>medium</b> (Gaussian integral, binomial theorem…), <b>hard</b> (Schrödinger, Cauchy integral formula, zeta functional equation…), <b>multiline</b> (prose passages with several embedded equations).</p>
  <div class="wrap-x">{table(text_rows, ["easy", "medium", "hard", "multiline"], "Accuracy @ median latency, in-browser models on an Apple M-series laptop")}</div>
  <p>What it decided: the 220M specialist goes first for single equations (it is 100% on easy and medium at a fraction of the cost); prose passages skip browser models below 4B-class; the “tiny model segments, specialist converts” idea was dropped, because small models fail at instruction-following long before they fail at math.</p>

  <h2>Image → LaTeX ({n_img_items} rendered images)</h2>
  <p>Equations rendered with KaTeX in headless Chrome, plus prose paragraphs with inline math and two hostile renders (12px font, dark mode). Both models run entirely in the browser.</p>
  <div class="wrap-x">{table(img_rows, ["easy", "medium", "hard", "mixed", "degraded"], "Texo vs Texify — accuracy @ median latency")}</div>
  <p>Texo's two misses on medium/degraded were the KaTeX-only alias <code>\\infin</code>, which the app now canonicalizes to <code>\\infty</code>; Texify's misses on easy images were a degenerate repeat-until-token-cap loop, which the app now collapses. Result: Texo reads first (fast, tiny, robust to small fonts and dark backgrounds); Texify takes over for prose and matrices.</p>

  <h2>Reproduce it</h2>
  <p>Everything is in the repository: the eval sets, the in-browser harnesses (<code>public/bench.html</code>, <code>public/bench-images.html</code>), the renderer for the image set, and the judge scripts. Run them against a new model and send the numbers.</p>
</main>
<footer class="foot"><span>Free, unlimited, and private: your text and images are processed on this device. Nothing is uploaded.</span></footer>
</body>
</html>
"""
(ROOT / "public" / "benchmarks.html").write_text(page)
print("wrote public/benchmarks.html", len(page), "bytes")
