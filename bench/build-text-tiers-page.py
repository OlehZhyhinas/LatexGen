#!/usr/bin/env python3
"""Render the tier-matched text benchmark (results + judge verdicts) as one HTML page.

  python3 bench/build-text-tiers-page.py bench/results-text-tiers-2026-09-07.json bench/judged-text-tiers-2026-09-07.json bench/bench-data-extended.json out.html
"""
import json, sys, html, collections, statistics, datetime, os

rows_path, judged_path, data_path, out_path = sys.argv[1:5]
rows = json.load(open(rows_path))
judged = json.load(open(judged_path)) if os.path.exists(judged_path) else []
items = json.load(open(data_path))
verdict = {(r["approach"], r["item"]): r for r in judged if "correct" in r}
n_items = len(items)

CLASSES = [
    ("Around 1B", "Shipped: Qwen3 0.6B", [("qwen3:0.6b", "Qwen3 0.6B", True), ("qwen3.5:0.8b", "Qwen3.5 0.8B", False), ("gemma3:1b", "Gemma 3 1B", False), ("openbmb/minicpm5:q4_K_M", "MiniCPM5 1B", False)]),
    ("Around 2B", "Shipped: Qwen3 1.7B", [("qwen3:1.7b", "Qwen3 1.7B", True), ("qwen3.5:2b-q4_K_M", "Qwen3.5 2B", False), ("hf.co/openbmb/MiniCPM5-2B-GGUF:q4_K_M", "MiniCPM5 2B", False), ("gemma4:e2b", "Gemma 4 E2B", False)]),
    ("Around 4B", "Shipped: Qwen3 4B", [("qwen3:4b", "Qwen3 4B", True), ("qwen3.5:4b", "Qwen3.5 4B", False), ("gemma3:4b", "Gemma 3 4B", False), ("gemma4:e4b", "Gemma 4 E4B", False)]),
]
TIERS = [("easy", "easy", 4), ("medium", "medium", 4), ("hard", "hard", 4), ("multiline", "prose", 3),
         ("pdf-paste", "PDF paste", 30), ("synth-pdf", "synth PDF", 24), ("synth-unicode", "synth unicode", 18), ("synth-latex", "synth LaTeX", 18)]
tier_n = collections.Counter(i["tier"] for i in items)

by = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    if r.get("tier") in tier_n and r.get("ms", -1) >= 0:
        by[r["model"]][r["tier"]].append(r)

def stats(model, tier):
    rs = by[model].get(tier, [])
    jd = [verdict.get((r["approach"], r["item"])) for r in rs]
    jd = [j for j in jd if j]
    return dict(gen=len(rs), judged=len(jd), correct=sum(bool(j["correct"]) for j in jd),
                valid=sum(bool(r.get("valid")) for r in rs), ms=[r["ms"] for r in rs])

def cell(st, total, ship_st=None, is_ship=False):
    if st["gen"] == 0:
        return '<td class="cell pending"><span class="muted">queued</span></td>'
    if st["judged"] == 0:
        return f'<td class="cell pending"><span class="muted">{st["gen"]}/{total} generated</span><span class="sub">judging…</span></td>'
    acc = st["correct"] / st["judged"]
    delta = ""
    if not is_ship and ship_st and ship_st["judged"]:
        d = st["correct"] / st["judged"] - ship_st["correct"] / ship_st["judged"]
        if abs(d) >= 0.005:
            cls = "up" if d > 0 else "down"
            delta = f'<span class="delta {cls}">{"+" if d > 0 else "−"}{abs(d) * 100:.0f} pt</span>'
    partial = "" if st["judged"] == total else f'<span class="sub">{st["judged"]} of {total} judged</span>'
    return (f'<td class="cell"><div class="frac"><span class="num">{st["correct"]}/{st["judged"]}</span>{delta}</div>'
            f'<div class="bar"><i style="width:{acc * 100:.0f}%"></i></div>{partial}</td>')

def section(title, sub, models):
    ship_id = next(m for m, _, s in models if s)
    head = "".join(f'<th><span>{lbl}</span><span class="n">n={tier_n[t]}</span></th>' for t, lbl, _ in TIERS)
    out = [f'<section class="tier"><div class="tier-head"><h2>{title}</h2><p>{sub}</p></div>',
           f'<div class="scroll"><table><thead><tr><th class="model">model</th>{head}<th>all judged</th><th>validator</th><th>median</th></tr></thead><tbody>']
    for mid, name, is_ship in models:
        allst = {"gen": 0, "judged": 0, "correct": 0, "valid": 0, "ms": []}
        cells = []
        for t, _, _ in TIERS:
            st = stats(mid, t); sh = stats(ship_id, t)
            for k in ("gen", "judged", "correct", "valid"): allst[k] += st[k]
            allst["ms"] += st["ms"]
            cells.append(cell(st, tier_n[t], sh, is_ship))
        tot_acc = f'{allst["correct"] / allst["judged"]:.0%}' if allst["judged"] else "—"
        if allst["judged"] and not is_ship:
            sh_all = {"j": 0, "c": 0}
            for t, _, _ in TIERS:
                s = stats(ship_id, t); sh_all["j"] += s["judged"]; sh_all["c"] += s["correct"]
            # compare on the same judged set size only when both complete
            if sh_all["j"] == allst["judged"] == n_items:
                d = (allst["correct"] - sh_all["c"]) / n_items * 100
                tot_acc += f' <span class="delta {"up" if d > 0 else "down" if d < 0 else ""}">{"+" if d > 0 else "−" if d < 0 else "±"}{abs(d):.0f} pt</span>'
        valid = f'{allst["valid"] / allst["gen"]:.0%}' if allst["gen"] else "—"
        med = f'{statistics.median(allst["ms"]) / 1000:.1f} s' if allst["ms"] else "—"
        prog = f'<span class="prog">{allst["gen"]}/{n_items} generated · {allst["judged"]} judged</span>'
        out.append(f'<tr class="{"ship" if is_ship else ""}"><td class="model"><span class="name">{name}</span>{"<span class=tag>shipped</span>" if is_ship else ""}{prog}</td>'
                   + "".join(cells) + f'<td class="cell total">{tot_acc}</td><td class="cell">{valid}</td><td class="cell">{med}</td></tr>')
    out.append("</tbody></table></div>")
    # sample outputs: one prose item and one dirty pdf paste
    samples = ["mechanics-homework-passage", "euler-formula-passage-pdf-dirty", "riemann-zeta-functional-equation"]
    out.append('<details class="samples"><summary>Sample outputs for this size class</summary>')
    for sid in samples:
        it = next((i for i in items if i["id"] == sid), None)
        if not it: continue
        out.append(f'<div class="sample"><h3>{html.escape(sid)} <span class="muted">({it["tier"]})</span></h3><pre class="in">{html.escape(it["input"][:600])}</pre>')
        for mid, name, is_ship in models:
            r = next((x for x in rows if x["model"] == mid and x["item"] == sid), None)
            if not r: continue
            v = verdict.get((r["approach"], sid))
            mark = "" if not v else (f'<b class="ok">judge: correct</b>' if v["correct"] else f'<b class="bad">judge: wrong</b> <i>{html.escape(v.get("why", ""))}</i>')
            val = 'validator: pass' if r.get("valid") else 'validator: ' + html.escape("; ".join(r.get("issues", []))[:160])
            out.append(f'<div class="out"><div class="who">{name}{" · shipped" if is_ship else ""} <span class="muted">{val}</span> {mark}</div><pre>{html.escape((r.get("output") or "(empty)")[:700])}</pre></div>')
        out.append("</div>")
    out.append("</details></section>")
    return "\n".join(out)

gen_total = sum(1 for r in rows if r.get("ms", -1) >= 0 and r.get("tier") in tier_n)
judged_total = len(verdict)
n_models = sum(len(m) for _, _, m in CLASSES)
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
progress_pct = gen_total / (n_items * n_models) * 100
judge_pct = judged_total / (n_items * n_models) * 100

page = f'''<title>Small-Model Tier Bench</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#F2F5F3;--panel:#FBFCFB;--ink:#17201C;--ink2:#4B5A53;--line:#D7DED9;--accent:#0B6E60;--accent-ink:#0B6E60;--ship:#5C6B76;--good:#1F8A5B;--bad:#B24630;--bar:#0B6E60;--bar-track:#E1E8E4;--chip:#E6EEEA}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#121816;--panel:#1A221F;--ink:#E4EBE7;--ink2:#9AAAA2;--line:#2B3733;--accent:#4FC2AE;--accent-ink:#7ED7C7;--ship:#93A3AC;--good:#5AC98F;--bad:#E27B63;--bar:#4FC2AE;--bar-track:#26312D;--chip:#243029}}}}
:root[data-theme="dark"]{{--bg:#121816;--panel:#1A221F;--ink:#E4EBE7;--ink2:#9AAAA2;--line:#2B3733;--accent:#4FC2AE;--accent-ink:#7ED7C7;--ship:#93A3AC;--good:#5AC98F;--bad:#E27B63;--bar:#4FC2AE;--bar-track:#26312D;--chip:#243029}}
body{{background:var(--bg);color:var(--ink);font:15px/1.5 "IBM Plex Sans",system-ui,sans-serif;margin:0}}
main{{max-width:1180px;margin:0 auto;padding:40px 28px 72px}}
h1,h2,h3{{font-family:"Source Serif 4",Georgia,serif;font-weight:600;text-wrap:balance;margin:0}}
h1{{font-size:34px;line-height:1.15}} h2{{font-size:22px}} h3{{font-size:15px;font-family:"IBM Plex Mono",ui-monospace,monospace;font-weight:500}}
header p{{max-width:64ch;color:var(--ink2);margin:10px 0 0}}
.status{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:26px 0 8px;padding:18px 20px;background:var(--panel);border:1px solid var(--line);border-radius:6px}}
.status .k{{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink2)}}
.status .v{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:20px;font-variant-numeric:tabular-nums;margin-top:2px}}
.status .bar{{margin-top:8px}}
.bar{{height:5px;background:var(--bar-track);border-radius:3px;overflow:hidden}} .bar i{{display:block;height:100%;background:var(--bar)}}
.tier{{margin-top:44px}}
.tier-head{{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap;margin-bottom:12px}} .tier-head p{{margin:0;color:var(--ink2)}}
.scroll{{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--panel)}}
table{{border-collapse:collapse;width:100%;min-width:1080px;font-variant-numeric:tabular-nums}}
th{{text-align:left;font-weight:500;font-size:12px;color:var(--ink2);padding:12px 12px 8px;border-bottom:1px solid var(--line);vertical-align:bottom}}
th span{{display:block}} th .n{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;opacity:.75}}
td{{padding:12px;border-bottom:1px solid var(--line);vertical-align:top}} tr:last-child td{{border-bottom:0}}
td.model{{width:170px}} .name{{font-weight:600;display:block}} .tag{{display:inline-block;font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--ship);border:1px solid var(--ship);border-radius:3px;padding:0 6px;margin-top:4px}}
.prog{{display:block;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;color:var(--ink2);margin-top:6px}}
tr.ship td{{background:color-mix(in srgb,var(--ship) 7%,transparent)}}
.cell{{min-width:104px}} .cell .bar{{margin-top:6px}} .frac{{display:flex;justify-content:space-between;gap:8px;align-items:baseline}}
.num{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:15px;font-weight:500}}
.delta{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;padding:0 5px;border-radius:3px;background:var(--chip)}} .delta.up{{color:var(--good)}} .delta.down{{color:var(--bad)}}
.sub,.muted{{color:var(--ink2);font-size:12px}} .sub{{display:block;margin-top:4px}}
.total{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-weight:500;font-size:16px}}
details.samples{{margin-top:12px}} summary{{cursor:pointer;color:var(--accent-ink);font-weight:500}}
.sample{{margin-top:18px;padding:14px 16px;background:var(--panel);border:1px solid var(--line);border-radius:6px}}
pre{{font:12.5px/1.45 "IBM Plex Mono",ui-monospace,monospace;white-space:pre-wrap;word-break:break-word;margin:6px 0 0;padding:10px 12px;background:var(--bg);border-radius:4px;border:1px solid var(--line)}}
pre.in{{border-left:3px solid var(--accent)}}
.out{{margin-top:12px}} .who{{font-size:13px;font-weight:500}} .ok{{color:var(--good)}} .bad{{color:var(--bad)}} .who i{{color:var(--ink2);font-weight:400}}
.method{{margin-top:48px;max-width:70ch;color:var(--ink2);font-size:14px}} .method h2{{color:var(--ink);font-size:18px;margin-bottom:8px}} .method li{{margin:4px 0}}
</style>
<main>
<header>
<h1>Small-Model Tier Bench</h1>
<p>Each shipped Qwen3 size against the candidates of the same size, on the pipeline's exact prompts with greedy decoding and thinking off. Cells are judge-correct over judged items; the delta is against the shipped model in that row's size class. Rows fill in as the run proceeds. Last built {now}.</p>
</header>
<div class="status">
<div><div class="k">Generation</div><div class="v">{gen_total} / {n_items * n_models}</div><div class="bar"><i style="width:{progress_pct:.1f}%"></i></div></div>
<div><div class="k">Judged by Qwen3.8 27B</div><div class="v">{judged_total} / {n_items * n_models}</div><div class="bar"><i style="width:{judge_pct:.1f}%"></i></div></div>
</div>
{"".join(section(*c) for c in CLASSES)}
<section class="method">
<h2>How this was measured</h2>
<ul>
<li>Corpus: {n_items} items. The 15 original Wikipedia-referenced items (easy, medium, hard, prose), 30 PDF pastes of the reference passages through headless Chrome and pdftotext (15 clean, 15 with two-column, footnote, ligature and running-head garbling), and 60 synthetic prose-with-math passages sampled from the text-gates set (PDF-style, Unicode-style and LaTeX-style math, ten of them prose only where the right answer is to change nothing).</li>
<li>Runtime: Ollama on the M5 Pro, all models at 4-bit (q4_K_M; Qwen3.5 0.8B at q8_0 since no 4-bit tag exists), temperature 0, seed 0, 4K context, thinking disabled. Max tokens follow the pipeline's own formula. Prompts and post-processing are read out of public/pipeline.js at run time.</li>
<li>Validator: the same KaTeX syntax, fidelity and not-answered checks the app uses to decide escalation.</li>
<li>Judge: Qwen3.8 27B at q8_0 with the repo's judge prompt, mathematical equivalence to the reference; for prose items the candidate must keep the prose and convert all the math.</li>
<li>Medians are Ollama wall-clock per item and only comparable within this page, not to the browser numbers.</li>
</ul>
</section>
</main>'''
open(out_path, "w").write(page)
print(f"{gen_total} generated, {judged_total} judged -> {out_path}")
