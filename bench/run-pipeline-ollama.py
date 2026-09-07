#!/usr/bin/env python3
"""Run text-tier benchmarks through Ollama with direct/pipeline modes.

usage:
  .venv/bin/python bench/run-pipeline-ollama.py --models qwen3:0.6b --mode direct
"""
import argparse
import importlib.util
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
PIPELINE_JS = ROOT / "public" / "pipeline.js"
PDF_PASTES = BENCH / "pdf-pastes.json"
SYNTH_SPANS = BENCH / "synth-spans.jsonl"
BENCH_DATA = BENCH / "bench-data.json"
ITEMS_FILE = BENCH / "text-tiers-items.json"
PDF_PIPELINE_FILE = BENCH / "pdf_pipeline.py"

SPAN_SYSTEM = (
    "Convert this fragment of mathematics, copied from a PDF, to LaTeX. "
    "The fragment may have lost superscripts, fraction bars and spacing: `x2` means x^2, "
    "`12 mv 2` means \\frac{1}{2} m v^2, `eix` means e^{ix}, a number above a number is a "
    "fraction, `∑ n=1 ∞` are limits. Output only the LaTeX body, no delimiters, no words, "
    "no commentary. Never use \\boxed, \\text, or environments such as aligned or equation. "
    "Output one bare expression, exactly the mathematics in the fragment, nothing added and "
    "nothing dropped."
)
SPAN_SYSTEM_BATCH = (
    SPAN_SYSTEM
    + " Fragments are numbered. Answer with one line per fragment: `<n>. <latex>`."
)
SPAN_SYSTEM_CTX = (
    SPAN_SYSTEM
    + " You are given the sentence the fragment came from for context, then the fragment "
    "marked between « and ». Convert ONLY the marked fragment. Output the bare LaTeX "
    "of the fragment, nothing else."
)
SPAN_SYSTEM_CTX_BATCH = (
    SPAN_SYSTEM
    + " You are given a clean passage where each fragment is marked as «n: ...». Convert ONLY "
    "the marked fragments. Output one line per fragment: `<n>. <latex>`."
)
PIPELINE_MODES = {"pipeline", "pipeline-batch", "pipeline-ctx", "pipeline-ctx-batch"}
SENT_BOUNDARY_RE = re.compile(r"[.!?]")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma list")
    ap.add_argument(
        "--mode",
        required=True,
        choices=[
            "direct",
            "direct-norm",
            "pipeline",
            "pipeline-batch",
            "pipeline-ctx",
            "pipeline-ctx-batch",
        ],
    )
    ap.add_argument("--out", default=str(BENCH / "results-pipeline.json"))
    ap.add_argument("--items", default="", help="comma item ids")
    ap.add_argument("--stub-convert", action="store_true", help="pipeline modes only")
    return ap.parse_args()


def read_json(path):
    with open(path) as f:
        return json.load(f)


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def decode_js_template(text):
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        i += 1
        if i >= n:
            out.append("\\")
            break
        esc = text[i]
        if esc == "n":
            out.append("\n")
        elif esc == "r":
            out.append("\r")
        elif esc == "t":
            out.append("\t")
        elif esc in {"\\", "`", "$"}:
            out.append(esc)
        elif esc == "u" and i + 4 < n:
            code = text[i + 1 : i + 5]
            if all(c in "0123456789abcdefABCDEF" for c in code):
                out.append(chr(int(code, 16)))
                i += 4
            else:
                out.append("u")
        else:
            out.append(esc)
        i += 1
    return "".join(out)


def load_prompts():
    src = PIPELINE_JS.read_text()
    m_sys = re.search(r"const SYSTEM_PROMPT = `([\s\S]*?)`;", src)
    m_user = re.search(r"const CONVERT_USER = \(text\) => `([\s\S]*?)`;", src)
    if not m_sys or not m_user:
        raise RuntimeError("could not read SYSTEM_PROMPT/CONVERT_USER from public/pipeline.js")
    system_prompt = decode_js_template(m_sys.group(1))
    user_tmpl = decode_js_template(m_user.group(1))
    if "${text}" not in user_tmpl:
        raise RuntimeError("CONVERT_USER template missing ${text}")

    def convert_user(text):
        return user_tmpl.replace("${text}", text)

    return system_prompt, convert_user


def needs_no_think(model):
    m = model.strip().lower()
    return m.startswith("qwen3") and not m.startswith("qwen3.5")


def append_no_think_if_needed(model, user_text):
    if needs_no_think(model):
        return user_text + "\n/no_think"
    return user_text


def strip_think(text):
    total = 0
    for m in re.finditer(r"<think>[\s\S]*?</think>", text):
        total += len(m.group(0))
    clean = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    return clean, total


def strip_fences(text):
    m = re.match(r"^\s*```(?:latex|tex)?\s*\n([\s\S]*?)\n?```\s*$", text)
    if m:
        return m.group(1).strip()
    return text.strip()


def balanced_braces(text):
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
        i += 1
    return depth == 0


QUESTION_RE = re.compile(r"^\s*(what(?:'s| is| are)?|find|solve|evaluate|compute|calculate)\b", re.I)
EQUALS_RE = re.compile(r"=|\bequals?\b|\bis equal to\b", re.I)
ANSWERY_RE = re.compile(r"\b(the answer is|therefore|thus we|hence)\b", re.I)


def validate_output(inp, out):
    issues = []
    if not out.strip():
        issues.append("empty output")
        return {"ok": False, "issues": issues}
    if "```" in out:
        issues.append("contains fence")
    if not balanced_braces(out):
        issues.append("unbalanced braces")
    if out.count(r"\(") != out.count(r"\)"):
        issues.append("unbalanced \\( \\)")
    if out.count(r"\[") != out.count(r"\]"):
        issues.append("unbalanced \\[ \\]")
    t = inp.strip()
    if (QUESTION_RE.search(t) or t.endswith("?")) and not EQUALS_RE.search(t) and "=" in out:
        issues.append("looks like answered question")
    if ANSWERY_RE.search(out):
        issues.append("looks like answering/commentary")
    return {"ok": not issues, "issues": issues}


def ollama_chat(model, messages):
    ollama = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": {"temperature": 0, "num_predict": 512},
        "messages": messages,
    }
    req = urllib.request.Request(
        ollama + "/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            j = json.load(r)
    except urllib.error.HTTPError as e:
        payload = e.read().decode("utf-8", "ignore")
        raise RuntimeError(f"HTTP {e.code}: {payload[:200]}") from e
    except Exception as e:
        raise RuntimeError(str(e)) from e
    if j.get("error"):
        raise RuntimeError(j["error"])
    msg = (j.get("message") or {}).get("content", "")
    msg, thinking = strip_think(msg)
    msg = strip_fences(msg)
    return msg, {
        "thinking": thinking,
        "evalTokens": int(j.get("eval_count") or 0),
        "promptTokens": int(j.get("prompt_eval_count") or 0),
        "evalMs": round(float(j.get("eval_duration") or 0) / 1e6),
    }


class Stats:
    def __init__(self):
        self.requests = 0
        self.thinking = 0
        self.eval_tokens = 0
        self.prompt_tokens = 0
        self.eval_ms = 0

    def add(self, meta):
        self.requests += 1
        self.thinking += int(meta.get("thinking") or 0)
        self.eval_tokens += int(meta.get("evalTokens") or 0)
        self.prompt_tokens += int(meta.get("promptTokens") or 0)
        self.eval_ms += int(meta.get("evalMs") or 0)


def parse_numbered_lines(text, n):
    out = {}
    pat = re.compile(r"(?ms)^\s*(\d+)\s*[\).\:-]\s*(.*?)(?=^\s*\d+\s*[\).\:-]\s*|\Z)")
    for m in pat.finditer(text):
        idx = int(m.group(1))
        if 1 <= idx <= n:
            out[idx - 1] = m.group(2).strip()
    if len(out) == n:
        return [out[i] for i in range(n)]
    return out


def infer_span_texts(clean_text, spans):
    out = []
    for pair in spans:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        a, b = int(pair[0]), int(pair[1])
        out.append(clean_text[a:b])
    return out


def make_stub_pdf_pipeline():
    class Stub:
        @staticmethod
        def normalize(text):
            return text, list(range(len(text) + 1))

        @staticmethod
        def segment(clean_text):
            return []

        @staticmethod
        def splice(clean_text, spans, latex_list):
            out = []
            cursor = 0
            for (a, b), repl in zip(spans, latex_list):
                a, b = int(a), int(b)
                out.append(clean_text[cursor:a])
                out.append(repl)
                cursor = b
            out.append(clean_text[cursor:])
            return "".join(out)

        @staticmethod
        def run(text, convert):
            clean, _ = Stub.normalize(text)
            spans = Stub.segment(clean)
            span_texts = [clean[a:b] for a, b in spans]
            latex = convert(span_texts) if span_texts else []
            output = Stub.splice(clean, spans, latex)
            return {
                "clean": clean,
                "spans": spans,
                "span_texts": span_texts,
                "latex": latex,
                "output": output,
            }

    return Stub


def load_pdf_pipeline():
    if not PDF_PIPELINE_FILE.exists():
        return make_stub_pdf_pipeline(), True, "missing bench/pdf_pipeline.py"
    try:
        spec = importlib.util.spec_from_file_location("bench_pdf_pipeline", PDF_PIPELINE_FILE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in ("normalize", "segment", "splice", "run"):
            if not hasattr(mod, name):
                raise RuntimeError(f"pdf_pipeline missing {name}")
        return mod, False, ""
    except Exception as e:
        return make_stub_pdf_pipeline(), True, str(e)


def load_catalog():
    out = {}

    pdf_items = read_json(PDF_PASTES)
    for r in pdf_items:
        out[r["id"]] = {
            "id": r["id"],
            "tier": "pdf-paste",
            "profile": r.get("profile"),
            "input": r["input"],
            "reference": r.get("reference", ""),
            "kind": "pdf-paste",
        }

    synth_rows = read_jsonl(SYNTH_SPANS)
    kind_to_tier = {"pdf": "synth-pdf", "unicode": "synth-unicode", "latex": "synth-latex"}
    for r in synth_rows:
        tier = kind_to_tier.get(r.get("kind"))
        if not tier:
            continue
        out[r["id"]] = {
            "id": r["id"],
            "tier": tier,
            "profile": r.get("profile"),
            "input": r["text"],
            "reference": r.get("reference", ""),
            "kind": r.get("kind"),
            "ok": r.get("ok"),
        }

    bench_rows = read_json(BENCH_DATA)
    for r in bench_rows:
        out[r["id"]] = {
            "id": r["id"],
            "tier": r.get("tier", "legacy"),
            "profile": None,
            "input": r["input"],
            "reference": r.get("reference", ""),
            "kind": "legacy",
        }
    return out, pdf_items, synth_rows


def create_items_file_if_missing(catalog, pdf_items, synth_rows):
    if ITEMS_FILE.exists():
        return
    pdf_ids = sorted(r["id"] for r in pdf_items)
    synth_ok = [r for r in synth_rows if r.get("ok") is True]
    synth_pdf_clean = sorted(
        r["id"] for r in synth_ok if r.get("kind") == "pdf" and r.get("profile") == "clean"
    )[:12]
    synth_pdf_dirty = sorted(
        r["id"] for r in synth_ok if r.get("kind") == "pdf" and r.get("profile") == "dirty"
    )[:12]
    synth_unicode = sorted(r["id"] for r in synth_ok if r.get("kind") == "unicode")[:18]
    synth_latex = sorted(r["id"] for r in synth_ok if r.get("kind") == "latex")[:18]
    payload = {
        "pdf-paste": pdf_ids,
        "synth-pdf-clean": synth_pdf_clean,
        "synth-pdf-dirty": synth_pdf_dirty,
        "synth-unicode": synth_unicode,
        "synth-latex": synth_latex,
    }
    # guard against stale source files
    all_ids = []
    for key in (
        "pdf-paste",
        "synth-pdf-clean",
        "synth-pdf-dirty",
        "synth-unicode",
        "synth-latex",
    ):
        all_ids.extend(payload[key])
    missing = [x for x in all_ids if x not in catalog]
    if missing:
        raise RuntimeError(f"cannot build {ITEMS_FILE.name}, unknown ids: {missing[:5]}")
    ITEMS_FILE.write_text(json.dumps(payload, indent=2) + "\n")


def default_item_ids(catalog):
    cfg = read_json(ITEMS_FILE)
    if isinstance(cfg, list):
        ids = list(cfg)
    elif isinstance(cfg, dict):
        ids = []
        ids.extend(cfg.get("pdf-paste", []))
        ids.extend(cfg.get("synth-pdf-clean", []))
        ids.extend(cfg.get("synth-pdf-dirty", []))
        ids.extend(cfg.get("synth-unicode", []))
        ids.extend(cfg.get("synth-latex", []))
    else:
        raise RuntimeError(f"bad {ITEMS_FILE}: expected list/dict")
    missing = [x for x in ids if x not in catalog]
    if missing:
        raise RuntimeError(f"{ITEMS_FILE.name} has unknown ids: {missing[:5]}")
    return ids


def convert_direct(model, system_prompt, convert_user, text, stats):
    user = append_no_think_if_needed(model, convert_user(text))
    out, meta = ollama_chat(
        model,
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
    )
    stats.add(meta)
    return out


def convert_span_single(model, span_text, stats):
    user = append_no_think_if_needed(model, span_text)
    out, meta = ollama_chat(
        model,
        [{"role": "system", "content": SPAN_SYSTEM}, {"role": "user", "content": user}],
    )
    stats.add(meta)
    return out


def convert_spans_pipeline(model, span_texts, stats, stub_convert=False):
    if stub_convert:
        return list(span_texts)
    return [convert_span_single(model, s, stats) for s in span_texts]


def convert_spans_batch(model, span_texts, stats, stub_convert=False):
    if stub_convert:
        return list(span_texts)
    if not span_texts:
        return []
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(span_texts))
    user = append_no_think_if_needed(model, numbered)
    out, meta = ollama_chat(
        model,
        [{"role": "system", "content": SPAN_SYSTEM_BATCH}, {"role": "user", "content": user}],
    )
    stats.add(meta)
    parsed = parse_numbered_lines(out, len(span_texts))
    if isinstance(parsed, list):
        return parsed
    final = [""] * len(span_texts)
    for i in range(len(span_texts)):
        if i in parsed and parsed[i]:
            final[i] = parsed[i]
        else:
            final[i] = convert_span_single(model, span_texts[i], stats)
    return final


def load_rows(path):
    p = pathlib.Path(path)
    if not p.exists():
        return []
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return []


def save_rows(path, rows):
    with open(path, "w") as f:
        json.dump(rows, f, indent=1)


def first60(text):
    return re.sub(r"\s+", " ", text.strip())[:60]


def collapse_inline(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def previous_boundary(text, idx):
    idx = max(0, min(int(idx), len(text)))
    prev_blank = text.rfind("\n\n", 0, idx)
    prev_punct = -1
    for ch in (".", "!", "?"):
        prev_punct = max(prev_punct, text.rfind(ch, 0, idx))
    out = 0
    if prev_blank >= 0:
        out = max(out, prev_blank + 2)
    if prev_punct >= 0:
        out = max(out, prev_punct + 1)
    return out


def next_boundary(text, idx):
    idx = max(0, min(int(idx), len(text)))
    next_blank = text.find("\n\n", idx)
    punct = SENT_BOUNDARY_RE.search(text, idx)
    options = []
    if next_blank >= 0:
        options.append(next_blank)
    if punct:
        options.append(punct.end())
    if not options:
        return len(text)
    return min(options)


def crop_marked_context(marked, max_chars=400):
    marked = marked.strip()
    if len(marked) <= max_chars:
        return marked
    a = marked.find("«")
    b = marked.find("»", a + 1)
    if a < 0 or b < 0:
        return marked[:max_chars].strip()
    b += 1
    span_len = b - a
    if span_len >= max_chars:
        return marked[a : a + max_chars].strip()
    left = (max_chars - span_len) // 2
    start = max(0, a - left)
    end = start + max_chars
    if end < b:
        end = b
        start = max(0, end - max_chars)
    if end > len(marked):
        end = len(marked)
        start = max(0, end - max_chars)
    return marked[start:end].strip()


def span_context(clean_text, span, max_chars=400):
    a, b = int(span[0]), int(span[1])
    left = previous_boundary(clean_text, a)
    right = next_boundary(clean_text, b)
    if right <= left:
        left = max(0, a - max_chars // 2)
        right = min(len(clean_text), b + max_chars // 2)
    base = clean_text[left:right]
    rel_a = max(0, min(len(base), a - left))
    rel_b = max(rel_a, min(len(base), b - left))
    marked = base[:rel_a] + "«" + base[rel_a:rel_b] + "»" + base[rel_b:]
    marked = collapse_inline(marked)
    return crop_marked_context(marked, max_chars=max_chars)


def mark_passage_fragments(clean_text, spans):
    out = []
    cur = 0
    for i, (a, b) in enumerate(spans, 1):
        a, b = int(a), int(b)
        out.append(clean_text[cur:a])
        out.append(f"«{i}: {clean_text[a:b]}»")
        cur = b
    out.append(clean_text[cur:])
    return "".join(out)


def main():
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise SystemExit("no models")

    system_prompt, convert_user = load_prompts()
    catalog, pdf_items, synth_rows = load_catalog()
    create_items_file_if_missing(catalog, pdf_items, synth_rows)

    if args.items.strip():
        item_ids = [x.strip() for x in args.items.split(",") if x.strip()]
    else:
        item_ids = default_item_ids(catalog)

    missing = [x for x in item_ids if x not in catalog]
    if missing:
        raise SystemExit(f"unknown item ids: {missing}")
    items = [catalog[x] for x in item_ids]

    pdf_pipeline, using_stub_pipeline, stub_reason = load_pdf_pipeline()
    if using_stub_pipeline:
        print(f"note: using stub pdf_pipeline ({stub_reason})", file=sys.stderr)
    if args.stub_convert and args.mode in {"direct", "direct-norm"}:
        print("note: --stub-convert ignored for direct modes", file=sys.stderr)

    rows = load_rows(args.out)
    done = {(r.get("approach"), r.get("item")) for r in rows if r.get("ms", -1) >= 0}

    for model in models:
        approach = f"{args.mode}:{model}"
        for it in items:
            key = (approach, it["id"])
            if key in done:
                continue
            t0 = time.perf_counter()
            stats = Stats()
            row = {
                "approach": approach,
                "model": model,
                "item": it["id"],
                "tier": it["tier"],
            }
            try:
                if args.mode == "direct":
                    output = convert_direct(model, system_prompt, convert_user, it["input"], stats)
                elif args.mode == "direct-norm":
                    clean_text, _ = pdf_pipeline.normalize(it["input"])
                    output = convert_direct(model, system_prompt, convert_user, clean_text, stats)
                    row["clean"] = clean_text
                else:
                    clean, _ = pdf_pipeline.normalize(it["input"])
                    spans = pdf_pipeline.segment(clean)
                    span_texts = [clean[int(a) : int(b)] for a, b in spans]
                    model_span_texts = [collapse_inline(s) for s in span_texts]
                    span_contexts = [span_context(clean, sp, max_chars=400) for sp in spans]

                    if args.mode == "pipeline":
                        span_latex = convert_spans_pipeline(
                            model,
                            model_span_texts,
                            stats,
                            stub_convert=args.stub_convert,
                        )
                    elif args.mode == "pipeline-batch":
                        span_latex = convert_spans_batch(
                            model,
                            model_span_texts,
                            stats,
                            stub_convert=args.stub_convert,
                        )
                    elif args.mode == "pipeline-ctx":
                        if args.stub_convert:
                            span_latex = list(model_span_texts)
                        else:
                            span_latex = []
                            for ctx, frag in zip(span_contexts, model_span_texts):
                                user = append_no_think_if_needed(
                                    model, f"Context: {ctx}\nFragment: {frag}"
                                )
                                out, meta = ollama_chat(
                                    model,
                                    [
                                        {"role": "system", "content": SPAN_SYSTEM_CTX},
                                        {"role": "user", "content": user},
                                    ],
                                )
                                stats.add(meta)
                                span_latex.append(out)
                    else:
                        if args.stub_convert:
                            span_latex = list(model_span_texts)
                        elif not model_span_texts:
                            span_latex = []
                        else:
                            marked = mark_passage_fragments(clean, spans)
                            user = append_no_think_if_needed(
                                model,
                                "Context: "
                                + marked
                                + "\nOutput one line per fragment exactly as '<n>. <latex>'.",
                            )
                            out, meta = ollama_chat(
                                model,
                                [
                                    {"role": "system", "content": SPAN_SYSTEM_CTX_BATCH},
                                    {"role": "user", "content": user},
                                ],
                            )
                            stats.add(meta)
                            parsed = parse_numbered_lines(out, len(model_span_texts))
                            if isinstance(parsed, list):
                                span_latex = parsed
                            else:
                                span_latex = [""] * len(model_span_texts)
                                for i in range(len(model_span_texts)):
                                    if i in parsed and parsed[i]:
                                        span_latex[i] = parsed[i]
                                        continue
                                    user = append_no_think_if_needed(
                                        model,
                                        f"Context: {span_contexts[i]}\nFragment: {model_span_texts[i]}",
                                    )
                                    single_out, single_meta = ollama_chat(
                                        model,
                                        [
                                            {"role": "system", "content": SPAN_SYSTEM_CTX},
                                            {"role": "user", "content": user},
                                        ],
                                    )
                                    stats.add(single_meta)
                                    span_latex[i] = single_out

                    cleaner = getattr(pdf_pipeline, "clean_span_latex", lambda s: str(s or "").strip())
                    span_latex = [cleaner(s) for s in list(span_latex)]
                    span_latex = [lx if lx else span_texts[i] for i, lx in enumerate(span_latex)]
                    output = pdf_pipeline.splice(clean, spans, span_latex)
                    row["spans"] = len(span_texts)
                    row["span_texts"] = span_texts
                    row["span_latex"] = span_latex
                    row["clean"] = clean
                ms = round((time.perf_counter() - t0) * 1000)
                v = validate_output(it["input"], output)
                row.update(
                    {
                        "ms": ms,
                        "output": output,
                        "valid": bool(v["ok"]),
                        "issues": v["issues"],
                        "requests": stats.requests,
                        "thinking": stats.thinking,
                        "evalTokens": stats.eval_tokens,
                        "promptTokens": stats.prompt_tokens,
                        "evalMs": stats.eval_ms,
                    }
                )
                print(
                    f"{approach} {it['id']} {ms}ms {row['valid']} {first60(output)}",
                    file=sys.stderr,
                )
            except Exception as e:
                row.update(
                    {
                        "ms": -1,
                        "output": "",
                        "valid": False,
                        "issues": [str(e)],
                        "requests": stats.requests,
                        "thinking": stats.thinking,
                        "evalTokens": stats.eval_tokens,
                        "promptTokens": stats.prompt_tokens,
                        "evalMs": stats.eval_ms,
                        "error": str(e)[:300],
                    }
                )
                if args.mode in PIPELINE_MODES:
                    row.setdefault("spans", 0)
                    row.setdefault("span_texts", [])
                print(
                    f"{approach} {it['id']} FAILED {first60(str(e))}",
                    file=sys.stderr,
                )
            rows.append(row)
            save_rows(args.out, rows)

    save_rows(args.out, rows)
    print(f"wrote {len(rows)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
