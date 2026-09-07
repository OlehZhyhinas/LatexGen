#!/usr/bin/env python3
"""Synthesize prose+math passages with char-level span labels.

Math appears in three supported forms:
  pdf     - PDF copy/paste artifacts produced from LaTeX passage rendering
  unicode - plain-text/unicode math
  latex   - "$...$" or bare LaTeX math

Each example: {"id", "text", "reference", "spans": [{"start","end","latex","kind"}], "source", "ok"}.
Also emits a token/tag view (B-MATH/I-MATH/O).
"""
import argparse, collections, json, random, re, sys

import pdf_paste

# ~40 Wikipedia-verified classics: (id, latex, spoken_legacy, unicode, wiki source).
FORMULAS = [
    ("quadratic-formula", r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}",
     "x equals negative b plus or minus the square root of b squared minus four a c, all over two a",
     "x = (-b ± √(b²-4ac)) / 2a",
     "https://en.wikipedia.org/wiki/Quadratic_formula"),
    ("eulers-identity", r"e^{i\pi} + 1 = 0",
     "e to the power of i pi, plus one, equals zero",
     "e^(iπ) + 1 = 0",
     "https://en.wikipedia.org/wiki/Euler%27s_identity"),
    ("pythagorean-theorem", r"a^2 + b^2 = c^2",
     "a squared plus b squared equals c squared",
     "a² + b² = c²",
     "https://en.wikipedia.org/wiki/Pythagorean_theorem"),
    ("circle-area", r"A = \pi r^2",
     "the area A equals pi times r squared",
     "A = πr²",
     "https://en.wikipedia.org/wiki/Area_of_a_circle"),
    ("gaussian-integral", r"\int_{-\infty}^{\infty} e^{-x^2}\,dx = \sqrt{\pi}",
     "the integral from minus infinity to infinity of e to the minus x squared dx equals the square root of pi",
     "∫₋∞^∞ e^(-x²) dx = √π",
     "https://en.wikipedia.org/wiki/Gaussian_integral"),
    ("basel-problem", r"\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}",
     "the sum from n equals one to infinity of one over n squared equals pi squared over six",
     "∑ 1/n² = π²/6",
     "https://en.wikipedia.org/wiki/Basel_problem"),
    ("binomial-theorem", r"(x+y)^n = \sum_{k=0}^{n} \binom{n}{k} x^{k} y^{n-k}",
     "the quantity x plus y raised to the n equals the sum from k equals zero to n of n choose k times x to the k times y to the n minus k",
     "(x+y)^n = ∑ C(n,k) x^k y^(n-k)",
     "https://en.wikipedia.org/wiki/Binomial_theorem"),
    ("bayes-theorem", r"P(A\vert B) = \frac{P(B \vert A) P(A)}{P(B)}",
     "the probability of A given B equals the probability of B given A times the probability of A all over the probability of B",
     "P(A|B) = P(B|A)P(A) / P(B)",
     "https://en.wikipedia.org/wiki/Bayes%27_theorem"),
    ("schrodinger-equation", r"i \hbar \frac{d}{d t}\vert\Psi(t)\rangle = \hat H\vert\Psi(t)\rangle",
     "i times h-bar times the derivative with respect to t of ket Psi of t equals the Hamiltonian operator H hat applied to ket Psi of t",
     "iħ d/dt |Ψ(t)⟩ = Ĥ|Ψ(t)⟩",
     "https://en.wikipedia.org/wiki/Schr%C3%B6dinger_equation"),
    ("cauchy-integral-formula", r"f(a) = \frac{1}{2\pi i} \oint_\gamma \frac{f(z)}{z-a}\,dz",
     "f of a equals one over two pi i times the closed contour integral over gamma of f of z divided by z minus a dz",
     "f(a) = 1/(2πi) ∮ f(z)/(z-a) dz",
     "https://en.wikipedia.org/wiki/Cauchy%27s_integral_formula"),
    ("riemann-zeta-functional-equation", r"\zeta(s) = 2^s \pi^{s-1} \sin\left( \frac{\pi s}{2} \right) \Gamma(1-s) \zeta(1-s)",
     "zeta of s equals two to the s times pi to the s minus one times sine of pi s over two times the gamma function of one minus s times zeta of one minus s",
     "ζ(s) = 2^s π^(s-1) sin(πs/2) Γ(1-s) ζ(1-s)",
     "https://en.wikipedia.org/wiki/Riemann_zeta_function"),
    ("einstein-field-equations", r"G_{\mu \nu} + \Lambda g_{\mu \nu} = \kappa T_{\mu \nu}",
     "G mu nu plus capital lambda times g mu nu equals kappa times T mu nu",
     "G_μν + Λg_μν = κT_μν",
     "https://en.wikipedia.org/wiki/Einstein_field_equations"),
    ("newtons-second-law", r"\mathbf{F} = m\mathbf{a}",
     "the vector F equals m times the vector a",
     "F = ma",
     "https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion"),
    ("kinetic-energy", r"E_\text{k} = \frac{1}{2} mv^2",
     "E sub k equals one half m v squared",
     "E_k = ½mv²",
     "https://en.wikipedia.org/wiki/Kinetic_energy"),
    ("derivative-definition", r"f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}",
     "f prime of x equals the limit as h approaches zero of f of x plus h minus f of x all over h",
     "f'(x) = lim_(h→0) (f(x+h)-f(x))/h",
     "https://en.wikipedia.org/wiki/Derivative"),
    ("taylor-series", r"f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n",
     "f of x equals the sum from n equals zero to infinity of the nth derivative of f at a over n factorial times x minus a to the n",
     "f(x) = ∑ f^(n)(a)/n! (x-a)^n",
     "https://en.wikipedia.org/wiki/Taylor_series"),
    ("maxwell-gauss-law", r"\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}",
     "the divergence of E equals rho over epsilon naught",
     "∇·E = ρ/ε₀",
     "https://en.wikipedia.org/wiki/Gauss%27s_law"),
    ("maxwell-ampere-law", r"\nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0 \varepsilon_0 \frac{\partial \mathbf{E}}{\partial t}",
     "the curl of B equals mu naught J plus mu naught epsilon naught times the partial derivative of E with respect to t",
     "∇×B = μ₀J + μ₀ε₀ ∂E/∂t",
     "https://en.wikipedia.org/wiki/Amp%C3%A8re%27s_circuital_law"),
    ("determinant-2x2", r"\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc",
     "the determinant of the two by two matrix a b c d equals a d minus b c",
     "det([[a,b],[c,d]]) = ad-bc",
     "https://en.wikipedia.org/wiki/Determinant"),
    ("maxwell-aligned-pair", r"\begin{aligned} \nabla \cdot \mathbf{E} &= \frac{\rho}{\varepsilon_0} \\ \nabla \times \mathbf{B} &= \mu_0 \mathbf{J} + \mu_0 \varepsilon_0 \frac{\partial \mathbf{E}}{\partial t} \end{aligned}",
     "the divergence of E equals rho over epsilon naught and the curl of B equals mu naught J plus mu naught epsilon naught partial E partial t",
     "∇·E = ρ/ε₀; ∇×B = μ₀J + μ₀ε₀ ∂E/∂t",
     "https://en.wikipedia.org/wiki/Maxwell%27s_equations"),
    ("law-of-cosines", r"c^2 = a^2 + b^2 - 2ab\cos C",
     "c squared equals a squared plus b squared minus two a b cosine C",
     "c² = a² + b² - 2ab·cos C",
     "https://en.wikipedia.org/wiki/Law_of_cosines"),
    ("distance-formula", r"d = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}",
     "d equals the square root of x two minus x one squared plus y two minus y one squared",
     "d = √((x₂-x₁)² + (y₂-y₁)²)",
     "https://en.wikipedia.org/wiki/Euclidean_distance"),
    ("compound-interest", r"A = P\left(1 + \frac{r}{n}\right)^{nt}",
     "A equals P times the quantity one plus r over n raised to the n t",
     "A = P(1 + r/n)^(nt)",
     "https://en.wikipedia.org/wiki/Compound_interest"),
    ("normal-distribution-pdf", r"f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}",
     "f of x equals one over sigma times the square root of two pi times e to the minus x minus mu squared over two sigma squared",
     "f(x) = 1/(σ√(2π)) e^(-(x-μ)²/2σ²)",
     "https://en.wikipedia.org/wiki/Normal_distribution"),
    ("geometric-series-sum", r"\sum_{k=0}^{\infty} ar^k = \frac{a}{1-r}",
     "the sum from k equals zero to infinity of a r to the k equals a over one minus r",
     "∑ ar^k = a/(1-r)",
     "https://en.wikipedia.org/wiki/Geometric_series"),
    ("de-moivres-formula", r"(\cos\theta + i\sin\theta)^n = \cos(n\theta) + i\sin(n\theta)",
     "the quantity cosine theta plus i sine theta raised to the n equals cosine of n theta plus i sine of n theta",
     "(cos θ + i sin θ)^n = cos(nθ) + i sin(nθ)",
     "https://en.wikipedia.org/wiki/De_Moivre%27s_formula"),
    ("triangle-inequality", r"|a + b| \le |a| + |b|",
     "the absolute value of a plus b is less than or equal to the absolute value of a plus the absolute value of b",
     "|a+b| ≤ |a|+|b|",
     "https://en.wikipedia.org/wiki/Triangle_inequality"),
    ("chain-rule", r"\frac{dy}{dx} = \frac{dy}{du}\cdot\frac{du}{dx}",
     "d y d x equals d y d u times d u d x",
     "dy/dx = dy/du · du/dx",
     "https://en.wikipedia.org/wiki/Chain_rule"),
    ("product-rule", r"(fg)' = f'g + fg'",
     "the derivative of f g equals f prime g plus f g prime",
     "(fg)' = f'g + fg'",
     "https://en.wikipedia.org/wiki/Product_rule"),
    ("mean-value-theorem", r"f'(c) = \frac{f(b)-f(a)}{b-a}",
     "f prime of c equals f of b minus f of a all over b minus a",
     "f'(c) = (f(b)-f(a))/(b-a)",
     "https://en.wikipedia.org/wiki/Mean_value_theorem"),
    ("fourier-transform", r"\hat f(\xi) = \int_{-\infty}^{\infty} f(x) e^{-2\pi i x \xi}\,dx",
     "f hat of xi equals the integral from minus infinity to infinity of f of x times e to the minus two pi i x xi dx",
     "f̂(ξ) = ∫ f(x) e^(-2πixξ) dx",
     "https://en.wikipedia.org/wiki/Fourier_transform"),
    ("planck-einstein-relation", r"E = h\nu",
     "E equals h times nu",
     "E = hν",
     "https://en.wikipedia.org/wiki/Planck%E2%80%93Einstein_relation"),
    ("mass-energy-equivalence", r"E = mc^2",
     "E equals m c squared",
     "E = mc²",
     "https://en.wikipedia.org/wiki/Mass%E2%80%93energy_equivalence"),
    ("ideal-gas-law", r"PV = nRT",
     "P V equals n R T",
     "PV = nRT",
     "https://en.wikipedia.org/wiki/Ideal_gas_law"),
    ("coulombs-law", r"F = k_e \frac{q_1 q_2}{r^2}",
     "F equals k sub e times q one q two over r squared",
     "F = k_e q₁q₂/r²",
     "https://en.wikipedia.org/wiki/Coulomb%27s_law"),
    ("ohms-law", r"V = IR",
     "V equals I R",
     "V = IR",
     "https://en.wikipedia.org/wiki/Ohm%27s_law"),
    ("newtons-law-of-gravitation", r"F = G\frac{m_1 m_2}{r^2}",
     "F equals G times m one m two over r squared",
     "F = Gm₁m₂/r²",
     "https://en.wikipedia.org/wiki/Newton%27s_law_of_universal_gravitation"),
    ("momentum", r"p = mv",
     "p equals m v",
     "p = mv",
     "https://en.wikipedia.org/wiki/Momentum"),
    ("work-energy-theorem", r"W = \Delta E_k",
     "W equals delta E sub k",
     "W = ΔE_k",
     "https://en.wikipedia.org/wiki/Work_(physics)"),
    ("entropy-boltzmann", r"S = k_B \ln \Omega",
     "S equals k sub B times the natural log of Omega",
     "S = k_B ln Ω",
     "https://en.wikipedia.org/wiki/Boltzmann%27s_entropy_formula"),
    ("logistic-function", r"\sigma(x) = \frac{1}{1+e^{-x}}",
     "sigma of x equals one over one plus e to the minus x",
     "σ(x) = 1/(1+e^(-x))",
     "https://en.wikipedia.org/wiki/Logistic_function"),
    ("limit-definition-of-e", r"e = \lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n",
     "e equals the limit as n approaches infinity of the quantity one plus one over n raised to the n",
     "e = lim_(n→∞) (1 + 1/n)^n",
     "https://en.wikipedia.org/wiki/E_(mathematical_constant)"),
]

# Sentence templates carrying exactly one {math} slot, homework/textbook register.
TEMPLATES = [
    "Recall that {math}.",
    "Newton's second law tells us that {math}, so the net force follows directly.",
    "For any real number x, we know that {math}.",
    "This converges because {math} holds.",
    "The homework asks us to verify that {math}.",
    "It follows that {math}, as shown in the derivation above.",
    "According to the textbook, {math} for all positive integers n.",
    "One classic result states that {math}.",
    "Substituting into the definition gives {math}.",
    "This can be rewritten as {math}.",
    "The lecture notes derive {math} from first principles.",
    "As an exercise, show that {math}.",
    "Physically, this means {math}.",
    "In the limit, we find {math}.",
    "The proof begins by noting that {math}.",
    "A well-known identity states that {math}.",
    "We are told that {math}, which we will use later.",
    "Consider the equation {math} and its consequences.",
    "The textbook defines {math} for the purposes of this chapter.",
    "Applying the theorem yields {math}.",
    "Note that {math}, which simplifies the rest of the argument.",
    "Here the key relation is {math}.",
]

# Connective sentences with no math at all: fill passages, add negative tokens.
CONNECTIVES = [
    "Which converges because the terms shrink rapidly.",
    "This ties together several fundamental constants in a single equation.",
    "The result generalizes to higher dimensions as well.",
    "Both answers involve pi even though neither problem mentions a circle.",
    "This is one of the most quoted results in the whole course.",
    "The derivation is left as an exercise for the reader.",
]

# Hard negatives: numbers, units, dates, variable names spelled out in words.
# None of these should be labelled math. Ambiguous ones are noted inline.
NEGATIVES = [
    "The variable x was measured five times during the experiment.",
    "At time t the sample had cooled to room temperature.",  # ambiguous: "t" reads like a math symbol but is prose here
    "5 apples were left on the table after lunch.",
    "The professor arrived at 3 p.m. on Tuesday, September 7.",
    "We used 10 milliliters of solution in the second trial.",
    "The value n was chosen arbitrarily by the teaching assistant.",  # ambiguous: bare "n" outside a formula
    "She scored 95 out of 100 on the midterm.",
    "The constant c in the problem refers to the speed of light, not a variable.",  # ambiguous: "c" named but not used symbolically
    "There were 12 students in the study group that evening.",
    "The textbook costs twenty dollars at the campus store.",
    "Chapter 3 covers roughly 40 pages of new material.",
    "The experiment ran for 2 hours before the readings stabilized.",
    "The letter e in his name is not related to Euler's number.",  # ambiguous: names "e" but is prose, not math
    "He labeled the box A, though it had nothing to do with a matrix.",  # ambiguous: names "A" but is prose
    "The room number was 214, right next to the lab.",
]

KINDS = ("pdf", "unicode", "latex")

# tokens: runs of unicode word chars, runs of digits, or single other chars
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
PROSE_RUN_RE = re.compile(r"\b[a-z]{4,}(?: [a-z]{4,}){2,}\b")


def formula_fields(formula):
    if len(formula) == 5:
        fid, latex, _spoken, unicode_, source = formula
    else:
        fid, latex, unicode_, source = formula
    return fid, latex, unicode_, source


def parse_mix(s):
    out = {k: 0.0 for k in KINDS}
    parts = [x.strip() for x in s.split(",") if x.strip()]
    if not parts:
        raise SystemExit("empty --mix")
    for part in parts:
        if "=" not in part:
            raise SystemExit(f"bad --mix segment: {part}")
        key, value = part.split("=", 1)
        key = key.strip()
        if key not in out:
            raise SystemExit(f"unknown kind in --mix: {key}")
        try:
            out[key] = float(value.strip())
        except ValueError as exc:
            raise SystemExit(f"bad weight for {key}: {value}") from exc
    total = sum(out.values())
    if total <= 0:
        raise SystemExit("--mix weights sum to zero")
    for key in out:
        out[key] /= total
    return out


def choose_kind(rng, mix):
    keys = list(mix.keys())
    weights = [mix[k] for k in keys]
    return rng.choices(keys, weights=weights, k=1)[0]


def render_non_pdf_math(rng, formula, kind):
    _fid, latex, unicode_, _source = formula_fields(formula)
    if kind == "unicode":
        return unicode_, latex
    if kind == "latex":
        return (f"${latex}$" if rng.random() < 0.5 else latex), latex
    raise ValueError(f"unexpected kind: {kind}")


def render_pdf_math(rng, latex):
    if rng.random() < 0.35:
        return f"\\[ {latex} \\]"
    return f"\\( {latex} \\)"


def gen_example_spec(rng, idx, kind):
    n_parts = rng.randint(1, 3)
    text = ""
    spans = []
    sources = []
    for _ in range(n_parts):
        roll = rng.random()
        if roll < 0.25:
            sentence = rng.choice(NEGATIVES)
        elif roll < 0.35:
            sentence = rng.choice(CONNECTIVES)
        else:
            tmpl = rng.choice(TEMPLATES)
            formula = rng.choice(FORMULAS)
            _fid, latex, _unicode, source = formula_fields(formula)
            if kind == "pdf":
                mathstr = render_pdf_math(rng, latex)
            else:
                mathstr, latex = render_non_pdf_math(rng, formula, kind)
            prefix, _, suffix = tmpl.partition("{math}")
            sentence = prefix + mathstr + suffix
            if kind != "pdf":
                start = len(text) + (1 if text else 0) + len(prefix)
                end = start + len(mathstr)
                spans.append({"start": start, "end": end, "latex": latex, "kind": kind})
            sources.append(source)
        text = f"{text} {sentence}" if text else sentence
    return {
        "id": f"synth-{idx:05d}",
        "text": text,
        "spans": spans if kind != "pdf" else [],
        "needs_pdf": kind == "pdf",
        "kind": kind,
        "source": ", ".join(sources) if sources else "synthetic (no math, hard negative)",
    }


def tokenize(text):
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def bio_tags(text, spans):
    tokens = tokenize(text)
    tags = []
    for _tok, start, _end in tokens:
        tag = "O"
        for sp in spans:
            if sp["start"] <= start < sp["end"]:
                tag = "B-MATH" if start == sp["start"] else "I-MATH"
                break
        tags.append(tag)
    return [t for t, _, _ in tokens], tags


def render_expected_output(text, spans):
    out = []
    cursor = 0
    for span in sorted(spans, key=lambda x: x["start"]):
        start = int(span["start"])
        end = int(span["end"])
        out.append(text[cursor:start])
        out.append(f"\\( {span['latex']} \\)")
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def _run_pdf_batch(batch, profile, layout, seed, knob_rows=None):
    passages = [x["text"] for x in batch]
    extracted, stats = pdf_paste.pdf_paste(
        passages,
        profile=profile,
        layout=layout,
        seed=seed,
        knob_rows=knob_rows,
    )
    out = {}
    for spec, ex in zip(batch, extracted):
        out[spec["id"]] = {
            "id": spec["id"],
            "text": ex["text"],
            "reference": spec["text"],
            "spans": ex["spans"],
            "kind": "pdf",
            "profile": profile,
            "knobs": ex.get("knobs", []),
            "source": spec["source"],
            "ok": bool(ex.get("ok", True)),
            "reason": ex.get("reason", ""),
        }
    return out, stats


def _seed_for_example(seed, ex_id, salt=0):
    idx = int(ex_id.split("-")[-1])
    return ((seed * 1_000_003) + (idx * 9_173) + salt) & 0xFFFFFFFF


def _dirty_knobs_for_example(seed, ex_id):
    rng = random.Random(_seed_for_example(seed, ex_id, salt=73))
    n = rng.randint(2, 4)
    return sorted(rng.sample(pdf_paste.DIRTY_KNOBS, k=n))


def generate(n, seed, mix, pdf_chunk_size, layout, dirty_frac):
    rng = random.Random(seed)
    specs = []
    for i in range(n):
        specs.append(gen_example_spec(rng, i, choose_kind(rng, mix)))

    finalized = {}
    bad_pdf = 0
    bad_pdf_by_profile = collections.Counter()
    pdf_attempted_by_profile = collections.Counter()
    dirty_knob_total = collections.Counter()
    dirty_knob_span_total = collections.Counter()
    dirty_knob_bad = collections.Counter()
    dirty_knob_extended = collections.Counter()
    dirty_knob_partial = collections.Counter()

    for spec in specs:
        if spec["needs_pdf"]:
            continue
        reference = render_expected_output(spec["text"], spec["spans"])
        finalized[spec["id"]] = {
            "id": spec["id"],
            "text": spec["text"],
            "reference": reference,
            "spans": spec["spans"],
            "kind": spec["kind"],
            "profile": None,
            "knobs": [],
            "source": spec["source"],
            "ok": True,
            "reason": "",
        }

    pdf_specs = [s for s in specs if s["needs_pdf"]]
    clean_specs = []
    dirty_specs = []
    for spec in pdf_specs:
        if rng.random() < dirty_frac:
            dirty_specs.append(spec)
        else:
            clean_specs.append(spec)

    for i in range(0, len(clean_specs), pdf_chunk_size):
        chunk = clean_specs[i : i + pdf_chunk_size]
        pdf_attempted_by_profile["clean"] += len(chunk)
        chunk_seed = _seed_for_example(seed, chunk[0]["id"], salt=101) if chunk else seed + 101
        batch_out, _stats = _run_pdf_batch(
            chunk,
            profile="clean",
            layout=layout,
            seed=chunk_seed,
        )
        finalized.update(batch_out)
        bad = sum(1 for ex in batch_out.values() if not ex.get("ok", True))
        bad_pdf += bad
        bad_pdf_by_profile["clean"] += bad

    for i in range(0, len(dirty_specs), pdf_chunk_size):
        chunk = dirty_specs[i : i + pdf_chunk_size]
        pdf_attempted_by_profile["dirty"] += len(chunk)
        chunk_knobs = [_dirty_knobs_for_example(seed, spec["id"]) for spec in chunk]
        chunk_seed = _seed_for_example(seed, chunk[0]["id"], salt=10001) if chunk else seed + 10001
        batch_out, stats = _run_pdf_batch(
            chunk,
            profile="dirty",
            layout=layout,
            seed=chunk_seed,
            knob_rows=chunk_knobs,
        )
        finalized.update(batch_out)
        bad = sum(1 for ex in batch_out.values() if not ex.get("ok", True))
        bad_pdf += bad
        bad_pdf_by_profile["dirty"] += bad
        for knob, row in stats["drop_rate_by_knob"].items():
            dirty_knob_total[knob] += row["total"]
            dirty_knob_span_total[knob] += row.get("span_total", 0)
            dirty_knob_bad[knob] += row["failed"]
            dirty_knob_extended[knob] += row.get("extended_spans", 0)
            dirty_knob_partial[knob] += row.get("partial_spans", 0)

    examples = []
    for i in range(n):
        ex_id = f"synth-{i:05d}"
        ex = finalized.get(ex_id)
        if ex is None:
            continue
        toks, tags = bio_tags(ex["text"], ex["spans"])
        ex["tokens"] = toks
        ex["tags"] = tags
        examples.append(ex)
    stats = {
        "pdf_attempted": int(sum(pdf_attempted_by_profile.values())),
        "pdf_bad": int(bad_pdf),
        "pdf_attempted_by_profile": dict(pdf_attempted_by_profile),
        "pdf_bad_by_profile": dict(bad_pdf_by_profile),
        "dirty_knob_total": dict(dirty_knob_total),
        "dirty_knob_span_total": dict(dirty_knob_span_total),
        "dirty_knob_bad": dict(dirty_knob_bad),
        "dirty_knob_extended": dict(dirty_knob_extended),
        "dirty_knob_partial": dict(dirty_knob_partial),
    }
    return examples, stats


def _mark_empty_spans(examples):
    for ex in examples:
        text = ex.get("text", "")
        for sp in ex.get("spans", []):
            piece = text[sp["start"]:sp["end"]]
            if piece.replace("\u200b", "").strip():
                continue
            ex["ok"] = False
            ex["reason"] = "empty span"
            break


def check(examples, pdf_stats):
    span_kind_counts = collections.Counter()
    example_kind_counts = collections.Counter()
    example_profile_counts = collections.Counter()
    bad_reason_counts = collections.Counter()
    math_tokens = 0
    total_tokens = 0
    bad = 0
    for ex in examples:
        kind = ex.get("kind", "unknown")
        profile = ex.get("profile") or "-"
        example_kind_counts[kind] += 1
        example_profile_counts[(kind, profile)] += 1
        text = ex["text"]
        token_bounds = {0, len(text)}
        for _tok, t_start, t_end in tokenize(text):
            token_bounds.add(t_start)
            token_bounds.add(t_end)
        spans = sorted(ex["spans"], key=lambda s: s["start"])
        for i, sp in enumerate(spans):
            piece = text[sp["start"]:sp["end"]]
            stripped = piece.replace("\u200b", "").strip()
            if sp["start"] not in token_bounds or sp["end"] not in token_bounds:
                print(
                    f"BAD {ex['id']}: span not on token boundary start={sp['start']} end={sp['end']}",
                    file=sys.stderr,
                )
                bad += 1
            if not stripped:
                ex["ok"] = False
                ex["reason"] = "empty span"
                print(f"BAD {ex['id']}: empty span slice", file=sys.stderr)
                bad += 1
            compact = re.sub(r"\s+", " ", piece.replace("\u200b", " ")).strip()
            latex = sp.get("latex", "")
            if compact and PROSE_RUN_RE.search(compact) and "\\text" not in latex:
                print(
                    f"BAD {ex['id']}: prose-like span slice without \\text ({compact!r})",
                    file=sys.stderr,
                )
                bad += 1
            if i > 0 and sp["start"] < spans[i - 1]["end"]:
                print(f"BAD {ex['id']}: overlapping spans", file=sys.stderr)
                bad += 1
            span_kind_counts[sp["kind"]] += 1
        for tag in ex["tags"]:
            total_tokens += 1
            if tag != "O":
                math_tokens += 1
        if not ex.get("ok", True):
            bad_reason_counts[ex.get("reason") or "unspecified"] += 1
    ok_false = sum(1 for ex in examples if not ex.get("ok", True))
    print(f"examples: {len(examples)}", file=sys.stderr)
    print(f"examples ok=false: {ok_false}", file=sys.stderr)
    print(f"ok=false by reason: {dict(bad_reason_counts)}", file=sys.stderr)
    print(f"example counts by kind: {dict(example_kind_counts)}", file=sys.stderr)
    prof_counts = {f"{k}/{p}": c for (k, p), c in sorted(example_profile_counts.items())}
    print(f"example counts by kind/profile: {prof_counts}", file=sys.stderr)
    print(f"spans by kind: {dict(span_kind_counts)}", file=sys.stderr)
    attempted = pdf_stats["pdf_attempted"]
    bad_pdf = pdf_stats["pdf_bad"]
    bad_rate = (bad_pdf / attempted) if attempted else 0.0
    print(
        f"pdf ok=false: {bad_pdf}/{attempted} ({bad_rate:.1%})",
        file=sys.stderr,
    )
    by_prof = {}
    for prof, n in sorted(pdf_stats["pdf_attempted_by_profile"].items()):
        d = pdf_stats["pdf_bad_by_profile"].get(prof, 0)
        by_prof[prof] = {
            "attempted": n,
            "bad": d,
            "bad_rate": (d / n) if n else 0.0,
        }
    print(f"pdf ok=false by profile: {by_prof}", file=sys.stderr)
    knob_drop = {}
    for knob, n in sorted(pdf_stats["dirty_knob_total"].items()):
        d = pdf_stats["dirty_knob_bad"].get(knob, 0)
        span_n = pdf_stats["dirty_knob_span_total"].get(knob, 0)
        ext = pdf_stats["dirty_knob_extended"].get(knob, 0)
        part = pdf_stats["dirty_knob_partial"].get(knob, 0)
        knob_drop[knob] = {
            "total_examples": n,
            "bad": d,
            "bad_rate": (d / n) if n else 0.0,
            "total_spans": span_n,
            "extended_spans": ext,
            "extended_rate": (ext / span_n) if span_n else 0.0,
            "partial_spans": part,
            "partial_rate": (part / span_n) if span_n else 0.0,
        }
    print(f"pdf dirty ok=false by knob: {knob_drop}", file=sys.stderr)
    ext_partial_by_profile = {}
    for prof in ("clean", "dirty"):
        prof_rows = [
            ex
            for ex in examples
            if ex.get("kind") == "pdf" and ex.get("profile") == prof
        ]
        total_spans = sum(len(ex.get("spans", [])) for ex in prof_rows)
        ext_spans = sum(
            1 for ex in prof_rows for sp in ex.get("spans", []) if sp.get("extended")
        )
        part_spans = sum(
            1 for ex in prof_rows for sp in ex.get("spans", []) if sp.get("partial")
        )
        ext_partial_by_profile[prof] = {
            "total_spans": total_spans,
            "extended_spans": ext_spans,
            "extended_rate": (ext_spans / total_spans) if total_spans else 0.0,
            "partial_spans": part_spans,
            "partial_rate": (part_spans / total_spans) if total_spans else 0.0,
        }
    print(f"pdf extended/partial by profile: {ext_partial_by_profile}", file=sys.stderr)
    frac = math_tokens / total_tokens if total_tokens else 0.0
    print(f"tokens tagged math: {math_tokens}/{total_tokens} ({frac:.1%})", file=sys.stderr)
    print(f"bad spans: {bad}", file=sys.stderr)
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="bench/synth-spans.jsonl")
    ap.add_argument("--mix", default="pdf=0.5,unicode=0.25,latex=0.25")
    ap.add_argument("--dirty-frac", type=float, default=0.5)
    ap.add_argument("--pdf-chunk-size", type=int, default=100)
    ap.add_argument("--layout", action="store_true")
    ap.add_argument("--drop-bad", action="store_true", help="omit examples with ok=false from output JSONL")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not (0.0 <= args.dirty_frac <= 1.0):
        raise SystemExit("--dirty-frac must be in [0,1]")

    mix = parse_mix(args.mix)
    examples, pdf_stats = generate(
        args.n,
        args.seed,
        mix=mix,
        pdf_chunk_size=args.pdf_chunk_size,
        layout=args.layout,
        dirty_frac=args.dirty_frac,
    )
    _mark_empty_spans(examples)
    to_write = [ex for ex in examples if ex.get("ok", True)] if args.drop_bad else examples
    with open(args.out, "w") as f:
        for ex in to_write:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"{len(to_write)} examples -> {args.out}", file=sys.stderr)
    if args.drop_bad:
        print(f"dropped ok=false examples: {len(examples) - len(to_write)}", file=sys.stderr)
    print(f"mix: {mix}", file=sys.stderr)
    print(f"pdf dirty_frac: {args.dirty_frac}", file=sys.stderr)

    if args.check:
        ok = check(examples, pdf_stats=pdf_stats)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
