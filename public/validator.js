// Client-side output validation: (1) syntax — every math segment must parse
// with KaTeX; (2) fidelity — numbers and math concepts named in the input
// must show up in the LaTeX; (3) transcription — a homework question must
// not come back solved. Used to decide when to escalate from the tiny
// browser model to the server model.

const DELIM_RE = /\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|\$([^$\n]+?)\$/g;

export function extractMathSegments(latex) {
  const segments = [];
  let m;
  DELIM_RE.lastIndex = 0;
  while ((m = DELIM_RE.exec(latex)) !== null) {
    segments.push(m[1] ?? m[2] ?? m[3] ?? m[4]);
  }
  // No delimiters at all -> treat the whole output as one math segment
  // (mirrors the preview's wrapping heuristic).
  if (segments.length === 0 && latex.trim()) segments.push(latex.trim());
  return segments;
}

export function checkSyntax(latex) {
  const issues = [];
  for (const seg of extractMathSegments(latex)) {
    try {
      katex.renderToString(seg, { throwOnError: true, strict: false, displayMode: true });
    } catch (err) {
      const msg = String(err.message || err).replace(/^KaTeX parse error:\s*/, "");
      issues.push(`syntax: ${msg.slice(0, 120)}`);
    }
  }
  return issues;
}

// input pattern -> what the LaTeX must contain if the concept is mentioned
const CONCEPTS = [
  [/\bintegrals?\b/i, /\\o?int/, "integral → \\int"],
  [/\bsums?\b|\bsummation\b/i, /\\sum/, "sum → \\sum"],
  [/\bsquare roots?\b|\bsqrt\b/i, /\\sqrt/, "square root → \\sqrt"],
  [/\binfinity\b/i, /\\infty/, "infinity → \\infty"],
  [/\bplus or minus\b/i, /\\pm/, "plus or minus → \\pm"],
  [/\bfractions?\b|\bover\b|\bdivided by\b/i, /\\[dt]?frac|\//, "fraction → \\frac"],
  [/\bmatrix\b|\bmatrices\b/i, /matrix|\\begin\{array\}/, "matrix → matrix environment"],
  [/\blimits?\b/i, /\\lim/, "limit → \\lim"],
  [/\bequals?\b|\bis equal to\b/i, /=/, "equals → ="],
  [/\bgreater than or equal\b|\bat least\b/i, /\\geq?\b/, "≥ → \\geq"],
  [/\bless than or equal\b|\bat most\b/i, /\\leq?\b/, "≤ → \\leq"],
  [/\bnot equal\b/i, /\\neq?\b/, "≠ → \\neq"],
  [/\bderivatives?\b|\bpartial\b/i, /\\partial|\\frac\{\s*\\?d|'/, "derivative notation"],
];

const GREEK = ["alpha", "beta", "gamma", "delta", "epsilon", "theta", "lambda", "mu", "sigma", "phi", "omega", "pi"];

const NUMBER_WORDS = {
  zero: "0", one: "1", two: "2", three: "3", four: "4", five: "5",
  six: "6", seven: "7", eight: "8", nine: "9", ten: "10",
};

export function checkFidelity(input, latex) {
  const issues = [];

  for (const [inputRe, latexRe, label] of CONCEPTS) {
    if (inputRe.test(input) && !latexRe.test(latex)) {
      issues.push(`missing: ${label}`);
    }
  }

  for (const g of GREEK) {
    if (new RegExp(`\\b${g}\\b`, "i").test(input) && !new RegExp(`\\\\${g}`, "i").test(latex)) {
      issues.push(`missing: ${g} → \\${g}`);
    }
  }

  // Every literal number in the input should survive into the output.
  const numbers = new Set(input.match(/\d+(?:\.\d+)?/g) ?? []);
  for (const [word, digit] of Object.entries(NUMBER_WORDS)) {
    if (new RegExp(`\\b${word}\\b`, "i").test(input)) numbers.add(digit);
  }
  for (const n of numbers) {
    if (!latex.includes(n)) issues.push(`missing number: ${n}`);
  }

  return issues;
}

const QUESTION_LEAD = /^(what(?:'s| is| are)?|find|solve|evaluate|compute|calculate)\b/i;
const EQUALS_IN_INPUT = /=|\bequals?\b|\bis equal to\b/i;

// Chat models treat a pasted homework question as something to answer.
// A new "=" that was not in the input is the usual tell (they converted
// the integral and then appended the value).
export function checkNotAnswered(input, latex) {
  const t = input.trim();
  if (!QUESTION_LEAD.test(t) && !/\?\s*$/.test(t)) return [];
  if (EQUALS_IN_INPUT.test(t)) return [];
  if (!/=/.test(latex)) return [];
  return ["answered: output solves the question; convert the question text, do not evaluate"];
}

export function validateLatex(input, latex) {
  if (!latex || !latex.trim()) return { ok: false, issues: ["empty output"] };
  const issues = [...checkSyntax(latex), ...checkFidelity(input, latex), ...checkNotAnswered(input, latex)];
  return { ok: issues.length === 0, issues };
}
