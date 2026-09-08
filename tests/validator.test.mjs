// Unit tests for public/validator.js. Run: npm test  (node --test)
// validator.js expects a global `katex`; we provide the vendored build.
import { test } from "node:test";
import assert from "node:assert/strict";
import katex from "../public/vendor/katex/katex.mjs";

globalThis.katex = katex;

const { validateLatex, checkSyntax, checkFidelity, checkNotAnswered, extractMathSegments } = await import("../public/validator.js");

test("valid display math passes syntax and fidelity", () => {
  const v = validateLatex(
    "the integral from 0 to infinity of e to the minus x squared",
    "\\[\\int_0^\\infty e^{-x^2}\\,dx\\]"
  );
  assert.equal(v.ok, true, JSON.stringify(v.issues));
});

test("unbalanced brace is a syntax issue", () => {
  const issues = checkSyntax("\\[x^{2\\]");
  assert.equal(issues.length, 1);
  assert.match(issues[0], /^syntax:/);
});

test("dropped concepts are reported as missing", () => {
  const issues = checkFidelity("the sum from n equals 1 to infinity of 1 over n squared", "\\[n^2 = 1\\]");
  assert.ok(issues.some((i) => i.includes("\\sum")), "sum");
  assert.ok(issues.some((i) => i.includes("\\infty")), "infinity");
  assert.ok(issues.some((i) => i.includes("\\frac")), "fraction");
});

test("numbers in the input must survive into the output", () => {
  const issues = checkFidelity("b squared minus 4ac over 2a", "\\[\\frac{b^2 - ac}{2a}\\]");
  assert.deepEqual(issues, ["missing number: 4"]);
});

test("number words count as numbers", () => {
  const issues = checkFidelity("one half m v squared", "\\[\\frac{1}{2}mv^2\\]");
  assert.deepEqual(issues, []);
});

test("greek letters are checked by name", () => {
  const issues = checkFidelity("alpha plus beta", "\\[a + b\\]");
  assert.ok(issues.some((i) => i.includes("\\alpha")));
  assert.ok(issues.some((i) => i.includes("\\beta")));
});

test("segments are extracted from every delimiter style", () => {
  const segs = extractMathSegments("text $a$ more \\(b\\) and $$c$$ then \\[d\\]");
  assert.deepEqual([...segs].sort(), ["a", "b", "c", "d"]);
});

test("bare math without delimiters is treated as one segment", () => {
  assert.deepEqual(extractMathSegments("x^2 + y^2"), ["x^2 + y^2"]);
});

test("empty output is rejected", () => {
  assert.equal(validateLatex("x squared", "").ok, false);
});

test("a solved homework question is rejected", () => {
  const issues = checkNotAnswered(
    "What is the integral of x squared from 0 to 1?",
    "\\[\\int_0^1 x^2\\,dx = \\frac{1}{3}\\]"
  );
  assert.equal(issues.length, 1);
  assert.match(issues[0], /^answered:/);
});

test("transcribing a question without evaluating it is fine", () => {
  const v = validateLatex(
    "What is the integral of x squared from 0 to 1?",
    "What is \\(\\int_0^1 x^{2}\\,dx\\)?"
  );
  assert.equal(v.ok, true, JSON.stringify(v.issues));
});

test("an equation that already contains equals is not treated as answered", () => {
  const issues = checkNotAnswered(
    "x plus 2 equals 5",
    "\\[x + 2 = 5\\]"
  );
  assert.deepEqual(issues, []);
});

// ---- unescapeDoubledBackslashes: JSON-escaped LaTeX from small models ----
import { unescapeDoubledBackslashes } from "../public/validator.js";

test("systematically double-escaped LaTeX is unescaped", () => {
  assert.equal(unescapeDoubledBackslashes("$\\\\mathbf{F}$ and $v = \\\\frac{ds}{dt}$"), "$\\mathbf{F}$ and $v = \\frac{ds}{dt}$");
  assert.equal(unescapeDoubledBackslashes("\\\\[ x^2 \\\\]"), "\\[ x^2 \\]");
});

test("matrix row breaks and normal LaTeX are left alone", () => {
  const matrix = "\\begin{pmatrix} a \\\\ b \\end{pmatrix}";
  assert.equal(unescapeDoubledBackslashes(matrix), matrix);
  assert.equal(unescapeDoubledBackslashes("\\[ x = \\frac{a}{b} \\]"), "\\[ x = \\frac{a}{b} \\]");
  assert.equal(unescapeDoubledBackslashes("plain prose"), "plain prose");
});

test("mixed single and double escaping is not touched", () => {
  const mixed = "\\\\( a \\\\) then \\frac{1}{2}";
  assert.equal(unescapeDoubledBackslashes(mixed), mixed);
});
