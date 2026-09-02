// Unit tests for public/validator.js. Run: npm test  (node --test)
// validator.js expects a global `katex`; we provide the vendored build.
import { test } from "node:test";
import assert from "node:assert/strict";
import katex from "../public/vendor/katex/katex.mjs";

globalThis.katex = katex;

const { validateLatex, checkSyntax, checkFidelity, extractMathSegments } = await import("../public/validator.js");

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
