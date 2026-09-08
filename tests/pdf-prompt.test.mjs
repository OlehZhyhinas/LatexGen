import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { PDF_HINT, PDF_FEWSHOT } from "../public/pdf-prompt.js";
import { looksLikePdfPaste } from "../public/pdf-pipeline.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const pdfPastes = JSON.parse(readFileSync(join(ROOT, "bench/pdf-pastes.json"), "utf8"));
const samplePdfInput = pdfPastes.find((row) => row?.profile === "clean" && typeof row?.input === "string")?.input ?? pdfPastes[0]?.input ?? "";

test("PDF_HINT and PDF_FEWSHOT are populated", () => {
  assert.equal(typeof PDF_HINT, "string");
  assert.ok(PDF_HINT.trim().length > 0);
  assert.ok(Array.isArray(PDF_FEWSHOT));
  assert.ok(PDF_FEWSHOT.length > 0);
  for (const pair of PDF_FEWSHOT) {
    assert.equal(typeof pair?.user, "string");
    assert.equal(typeof pair?.assistant, "string");
    assert.ok(pair.user.trim().length > 0);
    assert.ok(pair.assistant.trim().length > 0);
  }
});

test("looksLikePdfPaste stays conservative on typed equation and clean prose", () => {
  assert.equal(looksLikePdfPaste("x^2 + y^2 = r^2"), false);
  assert.equal(looksLikePdfPaste("The quick brown fox jumps over the lazy dog."), false);
});

test("looksLikePdfPaste detects a known PDF paste sample", () => {
  assert.ok(samplePdfInput.length > 0, "bench/pdf-pastes.json must contain an input sample");
  assert.equal(looksLikePdfPaste(samplePdfInput), true);
});
