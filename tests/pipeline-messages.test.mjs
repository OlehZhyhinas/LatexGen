import { test } from "node:test";
import assert from "node:assert/strict";

import { PDF_HINT, PDF_FEWSHOT } from "../public/pdf-prompt.js";
import { normalize } from "../public/pdf-pipeline.js";

if (!globalThis.Worker) {
  globalThis.Worker = class {
    constructor() { this.onmessage = null; }
    postMessage() {}
  };
}

const { buildConvertMessages } = await import("../public/pipeline.js");

const CONVERT_USER = (text) => `Convert to LaTeX. Do not solve or answer.\n${text}`;

test("buildConvertMessages with pdf:false uses base prompt and raw text", () => {
  const rawText = "Body line\n\n1. Footnote note: extra";
  const messages = buildConvertMessages(rawText, { pdf: false });
  const expectedSystem = messages[0]?.content;
  assert.deepEqual(messages, [
    { role: "system", content: expectedSystem },
    { role: "user", content: CONVERT_USER(rawText) },
  ]);
});

test("buildConvertMessages with pdf:true appends hint/fewshot and normalizes user text", () => {
  const rawText = "Body line\n\n1. Footnote note: extra";
  const [normalized] = normalize(rawText);
  assert.notEqual(normalized, rawText);

  const baseSystem = buildConvertMessages(rawText, { pdf: false })[0].content;
  const expectedFewshot = PDF_FEWSHOT.flatMap((pair) => [
    { role: "user", content: CONVERT_USER(pair.user) },
    { role: "assistant", content: pair.assistant },
  ]);

  const messages = buildConvertMessages(rawText, { pdf: true });
  assert.deepEqual(messages, [
    { role: "system", content: `${baseSystem}\n\n${PDF_HINT}` },
    ...expectedFewshot,
    { role: "user", content: CONVERT_USER(normalized) },
  ]);
});
