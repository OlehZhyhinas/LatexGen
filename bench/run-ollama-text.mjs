#!/usr/bin/env node
// Text benchmark through Ollama: every model gets the pipeline's exact
// prompts, greedy decoding, thinking off, and the same validator the app uses
// to decide escalation. Prompts and post-processing are read out of
// public/pipeline.js so this harness cannot drift from what ships.
//
//   node bench/run-ollama-text.mjs --models qwen3:4b,qwen3.5:4b --out bench/results-x.json [--data bench/bench-data-extended.json]
//   [--normalize] [--system-extra prompts/extra.txt] [--fewshot bench/fewshot.json] [--label normpdf]
//   [--pdf-gate]  with --pdf-gate, --normalize/--system-extra/--fewshot apply only when input
//                 looks like a PDF paste; other items run as plain direct. Each row has gated: true|false.
//   (re-running with the same --out resumes: finished (model, item) pairs are skipped)
//   python3 bench/judge.py --rows bench/results-x.json bench/judged-x.json
//
// Row format matches what bench.html posts to /api/bench (approach, item,
// tier, ms, output) plus model, valid and issues from validator.js.
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const OLLAMA = process.env.OLLAMA_URL || "http://localhost:11434";

const args = process.argv.slice(2);
const opt = (name, dflt) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : dflt; };
const models = opt("--models", "").split(",").map((s) => s.trim()).filter(Boolean);
const out = opt("--out", path.join(here, "results-ollama-text.json"));
const onlyItems = opt("--items", "").split(",").filter(Boolean);
const dataFile = opt("--data", path.join(here, "bench-data.json"));
const runLabel = opt("--label", "").trim();
const systemExtraFile = opt("--system-extra", "").trim();
const fewshotFile = opt("--fewshot", "").trim();
const useNormalize = args.includes("--normalize");
const usePdfGate = args.includes("--pdf-gate");
// Qwen3 (not 3.5) under Ollama sometimes ignores `think:false` and reasons in
// the visible content; the family's documented soft switch stops it. The app's
// WebLLM path disables thinking through the chat template instead.
const noThinkSuffix = opt("--no-think-suffix", "") ; // e.g. "/no_think"
if (!models.length) { console.error("usage: --models a,b,c [--out file] [--items id,id]"); process.exit(2); }

// ---- prompts and helpers lifted from pipeline.js ----
const pipelineJs = process.env.PIPELINE_JS
  ? path.resolve(process.env.PIPELINE_JS)
  : path.join(root, "public/pipeline.js");
const src = readFileSync(pipelineJs, "utf8");
const grab = (name) => {
  const m = src.match(new RegExp(`const ${name} = (\`[^\`]*\`|\\(text\\) => \`[^\`]*\`);`));
  if (!m) throw new Error(`could not find ${name} in ${pipelineJs}`);
  return new Function(`return ${m[1]};`)();
};
const SYSTEM_PROMPT = grab("SYSTEM_PROMPT");
const CONVERT_USER = grab("CONVERT_USER");
const SYSTEM_PROMPT_EXTRA = systemExtraFile ? readFileSync(path.resolve(systemExtraFile), "utf8") : "";
const SYSTEM_PROMPT_WITH_EXTRA = SYSTEM_PROMPT_EXTRA ? `${SYSTEM_PROMPT}\n\n${SYSTEM_PROMPT_EXTRA}` : SYSTEM_PROMPT;
const FEWSHOT_PAIRS = fewshotFile ? JSON.parse(readFileSync(path.resolve(fewshotFile), "utf8")) : [];
if (!Array.isArray(FEWSHOT_PAIRS)) throw new Error(`--fewshot must be a JSON array: ${fewshotFile}`);
for (const [idx, pair] of FEWSHOT_PAIRS.entries()) {
  if (!pair || typeof pair.user !== "string" || typeof pair.assistant !== "string") {
    throw new Error(`--fewshot item ${idx} must be {"user": "...", "assistant": "..."} in ${fewshotFile}`);
  }
}
const FEWSHOT_MESSAGES = FEWSHOT_PAIRS.flatMap((pair) => ([
  { role: "user", content: CONVERT_USER(pair.user) },
  { role: "assistant", content: pair.assistant },
]));
const stripThink = (t) => t.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
const stripFences = (t) => { const m = t.match(/^```(?:latex|tex)?\s*\n([\s\S]*?)\n?```\s*$/); return m ? m[1].trim() : t.trim(); };
const maxTokensFor = (userContent) => Math.min(1024, Math.max(256, Math.ceil(userContent.length / 3) * 2 + 128));

// validator.js expects a global `katex`
const katexMod = await import(path.join(root, "public/vendor/katex/katex.mjs"));
globalThis.katex = katexMod.default ?? katexMod;
const { validateLatex } = await import(path.join(root, "public/validator.js"));
let normalize = null;
let looksLikePdfPaste = null;
if (useNormalize || usePdfGate) {
  const pdfPipeline = await import(path.join(root, "public/pdf-pipeline.js"));
  normalize = pdfPipeline.normalize;
  if (usePdfGate) looksLikePdfPaste = pdfPipeline.looksLikePdfPaste;
}

const items = JSON.parse(readFileSync(dataFile, "utf8"))
  .filter((i) => !onlyItems.length || onlyItems.includes(i.id));

async function chat(model, messages, numPredict, think) {
  const body = {
    model, stream: false, keep_alive: "10m",
    options: { temperature: 0, seed: 0, num_predict: numPredict, num_ctx: 4096 },
    messages,
  };
  if (think !== undefined) body.think = think;
  const r = await fetch(`${OLLAMA}/api/chat`, { method: "POST", body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

// Some models reject `think:false` ("does not support thinking"); fall back per model.
async function generate(model, text, thinkMode, { systemPrompt = SYSTEM_PROMPT_WITH_EXTRA, fewshotMessages = FEWSHOT_MESSAGES } = {}) {
  const messages = [
    { role: "system", content: systemPrompt },
    ...fewshotMessages,
    { role: "user", content: CONVERT_USER(text) + (noThinkSuffix ? `\n${noThinkSuffix}` : "") },
  ];
  const numPredict = maxTokensFor(CONVERT_USER(text));
  if (thinkMode.value !== "unsupported") {
    try { return await chat(model, messages, numPredict, false); }
    catch (e) { if (!/think/i.test(String(e))) throw e; thinkMode.value = "unsupported"; console.error(`  ${model}: think:false rejected, retrying without`); }
  }
  return chat(model, messages, numPredict, undefined);
}

const rows = existsSync(out) ? JSON.parse(readFileSync(out, "utf8")) : [];
for (const model of models) {
  const approach = runLabel ? `${runLabel}:${model}` : `${useNormalize ? "norm" : "direct"}:${model}`;
  const modelName = runLabel ? `${model}+${runLabel}` : (useNormalize ? `${model}+norm` : model);
  const thinkMode = { value: "try" };
  console.error(`== ${model}`);
  try { await chat(model, [{ role: "user", content: "hi" }], 2, undefined); } // load + warm
  catch (e) { console.error(`  LOAD FAILED: ${e.message}`); rows.push({ approach, model: modelName, item: "__load__", tier: "meta", ms: -1, normMs: 0, output: "", error: String(e.message).slice(0, 200) }); continue; }
  const done = new Set(rows.filter((r) => r.approach === approach && r.ms >= 0).map((r) => r.item));
  for (const it of items) {
    if (done.has(it.id)) continue; // resume: the out file already has this (model, item)
    const gated = usePdfGate && looksLikePdfPaste(it.input);
    const useNormThis = usePdfGate ? (gated && useNormalize) : useNormalize;
    const systemPrompt = usePdfGate && !gated ? SYSTEM_PROMPT : SYSTEM_PROMPT_WITH_EXTRA;
    const fewshotMessages = usePdfGate && !gated ? [] : FEWSHOT_MESSAGES;
    let inputText = it.input;
    let normMs = 0;
    if (useNormThis) {
      const tNorm = performance.now();
      inputText = normalize(it.input)[0];
      normMs = Math.round(performance.now() - tNorm);
    }
    const t0 = performance.now();
    try {
      const j = await generate(model, inputText, thinkMode, { systemPrompt, fewshotMessages });
      const raw = j.message?.content ?? "";
      const latex = stripFences(stripThink(raw));
      const ms = Math.round(performance.now() - t0);
      const v = validateLatex(it.input, latex);
      rows.push({ approach, model: modelName, item: it.id, tier: it.tier, ms, normMs, gated, output: latex, valid: v.ok, issues: v.issues,
        thinking: j.message?.thinking ? String(j.message.thinking).length : 0,
        evalTokens: j.eval_count, promptTokens: j.prompt_eval_count, evalMs: Math.round((j.eval_duration ?? 0) / 1e6) });
      console.error(`  ${it.id.padEnd(28)} ${String(ms).padStart(6)}ms  ${v.ok ? "valid  " : "INVALID"} ${v.ok ? "" : v.issues.join(" | ").slice(0, 90)}`);
    } catch (e) {
      rows.push({ approach, model: modelName, item: it.id, tier: it.tier, ms: -1, normMs, gated, output: "", error: String(e.message).slice(0, 200) });
      console.error(`  ${it.id.padEnd(28)} FAILED ${e.message}`);
    }
    writeFileSync(out, JSON.stringify(rows, null, 1));
  }
}
writeFileSync(out, JSON.stringify(rows, null, 1));
console.error(`wrote ${rows.length} rows to ${out}`);
