#!/usr/bin/env node
// Text benchmark through Ollama: every model gets the pipeline's exact
// prompts, greedy decoding, thinking off, and the same validator the app uses
// to decide escalation. Prompts and post-processing are read out of
// public/pipeline.js so this harness cannot drift from what ships.
//
//   node bench/run-ollama-text.mjs --models qwen3:4b,qwen3.5:4b --out bench/results-x.json [--data bench/bench-data-extended.json]
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
const label = (m) => opt("--label", "") || m;
// Qwen3 (not 3.5) under Ollama sometimes ignores `think:false` and reasons in
// the visible content; the family's documented soft switch stops it. The app's
// WebLLM path disables thinking through the chat template instead.
const noThinkSuffix = opt("--no-think-suffix", "") ; // e.g. "/no_think"
if (!models.length) { console.error("usage: --models a,b,c [--out file] [--items id,id]"); process.exit(2); }

// ---- prompts and helpers lifted from pipeline.js ----
const src = readFileSync(path.join(root, "public/pipeline.js"), "utf8");
const grab = (name) => {
  const m = src.match(new RegExp(`const ${name} = (\`[^\`]*\`|\\(text\\) => \`[^\`]*\`);`));
  if (!m) throw new Error(`could not find ${name} in pipeline.js`);
  return new Function(`return ${m[1]};`)();
};
const SYSTEM_PROMPT = grab("SYSTEM_PROMPT");
const CONVERT_USER = grab("CONVERT_USER");
const stripThink = (t) => t.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
const stripFences = (t) => { const m = t.match(/^```(?:latex|tex)?\s*\n([\s\S]*?)\n?```\s*$/); return m ? m[1].trim() : t.trim(); };
const maxTokensFor = (userContent) => Math.min(1024, Math.max(256, Math.ceil(userContent.length / 3) * 2 + 128));

// validator.js expects a global `katex`
const katexMod = await import(path.join(root, "public/vendor/katex/katex.mjs"));
globalThis.katex = katexMod.default ?? katexMod;
const { validateLatex } = await import(path.join(root, "public/validator.js"));

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
async function generate(model, text, thinkMode) {
  const messages = [{ role: "system", content: SYSTEM_PROMPT }, { role: "user", content: CONVERT_USER(text) + (noThinkSuffix ? `\n${noThinkSuffix}` : "") }];
  const numPredict = maxTokensFor(CONVERT_USER(text));
  if (thinkMode.value !== "unsupported") {
    try { return await chat(model, messages, numPredict, false); }
    catch (e) { if (!/think/i.test(String(e))) throw e; thinkMode.value = "unsupported"; console.error(`  ${model}: think:false rejected, retrying without`); }
  }
  return chat(model, messages, numPredict, undefined);
}

const rows = existsSync(out) ? JSON.parse(readFileSync(out, "utf8")) : [];
for (const model of models) {
  const approach = `direct:${label(model)}`;
  const thinkMode = { value: "try" };
  console.error(`== ${model}`);
  try { await chat(model, [{ role: "user", content: "hi" }], 2, undefined); } // load + warm
  catch (e) { console.error(`  LOAD FAILED: ${e.message}`); rows.push({ approach, model, item: "__load__", tier: "meta", ms: -1, output: "", error: String(e.message).slice(0, 200) }); continue; }
  const done = new Set(rows.filter((r) => r.approach === approach && r.ms >= 0).map((r) => r.item));
  for (const it of items) {
    if (done.has(it.id)) continue; // resume: the out file already has this (model, item)
    const t0 = performance.now();
    try {
      const j = await generate(model, it.input, thinkMode);
      const raw = j.message?.content ?? "";
      const latex = stripFences(stripThink(raw));
      const ms = Math.round(performance.now() - t0);
      const v = validateLatex(it.input, latex);
      rows.push({ approach, model, item: it.id, tier: it.tier, ms, output: latex, valid: v.ok, issues: v.issues,
        thinking: j.message?.thinking ? String(j.message.thinking).length : 0,
        evalTokens: j.eval_count, promptTokens: j.prompt_eval_count, evalMs: Math.round((j.eval_duration ?? 0) / 1e6) });
      console.error(`  ${it.id.padEnd(28)} ${String(ms).padStart(6)}ms  ${v.ok ? "valid  " : "INVALID"} ${v.ok ? "" : v.issues.join(" | ").slice(0, 90)}`);
    } catch (e) {
      rows.push({ approach, model, item: it.id, tier: it.tier, ms: -1, output: "", error: String(e.message).slice(0, 200) });
      console.error(`  ${it.id.padEnd(28)} FAILED ${e.message}`);
    }
    writeFileSync(out, JSON.stringify(rows, null, 1));
  }
}
writeFileSync(out, JSON.stringify(rows, null, 1));
console.error(`wrote ${rows.length} rows to ${out}`);
