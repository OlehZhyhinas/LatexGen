// Headless conversion pipeline: the escalation ladder with no DOM. Both the
// UI handlers and the Tab API call this, so behaviour is identical and API
// jobs never touch the user's screen. Progress is reported through optional
// callbacks; ONNX models run in a worker (models.js); the WebLLM engine is
// guarded by a lock because it cannot generate concurrently.
import { validateLatex, checkSyntax } from "./validator.js";
import { loadModel, runModel, loaded } from "./models.js";
import { run as runPdfPipeline, looksLikePdfPaste, SPAN_SYSTEM_PROMPT } from "./pdf-pipeline.js";

// Kept terse: every token is prefilled on every on-device call.
const SYSTEM_PROMPT = `Convert the text to LaTeX. Never solve, evaluate, or answer — if the text is a question, transcribe the question. Output only LaTeX, no commentary or fences. Pure math: \\[ ... \\]. Prose with math: keep the prose, wrap math in \\( ... \\). Standard amsmath only.`;
const REPAIR_PROMPT = `Your LaTeX failed checks. Fix every listed issue, stay faithful to the original text, output only the corrected LaTeX. "syntax:" = does not parse; "missing:" = something from the text was dropped. "answered:" = you solved the question; transcribe the question, do not evaluate.`;
const CONVERT_USER = (text) => `Convert to LaTeX. Do not solve or answer.\n${text}`;
const REFINE_PROMPT = `The user gives feedback on your previous LaTeX. The feedback is about the LaTeX, never content to transcribe. Output only the corrected LaTeX; change only what the feedback concerns and keep the delimiters. If vague, fix your best guess.`;
const JUDGE_PROMPT = `You verify text-to-LaTeX conversions. Given the user's plain-English input and the produced LaTeX, decide whether the LaTeX expresses exactly what the text describes (same operations, grouping, exponents, limits, variables). Solving or answering instead of transcribing is not ok. Reply with ONLY a JSON object: {"ok": true/false, "reason": "<max 12 words>"}`;

export const MAX_REPAIR_ATTEMPTS = 5;
const REPAIR_BUDGET_MS = 8000;
export const IMAGE_MODELS = { texo: { name: "Texo", sizeMB: 77 }, texify: { name: "Texify", sizeMB: 305 } };
const PDF_PIPE = (() => {
  try {
    if (typeof location !== "undefined" && new URLSearchParams(location.search).get("pdfpipe") === "1") return true;
    return typeof localStorage !== "undefined" && localStorage.getItem("latexgen:pdfpipe") === "1";
  } catch {
    return false;
  }
})();

const stripThink = (t) => t.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
const stripFences = (t) => { const m = t.match(/^```(?:latex|tex)?\s*\n([\s\S]*?)\n?```\s*$/); return m ? m[1].trim() : t.trim(); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Trained on single short equations — gate on capability, not content.
export function specialistEligible(text) {
  const t = text.trim();
  return t.length > 0 && t.length <= 320 && !t.includes("\n");
}

// Output is complete when it ends on a closing delimiter and parses.
function looksComplete(text) {
  const t = text.trim();
  if (t.length < 4) return false;
  if (!/(\\\]|\\\))\s*$/.test(t) && !(/\$\$\s*$/.test(t) && (t.match(/\$\$/g) || []).length % 2 === 0)) return false;
  return checkSyntax(t).length === 0;
}

// ---- Texo post-processing ----
const KATEX_ALIASES = {
  infin: "infty", rarr: "rightarrow", larr: "leftarrow", lrarr: "leftrightarrow",
  Rarr: "Rightarrow", Larr: "Leftarrow", Lrarr: "Leftrightarrow", plusmn: "pm",
  empty: "emptyset", isin: "in", sub: "subset", sube: "subseteq", supe: "supseteq",
  sdot: "cdot", lang: "langle", rang: "rangle", real: "Re", image: "Im",
  alef: "aleph", thetasym: "vartheta",
};
export function canonicalizeTexo(s) {
  let out = s.replace(/\\([A-Za-z]+)/g, (m, cmd) => (cmd in KATEX_ALIASES ? `\\${KATEX_ALIASES[cmd]}` : m));
  out = out.replace(/\\(mathrm|operatorname\*?)\s*\{([^{}]*)\}/g, (m, cmd, body) =>
    `\\${cmd}{${body.split("~").map((w) => w.replace(/\s+/g, "")).join(" ")}}`);
  out = out.replace(/\s+/g, " ").replace(/ ([\^_{}()\[\],;])/g, "$1").replace(/([\^_{(\[]) /g, "$1").replace(/~/g, "\\,");
  return out.trim();
}
// Texo has no text mode: prose comes back spelled out inside \mathrm.
function texoProseSignal(raw) {
  const segs = raw.match(/\\(?:mathrm|operatorname\*?)\s*\{[^{}]*\}/g) || [];
  const words = segs.flatMap((seg) => seg.replace(/\\(?:mathrm|operatorname\*?)\s*\{|\}/g, "").split("~").map((w) => w.replace(/\s+/g, "")).filter((w) => w.length >= 3));
  return words.length >= 3;
}
// Texify sometimes repeats a line to the token cap on sparse images.
function dedupeRepeats(s) {
  const parts = [...new Set(s.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean))];
  while (parts.length > 1 && parts.slice(0, -1).some((earlier) => earlier.includes(parts[parts.length - 1]))) parts.pop();
  return parts.join("\n\n");
}

// Text outside math delimiters. A refinement must not introduce prose where
// the original had none (a chat answer is not an edit), nor grow the prose
// noticeably where it had some.
function outsideMath(t) {
  return (t.replace(/\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$[^$\n]*?\$/g, " ").match(/[A-Za-z]{2,}/g) || []);
}
function looksLikeProse(out, prev) {
  const before = outsideMath(prev).length, after = outsideMath(out).length;
  if (before === 0) return after > 0;
  return after > before + 3;
}
// A degenerate revision: the model echoed the feedback instead of editing.
export function isEcho(instruction, revised) {
  const norm = (s) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
  const a = norm(instruction), b = norm(revised);
  return !b || a === b || (b.length > 8 && (a.includes(b) || b.includes(a)));
}

// Read the server's NDJSON stream: {delta} lines, then {done, latex, model, ms}.
export async function streamServerChat(url, body, onDelta) {
  const r = await fetch(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) { const data = await r.json().catch(() => ({})); throw new Error(data.error || `server error ${r.status}`); }
  const reader = r.body.getReader(); const decoder = new TextDecoder();
  let buf = "", full = "", final = null;
  for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    buf += decoder.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i); buf = buf.slice(i + 1);
      if (!line.trim()) continue;
      const j = JSON.parse(line);
      if (j.done) final = j; else if (j.delta) { full += j.delta; onDelta?.(full); }
    }
  }
  if (!final) throw new Error("stream ended unexpectedly");
  return final;
}

// Output formats (single-segment outputs can be re-wrapped; others as-is).
function mathBody(latex) {
  const t = latex.trim();
  const m = t.match(/^(?:\$\$([\s\S]+)\$\$|\\\[([\s\S]+)\\\]|\\\(([\s\S]+)\\\)|\$([^$]+)\$)$/);
  if (m) return (m[1] ?? m[2] ?? m[3] ?? m[4]).trim();
  if (!/\$|\\\[|\\\(/.test(t)) return t;
  return null;
}
export function toFormat(latex, fmt) {
  const body = mathBody(latex);
  if (fmt === "latex" || !fmt) return latex;
  if (fmt === "display") return body != null ? `\\[\n${body}\n\\]` : latex;
  if (fmt === "inline") return body != null ? `\\(${body}\\)` : latex;
  if (fmt === "mathml") {
    try { return globalThis.katex.renderToString(body ?? latex, { output: "mathml", throwOnError: false, displayMode: true }).match(/<math[\s\S]*<\/math>/)?.[0] ?? null; } catch { return null; }
  }
  return null;
}

export function createPipeline(ctx) {
  // ctx: { getEngine, getEngineName, serverAvailable, loadedModelMultiline, browserIsSlow }
  let engineLock = Promise.resolve();
  const withEngine = (fn) => { const run = engineLock.catch(() => {}).then(fn); engineLock = run; return run; };

  async function streamBrowserChat(messages, onDelta) {
    const engine = ctx.getEngine();
    if (!engine) throw new Error("Load the browser model first.");
    const inputChars = messages[messages.length - 1].content.length;
    const maxTokens = Math.min(1024, Math.max(256, Math.ceil(inputChars / 3) * 2 + 128));
    return withEngine(async () => {
      const chunks = await engine.chat.completions.create({ messages, temperature: 0.2, max_tokens: maxTokens, stream: true, extra_body: { enable_thinking: false } });
      let full = "", sinceCheck = 0;
      for await (const c of chunks) {
        const delta = c.choices[0]?.delta?.content ?? "";
        if (!delta) continue;
        full += delta; onDelta?.(full);
        if (++sinceCheck >= 4) { sinceCheck = 0; if (looksComplete(full)) engine.interruptGenerate(); }
      }
      return { latex: stripFences(stripThink(full)), model: ctx.getEngineName() };
    });
  }
  const convertBrowser = (text, onDelta) => streamBrowserChat([{ role: "system", content: SYSTEM_PROMPT }, { role: "user", content: CONVERT_USER(text) }], onDelta);
  const repairBrowser = (text, badLatex, issues, onDelta) => streamBrowserChat([
    { role: "system", content: REPAIR_PROMPT },
    { role: "user", content: CONVERT_USER(text) },
    { role: "assistant", content: badLatex },
    { role: "user", content: `Checks failed:\n- ${issues.join("\n- ")}\nOutput the corrected LaTeX.` },
  ], onDelta);
  const convertServer = (text, onDelta) => streamServerChat("api/convert", { text, complex: !specialistEligible(text) }, onDelta);

  // Validator-guided repair: up to 5 turns while the error keeps changing;
  // one attempt when a server can take over or the device decodes slowly.
  async function repairLoop(contextText, latex, issues, validateFn, onDelta, onAttempt, opts = {}) {
    let prevSig = issues.join("|"), current = latex;
    const started = performance.now();
    const maxAttempts = opts.maxAttempts ?? ((ctx.serverAvailable() || ctx.browserIsSlow()) ? 1 : MAX_REPAIR_ATTEMPTS);
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      if (attempt > 1 && performance.now() - started > REPAIR_BUDGET_MS) return { ok: false, latex: current, attempts: attempt - 1, issues, budget: true };
      onAttempt?.(attempt, issues, maxAttempts);
      const repaired = await repairBrowser(contextText, current, issues, onDelta);
      const v = validateFn(repaired.latex);
      if (v.ok) return { ok: true, latex: repaired.latex, attempts: attempt };
      const sig = v.issues.join("|");
      if (sig === prevSig) return { ok: false, latex: repaired.latex, attempts: attempt, issues: v.issues, stuck: true };
      prevSig = sig; current = repaired.latex; issues = v.issues;
    }
    return { ok: false, latex: current, attempts: maxAttempts, issues };
  }

  // ---- mesh tier: another user's tab runs the job (text only), routed by the relay ----
  function meshUsable() { return !!(ctx.mesh && ctx.mesh.enabled() && ctx.mesh.tabId()); }
  async function runOnMesh(kind, body, onStatus) {
    onStatus?.("asking another LatexGen tab…");
    const r = await fetch(`api/pool/${ctx.mesh.tabId()}/${kind}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    const data = await r.json().catch(() => ({}));
    if (!r.ok || data.error || !data.latex) throw new Error(data.error || `mesh ${r.status}`);
    return { latex: data.latex, model: `peer · ${data.peerModel ?? "on-device model"}`, peer: data.peer };
  }
  // Ladder order after your own device: a self-hosted local server is more
  // trusted and faster than a stranger's tab, so it goes first; on a hosted
  // deployment (cloud API, costs, logs) peers go first.
  function afterLocalOrder() {
    const server = ctx.serverAvailable(), mesh = meshUsable();
    const kind = ctx.mesh?.serverKind?.() ?? "cloud";
    const order = [];
    if (server && kind === "local") order.push("server");
    if (mesh) order.push("mesh");
    if (server && kind !== "local") order.push("server");
    return order;
  }

  async function ensureSpecialist() { try { await loadModel("intellitex"); return true; } catch { return false; } }
  async function convertSpecialist(text) {
    return { latex: await runModel("intellitex", text), model: "IntelliTeX · specialist" };
  }

  async function judge(text, latex) {
    const parse = (s) => { try { const j = JSON.parse(s.slice(s.indexOf("{"), s.lastIndexOf("}") + 1)); return { ok: !!j.ok, reason: String(j.reason || "") }; } catch { return { ok: true, reason: "" }; } };
    try {
      if (ctx.serverAvailable()) {
        const r = await fetch("api/judge", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text, latex }) });
        if (r.ok) return parse(await r.text());
      }
      const engine = ctx.getEngine();
      if (engine) {
        return withEngine(async () => parse((await engine.chat.completions.create({
          messages: [{ role: "system", content: JUDGE_PROMPT }, { role: "user", content: `INPUT:\n${text}\n\nLATEX:\n${latex}` }],
          temperature: 0, max_tokens: 80, extra_body: { enable_thinking: false },
        })).choices[0].message.content));
      }
    } catch { /* judge unavailable */ }
    return { ok: true, reason: "", unavailable: true };
  }

  // ---- text -> LaTeX ----
  async function convertText(text, opts = {}) {
    const { engineChoice = "browser", onStatus = () => {}, onDelta = () => {}, onDraft = () => {}, strict = false, allowMesh = true, allowServer = true } = opts;
    const started = performance.now();
    const engine = ctx.getEngine();
    const server = allowServer && ctx.serverAvailable();
    // Escalation targets after the on-device attempt, in order.
    const tiers = afterLocalOrder().filter((t) => (t === "server" ? server : allowMesh));
    let peer = null;
    const escalate = async (why) => {
      for (const tier of tiers) {
        try {
          if (tier === "server") { onStatus(`${why} — using server model…`); const r = await convertServer(text, onDelta); return { r, tier }; }
          const r = await runOnMesh("convert", { text }, onStatus); peer = r.peer; return { r, tier };
        } catch (e) { onStatus(`${tier} unavailable (${String(e.message || e).slice(0, 40)})…`); }
      }
      return null;
    };
    const haveSpecialist = await ensureSpecialist();
    let result = null, validation = null, escalated = false, note = "", batch = false;

    // Batch: several lines, each a single equation -> one specialist call per line.
    const lines = text.split(/\n+/).map((l) => l.trim()).filter(Boolean);
    if (haveSpecialist && lines.length >= 2 && lines.every(specialistEligible)) {
      const outs = []; let allOk = true;
      for (const [i, line] of lines.entries()) {
        onStatus(`converting line ${i + 1}/${lines.length}…`);
        const r = await convertSpecialist(line);
        if (!validateLatex(line, r.latex).ok) { allOk = false; break; }
        outs.push(r.latex); onDelta(outs.join("\n\n"));
      }
      if (allOk) { result = { latex: outs.join("\n\n"), model: `IntelliTeX · specialist × ${lines.length}` }; validation = { ok: true, issues: [] }; note = `(${lines.length} equations, one per line)`; batch = true; }
    }

    // Tier 0: specialist.
    if (!result && haveSpecialist && specialistEligible(text)) {
      try {
        onStatus("converting…");
        const r0 = await convertSpecialist(text);
        const v0 = validateLatex(text, r0.latex);
        if (v0.ok) { result = r0; validation = v0; }
        else onDraft(r0.latex, v0);
      } catch { /* fall through */ }
    }

    const complexInput = !specialistEligible(text);
    const tierNote = (tier, why) => tier === "mesh" ? `(${why} — answered by another LatexGen user's device)` : `(${why} — routed to server)`;
    if (!result) {
      if (engineChoice === "browser" && complexInput && !ctx.loadedModelMultiline() && tiers.length) {
        const e = await escalate("long or mixed input");
        if (!e) throw new Error("No model available for this input.");
        result = e.r; validation = validateLatex(text, result.latex); note = tierNote(e.tier, "loaded on-device model is too small for prose passages");
      } else if (engineChoice === "browser" && !engine && tiers.length) {
        const e = await escalate("no on-device model loaded");
        if (!e) throw new Error("No model available: load an on-device model in settings.");
        result = e.r; validation = validateLatex(text, result.latex); note = tierNote(e.tier, "no on-device language model loaded");
      } else if (engineChoice === "browser") {
        if (!engine) throw new Error("No model available: load an on-device model in settings.");
        if (PDF_PIPE && looksLikePdfPaste(text)) {
          let spanMs = 0;
          const rec = await runPdfPipeline(text, async (spanTexts) => {
            const t0 = performance.now();
            const out = [];
            for (const spanText of spanTexts) {
              const r = await streamBrowserChat([{ role: "system", content: SPAN_SYSTEM_PROMPT }, { role: "user", content: spanText }], onDelta);
              out.push(r.latex);
            }
            spanMs = Math.round(performance.now() - t0);
            return out;
          });
          onStatus(`pdf: normalized (${Math.max(0, text.length - rec.clean.length)} chars removed)`);
          onStatus(`pdf: ${rec.spans.length} spans`);
          onStatus(`pdf: spans converted in ${spanMs} ms`);
          result = { latex: rec.output, model: `${ctx.getEngineName()} (pdf spans × ${rec.spans.length})` };
          validation = validateLatex(text, result.latex);
        } else {
          onStatus("converting with on-device model…");
          result = await convertBrowser(text, onDelta); validation = validateLatex(text, result.latex);
        }
        if (!validation.ok) {
          try {
            const r = await repairLoop(text, result.latex, validation.issues, (l) => validateLatex(text, l), onDelta,
              (attempt, issues, max) => onStatus(`repair attempt ${attempt}${server ? "" : `/${max}`} — ${issues[0]}…`, { issues }));
            result = { ...result, latex: r.latex };
            if (r.ok) { result.model += ` (self-corrected ×${r.attempts})`; validation = { ok: true, issues: [] }; note = `(self-corrected after ${r.attempts} attempt${r.attempts > 1 ? "s" : ""})`; }
            else validation = { ok: false, issues: r.issues };
          } catch { /* fall through to escalation */ }
        }
        if (!validation.ok && tiers.length) {
          const e = await escalate(`browser model failed checks (${validation.issues[0]}…)`);
          if (e) { escalated = true; result = e.r; validation = validateLatex(text, result.latex); note = tierNote(e.tier, "escalated — on-device output failed checks"); }
        }
      } else {
        if (!server) throw new Error("Server model is not reachable.");
        onStatus("converting with server model…");
        result = await convertServer(text, onDelta); validation = validateLatex(text, result.latex);
      }
    }
    const out = { latex: result.latex, validation, model: result.model, note, escalated, batch, peer, ms: Math.round(performance.now() - started) };
    if (strict && validation.ok && !batch) out.judge = await judge(text, result.latex);
    return out;
  }

  // ---- image -> LaTeX ----
  async function convertImage(blob, opts = {}) {
    const { ocr = "auto", onStatus = () => {}, onDelta = () => {}, onProgress } = opts;
    const started = performance.now();
    const syntaxOnly = (l) => validateLatex("(image)", l).issues.filter((i) => i.startsWith("syntax"));
    let used = ocr === "texify" ? "texify" : "texo", latex = "", escalatedWhy = "";
    if (used === "texo") {
      onStatus("loading image model (Texo)…"); await loadModel("texo", onProgress);
      onStatus("reading equation from image (Texo)…");
      const raw = await runModel("texo", blob);
      latex = canonicalizeTexo(raw);
      if (ocr === "auto") {
        if (texoProseSignal(raw)) escalatedWhy = "image contains prose";
        else if (!latex || syntaxOnly(latex).length) escalatedWhy = "output failed syntax checks";
        if (escalatedWhy) used = "texify";
      }
    }
    if (used === "texify") {
      onStatus(escalatedWhy ? `${escalatedWhy} — switching to Texify…` : "loading image model (Texify)…");
      await loadModel("texify", onProgress);
      onStatus("reading image (Texify)…");
      latex = dedupeRepeats(await runModel("texify", blob));
    }
    if (!latex) throw new Error("no text recognized in image");
    onDelta(latex);
    let issues = syntaxOnly(latex), repaired = 0;
    let note = `(from image via ${IMAGE_MODELS[used].name}${escalatedWhy ? `, escalated: ${escalatedWhy}` : ""} — processed locally, image never uploaded)`;
    if (issues.length && ctx.getEngine()) {
      try {
        const r = await repairLoop("(transcribed from an image of rendered math)", latex, issues, (l) => ({ ok: syntaxOnly(l).length === 0, issues: syntaxOnly(l) }), onDelta,
          (attempt, iss, max) => onStatus(`OCR repair attempt ${attempt}/${max} — ${iss[0]}…`, { issues: iss }));
        latex = r.latex; issues = r.ok ? [] : r.issues; repaired = r.ok ? r.attempts : 0;
        if (r.ok) note = `(from image via ${IMAGE_MODELS[used].name} — OCR self-corrected ×${r.attempts} locally, image never uploaded)`;
      } catch { /* keep OCR output */ }
    }
    return { latex, validation: { ok: issues.length === 0, issues }, model: `${IMAGE_MODELS[used].name} · local OCR`, used, escalatedWhy, note, repaired, ms: Math.round(performance.now() - started) };
  }

  // ---- check existing LaTeX ----
  async function checkLatex(latex, opts = {}) {
    const { onStatus = () => {}, onDelta = () => {} } = opts;
    const started = performance.now();
    let issues = checkSyntax(latex), out = latex, attempts = 0;
    let note = "(checked — nothing converted, just verified)";
    if (issues.length && ctx.getEngine()) {
      const syntaxOnly = (l) => ({ ok: checkSyntax(l).length === 0, issues: checkSyntax(l) });
      const r = await repairLoop("(user-supplied LaTeX; keep its mathematical meaning)", latex, issues, syntaxOnly, onDelta,
        (attempt, iss, max) => onStatus(`repair attempt ${attempt}/${max} — ${iss[0]}…`, { issues: iss }));
      out = r.latex; issues = r.ok ? [] : r.issues; attempts = r.attempts;
      if (r.ok) note = `(fixed after ${r.attempts} repair attempt${r.attempts > 1 ? "s" : ""} — compare with what you pasted)`;
    }
    return { latex: out, validation: { ok: issues.length === 0, issues }, note, repaired: attempts, model: "validator", ms: Math.round(performance.now() - started) };
  }

  // ---- refine with an instruction ----
  async function refine({ original, latex, instruction, engineChoice = "browser", onDelta = () => {}, allowMesh = true, allowServer = true }) {
    const started = performance.now();
    const engine = ctx.getEngine(), server = allowServer && ctx.serverAvailable();
    const useBrowser = engineChoice === "browser" && !!engine;
    const acceptable = (out) => out && !isEcho(instruction, out) && checkSyntax(out).length === 0 && !looksLikeProse(out, latex);
    if (!useBrowser && !server && allowMesh && meshUsable()) {
      const r = await runOnMesh("refine", { original, latex, instruction });
      const ok = acceptable(r.latex); const fl = ok ? r.latex : latex;
      return { latex: fl, validation: validateLatex(original, fl), model: r.model, echo: !ok, retried: false, peer: r.peer, ms: Math.round(performance.now() - started) };
    }
    if (!useBrowser && !server) throw new Error("No model available to refine with: load an on-device model in settings.");
    const viaBrowser = () => streamBrowserChat([
      { role: "system", content: REFINE_PROMPT }, { role: "user", content: `Convert to LaTeX:\n${original}` },
      { role: "assistant", content: latex }, { role: "user", content: instruction },
    ], onDelta);
    const viaServer = () => streamServerChat("api/refine", { original, latex, instruction }, onDelta);
    let result = useBrowser ? await viaBrowser() : await viaServer();
    let retried = false;
    if (!acceptable(result.latex) && useBrowser && server) { retried = true; result = await viaServer(); }
    const echo = !acceptable(result.latex);
    const finalLatex = echo ? latex : result.latex;
    return { latex: finalLatex, validation: validateLatex(original, finalLatex), model: result.model ?? "refine", echo, retried, ms: Math.round(performance.now() - started) };
  }

  return { convertText, convertImage, checkLatex, refine, judge, repairLoop, toFormat, specialistEligible, ensureSpecialist, loaded };
}
