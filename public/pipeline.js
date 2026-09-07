// Headless conversion pipeline: the escalation ladder with no DOM. Both the
// UI handlers and the Tab API call this, so behaviour is identical and API
// jobs never touch the user's screen. Progress is reported through optional
// callbacks; ONNX models run in a worker (models.js); the WebLLM engine is
// guarded by a lock because it cannot generate concurrently.
import { validateLatex, checkSyntax } from "./validator.js";
import { loadModel, runModel, loaded } from "./models.js";

// Kept terse: every token is prefilled on every on-device call.
const SYSTEM_PROMPT = `Convert the text to LaTeX. Never solve, evaluate, or answer — if the text is a question, transcribe the question. Output only LaTeX, no commentary or fences. Pure math: \\[ ... \\]. Prose with math: keep the prose, wrap math in \\( ... \\). Standard amsmath only.`;
const REPAIR_PROMPT = `Your LaTeX failed checks. Fix every listed issue, stay faithful to the original text, output only the corrected LaTeX. "syntax:" = does not parse; "missing:" = something from the text was dropped. "answered:" = you solved the question; transcribe the question, do not evaluate.`;
const CONVERT_USER = (text) => `Convert to LaTeX. Do not solve or answer.\n${text}`;
const REFINE_PROMPT = `The user gives feedback on your previous LaTeX. The feedback is about the LaTeX, never content to transcribe. Output only the corrected LaTeX; change only what the feedback concerns and keep the delimiters. If vague, fix your best guess.`;
const JUDGE_PROMPT = `You verify text-to-LaTeX conversions. Given the user's plain-English input and the produced LaTeX, decide whether the LaTeX expresses exactly what the text describes (same operations, grouping, exponents, limits, variables). Solving or answering instead of transcribing is not ok. Reply with ONLY a JSON object: {"ok": true/false, "reason": "<max 12 words>"}`;

export const MAX_REPAIR_ATTEMPTS = 5;
const REPAIR_BUDGET_MS = 8000;
export const IMAGE_MODELS = { texo: { name: "Texo", sizeMB: 77 }, texify: { name: "Texify", sizeMB: 305 } };

const stripThink = (t) => t.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
const stripFences = (t) => { const m = t.match(/^```(?:latex|tex)?\s*\n([\s\S]*?)\n?```\s*$/); return m ? m[1].trim() : t.trim(); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Optional structured log for the UI. Tab API jobs pass nothing and stay headless.
let streamSeq = 0;
function emit(onEvent, kind, title, extra = {}) {
  onEvent?.({ kind, title, ...extra });
}
export function runStats(ms, tokens, decodeMs) {
  if (ms == null || !Number.isFinite(ms) || ms < 0) return {};
  const out = { ms: Math.round(ms) };
  const n = Number(tokens);
  if (Number.isFinite(n) && n > 0) {
    out.tokens = n;
    const denom = n > 1 && Number.isFinite(decodeMs) && decodeMs > 0 ? decodeMs : ms;
    if (denom > 0) out.tokPerSec = Math.round((n / (denom / 1000)) * 10) / 10;
  }
  return out;
}
function fmtDur(ms) {
  if (ms == null || !Number.isFinite(ms)) return "";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(ms >= 10_000 ? 0 : 1)}s`;
}
function tapDelta(onDelta, onEvent, title) {
  const live = `stream-${++streamSeq}`;
  const t0 = performance.now();
  return (full, extra = {}) => {
    onDelta?.(full);
    const stats = extra.tokPerSec != null
      ? { ms: extra.ms ?? Math.round(performance.now() - t0), tokens: extra.tokens, tokPerSec: extra.tokPerSec }
      : runStats(extra.ms ?? (performance.now() - t0), extra.tokens, extra.decodeMs);
    emit(onEvent, "stream", title, {
      live, raw: full,
      detail: extra.done && stats.tokPerSec != null
        ? `Wrote ${stats.tokens} token${stats.tokens === 1 ? "" : "s"} in ${fmtDur(stats.ms)} (${stats.tokPerSec} tok/s).`
        : extra.done
          ? `Finished in ${fmtDur(stats.ms)}.`
          : "Output as it is written. This entry grows until the model stops.",
      ...stats,
    });
  };
}
function checkEvent(v) {
  if (v?.ok) return { kind: "check", title: "Checks passed", detail: "The LaTeX parses, and the numbers and names from your text appear in it.", raw: "ok" };
  const issues = v?.issues ?? [];
  return { kind: "check", title: "Checks found problems", detail: "The pipeline will repair or climb to a stronger model.", raw: issues.join("\n") || "failed", bad: true };
}

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
  const t0 = performance.now();
  const r = await fetch(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) { const data = await r.json().catch(() => ({})); throw new Error(data.error || `server error ${r.status}`); }
  const reader = r.body.getReader(); const decoder = new TextDecoder();
  let buf = "", full = "", final = null, pieces = 0;
  for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    buf += decoder.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i); buf = buf.slice(i + 1);
      if (!line.trim()) continue;
      const j = JSON.parse(line);
      if (j.done) final = j;
      else if (j.delta) { full += j.delta; pieces += 1; onDelta?.(full, runStats(performance.now() - t0, pieces)); }
    }
  }
  if (!final) throw new Error("stream ended unexpectedly");
  const ms = final.ms ?? Math.round(performance.now() - t0);
  const tokens = final.tokens ?? pieces;
  const stats = final.tokPerSec != null ? { ms, tokens, tokPerSec: final.tokPerSec } : runStats(ms, tokens);
  onDelta?.(full || final.latex, { ...stats, done: true });
  return { ...final, ...stats };
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

  async function streamBrowserChat(messages, onDelta, onEvent) {
    const engine = ctx.getEngine();
    if (!engine) throw new Error("Load the browser model first.");
    const inputChars = messages[messages.length - 1].content.length;
    const maxTokens = Math.min(1024, Math.max(256, Math.ceil(inputChars / 3) * 2 + 128));
    emit(onEvent, "step", "Running the on-device language model", {
      detail: `${ctx.getEngineName() || "WebLLM"} is writing LaTeX from your text.`,
      raw: messages.map((m) => `${m.role}: ${m.content}`).join("\n\n"),
    });
    const delta = tapDelta(onDelta, onEvent, "On-device model is writing");
    return withEngine(async () => {
      // The catalog runtime fast path needs greedy decode, which also keeps runs reproducible.
      const t0 = performance.now();
      const chunks = await engine.chat.completions.create({
        messages, temperature: 0, max_tokens: maxTokens, stream: true,
        stream_options: { include_usage: true }, extra_body: { enable_thinking: false },
      });
      let full = "", sinceCheck = 0, chunksOut = 0, usageTokens = null, firstTok = null;
      for await (const c of chunks) {
        if (c.usage?.completion_tokens) usageTokens = c.usage.completion_tokens;
        const d = c.choices[0]?.delta?.content ?? "";
        if (!d) continue;
        if (firstTok == null) firstTok = performance.now();
        full += d; chunksOut += 1;
        const tokens = usageTokens ?? chunksOut;
        delta(full, { ms: performance.now() - t0, tokens, decodeMs: firstTok != null ? performance.now() - firstTok : undefined });
        if (++sinceCheck >= 4) { sinceCheck = 0; if (looksComplete(full)) engine.interruptGenerate(); }
      }
      const ms = performance.now() - t0;
      const tokens = usageTokens ?? chunksOut;
      const decodeMs = firstTok != null ? performance.now() - firstTok : ms;
      const stats = runStats(ms, tokens, decodeMs);
      delta(full, { ...stats, decodeMs, done: true });
      const latex = stripFences(stripThink(full));
      emit(onEvent, "model", "On-device model finished", { detail: ctx.getEngineName() || "WebLLM", raw: latex, ...stats });
      return { latex, model: ctx.getEngineName() };
    });
  }
  const convertBrowser = (text, onDelta, onEvent) => streamBrowserChat([{ role: "system", content: SYSTEM_PROMPT }, { role: "user", content: CONVERT_USER(text) }], onDelta, onEvent);
  const repairBrowser = (text, badLatex, issues, onDelta, onEvent) => streamBrowserChat([
    { role: "system", content: REPAIR_PROMPT },
    { role: "user", content: CONVERT_USER(text) },
    { role: "assistant", content: badLatex },
    { role: "user", content: `Checks failed:\n- ${issues.join("\n- ")}\nOutput the corrected LaTeX.` },
  ], onDelta, onEvent);
  const convertServer = async (text, onDelta, onEvent) => {
    emit(onEvent, "step", "Running the server model", { detail: "This conversion left the tab. The text is sent to the configured server model.", raw: text });
    const r = await streamServerChat("api/convert", { text, complex: !specialistEligible(text) }, tapDelta(onDelta, onEvent, "Server model is writing"));
    emit(onEvent, "model", "Server model finished", { detail: r.model, raw: r.latex, ms: r.ms, tokens: r.tokens, tokPerSec: r.tokPerSec });
    return r;
  };

  // Validator-guided repair: up to 5 turns while the error keeps changing;
  // one attempt when a server can take over or the device decodes slowly.
  async function repairLoop(contextText, latex, issues, validateFn, onDelta, onAttempt, opts = {}) {
    let prevSig = issues.join("|"), current = latex;
    const started = performance.now();
    const maxAttempts = opts.maxAttempts ?? ((ctx.serverAvailable() || ctx.browserIsSlow()) ? 1 : MAX_REPAIR_ATTEMPTS);
    const onEvent = opts.onEvent;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      if (attempt > 1 && performance.now() - started > REPAIR_BUDGET_MS) return { ok: false, latex: current, attempts: attempt - 1, issues, budget: true };
      onAttempt?.(attempt, issues, maxAttempts);
      emit(onEvent, "step", `Repair attempt ${attempt} of ${maxAttempts}`, {
        detail: "The last LaTeX failed checks. The same model gets the exact error list and tries again.",
        raw: issues.join("\n"),
      });
      const repaired = await repairBrowser(contextText, current, issues, onDelta, onEvent);
      const v = validateFn(repaired.latex);
      const ev = checkEvent(v);
      emit(onEvent, ev.kind, ev.title, { detail: ev.detail, raw: ev.raw, bad: ev.bad });
      if (v.ok) return { ok: true, latex: repaired.latex, attempts: attempt };
      const sig = v.issues.join("|");
      if (sig === prevSig) return { ok: false, latex: repaired.latex, attempts: attempt, issues: v.issues, stuck: true };
      prevSig = sig; current = repaired.latex; issues = v.issues;
    }
    return { ok: false, latex: current, attempts: maxAttempts, issues };
  }

  // ---- mesh tier: another user's tab runs the job (text only), routed by the relay ----
  function meshUsable() { return !!(ctx.mesh && ctx.mesh.enabled() && ctx.mesh.tabId()); }
  async function runOnMesh(kind, body, onStatus, onEvent) {
    onStatus?.("asking another LatexGen tab…");
    emit(onEvent, "step", "Asking another LatexGen tab", { detail: "A peer device will run this conversion. Text only; images never leave this machine.", raw: JSON.stringify(body).slice(0, 2000) });
    const t0 = performance.now();
    const r = await fetch(`api/pool/${ctx.mesh.tabId()}/${kind}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    const data = await r.json().catch(() => ({}));
    if (!r.ok || data.error || !data.latex) throw new Error(data.error || `mesh ${r.status}`);
    emit(onEvent, "model", "Peer tab returned LaTeX", { detail: data.peerModel ?? "on-device model", raw: data.latex, ...runStats(performance.now() - t0, data.tokens) });
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
  async function convertSpecialist(text, onEvent) {
    emit(onEvent, "step", "Running the specialist", { detail: "IntelliTeX is a small model trained only to turn a single equation into LaTeX. It runs on this device.", raw: text });
    const t0 = performance.now();
    const r = await runModel("intellitex", text);
    const latex = r.output;
    emit(onEvent, "model", "Specialist returned LaTeX", { detail: "IntelliTeX · specialist", raw: latex, ...runStats(r.ms ?? (performance.now() - t0), r.tokens) });
    return { latex, model: "IntelliTeX · specialist" };
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
    const { engineChoice = "browser", onStatus = () => {}, onDelta = () => {}, onDraft = () => {}, onEvent, strict = false, allowMesh = true, allowServer = true } = opts;
    const started = performance.now();
    const engine = ctx.getEngine();
    const server = allowServer && ctx.serverAvailable();
    emit(onEvent, "run", "Starting a conversion", { detail: "Plain English to LaTeX, on this device unless a later step says otherwise.", raw: text });
    // Escalation targets after the on-device attempt, in order.
    const tiers = afterLocalOrder().filter((t) => (t === "server" ? server : allowMesh));
    let peer = null;
    const escalate = async (why) => {
      emit(onEvent, "step", "Climbing to a stronger tier", { detail: why, raw: tiers.join(" → ") || "(none available)" });
      for (const tier of tiers) {
        try {
          if (tier === "server") { onStatus(`${why} — using server model…`); const r = await convertServer(text, onDelta, onEvent); return { r, tier }; }
          const r = await runOnMesh("convert", { text }, onStatus, onEvent); peer = r.peer; return { r, tier };
        } catch (e) {
          const msg = String(e.message || e).slice(0, 40);
          onStatus(`${tier} unavailable (${msg})…`);
          emit(onEvent, "error", `${tier} unavailable`, { detail: "Trying the next tier if there is one.", raw: String(e.message || e) });
        }
      }
      return null;
    };
    if (!loaded.intellitex) emit(onEvent, "step", "Loading the specialist", { detail: "First use downloads IntelliTeX (about 200 MB) and may compile it for this machine. Later conversions skip this.", live: "load-intellitex" });
    const loadStarted = performance.now();
    const wasLoaded = loaded.intellitex;
    const haveSpecialist = await ensureSpecialist();
    if (!wasLoaded && haveSpecialist) emit(onEvent, "step", "Specialist is ready", { detail: "IntelliTeX is in memory for this visit.", live: "load-intellitex", ...runStats(performance.now() - loadStarted) });
    if (!haveSpecialist) emit(onEvent, "step", "Specialist is not available", { detail: "Continuing with the on-device language model or the server, if either is ready." });
    let result = null, validation = null, escalated = false, note = "", batch = false;

    // Batch: several lines, each a single equation -> one specialist call per line.
    const lines = text.split(/\n+/).map((l) => l.trim()).filter(Boolean);
    if (haveSpecialist && lines.length >= 2 && lines.every(specialistEligible)) {
      const outs = []; let allOk = true;
      for (const [i, line] of lines.entries()) {
        onStatus(`converting line ${i + 1}/${lines.length}…`);
        emit(onEvent, "step", `Converting line ${i + 1} of ${lines.length}`, { detail: "Each line is a separate equation for the specialist.", raw: line });
        const r = await convertSpecialist(line, onEvent);
        const v = validateLatex(line, r.latex);
        emit(onEvent, checkEvent(v).kind, checkEvent(v).title, { detail: checkEvent(v).detail, raw: checkEvent(v).raw, bad: checkEvent(v).bad });
        if (!v.ok) { allOk = false; break; }
        outs.push(r.latex); onDelta(outs.join("\n\n"));
      }
      if (allOk) { result = { latex: outs.join("\n\n"), model: `IntelliTeX · specialist × ${lines.length}` }; validation = { ok: true, issues: [] }; note = `(${lines.length} equations, one per line)`; batch = true; }
    }

    // Tier 0: specialist.
    if (!result && haveSpecialist && specialistEligible(text)) {
      try {
        onStatus("converting…");
        const r0 = await convertSpecialist(text, onEvent);
        const v0 = validateLatex(text, r0.latex);
        const ev = checkEvent(v0);
        emit(onEvent, ev.kind, ev.title, { detail: ev.detail, raw: ev.raw, bad: ev.bad });
        if (v0.ok) { result = r0; validation = v0; }
        else {
          emit(onEvent, "step", "Specialist draft did not pass", { detail: "Showing the draft and climbing the ladder.", raw: r0.latex });
          onDraft(r0.latex, v0);
        }
      } catch (e) {
        emit(onEvent, "error", "Specialist failed", { detail: "Climbing to the next model.", raw: String(e.message || e) });
      }
    } else if (!result && !specialistEligible(text)) {
      emit(onEvent, "step", "Skipping the specialist", { detail: "This input is longer than a single equation, or has multiple lines, so a larger model is next." });
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
        onStatus("converting with on-device model…");
        result = await convertBrowser(text, onDelta, onEvent); validation = validateLatex(text, result.latex);
        emit(onEvent, checkEvent(validation).kind, checkEvent(validation).title, { detail: checkEvent(validation).detail, raw: checkEvent(validation).raw, bad: checkEvent(validation).bad });
        if (!validation.ok) {
          try {
            const r = await repairLoop(text, result.latex, validation.issues, (l) => validateLatex(text, l), onDelta,
              (attempt, issues, max) => onStatus(`repair attempt ${attempt}${server ? "" : `/${max}`} — ${issues[0]}…`, { issues }), { onEvent });
            result = { ...result, latex: r.latex };
            if (r.ok) { result.model += ` (self-corrected ×${r.attempts})`; validation = { ok: true, issues: [] }; note = `(self-corrected after ${r.attempts} attempt${r.attempts > 1 ? "s" : ""})`; }
            else validation = { ok: false, issues: r.issues };
          } catch (e) { emit(onEvent, "error", "Repair failed", { raw: String(e.message || e) }); }
        }
        if (!validation.ok && tiers.length) {
          const e = await escalate(`browser model failed checks (${validation.issues[0]}…)`);
          if (e) { escalated = true; result = e.r; validation = validateLatex(text, result.latex); note = tierNote(e.tier, "escalated — on-device output failed checks"); }
        }
      } else {
        if (!server) throw new Error("Server model is not reachable.");
        onStatus("converting with server model…");
        result = await convertServer(text, onDelta, onEvent); validation = validateLatex(text, result.latex);
        emit(onEvent, checkEvent(validation).kind, checkEvent(validation).title, { detail: checkEvent(validation).detail, raw: checkEvent(validation).raw, bad: checkEvent(validation).bad });
      }
    }
    const out = { latex: result.latex, validation, model: result.model, note, escalated, batch, peer, ms: Math.round(performance.now() - started) };
    if (strict && validation.ok && !batch) {
      emit(onEvent, "step", "Asking a second model to verify", { detail: "Strict mode: another model judges whether the LaTeX says what you typed." });
      const tJ = performance.now();
      out.judge = await judge(text, result.latex);
      emit(onEvent, "check", out.judge.unavailable ? "Second opinion unavailable" : (out.judge.ok ? "Second model agrees" : "Second model disagrees"), { detail: out.judge.reason || "", raw: JSON.stringify(out.judge), bad: out.judge.ok ? undefined : true, ...runStats(performance.now() - tJ) });
    }
    emit(onEvent, "done", `Finished in ${(out.ms / 1000).toFixed(1)}s`, { detail: `${out.model}${out.escalated ? " (escalated)" : ""}`, raw: out.latex, ms: out.ms });
    return out;
  }

  // ---- image -> LaTeX ----
  async function convertImage(blob, opts = {}) {
    const { ocr = "auto", onStatus = () => {}, onDelta = () => {}, onProgress, onEvent } = opts;
    const started = performance.now();
    emit(onEvent, "run", "Reading an image", { detail: "OCR runs on this device. The image is never uploaded.", raw: `ocr=${ocr}` });
    const syntaxOnly = (l) => validateLatex("(image)", l).issues.filter((i) => i.startsWith("syntax"));
    let used = ocr === "texify" ? "texify" : "texo", latex = "", escalatedWhy = "";
    if (used === "texo") {
      onStatus("loading image model (Texo)…");
      emit(onEvent, "step", "Loading Texo", { detail: "A small OCR model for a single equation. Downloaded once, then cached.", live: "load-texo" });
      const tLoad = performance.now();
      const texoWasLoaded = loaded.texo;
      await loadModel("texo", onProgress);
      if (!texoWasLoaded) emit(onEvent, "step", "Texo is ready", { detail: "Downloaded once, then cached.", live: "load-texo", ...runStats(performance.now() - tLoad) });
      onStatus("reading equation from image (Texo)…");
      emit(onEvent, "step", "Reading the image with Texo", { detail: "Texo looks at the pixels and writes LaTeX." });
      const tRun = performance.now();
      const texoRun = await runModel("texo", blob);
      const raw = texoRun.output;
      emit(onEvent, "model", "Texo returned LaTeX", { detail: "Raw OCR, before cleanup.", raw, ...runStats(texoRun.ms ?? (performance.now() - tRun), texoRun.tokens) });
      latex = canonicalizeTexo(raw);
      if (latex !== raw) emit(onEvent, "step", "Cleaned Texo output", { detail: "Aliases and spacing normalized to what KaTeX accepts.", raw: latex });
      if (ocr === "auto") {
        if (texoProseSignal(raw)) escalatedWhy = "image contains prose";
        else if (!latex || syntaxOnly(latex).length) escalatedWhy = "output failed syntax checks";
        if (escalatedWhy) used = "texify";
      }
    }
    if (used === "texify") {
      onStatus(escalatedWhy ? `${escalatedWhy} — switching to Texify…` : "loading image model (Texify)…");
      if (escalatedWhy) emit(onEvent, "step", "Switching to Texify", { detail: escalatedWhy === "image contains prose" ? "Texo has no prose mode, so a larger OCR model takes over." : "Texo's LaTeX did not parse, so a larger OCR model takes over." });
      emit(onEvent, "step", "Loading Texify", { detail: "A larger OCR model. Downloaded once, then cached.", live: "load-texify" });
      const tLoad = performance.now();
      const texifyWasLoaded = loaded.texify;
      await loadModel("texify", onProgress);
      if (!texifyWasLoaded) emit(onEvent, "step", "Texify is ready", { detail: "Downloaded once, then cached.", live: "load-texify", ...runStats(performance.now() - tLoad) });
      onStatus("reading image (Texify)…");
      emit(onEvent, "step", "Reading the image with Texify", { detail: "Texify handles denser equations and prose around math." });
      const tRun = performance.now();
      const texifyRun = await runModel("texify", blob);
      latex = dedupeRepeats(texifyRun.output);
      emit(onEvent, "model", "Texify returned LaTeX", { raw: latex, ...runStats(texifyRun.ms ?? (performance.now() - tRun), texifyRun.tokens) });
    }
    if (!latex) throw new Error("no text recognized in image");
    onDelta(latex);
    let issues = syntaxOnly(latex), repaired = 0;
    let note = `(from image via ${IMAGE_MODELS[used].name}${escalatedWhy ? `, escalated: ${escalatedWhy}` : ""} — processed locally, image never uploaded)`;
    emit(onEvent, issues.length ? "check" : "check", issues.length ? "OCR LaTeX failed syntax checks" : "OCR LaTeX parsed", { detail: issues.length ? "The on-device language model will try to repair it." : "No syntax errors.", raw: issues.join("\n") || "ok", bad: issues.length ? true : undefined });
    if (issues.length && ctx.getEngine()) {
      try {
        const r = await repairLoop("(transcribed from an image of rendered math)", latex, issues, (l) => ({ ok: syntaxOnly(l).length === 0, issues: syntaxOnly(l) }), onDelta,
          (attempt, iss, max) => onStatus(`OCR repair attempt ${attempt}/${max} — ${iss[0]}…`, { issues: iss }), { onEvent });
        latex = r.latex; issues = r.ok ? [] : r.issues; repaired = r.ok ? r.attempts : 0;
        if (r.ok) note = `(from image via ${IMAGE_MODELS[used].name} — OCR self-corrected ×${r.attempts} locally, image never uploaded)`;
      } catch (e) { emit(onEvent, "error", "OCR repair failed", { raw: String(e.message || e) }); }
    }
    const out = { latex, validation: { ok: issues.length === 0, issues }, model: `${IMAGE_MODELS[used].name} · local OCR`, used, escalatedWhy, note, repaired, ms: Math.round(performance.now() - started) };
    emit(onEvent, "done", `Finished in ${(out.ms / 1000).toFixed(1)}s`, { detail: out.model, raw: out.latex, ms: out.ms });
    return out;
  }

  // ---- check existing LaTeX ----
  async function checkLatex(latex, opts = {}) {
    const { onStatus = () => {}, onDelta = () => {}, onEvent } = opts;
    const started = performance.now();
    emit(onEvent, "run", "Checking pasted LaTeX", { detail: "Nothing is converted. The text is parsed, and repaired only if it does not compile.", raw: latex });
    let issues = checkSyntax(latex), out = latex, attempts = 0;
    let note = "(checked — nothing converted, just verified)";
    emit(onEvent, issues.length ? "check" : "check", issues.length ? "LaTeX did not parse" : "LaTeX parsed", { raw: issues.join("\n") || "ok", bad: issues.length ? true : undefined });
    if (issues.length && ctx.getEngine()) {
      const syntaxOnly = (l) => ({ ok: checkSyntax(l).length === 0, issues: checkSyntax(l) });
      const r = await repairLoop("(user-supplied LaTeX; keep its mathematical meaning)", latex, issues, syntaxOnly, onDelta,
        (attempt, iss, max) => onStatus(`repair attempt ${attempt}/${max} — ${iss[0]}…`, { issues: iss }), { onEvent });
      out = r.latex; issues = r.ok ? [] : r.issues; attempts = r.attempts;
      if (r.ok) note = `(fixed after ${r.attempts} repair attempt${r.attempts > 1 ? "s" : ""} — compare with what you pasted)`;
    }
    const result = { latex: out, validation: { ok: issues.length === 0, issues }, note, repaired: attempts, model: "validator", ms: Math.round(performance.now() - started) };
    emit(onEvent, "done", result.validation.ok ? "Valid LaTeX" : "Issues remain", { detail: note, raw: result.latex, ms: result.ms });
    return result;
  }

  // ---- refine with an instruction ----
  async function refine({ original, latex, instruction, engineChoice = "browser", onDelta = () => {}, onEvent, allowMesh = true, allowServer = true }) {
    const started = performance.now();
    emit(onEvent, "run", "Applying a refinement", { detail: "The model edits the current LaTeX from your instruction. It must still parse.", raw: instruction });
    const engine = ctx.getEngine(), server = allowServer && ctx.serverAvailable();
    const useBrowser = engineChoice === "browser" && !!engine;
    const acceptable = (out) => out && !isEcho(instruction, out) && checkSyntax(out).length === 0 && !looksLikeProse(out, latex);
    if (!useBrowser && !server && allowMesh && meshUsable()) {
      const r = await runOnMesh("refine", { original, latex, instruction }, undefined, onEvent);
      const ok = acceptable(r.latex); const fl = ok ? r.latex : latex;
      emit(onEvent, ok ? "done" : "error", ok ? "Peer refined the LaTeX" : "Peer echo ignored", { raw: r.latex, ms: Math.round(performance.now() - started) });
      return { latex: fl, validation: validateLatex(original, fl), model: r.model, echo: !ok, retried: false, peer: r.peer, ms: Math.round(performance.now() - started) };
    }
    if (!useBrowser && !server) throw new Error("No model available to refine with: load an on-device model in settings.");
    const viaBrowser = () => streamBrowserChat([
      { role: "system", content: REFINE_PROMPT }, { role: "user", content: `Convert to LaTeX:\n${original}` },
      { role: "assistant", content: latex }, { role: "user", content: instruction },
    ], onDelta, onEvent);
    const viaServer = async () => {
      emit(onEvent, "step", "Refining with the server model", { raw: instruction });
      const r = await streamServerChat("api/refine", { original, latex, instruction }, tapDelta(onDelta, onEvent, "Server model is writing"));
      emit(onEvent, "model", "Server model finished", { detail: r.model, raw: r.latex, ms: r.ms, tokens: r.tokens, tokPerSec: r.tokPerSec });
      return r;
    };
    let result = useBrowser ? await viaBrowser() : await viaServer();
    let retried = false;
    if (!acceptable(result.latex) && useBrowser && server) {
      emit(onEvent, "step", "Retrying the refinement on the server", { detail: "The on-device edit did not look like LaTeX, so the server model gets a turn." });
      retried = true; result = await viaServer();
    }
    const echo = !acceptable(result.latex);
    const finalLatex = echo ? latex : result.latex;
    emit(onEvent, echo ? "error" : "done", echo ? "Kept the previous LaTeX" : "Refinement applied", { detail: echo ? "The model echoed the instruction instead of editing." : result.model, raw: result.latex, ms: Math.round(performance.now() - started) });
    return { latex: finalLatex, validation: validateLatex(original, finalLatex), model: result.model ?? "refine", echo, retried, ms: Math.round(performance.now() - started) };
  }

  return { convertText, convertImage, checkLatex, refine, judge, repairLoop, toFormat, specialistEligible, ensureSpecialist, loaded };
}
