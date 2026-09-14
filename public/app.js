// LatexGen UI. All conversion logic lives in pipeline.js (headless); this
// file wires DOM events to it and renders results. The Tab API at the bottom
// calls the same pipeline without touching the UI.
import * as webllm from "./vendor/webllm/index.js";
import { validateLatex, checkSyntax } from "./validator.js";
import { loadModel, onProgress as onModelProgress, runtimeUsed } from "./models.js";
import { createPipeline, streamServerChat, toFormat, IMAGE_MODELS, MAX_REPAIR_ATTEMPTS, specialistEligible } from "./pipeline.js";
import { isTransportError, loadCatalog, parseForce, planEngine, probeTarget, rememberRuntime } from "./qwen3-webllm.js";
import { prefetchModel } from "./webllm-store.js";
import { STATIC_BUILD } from "./config.js";

window.__validate = validateLatex; // debugging hook
window.__runtimeUsed = runtimeUsed; // debugging hook

const $ = (id) => document.getElementById(id);
const ddBtn = $("dd-btn");
const ddMenu = $("dd-menu");
const loadBtn = $("load-btn");
const loadStatus = $("load-status");
const progressFill = $("load-progress-fill");
const convertBtn = $("convert-btn");
const convertStatus = $("convert-status");
const outputCode = $("output-code");
const preview = $("preview");
const checksEl = $("checks");
const chatLog = $("chat-log");
const chatInput = $("chat-input");
const chatSend = $("chat-send");

let engine = null;
let loadedModel = null;
let currentLatex = null;
let currentInput = null;
let serverAvailable = false;
let browserTokPerSec = null;
let selectedModelId = null;
let selectedKey = null;   // rows are (model, decode path), so the id alone is ambiguous
let loadedKey = null;     // set from the plan that actually loaded, not the one picked
let modelRows = [];
let renderMenu = () => {}; // assigned by buildModelPicker; activate() re-renders to move the tick
let loadSeq = 0; // bumped by loadDefaultModel/loadPicked so a superseded load can bail instead of clobbering the one that won

// ---- settings persisted in localStorage ----
const PREFS_KEY = "latexgen.prefs";
function prefs() { try { return JSON.parse(localStorage.getItem(PREFS_KEY) || "{}"); } catch { return {}; } }
function setPref(k, v) { const p = prefs(); p[k] = v; try { localStorage.setItem(PREFS_KEY, JSON.stringify(p)); } catch {} }
function llmEnabled() { return prefs().consent === "full"; }
function strictMode() { return !!prefs().strict; }
function engineChoice() { return document.querySelector('input[name="engine"]:checked')?.value ?? "browser"; }

function toast(msg, ms = 1800) {
  const t = $("toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(t._timer);
  t._timer = setTimeout(() => { t.hidden = true; }, ms);
}

const KIND_LABEL = { run: "run", step: "step", model: "model", check: "check", stream: "out", error: "error", done: "done" };
function formatDuration(ms) {
  if (ms == null || !Number.isFinite(ms) || ms < 1) return "";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms >= 10_000 ? 0 : 1)} s`;
}
function formatStats(ms, tokPerSec) {
  const parts = [];
  const d = formatDuration(ms);
  if (d) parts.push(d);
  if (tokPerSec != null && Number.isFinite(tokPerSec) && tokPerSec >= 0.1) {
    parts.push(`${tokPerSec >= 10 ? Math.round(tokPerSec) : tokPerSec.toFixed(1)} tok/s`);
  }
  return parts.join(" · ");
}
function logEvent({ kind = "step", title, detail, raw, live, bad, ms, tokens, tokPerSec } = {}) {
  const box = $("activity-log");
  if (!box) return;
  box.querySelector(".activity-empty")?.remove();
  const stick = box.scrollHeight - box.scrollTop - box.clientHeight < 56;
  let el = live ? box.querySelector(`[data-live="${CSS.escape(String(live))}"]`) : null;
  if (!el) {
    el = document.createElement("article");
    el.className = `act act-${kind}`;
    el._t0 = performance.now();
    if (live) el.dataset.live = String(live);
    const meta = document.createElement("div"); meta.className = "act-meta";
    const time = document.createElement("span"); time.className = "act-time";
    time.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    const k = document.createElement("span"); k.className = "act-kind";
    const stats = document.createElement("span"); stats.className = "act-stats";
    meta.append(time, k, stats);
    const h = document.createElement("div"); h.className = "act-title";
    const d = document.createElement("div"); d.className = "act-detail";
    const pre = document.createElement("pre"); pre.className = "act-raw";
    el.append(meta, h, d, pre);
    box.appendChild(el);
  }
  el.className = `act act-${kind}${bad ? " is-bad" : ""}`;
  el.querySelector(".act-kind").textContent = KIND_LABEL[kind] ?? kind;
  el.querySelector(".act-title").textContent = title || "";
  const shownMs = ms != null ? ms : (live && el._t0 != null ? performance.now() - el._t0 : null);
  const statsEl = el.querySelector(".act-stats");
  statsEl.textContent = formatStats(shownMs, tokPerSec);
  statsEl.hidden = !statsEl.textContent;
  if (tokens != null && Number.isFinite(tokens)) statsEl.title = `${tokens} token${tokens === 1 ? "" : "s"}`;
  const det = el.querySelector(".act-detail");
  det.textContent = detail || "";
  det.hidden = !detail;
  const pre = el.querySelector(".act-raw");
  if (raw == null || raw === "") { pre.hidden = true; pre.textContent = ""; }
  else { pre.hidden = false; pre.textContent = String(raw); }
  const idle = $("activity-idle");
  if (idle) idle.textContent = (kind === "done" || kind === "error") ? "Idle" : "Working…";
  if (stick) box.scrollTop = box.scrollHeight;
}
window.__logEvent = logEvent;

// ---- pipeline ----
const SLOW_DEVICE_TOK_S = 25;
function browserIsSlow() { return browserTokPerSec != null && browserTokPerSec < SLOW_DEVICE_TOK_S; }
window.__setTokPerSec = (v) => { browserTokPerSec = v; }; // debugging hook
window.__setServerAvailable = (v) => { serverAvailable = v; }; // debugging hook
window.__setServerKind = (k) => { serverKind = k; }; // debugging hook
let serverKind = "cloud";
const meshHooks = { enabled: () => false, tabId: () => null, serverKind: () => serverKind }; // wired by the Tab API block
const pipe = createPipeline({
  getEngine: () => engine,
  getEngineName: () => modelRows.find((r) => r.id === loadedModel)?.name ?? loadedModel,
  serverAvailable: () => serverAvailable,
  loadedModelMultiline: () => !!modelRows.find((r) => r.id === loadedModel)?.multiline,
  browserIsSlow,
  mesh: meshHooks,
});
window.__repairLoop = pipe.repairLoop; // debugging hook
const specialistReady = pipe.ensureSpecialist(); // start the ~190MB specialist download immediately
{
  const mb = (n) => `${(Number(n) / 2 ** 20).toFixed(1)} MB`;
  const watch = (key, title, detail) => onModelProgress(key, (p) => {
    const file = p.file ?? key;
    const raw = p.total ? `${file}\n${mb(p.loaded ?? 0)} / ${mb(p.total)}` : (p.text ?? file);
    logEvent({ kind: "step", title, detail, raw, live: `load-${key}` });
  });
  watch("intellitex", "Loading the specialist", "IntelliTeX weights, and a one-time compile if WebNN is on.");
  watch("texo", "Loading Texo", "Small equation OCR. Downloaded once, then cached.");
  watch("texify", "Loading Texify", "Larger OCR model. Downloaded once, then cached.");
}

// ---- rendering helpers ----
function renderPreview(latex) {
  const hasDelims = /\\\[|\\\(|\$\$|(?:^|[^\\])\$/.test(latex);
  preview.textContent = hasDelims ? latex : `\\[${latex}\\]`;
  renderMathInElement(preview, {
    delimiters: [
      { left: "$$", right: "$$", display: true }, { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false }, { left: "$", right: "$", display: false },
    ],
    throwOnError: false,
  });
}
// Validation results render as status chips; a note becomes an info chip.
function showChecks(validation, note) {
  checksEl.innerHTML = "";
  const chip = (cls, label) => {
    const s = document.createElement("span"); s.className = cls;
    s.appendChild(document.createElement("i")); s.appendChild(document.createTextNode(label));
    checksEl.appendChild(s);
  };
  if (validation.ok) {
    chip("ok", "Syntax valid");
    if (!String(currentInput).startsWith("(") && !String(note).includes("image")) chip("ok", "Matches your input");
  } else {
    for (const issue of validation.issues.slice(0, 4)) chip("warn", issue);
    if (validation.issues.length > 4) chip("warn", `+${validation.issues.length - 4} more`);
  }
  if (note) chip("info", note.replace(/^\(|\)$/g, ""));
}
function setChatEnabled(on) { chatInput.disabled = !on; chatSend.disabled = !on; }
function addMsg(cls, text) {
  const el = document.createElement("div"); el.className = `msg ${cls}`; el.textContent = text;
  chatLog.appendChild(el); el.scrollIntoView({ block: "nearest" }); return el;
}
// Present a pipeline result in the UI.
function presentResult(input, r, statusText) {
  outputCode.textContent = r.latex;
  renderPreview(r.latex);
  currentLatex = r.latex; currentInput = input;
  showChecks(r.validation, r.note);
  recordHistory(input, r.latex);
  chatLog.innerHTML = ""; setChatEnabled(true);
  convertStatus.textContent = statusText;
}
const onStatusUI = (s, extra) => { convertStatus.textContent = s; if (extra?.issues) showChecks({ ok: false, issues: extra.issues }, ""); };
const onDeltaUI = (partial) => { outputCode.textContent = partial; };

// ---- text conversion (UI) ----
convertBtn.addEventListener("click", async () => {
  const text = $("input").value.trim();
  if (!text) return;
  if ($("input-card").dataset.mode === "latex") { await checkLatexUI(text); return; }
  convertBtn.disabled = true;
  convertStatus.textContent = "converting…";
  checksEl.textContent = "";
  outputCode.textContent = "";
  try {
    const r = await pipe.convertText(text, {
      engineChoice: engineChoice(), onStatus: onStatusUI, onDelta: onDeltaUI, onEvent: logEvent,
      onDraft: (latex, v) => { outputCode.textContent = latex; renderPreview(latex); showChecks(v, "(draft from the specialist — improving…)"); },
    });
    presentResult(text, r, `${(r.ms / 1000).toFixed(1)}s · ${r.model}${r.escalated ? " (escalated)" : ""}`);
    settleFirstRun(); maybeOfferWebnn(r.ms);
    if (r.peer) window.dispatchEvent(new CustomEvent("latexgen:converted", { detail: { peer: r.peer, summary: text.slice(0, 60), model: r.model, ms: r.ms } }));
    if (strictMode() && r.validation.ok && !r.batch) backgroundJudge(text, r.latex);
  } catch (err) {
    convertStatus.textContent = String(err.message || err);
    logEvent({ kind: "error", title: "Conversion failed", raw: String(err.message || err) });
  } finally {
    convertBtn.disabled = false;
  }
});

// ---- image conversion (UI) ----
let lastImageBlob = null, lastImageModel = null;
function imageProgressUI() {
  const bar = $("image-progress"), fill = $("image-progress-fill"), label = $("image-drop-label");
  const files = new Map();
  return {
    onProgress: (p) => {
      bar.hidden = false;
      files.set(p.file, { loaded: p.loaded, total: p.total });
      let loaded = 0, total = 0; for (const f of files.values()) { loaded += f.loaded; total += f.total; }
      fill.style.width = `${Math.round((loaded / total) * 100)}%`;
      label.textContent = `loading ${IMAGE_MODELS[p.key]?.name ?? "image"} model: ${(loaded / 2 ** 20).toFixed(0)} / ${(total / 2 ** 20).toFixed(0)} MB (one time — cached after this)`;
    },
    done: () => { bar.hidden = true; fill.style.width = "0"; },
  };
}
async function convertImage(fileOrBlob, forceModel = null) {
  const drop = $("image-drop"), label = $("image-drop-label"), altBtn = $("ocr-alt-btn");
  drop.classList.add("busy"); altBtn.hidden = true;
  convertStatus.textContent = ""; lastImageBlob = fileOrBlob;
  const prog = imageProgressUI();
  try {
    const r = await pipe.convertImage(fileOrBlob, {
      ocr: forceModel ?? "auto", onProgress: prog.onProgress, onDelta: onDeltaUI, onEvent: logEvent,
      onStatus: (s, extra) => { label.textContent = s; if (extra?.issues) showChecks({ ok: false, issues: extra.issues }, ""); },
      onDraft: (latex, why) => { currentInput = "(image)"; outputCode.textContent = latex; renderPreview(latex); const issues = checkSyntax(latex); showChecks({ ok: issues.length === 0, issues }, `(first reading from Texo — ${why}, loading Texify for a better one…)`); convertStatus.textContent = "draft from Texo · Texify loading…"; },
    });
    lastImageModel = r.used;
    presentResult("(image)", r, `${(r.ms / 1000).toFixed(1)}s · ${r.model}`);
    settleFirstRun(); maybeOfferWebnn(r.ms);
    const other = r.used === "texo" ? "texify" : "texo";
    altBtn.textContent = `looks wrong? read image with ${IMAGE_MODELS[other].name} instead`;
    altBtn.hidden = false;
  } catch (err) {
    convertStatus.textContent = `image conversion failed: ${err.message || err}`;
    logEvent({ kind: "error", title: "Image conversion failed", raw: String(err.message || err) });
  } finally {
    prog.done(); drop.classList.remove("busy");
    label.innerHTML = "Drop / paste / <u>choose</u> an image of an equation — processed entirely in your browser, never uploaded";
  }
}
window.__convertImage = convertImage; // debugging hook
$("ocr-alt-btn").addEventListener("click", () => { if (lastImageBlob) convertImage(lastImageBlob, lastImageModel === "texo" ? "texify" : "texo"); });
{
  const drop = $("image-drop"), fileInput = $("image-file");
  drop.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => { if (fileInput.files[0]) convertImage(fileInput.files[0]); fileInput.value = ""; });
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("dragover"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("dragover"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("dragover"); const f = e.dataTransfer.files?.[0]; if (f && f.type.startsWith("image/")) convertImage(f); });
  document.addEventListener("paste", (e) => { const item = [...(e.clipboardData?.items ?? [])].find((i) => i.type.startsWith("image/")); if (item) convertImage(item.getAsFile()); });
}

// ---- Check-LaTeX mode (UI) ----
async function checkLatexUI(latex) {
  convertBtn.disabled = true; checksEl.innerHTML = ""; convertStatus.textContent = "checking…";
  outputCode.textContent = latex; renderPreview(latex);
  try {
    const r = await pipe.checkLatex(latex, { onStatus: onStatusUI, onDelta: onDeltaUI, onEvent: logEvent });
    presentResult("(latex check)", r, r.validation.ok ? "valid LaTeX" : "issues found");
  } finally { convertBtn.disabled = false; }
}

// ---- refine chat (UI) ----
async function sendRefinement() {
  const instruction = chatInput.value.trim();
  if (!instruction || !currentLatex) return;
  chatInput.value = ""; setChatEnabled(false);
  addMsg("user", instruction);
  const pending = addMsg("note", "revising…");
  try {
    const r = await pipe.refine({ original: currentInput, latex: currentLatex, instruction, engineChoice: engineChoice(), onDelta: (p) => { pending.textContent = p; }, onEvent: logEvent });
    pending.remove();
    if (r.echo) {
      addMsg("note", 'The model returned your feedback instead of edited LaTeX, so I kept the previous version. Try stating the change directly, e.g. "change TV to T(V)".');
      return;
    }
    currentLatex = r.latex; recordHistory(currentInput, r.latex);
    if (r.peer) window.dispatchEvent(new CustomEvent("latexgen:converted", { detail: { peer: r.peer, summary: `refine: ${instruction.slice(0, 50)}`, model: r.model, ms: r.ms } }));
    addMsg("model", r.latex);
    outputCode.textContent = r.latex; renderPreview(r.latex);
    showChecks(r.validation, false);
  } catch (err) {
    pending.className = "msg note"; pending.textContent = `error: ${err.message || err}`;
  } finally { setChatEnabled(true); chatInput.focus(); }
}
chatSend.addEventListener("click", sendRefinement);
chatInput.addEventListener("keydown", (e) => { if (e.key === "Enter") sendRefinement(); });

// ---- strict mode, non-blocking: second opinion in the background ----
async function backgroundJudge(text, latex) {
  const pending = document.createElement("span"); pending.className = "info"; pending.innerHTML = "<i></i>second opinion…";
  checksEl.appendChild(pending);
  logEvent({ kind: "step", title: "Asking a second model to verify", detail: "Strict mode. This runs after the result is already on screen." });
  const tJ = performance.now();
  const j = await pipe.judge(text, latex);
  if (outputCode.textContent.trim() !== latex.trim()) { pending.remove(); return; } // user moved on
  pending.remove();
  if (j.unavailable) return;
  if (j.ok) { const ok = document.createElement("span"); ok.className = "ok"; ok.innerHTML = "<i></i>verified by a second model"; checksEl.appendChild(ok); logEvent({ kind: "check", title: "Second model agrees", raw: j.reason || "ok", ms: performance.now() - tJ }); return; }
  logEvent({ kind: "check", title: "Second model disagrees", detail: j.reason || "disagrees", raw: j.reason, bad: true, ms: performance.now() - tJ });
  const warn = document.createElement("span"); warn.className = "warn"; warn.innerHTML = `<i></i>second opinion: ${j.reason || "disagrees"}`;
  checksEl.appendChild(warn);
  if (!engine && !serverAvailable) return;
  const fix = document.createElement("button"); fix.className = "btn ghost"; fix.textContent = "Apply suggested fix";
  fix.addEventListener("click", async () => {
    fix.disabled = true; fix.textContent = "fixing…";
    try {
      let fixed = null;
      if (engine) {
        const r = await pipe.repairLoop(text, latex, [`judge: ${j.reason}`], (l) => validateLatex(text, l), onDeltaUI, null, { maxAttempts: 2 });
        if (r.ok) fixed = r.latex;
      }
      if (!fixed && serverAvailable) fixed = (await streamServerChat("api/refine", { original: text, latex, instruction: `A reviewer says: ${j.reason}. Fix exactly that.` }, onDeltaUI)).latex;
      if (fixed) { outputCode.textContent = fixed; renderPreview(fixed); currentLatex = fixed; showChecks(validateLatex(text, fixed), "(corrected after a second opinion)"); recordHistory(text, fixed); }
      else fix.textContent = "could not fix automatically";
    } catch (e) { fix.textContent = `fix failed: ${String(e.message || e).slice(0, 40)}`; }
  });
  checksEl.appendChild(fix);
}

// ---- curated model picker ----
// One tuned model per size class, ranked by the tier-matched text benchmark
// of 2026-09-07 (bench/judged-text-tiers-2026-09-07.json, 105 items: single
// equations, prose passages, PDF pastes, prose-with-math). Qwen 3.5 4B (64%)
// is the default: it fits an 8 GB laptop, and only the 4B class and up score
// on prose passages (`multiline`). MiniCPM5 2B (52%, quantized and compiled
// by this project — see PUBLISHED below) is picked automatically only where
// 4B is over the device's GPU budget. Qwen 3.5 9B is a manual pick, not
// auto-selected by size.
const CURATED = [
  { id: "Qwen3.5-9B-q4f16_1-MLC", name: "Qwen 3.5 \u00b7 9B", score: 5, multiline: true, manual: true },
  { id: "Qwen3.5-4B-q4f16_1-MLC", name: "Qwen 3.5 \u00b7 4B", score: 4, multiline: true },   // 64%
  { id: "MiniCPM5-2B-q4f16_1-MLC", name: "MiniCPM5 \u00b7 2B", score: 3 },                   // 52%
];
const gb = (mb) => `${(mb / 1024).toFixed(1)} GB`;
const stars = (n) => "★".repeat(n) + "☆".repeat(5 - n);
// Models this project quantized and compiled itself because no mlc-ai build
// exists. The Hugging Face weights repo also hosts the WebGPU model lib; WebLLM
// 0.2.84 loads it exactly like a prebuilt entry (tensor-cache.json manifest,
// q4f16_1 shards). See docs/benchmarks.md for why MiniCPM5 2B holds the 2B rung.
const PUBLISHED = [
  {
    model: "https://huggingface.co/ozhyhinas/MiniCPM5-2B-q4f16_1-MLC",
    model_id: "MiniCPM5-2B-q4f16_1-MLC",
    model_lib: "https://huggingface.co/ozhyhinas/MiniCPM5-2B-q4f16_1-MLC/resolve/main/libs/MiniCPM5-2B-q4f16_1-MLC-webgpu.wasm",
    vram_required_MB: 1900,
    low_resource_required: true,
    required_features: ["shader-f16"],
    overrides: { context_window_size: 4096 },
  },
];
const prebuilt = new Map([...webllm.prebuiltAppConfig.model_list, ...PUBLISHED].map((m) => [m.model_id, m]));

async function detectVramBudgetMB() {
  if (!navigator.gpu) return 0;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return 0;
    // Chrome caps deviceMemory at 8, so 8 is also what a 32 GB machine reports;
    // half of that admits the 3.8 GB default on any 8 GB laptop, and a device
    // reporting 4 GB falls to the 2B rung.
    const deviceGB = navigator.deviceMemory ?? 8; // Safari exposes nothing; assume 8GB+
    return deviceGB * 1024 * 0.5;
  } catch { return 0; }
}
const el = (tag, cls, text) => { const n = document.createElement(tag); n.className = cls; n.textContent = text; return n; };
function selectRow(r) {
  selectedKey = r.key; selectedModelId = r.id;
  ddBtn.replaceChildren(
    el("span", "dd-name", r.id),
    el("span", "dd-meta", `${r.sub} · ${stars(r.score)} · ${r.vram == null ? "size unknown" : gb(r.vram)}`),
    el("span", "dd-caret", "\u25be"),
  );
}
async function buildModelPicker() {
  const budget = await detectVramBudgetMB();
  let storageFreeMB = Infinity;
  try { const est = await navigator.storage.estimate(); if (est.quota) storageFreeMB = (est.quota - (est.usage ?? 0)) / (1024 * 1024); } catch {}
  // Two different questions. Browser storage is a hard blocker: without the
  // quota the weights cannot land at all. The GPU budget is a heuristic from
  // `navigator.deviceMemory`, which is coarse and often conservative, so a
  // model above it stays pickable by hand and is only kept out of automatic
  // selection. `canRun` gates auto-selection, `selectable` gates clicks.
  // Whether a model decodes through the graph catalog's tuned runtime is a
  // property of the model *and* this device: the probe wants an Apple GPU
  // exposing subgroups behind a recent Chromium. Everywhere else, and for any
  // model the catalog has no library for, WebLLM's stock path runs instead.
  let catalogModels = {}, catalogWhy = "catalog unreachable";
  try {
    const catalog = await loadCatalog();
    const probe = await probeTarget(catalog);
    catalogWhy = probe.why;
    if (probe.ok) catalogModels = catalog?.models ?? {};
  } catch (err) { /* no catalog reachable: every model runs the stock path */ }

  // One row per curated model: the catalog's tuned runtime when this device's
  // target probe passes and the catalog carries the model, WebLLM's stock path
  // otherwise. `?webllm=stock` and `?webllm=catalog:<variant>` remain the
  // benchmarking overrides used to force a path by hand. A model with no
  // WebLLM record cannot be loaded — it has no weights entry to point the
  // runtime at — but dropping it from the menu turned that into an invisible
  // gap, so it lists as disabled with the reason instead.
  modelRows = [];
  for (const c of CURATED) {
    const record = prebuilt.get(c.id);
    const vram = record?.vram_required_MB ?? null;
    const fitsGpu = vram != null && vram <= budget;
    const fitsStorage = vram != null && vram * 1.2 <= storageFreeMB;
    const fit = {
      vram, fitsGpu, canRun: fitsGpu && fitsStorage, selectable: vram != null && fitsStorage,
      why: vram == null ? "no WebLLM record" : !fitsStorage ? "not enough browser storage" : !fitsGpu ? "over budget" : "",
    };
    if (!record) {
      // No stock record means no catalog row either: a catalog plan still needs
      // the weights entry to graft its model_lib onto.
      modelRows.push({ ...c, ...fit, kind: "stock", variant: null, sub: "no WebLLM record", key: `${c.id}#stock`, force: "stock", preferred: true, manual: !!c.manual, note: "" });
      continue;
    }
    const model = catalogModels[c.id];
    if (model) {
      const variant = model.default;
      modelRows.push({
        ...c, ...fit, kind: "catalog", variant, sub: "tuned runtime", key: `${c.id}#${variant}`,
        force: { variant }, preferred: true, manual: !!c.manual, note: model.variants?.[variant]?.expected ?? "",
      });
    } else {
      modelRows.push({
        ...c, ...fit, kind: "stock", variant: null, sub: "stock WebLLM", key: `${c.id}#stock`,
        force: "stock", preferred: true, manual: !!c.manual, note: "",
      });
    }
  }
  const auto = modelRows.filter((r) => r.preferred && !r.manual);

  const renderRow = (r) => {
    const row = document.createElement("div");
    row.className = `dd-row${r.selectable ? (r.canRun ? "" : " over-budget") : " disabled"}${r.key === loadedKey ? " on" : ""}`;
    row.setAttribute("role", "option"); row.setAttribute("aria-selected", String(r.key === selectedKey));
    const main = el("span", "dd-main", "");
    main.append(el("span", "dd-name", r.id), el("span", "dd-sub", r.sub));
    row.append(main, el("span", "dd-meta", `${stars(r.score)} · ${r.vram == null ? "size unknown" : gb(r.vram)}${r.canRun ? "" : ` · ${r.why}`}`));
    const tips = [];
    if (r.note) tips.push(r.note);
    if (r.selectable && !r.fitsGpu) tips.push(`${gb(r.vram)} is above this device's estimated ${gb(budget)} budget, so it is not picked automatically. Select it to try anyway; if it fails to load you can pick a smaller one.`);
    if (tips.length) row.title = tips.join("\n");
    if (r.selectable) row.addEventListener("click", () => { selectRow(r); ddMenu.hidden = true; if (r.key !== loadedKey) loadPicked(r); });
    ddMenu.appendChild(row);
  };

  // Flat list: three models, one row each. A note up top stands in for the
  // group header this used to need, only when no row here decodes through the
  // catalog's tuned runtime.
  renderMenu = () => {
    ddMenu.replaceChildren();
    if (!modelRows.some((r) => r.kind === "catalog")) ddMenu.appendChild(el("div", "dd-note", `Stock WebLLM runtime on this device: ${catalogWhy}.`));
    modelRows.forEach(renderRow);
  };
  renderMenu();
  const best = auto.find((r) => r.canRun);
  if (best) {
    selectRow(best);
    if (llmEnabled()) loadDefaultModel(best);
    else {
      loadBtn.hidden = false;
      loadStatus.textContent = "On-device language model is off (specialist only). Turn it on in settings.";
      $("model-status").textContent = "On-device model: specialist only";
    }
  } else {
    // Nothing is inside the estimated budget. Storage permitting, still let the
    // smallest model be chosen by hand instead of dead-ending the picker.
    const manual = [...auto].reverse().find((r) => r.selectable);
    if (manual) {
      selectRow(manual);
      loadBtn.hidden = false;
      loadStatus.textContent = "No model fits this device automatically \u2014 pick one to try it anyway.";
      $("model-status").textContent = "On-device model: specialist only";
    } else {
      // Nothing here can load, but the list stays open so every model is still
      // visible with the reason beside it. Only loading is switched off.
      ddBtn.replaceChildren(el("span", "dd-name", "No browser model fits this device"), el("span", "dd-caret", "\u25be"));
      loadBtn.disabled = true;
    }
  }
}

// Warm-up pays the shader-compile cost and measures decode speed.
async function warmUp(eng) {
  try {
    await eng.chat.completions.create({ messages: [{ role: "user", content: "hi" }], max_tokens: 2, extra_body: { enable_thinking: false } });
    const t0 = performance.now();
    const r = await eng.chat.completions.create({ messages: [{ role: "user", content: "Count from one to forty as words, comma separated." }], max_tokens: 40, temperature: 0, extra_body: { enable_thinking: false } });
    browserTokPerSec = Math.round((r.usage?.completion_tokens ?? 40) / ((performance.now() - t0) / 1000));
  } catch { /* best-effort */ }
}
// Every shipped model has one single-tensor shard that WebLLM's default Cache
// API backend cannot split (Qwen 3.5 4B's params_shard_0.bin is 303 MB, the 9B
// ships two at 485 MB each), and in at least one Chromium build `cache.add` of
// a body over ~250 MB fails outright (`UnknownError: Unexpected internal
// error`), aborting the whole load — see issue #70. New downloads go through
// WebLLM's OPFS backend instead; a model whose shards are already complete
// under the Cache API keeps using it, so existing users never re-download.
async function withCacheBackend(modelId, plan) {
  const base = plan.appConfig ?? webllm.prebuiltAppConfig;
  const backend = (typeof navigator.storage?.getDirectory === "function")
    ? ((await webllm.hasModelInCache(modelId, base).catch(() => false)) ? "cache" : "opfs")
    : "cache";
  return { ...plan, appConfig: { ...base, cacheBackend: backend } };
}
// WebLLM downloads a model's shards as four contiguous quarters of the list,
// one connection each, so the oversized first shard makes one connection the
// long pole while the other three idle out (issue #71). When the weights are
// headed for OPFS, webllm-store.js fills the store first with a balanced 8-way
// pool, largest shard first; WebLLM then finds every key present and skips its
// own fetch. A failure here is only logged: WebLLM's loader fetches whatever
// is still missing.
async function prefetchWeights(modelId, plan, onProgress) {
  if (plan.appConfig?.cacheBackend !== "opfs") return;
  const record = plan.appConfig.model_list?.find((m) => m.model_id === modelId);
  if (!record?.model) return;
  const mb = (b) => (b / 2 ** 20).toFixed(0);
  try {
    const r = await prefetchModel(record, { concurrency: 8, onProgress: (p) => onProgress?.({ progress: p.total ? p.loaded / p.total : 0, text: `Downloading weights: ${mb(p.loaded)} / ${mb(p.total)} MB (${p.done}/${p.count} files)` }) });
    if (r.downloaded) console.info(`webllm prefetch ${modelId}: ${r.downloaded} shards, ${mb(r.bytes)} MB in ${(r.ms / 1000).toFixed(1)} s`);
  } catch (err) { console.warn(`webllm prefetch ${modelId} failed; WebLLM's loader fetches the rest`, err); }
}
async function createEngineForPlan(modelId, plan, onProgress) {
  const worker = new Worker(plan.workerUrl, { type: "module" });
  try {
    const eng = await webllm.CreateWebWorkerMLCEngine(
      worker,
      modelId,
      { initProgressCallback: onProgress, appConfig: plan.appConfig },
      { context_window_size: 2048 }
    );
    eng.latexgenWorker = worker;
    return eng;
  } catch (err) {
    worker.terminate(); // a failed create must not strand its worker
    throw err;
  }
}
// `unload()` frees the GPU buffers, but the worker keeps its WASM heap and the
// runtime bundle alive until it is terminated. Doing only the first leaves a
// live worker behind on every model swap, which is what made a few switches in
// a row bog the whole machine down.
async function releaseEngine(eng) {
  if (!eng) return;
  try { await eng.unload(); } catch { /* the worker goes away regardless */ }
  try { (eng.latexgenWorker ?? eng.worker)?.terminate(); } catch { /* already gone */ }
}
async function loadEngine(row, onProgress, { explicit = false } = {}) {
  const modelId = row.id;
  const stockRecord = prebuilt.get(modelId);
  if (!stockRecord) throw new Error(`Unknown WebLLM model: ${modelId}`);
  let catalog = null;
  try { catalog = await loadCatalog(); } catch { catalog = null; }
  // `?webllm=` stays the debugging override. Below it, a pick from the menu is
  // taken literally — a stock row stays stock, and a catalog row is loaded even
  // if an earlier failure had this model remembered as stock. The default load
  // passes no force at all, so there the remembered fallback and the catalog's
  // own default still decide.
  const force = parseForce(location.search) ?? (explicit ? row.force : undefined);
  let plan = await planEngine(modelId, catalog, { force, stockRecord });
  let eng;
  if (plan.kind === "catalog") {
    try {
      plan = await withCacheBackend(modelId, plan);
      await prefetchWeights(modelId, plan, onProgress);
      eng = await createEngineForPlan(modelId, plan, onProgress);
    } catch (err) {
      if (!isTransportError(err)) rememberRuntime(modelId, "stock", String(err)); // a download or storage failure is not the runtime's fault
      console.warn(`webllm catalog runtime failed for ${modelId}, falling back to stock`, err);
      onProgress?.({ progress: 0, text: "catalog runtime failed — loading stock WebLLM…" });
      plan = await planEngine(modelId, catalog, { force: "stock", stockRecord });
      plan = await withCacheBackend(modelId, plan);
      await prefetchWeights(modelId, plan, onProgress);
      eng = await createEngineForPlan(modelId, plan, onProgress);
    }
  } else {
    // Stock WebLLM only knows its prebuilt list; our published models travel in the app config.
    if (PUBLISHED.some((m) => m.model_id === modelId)) plan = { ...plan, appConfig: { model_list: [stockRecord] } };
    plan = await withCacheBackend(modelId, plan);
    await prefetchWeights(modelId, plan, onProgress);
    eng = await createEngineForPlan(modelId, plan, onProgress);
  }
  eng.latexgenPlan = plan;
  eng.latexgenCacheBackend = plan.appConfig.cacheBackend;
  await warmUp(eng);
  return eng;
}
function activate(eng, modelId, statusText) {
  const old = engine;
  engine = eng; loadedModel = modelId;
  // Which row is live is a property of the plan that loaded, not the one that
  // was clicked: a catalog pick that fails falls back to stock underneath us.
  const plan = eng?.latexgenPlan;
  loadedKey = `${modelId}#${plan?.kind === "catalog" ? plan.variant : "stock"}`;
  renderMenu();
  const planLabel = plan?.label ?? "stock WebLLM";
  loadStatus.textContent = statusText.startsWith("loaded: ") ? `${statusText} (${planLabel})` : statusText;
  const row = modelRows.find((r) => r.id === modelId);
  const speed = browserTokPerSec != null ? ` · ${browserTokPerSec} tok/s` : "";
  $("model-status").textContent = `On-device model: ${row?.name ?? modelId} · ${planLabel}${speed}`;
  if (old && old !== eng) releaseEngine(old);
  window.dispatchEvent(new CustomEvent("latexgen:caps-changed"));
}
// The default model loads directly once picked: the IntelliTeX specialist
// already covers single equations in the meantime.
function logLoadProgress(title, detail, live, p) {
  logEvent({ kind: "step", title, detail, raw: p.text ?? `${Math.round((p.progress ?? 0) * 100)}%`, live });
}
async function loadDefaultModel(best) {
  loadBtn.hidden = true;
  const t0 = performance.now();
  const seq = ++loadSeq; // a manual pick made after this call must win, not whichever finishes last
  try {
    const eng = await loadEngine(best, (p) => { loadStatus.textContent = p.text ?? "loading…"; progressFill.style.width = `${Math.round((p.progress ?? 0) * 100)}%`; logLoadProgress(`Loading ${best.name}`, "On-device language model. Downloaded once, then cached.", `load-webllm-${best.id}`, p); });
    if (seq !== loadSeq) {
      await releaseEngine(eng);
      logEvent({ kind: "step", title: `${best.name} load superseded`, detail: "A different model was picked while it was loading; the download stays cached." });
      return;
    }
    activate(eng, best.id, `loaded: ${best.name}`);
    progressFill.style.width = "100%";
    logEvent({ kind: "done", title: `${best.name} is ready`, detail: `${eng?.latexgenPlan?.label ?? "stock WebLLM"} · ${eng?.latexgenCacheBackend ?? "cache"} cache`, live: `load-webllm-${best.id}`, ms: performance.now() - t0 });
  } catch (err) {
    if (seq !== loadSeq) return; // superseded; the winning load owns loadStatus/loadBtn now
    loadStatus.textContent = `load failed: ${err}`;
    loadBtn.hidden = false;
    logEvent({ kind: "error", title: "On-device language model failed to load", raw: String(err) });
  }
}
async function loadPicked(row) {
  loadBtn.hidden = true;
  const t0 = performance.now();
  const seq = ++loadSeq; // a manual pick made after this call must win, not whichever finishes last
  try {
    // Release the current model before pulling the next one in. Holding both
    // doubles peak memory, which is exactly what fails on a device that only
    // just fits one.
    if (engine) {
      const previous = engine;
      engine = null; loadedModel = null;
      loadStatus.textContent = "unloading the current model\u2026";
      $("model-status").textContent = "On-device model: specialist only";
      window.dispatchEvent(new CustomEvent("latexgen:caps-changed"));
      await releaseEngine(previous);
    }
    const eng = await loadEngine(row, (p) => { loadStatus.textContent = p.text ?? "loading…"; progressFill.style.width = `${Math.round((p.progress ?? 0) * 100)}%`; logLoadProgress(`Loading ${row.name} (${row.sub})`, "On-device language model. Downloaded once, then cached.", `load-webllm-${row.key}`, p); }, { explicit: true });
    if (seq !== loadSeq) {
      await releaseEngine(eng);
      logEvent({ kind: "step", title: `${row.name} load superseded`, detail: "A different model was picked while it was loading; the download stays cached." });
      return;
    }
    activate(eng, row.id, `loaded: ${row.name}`); progressFill.style.width = "100%";
    logEvent({ kind: "done", title: `${row.name} is ready`, detail: `${eng?.latexgenPlan?.label ?? "stock WebLLM"} · ${eng?.latexgenCacheBackend ?? "cache"} cache`, live: `load-webllm-${row.key}`, ms: performance.now() - t0 });
  } catch (err) {
    if (seq !== loadSeq) return; // superseded; the winning load owns loadStatus/loadBtn now
    loadStatus.textContent = `load failed: ${err}`; loadBtn.hidden = false; logEvent({ kind: "error", title: "On-device language model failed to load", raw: String(err) });
  }
}
loadBtn.addEventListener("click", () => { const row = modelRows.find((r) => r.key === selectedKey); if (row) loadPicked(row); });
ddBtn.addEventListener("click", () => { ddMenu.hidden = !ddMenu.hidden; });
document.addEventListener("click", (e) => { if (!$("model-dd").contains(e.target)) ddMenu.hidden = true; });
specialistReady.then((ok) => {
  const rt = runtimeUsed.intellitex;
  logEvent({ kind: ok ? "done" : "error", title: ok ? "Specialist is ready" : "Specialist failed to load", detail: rt ? `${rt.device ?? "cpu"}${rt.dtype ? ` · ${rt.dtype}` : ""}` : "", raw: rt ? JSON.stringify(rt) : "", live: "load-intellitex" });
  return buildModelPicker();
}, () => { logEvent({ kind: "error", title: "Specialist failed to load", live: "load-intellitex" }); return buildModelPicker(); });

// ---- WebGPU availability ----
if (!navigator.gpu) {
  loadBtn.disabled = true; ddBtn.disabled = true;
  if (STATIC_BUILD) {
    loadStatus.textContent = "WebGPU not available in this browser — the specialist still runs on CPU.";
  } else {
    loadStatus.textContent = "WebGPU not available in this browser — use the server model.";
    document.querySelector('input[value="server"]').checked = true;
  }
}

// ---- server health ----
// The static build has no server tier, Tab API or mesh: hide the parts of the
// UI that would only ever report "unavailable".
if (STATIC_BUILD) {
  $("engine-group").hidden = true;
  $("api-btn").hidden = true;
} else fetch("api/health").then((r) => r.json()).then(({ routes, ollama, serverKind: kind }) => {
  serverAvailable = !!routes?.convert;
  serverKind = kind ?? "cloud";
  if (routes?.convert) $("server-hint").textContent = `(${routes.convert.kind} · ${routes.convert.model.split("/").pop()})`;
  else $("server-hint").textContent = ollama?.reachable ? `(Ollama · ${ollama.model})` : "(no backend reachable)";
}).catch(() => {});

// ---- direct manual editing of the LaTeX box ----
let editDebounce = null;
outputCode.addEventListener("input", () => {
  if (convertBtn.disabled) return;
  clearTimeout(editDebounce);
  editDebounce = setTimeout(() => {
    const latex = outputCode.textContent;
    currentLatex = latex; currentInput ??= "(manually entered LaTeX)";
    if (!latex.trim()) { checksEl.textContent = ""; preview.innerHTML = ""; return; }
    renderPreview(latex);
    const issues = checkSyntax(latex);
    showChecks({ ok: issues.length === 0, issues }, "(manually edited)");
    setChatEnabled(true);
  }, 300);
});

// ---- character counter ----
const INPUT_LIMIT = 6000;
$("input").addEventListener("input", () => {
  const len = $("input").value.length;
  $("char-count").textContent = `${len.toLocaleString()} / ${INPUT_LIMIT.toLocaleString()}`;
  $("char-count").classList.toggle("near-limit", len >= INPUT_LIMIT * 0.9);
});
$("input").addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") convertBtn.click(); });

// ---- chrome: segmented control, drawers, upload button ----
{
  const inputCard = $("input-card");
  for (const seg of document.querySelectorAll(".seg")) {
    seg.addEventListener("click", () => {
      document.querySelectorAll(".seg").forEach((s) => s.classList.toggle("active", s === seg));
      inputCard.dataset.mode = seg.dataset.mode;
      $("input").placeholder = seg.dataset.mode === "latex" ? "Paste LaTeX to check, render, and repair" : "e.g. the integral from 0 to infinity of e to the minus x squared dx equals square root of pi over 2";
      if (seg.dataset.mode !== "image") $("input").focus();
    });
  }
  $("upload-btn").addEventListener("click", () => $("image-file").click());
  const drawers = { settings: ["settings-btn", "settings-close"], history: ["history-btn", "history-close"], api: ["api-btn", "api-close"] };
  const openDrawer = (name, open) => {
    for (const n of Object.keys(drawers)) { const el = $(n); const btn = $(drawers[n][0]); const on = open && n === name; el.hidden = !on; btn.setAttribute("aria-expanded", String(on)); }
    if (open && name === "history") renderHistory();
  };
  for (const [name, [btnId, closeId]] of Object.entries(drawers)) {
    $(btnId).addEventListener("click", () => openDrawer(name, $(name).hidden));
    $(closeId).addEventListener("click", () => openDrawer(name, false));
  }
  $("change-model").addEventListener("click", () => { openDrawer("settings", true); ddMenu.hidden = false; });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") openDrawer(null, false); });
  document.addEventListener("click", (e) => {
    for (const [name, [btnId]] of Object.entries(drawers)) {
      const el = $(name);
      if (!el.hidden && !el.contains(e.target) && !$(btnId).contains(e.target) && e.target !== $("change-model")) el.hidden = true;
    }
  });
}

// ---- first-run notice + settings toggles ----
const maybeLoadDefault = () => { if (llmEnabled() && !engine) { const best = modelRows.find((r) => r.preferred && !r.manual && r.canRun); if (best) { loadBtn.hidden = true; loadDefaultModel(best); } } };
// The notice has done its job once a conversion has happened; call this after presentResult.
function settleFirstRun() {
  if (!prefs().consent) setPref("consent", "quick");
  $("first-run").hidden = true;
}
{
  const firstRun = $("first-run");
  if (!prefs().consent) firstRun.hidden = false;
  $("first-run-full").addEventListener("click", () => { setPref("consent", "full"); firstRun.hidden = true; $("llm-enabled").checked = true; maybeLoadDefault(); });
  $("first-run-dismiss").addEventListener("click", () => { setPref("consent", "quick"); firstRun.hidden = true; });
  const llmBox = $("llm-enabled");
  llmBox.checked = llmEnabled();
  llmBox.addEventListener("change", () => { setPref("consent", llmBox.checked ? "full" : "quick"); maybeLoadDefault(); });
  const strictBox = $("strict-mode");
  strictBox.checked = strictMode();
  strictBox.addEventListener("change", () => setPref("strict", strictBox.checked));
}

// ---- WebNN flags prompt (Chrome/Edge on Mac; navigator.ml is off for most users) ----
// Offered after the first successful conversion, not on load — see maybeOfferWebnn below.
function webnnAvailableNow() {
  return typeof navigator !== "undefined" && !!navigator.ml && typeof navigator.ml.createContext === "function";
}
function isIosFamily() {
  const ua = navigator.userAgent || "";
  return /iPhone|iPad|iPod/.test(ua) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
}
function isMacDesktop() {
  if (isIosFamily()) return false;
  const platform = navigator.userAgentData?.platform || "";
  return platform === "macOS" || /Mac/.test(navigator.platform || "") || /Macintosh/.test(navigator.userAgent || "");
}
function isChromium() {
  const brands = navigator.userAgentData?.brands ?? [];
  if (brands.some((b) => /Chromium|Google Chrome|Microsoft Edge|Brave|Opera/.test(b.brand || ""))) return true;
  const ua = navigator.userAgent || "";
  return /Chrome\//.test(ua) && !/Firefox\//.test(ua);
}
function flagsPageUrl() {
  return /Edg\//.test(navigator.userAgent || "") ? "edge://flags/#webnn" : "chrome://flags/#webnn";
}
function webnnPromptForced() {
  return new URLSearchParams(location.search).get("webnn") === "prompt";
}
function webnnPromptEligible() {
  if (webnnPromptForced()) return true;
  return !webnnAvailableNow() && isMacDesktop() && isChromium();
}
function showWebnnPrompt() {
  const modal = $("webnn-prompt");
  if (!modal) return;
  const open = $("webnn-open");
  const pasteUrl = $("webnn-flags-url");
  open.textContent = /Edg\//.test(navigator.userAgent || "") ? "Open Edge flags" : "Open Chrome flags";
  if (pasteUrl) pasteUrl.textContent = flagsPageUrl();
  $("webnn-paste").hidden = true;
  modal.hidden = false;
  open.focus();
}
function syncWebnnSettings() {
  const s = $("webnn-settings");
  if (s) s.hidden = !webnnPromptEligible();
}
// Shown once, after the first conversion actually took long enough to matter.
function maybeOfferWebnn(ms) {
  if (!webnnPromptEligible() || prefs().webnnPrompt === "dismissed") return;
  const bar = $("webnn-bar");
  if (!bar || !bar.hidden) return;
  const secs = ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
  $("webnn-bar-text").textContent = `That took ${secs}. On this Mac, Chrome can run it several times faster on Apple’s neural engine.`;
  bar.hidden = false;
}
{
  const modal = $("webnn-prompt");
  const bar = $("webnn-bar");
  $("webnn-dismiss")?.addEventListener("click", () => { setPref("webnnPrompt", "dismissed"); modal.hidden = true; bar.hidden = true; });
  $("webnn-settings-open")?.addEventListener("click", () => showWebnnPrompt());
  $("webnn-bar-open")?.addEventListener("click", () => showWebnnPrompt());
  $("webnn-bar-dismiss")?.addEventListener("click", () => { setPref("webnnPrompt", "dismissed"); bar.hidden = true; });
  $("webnn-open")?.addEventListener("click", async () => {
    // Pages cannot navigate to chrome:// or edge://. Copy the URL and show it
    // so the click still lands them on the flags page with the WebNN search.
    const url = flagsPageUrl();
    try { await navigator.clipboard.writeText(url); } catch { /* ignore */ }
    $("webnn-paste").hidden = false;
    $("webnn-flags-url").textContent = url;
    toast("Paste the copied address into the bar, then set those flags to Enabled.");
  });
  modal?.addEventListener("click", (e) => { if (e.target === modal) { setPref("webnnPrompt", "dismissed"); modal.hidden = true; bar.hidden = true; } });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || !modal || modal.hidden) return;
    setPref("webnnPrompt", "dismissed");
    modal.hidden = true;
    bar.hidden = true;
  });
  syncWebnnSettings();
  if (webnnPromptForced()) showWebnnPrompt();
}

// ---- history and favorites (this browser only) ----
const HISTORY_KEY = "latexgen.history", HISTORY_MAX = 100;
function loadHistory() { try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); } catch { return []; } }
function saveHistory(items) { try { localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, HISTORY_MAX))); } catch {} }
function recordHistory(input, latex) {
  if (!latex || !latex.trim()) return;
  const items = loadHistory();
  const dup = items.findIndex((h) => h.latex === latex);
  if (dup !== -1) { const [h] = items.splice(dup, 1); h.ts = Date.now(); items.unshift(h); }
  else items.unshift({ id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`, ts: Date.now(), input, latex, fav: false });
  saveHistory(items); renderHistory(); syncVisualEdit();
}
let historyStarredOnly = false;
function renderHistory() {
  const list = $("history-list"); if (!list) return;
  const items = loadHistory().filter((h) => !historyStarredOnly || h.fav);
  list.innerHTML = "";
  if (!items.length) { const e = document.createElement("div"); e.className = "history-empty"; e.textContent = historyStarredOnly ? "No starred conversions yet." : "Conversions you make will show up here."; list.appendChild(e); return; }
  for (const h of items) {
    const row = document.createElement("div"); row.className = "hist"; row.tabIndex = 0; row.setAttribute("role", "button");
    const star = document.createElement("button"); star.className = `star${h.fav ? " on" : ""}`; star.textContent = h.fav ? "★" : "☆"; star.title = h.fav ? "Unstar" : "Star"; star.setAttribute("aria-label", star.title);
    star.addEventListener("click", (e) => { e.stopPropagation(); const all = loadHistory(); const t = all.find((x) => x.id === h.id); if (t) { t.fav = !t.fav; saveHistory(all); renderHistory(); } });
    const body = document.createElement("div"); body.className = "body";
    const code = document.createElement("code"); code.textContent = h.latex;
    const meta = document.createElement("div"); meta.className = "meta";
    meta.textContent = `${new Date(h.ts).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })} · ${h.input === "(image)" ? "from image" : h.input}`;
    body.appendChild(code); body.appendChild(meta); row.appendChild(star); row.appendChild(body);
    const restore = () => {
      outputCode.textContent = h.latex; renderPreview(h.latex); currentLatex = h.latex; currentInput = h.input;
      if (!String(h.input).startsWith("(")) { $("input").value = h.input; $("input").dispatchEvent(new Event("input")); }
      const issues = checkSyntax(h.latex);
      showChecks({ ok: issues.length === 0, issues }, "(restored from history)");
      setChatEnabled(true); $("history").hidden = true; window.scrollTo({ top: 0, behavior: "smooth" });
    };
    row.addEventListener("click", restore);
    row.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); restore(); } });
    list.appendChild(row);
  }
}
$("history-filter").addEventListener("click", (e) => { historyStarredOnly = !historyStarredOnly; e.currentTarget.setAttribute("aria-pressed", String(historyStarredOnly)); e.currentTarget.textContent = historyStarredOnly ? "Show all" : "Starred only"; renderHistory(); });
$("history-clear").addEventListener("click", () => { if (confirm("Clear all history in this browser?")) { saveHistory([]); renderHistory(); } });

// ---- copy formats ----
async function writeClipboard(items, label) {
  try {
    if (typeof ClipboardItem !== "undefined" && items.some((i) => i.type !== "text/plain")) {
      await navigator.clipboard.write([new ClipboardItem(Object.fromEntries(items.map((i) => [i.type, i.blob ?? new Blob([i.text], { type: i.type })])))]);
    } else await navigator.clipboard.writeText(items.find((i) => i.type === "text/plain")?.text ?? "");
    toast(`Copied ${label}`);
  } catch (err) { toast(`Copy failed: ${err.message || err}`, 3000); }
}
// Render LaTeX to a PNG data URL without touching the visible preview.
async function renderPngDataUrl(latex) {
  const box = document.createElement("div");
  box.style.cssText = "position:fixed;left:-10000px;top:0;padding:16px;background:#fff;color:#111;font-size:18px;";
  box.textContent = /\\\[|\\\(|\$\$|(?:^|[^\\])\$/.test(latex) ? latex : `\\[${latex}\\]`;
  document.body.appendChild(box);
  try {
    renderMathInElement(box, { delimiters: [{ left: "$$", right: "$$", display: true }, { left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }, { left: "$", right: "$", display: false }], throwOnError: false });
    return await htmlToImage.toPng(box, { backgroundColor: "#ffffff", pixelRatio: 2 });
  } finally { box.remove(); }
}
async function copyAs(kind) {
  const latex = outputCode.textContent.trim();
  if (!latex) { toast("Nothing to copy yet"); return; }
  if (kind === "display" || kind === "inline") return writeClipboard([{ type: "text/plain", text: toFormat(latex, kind) }], `${kind} math`);
  if (kind === "mathml") {
    const mathml = toFormat(latex, "mathml");
    if (!mathml) { toast("Could not build MathML for this LaTeX", 3000); return; }
    return writeClipboard([{ type: "text/html", text: mathml }, { type: "text/plain", text: mathml }], "MathML");
  }
  if (kind === "png") {
    try { const blob = await htmlToImage.toBlob(preview, { backgroundColor: "#ffffff", pixelRatio: 2, style: { padding: "16px" } }); await writeClipboard([{ type: "image/png", blob }, { type: "text/plain", text: latex }], "PNG"); }
    catch (err) { toast(`PNG export failed: ${err.message || err}`, 3000); }
    return;
  }
  if (kind === "share") return writeClipboard([{ type: "text/plain", text: `${location.origin}${location.pathname}#l=${encodeURIComponent(latex)}` }], "share link");
  if (kind === "overleaf") {
    const doc = `\\documentclass{article}\n\\usepackage{amsmath,amssymb}\n\\begin{document}\n${toFormat(latex, "display")}\n\\end{document}\n`;
    const form = document.createElement("form"); form.method = "POST"; form.action = "https://www.overleaf.com/docs"; form.target = "_blank";
    const inp = document.createElement("input"); inp.type = "hidden"; inp.name = "snip"; inp.value = doc;
    form.appendChild(inp); document.body.appendChild(form); form.submit(); form.remove();
    toast("Opening in Overleaf…");
  }
}
{
  const menu = $("copy-menu"), menuBtn = $("copy-menu-btn");
  $("copy-btn").addEventListener("click", () => { const latex = outputCode.textContent.trim(); if (!latex) { toast("Nothing to copy yet"); return; } writeClipboard([{ type: "text/plain", text: latex }], "LaTeX"); });
  menuBtn.addEventListener("click", () => { menu.hidden = !menu.hidden; menuBtn.setAttribute("aria-expanded", String(!menu.hidden)); });
  for (const b of menu.querySelectorAll("[data-copy]")) b.addEventListener("click", () => { menu.hidden = true; menuBtn.setAttribute("aria-expanded", "false"); copyAs(b.dataset.copy); });
  document.addEventListener("click", (e) => { if (!menu.hidden && !menu.contains(e.target) && e.target !== menuBtn) { menu.hidden = true; menuBtn.setAttribute("aria-expanded", "false"); } });
}

// ---- installable + offline ----
if ("serviceWorker" in navigator) window.addEventListener("load", () => navigator.serviceWorker.register(new URL("sw.js", import.meta.url)).catch((e) => console.warn("sw:", e)));

// ---- shared link: LaTeX in the URL fragment (never sent to a server) ----
{
  const m = location.hash.match(/^#l=(.+)$/);
  if (m) {
    try {
      const latex = decodeURIComponent(m[1]);
      outputCode.textContent = latex; renderPreview(latex);
      currentLatex = latex; currentInput = "(shared link)";
      const issues = checkSyntax(latex);
      showChecks({ ok: issues.length === 0, issues }, "(opened from a shared link)");
      setChatEnabled(true); history.replaceState(null, "", location.pathname);
      setTimeout(() => syncVisualEdit(), 0);
    } catch {}
  }
}

// ---- MathLive visual editor ----
let mathliveReady = null;
function loadMathlive() {
  mathliveReady ??= import("./vendor/mathlive/mathlive.min.mjs").then((mod) => { mod.MathfieldElement.fontsDirectory = new URL("vendor/mathlive/fonts", import.meta.url).href; mod.MathfieldElement.soundsDirectory = null; return mod; });
  return mathliveReady;
}
function mathBodyOf(latex) { const t = latex.trim(); const m = t.match(/^(?:\$\$([\s\S]+)\$\$|\\\[([\s\S]+)\\\]|\\\(([\s\S]+)\\\)|\$([^$]+)\$)$/); if (m) return (m[1] ?? m[2] ?? m[3] ?? m[4]).trim(); return /\$|\\\[|\\\(/.test(t) ? null : t; }
function delimitersOf(latex) {
  const t = latex.trim();
  if (/^\$\$[\s\S]*\$\$$/.test(t)) return ["$$", "$$"]; if (/^\\\[[\s\S]*\\\]$/.test(t)) return ["\\[", "\\]"];
  if (/^\\\([\s\S]*\\\)$/.test(t)) return ["\\(", "\\)"]; if (/^\$[^$]*\$$/.test(t)) return ["$", "$"];
  return ["", ""];
}
function syncVisualEdit() {
  const btn = $("visual-edit-btn"); const body = mathBodyOf(outputCode.textContent.trim());
  btn.hidden = body == null;
  if (body == null && !$("math-field").hidden) setVisualEdit(false);
}
async function setVisualEdit(on) {
  const btn = $("visual-edit-btn"), mf = $("math-field");
  if (on) {
    await loadMathlive();
    const latex = outputCode.textContent.trim();
    mf.value = mathBodyOf(latex) ?? latex; mf._delims = delimitersOf(latex);
    mf.hidden = false; outputCode.hidden = true;
    btn.textContent = "</> LaTeX source"; btn.setAttribute("aria-pressed", "true"); mf.focus();
  } else {
    mf.hidden = true; outputCode.hidden = false;
    btn.textContent = "✎ Visual editor"; btn.setAttribute("aria-pressed", "false");
  }
}
{
  const btn = $("visual-edit-btn"), mf = $("math-field");
  btn.addEventListener("click", () => setVisualEdit(mf.hidden));
  let t = null;
  mf.addEventListener("input", () => {
    clearTimeout(t);
    t = setTimeout(() => {
      const [l, r] = mf._delims ?? ["", ""]; const body = mf.value;
      const latex = l === "$$" || l === "\\[" ? `${l}\n${body}\n${r}` : `${l}${body}${r}`;
      outputCode.textContent = latex; currentLatex = latex; renderPreview(latex);
      const issues = checkSyntax(latex);
      showChecks({ ok: issues.length === 0, issues }, "(edited visually)");
    }, 250);
  });
  outputCode.addEventListener("input", () => setTimeout(syncVisualEdit, 350));
}

// =====================================================================
// Tab API: other apps use this tab through a relay. Jobs run headlessly on
// the same pipeline as the UI — concurrently, without touching the screen —
// and the API drawer shows the address, status, running jobs and a log.
//
// Both this and the compute mesh need the relay in server.js, so the whole
// block is inert in the static build (mesh hooks stay at their "off" default).
// =====================================================================
if (!STATIC_BUILD) {
  const TABAPI_KEY = "latexgen.tabapi", LOG_KEY = "latexgen.apilog", LOG_MAX = 50, MAX_CONCURRENT = 3;
  const toggle = $("tabapi-toggle"), details = $("tabapi-details"), urlEl = $("tabapi-url");
  const statusEl = $("tabapi-status"), dotEl = $("tabapi-dot"), exampleEl = $("tabapi-example");
  const runningEl = $("api-running"), logEl = $("api-log"), badge = $("api-badge");
  const tprefs = () => { try { return JSON.parse(localStorage.getItem(TABAPI_KEY) || "{}"); } catch { return {}; } };
  const save = (p) => localStorage.setItem(TABAPI_KEY, JSON.stringify(p));
  const newId = () => [...crypto.getRandomValues(new Uint8Array(16))].map((b) => b.toString(16).padStart(2, "0")).join("");
  let state = tprefs();
  // The address identifies THIS tab (two tabs must not share one), so the id
  // lives in sessionStorage: stable across reloads of the tab, unique per tab.
  const ID_KEY = "latexgen.tabapi.id";
  let tabId = sessionStorage.getItem(ID_KEY) || state.id; // migrate an old localStorage id once
  if (!tabId) tabId = newId();
  sessionStorage.setItem(ID_KEY, tabId); delete state.id; save(state);
  state.id = tabId;
  let generation = 0;
  // mesh hooks for the pipeline: peers are usable only while this tab serves
  meshHooks.enabled = () => !!(state.enabled && state.pool);
  meshHooks.tabId = () => state.id;
  const poolToggle = $("pool-toggle"), poolDetails = $("pool-details"), poolKey = $("pool-key"), poolImages = $("pool-images"), poolStats = $("pool-stats");
  // Advertise what this tab can do; the relay uses it to route pool jobs.
  async function registerCaps() {
    if (!state.enabled) return;
    const row = modelRows.find((r) => r.id === loadedModel);
    const body = {
      caps: { specialist: !!pipe.loaded.intellitex, texo: !!pipe.loaded.texo, texify: !!pipe.loaded.texify, llm: !!engine, llmName: row?.name ?? null, multiline: !!row?.multiline, tokPerSec: browserTokPerSec },
      pool: state.pool ? { keys: [state.poolKey ? `k:${state.poolKey}` : "public"], images: !!state.poolImages, maxConcurrent: 1 } : null,
    };
    try {
      const r = await fetch(`api/relay/${state.id}/caps`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
      const d = await r.json();
      poolStats.textContent = `served ${d.servedJobs} · used ${d.usedJobs} · ${d.peers} peer${d.peers === 1 ? "" : "s"} online`;
    } catch {}
  }
  window.addEventListener("latexgen:caps-changed", registerCaps);
  setInterval(registerCaps, 60_000);
  const running = new Map(); // jobId -> { kind, summary, started }
  let log = (() => { try { return JSON.parse(sessionStorage.getItem(LOG_KEY) || "[]"); } catch { return []; } })();

  const render = () => {
    const base = new URL(`api/tab/${state.id}`, location.href).href;
    urlEl.textContent = `${base}/convert`;
    exampleEl.textContent = [
      `# text -> LaTeX (same ladder, checks and repair as the Convert button)`,
      `curl -X POST ${base}/convert -H 'content-type: application/json' \\`,
      `  -d '{"text": "the sum from n equals 1 to infinity of 1 over n squared"}'`,
      ``,
      `# options on /convert: "format": "latex"|"display"|"inline"|"mathml"|"png"`,
      `#   "strict": true (waits for the second-opinion judge), "engine": "browser"|"server"`,
      `#   image: {"imageBase64": "...", "ocr": "auto"|"texo"|"texify"}`,
      `#   existing LaTeX: {"latex": "..."}  (check + repair, like the Check LaTeX tab)`,
      `POST ${base}/refine   {"latex": "...", "instruction": "change TV to T(V)", "original": "..."}`,
      `POST ${base}/check    {"latex": "..."}`,
      `POST ${base}/format   {"latex": "...", "format": "mathml"}`,
      `POST ${base}/status   {}        # models loaded, runtime, speed, server`,
      `POST ${base}/history  {}        # this browser's conversion history`,
      ``,
      `# response: {"latex", "ok", "issues", "model", "note", "ms", ...}`,
    ].join("\n");
    toggle.checked = !!state.enabled; details.hidden = !state.enabled;
    poolToggle.checked = !!state.pool; poolDetails.hidden = !state.pool;
    poolKey.value = state.poolKey ?? ""; poolImages.checked = !!state.poolImages;
    renderRunning(); renderLog();
  };
  const renderRunning = () => {
    runningEl.innerHTML = "";
    for (const [id, j] of running) { const d = document.createElement("div"); d.className = "job"; d.textContent = `▸ ${j.kind} · ${j.summary} · ${Math.round((performance.now() - j.started) / 1000)}s…`; runningEl.appendChild(d); }
    badge.hidden = running.size === 0; badge.textContent = String(running.size);
    dotEl.className = `dot-ind${running.size ? " busy" : state.enabled ? " on" : ""}`;
  };
  const renderLog = () => {
    logEl.innerHTML = "";
    if (!log.length) { const e = document.createElement("div"); e.className = "history-empty"; e.textContent = "No requests yet."; logEl.appendChild(e); return; }
    for (const r of log) {
      const row = document.createElement("div"); row.className = `req ${r.ok ? "ok" : "err"}`;
      const kind = document.createElement("span"); kind.className = "kind"; kind.textContent = r.kind;
      const sum = document.createElement("span"); sum.className = "sum"; sum.textContent = r.summary; sum.title = r.summary;
      const meta = document.createElement("span"); meta.className = "meta";
      meta.textContent = `${new Date(r.t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })} · ${r.error ? "error" : r.ok ? "ok" : "issues"} · ${r.ms} ms${r.model ? ` · ${r.model}` : ""}`;
      row.appendChild(kind); row.appendChild(sum); row.appendChild(meta); logEl.appendChild(row);
    }
  };
  const logEntry = (e) => { log.unshift(e); log = log.slice(0, LOG_MAX); try { sessionStorage.setItem(LOG_KEY, JSON.stringify(log)); } catch {} renderLog(); };
  const summarize = (job) => (job.pool ? "peer · " : "") + (job.text ? job.text.slice(0, 60) : job.imageBase64 ? `image (${Math.round(job.imageBase64.length * 0.75 / 1024)} KB)` : job.latex ? job.latex.slice(0, 60) : job.kind);

  async function runJob(job) {
    const fmt = async (latex, format) => {
      if (!format || format === "latex") return { output: latex };
      if (format === "png") {
        // Font inlining can stall on a broken cache; never hang the job.
        try {
          const png = await Promise.race([renderPngDataUrl(latex), new Promise((_, rej) => setTimeout(() => rej(new Error("png render timed out")), 15000))]);
          return { output: png, format: "png" };
        } catch (e) { return { output: latex, format: "png", formatOk: false, formatError: String(e.message || e) }; }
      }
      const out = toFormat(latex, format);
      return { output: out ?? latex, format, formatOk: out != null };
    };
    const pack = (r, extra = {}) => ({ latex: r.latex, ok: r.validation.ok, issues: r.validation.issues, model: r.model, note: r.note?.replace(/^\(|\)$/g, "") ?? "", ms: r.ms, ...extra });
    const eng = job.engine === "server" ? "server" : "browser";
    const onEvent = (e) => logEvent({ ...e, title: `${job.pool ? "Peer" : "Tab API"} · ${e.title}` });
    if (job.kind === "status") {
      return { ok: true, models: { specialist: pipe.loaded.intellitex, texo: pipe.loaded.texo, texify: pipe.loaded.texify, llm: loadedModel ? (modelRows.find((r) => r.id === loadedModel)?.name ?? loadedModel) : null },
        webllm: engine?.latexgenPlan?.label ?? null, runtime: runtimeUsed, tokPerSec: browserTokPerSec, server: serverAvailable, strictMode: strictMode(), running: running.size };
    }
    if (job.kind === "history") return { ok: true, history: loadHistory() };
    if (job.kind === "format") return { ok: true, latex: job.latex, ...(await fmt(job.latex, job.format)) };
    if (job.kind === "check" || (job.kind === "convert" && job.latex && !job.text && !job.imageBase64)) {
      const r = await pipe.checkLatex(job.latex, { onEvent }); return pack(r, { repaired: r.repaired, ...(await fmt(r.latex, job.format)) });
    }
    if (job.kind === "refine") {
      const r = await pipe.refine({ original: job.original || "(tab api)", latex: job.latex, instruction: job.instruction, engineChoice: eng, allowMesh: !job.noMesh, allowServer: !job.noServer, onEvent });
      return pack(r, { echo: r.echo, retried: r.retried, ...(await fmt(r.latex, job.format)) });
    }
    if (job.imageBase64) {
      const bin = atob(job.imageBase64.replace(/^data:[^,]+,/, ""));
      const bytes = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      const r = await pipe.convertImage(new Blob([bytes], { type: job.mime || "image/png" }), { ocr: job.ocr || "auto", onEvent });
      return pack(r, { ocr: r.used, escalated: r.escalatedWhy || undefined, repaired: r.repaired, ...(await fmt(r.latex, job.format)) });
    }
    const local = { allowMesh: !job.noMesh, allowServer: !job.noServer };
    const r = await pipe.convertText(job.text, { engineChoice: eng, strict: !!job.strict, ...local, onEvent });
    if (r.validation.ok && !job.pool) recordHistory(job.text, r.latex);
    return pack(r, { escalated: r.escalated, batch: r.batch, judge: r.judge, peer: r.peer, ...(await fmt(r.latex, job.format)) });
  }

  async function handle(job) {
    const started = performance.now();
    running.set(job.jobId, { kind: job.pool ? "peer job" : job.kind, summary: summarize(job), started }); renderRunning();
    const tick = setInterval(renderRunning, 1000);
    let result;
    try { result = await runJob(job); } catch (e) { result = { error: String(e.message || e) }; }
    clearInterval(tick);
    result.ms ??= Math.round(performance.now() - started);
    try { await fetch(`api/relay/${state.id}/result/${job.jobId}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(result) }); } catch {}
    running.delete(job.jobId); renderRunning();
    logEntry({ t: Date.now(), kind: job.pool ? "peer job" : job.kind, summary: summarize(job), model: result.model, ms: result.ms, ok: !!result.ok && !result.error, error: result.error });
    statusEl.textContent = `answered ${job.pool ? "a peer's " : ""}${job.kind} in ${result.ms} ms · listening`;
    if (job.pool) registerCaps();
  }

  async function pollLoop(gen) {
    while (state.enabled && gen === generation) {
      try {
        if (running.size >= MAX_CONCURRENT) { await new Promise((r) => setTimeout(r, 200)); continue; }
        if (!running.size) statusEl.textContent = "listening for requests…";
        dotEl.className = `dot-ind${running.size ? " busy" : " on"}`;
        const r = await fetch(`api/relay/${state.id}/next`, { cache: "no-store" });
        if (gen !== generation) break;
        if (r.status === 204) continue;
        if (!r.ok) { await new Promise((res) => setTimeout(res, 3000)); continue; }
        handle(await r.json()); // don't await: keep polling so jobs run concurrently
      } catch { await new Promise((res) => setTimeout(res, 3000)); }
    }
    if (!state.enabled) { statusEl.textContent = "idle"; dotEl.className = "dot-ind"; }
  }

  toggle.addEventListener("change", () => {
    state.enabled = toggle.checked; if (!state.enabled) state.pool = false; save(state); generation++; render();
    if (state.enabled) { pollLoop(generation); registerCaps(); }
  });
  poolToggle.addEventListener("change", () => {
    state.pool = poolToggle.checked;
    if (state.pool && !state.enabled) { state.enabled = true; generation++; pollLoop(generation); } // serving is the price of using peers
    save(state); render(); registerCaps();
  });
  poolKey.addEventListener("change", () => { state.poolKey = poolKey.value.trim(); save(state); registerCaps(); });
  poolImages.addEventListener("change", () => { state.poolImages = poolImages.checked; save(state); registerCaps(); });
  $("tabapi-regen").addEventListener("click", () => { state.id = newId(); sessionStorage.setItem(ID_KEY, state.id); generation++; render(); if (state.enabled) { pollLoop(generation); registerCaps(); } toast("New address — the old one no longer works"); });
  $("tabapi-copy").addEventListener("click", () => { navigator.clipboard.writeText(urlEl.textContent); toast("Copied API address"); });
  $("api-log-clear").addEventListener("click", () => { log = []; sessionStorage.removeItem(LOG_KEY); renderLog(); });
  render();
  if (state.enabled) { pollLoop(generation); specialistReady.then(registerCaps, registerCaps); }
  // the UI's own conversions that went to a peer show up in the log too
  window.addEventListener("latexgen:converted", (e) => { if (e.detail?.peer) logEntry({ t: Date.now(), kind: "via peer", summary: e.detail.summary, model: e.detail.model, ms: e.detail.ms, ok: true }); });
}

// ---- resilience: surface unexpected errors, show connectivity ----
{
  const report = (msg) => { try { toast(`Something went wrong: ${String(msg).slice(0, 80)}`, 4000); } catch {} };
  window.addEventListener("error", (e) => { if (e.message && !/ResizeObserver/.test(e.message)) report(e.message); });
  window.addEventListener("unhandledrejection", (e) => report(e.reason?.message || e.reason));
  const chip = $("privacy-chip");
  const setOffline = (off) => {
    if (!chip) return;
    if (off) { chip.dataset.prev ??= chip.innerHTML; chip.className = "chip warn"; chip.innerHTML = "<i></i>Offline — on-device models still work"; }
    else if (chip.dataset.prev) { chip.className = "chip ok"; chip.innerHTML = chip.dataset.prev; }
  };
  window.addEventListener("offline", () => setOffline(true));
  window.addEventListener("online", () => setOffline(false));
  if (!navigator.onLine) setOffline(true);
}
