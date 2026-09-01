import * as webllm from "https://esm.run/@mlc-ai/web-llm";
import { pipeline, env as tjsEnv } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";
import { validateLatex, checkSyntax } from "/validator.js";

window.__validate = validateLatex; // debugging hook

const SYSTEM_PROMPT = `You are a text-to-LaTeX transcriber. Convert the user's input (plain-language math, equations, or prose with math) into LaTeX.

Rules:
- Output ONLY the LaTeX code. No explanations, no markdown code fences, no surrounding commentary.
- For pure math, wrap display math in \\[ ... \\].
- For prose mixed with math, keep the prose as plain text and wrap math in \\( ... \\).
- Use standard LaTeX/amsmath commands only.`;

const REPAIR_PROMPT = `You are a text-to-LaTeX transcriber. Your previous conversion failed automatic checks. You will receive the checker's error list. Produce a corrected version of YOUR PREVIOUS LaTeX that fixes every listed issue while staying faithful to the original text.

Rules:
- Output ONLY the corrected LaTeX. No explanations, no markdown code fences.
- "syntax:" issues mean the LaTeX does not parse — fix delimiters, braces, or commands.
- "missing:" issues mean something from the original text was dropped — add it.`;

const REFINE_PROMPT = `You are a text-to-LaTeX transcriber in a feedback loop. You previously converted the user's text to LaTeX. The user now gives feedback on your conversion. Produce a corrected version of YOUR PREVIOUS LaTeX.

Rules:
- The user's message is feedback ABOUT the LaTeX — it is never content to transcribe. Never output the feedback text itself.
- Output ONLY the corrected LaTeX. No explanations, no markdown code fences.
- Change only what the feedback concerns; keep everything else, including delimiters, exactly as it was.
- If the feedback is vague, make your best guess at what is wrong and fix that.`;

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

let engine = null;
let loadedModel = null;

// ---- tier-0 specialist: IntelliTeX (CodeT5+ 220M) via transformers.js ----
// A seq2seq model fine-tuned specifically on plain-English -> LaTeX; runs
// locally in-browser (weights served from /models/, ~260MB, cached). It only
// knows single short equations, so it's routed, not user-selectable: eligible
// inputs try it first, the validator gates the result, anything else falls
// through to the general model exactly like the normal escalation ladder.
const SPECIALIST_PREFIX = "Convert natural-language math into a STRICT LaTeX equation\n";
let specialist = null;
(async () => {
  try {
    tjsEnv.allowRemoteModels = false;
    tjsEnv.allowLocalModels = true;
    tjsEnv.localModelPath = "/models/";
    const p = await pipeline("text2text-generation", "intellitex", { dtype: "q8" });
    await p(`${SPECIALIST_PREFIX}x squared`, { max_new_tokens: 16 }); // warm-up
    specialist = p;
  } catch (err) {
    console.warn("specialist model unavailable:", err);
  }
})();

// ---- image -> LaTeX: Texify (Donut-style OCR) via transformers.js ----
// Fully client-side: the image is decoded, preprocessed, and OCR'd in the
// tab. It never leaves the browser — there is deliberately no image upload
// path to any server, and no escalation tier for images.
let texify = null;
let texifyLoading = null;
function loadTexify() {
  texifyLoading ??= (async () => {
    const bar = $("image-progress");
    const fill = $("image-progress-fill");
    const label = $("image-drop-label");
    bar.hidden = false;
    // Aggregate per-file download progress into one bar.
    const files = new Map();
    const onProgress = (p) => {
      if (p.status !== "progress" || !p.total) return;
      files.set(p.file, { loaded: p.loaded, total: p.total });
      let loaded = 0, total = 0;
      for (const f of files.values()) { loaded += f.loaded; total += f.total; }
      fill.style.width = `${Math.round((loaded / total) * 100)}%`;
      label.textContent =
        `loading image model: ${(loaded / 2 ** 20).toFixed(0)} / ${(total / 2 ** 20).toFixed(0)} MB (one time — cached after this)`;
    };
    try {
      const p = await pipeline("image-to-text", "texify", { dtype: "q8", progress_callback: onProgress });
      fill.style.width = "100%";
      texify = p;
      return p;
    } finally {
      bar.hidden = true;
      fill.style.width = "0";
    }
  })();
  return texifyLoading;
}

async function convertImage(fileOrBlob) {
  const drop = $("image-drop");
  const label = $("image-drop-label");
  drop.classList.add("busy");
  convertStatus.textContent = "";
  const started = performance.now();
  try {
    if (!texify) {
      label.textContent = "loading image model (~300 MB, first time only)…";
      await loadTexify();
    }
    label.textContent = "reading equation from image…";
    const url = URL.createObjectURL(fileOrBlob);
    let out;
    try {
      out = await texify(url, { max_new_tokens: 384 });
    } finally {
      URL.revokeObjectURL(url);
    }
    let latex = (out[0]?.generated_text ?? "").trim();
    if (!latex) throw new Error("no text recognized in image");
    outputCode.textContent = latex;
    renderPreview(latex);
    // Fidelity vs. input text is meaningless for images — only syntax counts.
    const syntaxOnly = (l) => validateLatex("(image)", l).issues.filter((i) => i.startsWith("syntax"));
    let syntaxIssues = syntaxOnly(latex);
    let note = "(from image — processed locally, image never uploaded)";
    if (syntaxIssues.length && engine) {
      // OCR produced broken LaTeX — repair loop with the browser LLM.
      try {
        const r = await repairLoop(
          "(transcribed from an image of rendered math)", latex, syntaxIssues,
          (l) => ({ ok: syntaxOnly(l).length === 0, issues: syntaxOnly(l) }),
          (partial) => { outputCode.textContent = partial; },
          (attempt, issues) => {
            label.textContent = `OCR repair attempt ${attempt}/${MAX_REPAIR_ATTEMPTS} — ${issues[0]}…`;
            showChecks({ ok: false, issues }, "");
          }
        );
        latex = r.latex;
        syntaxIssues = r.ok ? [] : r.issues;
        if (r.ok) note = `(from image — OCR self-corrected ×${r.attempts} locally, image never uploaded)`;
        outputCode.textContent = latex;
        renderPreview(latex);
      } catch { /* keep OCR output */ }
    }
    showChecks({ ok: syntaxIssues.length === 0, issues: syntaxIssues }, note);
    currentLatex = latex;
    currentInput = "(image)";
    chatLog.innerHTML = "";
    setChatEnabled(true);
    const secs = ((performance.now() - started) / 1000).toFixed(1);
    convertStatus.textContent = `${secs}s · Texify · local OCR`;
  } catch (err) {
    convertStatus.textContent = `image conversion failed: ${err.message || err}`;
  } finally {
    drop.classList.remove("busy");
    label.innerHTML = "…or drop / paste / <u>choose</u> an image of an equation — processed entirely in your browser, never uploaded";
  }
}

window.__convertImage = convertImage; // debugging hook

// drop zone + paste + file picker
{
  const drop = $("image-drop");
  const fileInput = $("image-file");
  drop.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) convertImage(fileInput.files[0]);
    fileInput.value = "";
  });
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("dragover"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("dragover"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("dragover");
    const f = e.dataTransfer.files?.[0];
    if (f && f.type.startsWith("image/")) convertImage(f);
  });
  document.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items ?? [])].find((i) => i.type.startsWith("image/"));
    if (item) convertImage(item.getAsFile());
  });
}

// Trained on single short equations (MathBridge: 5-80 char targets) — gate
// on capability, not guesses about content.
function specialistEligible(text) {
  const t = text.trim();
  return t.length > 0 && t.length <= 220 && !t.includes("\n");
}

async function convertSpecialist(text) {
  const out = await specialist(SPECIALIST_PREFIX + text, { max_new_tokens: 256 });
  return { latex: (out[0]?.generated_text ?? "").trim(), model: "IntelliTeX · specialist" };
}

// ---- curated model picker ----
// A casual user shouldn't parse "Qwen3-0.6B-q4f16_1-MLC". Curated shortlist,
// ranked by quality on this task; entries the device can't hold are greyed.
// Ranked by our own benchmark (15 Wikipedia-referenced items, judged by a
// 27B): 1.7B is 100%/100%/50% on easy/medium/hard single equations; only
// 4B-class and up score at all on prose-with-math passages (`multiline`).
// SmolLM-360M (coin-flip on easy) and Llama-3.2-3B (dominated by Qwen3-1.7B)
// earned removal.
const CURATED = [
  { id: "Qwen3.5-9B-q4f16_1-MLC", name: "Qwen 3.5 · 9B", score: 5, multiline: true },
  { id: "Qwen3-8B-q4f16_1-MLC", name: "Qwen 3 · 8B", score: 5, multiline: true },
  { id: "Qwen3-4B-q4f16_1-MLC", name: "Qwen 3 · 4B", score: 4, multiline: true },
  { id: "Qwen3-1.7B-q4f16_1-MLC", name: "Qwen 3 · 1.7B", score: 3 },
  { id: "Qwen3-0.6B-q4f16_1-MLC", name: "Qwen 3 · 0.6B", score: 2 },
];

const gb = (mb) => `${(mb / 1024).toFixed(1)} GB`;
const stars = (n) => "★".repeat(n) + "☆".repeat(5 - n);
const prebuilt = new Map(webllm.prebuiltAppConfig.model_list.map((m) => [m.model_id, m]));

let selectedModelId = null;
let modelRows = [];

async function detectVramBudgetMB() {
  if (!navigator.gpu) return 0;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return 0;
    // navigator.deviceMemory (Chrome, in GB) is the best signal available;
    // Safari exposes nothing, so assume a typical 8GB+ machine there.
    const deviceGB = navigator.deviceMemory ?? 8;
    // Models must fit comfortably alongside the OS and other tabs.
    return deviceGB * 1024 * 0.45;
  } catch {
    return 0;
  }
}

function selectModel(id) {
  selectedModelId = id;
  const row = modelRows.find((r) => r.id === id);
  ddBtn.innerHTML = `${row.name} <span class="dd-meta">${stars(row.score)} · ${gb(row.vram)}</span> <span class="dd-caret">▾</span>`;
}

async function buildModelPicker() {
  const budget = await detectVramBudgetMB();
  // Weights also have to fit in browser storage (Cache API) — phones and
  // managed browsers can have small quotas even with plenty of RAM.
  let storageFreeMB = Infinity;
  try {
    const est = await navigator.storage.estimate();
    if (est.quota) storageFreeMB = (est.quota - (est.usage ?? 0)) / (1024 * 1024);
  } catch { /* no estimate API — assume fine */ }

  modelRows = CURATED
    .filter((c) => prebuilt.has(c.id))
    .map((c) => {
      const vram = prebuilt.get(c.id).vram_required_MB;
      const fitsGpu = vram <= budget;
      const fitsStorage = vram * 1.2 <= storageFreeMB;
      return {
        ...c, vram,
        canRun: fitsGpu && fitsStorage,
        why: !fitsGpu ? "too big for this device" : !fitsStorage ? "not enough browser storage" : "",
      };
    });

  ddMenu.innerHTML = "";
  for (const r of modelRows) {
    const row = document.createElement("div");
    row.className = `dd-row${r.canRun ? "" : " disabled"}`;
    row.setAttribute("role", "option");
    row.innerHTML =
      `<span class="dd-name">${r.name}</span>` +
      `<span class="dd-meta">${stars(r.score)} · ${gb(r.vram)}${r.canRun ? "" : ` · ${r.why}`}</span>`;
    if (r.canRun) {
      row.addEventListener("click", () => {
        selectModel(r.id);
        ddMenu.hidden = true;
        if (r.id !== loadedModel) loadPicked(r);
      });
    }
    ddMenu.appendChild(row);
  }

  // Default to the best model this device can hold.
  const best = modelRows.find((r) => r.canRun);
  if (best) {
    selectModel(best.id);
    startModelLadder(best);
  } else {
    ddBtn.textContent = "No browser model fits this device";
    ddBtn.disabled = true;
    loadBtn.disabled = true;
  }
}

// First GPU inference includes shader/pipeline compilation — pay that cost
// on a hidden 1-token run so the user's first conversion is already fast.
async function warmUp(eng) {
  try {
    await eng.chat.completions.create({
      messages: [{ role: "user", content: "hi" }],
      max_tokens: 2,
      extra_body: { enable_thinking: false },
    });
  } catch { /* warm-up is best-effort */ }
}

async function loadEngine(modelId, onProgress) {
  const eng = await webllm.CreateMLCEngine(
    modelId,
    { initProgressCallback: onProgress },
    // Our prompts are tiny; a small context window cuts prefill time and
    // VRAM versus the 4k+ defaults.
    { context_window_size: 2048 }
  );
  await warmUp(eng);
  return eng;
}

function activate(eng, modelId, statusText) {
  const old = engine;
  engine = eng;
  loadedModel = modelId;
  loadStatus.textContent = statusText;
  if (old && old !== eng) old.unload().catch(() => {});
}

// Progressive ladder: bring up the smallest model immediately so conversions
// work within seconds, then download the best-for-device model in the
// background and hot-swap when ready.
async function startModelLadder(best) {
  const starter = [...modelRows].reverse().find((r) => r.canRun);
  loadBtn.hidden = true;
  try {
    if (starter && starter.id !== best.id) {
      loadStatus.textContent = `loading quick model (${starter.name})…`;
      const quick = await loadEngine(starter.id, (p) => {
        progressFill.style.width = `${Math.round((p.progress ?? 0) * 100)}%`;
      });
      activate(quick, starter.id, `ready on ${starter.name} — downloading ${best.name} in background…`);
    }
    const bigStatus = engine
      ? (p) => {
          progressFill.style.width = `${Math.round((p.progress ?? 0) * 100)}%`;
          loadStatus.textContent =
            `ready on ${modelRows.find((r) => r.id === loadedModel)?.name} — ${p.text ?? "downloading upgrade…"}`;
        }
      : (p) => {
          loadStatus.textContent = p.text ?? "loading…";
          progressFill.style.width = `${Math.round((p.progress ?? 0) * 100)}%`;
        };
    const bigEngine = await loadEngine(best.id, bigStatus);
    activate(bigEngine, best.id, `loaded: ${best.name}`);
    progressFill.style.width = "100%";
  } catch (err) {
    // Upgrade failed (storage quota, network, OOM) — keep serving on the
    // quick model rather than tearing everything down.
    const current = modelRows.find((r) => r.id === loadedModel);
    loadStatus.textContent = engine && current
      ? `upgrade failed (${String(err).slice(0, 60)}…) — continuing on ${current.name}`
      : `load failed: ${err}`;
    loadBtn.hidden = false;
  }
}

ddBtn.addEventListener("click", () => { ddMenu.hidden = !ddMenu.hidden; });
document.addEventListener("click", (e) => {
  if (!$("model-dd").contains(e.target)) ddMenu.hidden = true;
});
buildModelPicker();

// ---- WebGPU availability ----
if (!navigator.gpu) {
  loadStatus.textContent = "WebGPU not available in this browser — use the server model.";
  loadBtn.disabled = true;
  ddBtn.disabled = true;
  document.querySelector('input[value="server"]').checked = true;
}

// ---- server health ----
let serverAvailable = false;
fetch("/api/health")
  .then((r) => r.json())
  .then(({ routes, ollama }) => {
    serverAvailable = !!routes?.convert;
    if (routes?.convert) {
      const short = (m) => m.split("/").pop();
      $("server-hint").textContent =
        `(${routes.convert.kind} · ${short(routes.convert.model)})`;
    } else {
      $("server-hint").textContent = ollama?.reachable
        ? `(Ollama · ${ollama.model})`
        : "(no backend reachable)";
    }
  })
  .catch(() => {});

// ---- load browser model ----
// Manual pick from the dropdown (or retry after a failed ladder).
async function loadPicked(row) {
  loadBtn.hidden = true;
  try {
    const eng = await loadEngine(row.id, (p) => {
      loadStatus.textContent = p.text ?? "loading…";
      progressFill.style.width = `${Math.round((p.progress ?? 0) * 100)}%`;
    });
    activate(eng, row.id, `loaded: ${row.name}`);
    progressFill.style.width = "100%";
  } catch (err) {
    loadStatus.textContent = `load failed: ${err}`;
    loadBtn.hidden = false;
  }
}

loadBtn.addEventListener("click", () => {
  const row = modelRows.find((r) => r.id === selectedModelId);
  if (row) loadPicked(row);
});

// ---- conversion ----
function stripThink(text) {
  return text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
}
function stripFences(text) {
  const m = text.match(/^```(?:latex|tex)?\s*\n([\s\S]*?)\n?```\s*$/);
  return m ? m[1].trim() : text.trim();
}

// Output is complete when it ends on a closing math delimiter and every
// segment parses — at that point any further tokens are the model rambling.
function looksComplete(text) {
  const t = text.trim();
  if (t.length < 4) return false;
  if (!/(\\\]|\\\))\s*$/.test(t) && !(/\$\$\s*$/.test(t) && (t.match(/\$\$/g) || []).length % 2 === 0)) {
    return false;
  }
  return checkSyntax(t).length === 0;
}

async function streamBrowserChat(messages, onDelta) {
  if (!engine) throw new Error("Load the browser model first.");
  // LaTeX output is rarely longer than ~2x the input; don't let a small
  // model ramble for 1024 tokens on a one-line equation.
  const inputChars = messages[messages.length - 1].content.length;
  const maxTokens = Math.min(1024, Math.max(256, Math.ceil(inputChars / 3) * 2 + 128));
  const chunks = await engine.chat.completions.create({
    messages,
    temperature: 0.2,
    max_tokens: maxTokens,
    stream: true,
    extra_body: { enable_thinking: false },
  });
  let full = "";
  let sinceCheck = 0;
  for await (const c of chunks) {
    const delta = c.choices[0]?.delta?.content ?? "";
    if (delta) {
      full += delta;
      onDelta?.(full);
      // Early exit: check completeness at most every few tokens.
      sinceCheck += 1;
      if (sinceCheck >= 4) {
        sinceCheck = 0;
        if (looksComplete(full)) engine.interruptGenerate();
      }
    }
  }
  const name = modelRows.find((r) => r.id === loadedModel)?.name ?? loadedModel;
  return { latex: stripFences(stripThink(full)), model: name };
}

// Read the server's NDJSON stream: {delta} lines, then {done, latex, model, ms}.
async function streamServerChat(url, body, onDelta) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const data = await r.json().catch(() => ({}));
    throw new Error(data.error || `server error ${r.status}`);
  }
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = "", full = "", final = null;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i);
      buf = buf.slice(i + 1);
      if (!line.trim()) continue;
      const j = JSON.parse(line);
      if (j.done) final = j;
      else if (j.delta) {
        full += j.delta;
        onDelta?.(full);
      }
    }
  }
  if (!final) throw new Error("stream ended unexpectedly");
  return final;
}

const convertBrowser = (text, onDelta) =>
  streamBrowserChat(
    [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: text },
    ],
    onDelta
  );

// One validator-guided repair attempt: give the model its own output plus
// the exact check failures. Cheap (local), and the only fix path when no
// server escalation exists.
const repairBrowser = (text, badLatex, issues, onDelta) =>
  streamBrowserChat(
    [
      { role: "system", content: REPAIR_PROMPT },
      { role: "user", content: `Convert to LaTeX:\n${text}` },
      { role: "assistant", content: badLatex },
      { role: "user", content: `Checks failed:\n- ${issues.join("\n- ")}\nOutput the corrected LaTeX.` },
    ],
    onDelta
  );
window.__repairBrowser = repairBrowser; // debugging hook

// Validator-guided repair loop: up to 5 turns, but only while each attempt
// produces a DIFFERENT error than the last — the same error twice means the
// model is stuck, so stop rather than burn turns. Every attempt streams into
// the output box and reports via onAttempt so the user sees it trying.
const MAX_REPAIR_ATTEMPTS = 5;
async function repairLoop(contextText, latex, issues, validateFn, onDelta, onAttempt) {
  let prevSig = issues.join("|");
  let current = latex;
  for (let attempt = 1; attempt <= MAX_REPAIR_ATTEMPTS; attempt++) {
    onAttempt?.(attempt, issues);
    const repaired = await repairBrowser(contextText, current, issues, onDelta);
    const v = validateFn(repaired.latex);
    if (v.ok) return { ok: true, latex: repaired.latex, attempts: attempt };
    const sig = v.issues.join("|");
    if (sig === prevSig) return { ok: false, latex: repaired.latex, attempts: attempt, issues: v.issues, stuck: true };
    prevSig = sig;
    current = repaired.latex;
    issues = v.issues;
  }
  return { ok: false, latex: current, attempts: MAX_REPAIR_ATTEMPTS, issues };
}
window.__repairLoop = repairLoop; // debugging hook

const convertServer = (text, onDelta) =>
  streamServerChat("/api/convert", { text, complex: !specialistEligible(text) }, onDelta);

function renderPreview(latex) {
  // If the model returned bare math with no delimiters, wrap it for display.
  const hasDelims = /\\\[|\\\(|\$\$|(?:^|[^\\])\$/.test(latex);
  preview.textContent = hasDelims ? latex : `\\[${latex}\\]`;
  renderMathInElement(preview, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
      { left: "$", right: "$", display: false },
    ],
    throwOnError: false,
  });
}

const checksEl = $("checks");

function showChecks(validation, note) {
  if (validation.ok) {
    checksEl.className = "checks ok";
    checksEl.textContent = note ? `✓ checks passed ${note}` : "✓ syntax valid · matches input";
  } else {
    checksEl.className = "checks warn";
    checksEl.textContent = `⚠ ${validation.issues.join(" · ")}`;
  }
}

convertBtn.addEventListener("click", async () => {
  const text = $("input").value.trim();
  if (!text) return;
  const engineChoice = document.querySelector('input[name="engine"]:checked').value;
  convertBtn.disabled = true;
  convertStatus.textContent = "converting…";
  checksEl.textContent = "";
  const started = performance.now();
  try {
    let result, validation, escalated = false;
    const liveOutput = (partial) => { outputCode.textContent = partial; };
    outputCode.textContent = "";

    // Tier 0: the local specialist, when the input is in its wheelhouse.
    if (specialist && specialistEligible(text)) {
      try {
        const r0 = await convertSpecialist(text);
        const v0 = validateLatex(text, r0.latex);
        if (v0.ok) {
          result = r0;
          validation = v0;
          outputCode.textContent = r0.latex;
        }
      } catch { /* specialist failed — fall through to the general model */ }
    }

    let note = "";
    // Benchmark-driven routing: prose/multiline inputs score 0% on browser
    // models below 4B-class, so don't waste a hop on them.
    const loadedRow = modelRows.find((r) => r.id === loadedModel);
    const complexInput = !specialistEligible(text);

    if (result) {
      // specialist answered; nothing more to do
    } else if (engineChoice === "browser" && complexInput && !loadedRow?.multiline && serverAvailable) {
      convertStatus.textContent = "long or mixed input — using server model…";
      result = await convertServer(text, liveOutput);
      validation = validateLatex(text, result.latex);
      note = "(routed to server — loaded browser model is too small for prose passages)";
    } else if (engineChoice === "browser") {
      result = await convertBrowser(text, liveOutput);
      validation = validateLatex(text, result.latex);
      if (!validation.ok) {
        try {
          const r = await repairLoop(
            text, result.latex, validation.issues,
            (l) => validateLatex(text, l),
            liveOutput,
            (attempt, issues) => {
              convertStatus.textContent = `repair attempt ${attempt}/${MAX_REPAIR_ATTEMPTS} — ${issues[0]}…`;
              showChecks({ ok: false, issues }, "");
            }
          );
          result = { ...result, latex: r.latex };
          if (r.ok) {
            result.model += ` (self-corrected ×${r.attempts})`;
            validation = { ok: true, issues: [] };
            note = `(self-corrected after ${r.attempts} attempt${r.attempts > 1 ? "s" : ""})`;
          } else {
            validation = { ok: false, issues: r.issues };
          }
        } catch { /* repair failed — fall through to escalation */ }
      }
      if (!validation.ok && serverAvailable) {
        // Browser model came up short — escalate to the server model.
        convertStatus.textContent = `browser model failed checks (${validation.issues[0]}…) — escalating to server…`;
        escalated = true;
        note = "(escalated to server model — browser model output failed checks)";
        outputCode.textContent = "";
        result = await convertServer(text, liveOutput);
        validation = validateLatex(text, result.latex);
      }
    } else {
      result = await convertServer(text, liveOutput);
      validation = validateLatex(text, result.latex);
    }

    outputCode.textContent = result.latex;
    renderPreview(result.latex);
    showChecks(validation, note);
    currentLatex = result.latex;
    currentInput = text;
    chatLog.innerHTML = "";
    setChatEnabled(true);
    const secs = ((performance.now() - started) / 1000).toFixed(1);
    convertStatus.textContent = `${secs}s · ${result.model}${escalated ? " (escalated)" : ""}`;
  } catch (err) {
    convertStatus.textContent = String(err.message || err);
  } finally {
    convertBtn.disabled = false;
  }
});

// ---- refine chat ----
const chatLog = $("chat-log");
const chatInput = $("chat-input");
const chatSend = $("chat-send");
let currentLatex = null;
let currentInput = null;

function addMsg(cls, text) {
  const el = document.createElement("div");
  el.className = `msg ${cls}`;
  el.textContent = text;
  chatLog.appendChild(el);
  el.scrollIntoView({ block: "nearest" });
  return el;
}

function setChatEnabled(on) {
  chatInput.disabled = !on;
  chatSend.disabled = !on;
}

const refineBrowser = (instruction, onDelta) =>
  streamBrowserChat(
    [
      { role: "system", content: REFINE_PROMPT },
      { role: "user", content: `Convert to LaTeX:\n${currentInput}` },
      { role: "assistant", content: currentLatex },
      { role: "user", content: instruction },
    ],
    onDelta
  );

const refineServer = (instruction, onDelta) =>
  streamServerChat("/api/refine", { original: currentInput, latex: currentLatex, instruction }, onDelta);

// A degenerate revision: the model echoed the feedback back instead of
// editing the LaTeX. Compare normalized text.
function isEcho(instruction, revised) {
  const norm = (s) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
  const a = norm(instruction), b = norm(revised);
  return !b || a === b || (b.length > 8 && (a.includes(b) || b.includes(a)));
}

async function sendRefinement() {
  const instruction = chatInput.value.trim();
  if (!instruction || !currentLatex) return;
  const engineChoice = document.querySelector('input[name="engine"]:checked').value;
  chatInput.value = "";
  setChatEnabled(false);
  addMsg("user", instruction);
  const pending = addMsg("note", "revising…");
  try {
    const liveOutput = (partial) => { pending.textContent = partial; };
    let result = engineChoice === "browser"
      ? await refineBrowser(instruction, liveOutput)
      : await refineServer(instruction, liveOutput);

    if (isEcho(instruction, result.latex) && engineChoice === "browser") {
      pending.textContent = "browser model echoed the feedback — retrying with server model…";
      result = await refineServer(instruction, liveOutput);
    }
    pending.remove();

    if (isEcho(instruction, result.latex)) {
      addMsg("note",
        "The model returned your feedback instead of edited LaTeX, so I kept the previous version. " +
        'Try stating the change directly, e.g. "change TV to T(V)".');
      return;
    }

    currentLatex = result.latex;
    addMsg("model", result.latex);
    outputCode.textContent = result.latex;
    renderPreview(result.latex);
    showChecks(validateLatex(currentInput, result.latex), false);
  } catch (err) {
    pending.className = "msg note";
    pending.textContent = `error: ${err.message || err}`;
  } finally {
    setChatEnabled(true);
    chatInput.focus();
  }
}

chatSend.addEventListener("click", sendRefinement);
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendRefinement();
});

$("copy-btn").addEventListener("click", () => {
  navigator.clipboard.writeText(outputCode.textContent);
});

// Cmd/Ctrl+Enter converts
$("input").addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") convertBtn.click();
});

// ---- character counter ----
const INPUT_LIMIT = 6000;
const charCount = $("char-count");
$("input").addEventListener("input", () => {
  const len = $("input").value.length;
  charCount.textContent = `${len.toLocaleString()} / ${INPUT_LIMIT.toLocaleString()}`;
  charCount.classList.toggle("near-limit", len >= INPUT_LIMIT * 0.9);
});
