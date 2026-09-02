import * as webllm from "/vendor/webllm/index.js";
import {
  pipeline, env as tjsEnv, VisionEncoderDecoderModel, PreTrainedTokenizer, Tensor, cat,
} from "/vendor/transformers/transformers.min.js";
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
const specialistReady = (async () => {
  try {
    tjsEnv.allowRemoteModels = false;
    tjsEnv.allowLocalModels = true;
    tjsEnv.localModelPath = "/models/";
    tjsEnv.backends.onnx.wasm.wasmPaths = "/vendor/ort/";
    tjsEnv.backends.onnx.wasm.numThreads = 1; // multi-threaded ORT hung on load; 1 thread = ~0.9s/conversion
    const p = await pipeline("text2text-generation", "intellitex", { dtype: "q8" });
    await p(`${SPECIALIST_PREFIX}x squared`, { max_new_tokens: 16 }); // warm-up
    specialist = p;
  } catch (err) {
    console.warn("specialist model unavailable:", err);
  }
})();

// ---- image -> LaTeX: two in-browser OCR models, benchmarked into a ladder ----
// Fully client-side: the image is decoded, preprocessed, and OCR'd in the
// tab. It never leaves the browser — there is no image upload path at all.
//
// Tier 0: Texo (FormulaNet, 20M params, ~77MB). Our 18-image benchmark:
//   100% easy, 100% medium after alias canonicalization, ~0.7s/image, robust
//   to tiny fonts and dark backgrounds. No text mode (prose comes back spelled
//   letter-by-letter inside \mathrm) and it mangles matrices.
// Tier 1: Texify (~305MB int8). 100% on hard equations and prose-with-math,
//   but 3-5s/image and it degenerates into repeated lines on very simple
//   images. Loaded lazily — most users never download it.
const IMAGE_MODELS = {
  texo: { name: "Texo", sizeMB: 77 },
  texify: { name: "Texify", sizeMB: 305 },
};
const ocrLoaders = {}; // key -> loading promise
const ocrModels = {};  // key -> async (blob) => latex string

function ocrProgressUI(key) {
  const bar = $("image-progress"), fill = $("image-progress-fill"), label = $("image-drop-label");
  bar.hidden = false;
  const files = new Map();
  const onProgress = (p) => {
    if (p.status !== "progress" || !p.total) return;
    files.set(p.file, { loaded: p.loaded, total: p.total });
    let loaded = 0, total = 0;
    for (const f of files.values()) { loaded += f.loaded; total += f.total; }
    fill.style.width = `${Math.round((loaded / total) * 100)}%`;
    label.textContent =
      `loading ${IMAGE_MODELS[key].name} (image model): ${(loaded / 2 ** 20).toFixed(0)} / ${(total / 2 ** 20).toFixed(0)} MB (one time — cached after this)`;
  };
  return { onProgress, done: () => { bar.hidden = true; fill.style.width = "0"; } };
}

// Texo preprocessing, ported from Texo-web: grayscale -> invert if dark
// background -> crop to ink bbox -> letterbox into 384x384 on black ->
// normalize with UniMERNet mean/std.
const TEXO_MEAN = 0.7931, TEXO_STD = 0.1738, TEXO_SIZE = 384;
async function texoPreprocess(blob) {
  const bmp = await createImageBitmap(blob);
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = c.getContext("2d");
  ctx.fillStyle = "white"; ctx.fillRect(0, 0, c.width, c.height);
  ctx.drawImage(bmp, 0, 0);
  const { data, width, height } = ctx.getImageData(0, 0, c.width, c.height);
  const gray = new Uint8ClampedArray(width * height);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    gray[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
  }
  let dark = 0;
  for (const v of gray) if (v < 200) dark++;
  if (dark >= gray.length - dark) for (let i = 0; i < gray.length; i++) gray[i] = 255 - gray[i];
  let min = 255, max = 0;
  for (const v of gray) { if (v < min) min = v; if (v > max) max = v; }
  let x0 = width, y0 = height, x1 = -1, y1 = -1;
  if (max > min) {
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const n = ((gray[y * width + x] - min) / (max - min)) * 255;
      if (n < 200) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    }
  }
  if (x1 < x0 || y1 < y0) { x0 = 0; y0 = 0; x1 = width - 1; y1 = height - 1; }
  const cw = Math.max(1, x1 - x0), ch = Math.max(1, y1 - y0);
  const gImg = new ImageData(width, height);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    gImg.data[p] = gImg.data[p + 1] = gImg.data[p + 2] = gray[i]; gImg.data[p + 3] = 255;
  }
  const gc = new OffscreenCanvas(width, height);
  gc.getContext("2d").putImageData(gImg, 0, 0);
  const scale = TEXO_SIZE / Math.min(ch, cw);
  let nw = Math.round(cw * scale), nh = Math.round(ch * scale);
  if (nw > TEXO_SIZE || nh > TEXO_SIZE) {
    const r = Math.min(TEXO_SIZE / nw, TEXO_SIZE / nh);
    nw = Math.round(nw * r); nh = Math.round(nh * r);
  }
  const out = new OffscreenCanvas(TEXO_SIZE, TEXO_SIZE);
  const octx = out.getContext("2d");
  octx.fillStyle = "black"; octx.fillRect(0, 0, TEXO_SIZE, TEXO_SIZE);
  octx.drawImage(gc, x0, y0, cw, ch, Math.floor((TEXO_SIZE - nw) / 2), Math.floor((TEXO_SIZE - nh) / 2), nw, nh);
  const od = octx.getImageData(0, 0, TEXO_SIZE, TEXO_SIZE).data;
  const arr = new Float32Array(TEXO_SIZE * TEXO_SIZE);
  for (let i = 0, p = 0; i < arr.length; i++, p += 4) arr[i] = (od[p] / 255 - TEXO_MEAN) / TEXO_STD;
  return arr;
}

// Texo emits KaTeX/MathJax-only aliases (\infin, \rarr, ...) and token-spaced
// output ("x ^ { 2 }"). Map aliases to standard LaTeX so exports compile —
// a fixed alias table, not a guess about content — and tidy the spacing.
const KATEX_ALIASES = {
  infin: "infty", rarr: "rightarrow", larr: "leftarrow", lrarr: "leftrightarrow",
  Rarr: "Rightarrow", Larr: "Leftarrow", Lrarr: "Leftrightarrow", plusmn: "pm",
  empty: "emptyset", isin: "in", sub: "subset", sube: "subseteq", supe: "supseteq",
  sdot: "cdot", lang: "langle", rang: "rangle", real: "Re", image: "Im",
  alef: "aleph", thetasym: "vartheta",
};
function canonicalizeTexo(s) {
  let out = s.replace(/\\([A-Za-z]+)/g, (m, cmd) => (cmd in KATEX_ALIASES ? `\\${KATEX_ALIASES[cmd]}` : m));
  // spelled-out words inside \mathrm / \operatorname: "l i m" -> "lim"
  out = out.replace(/\\(mathrm|operatorname\*?)\s*\{([^{}]*)\}/g, (m, cmd, body) =>
    `\\${cmd}{${body.split("~").map((w) => w.replace(/\s+/g, "")).join(" ")}}`);
  out = out.replace(/\s+/g, " ")
    .replace(/ ([\^_{}()\[\],;])/g, "$1")
    .replace(/([\^_{(\[]) /g, "$1")
    .replace(/~/g, "\\,");
  return out.trim();
}

// Texo has no text mode: prose in the image comes back as words spelled out
// inside \mathrm/\operatorname. Several such words means the image is a text
// passage — Texify's domain. (Operator names like "lim"/"det" are 1-2 words.)
function texoProseSignal(raw) {
  const segs = raw.match(/\\(?:mathrm|operatorname\*?)\s*\{[^{}]*\}/g) || [];
  const words = segs.flatMap((seg) =>
    seg.replace(/\\(?:mathrm|operatorname\*?)\s*\{|\}/g, "").split("~")
      .map((w) => w.replace(/\s+/g, "")).filter((w) => w.length >= 3));
  return words.length >= 3;
}

// Texify sometimes degenerates into one line repeated to the token cap on
// very sparse images — collapse exact repeats.
function dedupeRepeats(s) {
  const parts = s.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
  return [...new Set(parts)].join("\n\n");
}

function loadOcr(key) {
  ocrLoaders[key] ??= (async () => {
    const ui = ocrProgressUI(key);
    try {
      if (key === "texify") {
        const p = await pipeline("image-to-text", "texify", { dtype: "q8", progress_callback: ui.onProgress });
        ocrModels.texify = async (blob) => {
          const url = URL.createObjectURL(blob);
          try {
            const out = await p(url, { max_new_tokens: 384 });
            return dedupeRepeats((out[0]?.generated_text ?? "").trim());
          } finally { URL.revokeObjectURL(url); }
        };
      } else {
        const model = await VisionEncoderDecoderModel.from_pretrained("texo", { dtype: "fp32", progress_callback: ui.onProgress });
        const tokenizer = await PreTrainedTokenizer.from_pretrained("texo");
        ocrModels.texo = async (blob) => {
          const arr = await texoPreprocess(blob);
          const t = new Tensor("float32", arr, [1, 1, TEXO_SIZE, TEXO_SIZE]);
          const outputs = await model.generate({ inputs: cat([t, t, t], 1), max_new_tokens: 512 });
          return tokenizer.batch_decode(outputs, { skip_special_tokens: true })[0].trim();
        };
      }
    } finally { ui.done(); }
  })();
  return ocrLoaders[key];
}

let lastImageBlob = null, lastImageModel = null;

async function convertImage(fileOrBlob, forceModel = null) {
  const drop = $("image-drop"), label = $("image-drop-label"), altBtn = $("ocr-alt-btn");
  drop.classList.add("busy");
  altBtn.hidden = true;
  convertStatus.textContent = "";
  lastImageBlob = fileOrBlob;
  const started = performance.now();
  // Fidelity vs. input text is meaningless for images — only syntax counts.
  const syntaxOnly = (l) => validateLatex("(image)", l).issues.filter((i) => i.startsWith("syntax"));
  try {
    let used = forceModel ?? "texo";
    let latex = "", escalatedWhy = "";
    if (used === "texo") {
      await loadOcr("texo");
      label.textContent = "reading equation from image (Texo)…";
      const raw = await ocrModels.texo(fileOrBlob);
      latex = canonicalizeTexo(raw);
      if (!forceModel) {
        if (texoProseSignal(raw)) escalatedWhy = "image contains prose";
        else if (!latex || syntaxOnly(latex).length) escalatedWhy = "output failed syntax checks";
        if (escalatedWhy) used = "texify";
      }
    }
    if (used === "texify") {
      if (escalatedWhy) label.textContent = `${escalatedWhy} — switching to Texify…`;
      await loadOcr("texify");
      label.textContent = "reading image (Texify)…";
      latex = await ocrModels.texify(fileOrBlob);
    }
    if (!latex) throw new Error("no text recognized in image");
    lastImageModel = used;
    outputCode.textContent = latex;
    renderPreview(latex);
    let syntaxIssues = syntaxOnly(latex);
    let note = `(from image via ${IMAGE_MODELS[used].name}${escalatedWhy ? `, escalated: ${escalatedWhy}` : ""} — processed locally, image never uploaded)`;
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
        if (r.ok) note = `(from image via ${IMAGE_MODELS[used].name} — OCR self-corrected ×${r.attempts} locally, image never uploaded)`;
        outputCode.textContent = latex;
        renderPreview(latex);
      } catch { /* keep OCR output */ }
    }
    showChecks({ ok: syntaxIssues.length === 0, issues: syntaxIssues }, note);
    currentLatex = latex;
    currentInput = "(image)";
    recordHistory("(image)", latex);
    chatLog.innerHTML = "";
    setChatEnabled(true);
    const other = used === "texo" ? "texify" : "texo";
    altBtn.textContent = `looks wrong? read image with ${IMAGE_MODELS[other].name} instead`;
    altBtn.hidden = false;
    const secs = ((performance.now() - started) / 1000).toFixed(1);
    convertStatus.textContent = `${secs}s · ${IMAGE_MODELS[used].name} · local OCR`;
  } catch (err) {
    convertStatus.textContent = `image conversion failed: ${err.message || err}`;
  } finally {
    drop.classList.remove("busy");
    label.innerHTML = "Drop / paste / <u>choose</u> an image of an equation — processed entirely in your browser, never uploaded";
  }
}

$("ocr-alt-btn").addEventListener("click", () => {
  if (lastImageBlob) convertImage(lastImageBlob, lastImageModel === "texo" ? "texify" : "texo");
});

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
    if (llmEnabled()) {
      startModelLadder(best);
    } else {
      loadBtn.hidden = false;
      loadStatus.textContent = "On-device language model is off (specialist only). Turn it on in settings.";
      $("model-status").textContent = "On-device model: specialist only";
    }
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
  const eng = await webllm.CreateWebWorkerMLCEngine(
    new Worker("/webllm-worker.js", { type: "module" }),
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
  const row = modelRows.find((r) => r.id === modelId);
  const ms = $("model-status");
  if (ms) ms.textContent = `On-device model: ${row?.name ?? modelId}${row && modelRows.find((r) => r.canRun) === row ? " (auto-selected for this device)" : ""}`;
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
specialistReady.then(buildModelPicker);

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
const REPAIR_BUDGET_MS = 8000;
async function repairLoop(contextText, latex, issues, validateFn, onDelta, onAttempt, opts = {}) {
  let prevSig = issues.join("|");
  let current = latex;
  const started = performance.now();
  // A reachable server answers in ~0.5-1s; more than one on-device repair
  // attempt is slower than just escalating.
  const maxAttempts = opts.maxAttempts ?? (serverAvailable ? 1 : MAX_REPAIR_ATTEMPTS);
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    if (attempt > 1 && performance.now() - started > REPAIR_BUDGET_MS) {
      return { ok: false, latex: current, attempts: attempt - 1, issues, budget: true };
    }
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
  return { ok: false, latex: current, attempts: maxAttempts, issues };
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

// Validation results render as status chips (design: trust signals, not a
// log line). A note becomes a muted info chip.
function showChecks(validation, note) {
  checksEl.innerHTML = "";
  const chip = (cls, label) => {
    const s = document.createElement("span");
    s.className = cls;
    const dot = document.createElement("i");
    s.appendChild(dot);
    s.appendChild(document.createTextNode(label));
    checksEl.appendChild(s);
  };
  if (validation.ok) {
    chip("ok", "Syntax valid");
    if (!String(currentInput).startsWith("(") && !String(note).includes("image")) chip("ok", "Matches your input");
  } else {
    for (const issue of validation.issues.slice(0, 4)) chip("warn", issue.replace(/^(syntax|missing):\s*/, (m) => m));
    if (validation.issues.length > 4) chip("warn", `+${validation.issues.length - 4} more`);
  }
  if (note) chip("info", note.replace(/^\(|\)$/g, ""));
}

convertBtn.addEventListener("click", async () => {
  const text = $("input").value.trim();
  if (!text) return;
  if ($("input-card").dataset.mode === "latex") { await checkLatexMode(text); return; }
  const engineChoice = document.querySelector('input[name="engine"]:checked').value;
  convertBtn.disabled = true;
  convertStatus.textContent = "converting…";
  checksEl.textContent = "";
  const started = performance.now();
  try {
    let result, validation, escalated = false;
    const liveOutput = (partial) => { outputCode.textContent = partial; };
    outputCode.textContent = "";

    // Batch: several lines, each a single equation -> one specialist call per
    // line (a structural split on the user's own newlines, not a heuristic).
    let batchNote = "";
    const lines = text.split(/\n+/).map((l) => l.trim()).filter(Boolean);
    if (specialist && lines.length >= 2 && lines.every(specialistEligible)) {
      const outs = [];
      let allOk = true;
      for (const [i, line] of lines.entries()) {
        convertStatus.textContent = `converting line ${i + 1}/${lines.length}…`;
        const r = await convertSpecialist(line);
        const v = validateLatex(line, r.latex);
        if (!v.ok) { allOk = false; break; }
        outs.push(r.latex);
        outputCode.textContent = outs.join("\n\n");
      }
      if (allOk) {
        result = { latex: outs.join("\n\n"), model: `IntelliTeX · specialist × ${lines.length}` };
        validation = { ok: true, issues: [] };
        batchNote = `(${lines.length} equations, one per line)`;
      } else {
        outputCode.textContent = "";
      }
    }

    // Tier 0: the local specialist, when the input is in its wheelhouse.
    if (!result && specialist && specialistEligible(text)) {
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

    let note = batchNote;
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
    } else if (engineChoice === "browser" && !engine && serverAvailable) {
      // No on-device LLM loaded (Quick mode) and the specialist declined:
      // go straight to the server instead of failing.
      convertStatus.textContent = "no on-device model loaded — using server model…";
      result = await convertServer(text, liveOutput);
      validation = validateLatex(text, result.latex);
      note = "(routed to server — no on-device language model loaded)";
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
              convertStatus.textContent = `repair attempt ${attempt}${serverAvailable ? "" : `/${MAX_REPAIR_ATTEMPTS}`} — ${issues[0]}…`;
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

    const strictPending = validation.ok && strictMode() && !batchNote;
    outputCode.textContent = result.latex;
    renderPreview(result.latex);
    showChecks(validation, note);
    currentLatex = result.latex;
    currentInput = text;
    recordHistory(text, result.latex);
    if (strictPending) backgroundJudge(text, result.latex);
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
    const useBrowser = engineChoice === "browser" && !!engine;
    if (!useBrowser && !serverAvailable) throw new Error("No model available to refine with: load an on-device model in settings.");
    let result = useBrowser
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
    recordHistory(currentInput, result.latex);
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

// (copy handling lives in the copy-formats module below)

// ---- direct manual editing of the LaTeX box ----
// No LLM involved: re-render the preview and re-run checks live as the user
// types (debounced). The edited version becomes the current LaTeX, so copy
// and the refine chat both operate on it.
let editDebounce = null;
outputCode.addEventListener("input", () => {
  if (convertBtn.disabled) return; // a conversion is streaming into the box
  clearTimeout(editDebounce);
  editDebounce = setTimeout(() => {
    const latex = outputCode.textContent;
    currentLatex = latex;
    currentInput ??= "(manually entered LaTeX)";
    if (!latex.trim()) { checksEl.textContent = ""; preview.innerHTML = ""; return; }
    renderPreview(latex);
    // Fidelity against the original text is meaningless once the user is
    // hand-editing toward what THEY want — check syntax only.
    const issues = checkSyntax(latex);
    showChecks({ ok: issues.length === 0, issues }, "(manually edited)");
    setChatEnabled(true);
  }, 300);
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

// ---- redesign chrome: segmented Text/Image, settings drawer, upload button ----
{
  const inputCard = $("input-card");
  for (const seg of document.querySelectorAll(".seg")) {
    seg.addEventListener("click", () => {
      document.querySelectorAll(".seg").forEach((s) => s.classList.toggle("active", s === seg));
      inputCard.dataset.mode = seg.dataset.mode;
      const latexMode = seg.dataset.mode === "latex";
      convertBtn.textContent = latexMode ? "Check" : "Convert";
      $("input").placeholder = latexMode
        ? "Paste LaTeX to check, e.g. \\frac{a}{b^2 + c"
        : "e.g. the integral from 0 to infinity of e to the minus x squared dx equals square root of pi over 2";
      if (seg.dataset.mode !== "image") $("input").focus();
    });
  }
  $("upload-btn").addEventListener("click", () => $("image-file").click());

  const settings = $("settings");
  const settingsBtn = $("settings-btn");
  const openSettings = (open) => {
    settings.hidden = !open;
    settingsBtn.setAttribute("aria-expanded", String(open));
  };
  settingsBtn.addEventListener("click", () => openSettings(settings.hidden));
  $("settings-close").addEventListener("click", () => openSettings(false));
  $("change-model").addEventListener("click", () => { openSettings(true); ddMenu.hidden = false; });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") openSettings(false); });
  document.addEventListener("click", (e) => {
    if (!settings.hidden && !settings.contains(e.target) && !settingsBtn.contains(e.target) && e.target !== $("change-model")) openSettings(false);
  });
}


// =====================================================================
// Feature batch: settings/consent, strict-mode judge, history/favorites,
// copy formats, toast.
// =====================================================================

// ---- settings persisted in localStorage ----
const PREFS_KEY = "latexgen.prefs";
function prefs() { try { return JSON.parse(localStorage.getItem(PREFS_KEY) || "{}"); } catch { return {}; } }
function setPref(k, v) { const p = prefs(); p[k] = v; try { localStorage.setItem(PREFS_KEY, JSON.stringify(p)); } catch {} }
function llmEnabled() { return prefs().consent === "full"; }
function strictMode() { return !!prefs().strict; }

function toast(msg, ms = 1800) {
  const t = $("toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(t._timer);
  t._timer = setTimeout(() => { t.hidden = true; }, ms);
}

// ---- first-visit consent ----
{
  const consent = $("consent");
  if (!prefs().consent) consent.hidden = false;
  for (const btn of consent.querySelectorAll(".consent-opt")) {
    btn.addEventListener("click", () => {
      setPref("consent", btn.dataset.consent);
      consent.hidden = true;
      $("llm-enabled").checked = llmEnabled();
      if (llmEnabled() && !engine) {
        const best = modelRows.find((r) => r.canRun);
        if (best) { loadBtn.hidden = true; startModelLadder(best); }
      }
    });
  }
  const llmBox = $("llm-enabled");
  llmBox.checked = llmEnabled();
  llmBox.addEventListener("change", () => {
    setPref("consent", llmBox.checked ? "full" : "quick");
    if (llmBox.checked && !engine) {
      const best = modelRows.find((r) => r.canRun);
      if (best) { loadBtn.hidden = true; startModelLadder(best); }
    }
  });
  const strictBox = $("strict-mode");
  strictBox.checked = strictMode();
  strictBox.addEventListener("change", () => setPref("strict", strictBox.checked));
}

// ---- strict mode: a second model judges input/LaTeX fidelity ----
const JUDGE_PROMPT = `You verify text-to-LaTeX conversions. Given the user's plain-English input and the produced LaTeX, decide whether the LaTeX expresses exactly what the text describes (same operations, grouping, exponents, limits, variables). Reply with ONLY a JSON object: {"ok": true/false, "reason": "<max 12 words>"}`;
async function strictJudge(text, latex) {
  const parse = (s) => {
    try { const j = JSON.parse(s.slice(s.indexOf("{"), s.lastIndexOf("}") + 1)); return { ok: !!j.ok, reason: String(j.reason || "") }; }
    catch { return { ok: true, reason: "" }; } // unparseable verdict never blocks the user
  };
  try {
    if (serverAvailable) {
      const r = await fetch("/api/judge", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text, latex }) });
      if (r.ok) return parse(await r.text());
    }
    if (engine) {
      const reply = await engine.chat.completions.create({
        messages: [{ role: "system", content: JUDGE_PROMPT }, { role: "user", content: `INPUT:\n${text}\n\nLATEX:\n${latex}` }],
        temperature: 0, max_tokens: 80, extra_body: { enable_thinking: false },
      });
      return parse(reply.choices[0].message.content);
    }
  } catch { /* judge unavailable */ }
  return { ok: true, reason: "" };
}

// ---- history and favorites (this browser only) ----
const HISTORY_KEY = "latexgen.history";
const HISTORY_MAX = 100;
function loadHistory() { try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); } catch { return []; } }
function saveHistory(items) { try { localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, HISTORY_MAX))); } catch {} }
function recordHistory(input, latex) {
  if (!latex || !latex.trim()) return;
  const items = loadHistory();
  const dup = items.findIndex((h) => h.latex === latex);
  if (dup !== -1) { const [h] = items.splice(dup, 1); h.ts = Date.now(); items.unshift(h); saveHistory(items); renderHistory(); syncVisualEdit(); return; }
  items.unshift({ id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`, ts: Date.now(), input, latex, fav: false });
  saveHistory(items);
  renderHistory();
  syncVisualEdit();
}
let historyStarredOnly = false;
function renderHistory() {
  const list = $("history-list");
  if (!list) return;
  const items = loadHistory().filter((h) => !historyStarredOnly || h.fav);
  list.innerHTML = "";
  if (!items.length) {
    const e = document.createElement("div"); e.className = "history-empty";
    e.textContent = historyStarredOnly ? "No starred conversions yet." : "Conversions you make will show up here.";
    list.appendChild(e); return;
  }
  for (const h of items) {
    const row = document.createElement("div"); row.className = "hist"; row.tabIndex = 0; row.setAttribute("role", "button");
    const star = document.createElement("button"); star.className = `star${h.fav ? " on" : ""}`; star.textContent = h.fav ? "★" : "☆";
    star.title = h.fav ? "Unstar" : "Star"; star.setAttribute("aria-label", star.title);
    star.addEventListener("click", (e) => { e.stopPropagation(); const all = loadHistory(); const t = all.find((x) => x.id === h.id); if (t) { t.fav = !t.fav; saveHistory(all); renderHistory(); } });
    const body = document.createElement("div"); body.className = "body";
    const code = document.createElement("code"); code.textContent = h.latex;
    const meta = document.createElement("div"); meta.className = "meta";
    meta.textContent = `${new Date(h.ts).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })} · ${h.input === "(image)" ? "from image" : h.input}`;
    body.appendChild(code); body.appendChild(meta);
    row.appendChild(star); row.appendChild(body);
    const restore = () => {
      outputCode.textContent = h.latex; renderPreview(h.latex);
      currentLatex = h.latex; currentInput = h.input;
      if (!String(h.input).startsWith("(")) { $("input").value = h.input; $("input").dispatchEvent(new Event("input")); }
      const issues = checkSyntax(h.latex);
      showChecks({ ok: issues.length === 0, issues }, "(restored from history)");
      setChatEnabled(true); $("history").hidden = true;
      window.scrollTo({ top: 0, behavior: "smooth" });
    };
    row.addEventListener("click", restore);
    row.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); restore(); } });
    list.appendChild(row);
  }
}
{
  const drawer = $("history"), btn = $("history-btn");
  const open = (o) => { drawer.hidden = !o; btn.setAttribute("aria-expanded", String(o)); if (o) { $("settings").hidden = true; renderHistory(); } };
  btn.addEventListener("click", () => open(drawer.hidden));
  $("history-close").addEventListener("click", () => open(false));
  $("history-filter").addEventListener("click", (e) => { historyStarredOnly = !historyStarredOnly; e.currentTarget.setAttribute("aria-pressed", String(historyStarredOnly)); e.currentTarget.textContent = historyStarredOnly ? "Show all" : "Starred only"; renderHistory(); });
  $("history-clear").addEventListener("click", () => { if (confirm("Clear all history in this browser?")) { saveHistory([]); renderHistory(); } });
  document.addEventListener("click", (e) => { if (!drawer.hidden && !drawer.contains(e.target) && !btn.contains(e.target)) open(false); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") open(false); });
}

// ---- copy formats ----
// Single-segment outputs can be re-wrapped; anything else is copied as-is.
function mathBody(latex) {
  const segs = extractSegmentsForCopy(latex);
  return segs.length === 1 && segs[0].isWhole ? segs[0].body : null;
}
function extractSegmentsForCopy(latex) {
  const t = latex.trim();
  const m = t.match(/^(?:\$\$([\s\S]+)\$\$|\\\[([\s\S]+)\\\]|\\\(([\s\S]+)\\\)|\$([^$]+)\$)$/);
  if (m) return [{ body: (m[1] ?? m[2] ?? m[3] ?? m[4]).trim(), isWhole: true }];
  if (!/\$|\\\[|\\\(/.test(t)) return [{ body: t, isWhole: true }]; // bare math
  return [{ body: t, isWhole: false }];
}
async function writeClipboard(items, label) {
  try {
    if (typeof ClipboardItem !== "undefined" && items.some((i) => i.type !== "text/plain")) {
      await navigator.clipboard.write([new ClipboardItem(Object.fromEntries(items.map((i) => [i.type, i.blob ?? new Blob([i.text], { type: i.type })])))]);
    } else {
      await navigator.clipboard.writeText(items.find((i) => i.type === "text/plain")?.text ?? "");
    }
    toast(`Copied ${label}`);
  } catch (err) {
    toast(`Copy failed: ${err.message || err}`, 3000);
  }
}
async function copyAs(kind) {
  const latex = outputCode.textContent.trim();
  if (!latex) { toast("Nothing to copy yet"); return; }
  const body = mathBody(latex);
  if (kind === "display") return writeClipboard([{ type: "text/plain", text: body != null ? `\\[\n${body}\n\\]` : latex }], "display math");
  if (kind === "inline") return writeClipboard([{ type: "text/plain", text: body != null ? `\\(${body}\\)` : latex }], "inline math");
  if (kind === "mathml") {
    const src = body ?? latex;
    let mathml;
    try { mathml = katex.renderToString(src, { output: "mathml", throwOnError: false, displayMode: true }).match(/<math[\s\S]*<\/math>/)?.[0]; } catch {}
    if (!mathml) { toast("Could not build MathML for this LaTeX", 3000); return; }
    return writeClipboard([{ type: "text/html", text: mathml }, { type: "text/plain", text: mathml }], "MathML");
  }
  if (kind === "png") {
    try {
      const blob = await htmlToImage.toBlob(preview, { backgroundColor: "#ffffff", pixelRatio: 2, style: { padding: "16px" } });
      await writeClipboard([{ type: "image/png", blob }, { type: "text/plain", text: latex }], "PNG");
    } catch (err) { toast(`PNG export failed: ${err.message || err}`, 3000); }
    return;
  }
  if (kind === "share") {
    const url = `${location.origin}${location.pathname}#l=${encodeURIComponent(latex)}`;
    return writeClipboard([{ type: "text/plain", text: url }], "share link");
  }
  if (kind === "overleaf") {
    const doc = `\\documentclass{article}\n\\usepackage{amsmath,amssymb}\n\\begin{document}\n${body != null ? `\\[\n${body}\n\\]` : latex}\n\\end{document}\n`;
    const form = document.createElement("form");
    form.method = "POST"; form.action = "https://www.overleaf.com/docs"; form.target = "_blank";
    const inp = document.createElement("input"); inp.type = "hidden"; inp.name = "snip"; inp.value = doc;
    form.appendChild(inp); document.body.appendChild(form); form.submit(); form.remove();
    toast("Opening in Overleaf…");
  }
}
{
  const menu = $("copy-menu"), menuBtn = $("copy-menu-btn");
  $("copy-btn").addEventListener("click", () => {
    const latex = outputCode.textContent.trim();
    if (!latex) { toast("Nothing to copy yet"); return; }
    writeClipboard([{ type: "text/plain", text: latex }], "LaTeX");
  });
  menuBtn.addEventListener("click", () => { menu.hidden = !menu.hidden; menuBtn.setAttribute("aria-expanded", String(!menu.hidden)); });
  for (const b of menu.querySelectorAll("[data-copy]")) b.addEventListener("click", () => { menu.hidden = true; menuBtn.setAttribute("aria-expanded", "false"); copyAs(b.dataset.copy); });
  document.addEventListener("click", (e) => { if (!menu.hidden && !menu.contains(e.target) && e.target !== menuBtn) { menu.hidden = true; menuBtn.setAttribute("aria-expanded", "false"); } });
}


// =====================================================================
// PWA, shared links, Check-LaTeX mode, MathLive visual editor
// =====================================================================

// ---- installable + offline ----
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => navigator.serviceWorker.register("/sw.js").catch((e) => console.warn("sw:", e)));
}

// ---- open a shared link: the LaTeX lives in the URL fragment (never sent to a server) ----
{
  const m = location.hash.match(/^#l=(.+)$/);
  if (m) {
    try {
      const latex = decodeURIComponent(m[1]);
      outputCode.textContent = latex;
      renderPreview(latex);
      const issues = checkSyntax(latex);
      currentLatex = latex; currentInput = "(shared link)";
      showChecks({ ok: issues.length === 0, issues }, "(opened from a shared link)");
      setChatEnabled(true);
      history.replaceState(null, "", location.pathname);
      setTimeout(() => syncVisualEdit(), 0);
    } catch { /* malformed fragment: ignore */ }
  }
}

// ---- Check-LaTeX mode: parse, render, and repair user-supplied LaTeX ----
async function checkLatexMode(latex) {
  convertBtn.disabled = true;
  checksEl.innerHTML = "";
  convertStatus.textContent = "checking…";
  try {
    let issues = checkSyntax(latex);
    let out = latex;
    let note = "(checked — nothing converted, just verified)";
    outputCode.textContent = latex;
    renderPreview(latex);
    if (issues.length && engine) {
      const syntaxOnly = (l) => ({ ok: checkSyntax(l).length === 0, issues: checkSyntax(l) });
      const r = await repairLoop(
        "(user-supplied LaTeX; keep its mathematical meaning)", latex, issues, syntaxOnly,
        (partial) => { outputCode.textContent = partial; },
        (attempt, iss) => { convertStatus.textContent = `repair attempt ${attempt}/${MAX_REPAIR_ATTEMPTS} — ${iss[0]}…`; showChecks({ ok: false, issues: iss }, ""); }
      );
      out = r.latex;
      issues = r.ok ? [] : r.issues;
      if (r.ok) note = `(fixed after ${r.attempts} repair attempt${r.attempts > 1 ? "s" : ""} — compare with what you pasted)`;
      outputCode.textContent = out;
      renderPreview(out);
    }
    currentLatex = out; currentInput = "(latex check)";
    showChecks({ ok: issues.length === 0, issues }, note);
    chatLog.innerHTML = ""; setChatEnabled(true);
    recordHistory("(latex check)", out);
    convertStatus.textContent = issues.length ? "issues found" : "valid LaTeX";
  } finally {
    convertBtn.disabled = false;
  }
}

// ---- MathLive: edit the rendered equation directly; LaTeX follows ----
let mathliveReady = null;
function loadMathlive() {
  mathliveReady ??= import("/vendor/mathlive/mathlive.min.mjs").then((mod) => {
    mod.MathfieldElement.fontsDirectory = "/vendor/mathlive/fonts";
    mod.MathfieldElement.soundsDirectory = null;
    return mod;
  });
  return mathliveReady;
}
function delimitersOf(latex) {
  const t = latex.trim();
  if (/^\$\$[\s\S]*\$\$$/.test(t)) return ["$$", "$$"];
  if (/^\\\[[\s\S]*\\\]$/.test(t)) return ["\\[", "\\]"];
  if (/^\\\([\s\S]*\\\)$/.test(t)) return ["\\(", "\\)"];
  if (/^\$[^$]*\$$/.test(t)) return ["$", "$"];
  return ["", ""];
}
function syncVisualEdit() {
  const btn = $("visual-edit-btn");
  const body = mathBody(outputCode.textContent.trim());
  btn.hidden = body == null; // only single-equation outputs are visually editable
  if (body == null && !$("math-field").hidden) setVisualEdit(false);
}
async function setVisualEdit(on) {
  const btn = $("visual-edit-btn"), mf = $("math-field");
  if (on) {
    await loadMathlive();
    const latex = outputCode.textContent.trim();
    mf.value = mathBody(latex) ?? latex;
    mf._delims = delimitersOf(latex);
    mf.hidden = false; outputCode.hidden = true;
    btn.textContent = "</> LaTeX source"; btn.setAttribute("aria-pressed", "true");
    mf.focus();
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
      const [l, r] = mf._delims ?? ["", ""];
      const body = mf.value;
      const latex = l === "$$" || l === "\\[" ? `${l}\n${body}\n${r}` : `${l}${body}${r}`;
      outputCode.textContent = latex;
      currentLatex = latex;
      renderPreview(latex);
      const issues = checkSyntax(latex);
      showChecks({ ok: issues.length === 0, issues }, "(edited visually)");
    }, 250);
  });
  // keep the button state in sync when the user edits the raw source
  outputCode.addEventListener("input", () => setTimeout(syncVisualEdit, 350));
}


// ---- strict mode, non-blocking: the result is already on screen; a second
// model checks it in the background and offers a fix if it disagrees.
async function backgroundJudge(text, latex) {
  const pending = document.createElement("span");
  pending.className = "info"; pending.innerHTML = "<i></i>second opinion…";
  checksEl.appendChild(pending);
  const j = await strictJudge(text, latex);
  if (outputCode.textContent.trim() !== latex.trim()) { pending.remove(); return; } // user moved on
  pending.remove();
  if (j.ok) {
    const ok = document.createElement("span"); ok.className = "ok"; ok.innerHTML = "<i></i>verified by a second model";
    checksEl.appendChild(ok);
    return;
  }
  const warn = document.createElement("span"); warn.className = "warn"; warn.innerHTML = `<i></i>second opinion: ${j.reason || "disagrees"}`;
  checksEl.appendChild(warn);
  if (!engine && !serverAvailable) return;
  const fix = document.createElement("button"); fix.className = "btn ghost"; fix.textContent = "Apply suggested fix";
  fix.addEventListener("click", async () => {
    fix.disabled = true; fix.textContent = "fixing…";
    try {
      let fixed = null;
      if (engine) {
        const r = await repairLoop(text, latex, [`judge: ${j.reason}`], (l) => validateLatex(text, l),
          (p) => { outputCode.textContent = p; }, null, { maxAttempts: 2 });
        if (r.ok) fixed = r.latex;
      }
      if (!fixed && serverAvailable) {
        const r = await streamServerChat("/api/refine", { original: text, latex, instruction: `A reviewer says: ${j.reason}. Fix exactly that.` }, (p) => { outputCode.textContent = p; });
        fixed = r.latex;
      }
      if (fixed) {
        outputCode.textContent = fixed; renderPreview(fixed); currentLatex = fixed;
        showChecks(validateLatex(text, fixed), "(corrected after a second opinion)");
        recordHistory(text, fixed);
      } else { fix.textContent = "could not fix automatically"; }
    } catch (e) { fix.textContent = `fix failed: ${String(e.message || e).slice(0, 40)}`; }
  });
  checksEl.appendChild(fix);
}
