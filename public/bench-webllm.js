// Head-to-head bench: stock WebLLM vs the catalog's patched runtimes for one
// Qwen3 model. Interleaves arms within each round (stock, v1, v2, stock, v1,
// v2, …), a fresh worker+engine per arm per round, and posts rows to
// /api/bench like bench.js (best-effort — the static build has no server).
//
// Query params:
//   model   default "Qwen3-1.7B-q4f16_1-MLC"
//   arms    comma list; default "stock" + every catalog variant for `model`
//   rounds  default 2
//   items   comma list of bench-data.json ids; default all
//   stream  default 1 (0 = non-streaming chat.completions.create)
//   post    default 1 (0 = don't POST rows to api/bench)
//
// Example: bench-webllm.html?model=Qwen3-0.6B-q4f16_1-MLC&arms=stock,sg32-burst4,sg32-burst4-flush64&rounds=2

import * as webllm from "./vendor/webllm/index.js";
import { loadCatalog, planEngine } from "./qwen3-webllm.js";

// Same prompts as bench.js — copied, not imported (importing bench.js would
// run its own main()).
const SYSTEM_PROMPT = `You are a text-to-LaTeX transcriber. Convert the user's input (plain-language math, equations, or prose with math) into LaTeX.

Rules:
- Output ONLY the LaTeX code. No explanations, no markdown code fences, no surrounding commentary.
- Never solve, evaluate, simplify, or answer. If the input is a question or problem, convert the question itself to LaTeX; do not produce the answer.
- For pure math, wrap display math in \\[ ... \\].
- For prose mixed with math, keep the prose as plain text and wrap math in \\( ... \\).
- Use standard LaTeX/amsmath commands only.`;
const CONVERT_USER = (text) => `Convert to LaTeX. Do not solve or answer.\n${text}`;

const stripThink = (t) => t.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
const stripFences = (t) => {
  const m = t.match(/^```(?:latex|tex)?\s*\n([\s\S]*?)\n?```\s*$/);
  return m ? m[1].trim() : t.trim();
};

const log = (cls, msg) => {
  const el = document.createElement("div");
  el.className = cls; el.textContent = msg;
  document.getElementById("log").prepend(el);
};
const progress = (msg) => { document.getElementById("progress").textContent = msg; };
const errText = (e) => {
  if (e instanceof Error) return `${e.name}: ${e.message}`;
  if (typeof e === "string") return e;
  // WebLLM worker rejections arrive as plain objects over postMessage, so
  // String(e) would be "[object Object]" and lose the whole failure.
  try { return JSON.stringify(e, Object.getOwnPropertyNames(e ?? {})); } catch { return String(e); }
};
const post = (row, enabled) => {
  if (!enabled) return;
  try { fetch("api/bench", { method: "POST", body: JSON.stringify(row) }).catch(() => {}); } catch {}
};

function median(nums) {
  const arr = nums.filter((n) => typeof n === "number" && Number.isFinite(n));
  if (!arr.length) return null;
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}
function min(nums) {
  const arr = nums.filter((n) => typeof n === "number" && Number.isFinite(n));
  return arr.length ? Math.min(...arr) : null;
}
const fmt = (n, d = 1) => (n == null ? "—" : n.toFixed(d));

function parseParams() {
  const qp = new URLSearchParams(location.search);
  const model = qp.get("model") || "Qwen3-1.7B-q4f16_1-MLC";
  const rounds = Math.max(1, parseInt(qp.get("rounds") ?? "2", 10) || 2);
  const stream = (qp.get("stream") ?? "1") !== "0";
  const post = (qp.get("post") ?? "1") !== "0";
  const armsParam = qp.get("arms");
  const itemsParam = qp.get("items");
  return { model, rounds, stream, post, armsParam, itemsParam };
}

async function genWebllm(engine, item, stream) {
  const messages = [{ role: "system", content: SYSTEM_PROMPT }, { role: "user", content: CONVERT_USER(item.input) }];
  const t0 = performance.now();
  if (stream) {
    const chunks = await engine.chat.completions.create({
      messages, temperature: 0, max_tokens: 768, stream: true,
      stream_options: { include_usage: true },
      extra_body: { enable_thinking: false },
    });
    let full = ""; let ttftMs = null; let usage = null;
    for await (const c of chunks) {
      const delta = c.choices?.[0]?.delta?.content;
      if (delta) {
        full += delta;
        if (ttftMs === null) ttftMs = performance.now() - t0;
      }
      if (c.usage) usage = c.usage;
    }
    const ms = performance.now() - t0;
    const completionTokens = usage?.completion_tokens ?? 0;
    const decodeMsPerTok = ttftMs !== null
      ? (ms - ttftMs) / Math.max(1, completionTokens - 1)
      : ms / Math.max(1, completionTokens);
    return {
      ms: Math.round(ms), ttftMs: ttftMs === null ? null : Math.round(ttftMs),
      completionTokens, decodeMsPerTok, usageExtra: usage?.extra ?? null,
      output: stripFences(stripThink(full)),
    };
  }
  const resp = await engine.chat.completions.create({
    messages, temperature: 0, max_tokens: 768, stream: false,
    extra_body: { enable_thinking: false },
  });
  const ms = performance.now() - t0;
  const completionTokens = resp.usage?.completion_tokens ?? 0;
  const decodeMsPerTok = ms / Math.max(1, completionTokens);
  return {
    ms: Math.round(ms), ttftMs: null, completionTokens, decodeMsPerTok,
    usageExtra: resp.usage?.extra ?? null,
    output: stripFences(stripThink(resp.choices?.[0]?.message?.content ?? "")),
  };
}

async function buildPlan(model, catalog, stockRecord, arm) {
  const force = arm === "stock" ? "stock" : { variant: arm };
  return planEngine(model, catalog, { force, stockRecord });
}

async function loadArmEngine(model, plan, onProgress) {
  const worker = new Worker(plan.workerUrl, { type: "module" });
  const eng = await webllm.CreateWebWorkerMLCEngine(worker, model, {
    initProgressCallback: onProgress,
    appConfig: plan.appConfig,
  }, { context_window_size: 2048 });
  return { eng, worker };
}

// ---- live summary table ----
const summaryBody = document.querySelector("#summary tbody");
const armRowEl = new Map();
function summarizeArm(rows, approach) {
  const own = rows.filter((r) => r.approach === approach && r.item !== "__load__" && !r.error);
  const loads = rows.filter((r) => r.approach === approach && r.item === "__load__" && !r.error);
  const decodeVals = own.map((r) => r.decodeMsPerTok);
  const tokPerSec = own
    .filter((r) => r.completionTokens > 0 && r.ms > 0)
    .map((r) => r.completionTokens / (r.ms / 1000));
  const ttftVals = own.map((r) => r.ttftMs).filter((v) => v != null);
  const identicalTotal = own.filter((r) => r.identicalToStock !== null).length;
  const identicalCount = own.filter((r) => r.identicalToStock === true).length;
  const totalMs = own.reduce((a, r) => a + (r.ms || 0), 0) + loads.reduce((a, r) => a + (r.ms || 0), 0);
  return {
    loadMs: median(loads.map((r) => r.ms)),
    medDecode: median(decodeVals),
    minDecode: min(decodeVals),
    medTokPerSec: median(tokPerSec),
    medTtft: ttftVals.length ? median(ttftVals) : null,
    identical: `${identicalCount}/${identicalTotal}`,
    totalMs,
  };
}
function renderArmSummary(rows, approach, label) {
  const s = summarizeArm(rows, approach);
  let tr = armRowEl.get(approach);
  if (!tr) {
    tr = document.createElement("tr");
    tr.innerHTML = "<td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td>";
    summaryBody.appendChild(tr);
    armRowEl.set(approach, tr);
  }
  const cells = tr.children;
  cells[0].textContent = label;
  cells[1].textContent = fmt(s.loadMs, 0);
  cells[2].textContent = fmt(s.medDecode, 2);
  cells[3].textContent = fmt(s.minDecode, 2);
  cells[4].textContent = fmt(s.medTokPerSec, 1);
  cells[5].textContent = s.medTtft == null ? "—" : fmt(s.medTtft, 0);
  cells[6].textContent = s.identical;
  cells[7].textContent = fmt(s.totalMs, 0);
  return s;
}

async function main() {
  const { model, rounds, stream, post: postEnabled, armsParam, itemsParam } = parseParams();

  progress("loading bench data…");
  let items = await fetch("bench-data.json").then((r) => r.json());
  if (itemsParam) {
    const ids = itemsParam.split(",").map((s) => s.trim()).filter(Boolean);
    const byId = new Map(items.map((i) => [i.id, i]));
    items = ids.map((id) => byId.get(id)).filter(Boolean);
  }
  log("ok", `${items.length} eval items, ${rounds} round(s), stream=${stream ? 1 : 0}, post=${postEnabled ? 1 : 0}`);

  let catalog = null;
  try { catalog = await loadCatalog(); } catch (e) { log("bad", `catalog load failed, stock-only: ${e}`); }

  const stockRecord = webllm.prebuiltAppConfig.model_list.find((m) => m.model_id === model);
  if (!stockRecord) {
    progress(`FATAL: unknown model ${model}`);
    log("err", `no stock model_list entry for ${model}`);
    return;
  }

  const defaultArms = () => {
    const variants = catalog?.models?.[model]?.variants ? Object.keys(catalog.models[model].variants) : [];
    return ["stock", ...variants];
  };
  const arms = armsParam ? armsParam.split(",").map((s) => s.trim()).filter(Boolean) : defaultArms();
  log("ok", `model=${model} arms=${arms.join(",")}`);

  const rows = [];
  // stockOutputs[`${round}:${item}`] = output text, populated once the stock
  // arm has produced it; other arms compare against it.
  const stockOutputs = new Map();

  function record(row) {
    rows.push(row);
    post(row, postEnabled);
    return row;
  }

  for (let round = 0; round < rounds; round++) {
    for (const arm of arms) {
      const approach = `webllm:${model}:${arm}`;
      progress(`round ${round + 1}/${rounds} · ${arm} · planning…`);

      let plan;
      try {
        plan = await buildPlan(model, catalog, stockRecord, arm);
      } catch (e) {
        record({ approach, round, item: "__load__", tier: "meta", ms: -1, ttftMs: null, completionTokens: null, decodeMsPerTok: null, usageExtra: null, output: "", identicalToStock: null, error: errText(e).slice(0, 600) });
        log("err", `${approach} round${round} PLAN FAILED: ${e}`);
        continue;
      }
      if (arm !== "stock" && plan.kind !== "catalog") {
        log("bad", `${approach} round${round} fell back to stock plan: ${plan.why ?? "unknown"}`);
      }

      let eng = null, worker = null;
      const tLoad = performance.now();
      try {
        progress(`round ${round + 1}/${rounds} · ${arm} · loading…`);
        ({ eng, worker } = await loadArmEngine(model, plan, (p) => progress(`${arm}: ${p.text?.slice(0, 90) ?? ""}`)));
        // Warm-up: one 2-token greedy request pays shader-compile cost.
        await eng.chat.completions.create({ messages: [{ role: "user", content: "hi" }], max_tokens: 2, temperature: 0, extra_body: { enable_thinking: false } });
        const loadMs = Math.round(performance.now() - tLoad);
        record({ approach, round, item: "__load__", tier: "meta", ms: loadMs, ttftMs: null, completionTokens: null, decodeMsPerTok: null, usageExtra: null, output: "", identicalToStock: null, planLabel: plan.label, planKind: plan.kind });
        log("ok", `${approach} round${round} loaded in ${loadMs}ms (${plan.label})`);
      } catch (e) {
        record({ approach, round, item: "__load__", tier: "meta", ms: -1, ttftMs: null, completionTokens: null, decodeMsPerTok: null, usageExtra: null, output: "", identicalToStock: null, error: errText(e).slice(0, 600) });
        log("err", `${approach} round${round} LOAD FAILED: ${e}`);
        try { if (eng) await eng.unload(); } catch {}
        try { if (worker) worker.terminate(); } catch {}
        continue;
      }

      try {
        for (const item of items) {
          progress(`round ${round + 1}/${rounds} · ${arm} · ${item.id}`);
          try {
            const res = await genWebllm(eng, item, stream);
            const key = `${round}:${item.id}`;
            let identicalToStock;
            if (arm === "stock") {
              stockOutputs.set(key, res.output);
              identicalToStock = true;
            } else if (stockOutputs.has(key)) {
              identicalToStock = res.output === stockOutputs.get(key);
            } else {
              identicalToStock = null;
            }
            record({ approach, round, item: item.id, tier: item.tier, ...res, identicalToStock });
            const tok = res.completionTokens > 0 && res.ms > 0 ? (res.completionTokens / (res.ms / 1000)).toFixed(1) : "?";
            log(identicalToStock === false ? "bad" : "ok", `${approach} round${round} ${item.id} ${res.ms}ms ttft=${res.ttftMs ?? "—"} ${tok}tok/s${identicalToStock === false ? " DIFF" : ""}`);
          } catch (e) {
            record({ approach, round, item: item.id, tier: item.tier, ms: -1, ttftMs: null, completionTokens: null, decodeMsPerTok: null, usageExtra: null, output: "", identicalToStock: null, error: errText(e).slice(0, 600) });
            log("err", `${approach} round${round} ${item.id} FAILED: ${e}`);
          }
          renderArmSummary(rows, approach, arm);
        }
      } finally {
        try { await eng.unload(); } catch {}
        try { worker.terminate(); } catch {}
      }
      renderArmSummary(rows, approach, arm);
    }
  }

  const summary = arms.map((arm) => ({ arm, ...summarizeArm(rows, `webllm:${model}:${arm}`) }));
  const params = { model, arms, rounds, items: items.map((i) => i.id), stream, post: postEnabled };
  window.__benchWebllm = { params, rows, summary };
  post({ approach: "__done__", item: "__done__", tier: "meta", ms: 0, output: "" }, postEnabled);
  progress("BENCH COMPLETE");
  log("ok", "all done");
  document.title = "done";
}

document.getElementById("copy-json").addEventListener("click", async () => {
  const text = JSON.stringify(window.__benchWebllm ?? { rows: [], summary: [] }, null, 2);
  try { await navigator.clipboard.writeText(text); log("ok", "copied JSON to clipboard"); }
  catch (e) { log("err", `copy failed: ${e}`); }
});

main().catch((e) => { progress(`FATAL: ${e}`); log("err", `FATAL: ${e}`); });
