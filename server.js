// Zero-dependency Node server: serves the static frontend and proxies
// conversion requests to a local Ollama instance (the "server model" path).
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const PORT = process.env.PORT || 8000;
const OLLAMA_URL = process.env.OLLAMA_URL || "http://host.docker.internal:11434";
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "qwen3.8:27b-q8_0";
// Refinement (editing LaTeX per feedback) is a harder task than conversion —
// small models echo feedback or hallucinate edits — so it gets its own model.
const OLLAMA_REFINE_MODEL = process.env.OLLAMA_REFINE_MODEL || OLLAMA_MODEL;
// Optional MLX backend (mlx_lm.server, OpenAI-compatible). On Apple Silicon
// MLX decodes noticeably faster than Ollama, so when it's reachable,
// conversions route to it automatically. Refinement stays on the strongest
// available model.
const MLX_URL = process.env.MLX_URL || "http://host.docker.internal:8080";
const MLX_MODEL = process.env.MLX_MODEL || "mlx-community/Qwen3-1.7B-4bit";
const MLX_REFINE_MODEL = process.env.MLX_REFINE_MODEL || null;

const PUBLIC_DIR = join(fileURLToPath(new URL(".", import.meta.url)), "public");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".json": "application/json",
  ".wasm": "application/wasm",
  ".woff2": "font/woff2",
  ".woff": "font/woff",
  ".ttf": "font/ttf",
  ".png": "image/png",
  ".webmanifest": "application/manifest+json",
};

const SYSTEM_PROMPT = `You are a text-to-LaTeX transcriber. Convert the user's input (plain-language math, equations, or prose with math) into LaTeX.

Rules:
- Output ONLY the LaTeX code. No explanations, no markdown code fences, no surrounding commentary.
- For pure math, wrap display math in \\[ ... \\].
- For prose mixed with math, keep the prose as plain text and wrap math in \\( ... \\).
- Use standard LaTeX/amsmath commands only.`;

const JUDGE_PROMPT = `You verify text-to-LaTeX conversions. Given the user's plain-English input and the produced LaTeX, decide whether the LaTeX expresses exactly what the text describes (same operations, grouping, exponents, limits, variables). Reply with ONLY a JSON object: {"ok": true/false, "reason": "<max 12 words>"}`;

const REFINE_PROMPT = `You are a text-to-LaTeX transcriber in a feedback loop. You previously converted the user's text to LaTeX. The user now gives feedback on your conversion. Produce a corrected version of YOUR PREVIOUS LaTeX.

Rules:
- The user's message is feedback ABOUT the LaTeX — it is never content to transcribe. Never output the feedback text itself.
- Output ONLY the corrected LaTeX. No explanations, no markdown code fences.
- Change only what the feedback concerns; keep everything else, including delimiters, exactly as it was.
- If the feedback is vague, make your best guess at what is wrong and fix that.`;

async function readBody(req) {
  const chunks = [];
  for await (const c of req) chunks.push(c);
  return Buffer.concat(chunks).toString("utf-8");
}

function stripThink(text) {
  // Qwen-family models may emit <think>...</think> blocks; drop them.
  return text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
}

function stripFences(text) {
  const m = text.match(/^```(?:latex|tex)?\s*\n([\s\S]*?)\n?```\s*$/);
  return m ? m[1].trim() : text.trim();
}

// ---- backend probing (cached ~20s so routing adapts if servers start/stop) ----
let probeCache = { at: 0, ollama: false, mlx: false };
async function probeBackends() {
  if (Date.now() - probeCache.at < 20_000) return probeCache;
  const check = (url) =>
    fetch(url, { signal: AbortSignal.timeout(1500) }).then((r) => r.ok).catch(() => false);
  const [ollama, mlx] = await Promise.all([
    check(`${OLLAMA_URL}/api/version`),
    check(`${MLX_URL}/v1/models`),
  ]);
  probeCache = { at: Date.now(), ollama, mlx };
  return probeCache;
}

// Route each task to the best available backend: conversions want speed
// (MLX when present), refinements want the strongest model (Ollama's big
// model, unless an MLX refine model is configured or Ollama is down).
async function routeFor(task) {
  const { ollama, mlx } = await probeBackends();
  if (task === "convert") {
    if (mlx) return { kind: "mlx", model: MLX_MODEL };
    if (ollama) return { kind: "ollama", model: OLLAMA_MODEL };
  } else {
    if (mlx && MLX_REFINE_MODEL) return { kind: "mlx", model: MLX_REFINE_MODEL };
    if (ollama) return { kind: "ollama", model: OLLAMA_REFINE_MODEL };
    if (mlx) return { kind: "mlx", model: MLX_MODEL };
  }
  return null;
}

// Stream a chat completion to the client as NDJSON: {delta} lines while
// generating, then a final {done, latex, model, ms} line. Speaks both
// dialects: Ollama's NDJSON and MLX/OpenAI's SSE.
async function streamChat(res, route, messages) {
  const started = Date.now();
  const isMlx = route.kind === "mlx";
  const r = await fetch(
    isMlx ? `${MLX_URL}/v1/chat/completions` : `${OLLAMA_URL}/api/chat`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(
        isMlx
          ? {
              model: route.model,
              stream: true,
              temperature: 0.2,
              max_tokens: 2048,
              chat_template_kwargs: { enable_thinking: false },
              messages,
            }
          : {
              model: route.model,
              stream: true,
              think: false,
              keep_alive: -1,
              options: { temperature: 0.2, num_predict: 2048 },
              messages,
            }
      ),
      signal: AbortSignal.timeout(120_000),
    }
  );
  if (!r.ok) {
    const detail = await r.text();
    res.writeHead(502, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: `${route.kind} ${r.status}`, detail: detail.slice(0, 500) }));
    return;
  }
  res.writeHead(200, { "content-type": "application/x-ndjson", "cache-control": "no-cache" });
  const decoder = new TextDecoder();
  let buf = "";
  let full = "";
  const emit = (delta) => {
    if (!delta) return;
    full += delta;
    res.write(JSON.stringify({ delta }) + "\n");
  };
  for await (const chunk of r.body) {
    buf += decoder.decode(chunk, { stream: true });
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      let line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (!line) continue;
      if (isMlx) {
        if (!line.startsWith("data:")) continue;
        line = line.slice(5).trim();
        if (line === "[DONE]") continue;
        emit(JSON.parse(line).choices?.[0]?.delta?.content ?? "");
      } else {
        emit(JSON.parse(line).message?.content ?? "");
      }
    }
  }
  const latex = stripFences(stripThink(full));
  res.end(JSON.stringify({ done: true, latex, model: `${route.model} (${route.kind})`, ms: Date.now() - started }) + "\n");
}

// Load both models into memory at boot so the first user request is warm.
function warmModels() {
  for (const model of new Set([OLLAMA_MODEL, OLLAMA_REFINE_MODEL])) {
    fetch(`${OLLAMA_URL}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model, stream: false, think: false, keep_alive: -1,
        options: { num_predict: 1 },
        messages: [{ role: "user", content: "hi" }],
      }),
    }).catch(() => {});
  }
}

const benchRows = [];

// ---- Tab API relay + compute mesh ----
// A browser tab can't accept connections, so tabs long-poll here for jobs and
// post results back. Two lanes per tab: the owner's own requests first, then
// pool jobs from other tabs. The relay forwards bytes only — all inference
// happens in tabs. State is in-memory (one process).
const TAB_ID_RE = /^[a-f0-9]{32}$/;
const tabs = new Map();      // tabId -> tab state
const results = new Map();   // jobId -> { resolve, timer, job }
const JOB_TIMEOUT_MS = 25_000, POLL_TIMEOUT_MS = 30_000, MAX_QUEUE = 8;
const PEER_ATTEMPT_MS = 12_000;   // re-dispatch to another peer after this
const SPOT_CHECK_RATE = 0.1;      // duplicate 1 in 10 pool jobs for agreement scoring
const CLASS_WEIGHT = { equation: 1, prose: 3, refine: 2, check: 1, image: 2 };
const json = (res, code, body) => { res.writeHead(code, { "content-type": "application/json" }); res.end(JSON.stringify(body)); };

function tabState(id) {
  if (!tabs.has(id)) tabs.set(id, {
    id, queue: [], poolQueue: [], waiter: null, lastSeen: 0,
    caps: {}, pool: null, // pool: { keys: [...], images: bool, maxConcurrent: n }
    inFlight: new Set(), servedWeight: 0, usedWeight: 0, servedJobs: 0, usedJobs: 0,
    agree: 0, disagree: 0, failures: 0, latency: {}, // class -> EMA ms
  });
  return tabs.get(id);
}
const isLive = (t) => t && Date.now() - t.lastSeen < 45_000;
function deliver(tab) {
  if (!tab.waiter) return;
  const job = tab.queue.shift() ?? tab.poolQueue.shift();
  if (!job) return;
  const res = tab.waiter; tab.waiter = null;
  tab.inFlight.add(job.jobId);
  job.dispatchedAt = Date.now();
  json(res, 200, job);
}
setInterval(() => { // drop tabs not seen for 5 minutes
  const cutoff = Date.now() - 300_000;
  for (const [id, t] of tabs) if (t.lastSeen < cutoff && !t.waiter) tabs.delete(id);
}, 60_000).unref();

// ---- scheduler ----
const classOf = (body) => body.imageBase64 ? "image" : body.instruction ? "refine" : body.latex && !body.text ? "check"
  : (body.text ?? "").includes("\n") || (body.text ?? "").length > 320 ? "prose" : "equation";
function reputationOk(t) { const n = t.agree + t.disagree; return n < 5 || t.agree / n >= 0.6; }
function canServe(t, cls, requester) {
  if (t === requester || !isLive(t) || !t.pool || !t.waiter && t.inFlight.size >= (t.pool.maxConcurrent || 1)) return false;
  if (t.inFlight.size >= (t.pool.maxConcurrent || 1)) return false;
  if (!t.pool.keys.some((k) => requester.pool.keys.includes(k))) return false;
  if (!reputationOk(t) || t.failures > 5) return false;
  if (cls === "image") return !!t.pool.images && !!(t.caps.texo || t.caps.texify);
  if (cls === "check") return !!t.caps.llm || !!t.caps.specialist;
  if (!t.caps.llm) return false;
  if (cls === "prose" && !t.caps.multiline) return false;
  return true;
}
// Expected finish time: queued work ahead + this job's decode time. Bigger
// models than the job needs get a small penalty so a lone 9B tab isn't the
// default target for every trivial equation.
function scoreFor(t, cls, body) {
  const avg = t.latency[cls] ?? 3000;
  const tokens = cls === "prose" ? 200 : cls === "refine" ? 80 : 60;
  const decode = t.caps.tokPerSec ? (tokens / t.caps.tokPerSec) * 1000 : avg;
  const oversize = (cls === "equation" || cls === "refine") && t.caps.multiline ? 400 : 0;
  return (t.inFlight.size + t.poolQueue.length) * avg + decode + 150 + oversize;
}
function pickPeer(requester, cls, body, exclude = new Set()) {
  const cands = [...tabs.values()].filter((t) => !exclude.has(t.id) && canServe(t, cls, requester));
  if (!cands.length) return null;
  const sample = cands.length <= 2 ? cands : [cands[Math.floor(Math.random() * cands.length)], cands[Math.floor(Math.random() * cands.length)]];
  return sample.reduce((a, b) => (scoreFor(b, cls, body) < scoreFor(a, cls, body) ? b : a));
}
// Good faith: your tab must be listening and pooled, and you can't consume
// much more than you serve (weighted by job class). Small grace for newcomers.
function reciprocityOk(requester) {
  if (!isLive(requester) || !requester.pool) return "your tab is not serving the pool";
  if (requester.usedWeight > requester.servedWeight * 2 + 6) return "contribute more before using peers (served/used ratio)";
  return null;
}
const normLatex = (s) => (s || "").replace(/\s+/g, "");

function dispatchToPeer(peer, job, requester, cls, weight, onDone) {
  const jobId = Math.random().toString(16).slice(2) + Date.now().toString(16);
  const entry = { job: { ...job, jobId, pool: true, noMesh: true, noServer: true }, peer, requester, cls, weight, started: Date.now(), onDone };
  entry.timer = setTimeout(() => { results.delete(jobId); peer.inFlight.delete(jobId); peer.failures++; onDone({ error: "peer did not answer in time", timeout: true }, entry); }, PEER_ATTEMPT_MS);
  results.set(jobId, entry);
  peer.poolQueue.push(entry.job);
  deliver(peer);
  return jobId;
}

async function runOnMesh(requester, body) {
  const cls = classOf(body), weight = CLASS_WEIGHT[cls] ?? 1;
  const tried = new Set();
  const deadline = Date.now() + JOB_TIMEOUT_MS;
  return new Promise((resolve) => {
    const attempt = () => {
      const peer = pickPeer(requester, cls, body, tried);
      if (!peer) return resolve({ error: tried.size ? "peers did not answer" : "no capable peer available", noPeer: true });
      tried.add(peer.id);
      dispatchToPeer(peer, body, requester, cls, weight, (result, entry) => {
        if (result.error && result.timeout && Date.now() + 5000 < deadline) return attempt(); // re-dispatch
        if (!result.error) {
          peer.servedWeight += weight; peer.servedJobs++;
          requester.usedWeight += weight; requester.usedJobs++;
          const ms = Date.now() - entry.started;
          peer.latency[cls] = peer.latency[cls] ? peer.latency[cls] * 0.7 + ms * 0.3 : ms;
          if (Math.random() < SPOT_CHECK_RATE) spotCheck(requester, body, cls, peer, result);
        }
        resolve({ ...result, peer: peer.id.slice(0, 6), peerModel: peer.caps.llmName ?? null });
      });
    };
    attempt();
  });
}
// Duplicate a job to a second peer purely for agreement scoring.
function spotCheck(requester, body, cls, first, firstResult) {
  const second = pickPeer(requester, cls, body, new Set([first.id]));
  if (!second) return;
  dispatchToPeer(second, body, requester, cls, 0, (r2) => {
    if (r2.error) return;
    const same = normLatex(r2.latex) === normLatex(firstResult.latex);
    for (const t of [first, second]) { if (same) t.agree++; else t.disagree++; }
  });
}

const server = createServer(async (req, res) => {
  try {
    if (req.method === "GET" && req.url === "/api/health") {
      const { ollama, mlx } = await probeBackends();
      const [convert, refine] = await Promise.all([routeFor("convert"), routeFor("refine")]);
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({
        ok: true,
        backends: { ollama, mlx },
        routes: { convert, refine },
        serverKind: "local", // self-hosted Ollama/MLX: trusted, fast — used before peers
        mesh: { tabs: [...tabs.values()].filter((t) => isLive(t)).length, pooled: [...tabs.values()].filter((t) => isLive(t) && t.pool).length },
        // kept for older clients
        ollama: { reachable: ollama, model: OLLAMA_MODEL, url: OLLAMA_URL },
      }));
      return;
    }

    let m;
    // --- mesh: a tab registers its capabilities and pool membership ---
    if (req.method === "POST" && (m = req.url.match(/^\/api\/relay\/([a-f0-9]{32})\/caps$/))) {
      const tab = tabState(m[1]);
      const body = JSON.parse(await readBody(req));
      tab.caps = body.caps ?? {};
      tab.pool = body.pool ? { keys: (body.pool.keys ?? ["public"]).map((k) => String(k).slice(0, 64)), images: !!body.pool.images, maxConcurrent: Math.min(3, Math.max(1, body.pool.maxConcurrent ?? 1)) } : null;
      tab.lastSeen = Date.now();
      json(res, 200, { ok: true, served: tab.servedWeight, used: tab.usedWeight, servedJobs: tab.servedJobs, usedJobs: tab.usedJobs, peers: [...tabs.values()].filter((t) => t !== tab && isLive(t) && t.pool && t.pool.keys.some((k) => tab.pool?.keys.includes(k))).length });
      return;
    }
    // --- mesh: a tab (or its owner's client) asks a peer to run a job ---
    if (req.method === "POST" && (m = req.url.match(/^\/api\/pool\/([a-f0-9]{32})\/(convert|refine|check)$/))) {
      const requester = tabs.get(m[1]);
      if (!requester) return json(res, 404, { error: "unknown tab" });
      const why = reciprocityOk(requester);
      if (why) return json(res, 429, { error: why });
      let body;
      try { body = JSON.parse(await readBody(req)); } catch { return json(res, 400, { error: "invalid json" }); }
      if ((body.text ?? "").length > 6000 || (body.latex ?? "").length > 6000) return json(res, 413, { error: "payload too large" });
      if (body.imageBase64 && !requester.pool.images) return json(res, 400, { error: "images are not shared with peers" });
      const out = await runOnMesh(requester, { kind: m[2], ...body });
      json(res, out.error ? (out.noPeer ? 503 : 504) : 200, out);
      return;
    }

    // --- Tab API: external client -> tab ---
    // POST /api/tab/<id>/convert {text} | {imageBase64, mime} | {latex, instruction}
    if (req.method === "POST" && (m = req.url.match(/^\/api\/tab\/([a-f0-9]{32})\/(convert|refine|check|format|status|history)$/))) {
      const [, tabId, kind] = m;
      const tab = tabs.get(tabId);
      if (!tab || Date.now() - tab.lastSeen > 45_000) {
        res.writeHead(404, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: "no tab is listening on this id — open LatexGen with the Tab API enabled" }));
        return;
      }
      if (tab.queue.length >= MAX_QUEUE) {
        res.writeHead(429, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: "tab is busy (queue full)" })); return;
      }
      let body;
      try { body = JSON.parse(await readBody(req)); } catch { res.writeHead(400, { "content-type": "application/json" }); res.end(JSON.stringify({ error: "invalid json" })); return; }
      const bad = (msg) => { res.writeHead(400, { "content-type": "application/json" }); res.end(JSON.stringify({ error: msg })); };
      if (kind === "convert" && !body.text && !body.imageBase64 && !body.latex) return bad("text, imageBase64 or latex required");
      if (kind === "refine" && (!body.latex || !body.instruction)) return bad("latex and instruction required");
      if (kind === "check" && !body.latex) return bad("latex required");
      if (kind === "format" && (!body.latex || !body.format)) return bad("latex and format required");
      if ((body.text ?? "").length > 6000 || (body.imageBase64 ?? "").length > 8_000_000) { res.writeHead(413, { "content-type": "application/json" }); res.end(JSON.stringify({ error: "payload too large" })); return; }
      const jobId = Math.random().toString(16).slice(2) + Date.now().toString(16);
      const answer = new Promise((resolve) => {
        const timer = setTimeout(() => { results.delete(jobId); tab.inFlight.delete(jobId); resolve({ error: "tab did not answer in time" }); }, JOB_TIMEOUT_MS);
        results.set(jobId, { onDone: (r) => resolve(r), timer, peer: tab });
      });
      tab.queue.push({ jobId, kind, ...body });
      deliver(tab);
      const out = await answer;
      json(res, out.error ? 504 : 200, out);
      return;
    }
    // --- Tab API: tab side ---
    if (req.method === "GET" && (m = req.url.match(/^\/api\/relay\/([a-f0-9]{32})\/next$/))) {
      const tab = tabState(m[1]);
      tab.lastSeen = Date.now();
      if (tab.waiter) { try { tab.waiter.writeHead(204); tab.waiter.end(); } catch { /* ignore */ } }
      tab.waiter = res;
      const t = setTimeout(() => { if (tab.waiter === res) { tab.waiter = null; res.writeHead(204); res.end(); } }, POLL_TIMEOUT_MS);
      res.on("close", () => { clearTimeout(t); if (tab.waiter === res) tab.waiter = null; });
      deliver(tab);
      return;
    }
    if (req.method === "POST" && (m = req.url.match(/^\/api\/relay\/([a-f0-9]{32})\/result\/([a-f0-9]+)$/))) {
      const entry = results.get(m[2]);
      const body = JSON.parse(await readBody(req));
      const tab = tabState(m[1]);
      tab.lastSeen = Date.now(); tab.inFlight.delete(m[2]);
      if (entry) { clearTimeout(entry.timer); results.delete(m[2]); if (body.error) tab.failures++; entry.onDone(body, entry); }
      res.writeHead(204); res.end();
      return;
    }

    // Dev benchmark collector: bench.html posts result rows here; a script
    // reads them back for judging. In-memory only.
    if (req.method === "POST" && req.url === "/api/bench") {
      benchRows.push(JSON.parse(await readBody(req)));
      res.writeHead(204); res.end();
      return;
    }
    if (req.method === "GET" && req.url === "/api/bench") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(benchRows));
      return;
    }
    if (req.method === "DELETE" && req.url === "/api/bench") {
      benchRows.length = 0;
      res.writeHead(204); res.end();
      return;
    }

    // Strict mode: a second model judges whether the LaTeX matches the input.
    if (req.method === "POST" && req.url === "/api/judge") {
      const { text, latex } = JSON.parse(await readBody(req));
      if (!text || !latex) { res.writeHead(400, { "content-type": "application/json" }); res.end(JSON.stringify({ error: "text and latex required" })); return; }
      const route = await routeFor("refine");
      if (!route) { res.writeHead(503, { "content-type": "application/json" }); res.end(JSON.stringify({ error: "no backend" })); return; }
      const isMlx = route.kind === "mlx";
      const r = await fetch(isMlx ? `${MLX_URL}/v1/chat/completions` : `${OLLAMA_URL}/api/chat`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify(isMlx
          ? { model: route.model, stream: false, temperature: 0, max_tokens: 80, chat_template_kwargs: { enable_thinking: false }, messages: [{ role: "system", content: JUDGE_PROMPT }, { role: "user", content: `INPUT:\n${text}\n\nLATEX:\n${latex}` }] }
          : { model: route.model, stream: false, think: false, keep_alive: -1, options: { temperature: 0, num_predict: 80 }, messages: [{ role: "system", content: JUDGE_PROMPT }, { role: "user", content: `INPUT:\n${text}\n\nLATEX:\n${latex}` }] }),
        signal: AbortSignal.timeout(60_000),
      });
      const data = await r.json();
      const content = isMlx ? (data.choices?.[0]?.message?.content ?? "") : (data.message?.content ?? "");
      res.writeHead(200, { "content-type": "application/json" });
      res.end(content.slice(content.indexOf("{"), content.lastIndexOf("}") + 1) || JSON.stringify({ ok: true, reason: "" }));
      return;
    }

    if (req.method === "POST" && req.url === "/api/refine") {
      const { original, latex, instruction } = JSON.parse(await readBody(req));
      if (!latex || !instruction || String(instruction).length > 2000) {
        res.writeHead(400, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: "latex and instruction required (instruction max 2000 chars)" }));
        return;
      }
      const route = await routeFor("refine");
      if (!route) {
        res.writeHead(503, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: "no inference backend reachable" }));
        return;
      }
      await streamChat(res, route, [
        { role: "system", content: REFINE_PROMPT },
        { role: "user", content: `Convert to LaTeX:\n${original ?? "(not provided)"}` },
        { role: "assistant", content: latex },
        { role: "user", content: instruction },
      ]);
      return;
    }

    if (req.method === "POST" && req.url === "/api/convert") {
      const { text, complex } = JSON.parse(await readBody(req));
      if (!text || typeof text !== "string" || text.length > 6000) {
        res.writeHead(400, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: "text required (max 6000 chars)" }));
        return;
      }
      // Prose/multiline conversions need the strong model (small models drop
      // prose — benchmarked); plain equations get the fast one.
      const route = await routeFor(complex ? "refine" : "convert");
      if (!route) {
        res.writeHead(503, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: "no inference backend reachable" }));
        return;
      }
      await streamChat(res, route, [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: text },
      ]);
      return;
    }

    // Static files
    if (req.method === "GET") {
      let path = req.url.split("?")[0];
      if (path === "/") path = "/index.html";
      const file = normalize(join(PUBLIC_DIR, path));
      if (!file.startsWith(PUBLIC_DIR)) {
        res.writeHead(403); res.end(); return;
      }
      try {
        const content = await readFile(file);
        // Vendored libraries and model weights never change under a path;
        // the app shell must always revalidate so deploys show up immediately.
        const immutable = path.startsWith("/vendor/") || path.startsWith("/models/") || path.startsWith("/icons/");
        res.writeHead(200, {
          "content-type": MIME[extname(file)] || "application/octet-stream",
          "cache-control": immutable ? "public, max-age=31536000, immutable" : "no-cache",
        });
        res.end(content);
        return;
      } catch {
        res.writeHead(404, { "content-type": "text/plain" });
        res.end("not found");
        return;
      }
    }

    res.writeHead(405); res.end();
  } catch (err) {
    res.writeHead(500, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: String(err) }));
  }
});

server.listen(PORT, () => {
  console.log(`LatexGen on http://localhost:${PORT}`);
  console.log(`ollama backend: ${OLLAMA_URL} (convert: ${OLLAMA_MODEL}, refine: ${OLLAMA_REFINE_MODEL})`);
  warmModels();
});
