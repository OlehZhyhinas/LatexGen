// Zero-dependency Node server: serves the static frontend, proxies the
// "server model" tier to any OpenAI-compatible API or Ollama, and hosts the
// Tab API / compute-mesh relay.
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const PORT = process.env.PORT || 8000;
const PROD = process.env.NODE_ENV === "production" || process.env.LATEXGEN_PROD === "1";
const TRUST_PROXY = process.env.TRUST_PROXY === "1";   // behind App Runner/CloudFront/ALB
const HSTS = process.env.HSTS === "1";                 // only when TLS terminates in front
const LOG_IP = process.env.LOG_IP === "1";             // access logs omit IPs by default

// --- Backend A: any OpenAI-compatible chat API (OpenRouter, OpenAI, Groq,
// Together, a vLLM box, or local mlx_lm.server). Provider-agnostic on purpose.
const OPENAI_BASE_URL = (process.env.OPENAI_BASE_URL
  || (process.env.MLX_URL ? `${process.env.MLX_URL}/v1` : "http://host.docker.internal:8080/v1")).replace(/\/+$/, "");
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const OPENAI_MODEL = process.env.OPENAI_MODEL || process.env.MLX_MODEL || "mlx-community/Qwen3-1.7B-4bit";
// Refinement, prose and judging want a stronger model than plain conversion.
const OPENAI_REFINE_MODEL = process.env.OPENAI_REFINE_MODEL || process.env.MLX_REFINE_MODEL || null;
const OPENAI_LOCAL = /^https?:\/\/(localhost|127\.0\.0\.1|host\.docker\.internal|\[::1\])(:|\/|$)/.test(OPENAI_BASE_URL);
const OPENAI_ENABLED = process.env.OPENAI_DISABLED !== "1";
// --- Backend B: Ollama (self-hosted).
const OLLAMA_URL = process.env.OLLAMA_URL || "http://host.docker.internal:11434";
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "qwen3:1.7b";
const OLLAMA_REFINE_MODEL = process.env.OLLAMA_REFINE_MODEL || OLLAMA_MODEL;
const OLLAMA_ENABLED = process.env.OLLAMA_DISABLED !== "1";
// Clients order their escalation ladder by this: a self-hosted local server is
// trusted and used before peers; a cloud API (costs, logs) comes after peers.
const SERVER_KIND = process.env.SERVER_KIND || (OPENAI_ENABLED && !OPENAI_LOCAL ? "cloud" : "local");

// --- limits ---
const BODY_LIMIT_TEXT = 64 * 1024, BODY_LIMIT_IMAGE = 6 * 1024 * 1024;
const RATE = { general: 600, inference: 120, relay: 1200 }; // requests per minute per IP
const MAX_TABS = 2000, MAX_RESULTS = 5000;

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
  ".gz": "application/gzip",
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

class HttpError extends Error { constructor(status, msg) { super(msg); this.status = status; } }
async function readBody(req, limit = BODY_LIMIT_TEXT) {
  const chunks = []; let size = 0, over = false;
  for await (const c of req) {
    size += c.length;
    if (size > limit) { over = true; if (size > limit * 4) break; continue; } // drain (bounded) so the 413 reaches the client
    chunks.push(c);
  }
  if (over) throw new HttpError(413, "payload too large");
  return Buffer.concat(chunks).toString("utf-8");
}
async function readJson(req, limit) {
  try { return JSON.parse(await readBody(req, limit) || "{}"); }
  catch (e) { if (e instanceof HttpError) throw e; throw new HttpError(400, "invalid json"); }
}

// ---- per-IP token buckets ----
const buckets = new Map();
function clientIp(req) {
  if (TRUST_PROXY) { const xff = req.headers["x-forwarded-for"]; if (xff) return String(xff).split(",")[0].trim(); }
  return req.socket.remoteAddress || "?";
}
function rateLimited(ip, cls) {
  const limit = RATE[cls], now = Date.now();
  const key = `${cls}:${ip}`;
  let b = buckets.get(key);
  if (!b) { b = { tokens: limit, at: now }; buckets.set(key, b); }
  b.tokens = Math.min(limit, b.tokens + ((now - b.at) / 60_000) * limit); b.at = now;
  if (b.tokens < 1) return true;
  b.tokens -= 1; return false;
}
setInterval(() => { const cutoff = Date.now() - 600_000; for (const [k, b] of buckets) if (b.at < cutoff) buckets.delete(k); }, 300_000).unref();
const rateClass = (path) => path.startsWith("/api/relay/") ? "relay"
  : /^\/api\/(convert|refine|judge|pool\/|tab\/)/.test(path) ? "inference" : "general";

// ---- security headers ----
const CSP = [
  "default-src 'self'",
  "script-src 'self' 'wasm-unsafe-eval'",
  "worker-src 'self' blob:",
  "connect-src 'self' https:", // model weights stream from HuggingFace CDNs
  "img-src 'self' data: blob:",
  "style-src 'self' 'unsafe-inline'", // KaTeX and MathLive set inline styles
  "font-src 'self' data:",
  "form-action 'self' https://www.overleaf.com",
  "frame-ancestors 'none'", "base-uri 'self'", "object-src 'none'",
].join("; ");
function securityHeaders(isHtml) {
  const h = {
    "x-content-type-options": "nosniff",
    "referrer-policy": "strict-origin-when-cross-origin",
    "permissions-policy": "camera=(self), microphone=(), geolocation=()",
    "x-frame-options": "DENY",
  };
  if (isHtml) h["content-security-policy"] = CSP;
  if (HSTS) h["strict-transport-security"] = "max-age=31536000; includeSubDomains";
  return h;
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
let probeCache = { at: 0, ollama: false, openai: false };
const openaiHeaders = () => ({ "content-type": "application/json", ...(OPENAI_API_KEY ? { authorization: `Bearer ${OPENAI_API_KEY}` } : {}) });
async function probeBackends() {
  // Cloud APIs are assumed up when configured (probing them costs requests and
  // some providers gate /models); local backends are probed and cached ~20s.
  if (Date.now() - probeCache.at < 20_000) return probeCache;
  const check = (url, headers) => fetch(url, { headers, signal: AbortSignal.timeout(1500) }).then((r) => r.ok).catch(() => false);
  const [ollama, openai] = await Promise.all([
    OLLAMA_ENABLED ? check(`${OLLAMA_URL}/api/version`) : false,
    OPENAI_ENABLED ? (OPENAI_LOCAL ? check(`${OPENAI_BASE_URL}/models`, openaiHeaders()) : !!OPENAI_API_KEY) : false,
  ]);
  probeCache = { at: Date.now(), ollama, openai };
  return probeCache;
}

// Route each task to the best available backend: conversions want speed
// (MLX when present), refinements want the strongest model (Ollama's big
// model, unless an MLX refine model is configured or Ollama is down).
async function routeFor(task) {
  const { ollama, openai } = await probeBackends();
  if (task === "convert") {
    if (openai) return { kind: "openai", model: OPENAI_MODEL };
    if (ollama) return { kind: "ollama", model: OLLAMA_MODEL };
  } else {
    if (openai && OPENAI_REFINE_MODEL) return { kind: "openai", model: OPENAI_REFINE_MODEL };
    if (ollama) return { kind: "ollama", model: OLLAMA_REFINE_MODEL };
    if (openai) return { kind: "openai", model: OPENAI_MODEL };
  }
  return null;
}
// One request body for either dialect.
function chatBody(route, messages, { stream, temperature, maxTokens }) {
  if (route.kind === "openai") {
    const b = { model: route.model, stream, temperature, max_tokens: maxTokens, messages };
    if (OPENAI_LOCAL) b.chat_template_kwargs = { enable_thinking: false }; // mlx_lm.server / vLLM with Qwen
    return b;
  }
  return { model: route.model, stream, think: false, keep_alive: -1, options: { temperature, num_predict: maxTokens }, messages };
}
const chatUrl = (route) => route.kind === "openai" ? `${OPENAI_BASE_URL}/chat/completions` : `${OLLAMA_URL}/api/chat`;
const chatHeaders = (route) => route.kind === "openai" ? openaiHeaders() : { "content-type": "application/json" };
async function chatOnce(route, messages, opts) {
  const r = await fetch(chatUrl(route), { method: "POST", headers: chatHeaders(route), body: JSON.stringify(chatBody(route, messages, { stream: false, ...opts })), signal: AbortSignal.timeout(60_000) });
  if (!r.ok) throw new HttpError(502, `${route.kind} ${r.status}`);
  const data = await r.json();
  return route.kind === "openai" ? (data.choices?.[0]?.message?.content ?? "") : (data.message?.content ?? "");
}

// Stream a chat completion to the client as NDJSON: {delta} lines while
// generating, then a final {done, latex, model, ms} line. Speaks both
// dialects: Ollama's NDJSON and MLX/OpenAI's SSE.
async function streamChat(res, route, messages) {
  const started = Date.now();
  const isMlx = route.kind === "openai"; // SSE dialect (OpenAI-compatible); else Ollama NDJSON
  const r = await fetch(chatUrl(route), {
    method: "POST", headers: chatHeaders(route),
    body: JSON.stringify(chatBody(route, messages, { stream: true, temperature: 0.2, maxTokens: 2048 })),
    signal: AbortSignal.timeout(120_000),
  });
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
  if (!OLLAMA_ENABLED) return;
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
  const started = Date.now();
  const path = (req.url || "/").split("?")[0];
  const ip = clientIp(req);
  // security headers on every response
  const baseHeaders = securityHeaders(path === "/" || path.endsWith(".html"));
  const origWriteHead = res.writeHead.bind(res);
  res.writeHead = (status, headers) => origWriteHead(status, { ...baseHeaders, ...(headers || {}) });
  res.on("finish", () => {
    if (path.startsWith("/api/relay/")) return; // long-poll chatter
    console.log(JSON.stringify({ t: new Date().toISOString(), m: req.method, p: path.replace(/[a-f0-9]{32}/g, "<id>"), s: res.statusCode, ms: Date.now() - started, ...(LOG_IP ? { ip } : {}) }));
  });
  try {
    if (path.startsWith("/api/") && rateLimited(ip, rateClass(path))) return json(res, 429, { error: "rate limit exceeded, slow down" });

    if (req.method === "GET" && path === "/api/health") {
      const { ollama, openai } = await probeBackends();
      const [convert, refine] = await Promise.all([routeFor("convert"), routeFor("refine")]);
      res.writeHead(200, { "content-type": "application/json", "cache-control": "no-store" });
      res.end(JSON.stringify({
        ok: true,
        backends: { ollama, openai, mlx: openai && OPENAI_LOCAL },
        routes: { convert, refine },
        serverKind: SERVER_KIND,
        mesh: { tabs: [...tabs.values()].filter((t) => isLive(t)).length, pooled: [...tabs.values()].filter((t) => isLive(t) && t.pool).length },
        ollama: { reachable: ollama, model: OLLAMA_MODEL },
      }));
      return;
    }

    let m;
    // --- mesh: a tab registers its capabilities and pool membership ---
    if (req.method === "POST" && (m = path.match(/^\/api\/relay\/([a-f0-9]{32})\/caps$/))) {
      if (!tabs.has(m[1]) && tabs.size >= MAX_TABS) return json(res, 503, { error: "relay is full" });
      const tab = tabState(m[1]);
      const body = await readJson(req);
      tab.caps = body.caps ?? {};
      tab.pool = body.pool ? { keys: (body.pool.keys ?? ["public"]).map((k) => String(k).slice(0, 64)), images: !!body.pool.images, maxConcurrent: Math.min(3, Math.max(1, body.pool.maxConcurrent ?? 1)) } : null;
      tab.lastSeen = Date.now();
      json(res, 200, { ok: true, served: tab.servedWeight, used: tab.usedWeight, servedJobs: tab.servedJobs, usedJobs: tab.usedJobs, peers: [...tabs.values()].filter((t) => t !== tab && isLive(t) && t.pool && t.pool.keys.some((k) => tab.pool?.keys.includes(k))).length });
      return;
    }
    // --- mesh: a tab (or its owner's client) asks a peer to run a job ---
    if (req.method === "POST" && (m = path.match(/^\/api\/pool\/([a-f0-9]{32})\/(convert|refine|check)$/))) {
      const requester = tabs.get(m[1]);
      if (!requester) return json(res, 404, { error: "unknown tab" });
      const why = reciprocityOk(requester);
      if (why) return json(res, 429, { error: why });
      const body = await readJson(req, BODY_LIMIT_IMAGE);
      if ((body.text ?? "").length > 6000 || (body.latex ?? "").length > 6000) return json(res, 413, { error: "payload too large" });
      if (body.imageBase64 && !requester.pool.images) return json(res, 400, { error: "images are not shared with peers" });
      const out = await runOnMesh(requester, { kind: m[2], ...body });
      json(res, out.error ? (out.noPeer ? 503 : 504) : 200, out);
      return;
    }

    // --- Tab API: external client -> tab ---
    // POST /api/tab/<id>/convert {text} | {imageBase64, mime} | {latex, instruction}
    if (req.method === "POST" && (m = path.match(/^\/api\/tab\/([a-f0-9]{32})\/(convert|refine|check|format|status|history)$/))) {
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
      const body = await readJson(req, BODY_LIMIT_IMAGE);
      const bad = (msg) => { res.writeHead(400, { "content-type": "application/json" }); res.end(JSON.stringify({ error: msg })); };
      if (kind === "convert" && !body.text && !body.imageBase64 && !body.latex) return bad("text, imageBase64 or latex required");
      if (kind === "refine" && (!body.latex || !body.instruction)) return bad("latex and instruction required");
      if (kind === "check" && !body.latex) return bad("latex required");
      if (kind === "format" && (!body.latex || !body.format)) return bad("latex and format required");
      if ((body.text ?? "").length > 6000) return json(res, 413, { error: "payload too large" });
      if (results.size >= MAX_RESULTS) return json(res, 503, { error: "relay is busy" });
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
    if (req.method === "GET" && (m = path.match(/^\/api\/relay\/([a-f0-9]{32})\/next$/))) {
      if (!tabs.has(m[1]) && tabs.size >= MAX_TABS) return json(res, 503, { error: "relay is full" });
      const tab = tabState(m[1]);
      tab.lastSeen = Date.now();
      if (tab.waiter) { try { tab.waiter.writeHead(204); tab.waiter.end(); } catch { /* ignore */ } }
      tab.waiter = res;
      const t = setTimeout(() => { if (tab.waiter === res) { tab.waiter = null; res.writeHead(204); res.end(); } }, POLL_TIMEOUT_MS);
      res.on("close", () => { clearTimeout(t); if (tab.waiter === res) tab.waiter = null; });
      deliver(tab);
      return;
    }
    if (req.method === "POST" && (m = path.match(/^\/api\/relay\/([a-f0-9]{32})\/result\/([a-f0-9]+)$/))) {
      const entry = results.get(m[2]);
      const body = await readJson(req, BODY_LIMIT_IMAGE);
      const tab = tabState(m[1]);
      tab.lastSeen = Date.now(); tab.inFlight.delete(m[2]);
      if (entry) { clearTimeout(entry.timer); results.delete(m[2]); if (body.error) tab.failures++; entry.onDone(body, entry); }
      res.writeHead(204); res.end();
      return;
    }

    // Dev benchmark collector: bench.html posts result rows here; a script
    // reads them back for judging. In-memory only.
    if (PROD && path === "/api/bench") return json(res, 404, { error: "not found" });
    if (req.method === "POST" && path === "/api/bench") {
      benchRows.push(await readJson(req, BODY_LIMIT_IMAGE));
      res.writeHead(204); res.end();
      return;
    }
    if (req.method === "GET" && path === "/api/bench") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(benchRows));
      return;
    }
    if (req.method === "DELETE" && path === "/api/bench") {
      benchRows.length = 0;
      res.writeHead(204); res.end();
      return;
    }

    // Strict mode: a second model judges whether the LaTeX matches the input.
    if (req.method === "POST" && path === "/api/judge") {
      const { text, latex } = await readJson(req);
      if (!text || !latex) return json(res, 400, { error: "text and latex required" });
      const route = await routeFor("refine");
      if (!route) return json(res, 503, { error: "no backend" });
      const content = await chatOnce(route, [{ role: "system", content: JUDGE_PROMPT }, { role: "user", content: `INPUT:\n${text}\n\nLATEX:\n${latex}` }], { temperature: 0, maxTokens: 80 });
      res.writeHead(200, { "content-type": "application/json" });
      res.end(content.slice(content.indexOf("{"), content.lastIndexOf("}") + 1) || JSON.stringify({ ok: true, reason: "" }));
      return;
    }

    if (req.method === "POST" && path === "/api/refine") {
      const { original, latex, instruction } = await readJson(req);
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

    if (req.method === "POST" && path === "/api/convert") {
      const { text, complex } = await readJson(req);
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
    if (req.method === "GET" || req.method === "HEAD") {
      let filePath = path === "/" ? "/index.html" : path;
      if (PROD && /^\/bench/.test(filePath)) { res.writeHead(404, { "content-type": "text/plain" }); res.end("not found"); return; }
      const file = normalize(join(PUBLIC_DIR, decodeURIComponent(filePath)));
      if (!file.startsWith(PUBLIC_DIR)) {
        res.writeHead(403); res.end(); return;
      }
      try {
        const { size } = await stat(file);
        // Vendored libraries and model weights never change under a path;
        // the app shell must always revalidate so deploys show up immediately.
        // Content-Length matters: transformers.js sizes its progress bar from
        // it, and the service worker's .gz swap-in relies on knowing sizes.
        const immutable = filePath.startsWith("/vendor/") || filePath.startsWith("/models/") || filePath.startsWith("/icons/");
        res.writeHead(200, {
          "content-type": MIME[extname(file)] || "application/octet-stream",
          "content-length": size,
          "cache-control": immutable ? "public, max-age=31536000, immutable" : "no-cache",
        });
        if (req.method === "HEAD") { res.end(); return; }
        createReadStream(file).on("error", () => res.destroy()).pipe(res);
        return;
      } catch {
        res.writeHead(404, { "content-type": "text/plain" });
        res.end("not found");
        return;
      }
    }

    res.writeHead(405); res.end();
  } catch (err) {
    if (res.headersSent) { try { res.end(); } catch {} return; }
    if (err instanceof HttpError) return json(res, err.status, { error: err.message });
    console.error(JSON.stringify({ t: new Date().toISOString(), error: String(err?.stack || err).slice(0, 500), p: path }));
    json(res, 500, { error: PROD ? "internal error" : String(err) });
  }
});
server.headersTimeout = 65_000;
server.requestTimeout = 120_000;
server.keepAliveTimeout = 65_000;

// Graceful shutdown: release long-polls so tabs reconnect to the next instance.
function shutdown(sig) {
  console.log(JSON.stringify({ t: new Date().toISOString(), shutdown: sig }));
  for (const t of tabs.values()) if (t.waiter) { try { t.waiter.writeHead(204); t.waiter.end(); } catch {} t.waiter = null; }
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 5000).unref();
}
process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));

if (process.env.LATEXGEN_NO_LISTEN !== "1") {
  server.listen(PORT, () => {
    console.log(JSON.stringify({ t: new Date().toISOString(), listening: PORT, prod: PROD, serverKind: SERVER_KIND,
      openai: OPENAI_ENABLED ? { base: OPENAI_BASE_URL, model: OPENAI_MODEL, refine: OPENAI_REFINE_MODEL, key: OPENAI_API_KEY ? "set" : "none" } : "off",
      ollama: OLLAMA_ENABLED ? { url: OLLAMA_URL, model: OLLAMA_MODEL, refine: OLLAMA_REFINE_MODEL } : "off" }));
    warmModels();
  });
}
export { server };
