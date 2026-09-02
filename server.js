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

// ---- Tab API relay ----
// A browser tab can't accept connections, so it long-polls here for jobs and
// posts results back; external clients POST a job and wait for the answer.
// The relay forwards bytes only — all inference happens in the tab. State is
// in-memory (one process); ids are unguessable capability tokens.
const TAB_ID_RE = /^[a-f0-9]{32}$/;
const tabs = new Map();      // tabId -> { queue: [job], waiter: res|null, lastSeen }
const results = new Map();   // jobId -> { resolve, timer }
const JOB_TIMEOUT_MS = 25_000, POLL_TIMEOUT_MS = 30_000, MAX_QUEUE = 8;
function tabState(id) {
  if (!tabs.has(id)) tabs.set(id, { queue: [], waiter: null, lastSeen: 0 });
  return tabs.get(id);
}
function deliver(tab) {
  if (tab.waiter && tab.queue.length) {
    const job = tab.queue.shift();
    const res = tab.waiter; tab.waiter = null;
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify(job));
  }
}
setInterval(() => { // drop tabs not seen for 5 minutes
  const cutoff = Date.now() - 300_000;
  for (const [id, t] of tabs) if (t.lastSeen < cutoff && !t.waiter) tabs.delete(id);
}, 60_000).unref();

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
        // kept for older clients
        ollama: { reachable: ollama, model: OLLAMA_MODEL, url: OLLAMA_URL },
      }));
      return;
    }

    // --- Tab API: external client -> tab ---
    // POST /api/tab/<id>/convert {text} | {imageBase64, mime} | {latex, instruction}
    let m;
    if (req.method === "POST" && (m = req.url.match(/^\/api\/tab\/([a-f0-9]{32})\/(convert|refine)$/))) {
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
      if (kind === "convert" && !body.text && !body.imageBase64) { res.writeHead(400, { "content-type": "application/json" }); res.end(JSON.stringify({ error: "text or imageBase64 required" })); return; }
      if (kind === "refine" && (!body.latex || !body.instruction)) { res.writeHead(400, { "content-type": "application/json" }); res.end(JSON.stringify({ error: "latex and instruction required" })); return; }
      if ((body.text ?? "").length > 6000 || (body.imageBase64 ?? "").length > 8_000_000) { res.writeHead(413, { "content-type": "application/json" }); res.end(JSON.stringify({ error: "payload too large" })); return; }
      const jobId = Math.random().toString(16).slice(2) + Date.now().toString(16);
      const answer = new Promise((resolve) => {
        const timer = setTimeout(() => { results.delete(jobId); resolve({ error: "tab did not answer in time" }); }, JOB_TIMEOUT_MS);
        results.set(jobId, { resolve, timer });
      });
      tab.queue.push({ jobId, kind, ...body });
      deliver(tab);
      const out = await answer;
      res.writeHead(out.error ? 504 : 200, { "content-type": "application/json" });
      res.end(JSON.stringify(out));
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
      if (entry) { clearTimeout(entry.timer); results.delete(m[2]); entry.resolve(body); }
      tabState(m[1]).lastSeen = Date.now();
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
