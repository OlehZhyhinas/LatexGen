// Integration tests: spawn server.js against a mock OpenAI-compatible upstream
// and exercise the server-model proxy, hardening, and the Tab API / mesh relay.
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";

let upstream, srv, base, lastChat;
const PORT = 8790 + Math.floor(Math.random() * 100);

before(async () => {
  // Mock OpenAI-compatible API: SSE stream for chat, /models for probing.
  upstream = createServer(async (req, res) => {
    if (req.url === "/v1/models") { res.writeHead(200, { "content-type": "application/json" }); return res.end(JSON.stringify({ data: [] })); }
    if (req.url === "/v1/chat/completions") {
      let body = ""; for await (const c of req) body += c;
      const j = JSON.parse(body);
      lastChat = j;
      assert.equal(req.headers.authorization, "Bearer test-key");
      const last = j.messages[j.messages.length - 1].content;
      const out = /judge|INPUT:/.test(last) ? '{"ok": true, "reason": "fine"}' : "$$x^{2}$$";
      if (!j.stream) { res.writeHead(200, { "content-type": "application/json" }); return res.end(JSON.stringify({ choices: [{ message: { content: out } }] })); }
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.write(": keepalive comment\n");
      for (const piece of out.match(/.{1,3}/g)) res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: piece } }] })}\n\n`);
      res.write("data: [DONE]\n\n"); return res.end();
    }
    res.writeHead(404); res.end();
  });
  await new Promise((r) => upstream.listen(0, "127.0.0.1", r));
  const up = `http://127.0.0.1:${upstream.address().port}/v1`;
  srv = spawn(process.execPath, ["server.js"], {
    env: { ...process.env, PORT: String(PORT), NODE_ENV: "production", OPENAI_BASE_URL: up, OPENAI_API_KEY: "test-key", OPENAI_MODEL: "mock-fast", OPENAI_REFINE_MODEL: "mock-strong", OLLAMA_DISABLED: "1", SERVER_KIND: "cloud" },
    stdio: ["ignore", "pipe", "pipe"],
  });
  base = `http://127.0.0.1:${PORT}`;
  for (let i = 0; i < 50; i++) { try { const r = await fetch(`${base}/api/health`); if (r.ok) break; } catch {} await new Promise((r) => setTimeout(r, 100)); }
});
after(() => { srv?.kill("SIGTERM"); upstream?.close(); });

const post = (path, body, headers = {}) => fetch(`${base}${path}`, { method: "POST", headers: { "content-type": "application/json", ...headers }, body: typeof body === "string" ? body : JSON.stringify(body) });
const ndjsonFinal = async (r) => { const lines = (await r.text()).trim().split("\n").map((l) => JSON.parse(l)); return lines[lines.length - 1]; };

test("health reports provider-agnostic routes and cloud server kind", async () => {
  const h = await (await fetch(`${base}/api/health`)).json();
  assert.equal(h.serverKind, "cloud");
  assert.equal(h.routes.convert.kind, "openai"); assert.equal(h.routes.convert.model, "mock-fast");
  assert.equal(h.routes.refine.model, "mock-strong");
  assert.equal(h.backends.openai, true); assert.equal(h.backends.ollama, false);
});

test("convert streams NDJSON from an OpenAI-compatible upstream (bearer key sent)", async () => {
  const r = await post("/api/convert", { text: "x squared" });
  assert.equal(r.status, 200);
  const final = await ndjsonFinal(r);
  assert.equal(final.done, true); assert.equal(final.latex, "$$x^{2}$$"); assert.match(final.model, /mock-fast \(openai\)/);
  assert.match(lastChat.messages[0].content, /Never solve/);
  assert.match(lastChat.messages.at(-1).content, /^Convert to LaTeX\. Do not solve or answer\.\nx squared$/);
});

test("prose conversions use the strong model; judge returns JSON", async () => {
  const final = await ndjsonFinal(await post("/api/convert", { text: "some prose\nwith lines", complex: true }));
  assert.match(final.model, /mock-strong/);
  const j = await (await post("/api/judge", { text: "x squared", latex: "$$x^2$$" })).json();
  assert.equal(j.ok, true);
});

test("hardening: security headers, size limits, bad json, bench disabled in prod, static traversal", async () => {
  const r = await fetch(`${base}/`);
  assert.equal(r.headers.get("x-content-type-options"), "nosniff");
  assert.match(r.headers.get("content-security-policy") || "", /wasm-unsafe-eval/);
  assert.equal((await post("/api/convert", "{not json")).status, 400);
  assert.equal((await post("/api/convert", { text: "x".repeat(70_000) })).status, 413);
  assert.equal((await fetch(`${base}/api/bench`)).status, 404);
  assert.equal((await fetch(`${base}/bench.html`)).status, 404);
  assert.ok([403, 404].includes((await fetch(`${base}/..%2F..%2Fetc%2Fpasswd`)).status), "traversal must be refused");
});

test("static files: Content-Length on GET and HEAD, immutable weights, .gz served as gzip", async () => {
  const r = await fetch(`${base}/app.js`);
  assert.equal(r.status, 200);
  assert.equal(Number(r.headers.get("content-length")), (await r.arrayBuffer()).byteLength, "progress bars need the real size");
  const h = await fetch(`${base}/vendor/transformers/transformers.min.js`, { method: "HEAD" });
  assert.equal(h.status, 200);
  assert.ok(Number(h.headers.get("content-length")) > 0, "HEAD carries the size transformers.js probes for");
  assert.match(h.headers.get("cache-control"), /immutable/);
  assert.equal((await h.arrayBuffer()).byteLength, 0);
  const gz = await fetch(`${base}/models/x.onnx.gz`, { method: "HEAD" });
  assert.equal(gz.status, 404); // no such artifact here, but the path is routable
  const webnn = await (await fetch(`${base}/models/texify-webnn/entry.json`)).json();
  assert.equal(webnn.family, "texify-420");
  assert.equal(webnn.compat.requires["backend.name"], "coreml");
});

test("tab API relay: long-poll delivers a job, result returns to the caller", async () => {
  const id = "a".repeat(32);
  const poll = fetch(`${base}/api/relay/${id}/next`);           // tab starts listening
  await new Promise((r) => setTimeout(r, 100));
  const client = post(`/api/tab/${id}/convert`, { text: "x squared" }); // external client asks
  const job = await (await poll).json();
  assert.equal(job.kind, "convert"); assert.equal(job.text, "x squared");
  await post(`/api/relay/${id}/result/${job.jobId}`, { latex: "$$x^2$$", ok: true, model: "IntelliTeX" });
  const out = await (await client).json();
  assert.equal(out.latex, "$$x^2$$"); assert.equal(out.model, "IntelliTeX");
  assert.equal((await post(`/api/tab/${"f".repeat(32)}/convert`, { text: "x" })).status, 404);
});

test("mesh: reciprocity gate, capability routing, owner lane first", async () => {
  const A = "b".repeat(32), B = "c".repeat(32);
  // A pooled but no LLM; B pooled with an LLM
  await post(`/api/relay/${A}/caps`, { caps: { specialist: true }, pool: { keys: ["public"] } });
  await post(`/api/relay/${B}/caps`, { caps: { specialist: true, llm: true, llmName: "Qwen 0.6B", multiline: false, tokPerSec: 100 }, pool: { keys: ["public"] } });
  const pollA = fetch(`${base}/api/relay/${A}/next`), pollB = fetch(`${base}/api/relay/${B}/next`);
  await new Promise((r) => setTimeout(r, 100));
  // prose needs multiline -> no capable peer
  const noPeer = await post(`/api/pool/${A}/convert`, { text: "line one\nline two" });
  assert.equal(noPeer.status, 503);
  // refine -> routed to B
  const req = post(`/api/pool/${A}/refine`, { latex: "$$x$$", instruction: "make it y", original: "x" });
  const jobB = await (await pollB).json();
  assert.equal(jobB.kind, "refine"); assert.equal(jobB.pool, true); assert.equal(jobB.noMesh, true);
  await post(`/api/relay/${B}/result/${jobB.jobId}`, { latex: "$$y$$", ok: true, model: "Qwen 0.6B" });
  const out = await (await req).json();
  assert.equal(out.latex, "$$y$$"); assert.equal(out.peer, B.slice(0, 6));
  // an unpooled, non-listening tab may not use peers
  const D = "d".repeat(32);
  assert.equal((await post(`/api/pool/${D}/convert`, { text: "x" })).status, 404);
  await post(`/api/relay/${D}/caps`, { caps: {}, pool: null });
  assert.equal((await post(`/api/pool/${D}/convert`, { text: "x" })).status, 429);
  // owner lane goes ahead of pool lane: queue a pool job and an owner job for B, poll once
  await post(`/api/relay/${B}/caps`, { caps: { llm: true, llmName: "Qwen 0.6B" }, pool: { keys: ["public"] } });
  const poolReq = post(`/api/pool/${A}/refine`, { latex: "$$x$$", instruction: "pool job", original: "x" });
  await new Promise((r) => setTimeout(r, 80));
  const ownerReq = post(`/api/tab/${B}/refine`, { latex: "$$x$$", instruction: "owner job", original: "x" });
  await new Promise((r) => setTimeout(r, 80));
  const first = await (await fetch(`${base}/api/relay/${B}/next`)).json();
  assert.equal(first.instruction, "owner job");
  await post(`/api/relay/${B}/result/${first.jobId}`, { latex: "$$o$$", ok: true });
  const second = await (await fetch(`${base}/api/relay/${B}/next`)).json();
  assert.equal(second.instruction, "pool job");
  await post(`/api/relay/${B}/result/${second.jobId}`, { latex: "$$p$$", ok: true });
  assert.equal((await (await ownerReq).json()).latex, "$$o$$");
  assert.equal((await (await poolReq).json()).latex, "$$p$$");
  pollA.catch(() => {});
});

test("rate limit trips on a burst of inference calls", async () => {
  const statuses = [];
  for (let i = 0; i < 140; i++) statuses.push((await post("/api/judge", { text: "a", latex: "b" })).status);
  assert.ok(statuses.includes(429), "expected a 429 within 140 rapid calls");
});
