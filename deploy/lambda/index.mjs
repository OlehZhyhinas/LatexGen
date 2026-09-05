// LatexGen stateless escalation proxy for the S3 + CloudFront path (AWS Lambda,
// Node 20+). Provider-agnostic: forwards to any OpenAI-compatible chat API
// (OpenRouter, OpenAI, Groq, Together, ...). The key lives ONLY here, in the
// Lambda environment. Without a key every inference endpoint returns 503 and
// the frontend shows escalation as unavailable. No Tab API / mesh relay here —
// that needs the stateful container (deploy/aws-apprunner.sh).
//
// Responses use the NDJSON contract of server.js: a single final line
// {done, latex, model, ms}, which the client parser accepts as-is.

const BASE = (process.env.OPENAI_BASE_URL || "").replace(/\/+$/, "");
const KEY = process.env.OPENAI_API_KEY || "";
const MODEL = process.env.OPENAI_MODEL || "";
const REFINE_MODEL = process.env.OPENAI_REFINE_MODEL || MODEL;
const MAX_INPUT = 6000;

const SYSTEM_PROMPT = `You are a text-to-LaTeX transcriber. Convert the user's input (plain-language math, equations, or prose with math) into LaTeX.

Rules:
- Output ONLY the LaTeX code. No explanations, no markdown code fences, no surrounding commentary.
- Never solve, evaluate, simplify, or answer. If the input is a question or problem, convert the question itself to LaTeX; do not produce the answer.
- For pure math, wrap display math in \\[ ... \\].
- For prose mixed with math, keep the prose as plain text and wrap math in \\( ... \\).
- Use standard LaTeX/amsmath commands only.`;
const CONVERT_USER = (text) => `Convert to LaTeX. Do not solve or answer.\n${text}`;
const REFINE_PROMPT = `You are a text-to-LaTeX transcriber in a feedback loop. You previously converted the user's text to LaTeX. The user now gives feedback on your conversion. Produce a corrected version of YOUR PREVIOUS LaTeX.

Rules:
- The user's message is feedback ABOUT the LaTeX — it is never content to transcribe. Never output the feedback text itself.
- Output ONLY the corrected LaTeX. No explanations, no markdown code fences.
- Change only what the feedback concerns; keep everything else, including delimiters, exactly as it was.
- If the feedback is vague, make your best guess at what is wrong and fix that.`;
const JUDGE_PROMPT = `You verify text-to-LaTeX conversions. Given the user's plain-English input and the produced LaTeX, decide whether the LaTeX expresses exactly what the text describes (same operations, grouping, exponents, limits, variables). Solving or answering instead of transcribing is not ok. Reply with ONLY a JSON object: {"ok": true/false, "reason": "<max 12 words>"}`;

const json = (status, body, ndjson = false) => ({
  statusCode: status,
  headers: { "content-type": ndjson ? "application/x-ndjson" : "application/json", "x-content-type-options": "nosniff" },
  body: typeof body === "string" ? body : JSON.stringify(body),
});
const stripThink = (t) => t.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
const stripFences = (t) => { const m = t.match(/^```(?:latex|tex)?\s*\n([\s\S]*?)\n?```\s*$/); return m ? m[1].trim() : t.trim(); };
const configured = () => !!(BASE && KEY && MODEL);

async function chat(model, messages, system, opts = {}) {
  const started = Date.now();
  const r = await fetch(`${BASE}/chat/completions`, {
    method: "POST",
    headers: { authorization: `Bearer ${KEY}`, "content-type": "application/json" },
    body: JSON.stringify({ model, messages: [{ role: "system", content: system }, ...messages], temperature: opts.temperature ?? 0.2, max_tokens: opts.maxTokens ?? 2048, stream: false }),
  });
  if (!r.ok) throw new Error(`upstream ${r.status}: ${(await r.text()).slice(0, 200)}`);
  const data = await r.json();
  const text = data.choices?.[0]?.message?.content ?? "";
  return { text: stripFences(stripThink(text)), model, ms: Date.now() - started };
}

export const handler = async (event) => {
  const path = event.rawPath ?? event.requestContext?.http?.path ?? "";
  const method = event.requestContext?.http?.method ?? "GET";

  if (path.endsWith("/api/health")) {
    return json(200, {
      ok: true, serverKind: "cloud",
      backends: { openai: configured(), ollama: false },
      routes: configured() ? { convert: { kind: "openai", model: MODEL }, refine: { kind: "openai", model: REFINE_MODEL } } : {},
      mesh: { tabs: 0, pooled: 0 },
    });
  }
  if (!configured()) return json(503, { error: "escalation not configured (OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL)" });

  let body;
  try { body = JSON.parse(event.isBase64Encoded ? Buffer.from(event.body, "base64").toString() : event.body ?? "{}"); }
  catch { return json(400, { error: "invalid json" }); }

  try {
    if (method === "POST" && path.endsWith("/api/convert")) {
      const { text, complex } = body;
      if (!text || typeof text !== "string" || text.length > MAX_INPUT) return json(400, { error: `text required (max ${MAX_INPUT} chars)` });
      const out = await chat(complex ? REFINE_MODEL : MODEL, [{ role: "user", content: CONVERT_USER(text) }], SYSTEM_PROMPT);
      return json(200, JSON.stringify({ done: true, latex: out.text, model: `${out.model} (openai)`, ms: out.ms }) + "\n", true);
    }
    if (method === "POST" && path.endsWith("/api/refine")) {
      const { original, latex, instruction } = body;
      if (!latex || !instruction || String(instruction).length > 2000) return json(400, { error: "latex and instruction required" });
      const out = await chat(REFINE_MODEL, [
        { role: "user", content: `Convert to LaTeX:\n${original ?? "(not provided)"}` }, { role: "assistant", content: latex }, { role: "user", content: instruction },
      ], REFINE_PROMPT);
      return json(200, JSON.stringify({ done: true, latex: out.text, model: `${out.model} (openai)`, ms: out.ms }) + "\n", true);
    }
    if (method === "POST" && path.endsWith("/api/judge")) {
      const { text, latex } = body;
      if (!text || !latex) return json(400, { error: "text and latex required" });
      const out = await chat(REFINE_MODEL, [{ role: "user", content: `INPUT:\n${text}\n\nLATEX:\n${latex}` }], JUDGE_PROMPT, { temperature: 0, maxTokens: 80 });
      const c = out.text; return json(200, c.slice(c.indexOf("{"), c.lastIndexOf("}") + 1) || { ok: true, reason: "" });
    }
  } catch (err) {
    return json(502, { error: String(err.message || err).slice(0, 300) });
  }
  return json(404, { error: "not found" });
};
