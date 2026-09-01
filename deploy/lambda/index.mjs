// LatexGen escalation proxy (AWS Lambda, Node 20+).
// Calls the Claude API for conversions/refinements that the in-browser tiers
// couldn't handle. The API key lives ONLY here (Lambda env var ANTHROPIC_API_KEY,
// set later — until then every conversion endpoint returns 503 and the
// frontend simply shows escalation as unavailable).
//
// Responses use the same NDJSON contract as the local server: zero or more
// {delta} lines, then {done, latex, model, ms}. This handler returns a single
// final line (no streaming), which the client parser accepts as-is.

const API_KEY = process.env.ANTHROPIC_API_KEY || "";
const MODEL = process.env.ESCALATION_MODEL || "claude-haiku-4-5";
const MAX_INPUT = 6000;

const SYSTEM_PROMPT = `You are a text-to-LaTeX transcriber. Convert the user's input (plain-language math, equations, or prose with math) into LaTeX.

Rules:
- Output ONLY the LaTeX code. No explanations, no markdown code fences, no surrounding commentary.
- For pure math, wrap display math in \\[ ... \\].
- For prose mixed with math, keep the prose as plain text and wrap math in \\( ... \\).
- Use standard LaTeX/amsmath commands only.`;

const REFINE_PROMPT = `You are a text-to-LaTeX transcriber in a feedback loop. You previously converted the user's text to LaTeX. The user now gives feedback on your conversion. Produce a corrected version of YOUR PREVIOUS LaTeX.

Rules:
- The user's message is feedback ABOUT the LaTeX — it is never content to transcribe. Never output the feedback text itself.
- Output ONLY the corrected LaTeX. No explanations, no markdown code fences.
- Change only what the feedback concerns; keep everything else, including delimiters, exactly as it was.
- If the feedback is vague, make your best guess at what is wrong and fix that.`;

const json = (status, body) => ({
  statusCode: status,
  headers: { "content-type": status === 200 ? "application/x-ndjson" : "application/json" },
  body: typeof body === "string" ? body : JSON.stringify(body),
});

const stripFences = (t) => {
  const m = t.match(/^```(?:latex|tex)?\s*\n([\s\S]*?)\n?```\s*$/);
  return m ? m[1].trim() : t.trim();
};

async function claude(messages, system) {
  const started = Date.now();
  const r = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": API_KEY,
      "anthropic-version": "2023-06-01",
      "content-type": "application/json",
    },
    body: JSON.stringify({ model: MODEL, max_tokens: 2048, system, messages }),
  });
  if (!r.ok) throw new Error(`claude api ${r.status}: ${(await r.text()).slice(0, 200)}`);
  const data = await r.json();
  const text = (data.content ?? []).filter((b) => b.type === "text").map((b) => b.text).join("");
  return { latex: stripFences(text), model: MODEL, ms: Date.now() - started };
}

export const handler = async (event) => {
  const path = event.rawPath ?? event.requestContext?.http?.path ?? "";
  const method = event.requestContext?.http?.method ?? "GET";

  if (path.endsWith("/api/health")) {
    return json(200, JSON.stringify({
      ok: true,
      backends: { api: !!API_KEY },
      routes: API_KEY ? { convert: { kind: "api", model: MODEL }, refine: { kind: "api", model: MODEL } } : {},
    }));
  }

  if (!API_KEY) {
    return json(503, { error: "escalation not configured (no API key set)" });
  }

  let body;
  try {
    body = JSON.parse(event.isBase64Encoded ? Buffer.from(event.body, "base64").toString() : event.body ?? "{}");
  } catch {
    return json(400, { error: "invalid json" });
  }

  try {
    if (method === "POST" && path.endsWith("/api/convert")) {
      const { text } = body;
      if (!text || typeof text !== "string" || text.length > MAX_INPUT) {
        return json(400, { error: `text required (max ${MAX_INPUT} chars)` });
      }
      const out = await claude([{ role: "user", content: text }], SYSTEM_PROMPT);
      return json(200, JSON.stringify({ done: true, ...out }) + "\n");
    }

    if (method === "POST" && path.endsWith("/api/refine")) {
      const { original, latex, instruction } = body;
      if (!latex || !instruction || String(instruction).length > 2000) {
        return json(400, { error: "latex and instruction required" });
      }
      const out = await claude(
        [
          { role: "user", content: `Convert to LaTeX:\n${original ?? "(not provided)"}` },
          { role: "assistant", content: latex },
          { role: "user", content: instruction },
        ],
        REFINE_PROMPT
      );
      return json(200, JSON.stringify({ done: true, ...out }) + "\n");
    }
  } catch (err) {
    return json(502, { error: String(err).slice(0, 300) });
  }

  return json(404, { error: "not found" });
};
