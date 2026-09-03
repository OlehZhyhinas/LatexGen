// Autonomous benchmark: runs every configured approach over the eval set,
// posting one row per (approach, item) to /api/bench. Drives WebLLM models
// sequentially (smallest first) and the transformers.js specialist.
import * as webllm from "https://esm.run/@mlc-ai/web-llm";
import { pipeline as tjsPipeline, env as tjsEnv } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const SYSTEM_PROMPT = `You are a text-to-LaTeX transcriber. Convert the user's input (plain-language math, equations, or prose with math) into LaTeX.

Rules:
- Output ONLY the LaTeX code. No explanations, no markdown code fences, no surrounding commentary.
- For pure math, wrap display math in \\[ ... \\].
- For prose mixed with math, keep the prose as plain text and wrap math in \\( ... \\).
- Use standard LaTeX/amsmath commands only.`;

const SEGMENT_PROMPT = `Repeat the user's text EXACTLY, word for word, with one change: wrap every mathematical expression (anything describing math in words or symbols) in double angle brackets like «...». Do NOT convert anything to LaTeX. Do NOT change, add, or remove any words. Output only the marked-up text.`;

const SPECIALIST_PREFIX = "Convert natural-language math into a STRICT LaTeX equation\n";

const WEBLLM_MODELS = [
  { id: "SmolLM2-360M-Instruct-q4f16_1-MLC", name: "SmolLM2-360M", pipeline: true },
  { id: "Qwen3-0.6B-q4f16_1-MLC", name: "Qwen3-0.6B", pipeline: true },
  { id: "Qwen3-1.7B-q4f16_1-MLC", name: "Qwen3-1.7B", pipeline: false },
  { id: "Llama-3.2-3B-Instruct-q4f16_1-MLC", name: "Llama-3.2-3B", pipeline: false },
  { id: "Qwen3-4B-q4f16_1-MLC", name: "Qwen3-4B", pipeline: false },
];

const log = (cls, msg) => {
  const el = document.createElement("div");
  el.className = cls; el.textContent = msg;
  document.getElementById("log").prepend(el);
};
const progress = (msg) => { document.getElementById("progress").textContent = msg; };
const post = (row) => fetch("api/bench", { method: "POST", body: JSON.stringify(row) });

const stripThink = (t) => t.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
const stripFences = (t) => {
  const m = t.match(/^```(?:latex|tex)?\s*\n([\s\S]*?)\n?```\s*$/);
  return m ? m[1].trim() : t.trim();
};

async function genWebllm(engine, system, user, maxTokens) {
  const chunks = await engine.chat.completions.create({
    messages: [{ role: "system", content: system }, { role: "user", content: user }],
    temperature: 0.2, max_tokens: maxTokens, stream: true,
    extra_body: { enable_thinking: false },
  });
  let full = "";
  for await (const c of chunks) full += c.choices[0]?.delta?.content ?? "";
  return stripFences(stripThink(full));
}

async function main() {
  const items = await fetch("bench-data.json").then((r) => r.json());
  progress(`loaded ${items.length} eval items`);

  // Specialist
  tjsEnv.allowRemoteModels = false;
  tjsEnv.allowLocalModels = true;
  tjsEnv.localModelPath = new URL("models/", import.meta.url).href;
  progress("loading specialist…");
  const specialist = await tjsPipeline("text2text-generation", "intellitex", { dtype: "q8" });
  await specialist(`${SPECIALIST_PREFIX}x squared`, { max_new_tokens: 16 });
  log("ok", "specialist ready");

  const convertSpecialist = async (text) => {
    const out = await specialist(SPECIALIST_PREFIX + text, { max_new_tokens: 256 });
    return (out[0]?.generated_text ?? "").trim();
  };

  // Approach 1: specialist solo on single-line items
  for (const it of items.filter((i) => i.tier !== "multiline")) {
    const t0 = performance.now();
    try {
      const out = await convertSpecialist(it.input);
      await post({ approach: "specialist", item: it.id, tier: it.tier, ms: Math.round(performance.now() - t0), output: out });
      log("ok", `specialist ${it.id} ${Math.round(performance.now() - t0)}ms`);
    } catch (e) {
      await post({ approach: "specialist", item: it.id, tier: it.tier, ms: -1, output: "", error: String(e) });
      log("err", `specialist ${it.id} FAILED ${e}`);
    }
  }

  // WebLLM models: direct on everything; pipeline (segment+specialist) where flagged
  for (const m of WEBLLM_MODELS) {
    progress(`loading ${m.name}…`);
    let engine;
    try {
      engine = await webllm.CreateMLCEngine(m.id, {
        initProgressCallback: (p) => progress(`${m.name}: ${p.text?.slice(0, 90) ?? ""}`),
      }, { context_window_size: 2048 });
      await genWebllm(engine, "You are helpful.", "hi", 2); // warm-up
      log("ok", `${m.name} loaded`);
    } catch (e) {
      log("err", `${m.name} LOAD FAILED: ${e}`);
      await post({ approach: `direct:${m.name}`, item: "__load__", tier: "meta", ms: -1, output: "", error: String(e).slice(0, 200) });
      continue;
    }

    for (const it of items) {
      // direct conversion
      const t0 = performance.now();
      try {
        const out = await genWebllm(engine, SYSTEM_PROMPT, it.input, 768);
        await post({ approach: `direct:${m.name}`, item: it.id, tier: it.tier, ms: Math.round(performance.now() - t0), output: out });
        log("ok", `direct:${m.name} ${it.id} ${Math.round(performance.now() - t0)}ms`);
      } catch (e) {
        await post({ approach: `direct:${m.name}`, item: it.id, tier: it.tier, ms: -1, output: "", error: String(e).slice(0, 200) });
        log("err", `direct:${m.name} ${it.id} FAILED`);
      }

      // segmenter pipeline
      if (m.pipeline) {
        const t1 = performance.now();
        try {
          const marked = await genWebllm(engine, SEGMENT_PROMPT, it.input, 768);
          const spans = [...marked.matchAll(/«([^«»]+)»/g)].map((x) => x[1]);
          let assembled;
          if (!spans.length) {
            assembled = ""; // segmenter found nothing — counts as failure
          } else {
            assembled = marked;
            for (const s of spans) {
              const latex = await convertSpecialist(s.trim());
              const inner = latex.replace(/^\$\$|\$\$$/g, "").trim();
              assembled = assembled.replace(`«${s}»`, `\\(${inner}\\)`);
            }
          }
          await post({ approach: `pipeline:${m.name}+spec`, item: it.id, tier: it.tier, ms: Math.round(performance.now() - t1), output: assembled, spans: spans.length });
          log(assembled ? "ok" : "bad", `pipeline:${m.name} ${it.id} ${Math.round(performance.now() - t1)}ms spans=${spans.length}`);
        } catch (e) {
          await post({ approach: `pipeline:${m.name}+spec`, item: it.id, tier: it.tier, ms: -1, output: "", error: String(e).slice(0, 200) });
          log("err", `pipeline:${m.name} ${it.id} FAILED`);
        }
      }
    }
    try { await engine.unload(); } catch {}
  }

  progress("BENCH COMPLETE");
  log("ok", "all done");
}

main().catch((e) => { progress(`FATAL: ${e}`); });
