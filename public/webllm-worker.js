// WebLLM runs here, off the main thread, so multi-GB model loads and token
// generation never freeze the UI. Bundle selection and tuning are driven by
// query params to let the app fall back without shipping multiple workers.
import { BUNDLE_RE, tuningFromSearch, applyTuning } from "./qwen3-webllm.js";

const queue = [];
let forward = null;
let initError = null;

self.onmessage = (msg) => {
  if (forward) {
    forward(msg);
    return;
  }
  if (initError) { rejectMessage(msg); return; }
  queue.push(msg);
};

async function boot() {
  const search = self.location?.search || "";
  const params = new URLSearchParams(search.startsWith("?") ? search.slice(1) : search);
  const bundle = params.get("bundle");
  if (bundle && !BUNDLE_RE.test(bundle)) throw new Error(`invalid bundle: ${bundle}`);

  const mod = await import(bundle ? `./vendor/webllm/${bundle}` : "./vendor/webllm/index.js");
  applyTuning(globalThis, tuningFromSearch(search));

  const handler = new mod.WebWorkerMLCEngineHandler();
  forward = (msg) => handler.onmessage(msg);
  for (const msg of queue.splice(0)) handler.onmessage(msg);
}

// WebLLM's worker protocol: every request carries a uuid and the main thread
// resolves it on {kind:"return"} or rejects on {kind:"throw"}. Answering a
// failed boot that way makes CreateWebWorkerMLCEngine reject, which is what
// lets app.js fall back to the stock bundle instead of hanging.
function rejectMessage(msg) {
  const uuid = msg?.data?.uuid;
  if (uuid) self.postMessage({ kind: "throw", uuid, content: `webllm-worker init failed: ${initError}` });
}

boot().catch((err) => {
  initError = String(err?.message || err);
  console.error("webllm-worker init failed:", err);
  for (const msg of queue.splice(0)) rejectMessage(msg);
});
