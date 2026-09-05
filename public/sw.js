// LatexGen service worker: makes the app installable and fully offline.
//
// Strategy:
//  - App shell (HTML/CSS/JS): network-first with cache fallback, so updates
//    land on the next load but the app opens offline.
//  - /vendor/** (pinned libraries, fonts, wasm): cache-first, immutable.
//  - Model weights (this origin's /models/** or the Hugging Face repo the
//    static build streams from): not cached here (transformers.js and WebLLM
//    already keep them in the Cache API; double-caching would double the disk
//    use), but swapped for a pre-compressed .gz when the model ships one
//    (scripts/compress-models.mjs), decompressed in-flight. Neither server
//    negotiates Content-Encoding, and the int8 weights are ~30% smaller
//    gzipped. The response carries the real Content-Length so progress bars
//    stay honest.
//  - /api/**: network only (escalation is optional and online-only).
//
// Every path here is relative to the worker's own URL, so the app works both
// at a domain root and under a deploy prefix such as /LatexGen/.
const VERSION = "v3-gz";
const SHELL_CACHE = `latexgen-shell-${VERSION}`;
const VENDOR_CACHE = `latexgen-vendor-${VERSION}`;
const SCOPE = new URL("./", location.href).pathname;
const SHELL = ["./", "index.html", "style.css", "app.js", "config.js", "validator.js", "webllm-worker.js", "onnx-worker.js", "texo-webnn.js", "models.js", "pipeline.js", "manifest.webmanifest", "benchmarks.html"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(SHELL_CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k.startsWith("latexgen-") && ![SHELL_CACHE, VENDOR_CACHE].includes(k)).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// ---- compressed weights ----
// A weight file lives at <model root>/onnx/<file>.onnx or <model root>/<x>.json;
// its manifest is <model root>/compressed.json. Manifests are small and
// remembered per root for the worker's lifetime (a miss is remembered too).
const WEIGHT = /\/(?:onnx\/[^/]+\.onnx|[^/]+\.json|[^/]+\.txt)$/;
const manifests = new Map();
function modelRoot(url) {
  const i = url.pathname.indexOf("/onnx/");
  const path = i >= 0 ? url.pathname.slice(0, i + 1) : url.pathname.slice(0, url.pathname.lastIndexOf("/") + 1);
  return url.origin + path;
}
async function manifestFor(root) {
  if (!manifests.has(root)) {
    manifests.set(root, fetch(root + "compressed.json").then((r) => (r.ok ? r.json() : null)).catch(() => null));
  }
  return manifests.get(root);
}
const isWeightRequest = (url) => WEIGHT.test(url.pathname) && !url.pathname.endsWith("/compressed.json");

async function weightResponse(request) {
  const url = new URL(request.url);
  // Only whole-file requests are swapped; Range probes see the raw file.
  if (request.headers.has("range") || typeof DecompressionStream === "undefined") return fetch(request);
  const root = modelRoot(url);
  const manifest = await manifestFor(root);
  const entry = manifest?.[url.href.slice(root.length)];
  if (!entry) return fetch(request);
  let gz;
  try { gz = await fetch(url.href + ".gz"); } catch { return fetch(request); }
  if (!gz.ok || !gz.body) return fetch(request);
  // Some CDNs decode .gz themselves: check the magic bytes before inflating.
  const reader = gz.body.getReader();
  const first = await reader.read();
  if (first.done) return fetch(request);
  const gzipped = first.value[0] === 0x1f && first.value[1] === 0x8b;
  const body = new ReadableStream({
    start(c) { c.enqueue(first.value); },
    async pull(c) { const { done, value } = await reader.read(); if (done) c.close(); else c.enqueue(value); },
    cancel() { reader.cancel(); },
  });
  return new Response(gzipped ? body.pipeThrough(new DecompressionStream("gzip")) : body, {
    status: 200,
    headers: {
      "content-type": url.pathname.endsWith(".json") ? "application/json" : "application/octet-stream",
      "content-length": String(entry.size),
      "cache-control": "public, max-age=31536000, immutable",
    },
  });
}

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET") return;
  if (url.origin !== location.origin) {
    if (/\/resolve\//.test(url.pathname) && isWeightRequest(url)) e.respondWith(weightResponse(e.request));
    return;
  }
  if (!url.pathname.startsWith(SCOPE)) return;
  const rel = url.pathname.slice(SCOPE.length);
  if (rel.startsWith("api/")) return;
  if (rel.startsWith("models/")) { if (isWeightRequest(url)) e.respondWith(weightResponse(e.request)); return; }

  if (rel.startsWith("vendor/") || rel.startsWith("icons/")) {
    e.respondWith(
      caches.open(VENDOR_CACHE).then(async (c) => {
        const hit = await c.match(e.request);
        if (hit) return hit;
        const res = await fetch(e.request);
        if (res.ok) c.put(e.request, res.clone());
        return res;
      })
    );
    return;
  }

  // app shell: network-first, fall back to cache when offline
  e.respondWith(
    fetch(e.request)
      .then((res) => {
        if (res.ok) caches.open(SHELL_CACHE).then((c) => c.put(e.request, res.clone()));
        return res;
      })
      .catch(() => caches.match(e.request).then((hit) => hit || caches.match("index.html")))
  );
});
