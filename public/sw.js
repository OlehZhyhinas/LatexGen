// LatexGen service worker: makes the app installable and fully offline.
//
// Strategy:
//  - App shell (HTML/CSS/JS): network-first with cache fallback, so updates
//    land on the next load but the app opens offline.
//  - /vendor/** (pinned libraries, fonts, wasm): cache-first, immutable.
//  - /models/** and HuggingFace weights: NOT handled here. transformers.js and
//    WebLLM already cache them in the Cache API; double-caching would double
//    the disk use.
//  - /api/**: network only (escalation is optional and online-only).
//
// Every path here is relative to the worker's own URL, so the app works both
// at a domain root and under a deploy prefix such as /LatexGen/.
const VERSION = "v2-relative";
const SHELL_CACHE = `latexgen-shell-${VERSION}`;
const VENDOR_CACHE = `latexgen-vendor-${VERSION}`;
const SCOPE = new URL("./", location.href).pathname;
const SHELL = ["./", "index.html", "style.css", "app.js", "config.js", "validator.js", "webllm-worker.js", "onnx-worker.js", "models.js", "pipeline.js", "manifest.webmanifest", "benchmarks.html"];

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

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin) return;
  if (!url.pathname.startsWith(SCOPE)) return;
  const rel = url.pathname.slice(SCOPE.length);
  if (rel.startsWith("api/") || rel.startsWith("models/")) return;

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
