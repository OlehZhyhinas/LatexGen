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
const VERSION = "v1";
const SHELL_CACHE = `latexgen-shell-${VERSION}`;
const VENDOR_CACHE = `latexgen-vendor-${VERSION}`;
const SHELL = ["/", "/index.html", "/style.css", "/app.js", "/validator.js", "/webllm-worker.js", "/manifest.webmanifest", "/benchmarks.html"];

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
  if (url.pathname.startsWith("/api/") || url.pathname.startsWith("/models/")) return;

  if (url.pathname.startsWith("/vendor/") || url.pathname.startsWith("/icons/")) {
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
      .catch(() => caches.match(e.request).then((hit) => hit || caches.match("/index.html")))
  );
});
