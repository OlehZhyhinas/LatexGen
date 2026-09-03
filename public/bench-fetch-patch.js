// Classic script, loaded before bench-load.js: transformers.js binds
// globalThis.fetch when its module evaluates, so the wrapper must be in
// place first. In cold mode weight requests bypass the HTTP cache too (the
// files are served immutable, so a plain fetch would come back from disk).
(() => {
  const mode = new URLSearchParams(location.search).get("mode") ?? "cold";
  const stats = (globalThis.__benchNet = { requests: 0, bytes: 0, files: [] });
  const real = globalThis.fetch.bind(globalThis);
  globalThis.fetch = async (input, init = {}) => {
    const url = typeof input === "string" ? input : input instanceof Request ? input.url : String(input);
    const isWeight = /\/(models\/|resolve\/)/.test(url);
    if (isWeight && mode === "cold") init = { ...init, cache: "reload" };
    const res = await real(input, init);
    if (isWeight && res.status === 200) {
      const len = Number(res.headers.get("content-length") || 0);
      stats.requests++; stats.bytes += len;
      stats.files.push({ file: url.split("/").slice(-2).join("/"), len, encoding: res.headers.get("content-encoding"), type: res.headers.get("content-type") });
    }
    return res;
  };
})();
