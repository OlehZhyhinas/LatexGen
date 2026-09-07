#!/usr/bin/env node
// Vendor the three patched WebLLM 0.2.84 bundles the qwen3-webllm catalog
// points at into public/vendor/webllm/ (same-origin, required by the CSP —
// dynamic import() of a blob:/https: module is not allowed). Weights are
// untouched: only these bundles and the per-model .wasm libs (streamed
// straight from the catalog by the browser, never vendored) differ from
// stock WebLLM. Node built-ins only: fetch, node:crypto, node:fs.
//
//   node scripts/vendor-qwen3-webllm.mjs           # download missing/stale bundles
//   node scripts/vendor-qwen3-webllm.mjs --check   # verify only; exit 1 on mismatch/missing
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const CATALOG_PATH = path.join(ROOT, "public/models/qwen3-webllm/catalog.json");
const OUT_DIR = path.join(ROOT, "public/vendor/webllm");

const checkOnly = process.argv.includes("--check");

const catalog = JSON.parse(readFileSync(CATALOG_PATH, "utf8"));

const sha256File = (p) => {
  const h = createHash("sha256");
  h.update(readFileSync(p));
  return h.digest("hex");
};

// null when ok, else a short human reason.
function mismatch(file, bytes, sha256) {
  const dest = path.join(OUT_DIR, file);
  if (!existsSync(dest)) return "missing";
  const gotBytes = readFileSync(dest).byteLength;
  if (gotBytes !== bytes) return `bytes ${gotBytes} != ${bytes}`;
  const gotSha256 = sha256File(dest);
  if (gotSha256 !== sha256) return `sha256 ${gotSha256.slice(0, 12)}… != ${sha256.slice(0, 12)}…`;
  return null;
}

async function download(url, dest) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} fetching ${url}`);
  writeFileSync(dest, Buffer.from(await res.arrayBuffer()));
}

const runtimes = Object.entries(catalog.runtimes).map(([id, r]) => ({ id, ...r }));
const bad = [];

for (const { id, file, bytes, sha256, url } of runtimes) {
  const before = mismatch(file, bytes, sha256);
  if (!before) {
    console.log(`${id}: ok (${file})`);
    continue;
  }
  if (checkOnly) {
    console.log(`${id}: ${before} (${file})`);
    bad.push(id);
    continue;
  }
  console.log(`${id}: ${before} — downloading ${file}…`);
  mkdirSync(OUT_DIR, { recursive: true });
  await download(url, path.join(OUT_DIR, file));
  const after = mismatch(file, bytes, sha256);
  if (after) {
    console.log(`${id}: FAILED after download — ${after}`);
    bad.push(id);
  } else {
    console.log(`${id}: downloaded and verified (${file})`);
  }
}

if (bad.length) {
  console.error(`${checkOnly ? "check" : "vendor"} failed for: ${bad.join(", ")}`);
  process.exitCode = 1;
} else {
  console.log(checkOnly ? "all runtimes verified" : "all runtimes vendored and verified");
}
