#!/usr/bin/env node
// Drive bench-load.html in a dedicated Chrome profile and collect /api/bench rows.
// Records host loadavg around each run so contended-machine noise is visible.
import { spawn, execSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { loadavg, cpus, tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const PORT = process.env.PORT || "8001";
const ORIGIN = `http://127.0.0.1:${PORT}`;
const CHROME = process.env.CHROME
  || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const FLAGS = [
  "--no-first-run",
  "--no-default-browser-check",
  "--disable-sync",
  "--disable-background-networking",
  "--enable-unsafe-webgpu",
  "--enable-features=WebMachineLearningNeuralNetwork,WebMachineLearningNeuralNetworkExperimentalFeatures,WebNNCoreML,WebGPUExperimentalFeatures",
  "--enable-dawn-features=allow_unsafe_apis",
];

const ALL_JOBS = [
  { label: "cold-hf", model: "intellitex", device: "webgpu", dtype: "q4", timeoutMs: 300_000 },
  { label: "cold-hf", model: "texify", device: "webgpu", dtype: "q4", timeoutMs: 300_000 },
  { label: "cold-hf", model: "texo", device: "wasm", dtype: "fp32", timeoutMs: 300_000 },
  { label: "cold-hf", model: "texo", device: "webnn", dtype: "fp16", timeoutMs: 300_000 },
  { label: "cold-hf", model: "texify", device: "webnn", dtype: "fp16", timeoutMs: 600_000 },
  { label: "cold-hf", model: "intellitex", device: "webnn", dtype: "fp16", timeoutMs: 900_000 },
];
const only = new Set((process.env.ONLY || "").split(",").filter(Boolean));
const JOBS = only.size
  ? ALL_JOBS.filter((job) => only.has(`${job.model}:${job.device}`))
  : ALL_JOBS;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const hostSnap = () => {
  let chromeHelpers = 0;
  try {
    chromeHelpers = execSync("pgrep -lf 'Google Chrome Helper' | wc -l", { encoding: "utf8" }).trim();
  } catch { /* ignore */ }
  return {
    at: new Date().toISOString(),
    loadavg: loadavg(),
    cpuCount: cpus().length,
    chromeHelpers: Number(chromeHelpers),
    contended: loadavg()[0] > cpus().length * 0.6,
  };
};

async function api(method, path = "/api/bench", body = null) {
  const r = await fetch(`${ORIGIN}${path}`, {
    method,
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (method === "GET") return r.json();
  return r;
}

function urlFor(job) {
  const p = new URLSearchParams({
    label: job.label,
    model: job.model,
    device: job.device,
    dtype: job.dtype,
    src: "hf",
    mode: "cold",
    items: "0",
  });
  return `${ORIGIN}/bench-load.html?${p}`;
}

let chromeProc = null, activeProfile = null;
function startChrome(url) {
  stopChrome();
  activeProfile = mkdtempSync(join(tmpdir(), "latexgen-load-cold-"));
  chromeProc = spawn(CHROME, [`--user-data-dir=${activeProfile}`, ...FLAGS, url], { stdio: "ignore" });
  return activeProfile;
}
function stopChrome() {
  if (!chromeProc) return;
  try { chromeProc.kill(); } catch { /* ignore */ }
  chromeProc = null;
}

async function waitForRow(beforeCount, timeoutMs) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    const rows = await api("GET");
    if (rows.length > beforeCount) {
      const row = rows[rows.length - 1];
      if (row.tier === "load" && (row.loadMs != null || row.error)) return row;
    }
    await sleep(2000);
  }
  throw new Error(`timeout waiting for bench row after ${timeoutMs}ms`);
}

const collected = [];
try {
  await api("GET", "/api/health");
} catch {
  console.error(`server not reachable at ${ORIGIN}; start with PORT=${PORT} npm start`);
  process.exit(1);
}

await api("DELETE");
for (const job of JOBS) {
  const before = hostSnap();
  console.log(JSON.stringify({ starting: job, host: before }));
  const rowsBefore = (await api("GET")).length;
  const profile = startChrome(urlFor(job));
  let row;
  try {
    row = await waitForRow(rowsBefore, job.timeoutMs);
  } catch (e) {
    row = { tier: "load", ...job, src: "hf", mode: "cold", error: String(e) };
  }
  const after = hostSnap();
  row.runnerHost = { before, after, isolation: "new regular Chrome user-data-dir per row" };
  collected.push(row);
  console.log(JSON.stringify({
    done: job.model + ":" + job.device,
    loadMs: row.loadMs,
    warmupMs: row.warmupMs,
    firstGenMs: row.firstGenMs,
    secondGenMs: row.secondGenMs,
    coldToFirstMs: row.coldToFirstMs,
    error: row.error,
    loadavg: after.loadavg,
    contended: after.contended,
  }));
  stopChrome();
  await sleep(2000);
  try { rmSync(profile, { recursive: true, force: true }); } catch { /* Chrome may still be releasing files */ }
}
stopChrome();

const out = process.env.LOAD_BENCH_OUT
  ? join(ROOT, process.env.LOAD_BENCH_OUT)
  : join(ROOT, "bench", `results-load-cold-hf-${new Date().toISOString().slice(0, 10)}.json`);
writeFileSync(out, JSON.stringify(collected, null, 1) + "\n");
console.log("wrote", out);
