import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { normalize, segment } from "../public/pdf-pipeline.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const root = path.resolve(__dirname, "..");

function loadJson(p) {
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

function loadJsonl(p) {
  return fs.readFileSync(p, "utf8")
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function sameSpans(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (!Array.isArray(a[i]) || !Array.isArray(b[i])) return false;
    if (a[i].length !== 2 || b[i].length !== 2) return false;
    if (a[i][0] !== b[i][0] || a[i][1] !== b[i][1]) return false;
  }
  return true;
}

function main() {
  const pdfRows = loadJson(path.join(root, "bench", "pdf-pastes.json"));
  const synthRows = loadJsonl(path.join(root, "bench", "synth-spans.jsonl"));
  const items = [
    ...pdfRows.map((r) => ({ id: r.id, text: r.input })),
    ...synthRows.map((r) => ({ id: r.id, text: r.text })),
  ];

  const js = {};
  for (const item of items) {
    const [clean] = normalize(item.text);
    js[item.id] = { clean, spans: segment(clean) };
  }

  const stamp = `${Date.now()}-${process.pid}`;
  const inputPath = path.join(root, "bench", `.tmp-pdf-pipeline-input-${stamp}.json`);
  const outputPath = path.join(root, "bench", `.tmp-pdf-pipeline-output-${stamp}.json`);
  fs.writeFileSync(inputPath, JSON.stringify(items), "utf8");

  const pyCode = [
    "import runpy,sys",
    "g=runpy.run_path('bench/dump-pdf-pipeline.py')",
    "g['dump'](sys.argv[1], sys.argv[2])",
  ].join(";");
  const py = spawnSync(".venv/bin/python", ["-c", pyCode, inputPath, outputPath], {
    cwd: root,
    encoding: "utf8",
  });

  let pyOut = null;
  try {
    if (py.status !== 0) {
      throw new Error(`python exited ${py.status}\n${py.stderr || py.stdout}`);
    }
    pyOut = loadJson(outputPath);
  } finally {
    try { fs.unlinkSync(inputPath); } catch {}
    try { fs.unlinkSync(outputPath); } catch {}
  }

  let sameClean = 0;
  let sameSpan = 0;
  const mismatches = [];
  for (const item of items) {
    const a = js[item.id];
    const b = pyOut[item.id];
    const cleanEq = !!b && a.clean === b.clean;
    const spansEq = !!b && sameSpans(a.spans, b.spans);
    if (cleanEq) sameClean++;
    if (spansEq) sameSpan++;
    if (!cleanEq || !spansEq) {
      mismatches.push({
        id: item.id,
        cleanEq,
        spansEq,
        js: a,
        py: b || { clean: null, spans: null },
      });
    }
  }

  console.log(`inputs: ${items.length}`);
  console.log(`identical clean: ${sameClean}`);
  console.log(`identical spans: ${sameSpan}`);
  console.log(`mismatches: ${mismatches.length}`);

  for (const m of mismatches.slice(0, 10)) {
    console.log(`\n--- ${m.id} ---`);
    console.log(`clean match: ${m.cleanEq}`);
    console.log(`spans match: ${m.spansEq}`);
    console.log(`js clean : ${JSON.stringify(m.js.clean)}`);
    console.log(`py clean : ${JSON.stringify(m.py.clean)}`);
    console.log(`js spans : ${JSON.stringify(m.js.spans)}`);
    console.log(`py spans : ${JSON.stringify(m.py.spans)}`);
  }

  if (mismatches.length) process.exitCode = 1;
}

main();
