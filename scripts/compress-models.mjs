#!/usr/bin/env node
// Pre-compress the model weights that gain from it, and write a manifest the
// service worker uses to swap in the .gz at fetch time (public/sw.js).
//
//   node scripts/compress-models.mjs            # public/models/*
//   node scripts/compress-models.mjs path/to/models
//
// Neither delivery path compresses on the wire: server.js sends files raw and
// the Hugging Face CDN does no content negotiation. The int8 files gzip to
// ~72%, JSON to ~20%; int4 and fp32 weights are ~93% and are skipped so the
// GPU path does not pay for a .gz that saves nothing. Output per model dir:
//   <file>.gz            next to each compressible file
//   compressed.json      { "<relative path>": { size, gz } }
// Both are build artifacts (gitignored) and go to Hugging Face with the rest
// (scripts/upload-models-hf.sh).
import { createReadStream, createWriteStream } from "node:fs";
import { readdir, stat, writeFile, rm } from "node:fs/promises";
import { join, relative } from "node:path";
import { pipeline } from "node:stream/promises";
import { createGzip } from "node:zlib";

const ROOT = process.argv[2] ?? join(new URL("..", import.meta.url).pathname, "public", "models");
const MAX_RATIO = 0.9;            // keep the .gz only if it is at most this fraction of the raw size
const CANDIDATE = /\.(onnx|json|txt)$/;

async function* walk(dir) {
  for (const e of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) yield* walk(p); else yield p;
  }
}

for (const modelDir of (await readdir(ROOT, { withFileTypes: true })).filter((e) => e.isDirectory()).map((e) => join(ROOT, e.name))) {
  const manifest = {};
  for await (const file of walk(modelDir)) {
    const rel = relative(modelDir, file);
    if (!CANDIDATE.test(rel) || rel === "compressed.json") continue;
    const size = (await stat(file)).size;
    const gzPath = file + ".gz";
    await pipeline(createReadStream(file), createGzip({ level: 9 }), createWriteStream(gzPath));
    const gz = (await stat(gzPath)).size;
    const ratio = gz / size;
    if (ratio > MAX_RATIO) { await rm(gzPath); console.log(`  skip ${rel}  ${(ratio * 100).toFixed(0)}%`); continue; }
    manifest[rel] = { size, gz };
    console.log(`  gz   ${rel}  ${(size / 2 ** 20).toFixed(1)} MB -> ${(gz / 2 ** 20).toFixed(1)} MB (${(ratio * 100).toFixed(0)}%)`);
  }
  await writeFile(join(modelDir, "compressed.json"), JSON.stringify(manifest, null, 1) + "\n");
  console.log(`${relative(process.cwd(), modelDir)}: ${Object.keys(manifest).length} compressed file(s)`);
}
