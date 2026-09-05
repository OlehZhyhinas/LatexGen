#!/usr/bin/env node
// Copy the published webnn-catalog IntelliTeX entry into
// public/models/intellitex-webnn/. Constants blobs (~994 MiB) stay out of git;
// the manifest already names Hugging Face URLs.
//
//   node scripts/vendor-intellitex-webnn.mjs \
//     [--src ../webnn-catalog-intellitex/families/intellitex-t5-220m] \
//     [--link-bins]
import { cpSync, mkdirSync, existsSync, linkSync, symlinkSync, unlinkSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const argv = process.argv.slice(2);
const o = {
  src: path.resolve(ROOT, "../webnn-catalog-intellitex/families/intellitex-t5-220m"),
  out: path.join(ROOT, "public/models/intellitex-webnn"),
  linkBins: false,
};
for (let i = 0; i < argv.length; i++) {
  const a = argv[i], take = () => argv[++i];
  if (a === "--src") o.src = path.resolve(take());
  else if (a === "--out") o.out = path.resolve(take());
  else if (a === "--link-bins") o.linkBins = true;
  else throw new Error(`unknown arg ${a}`);
}

const entryDir = path.join(o.src, "entries", "coreml-apple-m5-pro-macos26-chrome152");
if (!existsSync(path.join(entryDir, "entry.json"))) throw new Error(`no catalog entry at ${entryDir}`);
mkdirSync(o.out, { recursive: true });

cpSync(path.join(o.src, "family.json"), path.join(o.out, "family.json"));
const keep = new Set(["entry.json", "manifest.json"]);
for (const name of readdirSync(entryDir)) {
  if (keep.has(name) || /^recipe\./.test(name)) cpSync(path.join(entryDir, name), path.join(o.out, name));
}
const tokSrc = path.join(o.src, "tokenizer");
if (existsSync(tokSrc)) cpSync(tokSrc, path.join(o.out, "tokenizer"), { recursive: true });

if (o.linkBins) {
  for (const name of readdirSync(entryDir).filter((n) => n.endsWith(".bin"))) {
    const dest = path.join(o.out, name);
    try { unlinkSync(dest); } catch { /* missing is fine */ }
    try { linkSync(path.join(entryDir, name), dest); }
    catch { symlinkSync(path.join(entryDir, name), dest); }
  }
}
console.log(`wrote ${path.relative(ROOT, o.out)}`);
