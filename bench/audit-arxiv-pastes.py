#!/usr/bin/env python3
"""Audit arXiv paste outputs against cleanup/alignment invariants."""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / "arxiv" / "cache"


def load_build_module():
    script_path = ROOT / "build-arxiv-pastes.py"
    spec = importlib.util.spec_from_file_location("build_arxiv_pastes", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_custom_macros(mod, aid: str, cache: dict[str, set[str]]) -> set[str]:
    if aid in cache:
        return cache[aid]
    pdir = CACHE_ROOT / aid
    src_blob = pdir / "source.bin"
    source_dir = pdir / "source"
    macros: set[str] = set()
    if src_blob.exists():
        mod.ensure_source_tree(src_blob, source_dir)
    if source_dir.exists():
        tex_map = mod.load_tex_files(source_dir)
        style_map = mod.load_style_files(source_dir)
        if tex_map:
            main_tex = mod.choose_main_tex(tex_map)
            if main_tex is not None:
                expanded = mod.expand_inputs(main_tex, tex_map)
                doc_idx = expanded.find("\\begin{document}")
                preamble = expanded[:doc_idx] if doc_idx >= 0 else expanded
                macros.update(mod.extract_custom_macros(preamble))
        for sty_txt in style_map.values():
            macros.update(mod.extract_custom_macros(sty_txt))
    cache[aid] = macros
    return macros


def math_commands(inner: str) -> set[str]:
    return set(re.findall(r"\\([A-Za-z@]+)", inner))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "arxiv-pastes.json"))
    args = ap.parse_args()

    mod = load_build_module()
    rows = json.loads(Path(args.input).read_text())
    if not isinstance(rows, list):
        raise RuntimeError("input JSON must be a list")

    violations = collections.Counter()
    bad_ids: dict[str, list[str]] = collections.defaultdict(list)
    by_category = collections.Counter()
    by_extractor = collections.Counter()
    macro_cache: dict[str, set[str]] = {}

    for it in rows:
        item_id = str(it.get("id", ""))
        inp = str(it.get("input", "")).strip()
        ref = str(it.get("reference", "")).strip()
        meta = it.get("meta", {}) if isinstance(it.get("meta"), dict) else {}
        aid = str(meta.get("paper_id", "")).strip()
        by_category[str(meta.get("category", "unknown"))] += 1
        by_extractor[str(meta.get("extractor", "unknown"))] += 1

        # (1) reference cleanup and no commands outside math
        if not mod.reference_delimiters_ok(ref):
            violations["cleanup_delimiters"] += 1
            bad_ids["cleanup_delimiters"].append(item_id)
        if not mod.reference_cleanup_ok(ref):
            violations["cleanup_commands"] += 1
            bad_ids["cleanup_commands"].append(item_id)

        # parse math fragments once for checks (2) and (3)
        frags = mod.parse_math_fragments(ref)
        if frags is None:
            violations["math_parse"] += 1
            bad_ids["math_parse"].append(item_id)
            continue

        # (2) no author-defined macros in math
        if aid:
            custom_macros = load_custom_macros(mod, aid, macro_cache)
            used = set()
            for frag in frags:
                used |= (math_commands(frag.inner) & custom_macros)
            if used:
                violations["author_macros"] += 1
                bad_ids["author_macros"].append(item_id)

        # (3) at least one real math fragment
        if not any(mod.math_fragment_has_real_signal(frag.inner) for frag in frags):
            violations["real_math"] += 1
            bad_ids["real_math"].append(item_id)

        # (4) anchor-bounded input alignment and overlap
        ref_prose = str(meta.get("reference_prose", "")).strip() or mod.reference_without_math(ref)
        aligned = mod.align_region(inp, ref_prose)
        if aligned is None:
            violations["alignment"] += 1
            bad_ids["alignment"].append(item_id)
        elif aligned.get("region", "").strip() != inp:
            violations["alignment_bounds"] += 1
            bad_ids["alignment_bounds"].append(item_id)

    print(f"items: {len(rows)}")
    print("category_counts:")
    for cat, n in sorted(by_category.items()):
        print(f"- {cat}: {n}")
    print("extractor_counts:")
    for ext, n in sorted(by_extractor.items()):
        print(f"- {ext}: {n}")
    print("violations:")
    if violations:
        for key in sorted(violations):
            print(f"- {key}: {violations[key]}")
            sample = ", ".join(bad_ids[key][:5])
            if sample:
                print(f"  sample_ids: {sample}")
    else:
        print("- none")

    assert sum(violations.values()) == 0, "audit violations found"


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
