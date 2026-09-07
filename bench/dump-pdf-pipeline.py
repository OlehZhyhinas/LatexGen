#!/usr/bin/env python3
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
PIPE = HERE / "pdf_pipeline.py"


def _load_pipeline():
    spec = importlib.util.spec_from_file_location("pdf_pipeline", PIPE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pdf_pipeline = _load_pipeline()


def dump(input_path, output_path):
    rows = json.loads(pathlib.Path(input_path).read_text())
    out = {}
    for row in rows:
        text = row.get("text", "")
        clean, _ = pdf_pipeline.normalize(text)
        spans = [[int(a), int(b)] for a, b in pdf_pipeline.segment(clean)]
        out[row["id"]] = {"clean": clean, "spans": spans}
    pathlib.Path(output_path).write_text(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: dump-pdf-pipeline.py INPUT_JSON OUTPUT_JSON")
    dump(sys.argv[1], sys.argv[2])
