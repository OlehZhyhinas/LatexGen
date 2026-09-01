#!/usr/bin/env bash
# Rebuild public/models/intellitex from scratch: download IntelliTeX (MIT,
# CodeT5+ 220M), export to ONNX, quantize to int8 for transformers.js.
# The converted weights are committed via git-lfs, so you only need this if
# you're regenerating them (e.g. after a base-model update).
#
# Requires Python 3.12 (3.14's tokenizers wheel gap breaks the old optimum).
set -euo pipefail

PY="${PY:-python3.12}"
VENV="${VENV:-$HOME/.venvs/onnx-export}"
OUT="$(cd "$(dirname "$0")/.." && pwd)/public/models/intellitex"
WORK=$(mktemp -d)

if [ ! -x "$VENV/bin/python" ]; then
  "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install -q "optimum[exporters]==1.23.3" "transformers==4.46.3" \
    torch==2.5.1 onnx onnxruntime onnxscript sentencepiece
  "$VENV/bin/pip" uninstall -q -y torchvision torchaudio 2>/dev/null || true
fi

echo "==> exporting to ONNX"
"$VENV/bin/optimum-cli" export onnx -m duanxianpi/IntelliTex \
  --task text2text-generation-with-past "$WORK/onnx"

echo "==> quantizing to int8"
"$VENV/bin/python" - "$WORK/onnx" <<'EOF'
import os, sys
from onnxruntime.quantization import quantize_dynamic, QuantType
src = sys.argv[1]
os.makedirs(f"{src}/onnx", exist_ok=True)
for a, b in [("encoder_model.onnx", "onnx/encoder_model_quantized.onnx"),
             ("decoder_model_merged.onnx", "onnx/decoder_model_merged_quantized.onnx")]:
    quantize_dynamic(f"{src}/{a}", f"{src}/{b}", weight_type=QuantType.QUInt8,
                     extra_options={"EnableSubgraph": True})
    print(b, round(os.path.getsize(f'{src}/{b}')/2**20), "MB")
EOF

echo "==> installing into $OUT"
mkdir -p "$OUT/onnx"
cp "$WORK"/onnx/{config.json,generation_config.json,tokenizer.json,tokenizer_config.json,special_tokens_map.json,vocab.json,merges.txt} "$OUT/"
cp "$WORK"/onnx/onnx/*.onnx "$OUT/onnx/"
rm -rf "$WORK"
echo "done."
