#!/usr/bin/env bash
# Regenerate the WebGPU int4 model variants (not committed: ~680MB, over the
# LFS quota). Requires the export venv from scripts/setup-export-venv.sh.
#   IntelliTeX int4 : built from the plain ONNX export with MatMulNBits (block 32)
#   Texify int4     : downloaded from Xenova/texify (upstream ships it)
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="${EXPORT_PY:-$HOME/.venvs/onnx-export/bin/python}"
WORK="${WORK:-/tmp/latexgen-export}"
mkdir -p "$WORK"

echo "==> IntelliTeX: plain ONNX export (fp32)"
if [ ! -f "$WORK/intellitex-onnx/encoder_model.onnx" ]; then
  "$(dirname "$PY")/optimum-cli" export onnx -m duanxianpi/IntelliTex --task text2text-generation-with-past "$WORK/intellitex-onnx"
fi
echo "==> IntelliTeX: int4 (MatMulNBits) for WebGPU"
"$PY" - <<PYEOF
import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
for part in ["encoder_model", "decoder_model_merged"]:
    q = MatMulNBitsQuantizer(onnx.load("$WORK/intellitex-onnx/%s.onnx" % part), block_size=32, is_symmetric=True, accuracy_level=4)
    q.process(); q.model.save_model_to_file("$DIR/public/models/intellitex/onnx/%s_q4.onnx" % part, use_external_data_format=False)
    print("wrote", part + "_q4.onnx")
PYEOF

echo "==> Texify: int4 from upstream"
for f in encoder_model_q4.onnx decoder_model_merged_q4.onnx; do
  curl -sL "https://huggingface.co/Xenova/texify/resolve/main/onnx/$f" -o "$DIR/public/models/texify/onnx/$f"
  echo "downloaded $f"
done
echo "done"
