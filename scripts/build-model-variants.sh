#!/usr/bin/env bash
# Regenerate the WebGPU int4 model variants (not committed: ~680MB, over the
# LFS quota). Requires the export venv from scripts/setup-export-venv.sh.
#   IntelliTeX int4 : built from the plain ONNX export with MatMulNBits (block 32)
#   Texify int4     : downloaded from Xenova/texify (upstream ships it)
# MatMulNBits leaves the token-embedding tables in fp32 (they feed a Gather,
# not a MatMul), which made the int4 files bigger than the int8 ones; both
# variants are then passed through scripts/quantize-embeddings.py (issue #1).
# Finally the compressible files get .gz siblings for the service worker
# (scripts/compress-models.mjs, issue #2).
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
    q.process(); q.model.save_model_to_file("$WORK/intellitex-onnx/%s_q4_fp32emb.onnx" % part, use_external_data_format=False)
    print("wrote", part + "_q4_fp32emb.onnx")
PYEOF
for part in encoder_model decoder_model_merged; do
  "$PY" "$DIR/scripts/quantize-embeddings.py" "$WORK/intellitex-onnx/${part}_q4_fp32emb.onnx" "$DIR/public/models/intellitex/onnx/${part}_q4.onnx"
done

echo "==> Texify: int4 from upstream"
for f in encoder_model_q4.onnx decoder_model_merged_q4.onnx; do
  curl -sL "https://huggingface.co/Xenova/texify/resolve/main/onnx/$f" -o "$WORK/texify-$f"
  echo "downloaded $f"
  "$PY" "$DIR/scripts/quantize-embeddings.py" "$WORK/texify-$f" "$DIR/public/models/texify/onnx/$f"
done

echo "==> pre-compressing weights for the service worker"
node "$DIR/scripts/compress-models.mjs"
echo "done"
