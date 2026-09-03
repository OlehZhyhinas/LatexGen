#!/usr/bin/env bash
# Publish the ONNX weights to a Hugging Face model repo, which is what the
# static build streams them from (a static host cannot serve 1.3 GB, and
# GitHub Pages serves Git LFS pointers rather than the files themselves).
#
# Run once, and again whenever the weights change:
#   scripts/upload-models-hf.sh
#
# Needs the Hugging Face CLI and a write token:
#   brew install hf     (or: pipx install "huggingface_hub[cli]")
#   hf auth login
set -euo pipefail

REPO="${MODEL_REPO:-ozhyhinas/latexgen-models}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/public/models"

command -v hf >/dev/null || { echo "hf CLI not found: brew install hf" >&2; exit 1; }
hf auth whoami >/dev/null 2>&1 || { echo "not logged in: hf auth login (needs a write token)" >&2; exit 1; }

# The q4 variants are build artifacts (scripts/build-model-variants.sh) and are
# gitignored, but the static build's WebGPU path needs them, so they must be
# present locally before uploading.
for f in intellitex/onnx/encoder_model_q4.onnx intellitex/onnx/decoder_model_merged_q4.onnx \
         texify/onnx/encoder_model_q4.onnx texify/onnx/decoder_model_merged_q4.onnx; do
  [ -f "$SRC/$f" ] || { echo "missing $f — run scripts/build-model-variants.sh first" >&2; exit 1; }
done

hf repos create "$REPO" --repo-type model --exist-ok

# One commit per model keeps each upload resumable.
for model in intellitex texo texify; do
  echo "== $model"
  hf upload "$REPO" "$SRC/$model" "$model" \
    --repo-type model \
    --commit-message "LatexGen: $model weights"
done

echo
echo "Done. The static build reads them as:"
echo "  https://huggingface.co/$REPO/resolve/main/<model>/onnx/<file>"
echo "If REPO differs from the default, set MODEL_REPO for scripts/build-static.mjs too."
