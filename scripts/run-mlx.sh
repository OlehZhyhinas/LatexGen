#!/usr/bin/env bash
# Start the optional MLX fast tier (Apple Silicon). The app's server probes
# this every ~20s and routes simple conversions to it when reachable.
set -euo pipefail
VENV="${VENV:-$HOME/.venvs/mlx}"
MODEL="${MLX_MODEL:-mlx-community/Qwen3-1.7B-4bit}"

if [ ! -x "$VENV/bin/mlx_lm.server" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q mlx-lm
fi
exec "$VENV/bin/mlx_lm.server" --model "$MODEL" --port "${MLX_PORT:-8080}"
