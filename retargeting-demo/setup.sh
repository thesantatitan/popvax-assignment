#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

uv sync --extra gpu --extra tensorrt --extra dev

# rtmlib depends on the CPU distribution named `onnxruntime`. Both packages
# expose the same Python module, so reinstall the pinned GPU wheel last.
if [[ "$(uname -s)" == "Linux" ]]; then
  uv pip install --reinstall "onnxruntime-gpu==1.26.0"
  ./prepare_tensorrt_engine.sh
fi

uv run python verify_runtime.py
uv run pytest -q
