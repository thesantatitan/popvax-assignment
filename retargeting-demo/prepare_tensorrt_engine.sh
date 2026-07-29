#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

exec uv run \
  --with "torch==2.5.1" \
  --with "mmpose==1.3.2" \
  --with "mmdet==3.3.0" \
  --with "mmengine==0.10.7" \
  --with "mmcv-lite==2.1.0" \
  --with "onnx>=1.17" \
  --with "addict" \
  --with "json-tricks" \
  --with "matplotlib" \
  --with "munkres" \
  --with "platformdirs" \
  --with "pyyaml" \
  --with "rich" \
  --with "scipy" \
  --with "termcolor" \
  --with "yapf" \
  python prepare_tensorrt_engine.py "$@"
