#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

# ONNX Runtime GPU needs the CUDA/cuDNN 9 libraries already installed with the
# WSL PyTorch environment. TensorRT 8.6 needs its pinned cuDNN 8 libraries.
cuda_site_packages="${RTMW3D_CUDA_SITE_PACKAGES:-/home/dev/aigrandprix/.venv/lib/python3.12/site-packages}"
cuda_libs=""
if [[ -d "$cuda_site_packages/nvidia" ]]; then
  while IFS= read -r lib_dir; do
    if [[ -z "$cuda_libs" ]]; then
      cuda_libs="$lib_dir"
    else
      cuda_libs="$cuda_libs:$lib_dir"
    fi
  done < <(find "$cuda_site_packages/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print)
fi

trt_libs=""
for lib_dir in \
  "$script_dir/.venv/lib/python3.11/site-packages/tensorrt_libs" \
  "$script_dir/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib"; do
  if [[ -d "$lib_dir" ]]; then
    trt_libs="$trt_libs:$lib_dir"
  fi
done

export LD_LIBRARY_PATH="${trt_libs#:}:$cuda_libs:/usr/local/cuda/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export RTMW3D_DEVICE="${RTMW3D_DEVICE:-cuda}"
export RTMW3D_BACKEND="${RTMW3D_BACKEND:-tensorrt}"

if [[ "$RTMW3D_BACKEND" == "tensorrt" ]]; then
  ./prepare_tensorrt_engine.sh
fi

exec uv run python -m retargeting_demo.main \
  --host "${RETARGETING_HOST:-0.0.0.0}" \
  --port "${RETARGETING_PORT:-8000}" \
  "$@"
