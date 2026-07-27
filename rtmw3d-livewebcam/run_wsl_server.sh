#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

# The WSL CUDA machine already has the CUDA/cuDNN libraries that ship with
# its PyTorch environment. Add them to this uv environment without copying
# large runtime libraries into the demo repository.
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

# TensorRT 8.6 uses cuDNN 8 while the PyTorch environment may also contain
# cuDNN 9. Put the explicit TensorRT/cuDNN-8 directories first for the live
# TensorRT backend; the regular CUDA libraries remain available afterwards.
trt_libs=""
for lib_dir in \
    "$script_dir/.venv/lib/python3.11/site-packages/tensorrt_libs" \
    "$script_dir/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib"; do
    if [[ -d "$lib_dir" ]]; then
        trt_libs="$trt_libs:$lib_dir"
    fi
done

export LD_LIBRARY_PATH="${trt_libs#:}:$cuda_libs:/usr/local/cuda/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export RTMW3D_DEVICE="${RTMW3D_DEVICE:-cuda}"
export RTMW3D_BACKEND="${RTMW3D_BACKEND:-tensorrt}"
export RTMW3D_HOST="${RTMW3D_HOST:-127.0.0.1}"
export RTMW3D_PORT="${RTMW3D_PORT:-8000}"

exec uv run uvicorn server:app --host "$RTMW3D_HOST" --port "$RTMW3D_PORT"
