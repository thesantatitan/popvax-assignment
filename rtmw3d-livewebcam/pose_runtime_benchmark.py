"""Measure RTMW3D-X ONNX Runtime CUDA lower bounds and I/O paths.

This benchmark deliberately bypasses detection, cropping, visualization, and
the live server. It runs a fixed preprocessed 1x3x384x288 tensor so runtime
and model costs can be separated from camera/application work.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnxruntime as ort

from rtmlib.tools.pose_estimation.post_processings import get_simcc_maximum3d


CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2


class CudaRuntime:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcudart.so.12")
        self.device_synchronize = self.lib.cudaDeviceSynchronize
        self.device_synchronize.argtypes = []
        self.device_synchronize.restype = ctypes.c_int
        self.malloc_fn = self.lib.cudaMalloc
        self.malloc_fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.malloc_fn.restype = ctypes.c_int
        self.free_fn = self.lib.cudaFree
        self.free_fn.argtypes = [ctypes.c_void_p]
        self.free_fn.restype = ctypes.c_int
        self.memcpy_fn = self.lib.cudaMemcpy
        self.memcpy_fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.memcpy_fn.restype = ctypes.c_int

    def sync(self) -> None:
        code = self.device_synchronize()
        if code:
            raise RuntimeError(f"cudaDeviceSynchronize failed with {code}")

    def malloc(self, size: int) -> int:
        pointer = ctypes.c_void_p()
        code = self.malloc_fn(ctypes.byref(pointer), size)
        if code:
            raise RuntimeError(f"cudaMalloc failed with {code} for {size} bytes")
        assert pointer.value is not None
        return int(pointer.value)

    def free(self, pointer: int) -> None:
        code = self.free_fn(ctypes.c_void_p(pointer))
        if code:
            raise RuntimeError(f"cudaFree failed with {code}")

    def memcpy(self, destination: int, source: int, size: int, kind: int) -> None:
        code = self.memcpy_fn(
            ctypes.c_void_p(destination),
            ctypes.c_void_p(source),
            size,
            kind,
        )
        if code:
            raise RuntimeError(f"cudaMemcpy failed with {code}")


class GpuMonitor:
    def __init__(self, interval_s: float = 0.2) -> None:
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.samples: list[dict[str, float]] = []
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=1,
                )
                values = [float(value.strip()) for value in result.stdout.split(",")]
                if len(values) == 3:
                    self.samples.append(
                        {
                            "utilization_pct": values[0],
                            "memory_used_mib": values[1],
                            "memory_total_mib": values[2],
                        }
                    )
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self.stop_event.wait(self.interval_s)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)

    def summary(self) -> dict[str, float | int | None]:
        if not self.samples:
            return {"samples": 0, "utilization_mean_pct": None, "memory_peak_mib": None}
        utilization = np.asarray([sample["utilization_pct"] for sample in self.samples])
        memory = np.asarray([sample["memory_used_mib"] for sample in self.samples])
        return {
            "samples": len(self.samples),
            "utilization_mean_pct": round(float(utilization.mean()), 2),
            "utilization_p95_pct": round(float(np.percentile(utilization, 95)), 2),
            "utilization_peak_pct": round(float(utilization.max()), 2),
            "memory_mean_mib": round(float(memory.mean()), 2),
            "memory_peak_mib": round(float(memory.max()), 2),
            "memory_total_mib": round(float(self.samples[-1]["memory_total_mib"]), 2),
        }


def summarize(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": round(float(values.mean()), 3),
        "p50_ms": round(float(np.percentile(values, 50)), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
        "min_ms": round(float(values.min()), 3),
        "max_ms": round(float(values.max()), 3),
        "fps_from_mean": round(float(1000.0 / values.mean()), 3),
        "samples": int(values.size),
    }


def measure(
    fn: Callable[[], Any],
    synchronizer: CudaRuntime,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, float], Any]:
    for _ in range(warmup):
        synchronizer.sync()
        fn()
        synchronizer.sync()
    samples: list[float] = []
    last: Any = None
    for _ in range(iterations):
        synchronizer.sync()
        started = time.perf_counter()
        last = fn()
        synchronizer.sync()
        samples.append((time.perf_counter() - started) * 1000.0)
    return summarize(samples), last


def make_session(
    model_path: str,
    profile: bool,
    profile_prefix: str,
    execution_mode: str,
    enable_cuda_graph: bool,
) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = (
        ort.ExecutionMode.ORT_PARALLEL
        if execution_mode == "parallel"
        else ort.ExecutionMode.ORT_SEQUENTIAL
    )
    options.enable_profiling = profile
    options.profile_file_prefix = profile_prefix
    provider_options = {
        "device_id": "0",
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": "1",
    }
    if enable_cuda_graph:
        provider_options["enable_cuda_graph"] = "1"
    return ort.InferenceSession(
        model_path,
        sess_options=options,
        providers=[("CUDAExecutionProvider", provider_options), "CPUExecutionProvider"],
    )


def profile_top_ops(profile_path: str) -> list[dict[str, float | str]]:
    events = json.loads(Path(profile_path).read_text())
    totals: defaultdict[str, float] = defaultdict(float)
    counts: defaultdict[str, int] = defaultdict(int)
    for event in events:
        if event.get("ph") != "X":
            continue
        args = event.get("args", {})
        provider = str(args.get("provider", ""))
        if "CUDA" not in provider:
            continue
        name = str(args.get("op_name") or event.get("name") or "unknown")
        totals[name] += float(event.get("dur", 0.0)) / 1000.0
        counts[name] += 1
    return [
        {"op": name, "total_ms": round(total, 3), "calls": counts[name]}
        for name, total in sorted(totals.items(), key=lambda item: item[1], reverse=True)[:20]
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    cuda = CudaRuntime()
    profile_prefix = str(Path(args.profile_prefix).resolve())
    session = make_session(
        args.model,
        profile=args.profile,
        profile_prefix=profile_prefix,
        execution_mode=args.execution_mode,
        enable_cuda_graph=args.cuda_graph,
    )
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()
    input_shape = tuple(int(dim) for dim in input_meta.shape)
    output_shapes = [tuple(int(dim) for dim in meta.shape) for meta in output_meta]
    rng = np.random.default_rng(1234)
    input_host = rng.standard_normal(input_shape, dtype=np.float32)
    input_name = input_meta.name
    output_names = [meta.name for meta in output_meta]

    if args.cuda_graph:
        session_run: dict[str, str] = {
            "status": "skipped; CUDA Graph requires I/O Binding"
        }
        last_outputs: list[np.ndarray] = []
        gpu_summary: dict[str, Any] = {}
    else:
        monitor = GpuMonitor()
        monitor.start()
        session_run, last_outputs = measure(
            lambda: session.run(None, {input_name: input_host}),
            cuda,
            args.warmup,
            args.iterations,
        )
        monitor.stop()
        gpu_summary = monitor.summary()

    input_device = cuda.malloc(input_host.nbytes)
    output_devices = [cuda.malloc(int(np.prod(shape)) * 4) for shape in output_shapes]
    output_hosts = [np.empty(shape, dtype=np.float32) for shape in output_shapes]
    try:
        h2d, _ = measure(
            lambda: cuda.memcpy(
                input_device,
                int(input_host.ctypes.data),
                input_host.nbytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ),
            cuda,
            args.warmup,
            args.iterations,
        )

        io_binding = session.io_binding()
        io_binding.bind_input(
            input_name,
            "cuda",
            0,
            np.float32,
            input_shape,
            input_device,
        )
        for name, shape, pointer in zip(output_names, output_shapes, output_devices):
            io_binding.bind_output(name, "cuda", 0, np.float32, shape, pointer)

        def run_bound() -> None:
            session.run_with_iobinding(io_binding)

        bound, _ = measure(
            run_bound,
            cuda,
            args.warmup,
            args.iterations,
        )

        def copy_outputs() -> None:
            for host, pointer in zip(output_hosts, output_devices):
                cuda.memcpy(
                    int(host.ctypes.data),
                    pointer,
                    host.nbytes,
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                )

        d2h, _ = measure(
            copy_outputs,
            cuda,
            args.warmup,
            args.iterations,
        )
        decode_inputs = last_outputs or output_hosts
        decode, _ = measure(
            lambda: get_simcc_maximum3d(
                decode_inputs[0], decode_inputs[1], decode_inputs[2]
            ),
            cuda,
            args.warmup,
            args.iterations,
        )
        if args.cuda_graph:
            gpu_summary = {"note": "GPU utilization was sampled by the caller for graph mode"}
    finally:
        cuda.free(input_device)
        for pointer in output_devices:
            cuda.free(pointer)

    profile_path = session.end_profiling() if args.profile else None
    result: dict[str, Any] = {
        "configuration": {
            "model": args.model,
            "input_shape": list(input_shape),
            "output_shapes": [list(shape) for shape in output_shapes],
            "warmup": args.warmup,
            "iterations": args.iterations,
            "execution_mode": args.execution_mode,
            "graph_optimization": "ORT_ENABLE_ALL",
            "cuda_graph_requested": args.cuda_graph,
            "providers": session.get_providers(),
        },
        "session_run": session_run,
        "h2d_copy": h2d,
        "gpu_execution_iobinding": bound,
        "d2h_copy": d2h,
        "decoding": decode,
        "gpu": gpu_summary,
        "profile_path": profile_path,
        "profile_top_cuda_ops": profile_top_ops(profile_path) if profile_path else [],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-prefix", default="/tmp/rtmw3d_ort_profile")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--execution-mode", choices=("sequential", "parallel"), default="sequential")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
