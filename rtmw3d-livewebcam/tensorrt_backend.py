"""TensorRT 8.6 RTMW3D-L pose backend for the WSL live server."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

import numpy as np
from rtmlib import RTMPose3d as RtmlibRTMPose3d
from rtmlib import YOLOX


CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2


class _CudaRuntime:
    """Small CUDA Runtime wrapper used to avoid a PyCUDA dependency."""

    def __init__(self) -> None:
        self.lib = ctypes.CDLL(os.getenv("RTMW3D_CUDART_LIBRARY", "libcudart.so.12"))
        self._sync = self.lib.cudaDeviceSynchronize
        self._sync.argtypes = []
        self._sync.restype = ctypes.c_int
        self._malloc = self.lib.cudaMalloc
        self._malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._malloc.restype = ctypes.c_int
        self._free = self.lib.cudaFree
        self._free.argtypes = [ctypes.c_void_p]
        self._free.restype = ctypes.c_int
        self._memcpy = self.lib.cudaMemcpy
        self._memcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self._memcpy.restype = ctypes.c_int

    def synchronize(self) -> None:
        code = self._sync()
        if code:
            raise RuntimeError(f"cudaDeviceSynchronize failed with code {code}")

    def malloc(self, size: int) -> int:
        pointer = ctypes.c_void_p()
        code = self._malloc(ctypes.byref(pointer), size)
        if code or pointer.value is None:
            raise RuntimeError(f"cudaMalloc failed with code {code} for {size} bytes")
        return int(pointer.value)

    def free(self, pointer: int) -> None:
        code = self._free(ctypes.c_void_p(pointer))
        if code:
            raise RuntimeError(f"cudaFree failed with code {code}")

    def memcpy(self, destination: int, source: int, size: int, kind: int) -> None:
        code = self._memcpy(
            ctypes.c_void_p(destination),
            ctypes.c_void_p(source),
            size,
            kind,
        )
        if code:
            raise RuntimeError(f"cudaMemcpy failed with code {code}")


class RTMPose3d(RtmlibRTMPose3d):
    """rtmlib-compatible RTMW3D-L pose model backed by TensorRT 8.6."""

    def __init__(
        self,
        onnx_model: str,
        model_input_size: tuple = (288, 384),
        mean: tuple = (123.675, 116.28, 103.53),
        std: tuple = (58.395, 57.12, 57.375),
        to_openpose: bool = False,
        backend: str = "tensorrt",
        device: str = "cuda",
        z_range: float | None = None,
    ) -> None:
        if device != "cuda":
            raise ValueError("The TensorRT RTMW3D-L backend requires device='cuda'.")

        engine_path = Path(onnx_model).expanduser()
        if not engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        try:
            import tensorrt_bindings as trt
        except ImportError as exc:  # pragma: no cover - depends on WSL-only extra.
            raise RuntimeError(
                "TensorRT 8.6 is not installed. Run `uv sync --extra gpu "
                "--extra tensorrt` in the WSL demo directory."
            ) from exc

        self.onnx_model = str(engine_path)
        self.model_input_size = model_input_size
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.backend = backend
        self.device = device
        self.to_openpose = to_openpose
        self.z_range = z_range if z_range is not None else 2.1744869
        self._cuda = _CudaRuntime()
        self._closed = False

        logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(logger)
        with engine_path.open("rb") as handle:
            self._engine = self._runtime.deserialize_cuda_engine(handle.read())
        if self._engine is None:
            raise RuntimeError(f"TensorRT could not deserialize {engine_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("TensorRT could not create an execution context")

        self._bindings: list[int] = [0] * self._engine.num_bindings
        self._output_bindings: list[tuple[int, np.ndarray]] = []
        self._input_shape: tuple[int, ...] | None = None
        try:
            for index in range(self._engine.num_bindings):
                shape = tuple(int(dim) for dim in self._engine.get_binding_shape(index))
                dtype = self._numpy_dtype(self._engine.get_binding_dtype(index), trt)
                if dtype != np.dtype(np.float32):
                    raise RuntimeError(
                        f"RTMW3D-L TensorRT engine must be FP32, got {dtype}"
                    )
                pointer = self._cuda.malloc(int(np.prod(shape)) * dtype.itemsize)
                self._bindings[index] = pointer
                if self._engine.binding_is_input(index):
                    self._input_shape = shape
                else:
                    self._output_bindings.append(
                        (pointer, np.empty(shape, dtype=dtype))
                    )
        except Exception:
            self.close()
            raise

        if self._input_shape != (1, 3, 384, 288):
            self.close()
            raise RuntimeError(
                f"Unexpected RTMW3D-L TensorRT input shape: {self._input_shape}"
            )
        if [host.shape for _, host in self._output_bindings] != [
            (1, 133, 576),
            (1, 133, 768),
            (1, 133, 576),
        ]:
            self.close()
            raise RuntimeError("Unexpected RTMW3D-L TensorRT output shapes")

        print(f"load {engine_path} with TensorRT 8.6 FP32 backend")

    @staticmethod
    def _numpy_dtype(dtype: Any, trt: Any) -> np.dtype:
        if dtype == trt.DataType.FLOAT:
            return np.dtype(np.float32)
        if dtype == trt.DataType.HALF:
            return np.dtype(np.float16)
        raise RuntimeError(f"Unsupported TensorRT binding dtype: {dtype}")

    def inference(self, img: np.ndarray) -> list[np.ndarray]:
        chw = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)[None]
        if chw.shape != self._input_shape:
            raise ValueError(
                f"Expected RTMW3D-L input {self._input_shape}, got {chw.shape}"
            )

        input_pointer = next(
            pointer
            for index, pointer in enumerate(self._bindings)
            if self._engine.binding_is_input(index)
        )
        self._cuda.memcpy(
            input_pointer,
            int(chw.ctypes.data),
            chw.nbytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
        if not self._context.execute_v2(self._bindings):
            raise RuntimeError("TensorRT execution failed")
        self._cuda.synchronize()

        for pointer, host in self._output_bindings:
            self._cuda.memcpy(
                int(host.ctypes.data),
                pointer,
                host.nbytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        self._cuda.synchronize()
        return [host for _, host in self._output_bindings]

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        for pointer in getattr(self, "_bindings", []):
            if pointer:
                try:
                    self._cuda.free(pointer)
                except Exception:
                    pass
        self._bindings = []
        self._output_bindings = []
        self._context = None
        self._engine = None
        self._runtime = None

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown path.
        try:
            self.close()
        except Exception:
            pass


class TensorRTWholebody3d:
    """rtmlib solution preserving YOLOX detection and replacing only pose."""

    def __init__(
        self,
        det: str,
        det_input_size: tuple,
        pose: str,
        pose_input_size: tuple = (288, 384),
        mode: str = "balanced",
        to_openpose: bool = False,
        backend: str = "tensorrt",
        device: str = "cuda",
    ) -> None:
        del mode, backend
        detector_backend = os.getenv("RTMW3D_DETECTOR_BACKEND", "onnxruntime")
        self.det_model = YOLOX(
            det,
            model_input_size=det_input_size,
            backend=detector_backend,
            device=device,
        )
        self.pose_model = RTMPose3d(
            pose,
            model_input_size=pose_input_size,
            to_openpose=to_openpose,
            backend="tensorrt",
            device=device,
        )

