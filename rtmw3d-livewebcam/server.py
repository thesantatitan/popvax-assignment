"""WebSocket server for running RTMW3D remotely while viewing it in a browser."""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from rtmlib import Custom, PoseTracker, Wholebody3d, draw_skeleton

from demo import as_arrays, draw_3d_inset
from roi_tracker import PersistentRoiPoseTracker
from tensorrt_backend import TensorRTWholebody3d


ROOT = Path(__file__).resolve().parent
WEB_INDEX = ROOT / "web" / "index.html"
YOLOX_NANO_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "onnx_sdk/yolox_nano_8xb8-300e_humanart-40f6f0d0.zip"
)


def _requested_device() -> str:
    requested = os.getenv("RTMW3D_DEVICE", "cuda").strip().lower()
    if requested == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
        print(
            "CUDAExecutionProvider is not available in this environment; "
            "falling back to CPU. Install onnxruntime-gpu to enable WSL GPU inference."
        )
        return "cpu"
    return requested


def _make_tracker() -> tuple[PoseTracker, str, str]:
    device = _requested_device()
    backend = os.getenv("RTMW3D_BACKEND", "tensorrt").strip().lower()
    det_frequency = int(os.getenv("RTMW3D_DET_FREQUENCY", "10"))
    detector = os.getenv("RTMW3D_DETECTOR", "yolox_nano").strip().lower()
    trt_engine = os.getenv(
        "RTMW3D_TRT_ENGINE",
        "/home/dev/.cache/rtmlib/hub/checkpoints/rtmw3d-l-fp32.plan",
    )
    redetect_confidence = float(
        os.getenv("RTMW3D_REDETECT_CONFIDENCE", "0.35")
    )
    boundary_fraction = float(os.getenv("RTMW3D_BOUNDARY_FRACTION", "0.08"))
    print(
        f"Loading RTMW3D on {device} with {backend}, detector={detector}, "
        f"det_frequency={det_frequency}; "
        "the first run may download model files..."
    )
    if detector == "yolox_nano":
        detector_model = YOLOX_NANO_URL
        detector_input_size = (416, 416)
        solution = partial(
            Custom,
            det_class="YOLOX",
            det=detector_model,
            det_input_size=detector_input_size,
            pose_class="RTMPose3d",
            pose=Wholebody3d.MODE["balanced"]["pose"],
            pose_input_size=(288, 384),
        )
    elif detector in {"yolox_m", "default"}:
        detector_model = Wholebody3d.MODE["balanced"]["det"]
        detector_input_size = Wholebody3d.MODE["balanced"]["det_input_size"]
        solution = Wholebody3d
    else:
        raise ValueError(
            f"Unsupported RTMW3D_DETECTOR={detector!r}; use yolox_nano or yolox_m"
        )

    if backend == "tensorrt":
        if device != "cuda":
            raise ValueError("RTMW3D_BACKEND=tensorrt requires RTMW3D_DEVICE=cuda")
        if not Path(trt_engine).expanduser().is_file():
            raise FileNotFoundError(
                f"RTMW3D TensorRT engine does not exist: {trt_engine}"
            )
        solution = partial(
            TensorRTWholebody3d,
            det=detector_model,
            det_input_size=detector_input_size,
            pose=trt_engine,
            pose_input_size=(288, 384),
        )
    tracker = PersistentRoiPoseTracker(
        solution,
        det_frequency=det_frequency,
        tracking=False,
        mode="balanced",
        to_openpose=False,
        backend=backend,
        device=device,
        redetect_confidence=redetect_confidence,
        boundary_fraction=boundary_fraction,
    )
    return tracker, device, backend


class InferenceService:
    """Serialize inference so one model instance can serve a browser client."""

    def __init__(self) -> None:
        self.tracker, self.device, self.backend = _make_tracker()
        self.model_label = (
            "RTMW3D-L + TensorRT 8.6 FP32"
            if self.backend == "tensorrt"
            else "RTMW3D-X + ONNX Runtime"
        )
        self.lock = asyncio.Lock()
        self.frame_index = 0
        self.fps = 0.0
        self.last_tick = time.perf_counter()

    def process(self, payload: bytes) -> tuple[bytes, dict[str, Any]]:
        started = time.perf_counter()
        encoded = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("The server could not decode the browser frame.")

        keypoints_3d, scores, keypoints_2d = as_arrays(self.tracker(frame))
        people = keypoints_2d.shape[0] if keypoints_2d.ndim >= 1 else 0
        rendered = frame.copy()
        if people:
            rendered = draw_skeleton(
                rendered,
                keypoints_2d,
                scores,
                openpose_skeleton=False,
                kpt_thr=0.5,
            )
        draw_3d_inset(rendered, keypoints_3d, scores, 0.5)

        now = time.perf_counter()
        instant_fps = 1.0 / max(now - self.last_tick, 1e-6)
        self.last_tick = now
        self.fps = instant_fps if self.fps == 0.0 else 0.9 * self.fps + 0.1 * instant_fps
        self.frame_index += 1
        processing_ms = (time.perf_counter() - started) * 1000.0

        cv2.rectangle(rendered, (0, 0), (min(rendered.shape[1], 640), 44), (18, 22, 28), -1)
        cv2.putText(
            rendered,
            f"{self.model_label} | {self.device} | people: {people} | FPS: {self.fps:4.1f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            (235, 240, 248),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            rendered,
            f"inference: {processing_ms:.0f} ms",
            (12, rendered.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (225, 230, 238),
            1,
            cv2.LINE_AA,
        )

        ok, output = cv2.imencode(
            ".jpg",
            rendered,
            [int(cv2.IMWRITE_JPEG_QUALITY), 82],
        )
        if not ok:
            raise RuntimeError("The server could not encode the annotated frame.")

        stats = {
            "frame": self.frame_index,
            "people": int(people),
            "fps": round(self.fps, 2),
            "processing_ms": round(processing_ms, 1),
            "device": self.device,
            "backend": self.backend,
            "model": self.model_label,
        }
        return output.tobytes(), stats

    async def process_async(self, payload: bytes) -> tuple[bytes, dict[str, Any]]:
        async with self.lock:
            return await asyncio.to_thread(self.process, payload)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inference = InferenceService()
    yield
    app.state.inference = None


app = FastAPI(title="RTMW3D WSL live webcam", lifespan=lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_INDEX)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    service: InferenceService = websocket.app.state.inference
    try:
        while True:
            payload = await websocket.receive_bytes()
            try:
                output, stats = await service.process_async(payload)
            except Exception as exc:  # Keep the browser session alive for recoverable bad frames.
                await websocket.send_text(json.dumps({"error": str(exc)}))
                continue
            await websocket.send_text(json.dumps(stats))
            await websocket.send_bytes(output)
    except WebSocketDisconnect:
        print("Browser disconnected")
