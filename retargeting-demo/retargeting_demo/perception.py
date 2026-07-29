"""RTMW3D inference process and robot-target production."""

from __future__ import annotations

import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from rtmlib import YOLOX, Custom, Wholebody3d, draw_skeleton

from .contracts import BrowserFrame, RenderedFrame
from .ipc import drain_latest, put_latest
from .retarget import SimccRetargeter, target_record

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
RTMW3D_DEMO = REPOSITORY_ROOT / "rtmw3d-livewebcam"
sys.path.insert(0, str(RTMW3D_DEMO))

from roi_tracker import PersistentRoiPoseTracker
from tensorrt_backend import RTMPose3d as TensorRTPose3d

YOLOX_NANO_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "onnx_sdk/yolox_nano_8xb8-300e_humanart-40f6f0d0.zip"
)


class SplitProviderTensorRTWholebody3d:
    """Use TensorRT for pose while selecting the detector provider separately."""

    def __init__(
        self,
        det: str,
        det_input_size: tuple[int, int],
        pose: str,
        pose_input_size: tuple[int, int] = (288, 384),
        mode: str = "balanced",
        to_openpose: bool = False,
        backend: str = "tensorrt",
        device: str = "cuda",
    ) -> None:
        del mode, backend, device
        detector_device = (
            "cuda"
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else "cpu"
        )
        self.det_model = YOLOX(
            det,
            model_input_size=det_input_size,
            backend=os.getenv("RTMW3D_DETECTOR_BACKEND", "onnxruntime"),
            device=detector_device,
        )
        self.pose_model = TensorRTPose3d(
            pose,
            model_input_size=pose_input_size,
            to_openpose=to_openpose,
            backend="tensorrt",
            device="cuda",
        )


def _make_tracker() -> tuple[PersistentRoiPoseTracker, str, str]:
    requested = os.getenv("RTMW3D_DEVICE", "cuda").strip().lower()
    backend = os.getenv("RTMW3D_BACKEND", "tensorrt").strip().lower()
    device = requested
    if (
        backend != "tensorrt"
        and requested == "cuda"
        and "CUDAExecutionProvider" not in ort.get_available_providers()
    ):
        print("CUDAExecutionProvider unavailable; falling back to CPU", flush=True)
        device = "cpu"
    if backend == "tensorrt" and device != "cuda":
        raise ValueError("RTMW3D_BACKEND=tensorrt requires RTMW3D_DEVICE=cuda")
    detector_model = YOLOX_NANO_URL
    solution: object = partial(
        Custom,
        det_class="YOLOX",
        det=detector_model,
        det_input_size=(416, 416),
        pose_class="RTMPose3d",
        pose=Wholebody3d.MODE["balanced"]["pose"],
        pose_input_size=(288, 384),
    )
    if backend == "tensorrt":
        engine = Path(
            os.getenv(
                "RTMW3D_TRT_ENGINE",
                "/home/dev/.cache/rtmlib/hub/checkpoints/rtmw3d-l-fp32.plan",
            )
        ).expanduser()
        solution = partial(
            SplitProviderTensorRTWholebody3d,
            det=detector_model,
            det_input_size=(416, 416),
            pose=str(engine),
            pose_input_size=(288, 384),
        )
    tracker = PersistentRoiPoseTracker(
        solution,
        det_frequency=int(os.getenv("RTMW3D_DET_FREQUENCY", "10")),
        tracking=False,
        mode="balanced",
        to_openpose=False,
        backend=backend,
        device=device,
        redetect_confidence=float(os.getenv("RTMW3D_REDETECT_CONFIDENCE", "0.35")),
        boundary_fraction=float(os.getenv("RTMW3D_BOUNDARY_FRACTION", "0.08")),
    )
    return tracker, device, backend


def _annotate(
    frame: np.ndarray,
    keypoints_2d: np.ndarray,
    scores: np.ndarray,
    status: str,
) -> bytes:
    rendered = frame.copy()
    if len(keypoints_2d):
        rendered = draw_skeleton(
            rendered,
            keypoints_2d,
            scores,
            openpose_skeleton=False,
            kpt_thr=0.35,
        )
    cv2.rectangle(rendered, (0, 0), (rendered.shape[1], 42), (14, 20, 28), -1)
    cv2.putText(
        rendered,
        status,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (235, 242, 248),
        1,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(
        ".jpg", rendered, [int(cv2.IMWRITE_JPEG_QUALITY), 82]
    )
    if not ok:
        raise RuntimeError("Could not encode annotated frame")
    return encoded.tobytes()


def perception_worker(
    frame_queue,
    target_queue,
    pose_frame_queue,
    telemetry_queue,
    calibrate_event,
    engaged_event,
    stop_event,
    log_directory: str,
) -> None:
    """Process entrypoint. CUDA and TensorRT are initialized only in this process."""

    parent_pid = os.getppid()
    tracker, device, backend = _make_tracker()
    retargeter = SimccRetargeter(
        confidence_threshold=float(os.getenv("RETARGET_CONFIDENCE", "0.35"))
    )
    log_path = Path(log_directory)
    log_path.mkdir(parents=True, exist_ok=True)
    target_log = log_path / f"targets-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    last_tick = time.perf_counter()
    smoothed_fps = 0.0
    with target_log.open("a", encoding="utf-8", buffering=1) as output:
        while not stop_event.is_set() and os.getppid() == parent_pid:
            incoming = drain_latest(frame_queue)
            if incoming is None:
                time.sleep(0.002)
                continue
            assert isinstance(incoming, BrowserFrame)
            started = time.perf_counter()
            encoded = np.frombuffer(incoming.jpeg, dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            try:
                result = tracker(frame)
                if not isinstance(result, tuple) or len(result) != 4:
                    raise ValueError("RTMW3D did not return a valid pose")
                _keypoints_3d, scores_raw, simcc_raw, keypoints_2d_raw = result
                scores = np.asarray(scores_raw)
                simcc = np.asarray(simcc_raw)
                keypoints_2d = np.asarray(keypoints_2d_raw)
                people = len(scores) if scores.ndim == 2 else 0
                calibration_requested = calibrate_event.is_set()
                target = retargeter.make_target(
                    sequence=incoming.sequence,
                    capture_time_ns=incoming.capture_time_ns,
                    inference_time_ns=time.monotonic_ns(),
                    simcc=simcc,
                    scores=scores,
                    calibrate=calibration_requested,
                )
                if calibration_requested:
                    calibrate_event.clear()
                if engaged_event.is_set() and retargeter.calibrated:
                    put_latest(target_queue, target)
                    output.write(json.dumps(target_record(target), separators=(",", ":")) + "\n")
                error = None
            # Model/runtime failures are reported to the browser while this
            # long-lived worker stays available for the next camera frame.
            except Exception as exc:  # noqa: BLE001
                keypoints_2d = np.empty((0, 133, 2), dtype=np.float32)
                scores = np.empty((0, 133), dtype=np.float32)
                people = 0
                error = str(exc)

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_tick, 1e-6)
            last_tick = now
            smoothed_fps = (
                instant_fps
                if smoothed_fps == 0.0
                else 0.9 * smoothed_fps + 0.1 * instant_fps
            )
            processing_ms = (now - started) * 1000.0
            mode = "ENGAGED" if engaged_event.is_set() and retargeter.calibrated else "READY"
            if calibrate_event.is_set():
                mode = "HOLD NEUTRAL FOR CALIBRATION"
            status = (
                f"RTMW3D {backend} | {mode} | {smoothed_fps:.1f} FPS | "
                f"{processing_ms:.0f} ms"
            )
            annotated = _annotate(frame, keypoints_2d, scores, status)
            put_latest(
                pose_frame_queue,
                RenderedFrame(sequence=incoming.sequence, jpeg=annotated),
            )
            put_latest(
                telemetry_queue,
                {
                    "type": "perception",
                    "sequence": incoming.sequence,
                    "people": people,
                    "fps": round(smoothed_fps, 2),
                    "processing_ms": round(processing_ms, 1),
                    "device": device,
                    "backend": backend,
                    "calibrated": retargeter.calibrated,
                    "engaged": engaged_event.is_set(),
                    "error": error,
                },
            )
