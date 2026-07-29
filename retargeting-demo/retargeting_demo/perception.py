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

from .confidence import ContinuousConfidenceGate
from .contracts import RETARGET_MODES, BrowserFrame, RenderedFrame, RetargetMode
from .ipc import configure_parent_death_signal, drain_latest, put_latest
from .retarget import SimccRetargeter, target_record

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
RTMW3D_DEMO = REPOSITORY_ROOT / "rtmw3d-livewebcam"
sys.path.insert(0, str(RTMW3D_DEMO))

from demo import draw_3d_inset
from roi_tracker import PersistentRoiPoseTracker
from tensorrt_backend import RTMPose3d as TensorRTPose3d
from runtime_paths import tensorrt_engine_path

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
        engine = tensorrt_engine_path()
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
    keypoints_3d: np.ndarray,
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
    draw_3d_inset(rendered, keypoints_3d, scores, 0.35)
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
    mode_queue,
    camera_config_queue,
    target_queue,
    pose_frame_queue,
    telemetry_queue,
    tracking_reset_event,
    engaged_event,
    stop_event,
    log_directory: str,
) -> None:
    """Process entrypoint. CUDA and TensorRT are initialized only in this process."""

    parent_pid = configure_parent_death_signal()
    tracker, device, backend = _make_tracker()
    retargeter = SimccRetargeter(
        confidence_threshold=float(os.getenv("RETARGET_CONFIDENCE", "0.35")),
        smoothing_time_constant_s=float(
            os.getenv("RETARGET_SMOOTHING_TAU_S", "0.5")
        ),
    )
    gate = ContinuousConfidenceGate(
        required_seconds=float(os.getenv("RETARGET_CONFIDENCE_SECONDS", "2.0"))
    )
    gate_state = gate.reset()
    log_path = Path(log_directory)
    log_path.mkdir(parents=True, exist_ok=True)
    target_log = log_path / f"targets-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    last_tick = time.perf_counter()
    last_input_wall: float | None = None
    timeout_reported = False
    smoothed_fps = 0.0
    mode: RetargetMode = "both"
    camera_config: dict[str, object] = {"enabled": False, "source": "disabled"}
    with target_log.open("a", encoding="utf-8", buffering=1) as output:
        while not stop_event.is_set() and os.getppid() == parent_pid:
            requested_mode = drain_latest(mode_queue)
            if requested_mode in RETARGET_MODES and requested_mode != mode:
                mode = requested_mode
                retargeter.reset_smoothing()
                engaged_event.clear()
                gate_state = gate.reset()
            requested_camera_config = drain_latest(camera_config_queue)
            if isinstance(requested_camera_config, dict):
                camera_config = requested_camera_config
                retargeter.reset_smoothing()
                engaged_event.clear()
                gate_state = gate.reset()
            if tracking_reset_event.is_set():
                tracking_reset_event.clear()
                retargeter.reset_smoothing()
                engaged_event.clear()
                gate_state = gate.reset()
            incoming = drain_latest(frame_queue)
            if incoming is None:
                if (
                    last_input_wall is not None
                    and time.perf_counter() - last_input_wall > 0.5
                    and not timeout_reported
                ):
                    gate_state = gate.update(False, time.monotonic_ns())
                    engaged_event.clear()
                    timeout_reported = True
                    put_latest(
                        telemetry_queue,
                        {
                            "type": "perception",
                            "people": 0,
                            "fps": round(smoothed_fps, 2),
                            "processing_ms": None,
                            "device": device,
                            "backend": backend,
                            "mode": mode,
                            "camera_intrinsics_enabled": bool(
                                camera_config.get("enabled")
                            ),
                            "camera_intrinsics_source": camera_config.get("source"),
                            "engaged": False,
                            "tracking_state": gate_state.state,
                            "confidence_seconds": 0.0,
                            "confidence_required_seconds": gate_state.required_seconds,
                            "minimum_confidence": 0.0,
                            "mean_confidence": 0.0,
                            "error": "Camera frames timed out; holding the last pose.",
                        },
                    )
                time.sleep(0.002)
                continue
            assert isinstance(incoming, BrowserFrame)
            last_input_wall = time.perf_counter()
            timeout_reported = False
            started = time.perf_counter()
            encoded = np.frombuffer(incoming.jpeg, dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            keypoints_3d = np.empty((0, 133, 3), dtype=np.float32)
            keypoints_2d = np.empty((0, 133, 2), dtype=np.float32)
            scores = np.empty((0, 133), dtype=np.float32)
            people = 0
            minimum_confidence = 0.0
            mean_confidence = 0.0
            try:
                result = tracker(frame)
                if not isinstance(result, tuple) or len(result) != 4:
                    raise ValueError("RTMW3D did not return a valid pose")
                keypoints_3d_raw, scores_raw, simcc_raw, keypoints_2d_raw = result
                keypoints_3d = np.asarray(keypoints_3d_raw)
                scores = np.asarray(scores_raw)
                simcc = np.asarray(simcc_raw)
                keypoints_2d = np.asarray(keypoints_2d_raw)
                people = len(scores) if scores.ndim == 2 else 0
                minimum_confidence, mean_confidence = (
                    retargeter.confidence_summary(scores, mode)
                )
                target = retargeter.make_target(
                    sequence=incoming.sequence,
                    capture_time_ns=incoming.capture_time_ns,
                    inference_time_ns=time.monotonic_ns(),
                    simcc=simcc,
                    scores=scores,
                    mode=mode,
                    keypoints_2d=keypoints_2d,
                    camera_intrinsics=(
                        camera_config
                        if bool(camera_config.get("enabled"))
                        else None
                    ),
                )
                gate_state = gate.update(True, time.monotonic_ns())
                if gate_state.ready:
                    engaged_event.set()
                    put_latest(target_queue, target)
                    person_index = retargeter.select_person(scores)
                    output.write(
                        json.dumps(
                            target_record(
                                target,
                                keypoints_simcc=simcc[person_index],
                                person_index=person_index,
                            ),
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                else:
                    engaged_event.clear()
                error = None
            # Model/runtime and low-confidence failures are reported while
            # this long-lived worker remains available for the next frame.
            except Exception as exc:  # noqa: BLE001
                gate_state = gate.update(False, time.monotonic_ns())
                engaged_event.clear()
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
            if gate_state.state == "tracking":
                tracking_label = "TRACKING"
            elif gate_state.state in {"acquiring", "reacquiring"}:
                tracking_label = (
                    f"CONFIDENCE {gate_state.continuous_seconds:.1f}/"
                    f"{gate_state.required_seconds:.1f}s"
                )
            elif gate_state.state == "holding":
                tracking_label = "LOW CONFIDENCE - HOLDING"
            else:
                tracking_label = "WAITING FOR POSE"
            status = (
                f"RTMW3D {backend} | {mode.upper()} | {tracking_label} | "
                f"{smoothed_fps:.1f} FPS | {processing_ms:.0f} ms"
            )
            annotated = _annotate(
                frame, keypoints_3d, keypoints_2d, scores, status
            )
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
                    "mode": mode,
                    "camera_intrinsics_enabled": bool(
                        camera_config.get("enabled")
                    ),
                    "camera_intrinsics_source": camera_config.get("source"),
                    "camera_profile_id": camera_config.get("profile_id"),
                    "engaged": gate_state.ready,
                    "tracking_state": gate_state.state,
                    "confidence_seconds": round(
                        gate_state.continuous_seconds, 2
                    ),
                    "confidence_required_seconds": gate_state.required_seconds,
                    "minimum_confidence": round(minimum_confidence, 3),
                    "mean_confidence": round(mean_confidence, 3),
                    "smoothed_end_effectors": (
                        {
                            side: list(
                                getattr(target, side).wrist_position_m
                            )
                            for side in ("left", "right")
                            if getattr(target, side).wrist_position_m is not None
                        }
                        if error is None
                        else None
                    ),
                    "error": error,
                },
            )
