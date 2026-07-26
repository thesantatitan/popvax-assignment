"""Profile the RTMW3D live pipeline without changing its inference behavior.

The benchmark intentionally uses the same rtmlib PoseTracker configuration as
the live server. It also exposes a pose-only path so detector and pose costs
can be separated before making runtime changes.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

import cv2
import numpy as np
from rtmlib import Custom, PoseTracker, Wholebody3d, draw_skeleton

from demo import as_arrays, draw_3d_inset
from roi_tracker import PersistentRoiPoseTracker


STAGES = (
    "acquisition",
    "det_preprocess",
    "det_inference",
    "det_postprocess",
    "pose_preprocess",
    "pose_inference",
    "pose_postprocess",
    "visualization",
    "total",
)


class CudaSynchronizer:
    """Synchronize the CUDA device without requiring PyTorch in this venv."""

    def __init__(self) -> None:
        self._sync: Callable[[], int] | None = None
        self.error: str | None = None
        for library in ("libcudart.so.12", "libcudart.so"):
            try:
                runtime = ctypes.CDLL(library)
                function = runtime.cudaDeviceSynchronize
                function.argtypes = []
                function.restype = ctypes.c_int
                self._sync = function
                break
            except (OSError, AttributeError) as exc:
                self.error = str(exc)

    @property
    def available(self) -> bool:
        return self._sync is not None

    def __call__(self) -> None:
        if self._sync is None:
            return
        code = self._sync()
        if code:
            raise RuntimeError(f"cudaDeviceSynchronize returned error {code}")


@dataclass
class FrameTiming:
    values: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def add(self, stage: str, elapsed_ms: float) -> None:
        self.values[stage] += elapsed_ms


class StageProfiler:
    def __init__(self, synchronizer: CudaSynchronizer) -> None:
        self.synchronizer = synchronizer
        self.current = FrameTiming()
        self.frames: list[dict[str, float]] = []

    def begin_frame(self) -> None:
        self.current = FrameTiming()

    def add(self, stage: str, elapsed_ms: float) -> None:
        self.current.add(stage, elapsed_ms)

    def finish_frame(self, total_ms: float) -> None:
        values = {stage: self.current.values.get(stage, 0.0) for stage in STAGES}
        values["total"] = total_ms
        self.frames.append(values)


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
        query = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(
                    query,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=1.0,
                )
                parts = [float(value.strip()) for value in result.stdout.split(",")]
                if len(parts) == 3:
                    self.samples.append(
                        {
                            "utilization_gpu_pct": parts[0],
                            "memory_used_mib": parts[1],
                            "memory_total_mib": parts[2],
                        }
                    )
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self.stop_event.wait(self.interval_s)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def summary(self) -> dict[str, float | int | None]:
        if not self.samples:
            return {"samples": 0, "utilization_mean_pct": None, "utilization_p95_pct": None, "memory_peak_mib": None}
        utilization = np.asarray([sample["utilization_gpu_pct"] for sample in self.samples])
        memory = np.asarray([sample["memory_used_mib"] for sample in self.samples])
        return {
            "samples": len(self.samples),
            "utilization_mean_pct": round(float(utilization.mean()), 1),
            "utilization_p95_pct": round(float(np.percentile(utilization, 95)), 1),
            "utilization_peak_pct": round(float(utilization.max()), 1),
            "memory_peak_mib": round(float(memory.max()), 1),
            "memory_mean_mib": round(float(memory.mean()), 1),
            "memory_total_mib": round(float(self.samples[-1]["memory_total_mib"]), 1),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--backend", default="onnxruntime", choices=("onnxruntime", "opencv", "openvino"))
    parser.add_argument("--det-frequency", type=int, default=7)
    parser.add_argument(
        "--det-model",
        help="Optional detector ONNX/ZIP URL or local path for a detector-only experiment.",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        nargs=2,
        default=(640, 640),
        metavar=("WIDTH", "HEIGHT"),
        help="Detector input size used with --det-model.",
    )
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--image",
        help="Repeat a real image instead of the deterministic gradient benchmark input.",
    )
    parser.add_argument("--no-visualization", action="store_true")
    parser.add_argument("--pose-only", action="store_true")
    parser.add_argument(
        "--persistent-roi",
        action="store_true",
        help="Use the live server's confidence/boundary-aware ROI policy.",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=(100.0, 20.0, 540.0, 470.0),
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Fixed xyxy crop used by --pose-only.",
    )
    args = parser.parse_args()
    if args.det_frequency < 1:
        parser.error("--det-frequency must be at least 1")
    if args.warmup < 1 or args.frames < 1:
        parser.error("--warmup and --frames must be positive")
    return args


def make_frame(width: int, height: int, image_path: str | None = None) -> np.ndarray:
    """Make a stable input with real image memory traffic, not a zero sentinel."""

    if image_path:
        frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read benchmark image: {image_path}")
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return frame

    frame = np.empty((height, width, 3), dtype=np.uint8)
    rows = np.arange(height, dtype=np.uint16)[:, None]
    cols = np.arange(width, dtype=np.uint16)[None, :]
    frame[..., 0] = (cols + rows) % 256
    frame[..., 1] = (2 * cols + rows) % 256
    frame[..., 2] = (cols + 2 * rows) % 256
    return frame


def timed_wrapper(
    owner: Any,
    name: str,
    stage: str,
    profiler: StageProfiler,
    synchronize: bool = False,
) -> None:
    original = getattr(owner, name)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if synchronize:
            profiler.synchronizer()
        started = time.perf_counter()
        result = original(*args, **kwargs)
        if synchronize:
            profiler.synchronizer()
        profiler.add(stage, (time.perf_counter() - started) * 1000.0)
        return result

    setattr(owner, name, wrapped)


def instrument_tracker(tracker: PoseTracker, profiler: StageProfiler) -> None:
    if tracker.det_model is not None:
        timed_wrapper(tracker.det_model, "preprocess", "det_preprocess", profiler)
        timed_wrapper(tracker.det_model, "inference", "det_inference", profiler, synchronize=True)
        timed_wrapper(tracker.det_model, "postprocess", "det_postprocess", profiler)
    timed_wrapper(tracker.pose_model, "preprocess", "pose_preprocess", profiler)
    timed_wrapper(tracker.pose_model, "inference", "pose_inference", profiler, synchronize=True)
    timed_wrapper(tracker.pose_model, "postprocess", "pose_postprocess", profiler)


def render(
    frame: np.ndarray,
    result: tuple[object, ...],
    threshold: float = 0.5,
) -> None:
    keypoints_3d, scores, keypoints_2d = as_arrays(result)
    people = keypoints_2d.shape[0] if keypoints_2d.ndim >= 1 else 0
    if people:
        draw_skeleton(
            frame,
            keypoints_2d,
            scores,
            openpose_skeleton=False,
            kpt_thr=threshold,
        )
    draw_3d_inset(frame, keypoints_3d, scores, threshold)


def summarize(values: np.ndarray) -> dict[str, float]:
    if float(values.mean()) <= 0:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "fps": 0.0}
    return {
        "mean_ms": round(float(values.mean()), 2),
        "median_ms": round(float(np.median(values)), 2),
        "p95_ms": round(float(np.percentile(values, 95)), 2),
        "fps": round(float(1000.0 / values.mean()), 2),
    }


def stage_summary(frames: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for stage in STAGES:
        result[stage] = summarize(np.asarray([frame[stage] for frame in frames]))
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    synchronizer = CudaSynchronizer()
    profiler = StageProfiler(synchronizer)
    frame_template = make_frame(args.width, args.height, args.image)

    if args.pose_only:
        print("Loading pose-only RTMW3D model")
        solution = Wholebody3d(
            mode="balanced",
            to_openpose=False,
            backend=args.backend,
            device=args.device,
        )
        pose_model = solution.pose_model
        timed_wrapper(pose_model, "preprocess", "pose_preprocess", profiler)
        timed_wrapper(pose_model, "inference", "pose_inference", profiler, synchronize=True)
        timed_wrapper(pose_model, "postprocess", "pose_postprocess", profiler)
        tracker = None
    else:
        print(
            f"Loading full RTMW3D pipeline: det_frequency={args.det_frequency}, "
            f"visualization={not args.no_visualization}"
        )
        if args.det_model:
            solution = partial(
                Custom,
                det_class="YOLOX",
                det=args.det_model,
                det_input_size=tuple(args.det_size),
                pose_class="RTMPose3d",
                pose=Wholebody3d.MODE["balanced"]["pose"],
                pose_input_size=(288, 384),
            )
            print(
                f"Using custom YOLOX detector {args.det_model} "
                f"at {tuple(args.det_size)}"
            )
        else:
            solution = Wholebody3d
        tracker_class = PersistentRoiPoseTracker if args.persistent_roi else PoseTracker
        tracker_kwargs: dict[str, Any] = {}
        if args.persistent_roi:
            tracker_kwargs.update(
                redetect_confidence=0.35,
                boundary_fraction=0.08,
            )
        tracker = tracker_class(
            solution,
            det_frequency=args.det_frequency,
            tracking=False,
            mode="balanced",
            to_openpose=False,
            backend=args.backend,
            device=args.device,
            **tracker_kwargs,
        )
        instrument_tracker(tracker, profiler)
        pose_model = None

    if not synchronizer.available and args.device == "cuda":
        print(f"WARNING: CUDA runtime synchronization unavailable: {synchronizer.error}")
    print(f"cuda_synchronize={'available' if synchronizer.available else 'unavailable'}")

    def process_frame() -> None:
        profiler.begin_frame()
        started = time.perf_counter()
        acquisition_started = time.perf_counter()
        frame = frame_template.copy()
        profiler.add("acquisition", (time.perf_counter() - acquisition_started) * 1000.0)

        model_started = time.perf_counter()
        if args.pose_only:
            result = pose_model(frame, bboxes=[args.bbox])
        else:
            result = tracker(frame)
        model_elapsed = (time.perf_counter() - model_started) * 1000.0

        if not args.no_visualization:
            visualization_started = time.perf_counter()
            render(frame, result)
            profiler.add("visualization", (time.perf_counter() - visualization_started) * 1000.0)
        profiler.finish_frame((time.perf_counter() - started) * 1000.0)
        # Keep this value visible while debugging a stage wrapper mismatch.
        if model_elapsed < 0:
            raise AssertionError(model_elapsed)

    print(f"warmup_frames={args.warmup}")
    for _ in range(args.warmup):
        process_frame()
    profiler.frames.clear()

    monitor = GpuMonitor()
    monitor.start()
    print(f"measured_frames={args.frames}")
    for _ in range(args.frames):
        process_frame()
    monitor.stop()

    stages = stage_summary(profiler.frames)
    result: dict[str, Any] = {
        "configuration": {
            "device": args.device,
            "backend": args.backend,
            "det_frequency": None if args.pose_only else args.det_frequency,
            "det_model": args.det_model,
            "det_size": list(args.det_size) if not args.pose_only else None,
            "pose_only": args.pose_only,
            "persistent_roi": args.persistent_roi,
            "visualization": not args.no_visualization,
            "input": [args.width, args.height],
            "image": args.image,
            "warmup_frames": args.warmup,
            "measured_frames": args.frames,
            "bbox": list(args.bbox) if args.pose_only else None,
        },
        "stages": stages,
        "gpu": monitor.summary(),
        "cuda_synchronize": synchronizer.available,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run(parse_args())
