"""Run RTMW3D on a live webcam or a video file."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np
from rtmlib import PoseTracker, Wholebody3d, draw_skeleton


# COCO body edges. RTMW3D exposes the 17 body joints first in its whole-body
# output; the 2D renderer from rtmlib handles the full 133-keypoint skeleton.
BODY_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RTMW3D whole-body 3D pose estimation on a webcam."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (for example 0 or 1) or a video-file path (default: 0).",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default="cpu",
        help="Inference device passed to rtmlib (default: cpu).",
    )
    parser.add_argument(
        "--backend",
        choices=("onnxruntime", "opencv", "openvino"),
        default="onnxruntime",
        help="Inference backend passed to rtmlib (default: onnxruntime).",
    )
    parser.add_argument(
        "--mode",
        choices=("balanced",),
        default="balanced",
        help="RTMW3D model configuration (default: balanced).",
    )
    parser.add_argument(
        "--det-frequency",
        type=int,
        default=7,
        metavar="N",
        help="Run the person detector every N frames (default: 7).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        metavar="T",
        help="Keypoint confidence threshold for drawing (default: 0.5).",
    )
    parser.add_argument("--width", type=int, help="Requested camera width.")
    parser.add_argument("--height", type=int, help="Requested camera height.")
    parser.add_argument(
        "--no-mirror",
        dest="mirror",
        action="store_false",
        help="Do not mirror the camera image.",
    )
    parser.set_defaults(mirror=True)
    parser.add_argument(
        "--save-jsonl",
        type=Path,
        metavar="PATH",
        help="Write per-frame 3D keypoints and scores to PATH.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        metavar="N",
        help="Stop after N frames; useful with a video file.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without opening an OpenCV window.",
    )
    args = parser.parse_args()
    if args.det_frequency < 1:
        parser.error("--det-frequency must be at least 1")
    if not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be between 0 and 1")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    return args


def resolve_source(source: str) -> int | str:
    """Treat a numeric source as a camera index and everything else as a path."""

    try:
        return int(source)
    except ValueError:
        return source


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    source = resolve_source(args.source)
    capture = cv2.VideoCapture(source)
    if args.width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        capture.release()
        kind = "camera" if isinstance(source, int) else "video file"
        raise RuntimeError(
            f"Could not open {kind} {args.source!r}. "
            "Check the camera index/path and camera permissions."
        )
    return capture


def make_tracker(args: argparse.Namespace) -> PoseTracker:
    print(
        f"Loading RTMW3D ({args.mode}) on {args.device} with {args.backend}; "
        "the first run may download model files..."
    )
    return PoseTracker(
        Wholebody3d,
        det_frequency=args.det_frequency,
        tracking=False,
        mode=args.mode,
        to_openpose=False,
        backend=args.backend,
        device=args.device,
    )


def as_arrays(
    result: tuple[object, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize the rtmlib result and keep only values used by this demo."""

    if len(result) != 4:
        raise RuntimeError(
            "RTMW3D returned an unexpected result. "
            "Try lowering --det-frequency to 1."
        )
    keypoints_3d, scores, _keypoints_simcc, keypoints_2d = result
    return (
        np.asarray(keypoints_3d),
        np.asarray(scores),
        np.asarray(keypoints_2d),
    )


def draw_3d_inset(
    image: np.ndarray,
    keypoints_3d: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> None:
    """Draw a small perspective projection of the first detected 3D body."""

    if keypoints_3d.ndim != 3 or keypoints_3d.shape[1] < 17:
        return
    if scores.ndim != 2 or scores.shape[1] < 17:
        return

    body_scores = scores[:, :17]
    person_index = int(np.argmax(np.mean(body_scores, axis=1)))
    points = keypoints_3d[person_index, :17, :3].astype(np.float32, copy=True)
    visible = body_scores[person_index] >= threshold
    finite = np.isfinite(points).all(axis=1)
    visible &= finite
    if visible.sum() < 2:
        return

    center = np.median(points[visible], axis=0)
    points -= center
    span = np.ptp(points[visible], axis=0)
    scale = max(float(np.max(span)), 1e-6)
    points /= scale

    height, width = image.shape[:2]
    inset = max(180, min(300, height // 3, width // 3))
    x0 = width - inset - 16
    y0 = 52
    if x0 < 0 or y0 + inset > height:
        return

    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + inset, y0 + inset), (18, 22, 28), -1)
    cv2.addWeighted(overlay, 0.86, image, 0.14, 0, image)
    cv2.rectangle(image, (x0, y0), (x0 + inset, y0 + inset), (85, 92, 105), 1)

    center_xy = np.array([x0 + inset // 2, y0 + inset // 2 + 12])
    projected = np.empty((17, 2), dtype=np.int32)
    projected[:, 0] = center_xy[0] + (points[:, 0] - 0.35 * points[:, 2]) * inset * 0.42
    projected[:, 1] = center_xy[1] - (points[:, 1] + 0.15 * points[:, 2]) * inset * 0.42

    for start, end in BODY_EDGES:
        if visible[start] and visible[end]:
            cv2.line(
                image,
                tuple(projected[start]),
                tuple(projected[end]),
                (75, 205, 255),
                2,
                cv2.LINE_AA,
            )
    for index, point in enumerate(projected):
        if visible[index]:
            cv2.circle(image, tuple(point), 3, (80, 240, 160), -1, cv2.LINE_AA)

    cv2.putText(
        image,
        "3D pose (first person)",
        (x0 + 10, y0 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (235, 240, 248),
        1,
        cv2.LINE_AA,
    )


def draw_status(
    image: np.ndarray,
    people: int,
    fps: float,
    show_skeleton: bool,
    show_3d: bool,
) -> None:
    status = f"RTMW3D | people: {people} | FPS: {fps:4.1f}"
    cv2.rectangle(image, (0, 0), (min(image.shape[1], 470), 38), (18, 22, 28), -1)
    cv2.putText(
        image,
        status,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (235, 240, 248),
        1,
        cv2.LINE_AA,
    )
    controls = f"q quit | s skeleton {'on' if show_skeleton else 'off'} | v 3D {'on' if show_3d else 'off'}"
    cv2.putText(
        image,
        controls,
        (12, image.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (225, 230, 238),
        1,
        cv2.LINE_AA,
    )


def save_frame(
    output: TextIO,
    frame_index: int,
    fps: float,
    keypoints_3d: np.ndarray,
    scores: np.ndarray,
) -> None:
    record = {
        "frame": frame_index,
        "fps": round(fps, 3),
        "people": int(keypoints_3d.shape[0]) if keypoints_3d.ndim >= 1 else 0,
        "keypoints_3d": keypoints_3d.tolist(),
        "scores": scores.tolist(),
    }
    output.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> int:
    args = parse_args()
    capture = open_capture(args)
    output: TextIO | None = None
    if args.save_jsonl:
        args.save_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output = args.save_jsonl.open("w", encoding="utf-8")

    try:
        tracker = make_tracker(args)
        frame_index = 0
        fps = 0.0
        show_skeleton = True
        show_3d = True
        last_tick = time.perf_counter()

        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_index += 1
            if args.mirror:
                frame = cv2.flip(frame, 1)

            result = tracker(frame)
            keypoints_3d, scores, keypoints_2d = as_arrays(result)
            people = keypoints_2d.shape[0] if keypoints_2d.ndim >= 1 else 0

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_tick, 1e-6)
            last_tick = now
            fps = instant_fps if fps == 0.0 else 0.9 * fps + 0.1 * instant_fps

            rendered = frame.copy()
            if show_skeleton and people:
                rendered = draw_skeleton(
                    rendered,
                    keypoints_2d,
                    scores,
                    openpose_skeleton=False,
                    kpt_thr=args.score_threshold,
                )
            if show_3d:
                draw_3d_inset(rendered, keypoints_3d, scores, args.score_threshold)
            draw_status(rendered, people, fps, show_skeleton, show_3d)

            if output is not None:
                save_frame(output, frame_index, fps, keypoints_3d, scores)

            if not args.no_display:
                cv2.imshow("RTMW3D live webcam", rendered)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    show_skeleton = not show_skeleton
                if key == ord("v"):
                    show_3d = not show_3d

            if args.max_frames is not None and frame_index >= args.max_frames:
                break
    finally:
        capture.release()
        if output is not None:
            output.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"Processed {frame_index} frame(s).")
    if args.save_jsonl:
        print(f"Saved predictions to {args.save_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
