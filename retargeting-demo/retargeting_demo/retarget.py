"""Convert decoded RTMW3D SimCC landmarks into robot-base targets."""

from __future__ import annotations

import math
from dataclasses import asdict

import cv2
import numpy as np

from .contracts import RETARGET_MODES, ArmTarget, RetargetMode, RobotTarget

BODY = {
    "left": {"shoulder": 5, "elbow": 7, "wrist": 9, "hand_start": 91},
    "right": {"shoulder": 6, "elbow": 8, "wrist": 10, "hand_start": 112},
}
HAND_FRAME_OFFSETS = (0, 5, 9, 13, 17)
SHOULDER_POSITIONS = {
    "left": np.array([0.0, 0.1535, 0.0]),
    "right": np.array([0.0, -0.1535, 0.0]),
}
UPPER_ARM_LENGTH_M = 0.220
FOREARM_LENGTH_M = 0.216
ASSUMED_SHOULDER_WIDTH_M = 0.38
FALLBACK_ROOT_DEPTH_M = 2.5
# RTMW3D: +x image-right, +y image-down, +z depth. OpenArm arm_origin:
# +x forward into the cell, +y robot-left, +z up.
CAMERA_TO_ROBOT = np.array(
    [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)


def exponential_smoothing_alpha(period_s: float, time_constant_s: float) -> float:
    """Return a rate-independent exponential smoothing coefficient."""

    if period_s <= 0.0:
        raise ValueError("period_s must be positive")
    if time_constant_s < 0.0:
        raise ValueError("time_constant_s must be non-negative")
    if time_constant_s == 0.0:
        return 1.0
    return float(-math.expm1(-period_s / time_constant_s))


def required_keypoint_indices(mode: RetargetMode) -> list[int]:
    if mode not in RETARGET_MODES:
        raise ValueError(f"Unsupported retargeting mode: {mode}")
    required: list[int] = []
    for indices in BODY.values():
        required.extend([indices["shoulder"], indices["elbow"]])
        if mode != "elbow":
            required.append(indices["wrist"])
            required.extend(
                indices["hand_start"] + offset
                for offset in HAND_FRAME_OFFSETS
            )
    return sorted(set(required))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-6:
        raise ValueError("Degenerate limb or hand direction")
    return vector / norm


def hand_frame(
    points: np.ndarray,
    hand_start: int,
    side: str,
) -> np.ndarray:
    """Return an absolute right-handed hand frame in robot-base coordinates."""

    hand = points[hand_start : hand_start + 21]
    wrist = hand[0]
    finger_direction = _unit(
        np.mean(hand[[5, 9, 13, 17]], axis=0) - wrist
    )
    # Use the mirrored anatomical ordering so both hands map to the same tool
    # convention: +x along fingers, +y across the palm toward robot-left, and
    # +z along the palm normal.
    across_hint = (
        hand[17] - hand[5] if side == "left" else hand[5] - hand[17]
    )
    palm_normal = _unit(np.cross(finger_direction, across_hint))
    palm_across = _unit(np.cross(palm_normal, finger_direction))
    return np.column_stack((finger_direction, palm_across, palm_normal))


def smooth_rotation(
    previous: np.ndarray,
    current: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Interpolate on SO(3), preserving a valid rotation matrix."""

    relative = previous.T @ current
    rotation_vector, _ = cv2.Rodrigues(relative)
    incremental, _ = cv2.Rodrigues(alpha * rotation_vector)
    return previous @ incremental


def simcc_to_camera_points(simcc: np.ndarray) -> np.ndarray:
    """Convert decoded SimCC coordinates to an aspect-correct 3D proxy.

    rtmlib has already divided the raw SimCC maxima by its split ratio. The
    RTMW3D input is 288x384 and its depth bins use half the input height.
    Only relative directions are consumed downstream, so the crop translation
    cancels and robot segment lengths supply the metric scale.
    """

    points = np.asarray(simcc, dtype=np.float64).copy()
    if points.shape[-1] != 3:
        raise ValueError(f"Expected (..., 3) SimCC coordinates, got {points.shape}")
    half_height = 384.0 / 2.0
    points[..., 0] /= half_height
    points[..., 1] /= half_height
    points[..., 2] = (points[..., 2] / half_height - 1.0) * 2.1744869
    return points


def _estimate_root_depth(
    normalized_rays: np.ndarray,
    relative_depth: np.ndarray,
) -> float:
    """Solve root depth from the two shoulder rays and assumed shoulder width."""

    left = BODY["left"]["shoulder"]
    right = BODY["right"]["shoulder"]
    ray_left = np.array(
        [normalized_rays[left, 0], normalized_rays[left, 1], 1.0]
    )
    ray_right = np.array(
        [normalized_rays[right, 0], normalized_rays[right, 1], 1.0]
    )
    a = ray_left - ray_right
    b = ray_left * relative_depth[left] - ray_right * relative_depth[right]
    coefficients = (
        float(a @ a),
        float(2.0 * (a @ b)),
        float(b @ b - ASSUMED_SHOULDER_WIDTH_M**2),
    )
    roots = np.roots(coefficients)
    candidates = [
        float(root.real)
        for root in roots
        if abs(float(root.imag)) < 1e-7
        and 0.5 <= float(root.real) <= 6.0
        and float(root.real)
        + min(float(relative_depth[left]), float(relative_depth[right]))
        > 0.1
    ]
    if not candidates:
        return FALLBACK_ROOT_DEPTH_M
    return min(candidates, key=lambda value: abs(value - FALLBACK_ROOT_DEPTH_M))


def reconstruct_with_intrinsics(
    simcc: np.ndarray,
    keypoints_2d: np.ndarray,
    intrinsics: dict[str, object],
) -> tuple[np.ndarray, float]:
    """Backproject original-image points using decoded root-relative depth."""

    points_2d = np.asarray(keypoints_2d, dtype=np.float64)
    if points_2d.ndim != 2 or points_2d.shape[-1] != 2:
        raise ValueError(f"Expected (keypoints, 2), got {points_2d.shape}")
    matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(
        intrinsics["distortion_coefficients"], dtype=np.float64
    )
    normalized = cv2.undistortPoints(
        points_2d.reshape(-1, 1, 2), matrix, distortion
    ).reshape(-1, 2)
    relative_depth = (
        np.asarray(simcc, dtype=np.float64)[:, 2] / (384.0 / 2.0) - 1.0
    ) * 2.1744869
    root_depth = _estimate_root_depth(normalized, relative_depth)
    absolute_depth = root_depth + relative_depth
    arm_indices = [
        BODY[side][joint]
        for side in ("left", "right")
        for joint in ("shoulder", "elbow", "wrist")
    ]
    if np.min(absolute_depth[arm_indices]) <= 0.1:
        root_depth = max(
            FALLBACK_ROOT_DEPTH_M,
            0.2 - float(np.min(relative_depth[arm_indices])),
        )
        absolute_depth = root_depth + relative_depth
    points = np.column_stack(
        (
            normalized[:, 0] * absolute_depth,
            normalized[:, 1] * absolute_depth,
            absolute_depth,
        )
    )
    return points, root_depth


class SimccRetargeter:
    """Body-size-invariant absolute bimanual retargeting."""

    def __init__(
        self,
        confidence_threshold: float = 0.35,
        smoothing_time_constant_s: float = 0.25,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        if smoothing_time_constant_s < 0.0:
            raise ValueError("smoothing_time_constant_s must be non-negative")
        self.smoothing_time_constant_s = smoothing_time_constant_s
        self.last_upper_directions: dict[str, np.ndarray] = {}
        self.last_forearm_directions: dict[str, np.ndarray] = {}
        self.last_hand_rotations: dict[str, np.ndarray] = {}
        self.last_inference_time_ns: int | None = None

    def reset_smoothing(self) -> None:
        """Forget pose-filter history after a geometry or tracking reset."""

        self.last_upper_directions.clear()
        self.last_forearm_directions.clear()
        self.last_hand_rotations.clear()
        self.last_inference_time_ns = None

    def select_person(self, scores: np.ndarray) -> int:
        body_scores = np.asarray(scores)[:, :17]
        means = np.nanmean(body_scores, axis=1)
        if not len(means) or not np.isfinite(means).any():
            raise ValueError("No valid person")
        return int(np.nanargmax(means))

    def confidence_summary(
        self, scores: np.ndarray, mode: RetargetMode
    ) -> tuple[float, float]:
        person = self.select_person(scores)
        required_scores = np.asarray(scores, dtype=np.float64)[
            person, required_keypoint_indices(mode)
        ]
        return float(np.min(required_scores)), float(np.mean(required_scores))

    def make_target(
        self,
        *,
        sequence: int,
        capture_time_ns: int,
        inference_time_ns: int,
        simcc: np.ndarray,
        scores: np.ndarray,
        mode: RetargetMode,
        keypoints_2d: np.ndarray | None = None,
        camera_intrinsics: dict[str, object] | None = None,
    ) -> RobotTarget:
        if mode not in RETARGET_MODES:
            raise ValueError(f"Unsupported retargeting mode: {mode}")
        person = self.select_person(scores)
        person_scores = np.asarray(scores, dtype=np.float64)[person]
        estimated_root_depth: float | None = None
        if camera_intrinsics is not None:
            if keypoints_2d is None:
                raise ValueError("Camera intrinsics require original-image keypoints")
            camera_points, estimated_root_depth = reconstruct_with_intrinsics(
                np.asarray(simcc)[person],
                np.asarray(keypoints_2d)[person],
                camera_intrinsics,
            )
        else:
            camera_points = simcc_to_camera_points(np.asarray(simcc)[person])
        robot_points = camera_points @ CAMERA_TO_ROBOT.T

        for side, indices in BODY.items():
            required = [indices["shoulder"], indices["elbow"]]
            if mode != "elbow":
                required.append(indices["wrist"])
                required.extend(
                    indices["hand_start"] + offset
                    for offset in HAND_FRAME_OFFSETS
                )
            if np.min(person_scores[required]) < self.confidence_threshold:
                raise ValueError(f"{side} arm/hand confidence is too low")

        smoothing_alpha = 1.0
        if self.last_inference_time_ns is not None:
            period_s = (inference_time_ns - self.last_inference_time_ns) / 1e9
            if period_s > 0.0:
                smoothing_alpha = exponential_smoothing_alpha(
                    period_s, self.smoothing_time_constant_s
                )
        self.last_inference_time_ns = inference_time_ns

        arms: dict[str, ArmTarget] = {}
        for side, indices in BODY.items():
            shoulder = robot_points[indices["shoulder"]]
            elbow = robot_points[indices["elbow"]]
            upper_direction = _unit(elbow - shoulder)
            if side in self.last_upper_directions:
                upper_direction = _unit(
                    smoothing_alpha * upper_direction
                    + (1.0 - smoothing_alpha)
                    * self.last_upper_directions[side]
                )
            self.last_upper_directions[side] = upper_direction.copy()
            elbow_target = (
                SHOULDER_POSITIONS[side] + UPPER_ARM_LENGTH_M * upper_direction
            )

            wrist_target: np.ndarray | None = None
            wrist_rotation: np.ndarray | None = None
            if mode != "elbow":
                wrist = robot_points[indices["wrist"]]
                forearm_direction = _unit(wrist - elbow)
                if side in self.last_forearm_directions:
                    forearm_direction = _unit(
                        smoothing_alpha * forearm_direction
                        + (1.0 - smoothing_alpha)
                        * self.last_forearm_directions[side]
                    )
                self.last_forearm_directions[side] = forearm_direction.copy()
                wrist_target = (
                    elbow_target + FOREARM_LENGTH_M * forearm_direction
                )
                wrist_rotation = hand_frame(
                    robot_points, indices["hand_start"], side
                )
                if side in self.last_hand_rotations:
                    wrist_rotation = smooth_rotation(
                        self.last_hand_rotations[side],
                        wrist_rotation,
                        smoothing_alpha,
                    )
                self.last_hand_rotations[side] = wrist_rotation.copy()
            required = [indices["shoulder"], indices["elbow"]]
            if mode != "elbow":
                required.append(indices["wrist"])
                required.extend(
                    indices["hand_start"] + offset
                    for offset in HAND_FRAME_OFFSETS
                )
            arm_confidence = float(np.mean(person_scores[required]))
            arms[side] = ArmTarget(
                elbow_position_m=tuple(float(v) for v in elbow_target),
                wrist_position_m=(
                    tuple(float(v) for v in wrist_target)
                    if wrist_target is not None
                    else None
                ),
                confidence=arm_confidence,
                wrist_rotation=(
                    tuple(float(v) for v in wrist_rotation.reshape(-1))
                    if wrist_rotation is not None
                    else None
                ),
            )

        return RobotTarget(
            sequence=sequence,
            capture_time_ns=capture_time_ns,
            inference_time_ns=inference_time_ns,
            mode=mode,
            camera_intrinsics_enabled=camera_intrinsics is not None,
            camera_intrinsics_source=(
                str(camera_intrinsics["source"])
                if camera_intrinsics is not None
                else None
            ),
            estimated_root_depth_m=estimated_root_depth,
            left=arms["left"],
            right=arms["right"],
        )


def target_record(
    target: RobotTarget,
    *,
    keypoints_simcc: np.ndarray | None = None,
    person_index: int | None = None,
) -> dict[str, object]:
    """JSON-compatible target record used by the assessment logger."""

    record = asdict(target)
    if keypoints_simcc is not None:
        points = np.asarray(keypoints_simcc)
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError(
                "keypoints_simcc must have shape (keypoints, 3), "
                f"got {points.shape}"
            )
        record["keypoints_simcc"] = points.tolist()
        record["keypoints_simcc_person_index"] = person_index
    return record
