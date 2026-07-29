"""Convert decoded RTMW3D SimCC landmarks into robot-base targets."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .contracts import RETARGET_MODES, ArmTarget, RetargetMode, RobotTarget

BODY = {
    "left": {"shoulder": 5, "elbow": 7, "wrist": 9, "hand_start": 91},
    "right": {"shoulder": 6, "elbow": 8, "wrist": 10, "hand_start": 112},
}
SHOULDER_POSITIONS = {
    "left": np.array([0.0, 0.1535, 0.0]),
    "right": np.array([0.0, -0.1535, 0.0]),
}
UPPER_ARM_LENGTH_M = 0.220
FOREARM_LENGTH_M = 0.216
# RTMW3D: +x image-right, +y image-down, +z depth. OpenArm arm_origin:
# +x forward into the cell, +y robot-left, +z up.
CAMERA_TO_ROBOT = np.array(
    [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)


def required_keypoint_indices(mode: RetargetMode) -> list[int]:
    if mode not in RETARGET_MODES:
        raise ValueError(f"Unsupported retargeting mode: {mode}")
    required: list[int] = []
    for indices in BODY.values():
        required.extend([indices["shoulder"], indices["elbow"]])
        if mode != "elbow":
            required.append(indices["wrist"])
    return sorted(set(required))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-6:
        raise ValueError("Degenerate limb or hand direction")
    return vector / norm


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


class SimccRetargeter:
    """Body-size-invariant absolute bimanual retargeting."""

    def __init__(
        self,
        confidence_threshold: float = 0.35,
        smoothing_alpha: float = 0.55,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        self.smoothing_alpha = smoothing_alpha
        self.last_upper_directions: dict[str, np.ndarray] = {}
        self.last_forearm_directions: dict[str, np.ndarray] = {}

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
    ) -> RobotTarget:
        if mode not in RETARGET_MODES:
            raise ValueError(f"Unsupported retargeting mode: {mode}")
        person = self.select_person(scores)
        person_scores = np.asarray(scores, dtype=np.float64)[person]
        camera_points = simcc_to_camera_points(np.asarray(simcc)[person])
        robot_points = camera_points @ CAMERA_TO_ROBOT.T

        for side, indices in BODY.items():
            required = [indices["shoulder"], indices["elbow"]]
            if mode != "elbow":
                required.append(indices["wrist"])
            if np.min(person_scores[required]) < self.confidence_threshold:
                raise ValueError(f"{side} arm confidence is too low")

        arms: dict[str, ArmTarget] = {}
        for side, indices in BODY.items():
            shoulder = robot_points[indices["shoulder"]]
            elbow = robot_points[indices["elbow"]]
            upper_direction = _unit(elbow - shoulder)
            if side in self.last_upper_directions:
                alpha = self.smoothing_alpha
                upper_direction = _unit(
                    alpha * upper_direction
                    + (1.0 - alpha) * self.last_upper_directions[side]
                )
            self.last_upper_directions[side] = upper_direction.copy()
            elbow_target = (
                SHOULDER_POSITIONS[side] + UPPER_ARM_LENGTH_M * upper_direction
            )

            wrist_target: np.ndarray | None = None
            if mode != "elbow":
                wrist = robot_points[indices["wrist"]]
                forearm_direction = _unit(wrist - elbow)
                if side in self.last_forearm_directions:
                    alpha = self.smoothing_alpha
                    forearm_direction = _unit(
                        alpha * forearm_direction
                        + (1.0 - alpha) * self.last_forearm_directions[side]
                    )
                self.last_forearm_directions[side] = forearm_direction.copy()
                wrist_target = (
                    elbow_target + FOREARM_LENGTH_M * forearm_direction
                )
            required = [indices["shoulder"], indices["elbow"]]
            if mode != "elbow":
                required.append(indices["wrist"])
            arm_confidence = float(np.mean(person_scores[required]))
            arms[side] = ArmTarget(
                elbow_position_m=tuple(float(v) for v in elbow_target),
                wrist_position_m=(
                    tuple(float(v) for v in wrist_target)
                    if wrist_target is not None
                    else None
                ),
                confidence=arm_confidence,
            )

        return RobotTarget(
            sequence=sequence,
            capture_time_ns=capture_time_ns,
            inference_time_ns=inference_time_ns,
            mode=mode,
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
