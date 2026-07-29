"""Convert decoded RTMW3D SimCC landmarks into robot-base targets."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .contracts import ArmTarget, RobotTarget

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
HOME_WRIST_ROTATION = np.array(
    [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)
HOME_UPPER_DIRECTION = np.array([0.0, 0.0, -1.0])
HOME_FOREARM_DIRECTION = np.array([1.0, 0.0, 0.0])

# RTMW3D: +x image-right, +y image-down, +z depth. OpenArm arm_origin:
# +x forward into the cell, +y robot-left, +z up.
CAMERA_TO_ROBOT = np.array(
    [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-6:
        raise ValueError("Degenerate limb or hand direction")
    return vector / norm


def _rotation_between(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """Return the minimum rotation taking one unit direction to another."""

    source = _unit(source)
    destination = _unit(destination)
    cross = np.cross(source, destination)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, destination), -1.0, 1.0))
    if sine < 1e-8:
        if cosine > 0.0:
            return np.eye(3)
        basis = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.8:
            basis = np.array([0.0, 1.0, 0.0])
        axis = _unit(np.cross(source, basis))
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    axis = cross / sine
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)


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


def hand_frame(points: np.ndarray, hand_start: int) -> np.ndarray:
    """Return a right-handed palm frame from wrist and MCP landmarks."""

    hand = points[hand_start : hand_start + 21]
    wrist = hand[0]
    palm_forward = _unit((hand[5] + hand[9] + hand[13] + hand[17]) * 0.25 - wrist)
    palm_across = _unit(hand[5] - hand[17])
    palm_normal = _unit(np.cross(palm_forward, palm_across))
    palm_across = _unit(np.cross(palm_normal, palm_forward))
    return np.column_stack((palm_forward, palm_across, palm_normal))


class SimccRetargeter:
    """Body-size-invariant bimanual retargeting with hand-frame calibration."""

    def __init__(
        self,
        confidence_threshold: float = 0.35,
        smoothing_alpha: float = 0.55,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        self.smoothing_alpha = smoothing_alpha
        self.hand_reference: dict[str, np.ndarray] = {}
        self.last_rotations = {
            side: HOME_WRIST_ROTATION.copy() for side in ("left", "right")
        }
        self.limb_reference: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.last_directions: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def calibrated(self) -> bool:
        return len(self.hand_reference) == 2

    def clear_calibration(self) -> None:
        self.hand_reference.clear()
        self.limb_reference.clear()
        self.last_directions.clear()

    def select_person(self, scores: np.ndarray) -> int:
        body_scores = np.asarray(scores)[:, :17]
        means = np.nanmean(body_scores, axis=1)
        if not len(means) or not np.isfinite(means).any():
            raise ValueError("No valid person")
        return int(np.nanargmax(means))

    def make_target(
        self,
        *,
        sequence: int,
        capture_time_ns: int,
        inference_time_ns: int,
        simcc: np.ndarray,
        scores: np.ndarray,
        calibrate: bool = False,
    ) -> RobotTarget:
        person = self.select_person(scores)
        person_scores = np.asarray(scores, dtype=np.float64)[person]
        camera_points = simcc_to_camera_points(np.asarray(simcc)[person])
        robot_points = camera_points @ CAMERA_TO_ROBOT.T

        frames: dict[str, np.ndarray] = {}
        for side, indices in BODY.items():
            required = [
                indices["shoulder"],
                indices["elbow"],
                indices["wrist"],
                indices["hand_start"],
                indices["hand_start"] + 5,
                indices["hand_start"] + 9,
                indices["hand_start"] + 13,
                indices["hand_start"] + 17,
            ]
            if np.min(person_scores[required]) < self.confidence_threshold:
                raise ValueError(f"{side} arm/hand confidence is too low")
            frames[side] = hand_frame(robot_points, indices["hand_start"])

        if calibrate:
            self.hand_reference = {
                side: frame.copy() for side, frame in frames.items()
            }
            self.last_directions.clear()

        arms: dict[str, ArmTarget] = {}
        for side, indices in BODY.items():
            shoulder = robot_points[indices["shoulder"]]
            elbow = robot_points[indices["elbow"]]
            wrist = robot_points[indices["wrist"]]
            measured_upper = _unit(elbow - shoulder)
            measured_forearm = _unit(wrist - elbow)
            if calibrate:
                self.limb_reference[side] = (
                    measured_upper.copy(),
                    measured_forearm.copy(),
                )
            if side in self.limb_reference:
                reference_upper, reference_forearm = self.limb_reference[side]
                upper_direction = (
                    _rotation_between(reference_upper, measured_upper)
                    @ HOME_UPPER_DIRECTION
                )
                forearm_direction = (
                    _rotation_between(reference_forearm, measured_forearm)
                    @ HOME_FOREARM_DIRECTION
                )
            else:
                upper_direction = measured_upper
                forearm_direction = measured_forearm
            if side in self.last_directions and not calibrate:
                previous_upper, previous_forearm = self.last_directions[side]
                alpha = self.smoothing_alpha
                upper_direction = _unit(
                    alpha * upper_direction + (1.0 - alpha) * previous_upper
                )
                forearm_direction = _unit(
                    alpha * forearm_direction + (1.0 - alpha) * previous_forearm
                )
            self.last_directions[side] = (
                upper_direction.copy(),
                forearm_direction.copy(),
            )
            elbow_target = (
                SHOULDER_POSITIONS[side] + UPPER_ARM_LENGTH_M * upper_direction
            )
            wrist_target = elbow_target + FOREARM_LENGTH_M * forearm_direction

            reference = self.hand_reference.get(side, frames[side])
            delta = frames[side] @ reference.T
            wrist_rotation = delta @ HOME_WRIST_ROTATION
            if self.calibrated and not calibrate:
                alpha = self.smoothing_alpha
                blended = (
                    alpha * wrist_rotation
                    + (1.0 - alpha) * self.last_rotations[side]
                )
                left, _, right = np.linalg.svd(blended)
                wrist_rotation = left @ right
                if np.linalg.det(wrist_rotation) < 0.0:
                    left[:, -1] *= -1.0
                    wrist_rotation = left @ right
            self.last_rotations[side] = wrist_rotation
            arm_confidence = float(
                np.mean(
                    person_scores[
                        [
                            indices["shoulder"],
                            indices["elbow"],
                            indices["wrist"],
                            indices["hand_start"],
                            indices["hand_start"] + 5,
                            indices["hand_start"] + 9,
                            indices["hand_start"] + 13,
                            indices["hand_start"] + 17,
                        ]
                    ]
                )
            )
            arms[side] = ArmTarget(
                elbow_position_m=tuple(float(v) for v in elbow_target),
                wrist_position_m=tuple(float(v) for v in wrist_target),
                wrist_rotation=tuple(float(v) for v in wrist_rotation.reshape(-1)),
                confidence=arm_confidence,
            )

        return RobotTarget(
            sequence=sequence,
            capture_time_ns=capture_time_ns,
            inference_time_ns=inference_time_ns,
            left=arms["left"],
            right=arms["right"],
        )


def target_record(target: RobotTarget) -> dict[str, object]:
    """JSON-compatible target record used by the assessment logger."""

    return asdict(target)
