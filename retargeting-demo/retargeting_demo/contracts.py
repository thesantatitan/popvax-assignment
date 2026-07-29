"""Pickle-friendly contracts shared by the independent worker processes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RetargetMode = Literal[
    "elbow", "end_effector", "both", "both_orientation"
]
RETARGET_MODES: tuple[RetargetMode, ...] = (
    "elbow",
    "end_effector",
    "both",
    "both_orientation",
)
ELBOW_MODES = frozenset({"elbow", "both", "both_orientation"})
WRIST_MODES = frozenset({"end_effector", "both", "both_orientation"})
ORIENTATION_MODES = frozenset({"both_orientation"})


@dataclass(frozen=True, slots=True)
class BrowserFrame:
    sequence: int
    capture_time_ns: int
    jpeg: bytes


@dataclass(frozen=True, slots=True)
class ArmTarget:
    elbow_position_m: tuple[float, float, float]
    wrist_position_m: tuple[float, float, float] | None
    confidence: float
    wrist_rotation: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class RobotTarget:
    """Desired bimanual state in the OpenArm arm_origin frame and SI units."""

    sequence: int
    capture_time_ns: int
    inference_time_ns: int
    mode: RetargetMode
    camera_intrinsics_enabled: bool
    camera_intrinsics_source: str | None
    estimated_root_depth_m: float | None
    left: ArmTarget
    right: ArmTarget


@dataclass(frozen=True, slots=True)
class JointRetargetingTarget:
    """Desired bimanual OpenArm state passed from IK to joint control."""

    source_target_sequence: int
    mode: RetargetMode | None
    left_joint_positions_rad: tuple[float, ...]
    right_joint_positions_rad: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            len(self.left_joint_positions_rad) != 7
            or len(self.right_joint_positions_rad) != 7
        ):
            raise ValueError(
                "JointRetargetingTarget requires seven joints per arm"
            )

    @property
    def positions_rad(self) -> tuple[float, ...]:
        return (
            self.left_joint_positions_rad
            + self.right_joint_positions_rad
        )


@dataclass(frozen=True, slots=True)
class RenderedFrame:
    sequence: int
    jpeg: bytes
