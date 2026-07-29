"""Pickle-friendly contracts shared by the independent worker processes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RetargetMode = Literal["elbow", "end_effector", "both"]
RETARGET_MODES: tuple[RetargetMode, ...] = ("elbow", "end_effector", "both")


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


@dataclass(frozen=True, slots=True)
class RobotTarget:
    """Desired bimanual state in the OpenArm arm_origin frame and SI units."""

    sequence: int
    capture_time_ns: int
    inference_time_ns: int
    mode: RetargetMode
    left: ArmTarget
    right: ArmTarget


@dataclass(frozen=True, slots=True)
class RenderedFrame:
    sequence: int
    jpeg: bytes
