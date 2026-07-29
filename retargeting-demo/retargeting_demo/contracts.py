"""Pickle-friendly contracts shared by the independent worker processes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BrowserFrame:
    sequence: int
    capture_time_ns: int
    jpeg: bytes


@dataclass(frozen=True, slots=True)
class ArmTarget:
    elbow_position_m: tuple[float, float, float]
    wrist_position_m: tuple[float, float, float]
    wrist_rotation: tuple[float, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class RobotTarget:
    """Desired bimanual state in the OpenArm arm_origin frame and SI units."""

    sequence: int
    capture_time_ns: int
    inference_time_ns: int
    left: ArmTarget
    right: ArmTarget


@dataclass(frozen=True, slots=True)
class RenderedFrame:
    sequence: int
    jpeg: bytes
