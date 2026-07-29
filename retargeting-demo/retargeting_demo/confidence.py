"""Continuous-confidence engagement policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConfidenceState:
    ready: bool
    state: str
    continuous_seconds: float
    required_seconds: float


class ContinuousConfidenceGate:
    """Require an uninterrupted valid interval before emitting commands."""

    def __init__(self, required_seconds: float = 2.0) -> None:
        if required_seconds <= 0.0:
            raise ValueError("required_seconds must be positive")
        self.required_seconds = required_seconds
        self._valid_since_ns: int | None = None
        self._has_engaged = False

    def reset(self) -> ConfidenceState:
        self._valid_since_ns = None
        self._has_engaged = False
        return self._state(False, "waiting", 0.0)

    def update(self, valid: bool, now_ns: int) -> ConfidenceState:
        if not valid:
            self._valid_since_ns = None
            state = "holding" if self._has_engaged else "waiting"
            return self._state(False, state, 0.0)

        if self._valid_since_ns is None:
            self._valid_since_ns = now_ns
        elapsed = max(0.0, (now_ns - self._valid_since_ns) / 1_000_000_000.0)
        ready = elapsed >= self.required_seconds
        if ready:
            self._has_engaged = True
            state = "tracking"
        else:
            state = "reacquiring" if self._has_engaged else "acquiring"
        return self._state(ready, state, elapsed)

    def _state(
        self, ready: bool, state: str, elapsed: float
    ) -> ConfidenceState:
        return ConfidenceState(
            ready=ready,
            state=state,
            continuous_seconds=min(elapsed, self.required_seconds),
            required_seconds=self.required_seconds,
        )
