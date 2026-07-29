"""Shared locations for machine-specific runtime artifacts."""

from __future__ import annotations

import os
from pathlib import Path


def tensorrt_engine_path() -> Path:
    """Return the configured engine path or the portable per-user default."""
    configured = os.getenv("RTMW3D_TRT_ENGINE")
    if configured:
        return Path(configured).expanduser()
    cache_root = Path(
        os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    ).expanduser()
    return cache_root / "rtmlib" / "hub" / "checkpoints" / "rtmw3d-l-fp32.plan"
