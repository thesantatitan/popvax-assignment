import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from retargeting_demo.camera_calibration import (
    CalibrationStore,
    CharucoCalibrationSession,
    camera_profile_id,
    reasonable_default,
)
from retargeting_demo.retarget import _estimate_root_depth


def test_reasonable_default_is_centered_60_degree_camera() -> None:
    profile = reasonable_default(640, 480)
    matrix = np.asarray(profile["camera_matrix"])

    assert matrix[0, 0] == pytest.approx(320.0 / np.tan(np.deg2rad(30.0)))
    assert matrix[1, 1] == matrix[0, 0]
    assert matrix[0, 2] == pytest.approx(319.5)
    assert matrix[1, 2] == pytest.approx(239.5)
    assert profile["distortion_coefficients"] == [0.0] * 5


def test_saved_intrinsics_scale_to_stream_resolution(tmp_path: Path) -> None:
    path = tmp_path / "cameras.json"
    path.write_text('{"version": 1, "profiles": {}}\n', encoding="utf-8")
    store = CalibrationStore(path)
    profile_id = camera_profile_id("camera-one")
    store.save(
        profile_id,
        {
            "image_width": 1280,
            "image_height": 960,
            "camera_matrix": [[1000, 0, 640], [0, 1000, 480], [0, 0, 1]],
            "distortion_coefficients": [0, 0, 0, 0, 0],
            "reprojection_error_px": 0.2,
            "source": "charuco",
        },
    )

    _, resolved = store.resolve("camera-one", 640, 480)

    np.testing.assert_allclose(
        resolved["camera_matrix"],
        [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
    )
    assert resolved["source"] == "calibrated"
    assert json.loads(path.read_text())["profiles"][profile_id]


def test_printed_board_is_detectable() -> None:
    board_path = (
        Path(__file__).resolve().parents[1]
        / "calibration-board-a4-300dpi.png"
    )
    image = cv2.imread(str(board_path), cv2.IMREAD_GRAYSCALE)
    success = CharucoCalibrationSession().capture(
        cv2.imencode(".jpg", image)[1].tobytes()
    )

    assert success["accepted"]
    assert success["corners"] == 54

    duplicate = CharucoCalibrationSession()
    first = duplicate.capture(cv2.imencode(".jpg", image)[1].tobytes())
    second = duplicate.capture(cv2.imencode(".jpg", image)[1].tobytes())
    assert first["accepted"]
    assert not second["accepted"]
    assert second["views"] == 1
    assert "too similar" in second["message"]


def test_shoulder_width_recovers_root_depth() -> None:
    root_depth = 2.0
    depth_delta = 0.1
    lateral_delta = np.sqrt(0.38**2 - depth_delta**2)
    relative_depth = np.zeros(133)
    relative_depth[5] = depth_delta
    normalized_rays = np.zeros((133, 2))
    normalized_rays[5, 0] = (lateral_delta / 2.0) / (
        root_depth + depth_delta
    )
    normalized_rays[6, 0] = (-lateral_delta / 2.0) / root_depth

    estimated = _estimate_root_depth(normalized_rays, relative_depth)

    assert estimated == pytest.approx(root_depth)
