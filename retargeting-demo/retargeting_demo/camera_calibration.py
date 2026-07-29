"""ChArUco camera calibration and versioned intrinsic-profile storage."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_FILE = ROOT / "calibrations" / "cameras.json"
BOARD_COLUMNS = 7
BOARD_ROWS = 10
SQUARE_LENGTH_M = 0.024
MARKER_LENGTH_M = 0.018
MIN_CALIBRATION_VIEWS = 12
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
BOARD = cv2.aruco.CharucoBoard(
    (BOARD_COLUMNS, BOARD_ROWS),
    SQUARE_LENGTH_M,
    MARKER_LENGTH_M,
    DICTIONARY,
)


def camera_profile_id(device_id: str) -> str:
    """Create a non-identifying stable key from the browser camera device ID."""

    value = device_id or "default-camera"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def reasonable_default(width: int, height: int) -> dict[str, object]:
    """Return a centered zero-distortion pinhole model with a 60-degree HFOV."""

    focal = 0.5 * width / np.tan(np.deg2rad(30.0))
    return {
        "image_width": width,
        "image_height": height,
        "camera_matrix": [
            [float(focal), 0.0, (width - 1.0) * 0.5],
            [0.0, float(focal), (height - 1.0) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
        "reprojection_error_px": None,
        "source": "reasonable_default_60deg_hfov",
    }


class CalibrationStore:
    def __init__(self, path: Path = CALIBRATION_FILE) -> None:
        self.path = path

    def _read(self) -> dict[str, object]:
        if not self.path.is_file():
            return {"version": 1, "profiles": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def resolve(
        self, device_id: str, width: int, height: int
    ) -> tuple[str, dict[str, object]]:
        profile_id = camera_profile_id(device_id)
        profiles = self._read().get("profiles", {})
        profile = profiles.get(profile_id) if isinstance(profiles, dict) else None
        if not isinstance(profile, dict):
            return profile_id, reasonable_default(width, height)
        source_width = float(profile["image_width"])
        source_height = float(profile["image_height"])
        matrix = np.asarray(profile["camera_matrix"], dtype=np.float64).copy()
        matrix[0, :] *= width / source_width
        matrix[1, :] *= height / source_height
        resolved = dict(profile)
        resolved.update(
            {
                "image_width": width,
                "image_height": height,
                "camera_matrix": matrix.tolist(),
                "source": "calibrated",
            }
        )
        return profile_id, resolved

    def save(self, profile_id: str, profile: dict[str, object]) -> None:
        document = self._read()
        profiles = document.setdefault("profiles", {})
        if not isinstance(profiles, dict):
            raise TypeError("Calibration profiles must be a JSON object")
        profiles[profile_id] = profile
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


@dataclass
class CharucoCalibrationSession:
    image_size: tuple[int, int] | None = None
    object_points: list[np.ndarray] = field(default_factory=list)
    image_points: list[np.ndarray] = field(default_factory=list)
    view_descriptors: list[np.ndarray] = field(default_factory=list)

    @staticmethod
    def _view_descriptor(
        object_points: np.ndarray,
        image_points: np.ndarray,
        image_size: tuple[int, int],
    ) -> np.ndarray:
        """Represent board placement by its projected outer quadrilateral."""

        object_xy = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)[
            :, :2
        ]
        image_xy = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        normalized_image = image_xy / np.asarray(image_size, dtype=np.float64)
        homography, _ = cv2.findHomography(object_xy, normalized_image)
        if homography is None:
            raise ValueError("Could not estimate the board pose")
        width = (BOARD_COLUMNS - 1) * SQUARE_LENGTH_M
        height = (BOARD_ROWS - 1) * SQUARE_LENGTH_M
        outside = np.asarray(
            [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
            dtype=np.float64,
        )
        return cv2.perspectiveTransform(
            outside.reshape(-1, 1, 2), homography
        ).reshape(-1, 2)

    def capture(self, jpeg: bytes) -> dict[str, object]:
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise ValueError("Could not decode calibration image")
        height, width = frame.shape
        size = (width, height)
        if self.image_size is not None and self.image_size != size:
            raise ValueError(
                f"All calibration images must be {self.image_size}, got {size}"
            )
        detector = cv2.aruco.CharucoDetector(BOARD)
        corners, ids, _, _ = detector.detectBoard(frame)
        count = 0 if ids is None else len(ids)
        if count < 12:
            return {
                "accepted": False,
                "corners": count,
                "views": len(self.image_points),
                "required_views": MIN_CALIBRATION_VIEWS,
                "message": "Show more of the board; at least 12 corners are required.",
            }
        object_points, image_points = BOARD.matchImagePoints(corners, ids)
        descriptor = self._view_descriptor(object_points, image_points, size)
        if self.view_descriptors:
            distances = [
                float(np.sqrt(np.mean(np.square(descriptor - previous))))
                for previous in self.view_descriptors
            ]
            if min(distances) < 0.025:
                return {
                    "accepted": False,
                    "corners": count,
                    "views": len(self.image_points),
                    "required_views": MIN_CALIBRATION_VIEWS,
                    "message": "Move or tilt the board more; this view is too similar.",
                }
        self.image_size = size
        self.object_points.append(np.asarray(object_points, dtype=np.float32))
        self.image_points.append(np.asarray(image_points, dtype=np.float32))
        self.view_descriptors.append(descriptor)
        return {
            "accepted": True,
            "corners": count,
            "views": len(self.image_points),
            "required_views": MIN_CALIBRATION_VIEWS,
            "message": "View accepted. Move and tilt the board before the next capture.",
        }

    def calibrate(self) -> dict[str, object]:
        if self.image_size is None or len(self.image_points) < MIN_CALIBRATION_VIEWS:
            raise ValueError(
                f"Capture at least {MIN_CALIBRATION_VIEWS} accepted views first"
            )
        rms, matrix, distortion, _, _ = cv2.calibrateCamera(
            self.object_points,
            self.image_points,
            self.image_size,
            None,
            None,
        )
        return {
            "image_width": self.image_size[0],
            "image_height": self.image_size[1],
            "camera_matrix": matrix.tolist(),
            "distortion_coefficients": distortion.reshape(-1).tolist(),
            "reprojection_error_px": float(rms),
            "source": "charuco",
            "board": {
                "dictionary": "DICT_5X5_100",
                "columns": BOARD_COLUMNS,
                "rows": BOARD_ROWS,
                "square_length_m": SQUARE_LENGTH_M,
                "marker_length_m": MARKER_LENGTH_M,
            },
        }
