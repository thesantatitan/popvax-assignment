"""A small, single-person ROI policy layered on top of rtmlib's tracker."""

from __future__ import annotations

import numpy as np
from rtmlib import PoseTracker


class PersistentRoiPoseTracker(PoseTracker):
    """Reuse pose-derived ROIs and request an early detector refresh when needed.

    rtmlib already carries the last-frame boxes between detector calls. This
    subclass preserves that behavior and adds two conservative safety checks:

    * low-confidence body keypoints request a detector refresh;
    * confident keypoints close to the previous crop boundary request a
      detector refresh before the subject can leave the crop.

    It is intended for the one-person webcam case and deliberately keeps the
    existing RTMW3D model and rtmlib post-processing path unchanged.
    """

    def __init__(
        self,
        *args: object,
        redetect_confidence: float = 0.35,
        boundary_fraction: float = 0.08,
        **kwargs: object,
    ) -> None:
        self.redetect_confidence = redetect_confidence
        self.boundary_fraction = boundary_fraction
        self._force_detection = False
        self._early_refresh_block_until = 0
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self._force_detection = False
        self._early_refresh_block_until = 0

    def __call__(self, image: np.ndarray):
        frame_before = self.frame_cnt
        scheduled_detection = frame_before % self.det_frequency == 0
        force_detection = (
            self.det_model is not None
            and self._force_detection
            and not scheduled_detection
            and frame_before >= self._early_refresh_block_until
        )
        previous_boxes = [np.asarray(box, dtype=np.float32) for box in self.bboxes_last_frame]

        # PoseTracker decides whether to run detection from frame_cnt. Make a
        # forced refresh look like a scheduled frame, then restore the normal
        # counter so early refreshes do not permanently shift the cadence.
        if force_detection:
            self.frame_cnt = 0
        try:
            result = super().__call__(image)
        finally:
            if force_detection:
                self.frame_cnt = frame_before + 1

        needs_redetect = self._needs_redetect(
            result,
            previous_boxes,
            detection_ran=scheduled_detection or force_detection,
        )
        if force_detection:
            # Permit at most one early refresh before the next scheduled
            # detector frame. This prevents a difficult crop from turning the
            # detector back into an every-frame workload.
            self._early_refresh_block_until = frame_before + (
                self.det_frequency - (frame_before % self.det_frequency)
            )
            self._force_detection = False
        elif scheduled_detection:
            self._force_detection = False
        else:
            self._force_detection = needs_redetect
        return result

    def _needs_redetect(
        self,
        result: object,
        previous_boxes: list[np.ndarray],
        detection_ran: bool,
    ) -> bool:
        if not isinstance(result, tuple) or len(result) != 4:
            return True

        _, scores_raw, _, keypoints_2d_raw = result
        scores = np.asarray(scores_raw)
        keypoints_2d = np.asarray(keypoints_2d_raw)
        if scores.ndim != 2 or keypoints_2d.ndim != 3 or len(keypoints_2d) == 0:
            return True

        body_scores = scores[:, :17]
        body_confidence = np.nanmean(body_scores, axis=1)
        if not np.isfinite(body_confidence).any():
            return True
        primary_index = int(np.nanargmax(body_confidence))
        if body_confidence[primary_index] < self.redetect_confidence:
            return True

        # If this frame was a detector frame, the returned pose-derived boxes
        # are the new reference. Boundary checks start on the next frame.
        if detection_ran or not previous_boxes:
            return False

        # This policy is intentionally for the one-operator webcam case. If
        # the detector has produced several competing subjects, do not let a
        # false positive's hand/limb geometry force an extra full detector pass;
        # the next scheduled detector frame will resolve the assignment.
        if len(keypoints_2d) != 1 or len(previous_boxes) != 1:
            return False

        keypoints = keypoints_2d[primary_index]
        point_scores = scores[primary_index]
        # Use the 17 body joints for crop safety. They include wrists and
        # elbows, while excluding face/hand landmarks whose jitter would make
        # a stable whole-body ROI look like it is constantly overflowing.
        keypoints = keypoints[:17]
        point_scores = point_scores[:17]
        valid = (
            np.isfinite(keypoints).all(axis=1)
            & (point_scores >= self.redetect_confidence)
        )
        if valid.sum() < 2:
            return True

        points = keypoints[valid]
        center = points.mean(axis=0)
        distances = [
            float(np.linalg.norm(center - ((box[:2] + box[2:]) * 0.5)))
            for box in previous_boxes
        ]
        box = previous_boxes[int(np.argmin(distances))]
        # RTMPose3d's top-down preprocessing expands the input box by 1.25.
        # Compare against that effective crop, not the tighter pose-derived
        # box stored by PoseTracker.
        box_center = (box[:2] + box[2:]) * 0.5
        box_half_size = (box[2:] - box[:2]) * 0.5 * 1.25
        box = np.concatenate((box_center - box_half_size, box_center + box_half_size))
        width = max(float(box[2] - box[0]), 1.0)
        height = max(float(box[3] - box[1]), 1.0)
        margin_x = width * self.boundary_fraction
        margin_y = height * self.boundary_fraction
        near_boundary = (
            (points[:, 0] <= box[0] + margin_x).any()
            or (points[:, 0] >= box[2] - margin_x).any()
            or (points[:, 1] <= box[1] + margin_y).any()
            or (points[:, 1] >= box[3] - margin_y).any()
        )
        if near_boundary:
            return True

        return False
