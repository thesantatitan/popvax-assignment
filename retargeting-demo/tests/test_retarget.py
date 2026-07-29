import numpy as np

from retargeting_demo.retarget import (
    BODY,
    CAMERA_TO_ROBOT,
    FOREARM_LENGTH_M,
    SimccRetargeter,
    UPPER_ARM_LENGTH_M,
    hand_frame,
    simcc_to_camera_points,
)


def synthetic_pose() -> tuple[np.ndarray, np.ndarray]:
    simcc = np.zeros((1, 133, 3), dtype=np.float64)
    simcc[..., 2] = 192.0
    scores = np.ones((1, 133), dtype=np.float64)
    for side, sign in (("left", 1.0), ("right", -1.0)):
        indices = BODY[side]
        shoulder = np.array([144.0 + sign * 30.0, 120.0, 192.0])
        elbow = shoulder + np.array([sign * 25.0, 55.0, 4.0])
        wrist = elbow + np.array([sign * 18.0, 50.0, -5.0])
        simcc[0, indices["shoulder"]] = shoulder
        simcc[0, indices["elbow"]] = elbow
        simcc[0, indices["wrist"]] = wrist
        start = indices["hand_start"]
        simcc[0, start] = wrist
        for index, offset in {
            5: (sign * 10.0, 24.0, -2.0),
            9: (sign * 3.0, 28.0, -4.0),
            13: (sign * -4.0, 25.0, -3.0),
            17: (sign * -10.0, 20.0, -1.0),
        }.items():
            simcc[0, start + index] = wrist + np.asarray(offset)
    return simcc, scores


def test_camera_mapping_is_a_rotation() -> None:
    np.testing.assert_allclose(CAMERA_TO_ROBOT.T @ CAMERA_TO_ROBOT, np.eye(3))
    assert np.linalg.det(CAMERA_TO_ROBOT) == 1.0


def test_hand_frame_is_orthonormal() -> None:
    simcc, _ = synthetic_pose()
    points = simcc_to_camera_points(simcc[0]) @ CAMERA_TO_ROBOT.T
    frame = hand_frame(points, BODY["left"]["hand_start"])
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-7)
    assert np.linalg.det(frame) > 0.999


def test_retargeting_has_robot_segment_lengths_and_rotation() -> None:
    simcc, scores = synthetic_pose()
    retargeter = SimccRetargeter()
    target = retargeter.make_target(
        sequence=4,
        capture_time_ns=100,
        inference_time_ns=200,
        simcc=simcc,
        scores=scores,
        calibrate=True,
    )
    for side in ("left", "right"):
        arm = getattr(target, side)
        shoulder = np.array([0.0, 0.1535 if side == "left" else -0.1535, 0.0])
        elbow = np.asarray(arm.elbow_position_m)
        wrist = np.asarray(arm.wrist_position_m)
        rotation = np.asarray(arm.wrist_rotation).reshape(3, 3)
        assert np.isclose(np.linalg.norm(elbow - shoulder), UPPER_ARM_LENGTH_M)
        assert np.isclose(np.linalg.norm(wrist - elbow), FOREARM_LENGTH_M)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)


def test_low_hand_confidence_rejects_target() -> None:
    simcc, scores = synthetic_pose()
    scores[0, BODY["left"]["hand_start"] + 9] = 0.1
    retargeter = SimccRetargeter()
    try:
        retargeter.make_target(
            sequence=1,
            capture_time_ns=1,
            inference_time_ns=2,
            simcc=simcc,
            scores=scores,
        )
    except ValueError as error:
        assert "confidence" in str(error)
    else:
        raise AssertionError("low-confidence hand should not produce a target")
