import numpy as np

from retargeting_demo.retarget import (
    BODY,
    CAMERA_TO_ROBOT,
    FOREARM_LENGTH_M,
    UPPER_ARM_LENGTH_M,
    SimccRetargeter,
    required_keypoint_indices,
    simcc_to_camera_points,
    target_record,
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


def test_retargeting_has_robot_segment_lengths() -> None:
    simcc, scores = synthetic_pose()
    retargeter = SimccRetargeter()
    target = retargeter.make_target(
        sequence=4,
        capture_time_ns=100,
        inference_time_ns=200,
        simcc=simcc,
        scores=scores,
        mode="both",
    )
    robot_points = simcc_to_camera_points(simcc[0]) @ CAMERA_TO_ROBOT.T
    for side in ("left", "right"):
        indices = BODY[side]
        arm = getattr(target, side)
        shoulder = np.array([0.0, 0.1535 if side == "left" else -0.1535, 0.0])
        elbow = np.asarray(arm.elbow_position_m)
        wrist = np.asarray(arm.wrist_position_m)
        assert arm.wrist_position_m is not None
        assert np.isclose(np.linalg.norm(elbow - shoulder), UPPER_ARM_LENGTH_M)
        assert np.isclose(np.linalg.norm(wrist - elbow), FOREARM_LENGTH_M)
        measured_upper = (
            robot_points[indices["elbow"]] - robot_points[indices["shoulder"]]
        )
        measured_upper /= np.linalg.norm(measured_upper)
        measured_forearm = (
            robot_points[indices["wrist"]] - robot_points[indices["elbow"]]
        )
        measured_forearm /= np.linalg.norm(measured_forearm)
        np.testing.assert_allclose(
            elbow, shoulder + UPPER_ARM_LENGTH_M * measured_upper
        )
        np.testing.assert_allclose(
            wrist, elbow + FOREARM_LENGTH_M * measured_forearm
        )


def test_hand_confidence_is_ignored() -> None:
    simcc, scores = synthetic_pose()
    scores[0, BODY["left"]["hand_start"] + 9] = 0.1
    retargeter = SimccRetargeter()
    target = retargeter.make_target(
        sequence=1,
        capture_time_ns=1,
        inference_time_ns=2,
        simcc=simcc,
        scores=scores,
        mode="both",
    )
    assert target.mode == "both"


def test_modes_require_only_the_landmarks_they_control() -> None:
    assert required_keypoint_indices("elbow") == [5, 6, 7, 8]
    assert required_keypoint_indices("end_effector") == [5, 6, 7, 8, 9, 10]
    assert required_keypoint_indices("both") == [5, 6, 7, 8, 9, 10]

    simcc, scores = synthetic_pose()
    scores[0, BODY["left"]["wrist"]] = 0.1
    elbow_target = SimccRetargeter().make_target(
        sequence=2,
        capture_time_ns=1,
        inference_time_ns=2,
        simcc=simcc,
        scores=scores,
        mode="elbow",
    )
    assert elbow_target.left.wrist_position_m is None

    for mode in ("end_effector", "both"):
        try:
            SimccRetargeter().make_target(
                sequence=2,
                capture_time_ns=1,
                inference_time_ns=2,
                simcc=simcc,
                scores=scores,
                mode=mode,
            )
        except ValueError as error:
            assert "confidence" in str(error)
        else:
            raise AssertionError(f"{mode} should require wrist confidence")


def test_target_record_includes_selected_simcc_keypoints() -> None:
    simcc, scores = synthetic_pose()
    target = SimccRetargeter().make_target(
        sequence=7,
        capture_time_ns=100,
        inference_time_ns=200,
        simcc=simcc,
        scores=scores,
        mode="end_effector",
    )

    record = target_record(
        target,
        keypoints_simcc=simcc[0],
        person_index=0,
    )

    assert record["keypoints_simcc_person_index"] == 0
    np.testing.assert_allclose(record["keypoints_simcc"], simcc[0])
