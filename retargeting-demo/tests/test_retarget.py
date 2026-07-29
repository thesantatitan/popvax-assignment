import numpy as np
import pytest

from retargeting_demo.retarget import (
    BODY,
    CAMERA_TO_ROBOT,
    FOREARM_LENGTH_M,
    UPPER_ARM_LENGTH_M,
    SimccRetargeter,
    exponential_smoothing_alpha,
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


def test_pose_smoothing_is_rate_independent() -> None:
    alpha_10hz = exponential_smoothing_alpha(1.0 / 10.0, 0.12)
    alpha_20hz = exponential_smoothing_alpha(1.0 / 20.0, 0.12)

    remaining_10hz = (1.0 - alpha_10hz) ** 10
    remaining_20hz = (1.0 - alpha_20hz) ** 20
    assert remaining_10hz == pytest.approx(remaining_20hz)
    assert remaining_10hz == pytest.approx(np.exp(-1.0 / 0.12))


def test_zero_time_constant_disables_pose_smoothing() -> None:
    assert exponential_smoothing_alpha(1.0 / 20.0, 0.0) == 1.0


def test_retargeter_smooths_pose_and_reset_forgets_history() -> None:
    first_pose, scores = synthetic_pose()
    second_pose = first_pose.copy()
    second_pose[0, BODY["left"]["wrist"]] += np.array([45.0, -35.0, 20.0])
    filtered = SimccRetargeter(smoothing_time_constant_s=0.12)
    unfiltered = SimccRetargeter(smoothing_time_constant_s=0.0)
    for retargeter in (filtered, unfiltered):
        retargeter.make_target(
            sequence=1,
            capture_time_ns=0,
            inference_time_ns=1_000_000_000,
            simcc=first_pose,
            scores=scores,
            mode="both",
        )

    filtered_target = filtered.make_target(
        sequence=2,
        capture_time_ns=0,
        inference_time_ns=1_050_000_000,
        simcc=second_pose,
        scores=scores,
        mode="both",
    )
    unfiltered_target = unfiltered.make_target(
        sequence=2,
        capture_time_ns=0,
        inference_time_ns=1_050_000_000,
        simcc=second_pose,
        scores=scores,
        mode="both",
    )

    assert filtered_target.left.wrist_position_m != pytest.approx(
        unfiltered_target.left.wrist_position_m
    )
    filtered.reset_smoothing()
    reset_target = filtered.make_target(
        sequence=3,
        capture_time_ns=0,
        inference_time_ns=2_000_000_000,
        simcc=second_pose,
        scores=scores,
        mode="both",
    )
    assert reset_target.left.wrist_position_m == pytest.approx(
        unfiltered_target.left.wrist_position_m
    )


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
        assert arm.wrist_rotation is not None
        rotation = np.asarray(arm.wrist_rotation).reshape(3, 3)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
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


def test_hand_confidence_is_required_only_for_end_effector_modes() -> None:
    simcc, scores = synthetic_pose()
    scores[0, BODY["left"]["hand_start"] + 9] = 0.1
    elbow_target = SimccRetargeter().make_target(
        sequence=1,
        capture_time_ns=1,
        inference_time_ns=2,
        simcc=simcc,
        scores=scores,
        mode="elbow",
    )
    assert elbow_target.left.wrist_rotation is None

    with pytest.raises(ValueError, match="hand confidence"):
        SimccRetargeter().make_target(
            sequence=1,
            capture_time_ns=1,
            inference_time_ns=2,
            simcc=simcc,
            scores=scores,
            mode="both",
        )


def test_modes_require_only_the_landmarks_they_control() -> None:
    assert required_keypoint_indices("elbow") == [5, 6, 7, 8]
    end_effector_indices = [
        5,
        6,
        7,
        8,
        9,
        10,
        91,
        96,
        100,
        104,
        108,
        112,
        117,
        121,
        125,
        129,
    ]
    assert required_keypoint_indices("end_effector") == end_effector_indices
    assert required_keypoint_indices("both") == end_effector_indices

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
