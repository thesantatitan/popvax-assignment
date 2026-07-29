from pathlib import Path

import mujoco
import numpy as np
import pytest

from retargeting_demo.contracts import (
    ArmTarget,
    RobotTarget,
)
from retargeting_demo.simulation import (
    ArmConfigurationLimit,
    BimanualIk,
    _initialize_model,
    _lifter_top_command,
    joint_retargeting_target,
    joint_retargeting_target_record,
)

MODEL_PATH = Path(__file__).resolve().parents[1] / "vendor/openarm-v2/cell.xml"


def test_joint_retargeting_target_exposes_both_arms_in_radians() -> None:
    desired = np.arange(14, dtype=float) / 10.0

    target = joint_retargeting_target(
        desired,
        source_target_sequence=7,
        mode="both",
    )
    record = joint_retargeting_target_record(
        target,
        time_ns=123,
        simulation_time_s=0.5,
        control_timestep=30,
        tracking_active=True,
    )

    assert record["state"] == "desired_joint_positions"
    assert record["units"] == "radians"
    assert record["control_timestep"] == 30
    assert record["desired_joint_positions_rad"] == {
        "left": desired[:7].tolist(),
        "right": desired[7:].tolist(),
    }
    assert record["order"] == {
        "left": [f"left_joint{index}" for index in range(1, 8)],
        "right": [f"right_joint{index}" for index in range(1, 8)],
    }


def test_joint_retargeting_target_rejects_wrong_joint_count() -> None:
    with pytest.raises(ValueError, match="14 desired joint positions"):
        joint_retargeting_target(
            np.zeros(13),
            source_target_sequence=7,
            mode="both",
        )


def test_ik_records_solution_delta_and_cartesian_residuals() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    _initialize_model(model, data)
    ik = BimanualIk(model, data)
    achieved = ik.achieved_state(data, None)
    target = RobotTarget(
        sequence=42,
        capture_time_ns=1,
        inference_time_ns=2,
        mode="both",
        camera_intrinsics_enabled=False,
        camera_intrinsics_source=None,
        estimated_root_depth_m=None,
        left=ArmTarget(
            elbow_position_m=tuple(achieved["left"]["elbow_position_m"]),
            wrist_position_m=tuple(achieved["left"]["wrist_position_m"]),
            confidence=1.0,
        ),
        right=ArmTarget(
            elbow_position_m=tuple(achieved["right"]["elbow_position_m"]),
            wrist_position_m=tuple(achieved["right"]["wrist_position_m"]),
            confidence=1.0,
        ),
    )

    solution = ik.solve(target, data.qpos)

    assert solution.shape == (14,)
    assert ik.last_diagnostics is not None
    assert ik.last_diagnostics["target_sequence"] == 42
    assert ik.last_diagnostics["status"] == "converged"
    assert ik.last_diagnostics["solver"] == "mink:daqp"
    assert ik.last_diagnostics["iterations"] == 25
    assert ik.last_diagnostics["maximum_residual_m"] == pytest.approx(0.0)
    assert ik.last_diagnostics["solution_delta_norm_rad"] == pytest.approx(0.0)
    assert set(ik.last_diagnostics["residuals"]) == {"left", "right"}


def test_mink_reduces_position_error_without_moving_non_arm_joints() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    _initialize_model(model, data)
    ik = BimanualIk(model, data)
    achieved = ik.achieved_state(data, None)
    non_arm_qpos = np.setdiff1d(
        np.arange(model.nq), ik.all_qpos_addresses
    )

    arms = {}
    for side in ("left", "right"):
        wrist = np.asarray(achieved[side]["wrist_position_m"]).copy()
        wrist[0] += 0.03
        arms[side] = ArmTarget(
            elbow_position_m=tuple(achieved[side]["elbow_position_m"]),
            wrist_position_m=tuple(wrist),
            confidence=1.0,
        )
    target = RobotTarget(
        sequence=43,
        capture_time_ns=1,
        inference_time_ns=2,
        mode="end_effector",
        camera_intrinsics_enabled=False,
        camera_intrinsics_source=None,
        estimated_root_depth_m=None,
        left=arms["left"],
        right=arms["right"],
    )

    ik.solve(target, data.qpos)

    assert ik.last_diagnostics is not None
    assert ik.last_diagnostics["status"] == "converged"
    assert ik.last_diagnostics["maximum_residual_m"] < 0.01
    np.testing.assert_allclose(
        ik.data.qpos[non_arm_qpos],
        data.qpos[non_arm_qpos],
        atol=1e-12,
    )


def test_lifter_overshoot_is_excluded_from_mink_arm_limits() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    _initialize_model(model, data)
    lifter_actuator, lifter_top = _lifter_top_command(model)
    lifter_joint = model.actuator_trnid[lifter_actuator, 0]
    lifter_qpos = model.jnt_qposadr[lifter_joint]
    data.qpos[lifter_qpos] = lifter_top + 1e-5
    mujoco.mj_forward(model, data)
    ik = BimanualIk(model, data)
    achieved = ik.achieved_state(data, None)
    target = RobotTarget(
        sequence=44,
        capture_time_ns=1,
        inference_time_ns=2,
        mode="both",
        camera_intrinsics_enabled=False,
        camera_intrinsics_source=None,
        estimated_root_depth_m=None,
        left=ArmTarget(
            tuple(achieved["left"]["elbow_position_m"]),
            tuple(achieved["left"]["wrist_position_m"]),
            1.0,
        ),
        right=ArmTarget(
            tuple(achieved["right"]["elbow_position_m"]),
            tuple(achieved["right"]["wrist_position_m"]),
            1.0,
        ),
    )

    ik.solve(target, data.qpos)

    assert isinstance(ik.limits[0], ArmConfigurationLimit)
    assert ik.last_diagnostics is not None
    assert ik.last_diagnostics["solver_error"] is None
    assert ik.last_diagnostics["decision_dofs"] == 14
    assert ik.last_diagnostics["excluded_dofs"] == model.nv - 14
    assert data.qpos[lifter_qpos] == pytest.approx(lifter_top + 1e-5)
    assert ik.data.qpos[lifter_qpos] == pytest.approx(lifter_top + 1e-5)


def test_mink_tracks_end_effector_orientation() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    _initialize_model(model, data)
    ik = BimanualIk(model, data)
    achieved = ik.achieved_state(data, None)
    angle = 0.15
    cosine, sine = np.cos(angle), np.sin(angle)
    local_rotation = np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]
    )
    arms = {}
    for side in ("left", "right"):
        rotation = (
            np.asarray(achieved[side]["wrist_rotation"]).reshape(3, 3)
            @ local_rotation
        )
        arms[side] = ArmTarget(
            elbow_position_m=tuple(achieved[side]["elbow_position_m"]),
            wrist_position_m=tuple(achieved[side]["wrist_position_m"]),
            confidence=1.0,
            wrist_rotation=tuple(rotation.reshape(-1)),
        )
    target = RobotTarget(
        sequence=45,
        capture_time_ns=1,
        inference_time_ns=2,
        mode="both_orientation",
        camera_intrinsics_enabled=False,
        camera_intrinsics_source=None,
        estimated_root_depth_m=None,
        left=arms["left"],
        right=arms["right"],
    )

    ik.solve(target, data.qpos)

    assert ik.last_diagnostics is not None
    assert ik.last_diagnostics["status"] == "converged"
    assert ik.last_diagnostics["maximum_residual_m"] < 0.01
    assert ik.last_diagnostics["maximum_orientation_residual_rad"] < 0.01


def test_position_only_mode_ignores_supplied_orientation() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    _initialize_model(model, data)
    ik = BimanualIk(model, data)
    achieved = ik.achieved_state(data, None)
    arms = {}
    for side in ("left", "right"):
        rotation = np.asarray(achieved[side]["wrist_rotation"]).reshape(3, 3)
        arms[side] = ArmTarget(
            elbow_position_m=tuple(achieved[side]["elbow_position_m"]),
            wrist_position_m=tuple(achieved[side]["wrist_position_m"]),
            confidence=1.0,
            wrist_rotation=tuple(rotation.reshape(-1)),
        )
    target = RobotTarget(
        sequence=46,
        capture_time_ns=1,
        inference_time_ns=2,
        mode="both",
        camera_intrinsics_enabled=False,
        camera_intrinsics_source=None,
        estimated_root_depth_m=None,
        left=arms["left"],
        right=arms["right"],
    )

    ik.solve(target, data.qpos)

    assert ik.last_diagnostics is not None
    assert ik.last_diagnostics["maximum_orientation_residual_rad"] == 0.0
    for side in ("left", "right"):
        assert "wrist_orientation_rad" not in (
            ik.last_diagnostics["residuals"][side]
        )
