from pathlib import Path

import mujoco
import numpy as np
import pytest

from retargeting_demo.contracts import ArmTarget, RobotTarget
from retargeting_demo.simulation import BimanualIk, _initialize_model

MODEL_PATH = Path(__file__).resolve().parents[1] / "vendor/openarm-v2/cell.xml"


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
