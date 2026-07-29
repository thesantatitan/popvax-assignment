from pathlib import Path

import mujoco
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
    assert ik.last_diagnostics["iterations"] == 25
    assert ik.last_diagnostics["maximum_residual_m"] == pytest.approx(0.0)
    assert ik.last_diagnostics["solution_delta_norm_rad"] == pytest.approx(0.0)
    assert set(ik.last_diagnostics["residuals"]) == {"left", "right"}
