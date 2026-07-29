from pathlib import Path

import mujoco
import pytest

MODEL_PATH = Path(__file__).resolve().parents[1] / "vendor/openarm-v2/cell.xml"


def test_vendored_model_has_fixed_center_mount() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    assert (
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "openarm_lifter_joint"
        )
        == -1
    )
    assert (
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lifter_ctrl"
        )
        == -1
    )
    fixed_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "openarm_fixed_center_link"
    )
    assert fixed_body >= 0
    assert model.body_pos[fixed_body, 2] == pytest.approx(1.49)
    assert model.nu == 16
