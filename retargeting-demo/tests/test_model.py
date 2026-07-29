from pathlib import Path

import mujoco
import pytest

from retargeting_demo.simulation import _lifter_top_command

MODEL_PATH = Path(__file__).resolve().parents[1] / "vendor/openarm-v2/cell.xml"


def test_vendored_model_keeps_actuated_lifter() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "openarm_lifter_joint"
    )
    actuator, top = _lifter_top_command(model)

    assert joint >= 0
    assert model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_SLIDE
    assert model.actuator_trnid[actuator, 0] == joint
    assert top == pytest.approx(0.3)
    assert model.nu == 17
