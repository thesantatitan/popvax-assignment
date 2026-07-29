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


def test_vendored_model_has_no_roof_side_or_front_walls() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    removed_geometries = (
        "cell_vis",
        "cell_left_wall_col",
        "cell_right_wall_col",
        "cell_front_wall_col",
        "cell_roof_col",
    )
    for name in removed_geometries:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) == -1

    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0
    for name in ("sheet", "cell_table_col", "cell_rail_col1", "cell_rail_col2"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0
    assert (
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "body_lifter_col"
        )
        >= 0
    )
