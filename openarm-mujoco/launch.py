"""Launch the pinned OpenArm v1 bimanual scene in MuJoCo Viewer."""

from pathlib import Path

import mujoco
import mujoco.viewer

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "retargeting-demo"
    / "vendor"
    / "openarm-v2"
    / "cell.xml"
)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_key)
    for actuator_id in range(model.nu):
        joint_id = model.actuator_trnid[actuator_id, 0]
        if joint_id >= 0:
            data.ctrl[actuator_id] = data.qpos[model.jnt_qposadr[joint_id]]
    mujoco.mj_forward(model, data)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
