"""Validate the pinned bimanual OpenArm model and actuator-driven stepping."""

from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / ".assets" / "openarm_mujoco" / "v1" / "openarm_bimanual.xml"
ARM_JOINTS = [
    f"openarm_{side}_joint{index}"
    for side in ("left", "right")
    for index in range(1, 8)
]
ARM_ACTUATORS = [
    f"{side}_joint{index}_ctrl"
    for side in ("left", "right")
    for index in range(1, 8)
]


def named_ids(model: mujoco.MjModel, object_type: mujoco.mjtObj, names: list[str]) -> list[int]:
    ids = [mujoco.mj_name2id(model, object_type, name) for name in names]
    missing = [name for name, object_id in zip(names, ids, strict=True) if object_id < 0]
    if missing:
        raise RuntimeError(f"Missing model objects: {missing}")
    return ids


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"OpenArm model not found: {MODEL_PATH}. Run ./setup.sh first.")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    joint_ids = named_ids(model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS)
    actuator_ids = named_ids(model, mujoco.mjtObj.mjOBJ_ACTUATOR, ARM_ACTUATORS)

    collision_geoms = int(np.count_nonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
    if collision_geoms == 0:
        raise RuntimeError("Model contains no active collision geometry")
    if not np.all(model.jnt_limited[joint_ids]):
        raise RuntimeError("One or more arm joints do not have limits")

    initial_qpos = data.qpos.copy()
    data.ctrl[actuator_ids] = 0.2
    for _ in range(100):
        mujoco.mj_step(model, data)
    arm_qpos_addresses = model.jnt_qposadr[joint_ids]
    displacement = float(np.linalg.norm(data.qpos[arm_qpos_addresses] - initial_qpos[arm_qpos_addresses]))
    if displacement <= 1e-6:
        raise RuntimeError("Actuator command did not move the arm joints through physics")

    print(f"MuJoCo version: {mujoco.__version__}")
    print(f"Model: {MODEL_PATH}")
    print(f"Arm joints: {len(joint_ids)} (7 left + 7 right)")
    print(f"Arm actuators: {len(actuator_ids)}")
    print(f"Active collision geoms: {collision_geoms}")
    print(f"100-step actuator displacement: {displacement:.6f} rad")
    print("OpenArm MuJoCo setup verified.")


if __name__ == "__main__":
    main()

