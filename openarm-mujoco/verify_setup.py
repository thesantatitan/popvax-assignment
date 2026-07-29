"""Validate the pinned bimanual OpenArm model and actuator-driven stepping."""

from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / ".assets" / "openarm_mujoco" / "v2" / "cell.xml"
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
    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key < 0:
        raise RuntimeError("OpenArm v2 model is missing the home keyframe")
    mujoco.mj_resetDataKeyframe(model, data, home_key)
    for actuator_id in range(model.nu):
        joint_id = model.actuator_trnid[actuator_id, 0]
        if joint_id >= 0:
            data.ctrl[actuator_id] = data.qpos[model.jnt_qposadr[joint_id]]
    mujoco.mj_forward(model, data)
    joint_ids = named_ids(model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINTS)
    actuator_ids = named_ids(model, mujoco.mjtObj.mjOBJ_ACTUATOR, ARM_ACTUATORS)

    collision_geoms = int(np.count_nonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
    if collision_geoms == 0:
        raise RuntimeError("Model contains no active collision geometry")
    if not np.all(model.jnt_limited[joint_ids]):
        raise RuntimeError("One or more arm joints do not have limits")
    if not np.all(model.actuator_ctrllimited[actuator_ids]):
        raise RuntimeError("One or more v2 arm position actuators lack control limits")
    kp = model.actuator_gainprm[actuator_ids, 0]
    kv = -model.actuator_biasprm[actuator_ids, 2]
    if not np.all(kp > 0) or not np.all(kv > 0):
        raise RuntimeError("One or more v2 arm actuators lack internal PD gains")

    initial_qpos = data.qpos.copy()
    targets = data.qpos[model.jnt_qposadr[joint_ids]].copy()
    targets[0] += 0.2
    targets[7] += 0.2
    data.ctrl[actuator_ids] = targets
    for _ in range(500):
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
    print(f"Internal position-servo kp range: {kp.min():.1f}..{kp.max():.1f}")
    print(f"Internal position-servo kv range: {kv.min():.1f}..{kv.max():.1f}")
    print(f"500-step position-target displacement: {displacement:.6f} rad")
    print("OpenArm v2 MuJoCo setup verified.")


if __name__ == "__main__":
    main()
