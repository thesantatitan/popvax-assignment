"""Independent MuJoCo physics, IK, control, and rendering process."""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform.startswith("linux") else "glfw")
if Path("/dev/dxg").exists():
    os.environ.setdefault("GALLIUM_DRIVER", "d3d12")

import mujoco
import numpy as np
from PIL import Image

from .contracts import RenderedFrame, RobotTarget
from .ipc import configure_parent_death_signal, drain_latest, put_latest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = (
    ROOT.parent
    / "openarm-mujoco"
    / ".assets"
    / "openarm_mujoco"
    / "v2"
    / "cell.xml"
)
SIDES = ("left", "right")


def _object_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    result = mujoco.mj_name2id(model, kind, name)
    if result < 0:
        raise RuntimeError(f"OpenArm model is missing {name!r}")
    return result


def exponential_smoothing_alpha(period_s: float, time_constant_s: float) -> float:
    """Return a rate-independent exponential smoothing coefficient."""

    if period_s <= 0.0:
        raise ValueError("period_s must be positive")
    if time_constant_s < 0.0:
        raise ValueError("time_constant_s must be non-negative")
    if time_constant_s == 0.0:
        return 1.0
    return float(-np.expm1(-period_s / time_constant_s))


class BimanualIk:
    """Small damped-least-squares solver using MuJoCo analytic Jacobians."""

    def __init__(self, model: mujoco.MjModel, initial_data: mujoco.MjData) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = initial_data.qpos
        self.origin_site = _object_id(
            model, mujoco.mjtObj.mjOBJ_SITE, "arm_origin"
        )
        self.actuator_ids: dict[str, np.ndarray] = {}
        self.qpos_addresses: dict[str, np.ndarray] = {}
        self.dof_addresses: dict[str, np.ndarray] = {}
        self.elbow_bodies: dict[str, int] = {}
        self.wrist_sites: dict[str, int] = {}
        for side in SIDES:
            actuators = np.array(
                [
                    _object_id(
                        model,
                        mujoco.mjtObj.mjOBJ_ACTUATOR,
                        f"{side}_joint{index}_ctrl",
                    )
                    for index in range(1, 8)
                ],
                dtype=np.int32,
            )
            joints = model.actuator_trnid[actuators, 0]
            self.actuator_ids[side] = actuators
            self.qpos_addresses[side] = model.jnt_qposadr[joints]
            self.dof_addresses[side] = model.jnt_dofadr[joints]
            self.elbow_bodies[side] = _object_id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{side}_link4"
            )
            self.wrist_sites[side] = _object_id(
                model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_ee_control_point"
            )
        self.all_actuator_ids = np.concatenate(
            [self.actuator_ids[side] for side in SIDES]
        )
        self.all_qpos_addresses = np.concatenate(
            [self.qpos_addresses[side] for side in SIDES]
        )
        self.previous_solution = initial_data.qpos[self.all_qpos_addresses].copy()

    def _base_to_world(
        self, position: np.ndarray, rotation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        origin_position = self.data.site_xpos[self.origin_site]
        origin_rotation = self.data.site_xmat[self.origin_site].reshape(3, 3)
        return (
            origin_position + origin_rotation @ position,
            origin_rotation @ rotation,
        )

    def solve(self, target: RobotTarget, seed_qpos: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = seed_qpos
        # Warm-start from the previous kinematic solution rather than the
        # lagging, contact-affected physical state. Consecutive camera targets
        # are close, so this avoids branch flips and local minima.
        self.data.qpos[self.all_qpos_addresses] = self.previous_solution
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        damping = float(os.getenv("IK_DAMPING", "0.035"))
        iterations = int(os.getenv("IK_ITERATIONS", "25"))

        for _ in range(iterations):
            for side in SIDES:
                arm_target = getattr(target, side)
                wrist_target, _ = self._base_to_world(
                    np.asarray(arm_target.wrist_position_m), np.eye(3)
                )
                wrist_site = self.wrist_sites[side]
                dofs = self.dof_addresses[side]

                jac_wrist = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(
                    self.model,
                    self.data,
                    jac_wrist,
                    None,
                    wrist_site,
                )
                error = wrist_target - self.data.site_xpos[wrist_site]
                jacobian = jac_wrist[:, dofs]
                system = jacobian @ jacobian.T
                system.flat[:: system.shape[0] + 1] += damping**2
                delta = jacobian.T @ np.linalg.solve(system, error)
                delta = np.clip(delta, -0.10, 0.10)
                qpos_addresses = self.qpos_addresses[side]
                self.data.qpos[qpos_addresses] += delta
                joint_ids = self.model.actuator_trnid[self.actuator_ids[side], 0]
                limits = self.model.jnt_range[joint_ids]
                self.data.qpos[qpos_addresses] = np.clip(
                    self.data.qpos[qpos_addresses],
                    limits[:, 0] + 0.01,
                    limits[:, 1] - 0.01,
                )
                mujoco.mj_forward(self.model, self.data)

        solution = self.data.qpos[self.all_qpos_addresses].copy()
        self.previous_solution = solution
        return solution

    def achieved_state(
        self, data: mujoco.MjData, target: RobotTarget | None
    ) -> dict[str, object]:
        origin_position = data.site_xpos[self.origin_site]
        origin_rotation = data.site_xmat[self.origin_site].reshape(3, 3)
        result: dict[str, object] = {}
        for side in SIDES:
            elbow = origin_rotation.T @ (
                data.xpos[self.elbow_bodies[side]] - origin_position
            )
            wrist = origin_rotation.T @ (
                data.site_xpos[self.wrist_sites[side]] - origin_position
            )
            arm: dict[str, object] = {
                "elbow_position_m": elbow.tolist(),
                "wrist_position_m": wrist.tolist(),
            }
            if target is not None:
                desired = getattr(target, side)
                arm["elbow_error_m"] = float(
                    np.linalg.norm(elbow - np.asarray(desired.elbow_position_m))
                )
                arm["wrist_error_m"] = float(
                    np.linalg.norm(wrist - np.asarray(desired.wrist_position_m))
                )
            result[side] = arm
        return result


def _initialize_model(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    home = _object_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home)
    for actuator_id in range(model.nu):
        joint_id = model.actuator_trnid[actuator_id, 0]
        if joint_id >= 0:
            data.ctrl[actuator_id] = data.qpos[model.jnt_qposadr[joint_id]]
    mujoco.mj_forward(model, data)


def simulation_worker(
    target_queue,
    sim_frame_queue,
    telemetry_queue,
    camera_queue,
    engaged_event,
    stop_event,
    log_directory: str,
    width: int,
    height: int,
    render_fps: float,
) -> None:
    """Process entrypoint. MuJoCo steps independently from web and inference."""

    parent_pid = configure_parent_death_signal()
    model_path = Path(os.getenv("OPENARM_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if not model_path.is_file():
        raise FileNotFoundError(
            f"OpenArm model not found at {model_path}. Run ../openarm-mujoco/setup.sh."
        )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    data = mujoco.MjData(model)
    _initialize_model(model, data)
    ik = BimanualIk(model, data)
    arm_actuators = ik.all_actuator_ids
    desired = data.ctrl[arm_actuators].copy()
    ik_solution = desired.copy()
    active_target: RobotTarget | None = None
    target_dirty = False
    last_target_wall = 0.0
    latest_sequence = 0

    log_path = Path(log_directory)
    log_path.mkdir(parents=True, exist_ok=True)
    achieved_log = log_path / f"achieved-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    render_period = 1.0 / render_fps
    control_period = 1.0 / float(os.getenv("CONTROL_HZ", "60"))
    smoothing_time_constant = float(
        os.getenv("ROBOT_COMMAND_SMOOTHING_TAU_S", "0.12")
    )
    smoothing_alpha = exponential_smoothing_alpha(
        control_period, smoothing_time_constant
    )
    next_render = time.monotonic()
    next_control = next_render
    wall_start = next_render
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    frame_number = 0

    with achieved_log.open(  # noqa: SIM117
        "a", encoding="utf-8", buffering=1
    ) as output:
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            while not stop_event.is_set() and os.getppid() == parent_pid:
                now = time.monotonic()
                incoming = drain_latest(target_queue)
                if incoming is not None:
                    active_target = incoming
                    target_dirty = True
                    latest_sequence = incoming.sequence
                    last_target_wall = now
                if not engaged_event.is_set():
                    active_target = None
                    target_dirty = False

                if now >= next_control:
                    max_delta = 3.0 * control_period
                    desired += np.clip(
                        smoothing_alpha * (ik_solution - desired),
                        -max_delta,
                        max_delta,
                    )
                    next_control += control_period
                    if next_control <= now:
                        next_control = now + control_period

                if target_dirty and active_target is not None:
                    ik_solution = ik.solve(active_target, data.qpos)
                    target_dirty = False

                target_simulation_time = now - wall_start
                while data.time < target_simulation_time:
                    # OpenArm v2 uses position servos. This is the only command
                    # path: desired joints -> data.ctrl -> physics step.
                    data.ctrl[arm_actuators] = desired
                    mujoco.mj_step(model, data)

                commands = drain_latest(camera_queue)
                if commands is not None:
                    action, dx, dy = commands
                    if action == "reset":
                        mujoco.mjv_defaultFreeCamera(model, camera)
                    else:
                        mouse_action = {
                            "rotate": mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                            "pan": mujoco.mjtMouse.mjMOUSE_MOVE_V,
                            "zoom": mujoco.mjtMouse.mjMOUSE_ZOOM,
                        }.get(action)
                        if mouse_action is not None:
                            mujoco.mjv_moveCamera(
                                model,
                                mouse_action,
                                float(dx),
                                float(dy),
                                renderer.scene,
                                camera,
                            )

                if now >= next_render:
                    renderer.update_scene(data, camera=camera)
                    rgb = renderer.render()
                    encoded = io.BytesIO()
                    Image.fromarray(rgb).save(encoded, format="JPEG", quality=84)
                    frame_number += 1
                    put_latest(
                        sim_frame_queue,
                        RenderedFrame(sequence=frame_number, jpeg=encoded.getvalue()),
                    )
                    achieved = ik.achieved_state(data, active_target)
                    target_age_ms = (
                        (now - last_target_wall) * 1000.0
                        if active_target is not None
                        else None
                    )
                    state = {
                        "type": "simulation",
                        "simulation_time_s": round(float(data.time), 3),
                        "control_hz": round(1.0 / control_period, 1),
                        "render_fps": render_fps,
                        "target_sequence": latest_sequence,
                        "target_age_ms": (
                            round(target_age_ms, 1)
                            if target_age_ms is not None
                            else None
                        ),
                        "stale": target_age_ms is None or target_age_ms > 500.0,
                        "achieved": achieved,
                    }
                    put_latest(telemetry_queue, state)
                    if active_target is not None:
                        output.write(
                            json.dumps(
                                {
                                    "time_ns": time.monotonic_ns(),
                                    "simulation_time_s": float(data.time),
                                    "target_sequence": latest_sequence,
                                    "achieved": achieved,
                                },
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                    next_render += render_period
                    if next_render <= now:
                        next_render = now + render_period
                else:
                    time.sleep(min(0.001, next_render - now))
