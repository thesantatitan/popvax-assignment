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

import mink
import mujoco
import numpy as np
import qpsolvers
from PIL import Image

from .contracts import RenderedFrame, RobotTarget
from .ipc import configure_parent_death_signal, drain_latest, put_latest
from .retarget import exponential_smoothing_alpha

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = (
    ROOT
    / "vendor"
    / "openarm-v2"
    / "cell.xml"
)
SIDES = ("left", "right")


class ArmConfigurationLimit(mink.limits.Limit):
    """Hard position bounds projected onto only the OpenArm arm joints."""

    def __init__(
        self,
        model: mujoco.MjModel,
        joint_ids: np.ndarray,
        *,
        gain: float,
        margin_rad: float,
    ) -> None:
        if not 0.0 < gain <= 1.0:
            raise ValueError("IK_LIMIT_GAIN must be in (0, 1]")
        self.gain = gain
        self.dof_addresses = model.jnt_dofadr[joint_ids].astype(np.int32)
        self.qpos_addresses = model.jnt_qposadr[joint_ids].astype(np.int32)
        self.lower = model.jnt_range[joint_ids, 0] + margin_rad
        self.upper = model.jnt_range[joint_ids, 1] - margin_rad
        if np.any(self.lower >= self.upper):
            raise ValueError("IK_LIMIT_MARGIN leaves an empty arm-joint range")
        self.projection = np.eye(model.nv)[self.dof_addresses]

    def compute_qp_inequalities(
        self,
        configuration: mink.Configuration,
        dt: float,
    ) -> mink.limits.Constraint:
        del dt
        q = configuration.q[self.qpos_addresses]
        maximum_step = self.gain * (self.upper - q)
        minimum_step = self.gain * (q - self.lower)
        return mink.limits.Constraint(
            G=np.vstack([self.projection, -self.projection]),
            h=np.hstack([maximum_step, minimum_step]),
        )


def _object_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    result = mujoco.mj_name2id(model, kind, name)
    if result < 0:
        raise RuntimeError(f"OpenArm model is missing {name!r}")
    return result


class BimanualIk:
    """Continuity-regularized bimanual differential IK using Mink."""

    def __init__(self, model: mujoco.MjModel, initial_data: mujoco.MjData) -> None:
        self.model = model
        self.configuration = mink.Configuration(model, q=initial_data.qpos.copy())
        self.data = self.configuration.data
        self.origin_site = _object_id(
            model, mujoco.mjtObj.mjOBJ_SITE, "arm_origin"
        )
        self.actuator_ids: dict[str, np.ndarray] = {}
        self.qpos_addresses: dict[str, np.ndarray] = {}
        self.dof_addresses: dict[str, np.ndarray] = {}
        self.elbow_bodies: dict[str, int] = {}
        self.wrist_sites: dict[str, int] = {}
        self.elbow_tasks: dict[str, mink.FrameTask] = {}
        self.wrist_position_tasks: dict[str, mink.FrameTask] = {}
        self.wrist_tasks: dict[str, mink.FrameTask] = {}
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
            self.elbow_tasks[side] = mink.FrameTask(
                frame_name=f"openarm_{side}_link4",
                frame_type="body",
                position_cost=float(os.getenv("IK_ELBOW_COST", "1.0")),
                orientation_cost=0.0,
                gain=float(os.getenv("IK_TASK_GAIN", "0.7")),
                lm_damping=float(os.getenv("IK_LM_DAMPING", "1e-4")),
            )
            self.wrist_tasks[side] = mink.FrameTask(
                frame_name=f"{side}_ee_control_point",
                frame_type="site",
                position_cost=float(os.getenv("IK_WRIST_COST", "1.0")),
                orientation_cost=float(
                    os.getenv("IK_WRIST_ORIENTATION_COST", "0.25")
                ),
                gain=float(os.getenv("IK_TASK_GAIN", "0.7")),
                lm_damping=float(os.getenv("IK_LM_DAMPING", "1e-4")),
            )
            self.wrist_position_tasks[side] = mink.FrameTask(
                frame_name=f"{side}_ee_control_point",
                frame_type="site",
                position_cost=float(os.getenv("IK_WRIST_COST", "1.0")),
                orientation_cost=0.0,
                gain=float(os.getenv("IK_TASK_GAIN", "0.7")),
                lm_damping=float(os.getenv("IK_LM_DAMPING", "1e-4")),
            )
        self.all_actuator_ids = np.concatenate(
            [self.actuator_ids[side] for side in SIDES]
        )
        self.all_qpos_addresses = np.concatenate(
            [self.qpos_addresses[side] for side in SIDES]
        )
        self.all_dof_addresses = np.concatenate(
            [self.dof_addresses[side] for side in SIDES]
        )
        arm_joint_ids = model.actuator_trnid[self.all_actuator_ids, 0]
        self.previous_solution = initial_data.qpos[self.all_qpos_addresses].copy()
        self.posture_task = mink.PostureTask(
            model, cost=float(os.getenv("IK_POSTURE_COST", "0.05"))
        )
        velocity_limits: dict[str, float] = {}
        arm_velocity = float(os.getenv("IK_MAX_VELOCITY_RAD_S", "3.0"))
        for joint_id in arm_joint_ids:
            name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id)
            )
            if name is not None:
                velocity_limits[name] = arm_velocity
        self.limits = [
            ArmConfigurationLimit(
                model,
                arm_joint_ids,
                gain=float(os.getenv("IK_LIMIT_GAIN", "0.95")),
                margin_rad=float(os.getenv("IK_LIMIT_MARGIN", "0.0")),
            ),
            mink.VelocityLimit(model, velocity_limits),
        ]
        self.solver = os.getenv("IK_QP_SOLVER", "daqp")
        self.last_diagnostics: dict[str, object] | None = None

    def _solve_arm_velocity(
        self,
        tasks: list[mink.tasks.BaseTask],
        integration_dt: float,
        damping: float,
    ) -> np.ndarray:
        """Build with Mink, then solve a QP containing only the 14 arm DoFs."""

        full = mink.build_ik(
            self.configuration,
            tasks,
            dt=integration_dt,
            damping=damping,
            limits=self.limits,
        )
        indices = self.all_dof_addresses
        reduced = qpsolvers.Problem(
            P=full.P[indices][:, indices],
            q=full.q[indices],
            G=full.G[:, indices] if full.G is not None else None,
            h=full.h,
            A=full.A[:, indices] if full.A is not None else None,
            b=full.b,
            lb=full.lb[indices] if full.lb is not None else None,
            ub=full.ub[indices] if full.ub is not None else None,
        )
        result = qpsolvers.solve_problem(reduced, solver=self.solver)
        if not result.found or result.x is None:
            raise mink.exceptions.NoSolutionFound(self.solver)
        velocity = np.zeros(self.model.nv)
        velocity[indices] = result.x / integration_dt
        return velocity

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
        previous_solution = self.previous_solution.copy()
        configuration_q = np.asarray(seed_qpos).copy()
        # Preserve the current non-arm state, but warm-start both arms from the
        # previous continuous Mink solution rather than lagging physical qpos.
        configuration_q[self.all_qpos_addresses] = previous_solution
        self.configuration.update(configuration_q)
        self.posture_task.set_target_from_configuration(self.configuration)
        damping = float(os.getenv("IK_DAMPING", "1e-6"))
        iterations = int(os.getenv("IK_ITERATIONS", "25"))
        integration_dt = float(os.getenv("IK_INTEGRATION_DT_S", "0.02"))
        tasks: list[mink.tasks.BaseTask] = [self.posture_task]
        for side in SIDES:
            arm_target = getattr(target, side)
            if target.mode in {"elbow", "both"}:
                elbow_target, _ = self._base_to_world(
                    np.asarray(arm_target.elbow_position_m), np.eye(3)
                )
                self.elbow_tasks[side].set_target(
                    mink.SE3.from_translation(elbow_target)
                )
                tasks.append(self.elbow_tasks[side])
            if target.mode in {"end_effector", "both"}:
                if arm_target.wrist_position_m is None:
                    raise ValueError(
                        f"{target.mode} target is missing wrist_position_m"
                    )
                if arm_target.wrist_rotation is not None:
                    wrist_target, wrist_rotation = self._base_to_world(
                        np.asarray(arm_target.wrist_position_m),
                        np.asarray(arm_target.wrist_rotation).reshape(3, 3),
                    )
                    self.wrist_tasks[side].set_target(
                        mink.SE3.from_rotation_and_translation(
                            mink.SO3.from_matrix(wrist_rotation),
                            wrist_target,
                        )
                    )
                    tasks.append(self.wrist_tasks[side])
                else:
                    wrist_target, _ = self._base_to_world(
                        np.asarray(arm_target.wrist_position_m), np.eye(3)
                    )
                    self.wrist_position_tasks[side].set_target(
                        mink.SE3.from_translation(wrist_target)
                    )
                    tasks.append(self.wrist_position_tasks[side])

        solver_error: str | None = None
        completed_iterations = 0
        for _ in range(iterations):
            try:
                velocity = self._solve_arm_velocity(
                    tasks,
                    integration_dt,
                    damping,
                )
            except Exception as exc:  # noqa: BLE001
                solver_error = str(exc)
                break
            self.configuration.integrate_inplace(velocity, integration_dt)
            completed_iterations += 1

        solution = self.data.qpos[self.all_qpos_addresses].copy()
        residuals: dict[str, dict[str, float]] = {}
        maximum_residual = 0.0
        maximum_orientation_residual = 0.0
        for side in SIDES:
            arm_target = getattr(target, side)
            side_residuals: dict[str, float] = {}
            if target.mode in {"elbow", "both"}:
                elbow_target, _ = self._base_to_world(
                    np.asarray(arm_target.elbow_position_m), np.eye(3)
                )
                side_residuals["elbow_position_m"] = float(
                    np.linalg.norm(
                        elbow_target - self.data.xpos[self.elbow_bodies[side]]
                    )
                )
            if target.mode in {"end_effector", "both"}:
                if arm_target.wrist_position_m is None:
                    raise ValueError(
                        f"{target.mode} target is missing wrist_position_m"
                    )
                wrist_target, _ = self._base_to_world(
                    np.asarray(arm_target.wrist_position_m), np.eye(3)
                )
                side_residuals["wrist_position_m"] = float(
                    np.linalg.norm(
                        wrist_target - self.data.site_xpos[self.wrist_sites[side]]
                    )
                )
                if arm_target.wrist_rotation is not None:
                    wrist_rotation_target = np.asarray(
                        arm_target.wrist_rotation
                    ).reshape(3, 3)
                    _, wrist_rotation_target = self._base_to_world(
                        np.asarray(arm_target.wrist_position_m),
                        wrist_rotation_target,
                    )
                    wrist_rotation_actual = self.data.site_xmat[
                        self.wrist_sites[side]
                    ].reshape(3, 3)
                    relative_rotation = (
                        wrist_rotation_target.T @ wrist_rotation_actual
                    )
                    cosine = np.clip(
                        (np.trace(relative_rotation) - 1.0) * 0.5,
                        -1.0,
                        1.0,
                    )
                    orientation_residual = float(np.arccos(cosine))
                    side_residuals["wrist_orientation_rad"] = (
                        orientation_residual
                    )
                    maximum_orientation_residual = max(
                        maximum_orientation_residual,
                        orientation_residual,
                    )
            residuals[side] = side_residuals
            position_residuals = [
                value
                for key, value in side_residuals.items()
                if key.endswith("_position_m")
            ]
            maximum_residual = max(maximum_residual, *position_residuals)
        tolerance = float(os.getenv("IK_POSITION_TOLERANCE_M", "0.01"))
        orientation_tolerance = float(
            os.getenv("IK_ORIENTATION_TOLERANCE_RAD", "0.15")
        )
        self.last_diagnostics = {
            "target_sequence": target.sequence,
            "status": (
                "solver_error"
                if solver_error is not None
                else (
                    "converged"
                    if maximum_residual <= tolerance
                    and maximum_orientation_residual <= orientation_tolerance
                    else "residual_high"
                )
            ),
            "solver": f"mink:{self.solver}",
            "decision_dofs": len(self.all_dof_addresses),
            "excluded_dofs": int(
                self.model.nv - len(self.all_dof_addresses)
            ),
            "iterations": completed_iterations,
            "requested_iterations": iterations,
            "integration_dt_s": integration_dt,
            "damping": damping,
            "position_tolerance_m": tolerance,
            "orientation_tolerance_rad": orientation_tolerance,
            "maximum_residual_m": maximum_residual,
            "maximum_orientation_residual_rad": maximum_orientation_residual,
            "residuals": residuals,
            "solver_error": solver_error,
            "solution_delta_from_previous_rad": (
                solution - previous_solution
            ).tolist(),
            "solution_delta_norm_rad": float(
                np.linalg.norm(solution - previous_solution)
            ),
        }
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
            wrist_rotation = origin_rotation.T @ data.site_xmat[
                self.wrist_sites[side]
            ].reshape(3, 3)
            arm: dict[str, object] = {
                "elbow_position_m": elbow.tolist(),
                "wrist_position_m": wrist.tolist(),
                "wrist_rotation": wrist_rotation.reshape(-1).tolist(),
            }
            if target is not None:
                desired = getattr(target, side)
                arm["elbow_error_m"] = float(
                    np.linalg.norm(elbow - np.asarray(desired.elbow_position_m))
                )
                if desired.wrist_position_m is not None:
                    arm["wrist_error_m"] = float(
                        np.linalg.norm(
                            wrist - np.asarray(desired.wrist_position_m)
                        )
                    )
                if desired.wrist_rotation is not None:
                    desired_rotation = np.asarray(
                        desired.wrist_rotation
                    ).reshape(3, 3)
                    relative_rotation = desired_rotation.T @ wrist_rotation
                    cosine = np.clip(
                        (np.trace(relative_rotation) - 1.0) * 0.5,
                        -1.0,
                        1.0,
                    )
                    arm["wrist_orientation_error_rad"] = float(
                        np.arccos(cosine)
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


def _lifter_top_command(model: mujoco.MjModel) -> tuple[int, float]:
    actuator_id = _object_id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lifter_ctrl"
    )
    if not model.actuator_ctrllimited[actuator_id]:
        raise RuntimeError("OpenArm lifter_ctrl must have a bounded control range")
    return actuator_id, float(model.actuator_ctrlrange[actuator_id, 1])


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
        raise FileNotFoundError(f"OpenArm model not found at {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    data = mujoco.MjData(model)
    _initialize_model(model, data)
    lifter_actuator, lifter_top = _lifter_top_command(model)
    lifter_joint = model.actuator_trnid[lifter_actuator, 0]
    lifter_qpos_address = model.jnt_qposadr[lifter_joint]
    data.ctrl[lifter_actuator] = lifter_top
    ik = BimanualIk(model, data)
    arm_actuators = ik.all_actuator_ids
    joint_command = data.ctrl[arm_actuators].copy()
    ik_solution = joint_command.copy()
    active_target: RobotTarget | None = None
    target_dirty = False
    last_target_wall = 0.0
    latest_sequence = 0

    log_path = Path(log_directory)
    log_path.mkdir(parents=True, exist_ok=True)
    achieved_log = log_path / f"achieved-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    render_period = 1.0 / render_fps
    control_period = 1.0 / float(os.getenv("CONTROL_HZ", "60"))
    joint_smoothing_tau_s = float(
        os.getenv("ROBOT_COMMAND_SMOOTHING_TAU_S", "0")
    )
    joint_smoothing_alpha = exponential_smoothing_alpha(
        control_period, joint_smoothing_tau_s
    )
    joint_max_speed_rad_s = float(
        os.getenv("ROBOT_COMMAND_MAX_SPEED_RAD_S", "3.0")
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

                if target_dirty and active_target is not None:
                    ik_solution = ik.solve(active_target, data.qpos)
                    target_dirty = False

                if now >= next_control:
                    joint_command += np.clip(
                        joint_smoothing_alpha * (ik_solution - joint_command),
                        -joint_max_speed_rad_s * control_period,
                        joint_max_speed_rad_s * control_period,
                    )
                    next_control += control_period
                    if next_control <= now:
                        next_control = now + control_period

                target_simulation_time = now - wall_start
                while data.time < target_simulation_time:
                    # OpenArm v2 uses position servos. This is the only command
                    # path: filtered Cartesian target -> IK -> filtered joints
                    # -> data.ctrl.
                    data.ctrl[arm_actuators] = joint_command
                    data.ctrl[lifter_actuator] = lifter_top
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
                        "joint_smoothing_tau_s": joint_smoothing_tau_s,
                        "joint_smoothing_alpha": joint_smoothing_alpha,
                        "render_fps": render_fps,
                        "mode": (
                            active_target.mode
                            if active_target is not None
                            else None
                        ),
                        "lifter_target_m": lifter_top,
                        "lifter_position_m": round(
                            float(data.qpos[lifter_qpos_address]), 4
                        ),
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
                        joint_state = {
                            "order": {
                                side: [
                                    f"{side}_joint{index}"
                                    for index in range(1, 8)
                                ]
                                for side in SIDES
                            },
                            "ik_solution_rad": {
                                side: ik_solution[
                                    index * 7 : (index + 1) * 7
                                ].tolist()
                                for index, side in enumerate(SIDES)
                            },
                            "command_rad": {
                                side: joint_command[
                                    index * 7 : (index + 1) * 7
                                ].tolist()
                                for index, side in enumerate(SIDES)
                            },
                            "data_ctrl_rad": {
                                side: data.ctrl[ik.actuator_ids[side]].tolist()
                                for side in SIDES
                            },
                            "qpos_rad": {
                                side: data.qpos[
                                    ik.qpos_addresses[side]
                                ].tolist()
                                for side in SIDES
                            },
                            "qvel_rad_s": {
                                side: data.qvel[
                                    ik.dof_addresses[side]
                                ].tolist()
                                for side in SIDES
                            },
                        }
                        output.write(
                            json.dumps(
                                {
                                    "time_ns": time.monotonic_ns(),
                                    "simulation_time_s": float(data.time),
                                    "target_sequence": latest_sequence,
                                    "achieved": achieved,
                                    "joints": joint_state,
                                    "ik": ik.last_diagnostics,
                                    "joint_filter": {
                                        "time_constant_s": joint_smoothing_tau_s,
                                        "alpha": joint_smoothing_alpha,
                                        "maximum_speed_rad_s": (
                                            joint_max_speed_rad_s
                                        ),
                                    },
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
