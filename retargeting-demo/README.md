# Combined webcam-to-OpenArm retargeting demo

This demo combines the fast RTMW3D-L TensorRT webcam pipeline with the bimanual
OpenArm MuJoCo scene. The browser supplies the webcam, so the camera always belongs
to the device that opened the page. An SSH tunnel changes only the network route.

The runtime uses three independent OS processes:

1. FastAPI receives browser JPEGs and streams both rendered views.
2. RTMW3D decodes SimCC landmarks, constructs body-size-invariant elbow/wrist
   targets, renders both 2D and inset 3D pose views, and logs `RobotTarget` records.
3. MuJoCo runs damped-least-squares IK, writes joint targets to `data.ctrl`, steps
   physics at the model's 1 kHz timestep, and renders independently. Joint commands
   are exponentially smoothed before they reach `data.ctrl`. The model's actuated
   vertical lifter is continuously commanded to the top of its range (`0.3 m`).

Every cross-process queue has capacity one. Old camera frames, targets, and renders
are dropped instead of adding latency.

## WSL setup

First set up the existing OpenArm assets:

```bash
cd /home/dev/popvax-assignment/openarm-mujoco
./setup.sh
```

Then install and verify the combined Python 3.11 environment with `uv`:

```bash
cd /home/dev/popvax-assignment/retargeting-demo
chmod +x setup.sh
./setup.sh
```

The default TensorRT engine is:

```text
/home/dev/.cache/rtmlib/hub/checkpoints/rtmw3d-l-fp32.plan
```

Override it with `RTMW3D_TRT_ENGINE` if needed.

## Launch

```bash
cd /home/dev/popvax-assignment/retargeting-demo
./run_wsl_server.sh
```

Override the listener with `RETARGETING_HOST` or `RETARGETING_PORT`. Additional
arguments such as `--sim-width 1280 --sim-height 720` are forwarded to the module.

Open `http://localhost:8000` on the PC. For a Mac webcam, tunnel the same server:

```bash
ssh -N -L 8000:127.0.0.1:8000 windows-cuda-wsl
```

Then open `http://127.0.0.1:8000` on the Mac. Browsers treat localhost as a secure
camera context, and the browser asks for that Mac's camera permission.

## Operator procedure

1. Select **Start camera** and allow camera access.
2. Face the camera with shoulders, elbows, wrists, and both hands visible.
3. Keep every required keypoint confidently visible for two continuous seconds.
4. Tracking engages automatically. If confidence drops, the robot holds its last
   pose and requires another continuous two-second confident interval.

The simulation holds its last actuator command if targets become stale. Closing the
browser or selecting **Stop** disengages new target application.

## Coordinate and IK contract

`RobotTarget` is expressed in metres and rotation matrices in the model's
`arm_origin` frame:

- RTMW3D decoded SimCC coordinates supply relative shoulder-elbow-wrist directions.
- Human segment magnitude is discarded; directions are scaled to the OpenArm's
  measured 0.220 m upper arm and 0.216 m forearm.
- Limb directions map directly from camera coordinates into the robot base frame;
  there is no neutral-pose calibration or relative-motion offset.
- Wrist orientation and elbow position remain available in `RobotTarget` and the
  logs, but the IK objective uses only end-effector position.
- IK uses the analytic MuJoCo end-effector position Jacobian and joint-limit
  clipping. Its joint solution is exponentially smoothed at the control rate.

Intermediate targets, the selected person's decoded `keypoints_simcc`, and that
person's detection index are logged to `logs/targets-*.jsonl`; achieved elbow/wrist
poses and Cartesian errors are logged to `logs/achieved-*.jsonl`. `logs/` is
ignored by Git.

## Validation

```bash
uv run pytest
uv run python verify_runtime.py
```

Useful environment overrides include `CONTROL_HZ`, `IK_ITERATIONS`, `IK_DAMPING`,
`ROBOT_COMMAND_SMOOTHING_TAU_S`, `RETARGET_CONFIDENCE`,
`RETARGET_CONFIDENCE_SECONDS`, `RETARGET_SMOOTHING_ALPHA`,
`RTMW3D_DET_FREQUENCY`, and `OPENARM_MODEL_PATH`. Set the command smoothing time
constant to `0` to disable it; larger values make motion smoother and slower.
