# Combined webcam-to-OpenArm retargeting demo

This demo combines the fast RTMW3D-L TensorRT webcam pipeline with the bimanual
OpenArm MuJoCo scene. The browser supplies the webcam, so the camera always belongs
to the device that opened the page. An SSH tunnel changes only the network route.

The runtime uses three independent OS processes:

1. FastAPI receives browser JPEGs and streams both rendered views.
2. RTMW3D decodes SimCC landmarks, constructs body-size-invariant elbow/wrist
   targets and calibrated hand orientations, and logs `RobotTarget` records.
3. MuJoCo runs damped-least-squares IK, writes joint targets to `data.ctrl`, steps
   physics at the model's 1 kHz timestep, and renders independently.

Every cross-process queue has capacity one. Old camera frames, targets, and renders
are dropped instead of adding latency.

## WSL setup

First set up the existing OpenArm assets:

```bash
cd /home/dev/popvax-assignment/openarm-mujoco
./setup.sh
```

Then install the combined Python 3.11 environment with `uv`:

```bash
cd /home/dev/popvax-assignment/retargeting-demo
uv sync --extra gpu --extra tensorrt --extra dev
```

The default TensorRT engine is:

```text
/home/dev/.cache/rtmlib/hub/checkpoints/rtmw3d-l-fp32.plan
```

Override it with `RTMW3D_TRT_ENGINE` if needed.

## Launch

```bash
cd /home/dev/popvax-assignment/retargeting-demo
MUJOCO_GL=egl uv run python -m retargeting_demo.main --port 8000
```

Open `http://localhost:8000` on the PC. For a Mac webcam, tunnel the same server:

```bash
ssh -N -L 8000:127.0.0.1:8000 windows-cuda-wsl
```

Then open `http://127.0.0.1:8000` on the Mac. Browsers treat localhost as a secure
camera context, and the browser asks for that Mac's camera permission.

## Operator procedure

1. Select **Start camera** and allow camera access.
2. Face the camera with shoulders, elbows, wrists, and both hands visible.
3. Hold a comfortable neutral pose with open hands.
4. Select **Calibrate & engage** and remain still until the status becomes
   `Calibrated · engaged`.
5. Move within the robot's reach. If hand orientation drifts or the operator
   changes, recalibrate.

The simulation holds its last actuator command if targets become stale. Closing the
browser or selecting **Stop** disengages new target application.

## Coordinate and IK contract

`RobotTarget` is expressed in metres and rotation matrices in the model's
`arm_origin` frame:

- RTMW3D decoded SimCC coordinates supply relative shoulder-elbow-wrist directions.
- Human segment magnitude is discarded; directions are scaled to the OpenArm's
  measured 0.220 m upper arm and 0.216 m forearm.
- Wrist orientation comes from the wrist and four MCP landmarks. Calibration maps
  the neutral human hand frames to the OpenArm home tool frames.
- IK minimizes elbow position, end-effector position, and end-effector orientation
  with analytic MuJoCo Jacobians and joint-limit clipping.

Intermediate targets are logged to `logs/targets-*.jsonl`; achieved elbow/wrist
poses and Cartesian errors are logged to `logs/achieved-*.jsonl`. `logs/` is ignored
by Git.

## Validation

```bash
uv run pytest
```

Useful environment overrides include `CONTROL_HZ`, `IK_ITERATIONS`, `IK_DAMPING`,
`IK_ELBOW_WEIGHT`, `IK_ORIENTATION_WEIGHT`, `RETARGET_CONFIDENCE`,
`RTMW3D_DET_FREQUENCY`, and `OPENARM_MODEL_PATH`.
