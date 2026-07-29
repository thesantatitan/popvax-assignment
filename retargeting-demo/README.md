# Combined webcam-to-OpenArm retargeting demo

This demo combines the fast RTMW3D-L TensorRT webcam pipeline with the bimanual
OpenArm MuJoCo scene. The browser supplies the webcam, so the camera always belongs
to the device that opened the page. An SSH tunnel changes only the network route.

The runtime uses three independent OS processes:

1. FastAPI receives browser JPEGs and streams both rendered views.
2. RTMW3D decodes SimCC landmarks, constructs body-size-invariant elbow/wrist
   targets, exponentially filters the normalized Cartesian limb directions,
   renders both 2D and inset 3D pose views, and logs `RobotTarget` records.
3. MuJoCo and Mink run continuity-regularized differential QP IK with hard joint
   position and velocity limits on only the 14 arm joints, write filtered joint
   targets to `data.ctrl`, step physics at the model's 1 kHz timestep, and render
   independently. The lifter and grippers are fixed kinematic inputs, not IK
   variables. The model's actuated vertical lifter is continuously commanded to
   the top of its range (`0.3 m`).

Every cross-process queue has capacity one. Old camera frames, targets, and renders
are dropped instead of adding latency.

## WSL setup

Install and verify the combined Python 3.11 environment with `uv`:

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

The original OpenArm v2 model and all required meshes are tracked under
`vendor/openarm-v2`. Setup does not clone or download OpenArm. The vendored source
is based on upstream commit `8955afb54e4adfb59a236e2b4d15192b7a02865c` and
retains its Apache-2.0 license. The monolithic enclosure visual and its roof,
side-wall, and front-wall collision boxes are removed; the table, rails, and
actuated vertical lifter are unchanged.

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
2. Select **Elbow only**, **End effector only**, or **Elbow + end effector**.
3. Face the camera with both shoulders and elbows visible; wrist visibility is
   additionally required for modes that track the end effector.
4. Keep every required keypoint confidently visible for two continuous seconds.
5. Tracking engages automatically. If confidence drops, the robot holds its last
   pose and requires another continuous two-second confident interval.

The simulation holds its last actuator command if targets become stale. Closing the
browser or selecting **Stop** disengages new target application.

## Camera intrinsics and calibration

Enable **Use camera intrinsics** to reconstruct camera-space points before deriving
limb directions. The checked path:

1. Uses RTMW3D's original-image `keypoints_2d` as `(u, v)`.
2. Undistorts them with the selected camera's matrix and distortion coefficients.
3. Decodes SimCC Z into root-relative depth with `z_range = 2.1744869`.
4. Estimates absolute root depth from the two shoulder rays by assuming a `0.38 m`
   human shoulder width. It falls back to `2.5 m` only if there is no physically
   valid solution between `0.5 m` and `6 m`.
5. Backprojects each point with `X = (u-cx)/fx*Z`, `Y = (v-cy)/fy*Z`.
6. Computes Euclidean shoulder-elbow and elbow-wrist directions.

If no stored calibration matches the browser camera ID, the checked path uses zero
distortion, a centered principal point, and a 60-degree horizontal field of view.
The UI reports whether the active source is `calibrated` or
`reasonable_default_60deg_hfov`. With the checkbox clear, the original SimCC proxy
mapping remains active.

For calibration, print `calibration-board-a4.pdf` at **100% / Actual size** on A4
paper. Do not use Fit or Scale to page. Verify that every chessboard square measures
exactly `24 mm`; markers are `18 mm`. The board is 7 by 10 squares and uses
`DICT_5X5_100`. Mount it flat, then use the web calibration tool to capture at least
12 views distributed around the image with varied distance and tilt. After selecting
**Begin automatic calibration**, capture is hands-free: near-duplicate views are
rejected, and the profile is calibrated, saved, and enabled automatically after 12
accepted views. The provided `calibration-board-a4-300dpi.png` is the exact 300 DPI
raster alternative.

Calibrations are stored by a non-identifying hash of the browser camera device ID
in `calibrations/cameras.json`. This file is tracked by Git, so review and commit
new profiles after calibrating a camera.

## Coordinate and IK contract

`RobotTarget` positions are expressed in metres in the model's `arm_origin` frame:

- RTMW3D decoded SimCC coordinates supply relative shoulder-elbow-wrist directions.
- Human segment magnitude is discarded; directions are scaled to the OpenArm's
  measured 0.220 m upper arm and 0.216 m forearm.
- Limb directions map directly from camera coordinates into the robot base frame;
  there is no neutral-pose calibration or relative-motion offset.
- Wrist, index-, middle-, ring-, and little-finger MCP landmarks define an
  absolute right-handed palm frame. Its orientation is filtered with the same
  Cartesian time constant and tracked by Mink whenever the selected mode
  includes the end effector.
- **Elbow only** uses only the elbow-position residual and body Jacobian.
- **End effector only** uses only the end-effector-position residual and site
  Jacobian.
- **Elbow + end effector** stacks both position residuals and Jacobians.
- The normalized Cartesian limb directions are exponentially filtered before IK.
  Mink solves both arms together with configuration and velocity constraints plus
  a previous-posture objective. Its solution is exponentially filtered again
  before reaching the position actuators.

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
`IK_POSTURE_COST`, `IK_MAX_VELOCITY_RAD_S`, `IK_TASK_GAIN`,
`IK_INTEGRATION_DT_S`, `IK_QP_SOLVER`, `IK_WRIST_ORIENTATION_COST`,
`RETARGET_SMOOTHING_TAU_S`, `ROBOT_COMMAND_SMOOTHING_TAU_S`,
`ROBOT_COMMAND_MAX_SPEED_RAD_S`, `RETARGET_CONFIDENCE`,
`RETARGET_CONFIDENCE_SECONDS`, `RTMW3D_DET_FREQUENCY`, and
`OPENARM_MODEL_PATH`. Cartesian position and orientation smoothing default to
`0.25 s`; set `RETARGET_SMOOTHING_TAU_S=0` to disable it. Robot-joint
exponential smoothing defaults to `0`, while the separate joint-speed limit
remains active unless `ROBOT_COMMAND_MAX_SPEED_RAD_S` is increased.
