# OpenArm MuJoCo setup

This subfolder provides the assessment's simulation baseline: OpenArm v1 bimanual
(7 arm joints per side), pinned MuJoCo/Python dependencies, official collision
geometry and joint limits, and a verification that the arms move through actuators
while MuJoCo steps physics.

The official assets are fetched at the pinned OpenArm commit
`8955afb54e4adfb59a236e2b4d15192b7a02865c` into the ignored `.assets/`
directory. This avoids modifying the upstream model while keeping setup reproducible.

## WSL setup

From Ubuntu/WSL:

```bash
cd /home/dev/popvax-assignment/openarm-mujoco
chmod +x setup.sh
./setup.sh
```

The setup uses `uv`, creates a local `.venv`, installs the versions in `uv.lock`,
fetches the pinned OpenArm files, and runs the headless validation.

## Launch

With WSLg enabled:

```bash
cd /home/dev/popvax-assignment/openarm-mujoco
uv run python launch.py
```

When launching through SSH, provide the WSLg variables explicitly:

```bash
DISPLAY=:0 \
WAYLAND_DISPLAY=wayland-0 \
XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir \
uv run python launch.py
```

The MuJoCo viewer should show the complete bimanual OpenArm. To inspect collision
geometry, enable the collision/convex-hull rendering groups in the viewer.

## Headless browser stream

WSLg is not required. MuJoCo can render through EGL on the WSL GPU and stream
JPEG frames to a normal Windows browser:

```bash
cd /home/dev/popvax-assignment/openarm-mujoco
MUJOCO_GL=egl uv run python headless_server.py --demo-motion
```

Open <http://localhost:8080> on Windows. The optional demo motion is generated
with actuator torques and physics stepping; omit `--demo-motion` for a stationary
scene ready to be connected to the retargeting controller.

Useful options:

```bash
uv run python headless_server.py --help
uv run python headless_server.py --width 1280 --height 720 --fps 30
```

## Re-run validation

```bash
uv run python verify_setup.py
```

The check requires:

- exactly 14 named arm joints and 14 matching arm actuators;
- limits on all 14 arm joints;
- active collision geoms;
- measurable joint motion after applying actuator controls and stepping physics.

The retargeting controller should write desired torques to the 14 actuator controls
and call `mujoco.mj_step`; it must not write desired poses directly into `data.qpos`.
