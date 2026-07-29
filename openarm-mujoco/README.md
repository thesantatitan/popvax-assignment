# OpenArm MuJoCo setup

This subfolder provides the assessment's simulation baseline: OpenArm v2 bimanual
(7 arm joints per side), pinned MuJoCo/Python dependencies, official collision
geometry and joint limits, and a verification that the arms move through actuators
while MuJoCo steps physics.

The official assets are tracked in `../retargeting-demo/vendor/openarm-v2` at pinned
OpenArm commit `8955afb54e4adfb59a236e2b4d15192b7a02865c`. Setup does not clone or
download a separate model checkout.

## WSL setup

From Ubuntu/WSL:

```bash
cd /home/dev/popvax-assignment/openarm-mujoco
chmod +x setup.sh
./setup.sh
```

The setup uses `uv`, creates a local `.venv`, installs the versions in `uv.lock`,
and runs the headless validation against the tracked vendor model.

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

The MuJoCo viewer should show the complete v2 bimanual OpenArm cell. To inspect collision
geometry, enable the collision/convex-hull rendering groups in the viewer.

## Headless browser stream

WSLg is not required. MuJoCo can render through EGL on the WSL GPU and stream
JPEG frames to a normal Windows browser:

```bash
cd /home/dev/popvax-assignment/openarm-mujoco
MUJOCO_GL=egl uv run python headless_server.py --demo-motion
```

On WSL, the server automatically selects Mesa's D3D12 backend when `/dev/dxg`
is available. This prevents EGL from silently falling back to the CPU-based
`llvmpipe` renderer.

Open <http://localhost:8080> on Windows. The optional demo motion is generated
with actuator torques and physics stepping; omit `--demo-motion` for a stationary
scene ready to be connected to the retargeting controller.

The browser viewer supports:

- left-drag to orbit;
- right-drag to pan;
- mouse wheel to zoom;
- double-click or **Reset camera** to restore the default view.

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

- exactly 14 named arm joints and 14 matching position actuators;
- limits on all 14 arm joints;
- internal `kp`/`kv` PD gains and position-control ranges on every arm actuator;
- active collision geoms;
- measurable joint motion after applying position targets and stepping physics.

The retargeting controller should write desired joint positions to the 14 v2 actuator
controls and call `mujoco.mj_step`; the model's internal PD controller converts those
targets into bounded actuator forces. It must not write desired poses directly into
`data.qpos`.
