# PopVax retargeting demo

This repository contains the webcam-to-bimanual-OpenArm retargeting demo in
[`retargeting-demo/`](retargeting-demo/). The browser captures webcam frames,
RTMW3D estimates the operator's pose, and a MuJoCo OpenArm scene applies the
retargeted elbow and wrist targets.

The intended runtime is Linux/WSL with an NVIDIA GPU, CUDA, and the pinned
TensorRT dependencies. The browser can run on that same machine or on a Mac
through an SSH tunnel.

## Prerequisites

- Linux or WSL with NVIDIA GPU access and CUDA libraries
- Python 3.11
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- SSH access from the browser machine if using a separate Mac browser

The OpenArm MuJoCo model and meshes are already vendored under
`retargeting-demo/vendor/openarm-v2/`.

## Install

From the repository root, run this in the WSL/Linux environment:

```bash
cd retargeting-demo
chmod +x setup.sh run_wsl_server.sh prepare_tensorrt_engine.sh
./setup.sh
```

Setup creates the locked `uv` environment, installs the GPU/TensorRT and test
extras, downloads the official RTMW3D-L checkpoint, builds a TensorRT 8.6 FP32
engine for the local GPU, verifies the runtime, and runs the test suite. The
generated plan is stored outside the repository at
`~/.cache/rtmlib/hub/checkpoints/rtmw3d-l-fp32.plan`.

If the TensorRT engine is stored elsewhere, set the override before setup and
when launching:

```bash
export RTMW3D_TRT_ENGINE=/path/to/rtmw3d-l-fp32.plan
```

The setup and launch scripts honor that variable. They also support
`RTMW3D_TRT_ENGINE_URL` to download a prebuilt plan, or `RTMW3D_ONNX_MODEL` to
build from an existing ONNX export. TensorRT plans are tied to the GPU and
TensorRT runtime, so local generation is the portable default.

## Run

On the WSL/Linux machine:

```bash
cd retargeting-demo
./run_wsl_server.sh
```

Open [http://localhost:8000](http://localhost:8000) in a browser on that
machine and allow camera access.

The server listens on `0.0.0.0:8000` by default. Change it with
`RETARGETING_HOST` and `RETARGETING_PORT`, or pass simulation arguments through
the launcher, for example:

```bash
RETARGETING_PORT=8010 ./run_wsl_server.sh --sim-width 1280 --sim-height 720
```

If using a non-default port through SSH, update both tunnel endpoints, for
example `ssh -N -L 8010:127.0.0.1:8010 windows-cuda-wsl`.

## Operator workflow

1. Click **Start camera** and allow camera access.
2. Choose a tracking mode: **Elbow only**, **End effector only**, **Elbow + end
   effector**, or **Elbow + end effector + orientation**.
3. Keep the required shoulders, elbows, wrists, and—when applicable—hand
   landmarks visible for two continuous seconds.
4. Tracking engages automatically. If confidence drops, the robot holds its
   last pose until confidence is restored.

Use **Stop** to disengage new target application. The simulation continues to
hold the last safe actuator command when targets are stale.

## Validation and logs

Run the checks from `retargeting-demo/`:

```bash
uv run python verify_runtime.py
uv run pytest
```

The assignment-facing desired joint targets are written to
`retargeting-demo/assignment_logs/retargeting_target.jsonl`; each demo start
overwrites that file. Additional per-run perception and achieved-pose logs are
written under `retargeting-demo/logs/`.

For camera calibration, use the **Camera intrinsics** controls in the web UI.
Print [`retargeting-demo/calibration-board-a4.pdf`](retargeting-demo/calibration-board-a4.pdf)
at 100% / Actual size on A4 paper. Detailed calibration, coordinate-frame, IK,
and environment-variable documentation is in
[`retargeting-demo/README.md`](retargeting-demo/README.md).

## Troubleshooting

- **TensorRT engine missing:** rerun `./setup.sh`; the launcher also prepares it
  automatically. Set `RTMW3D_TRT_ENGINE` only to use a custom `.plan` path.
- **CUDA provider missing:** run `uv run python verify_runtime.py` and check
  that WSL can see the NVIDIA GPU and its CUDA libraries.
- **No camera prompt:** open the URL in the browser that owns the camera and
  grant permission; do not open the WSL URL on the Mac without the SSH tunnel.
- **Port already in use:** choose another port with `RETARGETING_PORT` and use
  the same local port in the SSH tunnel.
- **OpenArm model missing:** confirm that
  `retargeting-demo/vendor/openarm-v2/cell.xml` exists, or set
  `OPENARM_MODEL_PATH` to a compatible model.
