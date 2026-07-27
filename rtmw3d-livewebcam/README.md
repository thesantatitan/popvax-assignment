# RTMW3D live webcam demo

This is a self-contained webcam demo for RTMW3D through the lightweight
[`rtmlib`](https://github.com/Tau-J/rtmlib) package. It draws the detected
whole-body pose over the camera feed and includes a small 3D projection of the
first detected person. The model returns 3D keypoints; use `--save-jsonl` when
you also want the raw per-frame 3D predictions.

## Quick start

From this directory:

```bash
# Install the pinned interpreter if it is not already available.
uv python install 3.11

# .python-version pins this demo to Python 3.11.
uv sync
uv run python demo.py
```

The first run downloads the detector and RTMW3D ONNX models. This is a
one-time download and can take a few minutes because the pose model is large.

On macOS, allow the terminal or IDE running the command to access the camera
when prompted. If the default camera is not the one you want, select another
device with `--source 1`.

## Controls

- `q` or `Esc`: quit
- `v`: toggle the 3D inset
- `s`: toggle the 2D skeleton overlay

Useful examples:

```bash
# Explicit CPU mode, suitable for this Mac
uv run python demo.py --device cpu

# Use a second webcam and save 3D predictions
uv run python demo.py --source 1 --save-jsonl outputs/poses.jsonl

# Use a video file instead of a camera for a repeatable smoke run
uv run python demo.py --source ./sample.mp4 --max-frames 300
```

The default backend is ONNX Runtime and the default RTMW3D configuration is
the balanced configuration from `rtmlib`. Use `python demo.py --help` for all
options.

## Remote WSL mode

The browser mode keeps the camera and display on the Mac while running the
RTMW3D model on WSL. The browser sends compressed camera frames through an SSH
tunnel; WSL returns annotated frames and inference stats. The default WSL
configuration uses the official YOLOX-Nano person detector at 416x416, runs it
every 10 frames, and re-detects early when body confidence drops or the
pose-derived ROI approaches its previous crop boundary.

On the WSL machine, in the deployed demo directory:

```bash
cd /home/dev/popvax-assignment/rtmw3d-livewebcam
uv sync --extra gpu
./run_wsl_server.sh
```

From a second terminal on the Mac, keep the tunnel open:

```bash
ssh -N -L 8000:127.0.0.1:8000 windows-cuda-wsl
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) on the Mac and click
**Start camera**. Allow camera access when the browser asks. The server checks
whether the CUDA execution provider is available and falls back to CPU with a
clear log message if the WSL environment only has CPU ONNX Runtime installed.

The `gpu` extra is Linux-only and installs the CUDA 12-compatible
`onnxruntime-gpu` 1.26 series for the WSL machine. `run_wsl_server.sh` also
exposes the CUDA/cuDNN libraries already present in the WSL PyTorch environment
to the demo; the regular Mac environment continues to use CPU ONNX Runtime.

To compare against the original detector, start the server with
`RTMW3D_DETECTOR=yolox_m RTMW3D_DET_FREQUENCY=7`. The pose model remains the
same RTMW3D checkpoint in both configurations.

## Pose-only WSL benchmark

To separate pose execution from detector, camera, and browser costs, run the
fixed-input benchmark on WSL:

```bash
uv run python pose_runtime_benchmark.py \
  --model /home/dev/.cache/rtmlib/hub/checkpoints/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx \
  --warmup 50 --iterations 500
```

It reports ordinary `session.run`, host/device copies, I/O Binding execution,
decoding, GPU utilization, and optional ORT CUDA operator profiling. Add
`--cuda-graph` to measure the static I/O Binding CUDA Graph path.
