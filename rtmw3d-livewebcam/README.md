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
