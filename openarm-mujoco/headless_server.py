"""Run OpenArm physics and stream offscreen MuJoCo frames over HTTP."""

from __future__ import annotations

import argparse
import io
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / ".assets" / "openarm_mujoco" / "v1" / "scene.xml"
ARM_ACTUATORS = [
    f"{side}_joint{index}_ctrl"
    for side in ("left", "right")
    for index in range(1, 8)
]
HTML = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>OpenArm MuJoCo</title>
<style>
body{margin:0;background:#111;color:#eee;font:16px system-ui;text-align:center}
h1{font-size:20px;margin:14px}img{max-width:100vw;max-height:calc(100vh - 60px)}
</style></head><body><h1>OpenArm bimanual - headless MuJoCo/EGL</h1>
<img src="/stream.mjpg" alt="MuJoCo stream"></body></html>"""


class Simulation:
    def __init__(self, width: int, height: int, fps: float, demo_motion: bool) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.model.vis.global_.offwidth = width
        self.model.vis.global_.offheight = height
        self.data = mujoco.MjData(self.model)
        self.width = width
        self.height = height
        self.fps = fps
        self.demo_motion = demo_motion
        self.condition = threading.Condition()
        self.frame = b""
        self.frame_number = 0
        self.running = True
        self.actuator_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in ARM_ACTUATORS
            ]
        )

    def run(self) -> None:
        render_period = 1.0 / self.fps
        next_render = time.monotonic()
        start = next_render
        with mujoco.Renderer(self.model, height=self.height, width=self.width) as renderer:
            while self.running:
                now = time.monotonic()
                if self.demo_motion:
                    phase = now - start
                    values = 0.35 * np.sin(phase * 0.8 + np.arange(7) * 0.45)
                    self.data.ctrl[self.actuator_ids[:7]] = values
                    self.data.ctrl[self.actuator_ids[7:]] = -values
                mujoco.mj_step(self.model, self.data)

                if now >= next_render:
                    renderer.update_scene(self.data, camera="front_camera")
                    rgb = renderer.render()
                    output = io.BytesIO()
                    Image.fromarray(rgb).save(output, format="JPEG", quality=85)
                    with self.condition:
                        self.frame = output.getvalue()
                        self.frame_number += 1
                        self.condition.notify_all()
                    next_render = now + render_period
                else:
                    time.sleep(min(0.001, next_render - now))

    def wait_for_frame(self, after: int) -> tuple[int, bytes]:
        with self.condition:
            self.condition.wait_for(
                lambda: self.frame_number > after or not self.running, timeout=2.0
            )
            return self.frame_number, self.frame


class Handler(BaseHTTPRequestHandler):
    simulation: Simulation

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(HTML)))
            self.end_headers()
            self.wfile.write(HTML)
            return
        if self.path != "/stream.mjpg":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        frame_number = -1
        try:
            while self.simulation.running:
                frame_number, frame = self.simulation.wait_for_frame(frame_number)
                if not frame:
                    continue
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(frame)).encode()
                    + b"\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.client_address[0]} - {format % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--demo-motion",
        action="store_true",
        help="Apply low-amplitude mirrored torques to demonstrate actuator-driven motion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulation = Simulation(args.width, args.height, args.fps, args.demo_motion)
    Handler.simulation = simulation
    simulation_thread = threading.Thread(target=simulation.run, daemon=True)
    simulation_thread.start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://localhost:{args.port} in the Windows browser")
    print(f"MuJoCo GL backend: {os.environ['MUJOCO_GL']}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        simulation.running = False
        server.shutdown()
        simulation_thread.join(timeout=3.0)


if __name__ == "__main__":
    main()
