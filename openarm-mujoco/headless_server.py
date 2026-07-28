"""Run OpenArm physics and stream offscreen MuJoCo frames over HTTP."""

from __future__ import annotations

import argparse
import collections
import io
import json
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
html,body{height:100%;margin:0;background:#111;color:#eee;font:14px system-ui;overflow:hidden}
header{height:42px;display:flex;align-items:center;gap:16px;padding:0 14px;background:#191919}
h1{font-size:16px;margin:0}button{margin-left:auto;padding:5px 12px;cursor:pointer}
#viewport{height:calc(100% - 42px);display:grid;place-items:center}
img{display:block;max-width:100%;max-height:100%;user-select:none;cursor:grab}
img.dragging{cursor:grabbing}.help{color:#aaa}#state{font:12px ui-monospace;color:#8fd}
</style></head><body>
<header><h1>OpenArm bimanual - headless MuJoCo/EGL</h1>
<span class="help">Left drag: orbit | Right drag: pan | Wheel: zoom | Double-click: reset</span>
<span id="state">Loading state...</span>
<button id="reset">Reset camera</button></header>
<div id="viewport"><img id="stream" src="/stream.mjpg" draggable="false" alt="MuJoCo stream"></div>
<script>
const image=document.getElementById("stream");
let drag=null,pending=null,scheduled=false;
async function camera(action,dx=0,dy=0){
  try{await fetch("/camera",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({action,dx,dy})});}catch(_){}
}
function flush(){
  scheduled=false;if(!pending)return;
  const move=pending;pending=null;camera(move.action,move.dx,move.dy);
}
image.addEventListener("contextmenu",event=>event.preventDefault());
image.addEventListener("pointerdown",event=>{
  if(event.button!==0&&event.button!==2)return;
  drag={action:event.button===0?"rotate":"pan",x:event.clientX,y:event.clientY};
  image.setPointerCapture(event.pointerId);image.classList.add("dragging");
});
image.addEventListener("pointermove",event=>{
  if(!drag)return;
  const scale=Math.max(1,image.clientHeight);
  const dx=(event.clientX-drag.x)/scale,dy=(event.clientY-drag.y)/scale;
  drag.x=event.clientX;drag.y=event.clientY;
  if(pending&&pending.action===drag.action){pending.dx+=dx;pending.dy+=dy;}
  else pending={action:drag.action,dx,dy};
  if(!scheduled){scheduled=true;requestAnimationFrame(flush);}
});
function endDrag(){drag=null;image.classList.remove("dragging");}
image.addEventListener("pointerup",endDrag);
image.addEventListener("pointercancel",endDrag);
image.addEventListener("wheel",event=>{
  event.preventDefault();camera("zoom",0,event.deltaY/Math.max(1,image.clientHeight));
},{passive:false});
image.addEventListener("dblclick",()=>camera("reset"));
document.getElementById("reset").addEventListener("click",()=>camera("reset"));
async function updateState(){
  try{
    const state=await fetch("/state.json",{cache:"no-store"}).then(response=>response.json());
    document.getElementById("state").textContent=
      `t=${state.time.toFixed(1)}s | joint motion=${state.motion.toFixed(3)} rad`;
  }catch(_){}
}
setInterval(updateState,500);updateState();
</script></body></html>"""


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
        self.camera_commands: collections.deque[tuple[str, float, float]] = (
            collections.deque()
        )
        self.camera_lock = threading.Lock()
        self.actuator_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in ARM_ACTUATORS
            ]
        )
        self.joint_ids = self.model.actuator_trnid[self.actuator_ids, 0]
        self.qpos_addresses = self.model.jnt_qposadr[self.joint_ids]
        self.dof_addresses = self.model.jnt_dofadr[self.joint_ids]
        self.initial_arm_qpos = self.data.qpos[self.qpos_addresses].copy()
        self.kp = np.tile(np.array([18.0, 18.0, 12.0, 12.0, 5.0, 5.0, 5.0]), 2)
        self.kd = np.tile(np.array([2.0, 2.0, 1.5, 1.5, 0.7, 0.7, 0.7]), 2)
        self.demo_center = np.tile(np.array([0.5, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0]), 2)
        self.demo_amplitude = np.array([0.35, 0.25, 0.30, 0.40, 0.25, 0.20, 0.25])
        self.demo_phase = np.arange(7) * 0.55

    def run(self) -> None:
        render_period = 1.0 / self.fps
        wall_start = time.monotonic()
        next_render = wall_start
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, camera)
        with mujoco.Renderer(self.model, height=self.height, width=self.width) as renderer:
            while self.running:
                now = time.monotonic()
                target_simulation_time = now - wall_start
                while self.data.time < target_simulation_time:
                    if self.demo_motion:
                        phase = self.data.time
                        wave = self.demo_amplitude * np.sin(
                            phase * 0.7 + self.demo_phase
                        )
                        desired = self.demo_center.copy()
                        desired[:7] += wave
                        desired[7:] += wave * np.array(
                            [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
                        )
                        limits = self.model.jnt_range[self.joint_ids]
                        desired = np.clip(
                            desired, limits[:, 0] + 0.02, limits[:, 1] - 0.02
                        )
                        qpos = self.data.qpos[self.qpos_addresses]
                        qvel = self.data.qvel[self.dof_addresses]
                        gravity_and_bias = self.data.qfrc_bias[self.dof_addresses]
                        torque = (
                            self.kp * (desired - qpos)
                            - self.kd * qvel
                            + gravity_and_bias
                        )
                        force_limits = self.model.actuator_forcerange[
                            self.actuator_ids
                        ]
                        torque = np.clip(
                            torque, force_limits[:, 0], force_limits[:, 1]
                        )
                        self.data.ctrl[self.actuator_ids] = torque
                    mujoco.mj_step(self.model, self.data)

                if now >= next_render:
                    with self.camera_lock:
                        commands = list(self.camera_commands)
                        self.camera_commands.clear()
                    for action, dx, dy in commands:
                        if action == "reset":
                            mujoco.mjv_defaultFreeCamera(self.model, camera)
                        else:
                            mouse_action = {
                                "rotate": mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                                "pan": mujoco.mjtMouse.mjMOUSE_MOVE_V,
                                "zoom": mujoco.mjtMouse.mjMOUSE_ZOOM,
                            }[action]
                            mujoco.mjv_moveCamera(
                                self.model,
                                mouse_action,
                                dx,
                                dy,
                                renderer.scene,
                                camera,
                            )
                    renderer.update_scene(self.data, camera=camera)
                    rgb = renderer.render()
                    output = io.BytesIO()
                    Image.fromarray(rgb).save(output, format="JPEG", quality=85)
                    with self.condition:
                        self.frame = output.getvalue()
                        self.frame_number += 1
                        self.condition.notify_all()
                    next_render += render_period
                    if next_render <= now:
                        next_render = now + render_period
                else:
                    time.sleep(min(0.001, next_render - now))

    def wait_for_frame(self, after: int) -> tuple[int, bytes]:
        with self.condition:
            self.condition.wait_for(
                lambda: self.frame_number > after or not self.running, timeout=2.0
            )
            return self.frame_number, self.frame

    def queue_camera(self, action: str, dx: float, dy: float) -> None:
        if action not in {"rotate", "pan", "zoom", "reset"}:
            raise ValueError(f"Unknown camera action: {action}")
        with self.camera_lock:
            self.camera_commands.append((action, dx, dy))

    def state(self) -> dict[str, float]:
        qpos = self.data.qpos[self.qpos_addresses].copy()
        return {
            "time": float(self.data.time),
            "motion": float(np.linalg.norm(qpos - self.initial_arm_qpos)),
        }


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
        if self.path == "/state.json":
            payload = json.dumps(self.simulation.state()).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
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

    def do_POST(self) -> None:
        if self.path != "/camera":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 4096:
                raise ValueError("Request body is too large")
            payload = json.loads(self.rfile.read(length))
            self.simulation.queue_camera(
                str(payload["action"]),
                float(payload.get("dx", 0.0)),
                float(payload.get("dy", 0.0)),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.send_error(HTTPStatus.BAD_REQUEST, str(error))
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

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
