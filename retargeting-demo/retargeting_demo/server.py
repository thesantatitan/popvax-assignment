"""FastAPI web boundary for browser-local camera capture and worker streams."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .contracts import BrowserFrame, RenderedFrame
from .ipc import drain_latest, put_latest

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "web" / "index.html"


@dataclass(slots=True)
class Runtime:
    frame_queue: object
    pose_frame_queue: object
    sim_frame_queue: object
    perception_telemetry_queue: object
    simulation_telemetry_queue: object
    camera_queue: object
    calibrate_event: object
    engaged_event: object


def create_app(runtime: Runtime) -> FastAPI:
    app = FastAPI(title="PopVax bimanual retargeting demo")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(INDEX)

    async def receive_browser(websocket: WebSocket) -> None:
        sequence = 0
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect
            payload = message.get("bytes")
            if payload is not None:
                sequence += 1
                put_latest(
                    runtime.frame_queue,
                    BrowserFrame(
                        sequence=sequence,
                        capture_time_ns=time.monotonic_ns(),
                        jpeg=payload,
                    ),
                )
                continue
            text = message.get("text")
            if text is None:
                continue
            command = json.loads(text)
            action = command.get("action")
            if action == "calibrate":
                runtime.engaged_event.set()
                runtime.calibrate_event.set()
            elif action == "disengage":
                runtime.engaged_event.clear()
            elif action in {"rotate", "pan", "zoom", "reset"}:
                put_latest(
                    runtime.camera_queue,
                    (
                        action,
                        float(command.get("dx", 0.0)),
                        float(command.get("dy", 0.0)),
                    ),
                )

    async def send_workers(websocket: WebSocket) -> None:
        while True:
            sent = False
            pose_frame = drain_latest(runtime.pose_frame_queue)
            if isinstance(pose_frame, RenderedFrame):
                await websocket.send_bytes(b"P" + pose_frame.jpeg)
                sent = True
            sim_frame = drain_latest(runtime.sim_frame_queue)
            if isinstance(sim_frame, RenderedFrame):
                await websocket.send_bytes(b"S" + sim_frame.jpeg)
                sent = True
            for telemetry_queue in (
                runtime.perception_telemetry_queue,
                runtime.simulation_telemetry_queue,
            ):
                telemetry = drain_latest(telemetry_queue)
                if telemetry is not None:
                    await websocket.send_text(json.dumps(telemetry))
                    sent = True
            await asyncio.sleep(0.001 if sent else 0.005)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        runtime.engaged_event.clear()
        receiver = asyncio.create_task(receive_browser(websocket))
        sender = asyncio.create_task(send_workers(websocket))
        try:
            done, pending = await asyncio.wait(
                {receiver, sender}, return_when=asyncio.FIRST_EXCEPTION
            )
            for task in done:
                task.result()
            for task in pending:
                task.cancel()
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            runtime.engaged_event.clear()
            receiver.cancel()
            sender.cancel()

    return app
