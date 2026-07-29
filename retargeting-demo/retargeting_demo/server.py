"""FastAPI web boundary for browser-local camera capture and worker streams."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .contracts import RETARGET_MODES, BrowserFrame, RenderedFrame
from .ipc import drain_latest, put_latest

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "web" / "index.html"


@dataclass(slots=True)
class Runtime:
    frame_queue: object
    mode_queue: object
    pose_frame_queue: object
    sim_frame_queue: object
    perception_telemetry_queue: object
    simulation_telemetry_queue: object
    camera_queue: object
    tracking_reset_event: object
    engaged_event: object


class BroadcastHub:
    """Drain worker outputs once and fan them out to every connected page."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self.clients: set[asyncio.Queue] = set()
        self.latest: dict[str, object] = {}

    def register(self) -> asyncio.Queue:
        channel: asyncio.Queue = asyncio.Queue(maxsize=8)
        self.clients.add(channel)
        for item in self.latest.values():
            channel.put_nowait(item)
        return channel

    def unregister(self, channel: asyncio.Queue) -> None:
        self.clients.discard(channel)

    def publish(self, key: str, payload: object) -> None:
        item = (key, payload)
        self.latest[key] = item
        for channel in tuple(self.clients):
            if channel.full():
                try:
                    channel.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            channel.put_nowait(item)

    async def run(self) -> None:
        while True:
            sent = False
            pose_frame = drain_latest(self.runtime.pose_frame_queue)
            if isinstance(pose_frame, RenderedFrame):
                self.publish("pose", pose_frame)
                sent = True
            sim_frame = drain_latest(self.runtime.sim_frame_queue)
            if isinstance(sim_frame, RenderedFrame):
                self.publish("simulation-frame", sim_frame)
                sent = True
            perception = drain_latest(self.runtime.perception_telemetry_queue)
            if perception is not None:
                self.publish("perception", perception)
                sent = True
            simulation = drain_latest(self.runtime.simulation_telemetry_queue)
            if simulation is not None:
                self.publish("simulation", simulation)
                sent = True
            await asyncio.sleep(0.001 if sent else 0.005)


def create_app(runtime: Runtime) -> FastAPI:
    hub = BroadcastHub(runtime)
    controller: WebSocket | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        broadcaster = asyncio.create_task(hub.run())
        try:
            yield
        finally:
            broadcaster.cancel()
            with suppress(asyncio.CancelledError):
                await broadcaster

    app = FastAPI(
        title="PopVax bimanual retargeting demo",
        lifespan=lifespan,
    )

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(INDEX)

    async def receive_browser(websocket: WebSocket) -> None:
        nonlocal controller
        sequence = 0
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect
            payload = message.get("bytes")
            if payload is not None:
                controller = websocket
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
            if action == "disengage" and controller is websocket:
                runtime.engaged_event.clear()
                runtime.tracking_reset_event.set()
            elif action == "set_mode":
                mode = command.get("mode")
                if mode in RETARGET_MODES:
                    runtime.engaged_event.clear()
                    runtime.tracking_reset_event.set()
                    put_latest(runtime.mode_queue, mode)
            elif action in {"rotate", "pan", "zoom", "reset"}:
                put_latest(
                    runtime.camera_queue,
                    (
                        action,
                        float(command.get("dx", 0.0)),
                        float(command.get("dy", 0.0)),
                    ),
                )

    async def send_workers(
        websocket: WebSocket, client_channel: asyncio.Queue
    ) -> None:
        while True:
            kind, payload = await client_channel.get()
            if kind == "pose" and isinstance(payload, RenderedFrame):
                await websocket.send_bytes(b"P" + payload.jpeg)
            elif kind == "simulation-frame" and isinstance(payload, RenderedFrame):
                await websocket.send_bytes(b"S" + payload.jpeg)
            else:
                await websocket.send_text(json.dumps(payload))

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        nonlocal controller
        await websocket.accept()
        client_channel = hub.register()
        receiver = asyncio.create_task(receive_browser(websocket))
        sender = asyncio.create_task(send_workers(websocket, client_channel))
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
            if controller is websocket:
                runtime.engaged_event.clear()
                runtime.tracking_reset_event.set()
                controller = None
            hub.unregister(client_channel)
            receiver.cancel()
            sender.cancel()

    return app
