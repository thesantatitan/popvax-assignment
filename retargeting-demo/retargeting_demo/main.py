"""Launch the multiprocess webcam-to-OpenArm retargeting demo."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import uvicorn

from .perception import perception_worker
from .server import Runtime, create_app
from .simulation import simulation_worker

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--sim-width", type=int, default=960)
    parser.add_argument("--sim-height", type=int, default=720)
    parser.add_argument("--sim-fps", type=float, default=30.0)
    parser.add_argument("--log-directory", type=Path, default=ROOT / "logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = mp.get_context("spawn")
    stop_event = context.Event()
    tracking_reset_event = context.Event()
    engaged_event = context.Event()
    frame_queue = context.Queue(maxsize=1)
    mode_queue = context.Queue(maxsize=1)
    target_queue = context.Queue(maxsize=1)
    pose_frame_queue = context.Queue(maxsize=1)
    sim_frame_queue = context.Queue(maxsize=1)
    perception_telemetry_queue = context.Queue(maxsize=1)
    simulation_telemetry_queue = context.Queue(maxsize=1)
    camera_queue = context.Queue(maxsize=1)

    perception = context.Process(
        name="rtmw3d-perception",
        target=perception_worker,
        args=(
            frame_queue,
            mode_queue,
            target_queue,
            pose_frame_queue,
            perception_telemetry_queue,
            tracking_reset_event,
            engaged_event,
            stop_event,
            str(args.log_directory),
        ),
    )
    simulation = context.Process(
        name="mujoco-ik-control",
        target=simulation_worker,
        args=(
            target_queue,
            sim_frame_queue,
            simulation_telemetry_queue,
            camera_queue,
            engaged_event,
            stop_event,
            str(args.log_directory),
            args.sim_width,
            args.sim_height,
            args.sim_fps,
        ),
    )
    perception.start()
    simulation.start()
    runtime = Runtime(
        frame_queue=frame_queue,
        mode_queue=mode_queue,
        pose_frame_queue=pose_frame_queue,
        sim_frame_queue=sim_frame_queue,
        perception_telemetry_queue=perception_telemetry_queue,
        simulation_telemetry_queue=simulation_telemetry_queue,
        camera_queue=camera_queue,
        tracking_reset_event=tracking_reset_event,
        engaged_event=engaged_event,
    )
    app = create_app(runtime)

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        stop_event.set()
        for process in (perception, simulation):
            process.join(timeout=8.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)


if __name__ == "__main__":
    main()
