import queue
from types import SimpleNamespace

from retargeting_demo.server import BroadcastHub


def test_broadcast_hub_fans_out_and_replays_latest() -> None:
    runtime = SimpleNamespace(
        pose_frame_queue=queue.Queue(),
        sim_frame_queue=queue.Queue(),
        perception_telemetry_queue=queue.Queue(),
        simulation_telemetry_queue=queue.Queue(),
    )
    hub = BroadcastHub(runtime)
    first = hub.register()
    second = hub.register()

    payload = {"type": "perception", "fps": 15.0}
    hub.publish("perception", payload)

    assert first.get_nowait() == ("perception", payload)
    assert second.get_nowait() == ("perception", payload)

    late = hub.register()
    assert late.get_nowait() == ("perception", payload)
