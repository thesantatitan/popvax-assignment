from retargeting_demo.confidence import ContinuousConfidenceGate


def test_requires_two_continuous_seconds_and_reacquires_after_drop() -> None:
    gate = ContinuousConfidenceGate(required_seconds=2.0)

    assert gate.update(True, 0).state == "acquiring"
    assert not gate.update(True, 1_999_999_999).ready
    assert gate.update(True, 2_000_000_000).ready

    lost = gate.update(False, 2_100_000_000)
    assert not lost.ready
    assert lost.state == "holding"

    reacquiring = gate.update(True, 3_000_000_000)
    assert reacquiring.state == "reacquiring"
    assert not gate.update(True, 4_999_999_999).ready
    assert gate.update(True, 5_000_000_000).ready


def test_reset_forgets_previous_engagement() -> None:
    gate = ContinuousConfidenceGate(required_seconds=2.0)
    gate.update(True, 0)
    assert gate.update(True, 2_000_000_000).ready
    assert gate.reset().state == "waiting"
    assert gate.update(False, 3_000_000_000).state == "waiting"
