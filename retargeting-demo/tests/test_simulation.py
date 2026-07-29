import numpy as np
import pytest

from retargeting_demo.simulation import exponential_smoothing_alpha


def test_exponential_smoothing_is_rate_independent() -> None:
    alpha_60hz = exponential_smoothing_alpha(1.0 / 60.0, 0.12)
    alpha_120hz = exponential_smoothing_alpha(1.0 / 120.0, 0.12)

    remaining_60hz = (1.0 - alpha_60hz) ** 60
    remaining_120hz = (1.0 - alpha_120hz) ** 120
    assert remaining_60hz == pytest.approx(remaining_120hz)
    assert remaining_60hz == pytest.approx(np.exp(-1.0 / 0.12))


def test_zero_time_constant_disables_smoothing() -> None:
    assert exponential_smoothing_alpha(1.0 / 60.0, 0.0) == 1.0
