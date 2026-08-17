from __future__ import annotations

import numpy as np

from scripts.v914_reversible_transport_utils import (
    boundary_outflow_per_m,
    proportional_cancellation_density,
)


def test_negative_velocity_returns_mobile_to_surface() -> None:
    mobile = np.array([4.0, 3.0, 2.0])
    velocity = np.array([-2.0, -2.0, -2.0])
    returned, escaped = boundary_outflow_per_m(mobile, velocity, 0.5)
    assert returned == 4.0
    assert escaped == 0.0


def test_positive_velocity_escapes_at_far_boundary() -> None:
    mobile = np.array([4.0, 3.0, 2.0])
    velocity = np.array([2.0, 2.0, 2.0])
    returned, escaped = boundary_outflow_per_m(mobile, velocity, 0.5)
    assert returned == 0.0
    assert escaped == 2.0


def test_zero_time_has_no_boundary_fate() -> None:
    returned, escaped = boundary_outflow_per_m(
        np.ones(3), np.array([-1.0, 0.0, 1.0]), 0.0
    )
    assert returned == 0.0
    assert escaped == 0.0


def test_cancellation_is_proportional_and_conservative() -> None:
    net = np.array([2.0, 1.0, 1.0])
    increment, cancelled = proportional_cancellation_density(
        net, returned_line_content=1.0, cell_area_m2=0.5
    )
    # Available line content is 2.0, so half of every populated bin cancels.
    np.testing.assert_allclose(increment, 0.5 * net)
    assert cancelled == 1.0
    assert np.isclose(np.sum(increment) * 0.5, cancelled)


def test_cancellation_cannot_exceed_available_net_slip() -> None:
    net = np.array([2.0, 1.0])
    increment, cancelled = proportional_cancellation_density(
        net, returned_line_content=100.0, cell_area_m2=0.25
    )
    np.testing.assert_allclose(increment, net)
    assert cancelled == 0.75
