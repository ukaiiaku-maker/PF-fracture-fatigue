from __future__ import annotations

import math

import numpy as np
import pytest

from arrhenius_fracture import crystal
from arrhenius_fracture.kernel_extension_coordinate_v10228 import (
    ProjectedLigamentEquivalentCoordinate,
    install_direction_tracker,
    record_selected_direction,
    selected_direction,
)


def test_projected_coordinate_matches_nominal_straight_path():
    nominal = math.cos(math.radians(15.0))
    coordinate = ProjectedLigamentEquivalentCoordinate()

    assert coordinate.update(0.0, nominal, nominal) == pytest.approx(0.0)
    value = coordinate.update(10.0e-6, nominal, nominal)

    assert coordinate.projected_anchor_m == pytest.approx(10.0e-6 * nominal)
    assert value == pytest.approx(10.0e-6)


def test_projected_coordinate_accounts_for_direction_switches():
    nominal = math.cos(math.radians(15.0))
    low_forward = math.sin(math.radians(15.0))
    coordinate = ProjectedLigamentEquivalentCoordinate()

    coordinate.update(0.0, nominal, nominal)
    first = coordinate.update(10.0e-6, low_forward, nominal)
    second = coordinate.update(20.0e-6, nominal, nominal)

    expected_projected = 10.0e-6 * nominal + 10.0e-6 * low_forward
    assert first == pytest.approx(10.0e-6)
    assert coordinate.projected_anchor_m == pytest.approx(expected_projected)
    assert second == pytest.approx(expected_projected / nominal)
    assert second < 20.0e-6


def test_coordinate_rejects_invalid_nominal_forward_cosine():
    coordinate = ProjectedLigamentEquivalentCoordinate()
    with pytest.raises(RuntimeError, match="positive nominal forward cosine"):
        coordinate.update(1.0e-6, 1.0, 0.0)


def test_direction_tracker_preserves_continuous_selector_contract():
    install_direction_tracker()
    record_selected_direction([1.0, 0.0])

    sigma = np.array([[0.5, 0.0], [0.0, 2.0]])
    result = crystal.cleave_direction_competition(
        sigma,
        15.0,
        np.array([1.0, 0.0]),
        min_forward=0.2,
        gamma_aniso=0.3,
        branch_ratio=0.9,
    )

    assert isinstance(result, tuple)
    selected, all_candidates = result
    assert selected
    assert all_candidates
    np.testing.assert_allclose(selected_direction(), selected[0]["t"], atol=1.0e-14)
    assert selected_direction()[0] >= 0.2


def test_direction_tracker_preserves_discrete_selector_contract():
    install_direction_tracker()
    record_selected_direction([1.0, 0.0])

    planes = crystal.bcc_cleavage_traces(15.0, include_110=False)
    sigma = np.array([[0.5, 0.0], [0.0, 2.0]])
    selected = crystal.cleavage_branch_candidates(
        sigma,
        planes,
        forward=np.array([1.0, 0.0]),
        min_forward=0.2,
        branch_ratio=0.92,
    )

    assert isinstance(selected, list)
    assert selected
    np.testing.assert_allclose(selected_direction(), selected[0]["t"], atol=1.0e-14)
    assert selected_direction()[0] >= 0.2
