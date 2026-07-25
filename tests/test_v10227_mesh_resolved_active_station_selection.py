from __future__ import annotations

import numpy as np
import pytest

from arrhenius_fracture.physical_fem_station_responses_v10212 import (
    _station_indices,
)


def test_first_unresolved_mpz_bins_are_skipped_without_relocation():
    spacing = 0.625e-6
    coordinates = tuple((index + 0.5) * spacing for index in range(80))
    minimum_distance = 5.53832e-6

    selected = _station_indices(
        coordinates,
        minimum_spacing_m=minimum_distance,
        minimum_distance_m=minimum_distance,
    )

    assert selected[0] == 9
    assert selected[-1] == 79
    assert coordinates[selected[0]] == pytest.approx(5.9375e-6)
    assert all(coordinates[index] >= minimum_distance for index in selected)
    assert all(
        coordinates[right] - coordinates[left] >= minimum_distance
        for left, right in zip(selected[:-2], selected[1:-1])
    )
    assert all(coordinates[index] == (index + 0.5) * spacing for index in selected)


def test_no_mesh_resolved_station_returns_empty_selection():
    coordinates = (0.25e-6, 0.75e-6, 1.25e-6)
    assert _station_indices(
        coordinates,
        minimum_spacing_m=1.0e-6,
        minimum_distance_m=2.0e-6,
    ) == []


def test_default_zero_minimum_distance_preserves_legacy_endpoint_selection():
    coordinates = (1.0, 2.0, 3.0, 4.0)
    selected = _station_indices(coordinates, minimum_spacing_m=2.0)
    assert selected == [0, 2, 3]


def test_station_selection_rejects_invalid_resolution_contract():
    with pytest.raises(ValueError, match="minimum station distance"):
        _station_indices((1.0, 2.0), 1.0, minimum_distance_m=-1.0)
    with pytest.raises(ValueError, match="minimum station spacing"):
        _station_indices((1.0, 2.0), 0.0)


def test_selected_coordinates_remain_exact_source_grid_values():
    coordinates = tuple(np.linspace(0.3125e-6, 49.6875e-6, 80))
    selected = _station_indices(
        coordinates,
        minimum_spacing_m=5.53832e-6,
        minimum_distance_m=5.53832e-6,
    )
    measured = np.asarray([coordinates[index] for index in selected])
    source = np.asarray(coordinates)
    assert np.all(np.isin(measured, source))
