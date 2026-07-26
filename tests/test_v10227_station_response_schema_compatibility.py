from __future__ import annotations

import importlib.util
from pathlib import Path

from arrhenius_fracture import physical_fem_station_responses_v10212 as responses


EXPECTED = "v10.2.14_exact_endpoint_active_signed_spatial_station_responses"


def _entry_module():
    path = Path("scripts/evaluate_v10_2_14_active_load_invariance.py").resolve()
    spec = importlib.util.spec_from_file_location("v10214_load_invariance_entry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_invariance_entry_preserves_established_station_schema():
    module = _entry_module()
    assert module._STATION_RESPONSE_SCHEMA == EXPECTED
    assert responses.MODEL_ID == EXPECTED


def test_load_invariance_entry_preserves_exact_first_and_last_bins():
    module = _entry_module()
    coordinates = tuple((index + 0.5) * 0.625e-6 for index in range(80))

    selected = module._exact_first_last_station_indices(
        coordinates,
        minimum_spacing_m=50.0e-6,
        minimum_distance_m=15.5e-6,
    )

    assert selected[0] == 0
    assert selected[-1] == len(coordinates) - 1
    assert responses._station_indices(
        coordinates,
        minimum_spacing_m=50.0e-6,
        minimum_distance_m=15.5e-6,
    ) == selected
