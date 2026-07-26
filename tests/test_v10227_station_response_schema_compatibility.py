from __future__ import annotations

import importlib.util
from pathlib import Path

from arrhenius_fracture import physical_fem_station_responses_v10212 as responses


EXPECTED = "v10.2.14_exact_endpoint_active_signed_spatial_station_responses"


def test_load_invariance_entry_preserves_established_station_schema():
    path = Path("scripts/evaluate_v10_2_14_active_load_invariance.py").resolve()
    spec = importlib.util.spec_from_file_location("v10214_load_invariance_entry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._STATION_RESPONSE_SCHEMA == EXPECTED
    assert responses.MODEL_ID == EXPECTED
