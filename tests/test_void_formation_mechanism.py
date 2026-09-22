from __future__ import annotations

import pytest

from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.void_formation_mechanism import (
    VACANCY_CLUSTERING_V1, VacancyClusteringV1,
    active_void_formation_mechanism,
)
from arrhenius_fracture.voiding_production_v5 import build_production_void_state


def test_only_vacancy_clustering_v1_is_enabled_and_state_owned():
    mechanism = active_void_formation_mechanism()
    assert isinstance(mechanism, VacancyClusteringV1)
    assert mechanism.mechanism_id == VACANCY_CLUSTERING_V1
    with pytest.raises(ValueError, match="unimplemented"):
        active_void_formation_mechanism("PREEXISTING_CAVITY")
    state, _ = build_production_void_state(
        bundle=material_bundle("DBTT"), temperature_K=900.0,
    )
    checkpoint = mechanism.checkpoint_payload(state.vacancy_transport_state_v2)
    conservation = mechanism.conservation_audit(state.vacancy_transport_state_v2)
    assert checkpoint["mechanism_id"] == VACANCY_CLUSTERING_V1
    assert checkpoint["parameter_row_fingerprint"] == (
        state.vacancy_transport_state_v2.parameter_row_fingerprint
    )
    assert conservation["mechanism_id"] == VACANCY_CLUSTERING_V1
    assert conservation["inventory_backed_hits"]
