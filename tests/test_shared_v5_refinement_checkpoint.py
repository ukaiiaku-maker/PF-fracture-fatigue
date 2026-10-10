from __future__ import annotations

import pytest
import numpy as np
from types import SimpleNamespace

from arrhenius_fracture import shared_v5_cavity_adapter as adapter


def test_refinement_failure_retains_complete_accepted_prefix(monkeypatch):
    def state(value):
        return SimpleNamespace(
            value=value, vacancy_fraction_gp=np.array([1e-12]), rng_state={},
            void_state=None,
            vacancy_transport_state_v2=SimpleNamespace(transport_time_s=value),
        )
    monkeypatch.setattr(
        adapter, "complete_accepted_state_fingerprint", lambda item: f"state:{item.value}",
    )

    def advance(item, duration, arm):
        if duration <= 0.25 and item.value < 0.25:
            return state(item.value + duration), {"arm": arm, "duration": duration}
        raise RuntimeError(adapter.TRANSPORT_BOUND_FAILURE)

    with pytest.raises(adapter._PairedRefinementFailure) as caught:
        adapter._advance_paired_with_refinement(
            {"disabled": state(0.0), "enabled": state(0.0)}, 1.0, advance,
            minimum_duration_s=0.2,
        )

    failure = caught.value
    assert {key: item.value for key, item in failure.states.items()} == {
        "disabled": 0.25, "enabled": 0.25,
    }
    assert [segment[0] for segment in failure.segments] == [0.25]
    assert failure.diagnostic["rollback_result"] == "exact-paired-parent-retained"

