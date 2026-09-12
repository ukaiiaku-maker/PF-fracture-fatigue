from dataclasses import replace

from arrhenius_fracture.controlled_history_v3 import (
    DIFFUSION_SEPARATION_MARGIN, canonical_rate_channel, dormant_interval_audit,
    growth_limiter_audit, mirrored_source_audit,
)
from arrhenius_fracture.voiding_production_v5 import build_production_void_state


def test_canonical_channel_taxonomy_drops_units_suffix():
    assert canonical_rate_channel("plastic_accommodation_s") == "plastic_accommodation"


def test_diffusion_history_requires_every_interval_and_frozen_margin():
    rates = {"surface_reaction_s": 4.0, "plastic_accommodation_s": 2.0,
             "vacancy_transport_s": 1.8}
    operations = [{"api": "accepted_load_growth_interval", "rates": rates}] * 8
    result = growth_limiter_audit(operations, target_channel="vacancy_transport_s")
    assert result["passed"]
    assert min(row["relative_separation_margin"] for row in result["intervals"]) > DIFFUSION_SEPARATION_MARGIN
    assert not growth_limiter_audit(operations[:7], target_channel="vacancy_transport_s")["passed"]


def test_zero_rate_interval_requires_exact_owned_state_invariance():
    state, _ = build_production_void_state(stochastic=True)
    rates = [{"effective_rate_s": 0.0, "winner": False, "emitted_event_ids": []}]
    assert dormant_interval_audit(state, state, rates)["passed"]
    changed = replace(state, rng_state={"advanced": True})
    assert not dormant_interval_audit(state, changed, rates)["passed"]


def test_mirror_audit_applies_signed_tensor_transform():
    positive = {"tensor_Pa": [[1.0, 2.0], [2.0, 3.0]], "eta_n_max": 1.0,
                "eta_t_max": 1.0, "local_aspect_ratio_max": 2.0,
                "minimum_quality": 0.2, "local_minimum_quality": 0.3,
                "normalized_traction": 0.01}
    negative = {**positive, "tensor_Pa": [[1.0, -2.0], [-2.0, 3.0]]}
    assert mirrored_source_audit(positive, negative)["passed"]
