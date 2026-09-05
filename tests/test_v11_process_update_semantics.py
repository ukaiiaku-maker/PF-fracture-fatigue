import random
from types import SimpleNamespace

import pytest

from arrhenius_fracture.kinetic_tip_cell import (
    KineticMovingTipFrontEngine, KineticTipConfig,
)
from arrhenius_fracture.process_update_semantics_v11 import (
    classify_process_update, require_full_accepted_interval_consumption,
)


def test_disabled_legacy_hazard_checkpoint_sync_is_not_time_refinement():
    decision = classify_process_update(
        {
            "fired": True,
            "n_fire": 1,
            "stochastic_hazard_enabled": False,
            "physical_hazard_action_step": 0.0,
            "avalanche_checkpoint_synchronized": True,
            "checkpoint_committed_total_m": 20e-6,
        },
        directional_event_expected=False,
        permitted_physical_hazard_action=0.075,
    )
    assert decision.event_semantics == "process_checkpoint_synchronization"
    assert not decision.refinement_required
    assert decision.physical_hazard_action_step == 0.0


def test_genuine_legacy_physical_hazard_crossing_retains_refinement():
    decision = classify_process_update(
        {
            "fired": True,
            "n_fire": 1,
            "stochastic_hazard_enabled": True,
            "physical_hazard_action_step": 0.2,
            "avalanche_checkpoint_synchronized": True,
        },
        directional_event_expected=True,
        permitted_physical_hazard_action=0.075,
    )
    assert decision.event_semantics == "legacy_physical_hazard_crossing"
    assert decision.refinement_required
    assert decision.physical_hazard_action_step == pytest.approx(0.2)


def test_resolved_directional_clock_increment_is_ordinary_process_advance():
    decision = classify_process_update(
        {
            "fired": False,
            "n_fire": 0,
            "stochastic_hazard_enabled": False,
            "physical_hazard_action_step": 0.0,
            "avalanche_checkpoint_synchronized": True,
        },
        directional_event_expected=False,
        permitted_physical_hazard_action=0.15,
    )
    assert decision.event_semantics == "ordinary_process_state_advance"
    assert not decision.refinement_required


@pytest.mark.parametrize("rate", [1e-34, 1e-46, 1e-64])
def test_dormant_directional_rate_does_not_create_process_sentinel(rate):
    dH = rate * 8.4
    decision = classify_process_update(
        {
            "fired": False,
            "n_fire": 0,
            "stochastic_hazard_enabled": False,
            "physical_hazard_action_step": 0.0,
            "hazard_progress_rate": dH,
            "avalanche_checkpoint_synchronized": True,
        },
        directional_event_expected=False,
        permitted_physical_hazard_action=0.15,
    )
    assert not decision.refinement_required
    assert decision.physical_hazard_action_step == 0.0


def test_shared_owner_checkpoint_sync_preserves_membership_contract():
    member_tips = ("tip-a", "tip-b")
    decision = classify_process_update(
        {
            "fired": True,
            "n_fire": 1,
            "stochastic_hazard_enabled": False,
            "physical_hazard_action_step": 0.0,
            "avalanche_checkpoint_synchronized": True,
        },
        directional_event_expected=False,
        permitted_physical_hazard_action=0.075,
    )
    owner_by_tip = {tip: "shared-cluster" for tip in member_tips}
    assert not decision.refinement_required
    assert tuple(tip for tip, owner in owner_by_tip.items() if owner == "shared-cluster") == member_tips


def test_process_observer_must_consume_exact_accepted_interval():
    result = require_full_accepted_interval_consumption(
        {"kinetic_dt_consumed_s": 8.4, "kinetic_dt_unused_s": 0.0}, 8.4,
    )
    assert result["process_dt_consumed_s"] == 8.4
    assert result["process_dt_unused_s"] == 0.0
    with pytest.raises(RuntimeError, match="complete accepted physical interval"):
        require_full_accepted_interval_consumption(
            {"kinetic_dt_consumed_s": 0.1, "kinetic_dt_unused_s": 8.3}, 8.4,
        )


class _NoPlasticProcessState:
    def copy(self):
        return _NoPlasticProcessState()


def test_directional_topology_owner_suppresses_legacy_cleavage_for_full_interval():
    engine = KineticMovingTipFrontEngine.__new__(KineticMovingTipFrontEngine)
    engine.tip_cfg = KineticTipConfig(plasticity_enabled=False)
    engine.mpz = _NoPlasticProcessState()
    engine.f = SimpleNamespace(da=5.0e-6)
    engine.B = 0.99
    engine.W_emit = 0.0
    engine.t = 0.0
    engine.micro_advance_total_m = 0.0
    engine.packet_count_mean_total = 0.0
    engine.packet_variance_total_m2 = 0.0
    engine.checkpoint_advance_total_m = 0.0
    engine.n_adv = 0
    engine._directional_topology_owns_cleavage = True
    engine._hazard_rng = random.Random(3621)
    engine.sigma_tip = lambda _K: 1.0
    engine.lambda_cleave = lambda _stress, _temperature: (10.0, 10.0, 1.0)

    rng_before = engine._hazard_rng.getstate()
    result = engine._integrate_coupled(1.0, 700.0, 8.4)

    assert result["dt_consumed"] == pytest.approx(8.4)
    assert result["dt_unused"] == pytest.approx(0.0)
    assert result["fired"] is False
    assert result["da"] == pytest.approx(0.0)
    assert engine.B == pytest.approx(0.99)
    assert engine._hazard_rng.getstate() == rng_before
