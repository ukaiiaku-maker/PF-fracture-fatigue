import dataclasses
from pathlib import Path

import pytest

from arrhenius_fracture.v12_production_driver import (
    PhysicalCleavageEventContext,build_physical_cleavage_event_context,
    execute_event,physical_cleavage_resistance,
    validate_physical_cleavage_event_context)
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint


def context(hits=3.0):
    return PhysicalCleavageEventContext(
        accepted_state_fingerprint="state",candidate_id="candidate",
        front_id="b00000000",source_fingerprint="source",event_ids=("event",),
        hazard_barrier_J=1.2e-19,cooperative_hits=hits,
        burgers_vector_m=2.5e-10,gamma_relative=1.25,
        material_context_fingerprint="material")


def test_nondefault_cooperative_hits_are_consumed_by_native_resistance():
    baseline=physical_cleavage_resistance(context(3.0))
    nondefault=physical_cleavage_resistance(context(7.0))
    assert nondefault/baseline == pytest.approx(7.0/3.0)


def test_clock_multiplier_cannot_change_bound_resistance():
    bound=context(5.0)
    values=[physical_cleavage_resistance(bound) for _clock_scale in (1.0,1.0e60)]
    assert values[0] == values[1]
    assert "clock" not in {field.name for field in dataclasses.fields(bound)}


def test_missing_physical_context_is_rejected_before_state_access():
    with pytest.raises(RuntimeError,match="state-bound cleavage context"):
        execute_event(None,(1.0,0.0),transaction_identity="missing")


def test_stale_physical_context_is_rejected_before_trial_remesh():
    root=Path(__file__).resolve().parents[1]
    state=restore_checkpoint(
        root/"runs/managed_sustained_growth_v1_20260920/output/restart_000008475ms.json")
    candidate=state.competition.candidates[0]
    bound=build_physical_cleavage_event_context(
        state,candidate_id=candidate.candidate_id,front_id="b00000000",
        source_fingerprint="source",event_ids=("event",),hazard_barrier_J=1.0e-19)
    stale=dataclasses.replace(bound,accepted_state_fingerprint="stale")
    with pytest.raises(RuntimeError,match="stale physical cleavage event context"):
        validate_physical_cleavage_event_context(state,stale)
