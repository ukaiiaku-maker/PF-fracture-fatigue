from types import SimpleNamespace as NS
import math
import pytest
from arrhenius_fracture.inherited_residual_clock_v13 import InheritedCompanionClock, ResidualRaceParameters, residual_completion_time, apply_inherited_residual_race


def test_residual_integrates_existing_action_without_new_threshold():
    clock=InheritedCompanionClock('b',.9,1.,4,'existing-threshold-key')
    assert residual_completion_time(clock,[(.005,10),(.1,20)])==pytest.approx(.0075)
    assert clock.action==.9 and clock.threshold==1 and clock.event_ordinal==4
    assert math.isinf(residual_completion_time(clock,[(1.,0.)]))


def test_default_off_exact_fallback_and_enabled_explicit_rates():
    state=object(); clock=InheritedCompanionClock('b',.9,1.,4,'rng')
    out=apply_inherited_residual_race(parent=None,baseline_single_state=state,clock=clock,
        hazard_segments=[],parameters=ResidualRaceParameters(),exact_pair_trial=lambda *_:pytest.fail('called'))
    assert out.state is state
    with pytest.raises(ValueError):
        ResidualRaceParameters(enabled=True).validate()


def test_inherited_completion_races_only_private_commitment_then_exact_pair():
    parent=NS(primary_candidate_id='a',primary_event_id='e',primary_event_ordinal=1,front_lineage=('root',),
        accepted_state_id='state',process_state_sha256='process')
    state=NS(competition=NS(consumed_event_ids=('e',)),rng_state=object(),tip_process_state=object(),event_counters=object())
    clock=InheritedCompanionClock('b',.9,1.,4,'original-threshold-key')
    arguments=dict(parent=parent,baseline_single_state=state,clock=clock,hazard_segments=[(1.,10)],
        parameters=ResidualRaceParameters(True,1e-100,0.),observation_accepted_state_id='state',observation_process_sha256='process')
    out=apply_inherited_residual_race(**arguments,exact_pair_trial=lambda *_:NS(accepted=False,rejection_reason='energy_veto'))
    assert out.state is state and out.disposition=='SINGLE_COMPANION_REJECTED'
    accepted=NS(**vars(state))
    out=apply_inherited_residual_race(**arguments,exact_pair_trial=lambda *_:NS(accepted=True,state=accepted))
    assert out.state is accepted and out.mark.baseline_physical_time_increment_s==0
    assert clock.threshold_rng_identity=='original-threshold-key'
