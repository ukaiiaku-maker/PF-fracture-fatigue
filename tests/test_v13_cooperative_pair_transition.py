from dataclasses import replace
from types import SimpleNamespace as NS
import pytest
from arrhenius_fracture.cooperative_pair_transition_v13 import CooperativePairParameters, frozen_pair_probability, apply_cooperative_pair_transition


def test_default_off_does_not_call_pair_provider():
    state=object()
    out=apply_cooperative_pair_transition(parent=None,baseline_single_state=state,companion_id='b',
        parameters=CooperativePairParameters(),temperature_K=300,rate_balance=1,
        exact_pair_trial=lambda *_:pytest.fail('disabled provider called'))
    assert out.state is state


def test_enabled_model_has_no_silent_kinetic_defaults():
    with pytest.raises(ValueError,match='explicit'):
        frozen_pair_probability(CooperativePairParameters(enabled=True),temperature_K=300,rate_balance=1)


def test_competing_pair_probability_and_exact_fallback():
    p=CooperativePairParameters(True,0.,2.,1.,1.)
    assert frozen_pair_probability(p,temperature_K=300,rate_balance=1)['P_pair_embryo']==.5
    assert frozen_pair_probability(p,temperature_K=300,rate_balance=0)['P_pair_embryo']==0
    parent=NS(front_lineage=('a',),primary_event_id='event:1',primary_event_ordinal=1,primary_candidate_id='a')
    state=NS(competition=object(),rng_state=object(),tip_process_state=object(),event_counters=object())
    # Deterministic limiting fixtures, not seed screening or physical parameters.
    p=replace(p,pair_attempt_rate_per_s=1e100)
    veto=apply_cooperative_pair_transition(parent=parent,baseline_single_state=state,companion_id='b',
        parameters=p,temperature_K=300,rate_balance=1,
        exact_pair_trial=lambda *_:NS(accepted=False,rejection_reason='exact_energy_veto'))
    assert veto.state is state and veto.disposition=='SINGLE_COMPANION_REJECTED'
    accepted=NS(**vars(state))
    out=apply_cooperative_pair_transition(parent=parent,baseline_single_state=state,companion_id='b',
        parameters=p,temperature_K=300,rate_balance=1,
        exact_pair_trial=lambda *_:NS(accepted=True,state=accepted))
    assert out.state is accepted and out.mark.baseline_physical_time_increment_s==0
    accepted.competition=object()
    with pytest.raises(RuntimeError,match='competition'):
        apply_cooperative_pair_transition(parent=parent,baseline_single_state=state,companion_id='b',
            parameters=p,temperature_K=300,rate_balance=1,
            exact_pair_trial=lambda *_:NS(accepted=True,state=accepted))
