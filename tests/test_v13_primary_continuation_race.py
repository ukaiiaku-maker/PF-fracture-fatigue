from dataclasses import replace
import math
from types import SimpleNamespace as NS

import pytest
from scipy.integrate import quad

from arrhenius_fracture.inherited_primary_race_v13 import (
    ClockProjection, apply_primary_continuation_race, exponential_ensemble_probability)


def clock(cid='i', action=2., threshold=2.8, rate=1e6):
    return ClockProjection(cid,action,threshold,3,'stored-key:'+cid,rate,'accepted-state')


def fixture():
    base=NS(competition=NS(consumed_event_ids=('event',)),rng_state=object(),
            tip_process_state=object(),event_counters=object())
    pair=NS(**vars(base))
    parent=NS(primary_candidate_id='i',primary_event_id='event')
    return base,pair,parent


def run(primary=None,companions=None,trial=None,enabled=True):
    base,pair,parent=fixture()
    seen=[]
    def exact(p,m):
        seen.append(m.companion_id)
        return trial(base,pair,m) if trial else NS(accepted=True,state=pair)
    result=apply_primary_continuation_race(parent=parent,baseline_single_state=base,
        primary=primary or clock(),companions=companions or (clock('j',threshold=2.2),),
        tau_c=1e-6,exact_pair_trial=exact,enabled=enabled)
    return result,base,pair,seen


def test_cumulative_primary_threshold_not_absolute():
    assert clock().completion_s == pytest.approx(.8e-6)
    result,base,pair,seen=run()
    assert result.state is pair and seen==['j']
    assert result.mark.branch_rng_identities==()


@pytest.mark.parametrize('primary,secondary',[(clock(threshold=2.1),clock('j',threshold=2.2)),
    (clock(threshold=4.),clock('j',threshold=3.1)),(clock(),clock('j',threshold=2.8))])
def test_loss_expiry_and_tie_exact_fallback(primary,secondary):
    result,base,_,seen=run(primary,(secondary,))
    assert result.state is base and not seen


def test_default_off_exact_fallback():
    result,base,_,seen=run(enabled=False)
    assert result.state is base and not seen


def test_earliest_admissible_not_first_in_inventory():
    def trial(base,pair,mark):
        accepted=mark.companion_id!='j'
        return NS(accepted=accepted,state=pair if accepted else base,rejection_reason=None if accepted else 'energy')
    result,_,pair,seen=run(companions=(clock('k',threshold=2.3),clock('j',threshold=2.2)),trial=trial)
    assert result.state is pair and seen==['j','k']


def test_all_pair_vetoes_exact_fallback():
    result,base,_,seen=run(trial=lambda b,p,m:NS(accepted=False,state=b,rejection_reason='energy'))
    assert result.state is base and seen==['j']


def test_mixed_time_origins_rejected():
    with pytest.raises(ValueError,match='time origins'):
        run(companions=(replace(clock('j'),observation_identity='raw-crossing'),))


def test_parent_invariant_change_rejected():
    with pytest.raises(RuntimeError,match='canonical rng_state'):
        run(trial=lambda b,p,m:NS(accepted=True,state=NS(**dict(vars(p),rng_state=object()))))


@pytest.mark.parametrize('i,j',[(1e6,1e6),(1e6,1.),(0.,1e6),(0.,0.)])
def test_analytic_identity_by_deterministic_quadrature(i,j):
    tau=1e-6
    integral=quad(lambda t:j*math.exp(-(i+j)*t),0,tau)[0]
    assert exponential_ensemble_probability(i,j,tau)==pytest.approx(integral)
    assert exponential_ensemble_probability(i,j,tau,pair_admissible=False)==0.


def test_equal_saturated_not_deterministic():
    assert exponential_ensemble_probability(1e6,1e6,1e-6)==pytest.approx(.43233235838169365)


@pytest.mark.parametrize('kwargs',[{'action':math.nan},{'threshold':1.},{'rate':-1.}])
def test_invalid_or_pending_clock_fail_closed(kwargs):
    with pytest.raises(ValueError):
        clock(**kwargs)


def test_zero_rate_residual_is_infinite():
    assert math.isinf(clock(rate=0.).completion_s)


def test_existing_pending_event_is_completed_without_threshold_draw():
    pending=replace(clock('j'),action=3.,pending_event_id='j#event:3')
    assert pending.completion_s==0.
    result,_,pair,_=run(companions=(pending,))
    assert result.state is pair


def test_tied_inadmissible_candidate_does_not_hide_admissible_one():
    def trial(base,pair,mark):
        return NS(accepted=mark.companion_id=='k',state=pair if mark.companion_id=='k' else base,rejection_reason='energy')
    result,_,pair,_=run(companions=(clock('k',threshold=2.2),clock('j',threshold=2.2)),trial=trial)
    assert result.state is pair


def test_two_admissible_tied_companions_fail_closed():
    result,base,_,_=run(companions=(clock('k',threshold=2.2),clock('j',threshold=2.2)))
    assert result.state is base and result.disposition=='SINGLE_UNRESOLVED_COMPANION_TIE'
