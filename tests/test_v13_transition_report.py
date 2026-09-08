"""Reporting-only checks: no mechanics or trajectories."""
import pytest

from scripts.report_v13_transition_ensemble import km, process_parity, has_observed_mixed_incidence


def row(x, branch):
    return dict(survival_extension_um=x, branch_observed=branch)


def test_event_censor_tie_uses_full_risk_set():
    curve = km([row(10., True), row(10., False), row(75., False)])
    assert curve[1]['at_risk'] == 3
    assert curve[1]['events'] == 1
    assert curve[1]['censored'] == 1
    assert curve[1]['survival'] == pytest.approx(2/3)
    assert curve[-1]['at_risk'] == 1
    assert curve[-1]['survival'] == pytest.approx(2/3)


def test_all_censored_are_not_branch_events():
    curve = km([row(75., False)] * 8)
    assert curve[-1]['survival'] == 1.
    assert curve[-1]['events'] == 0
    assert curve[-1]['censored'] == 8


def test_early_censor_changes_risk_not_incidence():
    curve = km([row(5., False), row(10., True), row(75., False)])
    assert curve[1]['survival'] == 1.
    assert curve[2]['at_risk'] == 2
    assert curve[2]['survival'] == .5


def test_all_events_reach_zero_survival():
    assert km([row(10., True), row(20., True)])[-1]['survival'] == 0.


def test_process_comparison_excludes_only_observer_counter():
    a={'engine_fields':{'_anisotropic_drive':{'mechanics_serial':2,'tensor':3.}},'rate':7.}
    b={'engine_fields':{'_anisotropic_drive':{'mechanics_serial':4,'tensor':3.}},'rate':7.}
    r=process_parity(a,b)
    assert not r['raw_equal'] and r['physical_values_equal']
    assert a['engine_fields']['_anisotropic_drive']['mechanics_serial']==2
    b['engine_fields']['_anisotropic_drive']['tensor']=3.1
    assert not process_parity(a,b)['physical_values_equal']
    b['engine_fields']['_anisotropic_drive']['tensor']=3.
    b['rate']=7.1
    assert not process_parity(a,b)['physical_values_equal']


def test_early_gate_alone_cannot_establish_mixed_incidence():
    branch=dict(branch_observed=True,unbranched_75um_complete=False)
    gate=dict(branch_observed=False,unbranched_75um_complete=False)
    negative=dict(branch_observed=False,unbranched_75um_complete=True)
    assert not has_observed_mixed_incidence([branch,gate])
    assert not has_observed_mixed_incidence([negative,gate])
    assert has_observed_mixed_incidence([branch,negative,gate])
