import pytest
from scripts.report_v13_heldout_materials import distance_summary,mark_metrics,audit_accepted


def row(x,event,early=False):
    return dict(survival_extension_um=x,branch_observed=event,early_gate_censor=early)


def test_restricted_mean_with_administrative_censor():
    r=distance_summary([row(10.,True),row(75.,False)])
    assert r['restricted_mean_branch_free_reach_um']==42.5
    assert r['median_first_branch_reach_um']==10.


def test_no_unsupported_tail_extrapolation():
    r=distance_summary([row(20.,False,True)])
    assert r['restricted_mean_branch_free_reach_um'] is None
    assert r['median_first_branch_reach_um'] is None
    assert r['informative_early_censoring_possible']


def test_all_events_support_zero_survival_tail():
    assert distance_summary([row(10.,True),row(20.,True)])['restricted_mean_branch_free_reach_um']==15.
    assert distance_summary([row(75.,False)])['restricted_mean_branch_free_reach_um']==75.


def test_pending_event_not_erased_by_zero_instantaneous_rate():
    r=mark_metrics(dict(tau_c_s=1e-6,T_j_s=0.,primary={'effective_rate_per_s':1e6},
        companion={'effective_rate_per_s':0.,'pending_event_id':'completed'},pair_margin_J_per_m=.1))
    assert r['completed_pending_companion'] and not r['companion_near_saturation']
    assert r['primary_near_saturation'] and r['exact_pair_margin_J_per_m']==.1
    empty=mark_metrics({'tau_c_s':1e-6})
    assert empty['primary_effective_lambda_tau_c'] is None and empty['exact_pair_margin_J_per_m'] is None


def test_assessment_reductions_reproduce():
    assert audit_accepted()['status']=='PASS'


def test_chi_native_waits_missing_infinite_and_pending():
    r=dict(tau_c_s=1e-6,T_i_next_s=2e-6,T_j_s=5e-7)
    assert mark_metrics(r)['chi_B']==.5
    assert mark_metrics(dict(r,T_i_next_s=2e-7))['chi_B']==2.5
    assert mark_metrics(dict(r,T_j_s=float('inf')))['chi_B_status']=='INFINITE_COMPANION_WAIT'
    assert mark_metrics(dict(tau_c_s=1e-6))['chi_B'] is None
    assert mark_metrics(dict(r,T_i_next_s=0))['chi_B'] is None
    assert mark_metrics(dict(r,T_j_s=0,companion=dict(effective_rate_per_s=0,pending_event_id='done')))['chi_B']==0
