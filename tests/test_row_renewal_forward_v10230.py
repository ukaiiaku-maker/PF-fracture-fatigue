import math,json
from pathlib import Path
import numpy as np
import pytest
from dataclasses import replace
from scripts.analyze_row_renewal_forward_v10230 import row_controls,source_rows
from scripts.forward_temperature_model_v10230 import f0_root,f0_action,f0_derivatives,TransientBlunting


def test_complete_row_renewal_is_used_without_generic_fallback():
    for row,m,_ in source_rows().values():
        c=row_controls(row)
        assert c.hits==float(row['physics__cleavage_hits'])==3.2732414351776242
        assert c.tau==float(row['physics__cleavage_correlation_time_s'])==6.992153587194454e-7
        assert TransientBlunting(m,row,300,.005,c).c==c
    with pytest.raises(KeyError):row_controls({})

@pytest.mark.parametrize('xi',[.07509316036236147,math.log(2),1.])
def test_exact_row_saturated_root_and_derivatives_all_thresholds(xi):
    row,m,_=source_rows()['P25_TRANSFER_V1_RANK1'];c=row_controls(row)
    for rate in [.0005,.005,.05]:
        k=f0_root(m,1200.,rate,xi,c)
        assert math.isclose(k,xi*rate*c.tau,rel_tol=1e-10)
        assert math.isclose(f0_action(m,k,1200.,rate,c),xi,rel_tol=1e-10)
        ak,at=f0_derivatives(m,k,1200.,rate,xi,c)
        assert abs(ak-1)<1e-10 and abs(at)<1e-10


def test_noninteger_hits_changes_unsaturated_action():
    row,m,_=source_rows()['A_NATIVE'];c=row_controls(row)
    exact=f0_action(m,10,300,.005,c)
    generic=f0_action(m,10,300,.005,replace(c,hits=3.,tau=1e-6))
    assert not math.isclose(exact,generic,rel_tol=1e-3)
    k=f0_root(m,300,.005,math.log(2),c);ak,at=f0_derivatives(m,k,300,.005,math.log(2),c)
    fd=(math.log(f0_root(m,300.05,.005,math.log(2),c))-math.log(f0_root(m,299.95,.005,math.log(2),c)))/.1
    assert math.isclose(-at/ak,fd,rel_tol=5e-4,abs_tol=2e-7)


def test_original_gate_identity():
    root=Path(__file__).resolve().parents[1]/'artifacts'
    old=json.loads((root/'retained_controls_monotonic_forward/monotonic_forward_configuration.json').read_text())
    new=json.loads((root/'row_renewal_monotonic_forward/monotonic_forward_configuration.json').read_text())
    for key in ['accessibility_rules','topology_rules','derivative_relative_tolerance','derivative_absolute_tolerance']:
        assert new[key]==old[key]


def test_stiff_analytical_solve_fails_closed_and_releases_budget(monkeypatch):
    from scripts.analyze_row_renewal_forward_v10230 import TransientBlunting as Bounded
    row,m,_=source_rows()['P25_TRANSFER_V1_RANK1']
    model=Bounded(m,row,375.,.05,row_controls(row))
    model.max_state_evaluations=1
    def stalled(self,Kend,rtol=None):
        for _ in range(3):self.state(Kend,np.zeros(2))
        pytest.fail('stalled solve escaped the budget')
    monkeypatch.setattr(TransientBlunting,'solve',stalled)
    with pytest.raises(ValueError,match='budget exhausted'):model.solve(.001)
    assert not model._budget_active
    assert np.isfinite(model.state(.001,np.zeros(2))[0])


def test_pre_budget_cache_must_repeat_current_admission(monkeypatch):
    import scripts.analyze_row_renewal_forward_v10230 as audit
    calls=[]
    def reject_prior(result,row,manifest):
        calls.append(result)
        return dict(result,F1_status='STATE_CLOSURE_UNAVAILABLE')
    monkeypatch.setattr(audit,'qualify_independent_f1',reject_prior)
    old={'F1_status':'FIRST_PASSAGE'}
    new=audit.qualify_cached_result(old,{},None)
    assert new['F1_status']=='STATE_CLOSURE_UNAVAILABLE'
    assert old['F1_status']=='FIRST_PASSAGE'
    assert new['analytical_work_budget']==100000
    assert audit.qualify_cached_result(new,{},None) is new
    assert len(calls)==1


def test_saturated_action_does_not_admit_unstable_blunting_radius():
    from scripts.analyze_row_renewal_forward_v10230 import condition,qualify_independent_f1
    row,m,_=source_rows()['P25_TRANSFER_V1_RANK1']
    raw=condition('P25_TRANSFER_V1_RANK1',row,m,600.,.05,math.log(2),'MEDIAN_LN2')
    assert next(r for r in raw['records'] if r['tier']=='F1_EMISSION_BLUNTING')['status']=='FIRST_PASSAGE'
    admitted=qualify_independent_f1(raw,row,m)
    f1=next(r for r in admitted['records'] if r['tier']=='F1_EMISSION_BLUNTING')
    assert f1['K_FP'] is None and 'blunting-radius disagreement' in f1['reason']
    best=next(r for r in admitted['records'] if r['tier']=='BEST_CURRENT_MONOTONIC_FORWARD')
    assert best['selected_tier']=='F0_INTRINSIC_OPENING'
