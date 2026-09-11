import numpy as np
import pytest
from arrhenius_fracture.material_manifest import ExpFloorBarrier
from arrhenius_fracture.prospective_paris_candidate_engine_v10230 import build_frozen_manifest
from scripts.forward_temperature_model_v10230 import surface,f0_action,f0_root,f0_derivatives,Controls,accessibility,topology,TransientBlunting

@pytest.mark.parametrize('sigma',[0.,1e7,1e9,8e9,30e9])
@pytest.mark.parametrize('T',[300.,700.,1200.])
def test_piecewise_surface_exact_and_derivatives(sigma,T):
    b=ExpFloorBarrier(1.8,-1e-4,2e9,1e5,.4,2.2,.22)
    d=surface(b,sigma,T); assert np.isclose(d['G'],b.values_eV(sigma,T),rtol=1e-14)
    h=1e-4
    d1=-(b.values_eV(sigma*np.exp(h),T)-b.values_eV(sigma*np.exp(-h),T))/(2*h)
    d2=-(b.values_eV(sigma*np.exp(h),T)-2*b.values_eV(sigma,T)+b.values_eV(sigma*np.exp(-h),T))/h**2
    dt=(b.values_eV(sigma,T+.01)-b.values_eV(sigma,T-.01))/.02
    assert np.isclose(d['D1'],d1,rtol=1e-6,atol=1e-8)
    assert np.isclose(d['D2'],d2,rtol=1e-5,atol=1e-7)
    assert np.isclose(d['DT'],dt,rtol=1e-6,atol=1e-10)


def test_tiny_positive_root_is_not_numerical_floor():
    from dataclasses import replace
    m,_=build_frozen_manifest('A_NATIVE');m=replace(m,cleavage=replace(m.cleavage,G00_eV=.01))
    for rate in [.0005,.005,.05]:
        k=f0_root(m,1200.,rate,.075)
        assert np.isclose(k,.075*rate*1e-6,rtol=1e-10)
        assert np.isclose(f0_action(m,k,1200.,rate),.075,rtol=1e-10)


def test_cumulative_implicit_derivative_independent_roots():
    m,_=build_frozen_manifest('A_NATIVE');T=300.;rate=.005;xi=.075
    k=f0_root(m,T,rate,xi);ak,at=f0_derivatives(m,k,T,rate,xi)
    fd=(np.log(f0_root(m,T+.05,rate,xi))-np.log(f0_root(m,T-.05,rate,xi)))/.1
    assert np.isclose(-at/ak,fd,rtol=5e-4,atol=2e-7)

@pytest.mark.parametrize('label',['PEAK_T_FORWARD_PREDICTION','WEAK_T_FORWARD_PREDICTION'])
def test_saturation_never_promoted(label):
    rows=[dict(K_FP=k,thermal_derivative=d,accessibility='RENEWAL_CEILING_DOMINATED') for k,d in [(1e-9,1e-2),(2e-9,0),(1e-9,-1e-2)]]
    assert topology(rows,True) not in [label]


def test_root_failure_is_not_accessible():
    assert accessibility(0,0,0,1e-9,.1)[0]=='NUMERICAL_LOWER_BOUND_DOMINATED'
    assert accessibility(0,0,0,None,0)[0]=='RAMP_CENSORED'


def test_unavailable_state_does_not_become_weakT():
    rows=[dict(K_FP=1.,thermal_derivative=0.,accessibility='FRACTURE_RESPONSE_ACCESSIBLE')]*3
    assert topology(rows,False)=='UNCLASSIFIED_FORWARD_PREDICTION'


def test_f1_integrator_failure_fails_closed(monkeypatch):
    from types import SimpleNamespace
    from scripts import forward_temperature_model_v10230 as model
    m,_=build_frozen_manifest('A_NATIVE')
    monkeypatch.setattr(model,'solve_ivp',lambda *a,**k:SimpleNamespace(success=False,message='Required step size is less than spacing between numbers.',y=np.zeros((3,2))))
    with pytest.raises(ValueError,match='STATE_CLOSURE_UNAVAILABLE'):
        TransientBlunting(m,{'rho_source0_m2':1e15},600,.005).solve(1.)


def test_verifier_rejects_parameter_mutation():
    from scripts.verify_forward_temperature_v10230 import validate_registry
    from scripts.analyze_forward_temperature_v10230 import source_rows
    from arrhenius_fracture.prospective_paris_candidate_engine_v10230 import digest
    rows=source_rows();reg=[dict(candidate_id=k,complete_parameter_vector=dict(v[0]),complete_row_sha256=digest(v[0])) for k,v in rows.items()]
    reg[1]['complete_parameter_vector']['cleave_G00_eV']='99'
    with pytest.raises(ValueError,match='parameter changed'):validate_registry(reg,rows)


def test_verifier_rejects_saturated_peak_and_fabricated_plot():
    from scripts.verify_forward_temperature_v10230 import validate_classification,validate_series
    with pytest.raises(ValueError,match='classification'):
        validate_classification('PEAK_T_FORWARD_PREDICTION',[dict(accessibility='RENEWAL_CEILING_DOMINATED')])
    with pytest.raises(ValueError,match='figure coordinates'):
        validate_series(dict(x_column='T',y_column='K',x=[300],y=[10]),[dict(T='300',K='1e-9')])


def test_root_endpoint_stiffness_cannot_retain_F1_or_full_state(monkeypatch):
    from scripts import analyze_forward_temperature_v10230 as analysis
    base=dict(candidate_id='P25_TRANSFER_V1_RANK1',temperature_K=925.,Kdot=.005,threshold_action=.07509316036236147,threshold_mode='EXPONENTIAL_CRN_1720_ENGINE1',status='FIRST_PASSAGE',K_FP=3.7546580181180735e-10)
    result=dict(records=[dict(base,tier=t) for t in ['F0_INTRINSIC_OPENING','F1_EMISSION_BLUNTING','BEST_CURRENT_MONOTONIC_FORWARD']],traces=[dict(tier='F1_EMISSION_BLUNTING')],descriptors=[],derivatives=[])
    def fail(*a,**k):raise ValueError('Required step size is less than spacing between numbers.')
    monkeypatch.setattr(analysis.TransientBlunting,'solve',fail)
    m,_=build_frozen_manifest('A_NATIVE')
    final=analysis.qualify_independent_f1(result,{'rho_source0_m2':1e15},m)
    assert final['records'][1]['status']=='STATE_CLOSURE_UNAVAILABLE'
    assert final['records'][1]['K_FP'] is None
    assert final['records'][2]['best_scope']=='F0_ONLY_F1_UNAVAILABLE'
    assert not final['traces']
    assert result['records'][1]['status']=='FIRST_PASSAGE' # raw evidence preserved
