import numpy as np

from arrhenius_fracture.unified_mpz import UnifiedMPZState
from scripts.qste340tm_faithful_reduction import (
    PlasticParameters, _exchange_matrix, apply_cycles, cycle_operator,
    one_cycle_with_transfers, phase_operator,
)


def parameters(**updates):
    values = dict(emission_G0_298_eV=0.8, plastic_sigc_298_Pa=2e9,
                  transport_G0_298_eV=0.7, retained_recovery_per_s=1e-3,
                  initial_mobile_count=2.0, initial_retained_count=3.0)
    values.update(updates)
    return PlasticParameters(**values)


def test_scalar_exchange_is_exact_native_exchange_owner():
    mobile=np.array([[2.0]]);retained=np.array([[3.0]])
    native_m,native_r,_,_=UnifiedMPZState._exchange(mobile,retained,np.array([4.0]),np.array([1.5]),0.2)
    reduced=_exchange_matrix(4.0,1.5,0.2)@np.array([2.0,3.0])
    np.testing.assert_allclose(reduced,[native_m.item(),native_r.item()],rtol=2e-15)


def test_zero_source_phase_has_no_retained_credit_and_conserves_inventory():
    operator=phase_operator(0.0,2.0,3.0,1.0,0.4,0.2,0.1)
    before=np.array([2.0,3.0,0.7,1.0]);after=operator@before
    assert after[0]+after[1] <= before[0]+before[1]
    assert after[2] > before[2]


def test_phase_map_matches_native_owner_order_and_exchange():
    mobile=np.array([[2.0]]);retained=np.array([[3.0]]);q0=.7
    hazard,clear,encounter,release,escape,recovery,dt=4.,1.5,3.,.8,.4,.2,.1
    total=hazard+clear;decay=np.exp(-total*dt);qeq=clear/total
    integral_q=qeq*dt+(q0-qeq)*(1-decay)/total
    mobile=mobile+hazard*integral_q
    q1=qeq+(q0-qeq)*decay
    mobile,retained,_,_=UnifiedMPZState._exchange(
        mobile,retained,np.array([encounter]),np.array([release]),dt)
    retained*=np.exp(-recovery*dt)
    mobile*=np.exp(-escape*dt)
    native=np.array([mobile.item(),retained.item(),q1])
    reduced=(phase_operator(hazard,clear,encounter,release,escape,recovery,dt)
             @np.array([2.,3.,q0,1.]))[:3]
    np.testing.assert_allclose(reduced,native,rtol=3e-15,atol=2e-15)


def test_strong_escape_endpoint_survivor_is_not_direct_retained_production():
    p=parameters(emission_G0_298_eV=0.3,transport_G0_298_eV=0.3,
                 retained_recovery_per_s=0.0)
    initial=np.array([0.0,0.0,1.0])
    final,flow=one_cycle_with_transfers(45.0,0.1,initial,p,n_phase=128)
    assert flow["endpoint_mobile_survivor_diagnostic"] >= 0.0
    assert flow["retained_final"] == final[1]
    assert abs(flow["inventory_balance_residual"]) < 1e-12*max(flow["emitted"],1.0)
    assert flow["retained_final"] <= flow["emitted"]


def test_affine_cycle_power_matches_explicit_repetition():
    p=parameters();state=np.array([2.0,3.0,0.8])
    operator,_=cycle_operator(20.0,0.3,state,p,n_phase=64)
    explicit=np.r_[state,1.0]
    for _ in range(17):explicit=operator@explicit
    powered=apply_cycles(operator,state,17.0)
    np.testing.assert_allclose(powered,explicit[:3],rtol=2e-11,atol=1e-12)
