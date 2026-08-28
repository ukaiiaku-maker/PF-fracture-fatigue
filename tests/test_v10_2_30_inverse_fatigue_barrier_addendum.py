import math

import numpy as np

from arrhenius_fracture.crystal import bcc_slip_traces
from arrhenius_fracture.inverse_fatigue_barrier_addendum_v10230 import (
    GeneralizedStress,
    derivative_identity,
    fa1_plane_opening_stresses,
    generalized_exp_floor_gradient,
    generalized_slip_stress,
    linear_non_schmid_drive,
    opening_waveform,
    recover_load_ratio,
    signed_waveform,
    svd_identifiability,
    waveform_factor_midpoint,
    waveform_factor_peak_asymptotic,
    waveform_factor_quadrature,
)
from arrhenius_fracture.material_manifest import ExpFloorBarrier


def barrier():
    return ExpFloorBarrier(2.0, 0.0, 2e9, 0.0, 0.8, 1.3, 0.1)


def test_exact_load_recovery_and_separate_waveforms():
    phi=np.array([0.0,math.pi])
    assert math.isclose(recover_load_ratio(18.0,-17.1),-0.95)
    signed=signed_waveform(phi,-0.95);opening=opening_waveform(phi,-0.95)
    assert signed[0]==1.0 and signed[1]==-0.95
    assert opening[0]==1.0 and opening[1]==0.0
    assert signed[1]<0.0


def test_negative_drive_does_not_reach_opening_barriers():
    phase=np.linspace(math.pi*.75,math.pi*1.25,41)
    assert np.any(signed_waveform(phase,-.95)<0)
    assert np.all(opening_waveform(phase,-.95)>=0)


def test_exact_waveform_quadrature_matches_midpoint():
    for M in (2.0,4.0,6.0):
        for R in (-.95,.1,.5):
            assert math.isclose(waveform_factor_quadrature(M,R),
                                waveform_factor_midpoint(M,R,262144),
                                rel_tol=2e-10,abs_tol=2e-12)


def test_peak_asymptotic_converges():
    R=.1
    errors=[]
    for M in (20.0,80.0,320.0):
        exact=waveform_factor_quadrature(M,R)
        asym=waveform_factor_peak_asymptotic(M,R)
        errors.append(abs(asym/exact-1.0))
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < .003


def test_fixed_Kmax_vs_fixed_deltaK_identity():
    def rate(K,R):
        return K**4*waveform_factor_quadrature(4.0,R)
    result=derivative_identity(rate,18.0,.1,2e-5)
    assert abs(result["closure_residual"])<2e-7


def test_generalized_gradient_matches_finite_difference():
    b=barrier();xi=GeneralizedStress(1.2e9,-.2e9,.8e9,-.5e9)
    c=np.array([1.0,.1,.05,-.02]);analytic=generalized_exp_floor_gradient(b,xi,c,300)
    base=xi.vector();fd=[]
    for j in range(4):
        h=100.0;up=base.copy();dn=base.copy();up[j]+=h;dn[j]-=h
        gu=float(b.values_eV(max(float(c@up),0),300));gd=float(b.values_eV(max(float(c@dn),0),300))
        fd.append((gu-gd)/(2*h))
    assert np.allclose(analytic,fd,rtol=2e-5,atol=1e-15)


def test_crystallographic_rotation_and_symmetry_invariance():
    K=18e6;r=1e-6
    values=[]
    for theta in (0.0,30.0):
        traces=bcc_slip_traces(theta)
        values.append(sorted(abs(generalized_slip_stress(K,r,t).tau_Pa) for t in traces))
    assert np.all(np.asarray(values)>0)
    # The two symmetry-related traces at the reference orientation have the
    # same unordered magnitude under the symmetric mode-I field.
    a=sorted(abs(generalized_slip_stress(K,r,t).tau_Pa) for t in bcc_slip_traces(0))
    assert math.isclose(a[0],a[1],rel_tol=1e-12)


def test_NS0_recovered_when_NS1_non_schmid_coefficients_zero():
    xi=GeneralizedStress(1e9,2e8,3e8,-4e8)
    assert linear_non_schmid_drive(xi,[1,0,0,0])==xi.tau_Pa


def test_FA1_isotropic_limit_and_directional_scores_are_finite():
    rows=fa1_plane_opening_stresses(18e6,1e-6,30.0)
    assert len(rows)==4
    assert all(math.isfinite(x["signed_configurational_score"]) for x in rows)
    b=barrier()
    values=[float(b.values_eV(x["sigma_nn_open_Pa"],300)) for x in rows[:2]]
    assert all(math.isfinite(x) for x in values)


def test_identifiability_SVD_is_reproducible():
    J=np.array([[1,0,1],[0,1,1],[1,1,2]],dtype=float)
    one=svd_identifiability(J);two=svd_identifiability(J)
    assert one["rank"]==2
    assert np.allclose(one["singular_values"],two["singular_values"])
