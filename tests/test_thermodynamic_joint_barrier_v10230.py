import math
import numpy as np
from scripts.analyze_forward_temperature_v10230 import source_rows
from arrhenius_fracture.thermodynamic_joint_barrier_v10230 import (
    KB_EV_PER_K, EV_J, candidate_surface_from_parameters,
)

def parameters():
    return dict(cleavage_zero_target_eV=1.5,guard_log10_q_low=-5.,guard_exponent=4.,
        cleavage_entropy_zero_kB=-10.,cleavage_entropy_active_kB=5.,cleavage_entropy_infinity_kB=2.,
        emission_entropy_active_kB=35.,emission_entropy_infinity_kB=5.)

def surfaces():
    _,m,_=source_rows()['P25_TRANSFER_V1_RANK1']
    return candidate_surface_from_parameters(m,parameters(),4e9,7e9,3e9),m

def test_reference_surface_and_entropy_coordinates():
    (c,e),m=surfaces();p=parameters()
    assert math.isclose(float(c.G_eV(0.,300)),p['cleavage_zero_target_eV'])
    assert math.isclose(float(c.entropy_kB(0.,300)),p['cleavage_entropy_zero_kB'])
    assert math.isclose(float(c.entropy_kB(7e9,300)),p['cleavage_entropy_active_kB'])
    assert math.isclose(float(e.entropy_kB(3e9,300)),p['emission_entropy_active_kB'])
    assert np.allclose(e.G_eV(np.array([0.,3e9,10e9]),300),m.emission.values_eV(np.array([0.,3e9,10e9]),300))

def test_thermodynamic_identity_and_maxwell_finite_difference():
    (c,e),_=surfaces();s=np.array([1e8,2e9,7e9]);T=750.;dt=.02;ds=2e4
    for surf in (c,e):
        G=surf.G_eV(s,T);S=surf.entropy_kB(s,T);H=surf.enthalpy_eV(s,T)
        assert np.allclose(H,G+T*KB_EV_PER_K*S,rtol=1e-13)
        Sfd=-(surf.G_eV(s,T+dt)-surf.G_eV(s,T-dt))/(2*dt)/KB_EV_PER_K
        assert np.allclose(S,Sfd,rtol=2e-8,atol=2e-8)
        Vfd=-(surf.G_eV(s+ds,T)-surf.G_eV(s-ds,T))/(2*ds)*EV_J
        assert np.allclose(surf.activation_volume_m3(s,T),Vfd,rtol=2e-7,atol=1e-35)
        dvdT=(surf.activation_volume_m3(s,T+dt)-surf.activation_volume_m3(s,T-dt))/(2*dt)
        dSds=(surf.entropy_kB(s+ds,T)-surf.entropy_kB(s-ds,T))/(2*ds)*KB_EV_PER_K*EV_J
        assert np.allclose(dvdT,dSds,rtol=2e-6,atol=1e-36)

def test_no_independent_activation_volume_or_entropy_prefactor():
    fields=set(parameters())
    assert not any('volume' in f or 'prefactor' in f for f in fields)
