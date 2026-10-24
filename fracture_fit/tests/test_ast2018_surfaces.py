from dataclasses import dataclass

import pytest

from experimental_fits.ast2018.surfaces import surface_diagnostics


@dataclass(frozen=True)
class Surface:
    G00_eV:float=2.0
    gT_eV_per_K:float=-.001
    sigc0_Pa:float=3e9
    sT_Pa_per_K:float=1e6
    exp_a:float=.5
    exp_n:float=2.0
    floor_fraction:float=.1
    Tref_K:float=481.33
    floor_min_eV:float=1e-4
    floor_max_fraction:float=.95


def test_executed_derivatives_at_fixed_physical_stress():
    s=Surface();sigma=2e9;T=400.0
    d=surface_diagnostics(s,sigma,T,1e12)
    hT=1e-3;hS=1e4
    derivative_T=(surface_diagnostics(s,sigma,T+hT,1e12)["G_eV"]-
                  surface_diagnostics(s,sigma,T-hT,1e12)["G_eV"])/(2*hT)
    derivative_S=(surface_diagnostics(s,sigma+hS,T,1e12)["G_eV"]-
                  surface_diagnostics(s,sigma-hS,T,1e12)["G_eV"])/(2*hS)
    assert d["S_eV_per_K"] == pytest.approx(-derivative_T,rel=1e-6)
    assert d["V_eV_per_Pa"] == pytest.approx(-derivative_S,rel=1e-6)
    assert d["H_eV"] == pytest.approx(d["G_eV"]+T*d["S_eV_per_K"])


def test_floor_branch_and_zero_stress_limit():
    s=Surface(exp_n=2.0)
    d=surface_diagnostics(s,0,400,1e12)
    assert d["V_eV_per_Pa"]==0
    assert d["G_eV"]==pytest.approx(s.G00_eV+s.gT_eV_per_K*(400-s.Tref_K))
    assert d["S_eV_per_K"]==pytest.approx(-s.gT_eV_per_K)
