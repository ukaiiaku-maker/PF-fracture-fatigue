import math

import pytest

from experimental_fits.ast2018.mechanics import (
    HistoryPoint, Specimen, basic_J_at_failure, continuous_onset_K,
    lefm_Kq_at_failure, observe_like_ast,
)


def specimen():
    return Specimen(4e-6, 6e-6, 12.5e-6, 2.25e-6, 410e9, 0.28, 1e-3)


def test_elastic_loading_has_zero_plastic_area_and_J():
    s = specimen()
    h = [HistoryPoint(i, i*1e-6, i*1e-3, s.a0_m, 0, 0) for i in range(4)]
    obs = observe_like_ast(h, s)
    for p in obs:
        assert p.A_pl_J == pytest.approx(0, abs=1e-20)
        assert p.J_continuous_N_per_m == pytest.approx(
            (p.K_el_current_MPa_sqrt_m*1e6)**2/s.Eprime_Pa)
        assert p.K_nominal_MPa_sqrt_m == p.K_el_current_MPa_sqrt_m
        assert p.relative_stiffness == 1


def test_compliance_derivative_matches_energy_release_relation():
    s = specimen()
    a = s.a0_m+0.1e-6
    h = 1e-10
    numerical = (s.elastic_compliance(a+h)-s.elastic_compliance(a-h))/(2*h)
    exact = 2*s.B_m*s.geometry_factor(a)**2/s.Eprime_Pa
    assert numerical == pytest.approx(exact, rel=1e-7)


def test_rate_uses_initial_compliance_and_nominal_geometry():
    s = specimen()
    v = s.initial_displacement_rate(0.2)
    rate = (v/s.initial_compliance_m_per_N)*s.geometry_factor(s.a0_m)/1e6
    assert rate == pytest.approx(0.2)


def test_onset_requires_actual_extension_support():
    s = specimen()
    h = [HistoryPoint(0,0,0,s.a0_m,0,0),
         HistoryPoint(1,1e-6,1e-3,s.a0_m,0,0),
         HistoryPoint(1,1e-6,0.9e-3,s.a0_m+0.1e-6,0,0,event=True)]
    assert math.isnan(continuous_onset_K(observe_like_ast(h,s),s))


def test_final_cleavage_observers_require_physical_failure():
    s = specimen()
    h = [HistoryPoint(0,0,0,s.a0_m,0,0),
         HistoryPoint(1,1e-6,1e-3,s.a0_m,0,0)]
    obs = observe_like_ast(h,s)
    assert math.isnan(basic_J_at_failure(obs,s))
    assert math.isnan(lefm_Kq_at_failure(obs,s))
    h[-1] = HistoryPoint(1,1e-6,1e-3,s.a0_m,0,0,physical_failure=True)
    obs = observe_like_ast(h,s)
    assert basic_J_at_failure(obs,s) == pytest.approx(obs[-1].K_el_current_MPa_sqrt_m)
    assert lefm_Kq_at_failure(obs,s) == pytest.approx(obs[-1].K_nominal_MPa_sqrt_m)


def test_known_linear_plastic_displacement_gives_correct_work_area():
    s=specimen();fraction=.2;u=2e-6
    h=[HistoryPoint(0,0,0,s.a0_m,0,0)]
    for i in (1,2):
        ui=u*i/2
        upl=fraction*ui
        h.append(HistoryPoint(i,ui,s.equilibrium_force(ui,upl,s.a0_m),
                              s.a0_m,upl,0))
    p=observe_like_ast(h,s)[-1]
    expected=.5*fraction*(1-fraction)*u*u/s.initial_compliance_m_per_N
    assert p.A_pl_J==pytest.approx(expected)
    elastic_J=(p.K_el_current_MPa_sqrt_m*1e6)**2/s.Eprime_Pa
    assert p.J_continuous_N_per_m-elastic_J==pytest.approx(
        2*expected/(s.B_m*(s.W_m-s.a0_m)))
