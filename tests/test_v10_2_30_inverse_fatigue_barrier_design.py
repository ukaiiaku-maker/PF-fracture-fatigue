import math
import csv
from pathlib import Path

import numpy as np
import pytest
from scipy.special import gammainc

from arrhenius_fracture.inverse_fatigue_barrier_design_v10230 import (
    RenewalControls,
    centered_log_slope,
    cooperative_Q,
    cooperative_rate_from_raw,
    cycle_growth_and_slope,
    exact_power_law_target_barrier,
    exact_target_instantaneous_rate,
    exp_floor_drop_derivative_eV,
    exp_floor_rare_event_max_slope,
    integrate_slope_profile,
    invert_cooperative_rate_to_barrier,
    logistic_rare_event_barrier,
    slope_profile,
    waveform_factor_C,
    waveform_fraction,
    ideal_mode_I_drive_factors,
    b1_event_conditioned_emission,
    B1Controls,
)
from arrhenius_fracture.material_manifest import ExpFloorBarrier, MaterialManifest


def barrier():
    return ExpFloorBarrier(
        G00_eV=1.8433725545182824,
        gT_eV_per_K=0.0,
        sigc0_Pa=2.0952987279742956e9,
        sT_Pa_per_K=0.0,
        alpha=0.41946418629859783,
        exponent=2.9225786491297185,
        floor_fraction=0.22289373425133724,
    )


def test_exact_waveform_semantics():
    n = 16384
    h = waveform_fraction(-0.95, n)
    assert h.min() == 0.0
    assert h.max() == pytest.approx(1.0, rel=2e-8)
    tensile = waveform_fraction(0.1, n)
    expected_midpoint_minimum = 0.55 - 0.45 * math.cos(math.pi / n)
    assert tensile.min() == pytest.approx(expected_midpoint_minimum, rel=1e-13)


def test_fixed_R_Kmax_and_deltaK_log_slopes_are_equal():
    R, M = 0.1, 4.0
    g = lambda K: K ** M
    K = 18.0
    m_k = centered_log_slope(g, K)
    delta = (1.0 - R) * K
    m_delta = centered_log_slope(lambda d: g(d / (1.0 - R)), delta)
    assert m_delta == pytest.approx(m_k, rel=1e-10)


@pytest.mark.parametrize("x", [1e-4, 0.02, 0.8, 7.0])
def test_cooperative_Q_matches_finite_difference(x):
    m, h = 3.0, 1e-6
    numeric = (math.log(gammainc(m, x * math.exp(h))) -
               math.log(gammainc(m, x * math.exp(-h)))) / (2 * h)
    assert float(cooperative_Q(m, x)) == pytest.approx(numeric, rel=2e-8)


def test_cooperative_Q_asymptotic_limits():
    assert float(cooperative_Q(3.0, 1e-12)) == pytest.approx(3.0)
    assert float(cooperative_Q(3.0, 1e4)) == pytest.approx(0.0, abs=1e-14)


def test_inverse_gamma_round_trip():
    controls = RenewalControls()
    rate = np.geomspace(1e-8, 0.8 / controls.tau_s, 100)
    G = invert_cooperative_rate_to_barrier(
        rate, hits=controls.hits, tau_s=controls.tau_s,
        attempt_frequency_s=1e12, temperature_K=controls.temperature_K,
    )
    raw = 1e12 * np.exp(-G / (8.617333262145e-5 * controls.temperature_K))
    recovered = cooperative_rate_from_raw(raw, controls.hits, controls.tau_s)
    assert np.allclose(recovered, rate, rtol=2e-11, atol=0.0)


@pytest.mark.parametrize("M", [2.0, 4.0, 6.0])
def test_exact_power_law_is_recovered_by_forward_cycle_quadrature(M):
    controls = RenewalControls(n_phase=16384)
    R, Kref, gref, K = 0.1, 18.0, 4.473410023231299e-7, 20.0
    h = waveform_fraction(R, controls.n_phase)
    sigma = K * 1e6 * h / math.sqrt(2 * math.pi * controls.radius_m)
    target_rate = exact_target_instantaneous_rate(
        sigma, M=M, R=R, K_ref_MPa_sqrt_m=Kref,
        g_ref_m_per_cycle=gref, controls=controls,
    )
    growth = controls.event_length_m * np.mean(target_rate) / controls.frequency_Hz
    assert growth == pytest.approx(gref * (K / Kref) ** M, rel=2e-9)
    G = exact_power_law_target_barrier(
        sigma, M=M, R=R, K_ref_MPa_sqrt_m=Kref,
        g_ref_m_per_cycle=gref, attempt_frequency_s=1e12, controls=controls,
    )
    assert np.all(np.isfinite(G))


@pytest.mark.parametrize("kind,parameters", [
    ("CONSTANT_SLOPE", {"M": 4.0}),
    ("PIECEWISE_CONSTANT_SLOPE", {"edges": [-0.2, 0.2], "values": [1.0, 4.0, 0.5]}),
    ("LOGISTIC_SLOPE_WINDOW", {"M_P": 5.0, "s_on": -0.2, "s_off": 0.2,
                                "w_on": 0.04, "w_off": 0.05}),
    ("USER_TABULATED_SLOPE_PROFILE", {"s": [-0.5, 0.0, 0.5], "M": [1.0, 5.0, 2.0]}),
])
def test_arbitrary_slope_profile_recovery(kind, parameters):
    s = np.linspace(-0.6, 0.6, 20001)
    log_rate = integrate_slope_profile(kind, s, parameters, math.log(2.0))
    recovered = np.gradient(log_rate, s, edge_order=2)
    target = slope_profile(kind, s, parameters)
    # Exclude discontinuities and interpolation endpoints from the derivative check.
    mask = np.ones_like(s, dtype=bool)
    mask[:10] = False; mask[-10:] = False
    if kind == "PIECEWISE_CONSTANT_SLOPE":
        for edge in parameters["edges"]:
            mask &= np.abs(s - edge) > 5e-4
    assert np.max(np.abs(recovered[mask] - target[mask])) < 2e-3


def test_logistic_barrier_has_bounded_plateaus_and_declared_drop():
    controls = RenewalControls()
    values = logistic_rare_event_barrier(
        np.array([-20.0, 20.0]), G0_eV=2.0, M_P=4.0,
        s_on=-0.2, s_off=0.3, w_on=0.05, w_off=0.05,
        controls=controls,
    )
    expected_drop = (8.617333262145e-5 * controls.temperature_K * 4.0 /
                     controls.hits * (0.3 - (-0.2)))
    assert values[0] == pytest.approx(2.0, abs=1e-12)
    assert values[0] - values[1] == pytest.approx(expected_drop, rel=1e-10)


def test_exp_floor_logarithmic_derivative_matches_finite_difference():
    b = barrier(); sigma = np.geomspace(0.3e9, 10e9, 30); h = 1e-3
    numeric = -(b.values_eV(sigma * np.exp(-2*h), 300.0)
                - 8*b.values_eV(sigma * np.exp(-h), 300.0)
                + 8*b.values_eV(sigma * np.exp(h), 300.0)
                - b.values_eV(sigma * np.exp(2*h), 300.0)) / (12*h)
    analytic = exp_floor_drop_derivative_eV(b, sigma, 300.0)
    resolved = analytic > 1e-6
    assert np.allclose(analytic[resolved], numeric[resolved], rtol=1e-7, atol=1e-11)
    assert np.max(np.abs(numeric[~resolved] - analytic[~resolved])) < 1e-9


def test_exp_floor_maximum_slope_estimate_occurs_at_u_one():
    b = barrier(); controls = RenewalControls()
    sigma = b.sigc0_Pa * (1.0 / b.alpha) ** (1.0 / b.exponent)
    drop = float(exp_floor_drop_derivative_eV(b, sigma, controls.temperature_K))
    direct = controls.hits * drop / (8.617333262145e-5 * controls.temperature_K)
    assert direct == pytest.approx(exp_floor_rare_event_max_slope(
        b, controls.temperature_K, controls.hits), rel=1e-12)


@pytest.mark.parametrize("R", [-0.95, 0.1, 0.5])
@pytest.mark.parametrize("K", [12.0, 15.0, 18.0, 21.0, 24.3])
def test_exact_cycle_quadrature_derivative_matches_finite_difference(R, K):
    b = barrier(); controls = RenewalControls(n_phase=8192)
    analytic = cycle_growth_and_slope(b, K, R, controls)["local_slope"]
    numeric = centered_log_slope(
        lambda value: cycle_growth_and_slope(b, value, R, controls)["da_dN"], K,
        relative_step=2e-6,
    )
    assert analytic == pytest.approx(numeric, rel=1e-5)


def test_R_dependent_waveform_factor_has_known_tensile_value():
    assert waveform_factor_C(2.0, 0.1) == pytest.approx(0.40375, rel=1e-10)


def test_ideal_mode_I_channel_factors_follow_current_30_degree_BCC_traces():
    factors=ideal_mode_I_drive_factors(30.0,0.5)
    assert factors == pytest.approx([0.7663204807600036,0.2566048122925707],rel=1e-12)


def test_B1_event_map_is_fresh_stochastic_and_event_conditioned():
    root=Path(__file__).resolve().parents[1]
    row=next(r for r in csv.DictReader((root/"runs/A_native_plus_8PT_fatigue_v1/A_native_plus_8PT_registry.csv").open()) if r["option_key"]=="A_NATIVE")
    path=next((root/"runs/A_native_plus_8PT_fatigue_v1/developed/n80/A_NATIVE").glob("DK_*/selected_material_manifest_v10_2_22.csv"))
    result=b1_event_conditioned_emission(MaterialManifest.from_csv(path),row,18.0,0.1,
        controls=B1Controls(n_phase=32,burn_events=1,sample_events=1,maximum_cycles=10000,hazard_seed=1720))
    assert result["fixed_point_converged"]
    assert result["da_dN"]>0
    assert result["mean_event_length_m"]!=pytest.approx(5e-6)
    assert result["PT_coordinates"]=="held_at_A_NATIVE_not_in_B1_state"
