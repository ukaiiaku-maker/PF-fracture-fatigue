import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from arrhenius_fracture.two_scale_virtual_ct_v10230 import (
    LocalSurfaceOutOfDomain,
    LogPchipRateSurface,
    fixed_load_kmax_MPa,
    integrate_virtual_ct,
    protocol_kmax,
    shedding_kmax_MPa,
)
from arrhenius_fracture.virtual_ct_v10230 import (
    CompactTensionGeometry,
    ct_geometry_factor,
    ct_k_from_load_pa_sqrt_m,
    ct_load_from_k_pa_sqrt_m,
)


RS = (-0.95, 0.1, 0.5)
ROOT = Path(__file__).parents[1]


def synthetic_rows(constant=False):
    rows = []
    for R in RS:
        for K in (12.0, 15.0, 18.0, 24.0):
            rate = 2e-8 if constant else 1e-11 * K**4 * math.exp(0.2 * R)
            rows.append({"R": R, "Kmax_MPa_sqrt_m": K, "developed_da_dN": rate})
    return rows


def test_log_pchip_recovers_nodes_is_positive_monotone_and_has_analytic_derivative():
    surface = LogPchipRateSurface(synthetic_rows(), option="A_NATIVE", version="v0")
    for row in synthetic_rows():
        ev = surface.evaluate(row["Kmax_MPa_sqrt_m"], row["R"])
        assert float(ev.rate_m_per_cycle) == pytest.approx(row["developed_da_dN"], rel=2e-14)
    K = np.linspace(12, 24, 501)
    for R in RS:
        ev = surface.evaluate(K, R)
        assert np.all(ev.rate_m_per_cycle > 0)
        assert np.all(np.diff(ev.rate_m_per_cycle) >= 0)
        h = 1e-5
        numerical = (
            np.log(surface.evaluate(18 * math.exp(h), R).rate_m_per_cycle)
            - np.log(surface.evaluate(18 * math.exp(-h), R).rate_m_per_cycle)
        ) / (2 * h)
        assert float(ev.local_slope[len(K) // 2]) == pytest.approx(float(numerical), rel=2e-5)


def test_log_pchip_does_not_overshoot_neighbor_bounds_and_R_is_log_conservative():
    surface = LogPchipRateSurface(synthetic_rows(), option="A_NATIVE", version="v0")
    for R in RS:
        for lo, hi in zip((12, 15, 18), (15, 18, 24)):
            values = surface.evaluate(np.linspace(lo, hi, 101), R).rate_m_per_cycle
            endpoints = surface.evaluate(np.asarray([lo, hi]), R).rate_m_per_cycle
            assert values.min() >= endpoints.min() * (1 - 1e-14)
            assert values.max() <= endpoints.max() * (1 + 1e-14)
    mid = surface.evaluate(18, 0.3).rate_m_per_cycle
    bounds = surface.evaluate(18, np.asarray([0.1])[0]).rate_m_per_cycle, surface.evaluate(18, 0.5).rate_m_per_cycle
    assert min(bounds) <= mid <= max(bounds)


def test_surface_fails_closed_outside_K_and_R_domain():
    surface = LogPchipRateSurface(synthetic_rows(), option="A_NATIVE", version="v0")
    for K, R in ((11.999, 0.1), (24.001, 0.1), (18, -0.951), (18, 0.501)):
        with pytest.raises(LocalSurfaceOutOfDomain, match="LOCAL_SURFACE_OUT_OF_DOMAIN"):
            surface.evaluate(K, R)
    prediction = surface.prospective_anchor_prediction(24.3, 0.1)
    assert prediction["prospective_endpoint_extension"]
    assert not prediction["admitted_for_virtual_integration"]
    assert surface.K_max == 24.0


@pytest.mark.parametrize(
    "x,reference",
    [
        (0.45, 8.33958568327015),
        (0.50, 9.659078631008239),
        (0.55, 11.364286292888009),
        (0.60, 13.654145726628231),
        (0.65, 16.856887755643644),
    ],
)
def test_CT_geometry_reference_values(x, reference):
    assert ct_geometry_factor(x) == pytest.approx(reference, rel=2e-14)


def test_load_K_inverse_fixed_load_path_and_no_tip_radius_argument():
    geometry = CompactTensionGeometry(0.01, 0.0025, 0.0045)
    p = ct_load_from_k_pa_sqrt_m(12e6, geometry, geometry.initial_crack_m)
    for x in np.linspace(0.45, 0.65, 9):
        K = ct_k_from_load_pa_sqrt_m(p, geometry, x * geometry.width_m) / 1e6
        assert K == pytest.approx(float(fixed_load_kmax_MPa(x)), rel=2e-14)
    assert "radius" not in inspect.signature(ct_k_from_load_pa_sqrt_m).parameters


def test_shedding_endpoints_constant_normalized_gradient_and_ranges():
    x = np.linspace(0.45, 0.65, 101)
    K = shedding_kmax_MPa(x)
    assert K[0] == pytest.approx(24)
    assert K[-1] == pytest.approx(12)
    gradient = np.gradient(np.log(K), x)
    assert np.ptp(gradient) < 1e-10
    R = -0.95
    Kmin = R * K
    assert np.allclose((K - Kmin) / K, 1.95)
    assert np.allclose((K - np.maximum(Kmin, 0)) / K, 1.0)


def test_constant_K_life_and_integration_convergence_are_exact_and_monotone():
    rate = 2e-8
    surface = LogPchipRateSurface(synthetic_rows(constant=True), option="A_NATIVE", version="v1")
    geometry = CompactTensionGeometry(0.01, 0.0025, 0.0045)
    rows, metadata = integrate_virtual_ct(surface, R=0.1, protocol="CONSTANT_KMAX", geometry=geometry)
    expected = (0.65 - 0.45) * geometry.width_m / rate
    assert metadata["total_cycles"] == pytest.approx(expected, rel=1e-12)
    assert metadata["maximum_relative_convergence_difference"] < 1e-5
    assert np.all(np.diff([row["cumulative_cycles"] for row in rows]) > 0)
    assert np.allclose([row["Kmax_MPa_sqrt_m"] for row in rows], 18)
    assert all(row["Pmin_N"] / row["Pmax_N"] == pytest.approx(0.1) for row in rows)


def test_fixed_endpoint_matches_mandated_anchor_and_protocol_names_fail_closed():
    assert fixed_load_kmax_MPa(0.65) == pytest.approx(24.255719738394, rel=2e-14)
    with pytest.raises(ValueError):
        protocol_kmax("UNKNOWN", np.asarray([0.5]))


def test_anchor_controller_is_fresh_exact_bounded_and_freezes_predictions_first():
    controller = (ROOT / "scripts/complete_v10_2_30_two_scale_virtual_CT.py").read_text()
    worker = (ROOT / "scripts/run_v10_2_30_two_scale_anchor_worker.py").read_text()
    for phrase in (
        'ANCHOR_K = (13.5, 21.0, 24.3)',
        '"V10230_HIGH_CYCLE_EXPLICIT_ONLY": "1"',
        '"fresh_virgin_start": True',
        '"resume": False',
        '1 <= args.workers <= 3',
        'A_NATIVE_anchor_predictions_v0.csv',
        'prediction_frozen_unix_ns',
        'held_out_constant_load_rows_used_for_fit',
        'interrupted physical anchor cannot resume',
        'reconcile_prephysics_launch_failures',
        'kinetic_tip_cell_audit_v101.json',
        'high_cycle_live_checkpoint.json',
        'prephysics quarantine collision',
    ):
        assert phrase in controller
    assert "fresh anchor worker refuses an existing result path" in worker
    assert '"RESUME" in key.upper() or "RESTART" in key.upper()' in worker


def test_analysis_holds_dynamic_controls_out_and_reports_only_defined_K_ranges():
    analysis = (ROOT / "scripts/analyze_v10_2_30_two_scale_virtual_CT.py").read_text()
    for phrase in (
        'used_for_surface_fit": False',
        'constant_load_rows_used_for_fit": 0',
        'closure_corrected_deltaK_reported": False',
        'tip_radius_used": False',
        'FULL_DELTAK',
        'TENSILE_DELTAK',
        'KMAX18_SEED_SENSITIVITY',
        'probabilistic_confidence_interval": False',
        'STATE_HISTORY_REQUIRED',
    ):
        assert phrase in analysis


def test_conditional_PT_controller_resolves_endpoint_without_resume_or_extrapolated_admission():
    controller = (ROOT / "scripts/complete_v10_2_30_two_scale_PT_anchors.py").read_text()
    for phrase in (
        "FIRST_CONDITIONAL",
        "ENDPOINT_DOMAIN_RESOLUTION",
        "FIXED_LOAD_ENDPOINT_DOMAIN_AMBIGUITY",
        '"fresh_virgin_start":True',
        '"resume":False',
        'V10230_HIGH_CYCLE_EXPLICIT_ONLY',
        "conditional PT anchors cannot precede nine terminal native anchors",
        "PT first conditional interpolation requires midpoint refinement",
        "interrupted conditional PT physics cannot resume",
    ):
        assert phrase in controller
