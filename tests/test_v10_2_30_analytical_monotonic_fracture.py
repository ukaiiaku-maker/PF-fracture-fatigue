import csv
from dataclasses import replace
import math
from pathlib import Path

import numpy as np
from scipy.special import gammainc

from arrhenius_fracture.analytical_monotonic_fracture_v10230 import (
    MonotonicControls,
    _linear_positive_advance,
    cooperative_log_sensitivity,
    cooperative_rate,
    finite_difference_sensitivity,
    f0_implicit_sensitivities,
    solve_first_passage,
    stress_channels,
)
from arrhenius_fracture.material_manifest import MaterialManifest


ROOT = Path(__file__).resolve().parents[1]


def _native():
    registry = ROOT / "runs/A_native_plus_8PT_fatigue_v1/A_native_plus_8PT_registry.csv"
    row = next(r for r in csv.DictReader(registry.open()) if r["option_key"] == "A_NATIVE")
    manifest_path = next((ROOT / "runs/A_native_plus_8PT_fatigue_v1/developed/n80/A_NATIVE").glob(
        "DK_*/selected_material_manifest_v10_2_22.csv"
    ))
    return MaterialManifest.from_csv(manifest_path), row


def test_exact_cooperative_gamma_renewal_and_log_sensitivity():
    raw = np.geomspace(1.0e-6, 1.0e12, 101)
    tau = 1.0e-6
    got = cooperative_rate(raw, 3.0, tau)
    np.testing.assert_allclose(got, gammainc(3.0, raw * tau) / tau, rtol=2e-15)
    step = 1.0e-6
    fd = (
        np.log(cooperative_rate(raw * np.exp(step), 3.0, tau))
        - np.log(cooperative_rate(raw * np.exp(-step), 3.0, tau))
    ) / (2.0 * step)
    np.testing.assert_allclose(cooperative_log_sensitivity(raw, 3.0, tau), fd,
                               rtol=2e-5, atol=2e-7)


def test_opening_cleavage_and_emission_channels_are_separate():
    channels = stress_channels(20.0e6, 1.0e-6, 2.0e6, 1.0e9, 0.5)
    assert channels["sigma_open_Pa"] > channels["sigma_cleavage_Pa"]
    assert channels["sigma_emission_Pa"] != channels["sigma_cleavage_Pa"]
    assert math.isclose(
        channels["sigma_cleavage_Pa"], 18.0e6 / math.sqrt(2.0 * math.pi * 1.0e-6)
    )


def test_f0_first_passage_action_closes_and_endpoint_converges():
    manifest, row = _native()
    coarse = solve_first_passage(
        manifest, row, 300.0, "F0",
        MonotonicControls(dK_MPa_sqrt_m=0.02, Kmax_MPa_sqrt_m=30.0),
        return_history=True,
    )
    fine = solve_first_passage(
        manifest, row, 300.0, "F0",
        MonotonicControls(dK_MPa_sqrt_m=0.01, Kmax_MPa_sqrt_m=30.0),
    )
    assert coarse["first_passage_reached"]
    assert coarse["cleavage_action"] == 1.0
    assert abs(coarse["K_init_MPa_sqrt_m"] - fine["K_init_MPa_sqrt_m"]) < 0.03
    assert coarse["no_crack_advance_translation_before_first_event"]


def test_temperature_and_loading_rate_centered_derivatives_are_finite():
    manifest, row = _native()
    controls = MonotonicControls(dK_MPa_sqrt_m=0.04, Kmax_MPa_sqrt_m=30.0)
    dT = finite_difference_sensitivity(
        manifest, row, 300.0, "F0", controls, "temperature_K"
    )
    drate = finite_difference_sensitivity(
        manifest, row, 300.0, "F0", controls, "loading_rate_MPa_sqrt_m_s"
    )
    assert math.isfinite(dT)
    assert math.isfinite(drate)
    assert drate > 0.0


def test_implicit_temperature_and_rate_tangents_match_centered_differences():
    manifest, row = _native()
    controls = MonotonicControls(dK_MPa_sqrt_m=0.005, Kmax_MPa_sqrt_m=30.0)
    tangent = f0_implicit_sensitivities(manifest, row, 300.0, controls)
    dT = finite_difference_sensitivity(
        manifest, row, 300.0, "F0", controls, "temperature_K", 5e-4
    )
    dr = finite_difference_sensitivity(
        manifest, row, 300.0, "F0", controls,
        "loading_rate_MPa_sqrt_m_s", 5e-4,
    ) * controls.loading_rate_MPa_sqrt_m_s
    assert math.isclose(tangent["temperature_K"], dT, rel_tol=0.01)
    assert math.isclose(tangent["ln_loading_rate"], dr, rel_tol=0.02)


def test_f0_recovery_when_state_coupling_disabled():
    manifest, row = _native()
    row = dict(row)
    row["rho_source0_m2"] = "0"
    controls = MonotonicControls(dK_MPa_sqrt_m=0.04, Kmax_MPa_sqrt_m=30.0)
    values = [
        solve_first_passage(manifest, row, 300.0, level, controls)["K_init_MPa_sqrt_m"]
        for level in ("F0", "F1", "F2")
    ]
    np.testing.assert_allclose(values, values[0], rtol=0.0, atol=1e-12)


def test_positive_compartment_conservation_without_losses():
    manifest, _ = _native()
    controls = MonotonicControls(retained_recovery_rate_s=0.0)
    diagnostics = {
        "emission_rate_s": 2.0,
        "blunting_transport_rate_s": 0.0,
        "encounter_rate_s": 3.0,
        "escape_rate_s": 0.0,
        "taylor_rate_s": 5.0,
    }
    state = np.array([0.0, 4.0, 7.0])
    got = _linear_positive_advance("F2", state, 0.2, diagnostics, manifest, controls)
    assert np.all(got >= 0.0)
    assert math.isclose(got[1] + got[2], state[1] + state[2] + 0.4,
                        rel_tol=2e-14, abs_tol=2e-14)


def test_two_compartment_conservation_without_losses():
    manifest, _ = _native()
    controls = MonotonicControls(retained_recovery_rate_s=0.0,
                                 f2b_exchange_rate_s=4.0)
    diagnostics = {
        "emission_rate_s": 2.0,
        "blunting_transport_rate_s": 0.0,
        "encounter_rate_s": 3.0,
        "escape_rate_s": 0.0,
        "taylor_rate_s": 5.0,
    }
    state = np.array([0.0, 4.0, 7.0, 3.0, 2.0])
    got = _linear_positive_advance("F2B", state, 0.2, diagnostics, manifest, controls)
    assert np.all(got >= 0.0)
    assert math.isclose(got[1:].sum(), state[1:].sum() + 0.4,
                        rel_tol=2e-14, abs_tol=2e-14)


def test_loading_rate_control_is_immutable_dataclass_value():
    base = MonotonicControls()
    changed = replace(base, loading_rate_MPa_sqrt_m_s=base.loading_rate_MPa_sqrt_m_s * 2)
    assert base.loading_rate_MPa_sqrt_m_s * 2 == changed.loading_rate_MPa_sqrt_m_s
    assert base.loading_rate_MPa_sqrt_m_s != changed.loading_rate_MPa_sqrt_m_s
