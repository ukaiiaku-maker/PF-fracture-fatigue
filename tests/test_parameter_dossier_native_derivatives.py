import csv
from pathlib import Path

import numpy as np
import pytest

from arrhenius_fracture.material_manifest import ExpFloorBarrier
from arrhenius_fracture.parameter_dossier import _barrier, build_parameter_dossier
from arrhenius_fracture.prospective_campaign import REGISTRY


ROOT = Path(__file__).resolve().parents[1]


def _dbtt_emission():
    with (ROOT / REGISTRY).open(newline="") as handle:
        row = next(r for r in csv.DictReader(handle)
                   if r["candidate_id"] == "v913_zeroD_sobol_0202500")
    return _barrier(row, "emit")


def test_recorded_dbtt_counterexample_uses_complete_native_surface():
    barrier = _dbtt_emission()
    sigma = 266302770.76474476
    diag = barrier.diagnostic(sigma, 950.0)
    assert diag["barrier_eV"] == pytest.approx(5.337216317262275, abs=2e-14)
    assert diag["dG0_dT_eV_K"] == pytest.approx(0.0061668124636635)
    assert diag["partial_G_partial_T_fixed_stress_state_eV_K"] == pytest.approx(
        -0.0031065770305, abs=5e-12)
    assert diag["activation_entropy_fixed_stress_state_eV_K"] > 0.0
    assert diag["partial_G_partial_T_fixed_sigma_over_sigc_eV_K"] != pytest.approx(
        diag["partial_G_partial_T_fixed_stress_state_eV_K"])


@pytest.mark.parametrize("sigma", [2.6630277076474476e8, 1.0e9])
def test_fixed_stress_thermal_derivative_converges_without_mutating_surface(sigma):
    barrier = _dbtt_emission()
    before = barrier.__dict__.copy()
    analytic = barrier.diagnostic(sigma, 950.0)["partial_G_partial_T_fixed_stress_state_eV_K"]
    errors = []
    for h in (1.0, 0.25, 0.0625):
        fd = (float(barrier.values_eV(sigma, 950+h))
              - float(barrier.values_eV(sigma, 950-h))) / (2*h)
        errors.append(abs(fd-analytic))
    assert errors[2] < errors[1] < errors[0]
    assert barrier.__dict__ == before


def test_fixed_normalized_stress_derivative_is_separate_convention():
    barrier = _dbtt_emission()
    T, ratio = 950.0, 0.125
    sigma = ratio * (barrier.sigc0_Pa + barrier.sT_Pa_per_K*(T-barrier.Tref_K))
    diag = barrier.diagnostic(sigma, T)
    h = 1e-2
    def along(temp):
        sigc = barrier.sigc0_Pa + barrier.sT_Pa_per_K*(temp-barrier.Tref_K)
        return float(barrier.values_eV(ratio*sigc, temp))
    fd = (along(T+h)-along(T-h))/(2*h)
    assert fd == pytest.approx(diag["partial_G_partial_T_fixed_sigma_over_sigc_eV_K"], abs=2e-10)


def test_piecewise_floor_and_sign_gate_are_reported_without_fake_derivatives():
    absolute = ExpFloorBarrier(1.0, 0.0, 1e9, 0.0, 1.0, 0.5, 0.0,
                               floor_min_eV=0.2)
    d = absolute.diagnostic(-1.0, 500.0)
    assert d["floor_branch"] == "absolute_floor"
    assert d["stress_gate_branch"] == "negative_stress_gate"
    assert d["partial_G_partial_sigma_eV_Pa"] == 0.0
    zero = absolute.diagnostic(0.0, 500.0)
    assert zero["partial_G_partial_sigma_eV_Pa"] is None  # n<1 one-sided singularity


def test_corrected_dossier_is_native_v2_and_has_mechanism_specific_semantics():
    dossier = build_parameter_dossier(ROOT, temperatures=(950.0,), stress_fractions=(0.1,))
    assert dossier["schema"].endswith("/2")
    dbtt = next(p for p in dossier["parents"] if p["family"] == "DBTT")
    multiplicity = dbtt["physical_multiplicity"]
    assert "source_sites_per_system" not in multiplicity
    assert multiplicity["inherited_reference_source_sites_per_system"] == pytest.approx(
        141.0590567476921
    )
    assert "inactive for persistent-site" in multiplicity["reference_field_runtime_status"]
    transport = dbtt["native_transport_consumer"]
    assert transport["peierls_stress_projection"] == pytest.approx(1.0 / np.sqrt(3.0))
    assert transport["taylor_stress_projection"] == pytest.approx(1.0 / np.sqrt(3.0))
    assert transport["completion_law"]["mode"] == "mean_net_rate_over_density_order"
    assert transport["completion_law"]["window_s"] is None
    assert transport["completion_law"]["single_rate_cap_s"] is None
    for item in dbtt["mechanisms"]:
        assert item["native_callable"].endswith("values_eV/rate/diagnostic")
        assert "generic one-mark" not in item["completion_law"] or item["mechanism"] != "cleavage"
        assert "dG0_dT_eV_K" in item
        assert "partial_G_partial_T_fixed_stress_state_eV_K" in item
