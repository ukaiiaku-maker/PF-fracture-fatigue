from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture import sharp_front_v10_4_bulk_peierls_taylor as entry
from arrhenius_fracture.bulk_plasticity_manifest_v104 import (
    BulkManifestParameters,
    BulkPlasticityCoupling,
    KB_EV_PER_K,
)
from arrhenius_fracture.emission_derived_plasticity import (
    EmissionDerivedPeierlsTaylorModel,
    config_from_dislocation_config,
)


ROW = {
    "option_key": "test_option", "candidate_id": "test_candidate",
    "Tref_K": "481.33", "rho_forest_floor_m2": "5e12",
    "emit_G00_eV": "2.0", "emit_gT_eV_per_K": "0.002",
    "emit_sigc0_GPa": "3.0", "emit_sT_GPa_per_K": "-0.0005",
    "emit_exp_a": "0.1", "emit_exp_n": "1.2", "emit_floor_frac": "0.02",
    "peierls_H0_eV": "0.5", "peierls_activation_entropy_kB": "-10",
    "peierls_exp_a": "0.2", "peierls_exp_n": "1.1",
    "peierls_stress_fraction": "0.5773502691896258", "peierls_nu0_s": "1e12",
    "taylor_H0_eV": "0.8", "taylor_activation_entropy_kB": "-5",
    "taylor_exp_a": "0.3", "taylor_exp_n": "1.4",
    "taylor_stress_fraction": "0.5773502691896258", "taylor_nu0_s": "1e11",
    "taylor_corr_rho_c_m2": "1e14", "taylor_corr_scale": "2.5",
}


def test_exact_row_maps_to_native_bulk_configuration():
    parameters = BulkManifestParameters.from_row(ROW)
    cfg = SimpleNamespace()
    mapped = parameters.configure(cfg)
    assert cfg.bulk_kinetics_model == "emission_derived_peierls_taylor_multihit"
    assert cfg.thermo_consistency_mode == "time_cone"
    assert cfg.bulk_mult_frac == 1.0
    assert cfg.tip_source_rho_per_emit == 0.0
    assert cfg.rho_transport_c == 0.0
    assert cfg.exhaustion_enabled is False
    assert cfg.pt_peierls_energy_ratio == pytest.approx(0.25)
    assert cfg.pt_peierls_entropy_ratio == pytest.approx((10.0 * KB_EV_PER_K) / 0.002)
    assert mapped["pt_emit_sigc0_Pa"] == pytest.approx(3e9)


def test_native_bulk_rates_are_finite_activated_and_zero_without_drive():
    parameters = BulkManifestParameters.from_row(ROW)
    cfg = SimpleNamespace(
        pt_emit_floor_min_eV=1e-4, pt_emit_floor_max_frac=0.95,
        pt_taylor_renewal_time_s=1e-9, pt_mobile_fraction=0.01,
        pt_mobile_saturation_density_m2=1e14,
        pt_mobile_density_floor_m2=5e12, pt_jump_fraction=1.0,
        pt_jump_length_min_m=2.5e-10, pt_taylor_phi_max=20.0,
        pt_equivalent_strain_factor=1.0,
    )
    parameters.configure(cfg)
    model = EmissionDerivedPeierlsTaylorModel(config_from_dislocation_config(cfg))
    rho = np.full(3, 5e12)
    zero = model.rates(np.zeros(3), rho, 900.0, 2.74e-10)
    high = model.rates(np.array([1e9, 2e9, 3e9]), rho, 900.0, 2.74e-10)
    assert np.array_equal(zero["equivalent_plastic_rate_s"], np.zeros(3))
    assert np.all(np.isfinite(high["equivalent_plastic_rate_s"]))
    assert np.max(high["equivalent_plastic_rate_s"]) > 0.0
    assert np.all(high["series_rate_s"] <= high["peierls_rate_s"] * (1 + 1e-12))


def test_exact_row_direct_shapes_and_entropy_are_used_by_runtime_model():
    parameters = BulkManifestParameters.from_row(ROW)
    cfg = SimpleNamespace(
        pt_emit_floor_min_eV=1e-4, pt_emit_floor_max_frac=0.95,
        pt_taylor_renewal_time_s=1e-9, pt_mobile_fraction=0.01,
        pt_mobile_saturation_density_m2=1e14,
        pt_mobile_density_floor_m2=5e12, pt_jump_fraction=1.0,
        pt_jump_length_min_m=2.5e-10, pt_taylor_phi_max=20.0,
        pt_equivalent_strain_factor=1.0,
    )
    parameters.configure(cfg)
    model = EmissionDerivedPeierlsTaylorModel(config_from_dislocation_config(cfg))
    assert model.peierls.G00_eV == pytest.approx(float(ROW["peierls_H0_eV"]))
    assert model.peierls.gT_eV_per_K == pytest.approx(
        -float(ROW["peierls_activation_entropy_kB"]) * KB_EV_PER_K
    )
    assert model.peierls.alpha == pytest.approx(float(ROW["peierls_exp_a"]))
    assert model.peierls.exponent == pytest.approx(float(ROW["peierls_exp_n"]))
    assert model.taylor.alpha == pytest.approx(float(ROW["taylor_exp_a"]))
    assert model.taylor.exponent == pytest.approx(float(ROW["taylor_exp_n"]))


def test_historical_window_jump_floor_and_phi_cap_cannot_change_native_rates():
    parameters = BulkManifestParameters.from_row(ROW)
    cfg = SimpleNamespace(
        pt_emit_floor_min_eV=1e-4, pt_emit_floor_max_frac=0.95,
        pt_taylor_renewal_time_s=0.0, pt_mobile_fraction=0.01,
        pt_mobile_saturation_density_m2=1e14,
        pt_mobile_density_floor_m2=5e12, pt_jump_fraction=1.0,
        pt_jump_length_min_m=0.0, pt_taylor_phi_max=float("inf"),
        pt_equivalent_strain_factor=1.0,
    )
    parameters.configure(cfg)
    stress = np.array([2e8, 8e8])
    rho = np.array([1e8, 5e12])
    native = EmissionDerivedPeierlsTaylorModel(
        config_from_dislocation_config(cfg)).rates(stress, rho, 950.0, 2.74e-10)
    cfg.pt_taylor_renewal_time_s = 1.0
    cfg.pt_jump_length_min_m = 1.0
    cfg.pt_taylor_phi_max = 1.0
    compatibility = EmissionDerivedPeierlsTaylorModel(
        config_from_dislocation_config(cfg)).rates(stress, rho, 950.0, 2.74e-10)
    for key in ("equivalent_plastic_rate_s", "peierls_rate_s",
                "taylor_completion_rate_s", "taylor_m_eff"):
        np.testing.assert_array_equal(compatibility[key], native[key])


def test_v104_contract_forces_full_field_and_rejects_finite_or_fatigue():
    args = ["--bulk-plasticity-mode", "tip_only", "--rho-transport-c", "3"]
    entry._prepare_v104_args(args)
    assert entry._stage3._option_value(args, "--bulk-plasticity-mode") == "full_field"
    assert float(entry._stage3._option_value(args, "--bulk-mult-frac")) == 1.0
    assert float(entry._stage3._option_value(args, "--tip-source-rho-per-emit")) == 0.0
    assert float(entry._stage3._option_value(args, "--rho-transport-c")) == 0.0
    with pytest.raises(SystemExit, match="monotonic-only"):
        entry._prepare_v104_args(["--fatigue-cycles"])
    with pytest.raises(SystemExit, match="finite bulk-content"):
        entry._prepare_v104_args(["--exhaustion"])


def test_coupling_uses_real_update_signature_and_records_diagnostics():
    coupling = BulkPlasticityCoupling(BulkManifestParameters.from_row(ROW))
    cfg = SimpleNamespace()

    def original(ep, rho, sigma, mat, T, dt, plast_model, disl_cfg, return_info=False):
        assert disl_cfg is cfg
        info = {"dep_eq_accepted_gp": np.full_like(rho, 1e-5),
                "dWp_accepted_gp": np.ones_like(rho),
                "dep_eq_limited_gp": np.zeros_like(rho),
                "pt_peierls_rate_gp": np.full_like(rho, 3.0),
                "pt_taylor_completion_rate_gp": np.full_like(rho, 4.0),
                "pt_series_rate_gp": np.full_like(rho, 5.0),
                "thermo_mode": "time_cone"}
        result = (ep, rho + 1.0, np.full_like(rho, 2.0))
        return (*result, info) if return_info else result

    wrapped = coupling.wrap(original)
    wrapped(np.zeros((3, 2)), np.full(2, 5e12), np.zeros((3, 2)), object(),
            900.0, 1.0, object(), cfg, True)
    payload = coupling.diagnostics.payload()
    assert payload["plasticity_update_calls"] == 1
    assert payload["maximum_series_rate_s"] == pytest.approx(5.0)
    assert payload["local_plastic_work_nonnegative"] is True
