from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture.bulk_plasticity_manifest_v104 import (
    BulkManifestParameters,
    BulkPlasticityCoupling,
    KB_EV_PER_K,
)
from arrhenius_fracture import sharp_front_v10_4_bulk_peierls_taylor as entry


ROW = {
    "option_key": "test_option",
    "candidate_id": "test_candidate",
    "rho_forest_floor_m2": "5e12",
    "emit_G00_eV": "2.0",
    "emit_gT_eV_per_K": "0.002",
    "emit_sigc0_GPa": "3.0",
    "emit_sT_GPa_per_K": "-0.0005",
    "emit_exp_a": "0.1",
    "emit_exp_n": "1.2",
    "emit_floor_frac": "0.02",
    "peierls_H0_eV": "0.5",
    "peierls_activation_entropy_kB": "-10",
    "peierls_exp_a": "0.2",
    "peierls_exp_n": "1.1",
    "peierls_stress_fraction": "0.5773502691896258",
    "peierls_nu0_s": "1e12",
    "taylor_H0_eV": "0.8",
    "taylor_activation_entropy_kB": "-5",
    "taylor_exp_a": "0.3",
    "taylor_exp_n": "1.4",
    "taylor_stress_fraction": "0.5773502691896258",
    "taylor_nu0_s": "1e11",
    "taylor_corr_rho_c_m2": "1e14",
    "taylor_corr_scale": "2.5",
}


def test_exact_row_maps_to_bulk_configuration():
    parameters = BulkManifestParameters.from_row(ROW)
    cfg = SimpleNamespace()
    mapped = parameters.configure(cfg)

    assert cfg.bulk_kinetics_model == "emission_derived_peierls_taylor_multihit"
    assert cfg.thermo_consistency_mode == "time_cone"
    assert cfg.bulk_mult_frac == 1.0
    assert cfg.tip_source_rho_per_emit == 0.0
    assert cfg.rho_transport_c == 0.0
    assert cfg.exhaustion_enabled is False
    assert cfg.mobile_rho_floor == 5e12
    assert cfg.pt_peierls_energy_ratio == pytest.approx(0.25)
    assert cfg.pt_taylor_energy_ratio == pytest.approx(0.4)
    assert cfg.pt_peierls_entropy_ratio == pytest.approx(
        (10.0 * KB_EV_PER_K) / 0.002
    )
    assert cfg.pt_taylor_m_scale == pytest.approx(2.5)
    assert mapped["pt_emit_sigc0_Pa"] == pytest.approx(3e9)


def test_coupled_wrapper_calls_original_and_records_nonnegative_work():
    parameters = BulkManifestParameters.from_row(ROW)
    coupling = BulkPlasticityCoupling(parameters)
    cfg = SimpleNamespace()

    def original(ep, rho, sigma, mat, T, dt, disl_cfg, return_info=False):
        assert disl_cfg.bulk_kinetics_model == "emission_derived_peierls_taylor_multihit"
        rho_out = rho + 1.0
        rate = np.full_like(rho, 2.0)
        if return_info:
            return ep, rho_out, rate, {
                "dep_eq_accepted_gp": np.full_like(rho, 1e-5),
                "dWp_accepted_gp": np.ones_like(rho),
                "dep_eq_limited_gp": np.zeros_like(rho),
                "pt_peierls_rate_gp": np.full_like(rho, 3.0),
                "pt_taylor_completion_rate_gp": np.full_like(rho, 4.0),
                "pt_series_rate_gp": np.full_like(rho, 5.0),
                "thermo_mode": "time_cone",
            }
        return ep, rho_out, rate

    wrapped = coupling.wrap(original)
    ep = np.zeros((3, 2))
    rho = np.full(2, 5e12)
    sigma = np.zeros((3, 2))
    result = wrapped(ep, rho, sigma, object(), 900.0, 1.0, cfg, True)

    assert len(result) == 4
    payload = coupling.diagnostics.payload()
    assert payload["plasticity_update_calls"] == 1
    assert payload["maximum_series_rate_s"] == pytest.approx(5.0)
    assert payload["local_plastic_work_nonnegative"] is True
    assert payload["thermodynamic_modes"] == ["time_cone"]


def test_v104_args_force_audited_full_field_contract():
    args = ["--bulk-plasticity-mode", "tip_only", "--rho-transport-c", "3"]
    entry._prepare_v104_args(args)
    assert entry._stage3._option_value(args, "--bulk-plasticity-mode") == "full_field"
    assert float(entry._stage3._option_value(args, "--bulk-mult-frac")) == 1.0
    assert float(entry._stage3._option_value(args, "--tip-source-rho-per-emit")) == 0.0
    assert float(entry._stage3._option_value(args, "--rho-transport-c")) == 0.0


def test_v104_rejects_fatigue_and_finite_content():
    with pytest.raises(SystemExit, match="monotonic-only"):
        entry._prepare_v104_args(["--fatigue-cycles"])
    with pytest.raises(SystemExit, match="finite bulk-content"):
        entry._prepare_v104_args(["--exhaustion"])


def test_stage3_adapter_preserves_all_other_validity_checks():
    args = ["--bulk-plasticity-mode", "full_field"]
    seen = {}

    def original(local_args):
        seen["mode"] = entry._stage3._option_value(
            local_args, "--bulk-plasticity-mode"
        )
        return 3621

    seed = entry._bulk_capable_stage3_validity(original, args)
    assert seed == 3621
    assert seen["mode"] == "tip_only"
    assert entry._stage3._option_value(args, "--bulk-plasticity-mode") == "full_field"


def test_launcher_builder_generates_full_field_v104_contract(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    import importlib.util

    path = root / "scripts" / "build_v10_4_bulk_rate_orientation_launcher.py"
    spec = importlib.util.spec_from_file_location("v104_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source = (
        root / "scripts" / "run_v10_2_28_paper_four_class_theta30_1000um.sh"
    ).read_text()
    generated = module.transform(source)
    assert "sharp_front_v10_4_bulk_peierls_taylor_audited" in generated
    assert "--bulk-plasticity-mode full_field" in generated
    assert "--bulk-mult-frac 1" in generated
    assert "--tip-source-rho-per-emit 0" in generated
    assert "--rho-transport-c 0" in generated
    assert "v10_4_bulk_peierls_taylor_coupling_audit.json" in generated
    assert "v10_4_bulk_coupled_model_audit.json" in generated
    assert "v10.4_bulk_peierls_taylor_orientation_rate_campaign_v1" in generated
    assert "--bulk-plasticity-mode tip_only" not in generated


def test_emission_derived_series_rate_is_finite_and_stress_activated():
    from arrhenius_fracture.emission_derived_plasticity import (
        EmissionDerivedPeierlsTaylorModel,
        config_from_dislocation_config,
    )

    parameters = BulkManifestParameters.from_row(ROW)
    cfg = SimpleNamespace(
        pt_emit_floor_min_eV=1e-4,
        pt_emit_floor_max_frac=0.95,
        pt_emit_Tref_K=481.33,
        pt_taylor_renewal_time_s=1e-9,
        pt_mobile_fraction=0.01,
        pt_mobile_saturation_density_m2=1e14,
        pt_jump_fraction=1.0,
        pt_jump_length_min_m=2.5e-10,
        pt_taylor_phi_max=20.0,
    )
    parameters.configure(cfg)
    model = EmissionDerivedPeierlsTaylorModel(
        config_from_dislocation_config(cfg)
    )
    rho = np.full(3, 5e12)
    low = model.rates(np.array([0.0, 1e8, 2e8]), rho, 900.0, 2.74e-10)
    high = model.rates(np.array([1e9, 2e9, 3e9]), rho, 900.0, 2.74e-10)

    assert np.all(np.isfinite(low["equivalent_plastic_rate_s"]))
    assert np.all(np.isfinite(high["equivalent_plastic_rate_s"]))
    assert np.all(low["equivalent_plastic_rate_s"] >= 0.0)
    assert np.max(high["equivalent_plastic_rate_s"]) >= np.max(
        low["equivalent_plastic_rate_s"]
    )
    assert np.all(high["series_rate_s"] <= high["peierls_rate_s"] * (1 + 1e-12))
    assert np.all(
        high["series_rate_s"]
        <= high["taylor_completion_rate_s"] * (1 + 1e-12)
    )
