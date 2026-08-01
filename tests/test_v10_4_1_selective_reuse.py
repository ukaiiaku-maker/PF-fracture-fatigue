from __future__ import annotations

import json
from pathlib import Path

from arrhenius_fracture.reuse_v1040_v1041 import (
    CASE_SCHEMA,
    SCHEMA,
    materialize_reuse_cases,
    max_rate_correction_bound,
    sha256_file,
    verify_materialized_reuse,
)


ROW = {
    "option_key": "test_option",
    "candidate_id": "test_candidate",
    "Tref_K": "481.33",
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


def test_rate_correction_bound_includes_exact_zero_stress_fixed_point():
    result = max_rate_correction_bound(
        ROW,
        temperature_K=1000.0,
        rho_min_m2=5e12,
        rho_max_m2=1e14,
        max_stress_Pa=30e9,
        stress_points=101,
        rho_points=17,
    )
    assert result["zero_stress_old_rate_max_s"] > 0.0
    assert result["zero_stress_new_rate_max_s"] == 0.0
    assert result["maximum_absolute_equivalent_rate_difference_s"] > 0.0
    assert result["stress_at_maximum_difference_Pa"] >= 0.0


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_materialized_reuse_is_explicit_hash_checked_and_non_destructive(tmp_path: Path):
    source = tmp_path / "source_case"
    source.mkdir()
    (source / "COMPLETE").write_text("complete\n")
    _write_json(source / "stage3_case_status.json", {"complete": True})
    _write_json(
        source / "v10_2_27_case_contract.json",
        {
            "option": "test_option",
            "temperature_K": 900.0,
            "seed": 123,
            "target_extension_um": 1000.0,
            "theta_deg": 0.0,
            "loading_rate_factor": 1.0,
        },
    )
    _write_json(
        source / "v10_2_27_paper_four_class_parameter_transfer.json", {}
    )
    (source / "command.sh").write_text("#!/usr/bin/env bash\necho source\n")
    _write_json(source / "v10_2_30_hazard_energy_gate_audit.json", {})
    _write_json(source / "stochastic_avalanche_geometry_events.json", [])
    _write_json(
        source / "v10_4_bulk_peierls_taylor_coupling_audit.json",
        {
            "bulk_kinetics_model": "emission_derived_peierls_taylor_multihit",
            "runtime_diagnostics": {"local_plastic_work_nonnegative": True},
        },
    )
    _write_json(
        source / "v10_4_bulk_coupled_model_audit.json",
        {
            "bulk_plasticity_mode": "full_field",
            "v10_2_30_code_path_modified": False,
        },
    )

    required = [
        "COMPLETE",
        "stage3_case_status.json",
        "v10_2_27_case_contract.json",
        "v10_2_27_paper_four_class_parameter_transfer.json",
        "command.sh",
        "v10_2_30_hazard_energy_gate_audit.json",
        "stochastic_avalanche_geometry_events.json",
        "v10_4_bulk_peierls_taylor_coupling_audit.json",
        "v10_4_bulk_coupled_model_audit.json",
    ]
    hashes = {name: sha256_file(source / name) for name in required}
    record = {
        "schema": CASE_SCHEMA,
        "approved": True,
        "decision": "reuse",
        "source_case": str(source),
        "option": "test_option",
        "temperature_K": 900.0,
        "seed": 123,
        "source_commit": "old",
        "target_commit": "new",
        "source_model": "v10.4.0_one_way_arrhenius_bulk_slip",
        "target_model": "v10.4.1_detailed_balance_forward_minus_reverse",
        "acceptance_tolerance_cumulative_equivalent_strain": 1e-6,
        "upper_bound_cumulative_equivalent_strain_difference": 1e-8,
        "source_required_file_sha256": hashes,
        "reasons": [],
    }
    audit = tmp_path / "reuse.json"
    _write_json(
        audit,
        {
            "schema": SCHEMA,
            "theta_deg": 0.0,
            "records": [record],
        },
    )

    destination = tmp_path / "target"
    manifest = materialize_reuse_cases(audit, destination)
    assert manifest["materialized_case_count"] == 1
    case = destination / "test_option" / "T900K_th0_seed123"
    assert (case / "COMPLETE").is_symlink()
    assert not (case / "v10_2_27_case_contract.json").is_symlink()
    contract = json.loads((case / "v10_2_27_case_contract.json").read_text())
    assert contract["case_execution_mode"] == "audited_v10_4_0_reuse"
    assert contract["zero_stress_net_plastic_rate_exactly_zero"] is True
    model = json.loads((case / "v10_4_bulk_coupled_model_audit.json").read_text())
    assert model["source_one_way_arrhenius_rate_used_as_net_slip"] is True
    verified = verify_materialized_reuse(case)
    assert verified["approved"] is True
    assert not (source / "v10_4_1_reuse_audit.json").exists()


def test_launcher_contains_native_or_audited_reuse_verifier():
    text = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_v10_4_bulk_rate_orientation_launcher.py"
    ).read_text()
    assert "verify_materialized_reuse" in text
    assert "audited_v10_4_0_reuse" in text
    assert "selective_reuse_permitted_with_case_audit" in text
