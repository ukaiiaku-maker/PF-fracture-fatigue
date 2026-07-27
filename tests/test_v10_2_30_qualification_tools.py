import json
from pathlib import Path
import subprocess

from scripts.analyze_v10_2_30_energy_gated_qualification import summarize_case


ROOT = Path(__file__).resolve().parents[1]


def test_qualification_runner_has_valid_shell_syntax():
    completed = subprocess.run(
        ["bash", "-n", "scripts/run_v10_2_30_three_deltaK_energy_gate_qualification.sh"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_qualification_analyzer_validates_hazard_energy_event(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    control = {
        "parameter_option": "v913_paper_dbtt01_0202500_persistent_sites",
        "target_deltaK_MPa_sqrt_m": 10.0,
        "R": 0.1,
        "frequency_Hz": 1000.0,
        "cleavage_hazard_seed": 1720,
        "Gc0_athermal_active": False,
        "independent_fracture_energy_active": False,
        "fixed_deltaK_exact_within_relative_1e-12": True,
        "persistent_site_source": True,
        "finite_source_inventory": False,
        "source_refresh": False,
        "cleavage_first_passage_rate_changed": False,
        "continuum_energy_comparison_diagnostic_only": True,
        "continuum_energy_comparison_affects_hazard": False,
        "zero_length_hazard_attempts_consumed": True,
    }
    (case / "v10_2_30_fixed_deltaK_control.json").write_text(
        json.dumps(control)
    )
    (case / "qualification_case.json").write_text(
        json.dumps(
            {
                "temperature_K": 300.0,
                "deltaK_fraction": 0.8,
                "energy_gate_trial_fraction": 0.1,
            }
        )
    )
    event = {
        "inserted": True,
        "stochastic_proposed_event_length_m": 5.0e-6,
        "event_advance_m": 4.0e-6,
        "orientation_gamma_relative": 1.2,
        "direction_audit": {"source": "continuous_cubic_competition"},
        "athermal_Gc_used": False,
        "independent_toughness_floor_used": False,
        "mesh_resolved_commit_required": True,
        "trial_rows": [
            {
                "trial_length_m": 4.0e-6,
                "energy_residual_J_per_m": 1.0e-8,
                "energy_tolerance_J_per_m": 1.0e-12,
                "topology_changed": True,
            }
        ],
    }
    (case / "hazard_energy_gated_events_v10_2_30.json").write_text(
        json.dumps([event])
    )
    (case / "stochastic_avalanche_geometry_events.json").write_text(
        json.dumps([{"x0": 0.5e-3, "x1": 0.503e-3}])
    )
    (case / "kinetic_tip_cell_audit_v101.json").write_text(
        json.dumps(
            {
                "records": [
                    {"loading_mode": "cyclic", "cycles_consumed": 1000.0}
                ]
            }
        )
    )
    (case / "summary.json").write_text(
        json.dumps([{"geometry_projected_extension_m": 3.0e-6}])
    )

    row = summarize_case(case / "v10_2_30_fixed_deltaK_control.json")
    assert row["pass"] is True
    assert row["committed_events"] == 1
    assert row["truncated_events"] == 1
    assert row["cycles_consumed"] == 1000.0
    assert row["projected_extension_um"] == 3.0
    assert row["path_extension_um"] == 4.0
    assert row["path_tortuosity"] == 4.0 / 3.0
    assert row["projected_da_dN_m_per_cycle"] == 3.0e-9
    assert row["path_ds_dN_m_per_cycle"] == 4.0e-9
    assert row["first_passage_rate_preserved"] is True
    assert row["continuum_energy_diagnostic_only"] is True
