import json
from pathlib import Path

import pytest

from scripts import run_pf_current_source_branching_completion_pair_v5_4_2 as launch


BUNDLE = Path(__file__).parents[1] / (
    "analysis_outputs/pf_current_source_branching_completion_execution_v5_4_2"
)


def test_v5_4_2_pins_reviewed_execution_and_accepted_inputs():
    assert launch.EXECUTION_COMMIT == "ae9a06d8c42287428e917baef143b0c1142cefd8"
    assert launch.EXECUTION_TREE == "2d0edbc7c7b51d81ab5678065ca68a1cef45d134"
    assert launch.RECORD_PARENT == "040a124afb4c90befa214faa2d8aacfd06f2d95c"
    assert launch.FAMILY_SHA == "423bc3232326b8ccc3ffcca0aa6b5363c67bad2c64debee49925bf1e6413e8cb"
    assert launch.FAMILY_PHYSICS == "e0bc48eb8c3f5526877a21f8551e500046a614df8081693f340084fb73400302"
    assert launch.RESTORE_SENTINEL_SHA == "6912cd00a4e22990d6cc914e55dae7d1fdcec32a06a0d95ae12094e68e5d3516"


def test_v5_4_2_environment_base_is_exactly_pinned():
    assert launch.stable_json_hash(launch.environment_base()) == launch.ENVIRONMENT_BASE_SHA
    assert launch.environment_base()["CLEAVAGE_HAZARD_SEED"] == "3621"
    assert launch.environment_base()["ONED_V2_TP_STATE_DIAGNOSTICS"] == "events"


@pytest.mark.parametrize("maximum_fronts", (1, 2))
def test_v5_4_2_command_is_the_bounded_theta40_300um_case(maximum_fronts):
    command = launch.command(
        Path("/fresh/output"), Path("/sealed/checkpoint.json"),
        Path("/qualified/family.json"), maximum_fronts,
    )
    joined = " ".join(command)
    assert f"--maximum-fronts {maximum_fronts}" in joined
    assert "--crystal-theta-deg 40" in joined
    assert "--target-crack-extension-um 300" in joined
    assert "--da-phys 5e-6" in joined
    assert "--dU 2e-7" in joined
    assert "--dt 8.4" in joined
    assert "--save-snapshots 0" in joined
    assert "--no-plots" in command
    assert "daughter" not in joined.lower()
    assert "theta45" not in joined.lower()
    assert "1000" not in command


def test_v5_4_2_rejects_more_than_two_fronts():
    with pytest.raises(RuntimeError, match="maximum_fronts 1 and 2 only"):
        launch.command(
            Path("/fresh/output"), Path("/sealed/checkpoint.json"),
            Path("/qualified/family.json"), 3,
        )


def test_v5_4_2_counts_compute_worker_not_shell_supervisor():
    solver = (
        "/Volumes/Data/PF-sintering/.venv/bin/python "
        "scripts/pr_full_corrected_production_recover_v6_latest.py"
    )
    supervisor = "/bin/zsh -lc " + solver + " 2>&1 | tee live.log"
    assert launch.is_heavy_compute_command(solver)
    assert not launch.is_heavy_compute_command(supervisor)


def test_v5_4_2_terminal_bundle_preserves_bounded_scientific_decision():
    audit = json.loads(
        (BUNDLE / "pf_branching_completion_terminal_audit_v5_4_2.json").read_text()
    )
    assert audit["qualification"] == "PASS"
    assert audit["boundary"] == "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
    assert audit["pair_terminal_result"] == "CORRECTED_THETA40_COMPLETION_PAIR_REACHED_300UM"
    assert audit["bounded_300um_pair_completion"] == "COMPLETED"
    assert audit["corrected_process_state_branch_birth"] == "DEMONSTRATED"
    assert not audit["predictive_branching_physics_validated"]
    assert not audit["continuation_to_1000um_authorized"]
    assert not audit["maximum_fronts_greater_than_two_supported"]
    assert audit["analysis_pf_workers_started"] == 0
    assert audit["analysis_mechanics_solves"] == 0
    assert all(
        case["qualification"] == "PASS"
        and all(case["checks"].values())
        for case in audit["terminal_cases"].values()
    )


def test_v5_4_2_published_kinetic_k_figures_are_not_rcurves():
    audit = json.loads(
        (BUNDLE / "pf_branching_completion_terminal_audit_v5_4_2.json").read_text()
    )["figure_audit"]
    assert audit["qualification"] == "PASS"
    assert audit["event_owner_source"] == "directional_rates.selected_event_tip_id"
    assert not audit["invalid_contour_points_connected"]
    assert not audit["full_history_called_fracture_toughness_or_R_curve"]
    assert not audit["mechanics_solve_performed"]
    stems = (
        "pf_branching_completion_final_crack_structure_v5_4_2",
        "pf_branching_completion_final_damage_and_process_fields_v5_4_2",
        "pf_branching_completion_model_native_accepted_event_kinetic_K_vs_maximum_network_forward_reach_v5_4_2",
        "pf_branching_completion_model_native_accepted_event_kinetic_K_vs_selected_front_progress_v5_4_2",
    )
    for stem in stems:
        for suffix in (".png", ".pdf", ".svg"):
            assert (BUNDLE / f"{stem}{suffix}").is_file()
