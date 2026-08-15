from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_v1032_transition_inventory as inventory


def test_authoritative_material_identifiers_are_exact():
    assert inventory.MATERIALS == {
        "DBTT": ("v913_zeroD_sobol_0202500", "v913_paper_dbtt01_0202500_persistent_sites", 21.02530765128298),
        "Peak-T": ("v913_zeroD_sobol_0242980", "v913_paper_peak01_0242980_persistent_sites", 21.289546465050222),
        "weak-T": ("v913_zeroD_sobol_0129902", "v913_paper_weakT01_0129902_persistent_sites", 12.702935563752424),
        "ceramic-like": ("v913_zeroD_sobol_0077080", "v913_paper_ceramic01_0077080_persistent_sites", 12.259477791864454),
    }


def test_material_accelerated_censors_and_partials_have_no_rate():
    repo = Path(__file__).resolve().parents[1]
    rows = pd.DataFrame(inventory.material_accelerated_2d(repo))
    unresolved = rows[rows.plot_kind.isin(["censor", "partial"])]
    assert set(rows["class"]) == {"DBTT", "Peak-T", "weak-T", "ceramic-like"}
    assert not unresolved.empty
    assert unresolved.da_dN_m_per_cycle.isna().all()


def test_round_one_is_bounded_and_material_only():
    repo = Path(__file__).resolve().parents[1]
    matrix = pd.read_csv(repo / "runtime_inputs/v10_2_32/transition_refinement_1d_round1.csv")
    assert set(matrix.family) == set(inventory.MATERIALS)
    assert len(matrix) == 8
    assert matrix.maximum_cycles.max() == 5000
    script = (repo / "scripts/run_v1032_transition_refinement_1d.sh").read_text()
    assert "state-history-cycle-interval 10" in script
    assert "authoritative launch requires clean worktree" in script
    assert 'mode=${7:-explicit}' in script


def test_round_two_adaptively_concentrates_on_weakt_transition():
    repo = Path(__file__).resolve().parents[1]
    matrix = pd.read_csv(repo / "runtime_inputs/v10_2_32/transition_refinement_1d_round2.csv")
    assert len(matrix) == 11
    assert (matrix.family == "weak-T").sum() == 5
    assert set(matrix[matrix.family == "weak-T"].fraction) == {1.12, 1.14, 1.16, 1.18, 1.20}
    assert matrix.maximum_cycles.max() == 5000


def test_accelerated_matrix_matches_explicit_transition_loads():
    repo = Path(__file__).resolve().parents[1]
    matrix = pd.read_csv(repo / "runtime_inputs/v10_2_32/transition_refinement_1d_accelerated.csv")
    assert len(matrix) == 20
    assert set(matrix["mode"]) == {"accelerated"}
    assert set(matrix.family) == set(inventory.MATERIALS)
    assert matrix.maximum_cycles.min() == 10_000_000


def test_explicit_2d_matrix_is_sparse_adaptive_and_canonical():
    repo = Path(__file__).resolve().parents[1]
    matrix = pd.read_csv(repo / "runtime_inputs/v10_2_32/transition_refinement_2d_explicit.csv")
    assert len(matrix) == 17
    assert matrix.groupby("family").size().to_dict() == {
        "DBTT": 3, "Peak-T": 4, "weak-T": 5, "ceramic-like": 5,
    }
    assert set(matrix.family) == set(inventory.MATERIALS)
    for family, (candidate, option, _) in inventory.MATERIALS.items():
        rows = matrix[matrix.family == family]
        assert set(rows.candidate_id) == {candidate}
        assert set(rows.parameter_option) == {option}
    assert matrix.maximum_cycles.max() == 1200
    assert matrix.maximum_cycles.min() == 100


def test_transition_2d_launcher_is_clean_head_and_disk_guarded():
    repo = Path(__file__).resolve().parents[1]
    script = (repo / "scripts/run_v1032_transition_refinement_2d.sh").read_text()
    assert "authoritative launch requires clean worktree" in script
    assert "MIN_FREE_GIB" in script
    assert "wait -n" not in script
    assert 'REGISTRY=arrhenius_fracture/data/materials/v10_2_27_paper_four_class_registry.csv' in script
