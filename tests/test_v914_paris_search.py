from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_v914_paris_search import aggregate_seed_fits, best_fit, campaign_group_complete, expected_level0_points, fit_segment, level0_decision, slug
from propose_v914_paris_surrogate_batch import descriptor_response, gp_predict


CONFIG = json.loads((ROOT / "config/v914_paris_search_target.json").read_text())
ARTIFACT = ROOT / "artifacts/v914_paris_reparameterization"


def point(deltaK: float, *, usable: bool = True) -> dict:
    return {
        "DeltaK_MPa_sqrt_m": deltaK,
        "developed_da_dN_m_per_cycle": 1.0e-12 * deltaK**4,
        "developed_point_usable": usable,
        "censor_type": "none" if usable else "cycle_horizon_reached",
        "event_count": 5 if usable else 0,
    }


def test_exact_power_law_fit_recovers_m_and_C():
    values = [point(value) for value in (10.0, 12.0, 14.0, 16.0, 18.0)]
    fit = fit_segment(values)
    assert fit["m"] == pytest.approx(4.0)
    assert fit["C"] == pytest.approx(1.0e-12)
    assert fit["R2"] == pytest.approx(1.0)


def test_valid_paris_region_requires_four_contiguous_usable_points():
    valid = best_fit([point(value) for value in (10.0, 12.0, 14.0, 16.0, 18.0)], CONFIG)
    assert valid["valid_Paris_region"] is True
    assert valid["n_points"] >= 4

    broken = [point(value) for value in (10.0, 12.0, 14.0, 16.0, 18.0)]
    broken[2] = point(14.0, usable=False)
    rejected = best_fit(broken, CONFIG)
    assert rejected["valid_Paris_region"] is False
    assert rejected["n_points"] == 4


def test_two_points_never_define_a_paris_candidate():
    fit = best_fit([point(10.0), point(15.0)], CONFIG)
    assert fit["valid_Paris_region"] is False
    assert fit["n_points"] == 2


def test_three_contiguous_points_define_only_a_provisional_local_slope():
    fit = best_fit([point(10.0), point(12.0), point(14.0)], CONFIG)
    assert fit["descriptor_status"] == "PROVISIONAL_THREE_POINT_LOCAL_SLOPE"
    assert fit["m"] == pytest.approx(4.0)
    assert fit["valid_Paris_region"] is False


def test_provisional_descriptor_does_not_bridge_a_censor():
    values = [point(10.0), point(12.0), point(14.0, usable=False), point(16.0), point(18.0)]
    fit = best_fit(values, CONFIG)
    assert fit["descriptor_status"] == "NO_LOCAL_SLOPE_DESCRIPTOR"
    assert fit["m"] != fit["m"]


def test_surrogate_separates_invalid_cases_from_measured_descriptors():
    fits = [
        {"candidate_id": "a", "fidelity": "level1", "objective": "0.5", "descriptor_status": "PROVISIONAL_THREE_POINT_LOCAL_SLOPE"},
        {"candidate_id": "b", "fidelity": "level1", "objective": "inf", "descriptor_status": "NO_LOCAL_SLOPE_DESCRIPTOR"},
        {"candidate_id": "c", "fidelity": "level0", "objective": "0.1", "descriptor_status": "VALID_PARIS_REGION"},
    ]
    observed, response, invalid = descriptor_response({"a", "b", "c"}, fits, "level1")
    assert observed == ["a", "b"]
    assert response == {"a": 0.5}
    assert invalid == {"b"}


def test_shape_first_surrogate_uses_measured_m_without_fabricating_invalid_slopes():
    fits = [
        {"candidate_id": "a", "fidelity": "level1", "m": "20.5", "objective": "8", "descriptor_status": "PROVISIONAL_THREE_POINT_LOCAL_SLOPE"},
        {"candidate_id": "b", "fidelity": "level1", "m": "nan", "objective": "inf", "descriptor_status": "NO_LOCAL_SLOPE_DESCRIPTOR"},
    ]
    _, response, invalid = descriptor_response({"a", "b"}, fits, "level1", "m")
    assert response == {"a": 20.5}
    assert invalid == {"b"}


def test_surrogate_cli_rejects_less_than_two_descriptors(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "propose_v914_paris_surrogate_batch.py"),
            "--iteration", "1",
            "--minimum-descriptors", "1",
            "--out", str(tmp_path / "out"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "minimum-descriptors must be at least two" in result.stderr


def test_recovered_population_and_initial_design_are_audited():
    import csv

    with (ARTIFACT / "candidate_pool.csv").open(newline="") as stream:
        pool = list(csv.DictReader(stream))
    with (ARTIFACT / "candidate_registry.csv").open(newline="") as stream:
        initial = list(csv.DictReader(stream))
    assert len(pool) == 1024
    assert len(initial) == 32
    assert [row["paris_design_role"] for row in initial[:4]] == ["ABCD_A", "ABCD_B", "ABCD_C", "ABCD_D"]
    assert not any(field.startswith("physics__") for field in initial[0])
    assert len([field for field in initial[0] if field.startswith("u__")]) == 22


def test_surrogate_predictor_returns_finite_uncertainty():
    import numpy as np

    x = np.asarray([[0.0], [0.5], [1.0]])
    y = np.asarray([1.0, 0.0, 1.0])
    mean, std, length = gp_predict(x, y, np.asarray([[0.25], [0.75]]))
    assert np.all(np.isfinite(mean))
    assert np.all(std > 0.0)
    assert length > 0.0


def test_level0_never_rejects_from_partial_load_evidence():
    inert = {"status": "cycle_horizon_reached", "event_count": 0, "completed_physical_cycles": 1.0e8}
    assert level0_decision([inert, inert], 3, 1.0) == (False, "LEVEL0_INCOMPLETE", "level0")
    assert level0_decision([inert, inert, inert], 3, 1.0) == (False, "LEVEL0_EARLY_REJECTION", "none")


def test_high_k50_candidate_requires_supplemental_level0_probe():
    assert expected_level0_points({"historical_monotonic_K50_300K_MPa_sqrt_m": "30.0"}, CONFIG) == 4
    assert expected_level0_points({"historical_monotonic_K50_300K_MPa_sqrt_m": "12.0"}, CONFIG) == 3


def test_high_k50_candidate_with_finite_base_bracket_promotes_without_supplement():
    points = [
        {"status": "work_budget_exhausted", "event_count": 0, "completed_physical_cycles": 1.0e8},
        {"status": "work_budget_exhausted", "event_count": 0, "completed_physical_cycles": 1.0e7},
        {"status": "growth_target_reached", "event_count": 3, "completed_physical_cycles": 19},
    ]
    assert level0_decision(points, 4, 1.0) == (
        True, "LEVEL0_FINITE_NONCATASTROPHIC_GROWTH", "level1"
    )


def test_analysis_and_launcher_share_precise_load_slug():
    assert slug(30.3832779198) == "30p3833"


def test_analyzer_discovers_campaigns_independent_of_label():
    source = (SCRIPTS / "analyze_v914_paris_search.py").read_text()
    assert 'args.root.glob("*/launch_manifest.json")' in source
    assert 'args.root.glob("level*/launch_manifest.json")' not in source


def test_level3_launcher_preserves_case_specific_common_random_seeds(tmp_path):
    case_table = tmp_path / "cases.csv"
    case_table.write_text(
        "candidate_id,DeltaK_MPa_sqrt_m,seed,case_role\n"
        "v914_endurance_knee_0462,15.0,1720,LEVEL3_FINALIST_PRODUCTION_PARIS_INTERVAL\n"
        "v914_endurance_knee_0462,15.0,2718,LEVEL3_FINALIST_PRODUCTION_PARIS_INTERVAL\n"
    )
    out = tmp_path / "runs"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_v914_paris_search.py"),
            "--fidelity", "level3",
            "--case-table", str(case_table),
            "--campaign-label", "level3_test",
            "--out-root", str(out),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((out / "level3_test" / "launch_manifest.json").read_text())
    assert manifest["numerical_fidelity"]["n_bins"] == 128
    assert manifest["numerical_fidelity"]["target_extension_um"] == 100.0
    assert [case["seed"] for case in manifest["cases"]] == [1720, 2718]


def test_level1_search_descriptor_uses_validated_short_target_only(tmp_path):
    case_table = tmp_path / "cases.csv"
    case_table.write_text(
        "candidate_id,DeltaK_MPa_sqrt_m,seed,case_role\n"
        "v914_endurance_knee_0462,12.0,1720,SEARCH_DESCRIPTOR\n"
    )
    out = tmp_path / "runs"
    subprocess.run(
        [
            sys.executable, str(SCRIPTS / "run_v914_paris_search.py"),
            "--fidelity", "level1", "--search-descriptor",
            "--case-table", str(case_table), "--campaign-label", "descriptor_test",
            "--out-root", str(out), "--dry-run",
        ], cwd=ROOT, check=True, capture_output=True, text=True,
    )
    manifest = json.loads((out / "descriptor_test" / "launch_manifest.json").read_text())
    assert manifest["search_descriptor"] is True
    assert manifest["descriptor_only_not_formal_Paris_claim"] is True
    assert manifest["numerical_fidelity"]["n_bins"] == 64
    assert manifest["numerical_fidelity"]["maximum_physical_cycles"] == 10_000_000_000
    assert manifest["numerical_fidelity"]["target_extension_um"] == 9.0


def test_level1_search_case_designer_emits_three_bracketed_loads(tmp_path):
    registry = tmp_path / "registry.csv"
    registry.write_text("candidate_id,historical_monotonic_K50_300K_MPa_sqrt_m\ncandidate,15\n")
    promotions = tmp_path / "promotions.csv"
    promotions.write_text("candidate_id,next_fidelity,promote\ncandidate,level1,true\n")
    points = tmp_path / "points.csv"
    points.write_text(
        "candidate_id,fidelity,DeltaK_MPa_sqrt_m,status\n"
        "candidate,level0,10.8,work_budget_exhausted\n"
        "candidate,level0,16.2,growth_target_reached\n"
    )
    output = tmp_path / "cases.csv"
    subprocess.run(
        [
            sys.executable, str(SCRIPTS / "design_v914_paris_adaptive_cases.py"),
            "--mode", "level1-search-descriptors", "--registry", str(registry),
            "--promotions", str(promotions), "--points", str(points), "--out", str(output),
        ], cwd=ROOT, check=True, capture_output=True, text=True,
    )
    import csv
    rows = list(csv.DictReader(output.open()))
    assert len(rows) == 3
    assert all(row["case_role"] == "LEVEL1_SEARCH_DESCRIPTOR" for row in rows)
    assert [float(row["DeltaK_MPa_sqrt_m"]) for row in rows] == pytest.approx([11.34, (11.34 * 19.44) ** 0.5, 19.44])


def test_finalist_seed_statistics_exclude_invalid_fits():
    base = {
        "candidate_id": "candidate", "fidelity": "level3", "R": 0.1,
        "frequency_Hz": 1000.0, "temperature_K": 300.0, "log10_C": -12.0,
        "objective": 0.25,
    }
    rows = aggregate_seed_fits([
        {**base, "seed": 1720, "m": 4.0, "valid_Paris_region": True},
        {**base, "seed": 2718, "m": 6.0, "log10_C": -10.0, "valid_Paris_region": True},
        {**base, "seed": 3141, "m": 100.0, "valid_Paris_region": False},
    ])
    assert len(rows) == 1
    assert rows[0]["n_valid_seeds"] == 2
    assert rows[0]["m_mean"] == pytest.approx(5.0)
    assert rows[0]["m_std"] == pytest.approx(2.0**0.5)
    assert rows[0]["log10_C_mean"] == pytest.approx(-11.0)


def test_level1_campaign_cannot_reject_from_partial_load_evidence():
    base = {
        "candidate_id": "candidate", "fidelity": "level1", "seed": 1720,
        "R": 0.1, "frequency_Hz": 1000.0, "temperature_K": 300.0,
    }
    history = [
        {**base, "deltaK_MPa_sqrt_m": load}
        for load in (10.0, 12.0, 14.0, 16.0)
    ]
    partial = [
        {**base, "DeltaK_MPa_sqrt_m": load}
        for load in (10.0, 12.0, 14.0)
    ]
    assert campaign_group_complete("candidate", "level1", partial, history) is False
    complete = partial + [{**base, "DeltaK_MPa_sqrt_m": 16.0}]
    assert campaign_group_complete("candidate", "level1", complete, history) is True
