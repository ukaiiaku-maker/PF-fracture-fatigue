import numpy as np

from arrhenius_fracture.quality_diversity_v1029 import (
    QualityDiversityConfig,
    select_quality_diverse,
)


def _row(index, *, objective, parameter, response, passed=True, target="DBTT"):
    return {
        "candidate_id": f"C{index:03d}",
        "target_class": target,
        "analytical_pass": passed,
        "analytical_objective": objective,
        "cleave_G00_eV": 1.0 + parameter,
        "emit_G00_eV": 1.0 + 0.5 * parameter,
        "source_sites_per_system": 10.0 ** (1.0 + 0.1 * parameter),
        "analytical_K_cleave_300K": 10.0 + response,
        "analytical_K_cleave_1200K": 12.0 + 2.0 * response,
        "analytical_Kshield_1200K": 0.5 + response,
    }


def test_quality_diversity_retains_distinct_candidate_not_in_pure_top_n():
    rows = [
        _row(0, objective=0.00, parameter=0.00, response=0.00),
        _row(1, objective=0.01, parameter=0.01, response=0.01),
        _row(2, objective=0.02, parameter=0.02, response=0.02),
        _row(3, objective=0.03, parameter=0.03, response=0.03),
        _row(4, objective=0.08, parameter=4.00, response=3.00),
        _row(5, objective=0.09, parameter=2.00, response=-2.00),
    ]
    selected, audit = select_quality_diverse(
        rows,
        pass_key="analytical_pass",
        objective_key="analytical_objective",
        target_class="DBTT",
        stage="analytical",
        config=QualityDiversityConfig(
            count=3,
            quality_reserve_fraction=1.0 / 3.0,
            quality_weight=0.25,
            parameter_distance_weight=0.45,
            response_distance_weight=0.55,
            pool_factor=12,
            preserve_anchor_lineages=False,
        ),
    )
    ids = {row["candidate_id"] for row in selected}
    assert "C000" in ids
    assert ids.intersection({"C004", "C005"})
    assert audit["selection_changes_promotion_set"] is True
    assert audit["selected_response_pairwise_distance"]["minimum"] > audit[
        "quality_only_response_pairwise_distance"
    ]["minimum"]


def test_passed_candidates_exclude_near_passes_when_enough_passers_exist():
    rows = [
        _row(0, objective=0.10, parameter=0.0, response=0.0, passed=True),
        _row(1, objective=0.20, parameter=1.0, response=1.0, passed=True),
        _row(2, objective=0.00, parameter=5.0, response=5.0, passed=False),
    ]
    selected, audit = select_quality_diverse(
        rows,
        pass_key="analytical_pass",
        objective_key="analytical_objective",
        target_class="DBTT",
        stage="analytical",
        config=QualityDiversityConfig(count=2),
    )
    assert {row["candidate_id"] for row in selected} == {"C000", "C001"}
    assert audit["passed_count"] == 2


def test_anchor_lineage_reserve_preserves_all_available_dbtt_anchor_basins():
    rows = []
    for index, lineage in enumerate(range(3)):
        row = _row(
            index,
            objective=0.01 * index,
            parameter=0.2 * index,
            response=0.1 * index,
        )
        for anchor in range(3):
            row[f"anchor_weight_{anchor}"] = 0.05
        row[f"anchor_weight_{lineage}"] = 0.90
        rows.append(row)
    selected, _audit = select_quality_diverse(
        rows,
        pass_key="analytical_pass",
        objective_key="analytical_objective",
        target_class="DBTT",
        stage="analytical",
        config=QualityDiversityConfig(
            count=3,
            quality_reserve_fraction=1.0 / 3.0,
            preserve_anchor_lineages=True,
        ),
    )
    assert {row["dominant_anchor_lineage"] for row in selected} == {
        "anchor_weight_0",
        "anchor_weight_1",
        "anchor_weight_2",
    }


def test_selection_is_deterministic_for_fixed_rows():
    rows = [
        _row(
            index,
            objective=0.01 * index,
            parameter=float(index),
            response=float(np.sin(index)),
        )
        for index in range(12)
    ]
    kwargs = dict(
        pass_key="analytical_pass",
        objective_key="analytical_objective",
        target_class="DBTT",
        stage="analytical",
        config=QualityDiversityConfig(count=5),
    )
    first, _ = select_quality_diverse(rows, **kwargs)
    second, _ = select_quality_diverse(rows, **kwargs)
    assert [row["candidate_id"] for row in first] == [
        row["candidate_id"] for row in second
    ]
