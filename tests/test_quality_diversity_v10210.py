import numpy as np

from arrhenius_fracture.quality_diversity_v10210 import (
    QualityDiversityConfig,
    select_quality_diverse,
)


def _row(
    index,
    *,
    objective,
    parameter,
    response,
    passed=True,
    target="DBTT",
    pass_key="analytical_pass",
    objective_key="analytical_objective",
):
    return {
        "candidate_id": f"C{index:03d}",
        "target_class": target,
        pass_key: passed,
        objective_key: objective,
        "cleave_G00_eV": 1.0 + parameter,
        "emit_G00_eV": 1.0 + 0.5 * parameter,
        "source_sites_per_system": 10.0 ** (1.0 + 0.1 * parameter),
        "analytical_K_cleave_300K": 10.0 + response,
        "analytical_K_cleave_1200K": 12.0 + 2.0 * response,
        "analytical_Kshield_1200K": 0.5 + response,
        "K_first_300K": 10.0 + response,
        "K_first_1200K": 12.0 + response,
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
            preserve_anchor_lineages=False,
        ),
    )
    ids = {row["candidate_id"] for row in selected}
    assert "C000" in ids
    assert ids.intersection({"C004", "C005"})
    assert audit["selection_changes_promotion_set"] is True


def test_all_passers_are_hard_reserved_when_budget_exceeds_passing_population():
    rows = [
        _row(0, objective=0.50, parameter=0.0, response=0.0, passed=True),
        _row(1, objective=0.60, parameter=0.1, response=0.1, passed=True),
        _row(2, objective=0.00, parameter=10.0, response=10.0, passed=False),
        _row(3, objective=0.01, parameter=-0.9, response=-5.0, passed=False),
        _row(4, objective=0.02, parameter=5.0, response=7.0, passed=False),
    ]
    selected, audit = select_quality_diverse(
        rows,
        pass_key="analytical_pass",
        objective_key="analytical_objective",
        target_class="DBTT",
        stage="analytical",
        config=QualityDiversityConfig(
            count=4,
            quality_reserve_fraction=0.25,
            quality_weight=0.0,
            preserve_anchor_lineages=False,
        ),
    )
    ids = {row["candidate_id"] for row in selected}
    assert {"C000", "C001"}.issubset(ids)
    assert audit["all_passers_retained"] is True
    assert audit["selected_reasons"]["C000"] == "all_passers_reserve"
    assert audit["selected_reasons"]["C001"] == "all_passers_reserve"


def test_enough_passers_exclude_failed_candidates():
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
    assert audit["all_passers_retained"] is True


def test_anchor_lineage_reserve_is_scoped_to_analytical_dbtt():
    rows = []
    for index, lineage in enumerate(range(3)):
        row = _row(
            index,
            objective=0.01 * index,
            parameter=0.2 * index,
            response=0.1 * index,
            pass_key="first_passage_pass",
            objective_key="first_passage_objective",
        )
        for anchor in range(3):
            row[f"anchor_weight_{anchor}"] = 0.05
        row[f"anchor_weight_{lineage}"] = 0.90
        rows.append(row)
    selected, audit = select_quality_diverse(
        rows,
        pass_key="first_passage_pass",
        objective_key="first_passage_objective",
        target_class="DBTT",
        stage="first-passage",
        config=QualityDiversityConfig(count=2, preserve_anchor_lineages=True),
    )
    assert len(selected) == 2
    assert audit["lineage_reserve_active"] is False
    assert "anchor_lineage_reserve" not in set(audit["selected_reasons"].values())


def test_anchor_lineage_reserve_remains_active_for_analytical_dbtt():
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
    selected, audit = select_quality_diverse(
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
    assert audit["lineage_reserve_active"] is True
    assert {row["dominant_anchor_lineage"] for row in selected} == {
        "anchor_weight_0",
        "anchor_weight_1",
        "anchor_weight_2",
    }


def test_selection_is_deterministic():
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
