from pathlib import Path

from arrhenius_fracture.prospective_campaign_v3 import (
    ALL_GOALS, STATUS_AXES, build_cases_v3, summary, validate_plan,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v3_is_typed_disabled_and_adds_deliberate_cohesive_block():
    rows = build_cases_v3(ROOT)
    assert len(rows) == 692
    assert validate_plan(rows)["valid"]
    assert all(row["production_execution_authorized"] is False for row in rows)
    assert all(set(row["readiness"]) == set(STATUS_AXES) for row in rows)
    assert all(row["execution"]["launcher"] for row in rows)
    assert all(row["resource_estimate"]["peak_transient_GiB"] > 0 for row in rows)
    assert all(row["output_validation"]["dense_movie_required"] is False for row in rows)
    assert all(row["parameter_registry"].endswith("v10_2_32_taylor_domain_registry.csv") for row in rows)

    cohesive = [row for row in rows if row["block"] == "P9"]
    assert len(cohesive) == 48
    assert {row["cohesive_law"]["kind"] for row in cohesive} == {
        "original_activated_opening", "activated_bilinear",
        "autonomous_irreversible_bilinear",
    }
    fitted = [row for row in cohesive if row["cohesive_law"]["fit"] is not None]
    assert all(row["readiness"]["input_availability"] == "BLOCKED_PHYSICAL_COHESIVE_SUPPORT" for row in fitted)
    assert all(row["cohesive_law"]["fit"]["unassigned_physical_quantities"] for row in fitted)


def test_v3_summary_has_all_goal_records_and_distinct_digests():
    rows = build_cases_v3(ROOT)
    report = summary(rows, b"raw")
    assert [row["goal_id"] for row in report["goal_records"]] == list(ALL_GOALS)
    assert report["canonical_semantic_sha256"] != report["raw_jsonl_sha256"]
    assert report["target_machine"] is None and report["output_root"] is None
