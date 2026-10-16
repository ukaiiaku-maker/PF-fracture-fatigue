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
    assert all(
        row["parameter_registry"].endswith((
            "v10_2_32_taylor_domain_registry.csv",
            "dbtt_paper_minimal_qualification_registry.csv",
        ))
        for row in rows
    )

    cohesive = [row for row in rows if row["block"] == "P9"]
    assert len(cohesive) == 48
    assert {row["cohesive_law"]["kind"] for row in cohesive} == {
        "original_activated_opening", "activated_bilinear",
        "autonomous_irreversible_bilinear",
    }
    fitted = [row for row in cohesive if row["cohesive_law"]["fit"] is not None]
    assert all(row["readiness"]["input_availability"] == "BLOCKED_PHYSICAL_COHESIVE_SUPPORT" for row in fitted)
    assert all(row["cohesive_law"]["fit"]["unassigned_physical_quantities"] for row in fitted)
    assert all(row["qualification_execution"]["launcher"] for row in fitted)
    assert all(
        row["readiness"]["implementation"]
        == "IMPLEMENTED_CONSTITUTIVE_SPECIMEN_ADAPTER_MOVING_TIP_NOT_CONNECTED"
        for row in fitted
    )
    assert all(
        row["qualification_execution"]["production_case_substitution"] is False
        for row in fitted
    )
    native_opening = [
        row for row in cohesive
        if row["cohesive_law"]["kind"] == "original_activated_opening"
    ]
    assert len(native_opening) == 12
    assert all(row["native_args"] for row in native_opening)
    assert all(
        row["native_entrypoint"]
        == "arrhenius_fracture.generic_current_pf_companion"
        for row in native_opening
    )
    assert all(row["readiness"]["implementation"] == "IMPLEMENTED" for row in native_opening)

    dbtt_qualification = [
        row for row in rows
        if row.get("material_selection_scope") == "QUALIFICATION_ONLY"
    ]
    # 66 PF baseline cells, six P4 orientation cells, two P8 cells, and one P9 specimen.
    assert len(dbtt_qualification) == 75
    assert all(row["native_args"] for row in dbtt_qualification)
    assert all(row["production_parameter_option"] is None for row in dbtt_qualification)
    assert {row["parameter_option"] for row in dbtt_qualification} == {
        "dbtt_paper_minimal_sigc_taylor_0p10eV_qualification"
    }
    assert all(
        row["readiness"]["implementation"] == "IMPLEMENTED"
        and row["readiness"]["input_availability"].startswith("AVAILABLE_")
        for row in dbtt_qualification
    )

    fem = [row for row in rows if row["block"] == "P8"]
    assert len(fem) == 22
    assert all(row["native_args"] for row in fem)
    assert all(
        row["native_entrypoint"]
        == "arrhenius_fracture.generic_current_pf_companion"
        for row in fem
    )
    assert all(
        row["shared_options"]["fem_entry"] == "generic_current_pf_companion"
        for row in fem
    )
    assert all(row["command_status"] == "RESOLVED_DISABLED" for row in fem)
    assert all(row["readiness"]["implementation"] == "IMPLEMENTED" for row in fem)
    assert sum(
        row.get("material_selection_scope") == "QUALIFICATION_ONLY"
        for row in fem
    ) == 2

    p4_tip = [
        row for row in rows
        if row["block"] == "P4" and row["source_location"] == "tip_surface"
    ]
    assert len(p4_tip) == 40
    assert all(row["native_args"] for row in p4_tip)
    assert all(
        row["shared_options"]["mechanics"] == "spatial_pf"
        and row["readiness"]["implementation"] == "IMPLEMENTED"
        for row in p4_tip
    )
    p4_bulk = [
        row for row in rows
        if row["block"] == "P4"
        and row["source_location"] in {"bulk_uniform", "tip_plus_bulk"}
    ]
    assert len(p4_bulk) == 4
    assert all(row["native_args"] for row in p4_bulk)
    assert all(row["shared_options"]["material_manifest"] for row in p4_bulk)
    assert {
        row["shared_options"]["source_location"] for row in p4_bulk
    } == {"bulk_uniform", "tip_plus_bulk"}
    assert all(row["readiness"]["implementation"] == "IMPLEMENTED" for row in p4_bulk)

    p6 = [row for row in rows if row["block"] == "P6"]
    assert len(p6) == 64 and all(row["native_args"] for row in p6)
    branch = [row for row in p6 if row["crack_path"] == "branching_multifront"]
    controls = [row for row in p6 if row["crack_path"] == "single_front"]
    assert len(branch) == len(controls) == 32
    assert all("--mechanistic-branching" in row["native_args"] for row in branch)
    assert all(
        row["native_entrypoint"].endswith("sharp_front_v11_branching_audited")
        for row in branch
    )
    assert all(
        row["native_entrypoint"].endswith("sharp_front_v10_2_27_audited")
        for row in controls
    )

    p7_native = [
        row for row in rows
        if row["block"] == "P7" and row["study"] == "native_void_nucleation"
    ]
    assert len(p7_native) == 16 and all(row["native_args"] for row in p7_native)
    assert all(
        row["native_entrypoint"].endswith("shared_v5_cavity_adapter")
        and row["paired_arms_evolved_in_one_native_execution"] is True
        for row in p7_native
    )
    assert all("--target-extension-um" in row["native_args"] for row in p7_native)
    assert all("--steps" not in row["native_args"] for row in p7_native)
    assert {
        row["shared_options"]["cavity"] for row in p7_native
    } == {"disabled", "full_order_v5"}
    p7_preexisting = [
        row for row in rows
        if row["block"] == "P7" and row["study"] != "native_void_nucleation"
    ]
    assert len(p7_preexisting) == 28
    assert all(row["native_args"] for row in p7_preexisting)
    assert all(
        row["cavity_material_bundle"].startswith("current_registry:")
        for row in p7_preexisting
    )
    assert all(
        row["cavity_geometry"]["preparation"]
        == "native_preexisting_resolved_zero_vacancy_debit"
        for row in p7_preexisting
    )
    rate = [row for row in p7_preexisting if row["study"] == "conditional_void_rate"]
    assert len(rate) == 4
    assert all(row["cavity_geometry"]["qualification_geometry_selection"] for row in rate)
    assert all(row["cavity_geometry"]["production_geometry_selection"] is None for row in rate)


def test_v3_summary_has_all_goal_records_and_distinct_digests():
    rows = build_cases_v3(ROOT)
    report = summary(rows, b"raw")
    assert [row["goal_id"] for row in report["goal_records"]] == list(ALL_GOALS)
    assert report["canonical_semantic_sha256"] != report["raw_jsonl_sha256"]
    assert report["target_machine"] is None and report["output_root"] is None


def test_all_p3_operators_are_bound_and_reset_declares_populated_checkpoint_dependency():
    rows = build_cases_v3(ROOT)
    p3 = [row for row in rows if row["block"] == "P3"]
    runnable = [
        row for row in p3 if row["study"] != "history_reset_reequilibrated"
    ]
    resets = [
        row for row in p3 if row["study"] == "history_reset_reequilibrated"
    ]
    assert len(runnable) == 15
    assert len(resets) == 3
    assert all(row["native_args"] for row in p3)
    assert all(row["command_status"] == "RESOLVED_DISABLED" for row in p3)
    assert all(row["shared_options"]["diagnostic_intervention"] == row["study"]
               for row in p3)
    counterfactual = [row for row in p3 if row["study"] != "native"]
    assert all("--diagnostic-intervention" in row["native_args"]
               for row in counterfactual)
    assert all(row["requires_populated_restart_checkpoint"] for row in resets)
    assert all(row["readiness"]["implementation"] == "IMPLEMENTED" for row in resets)
    assert all(row["readiness"]["input_availability"]
               == "PENDING_MATCHED_POPULATED_CHECKPOINT" for row in resets)
