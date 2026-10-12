from collections import Counter

from arrhenius_fracture.prospective_campaign import (
    RATE_FACTORS, TEMPERATURES_K, build_cases, campaign_summary,
    load_material_rows,
)


def test_nominal_blocks_and_storage_are_frozen():
    cases = build_cases(".")
    summary = campaign_summary(cases)
    assert summary["case_count"] == 644
    assert summary["by_block"] == {
        "P1": 288, "P2": 68, "P3": 18, "P4": 44,
        "P5": 96, "P6": 64, "P7": 44, "P8": 22,
    }
    assert summary["estimated_storage_GiB"] == 28.35
    assert summary["production_execution_authorized"] is False
    assert len({case.case_id for case in cases}) == len(cases)


def test_p1_is_complete_nominal_factorial_without_hiding_blocked_domains():
    cases = [case for case in build_cases(".") if case.block == "P1"]
    assert len(cases) == 4 * len(TEMPERATURES_K) * len(RATE_FACTORS) * 2
    counts = Counter((case.family, case.temperature_K, case.rate_factor, case.arm) for case in cases)
    assert set(counts.values()) == {1}
    dbtt_1050 = [case for case in cases if case.family == "DBTT" and case.temperature_K == 1050]
    assert dbtt_1050
    assert {case.eligibility for case in dbtt_1050} == {"BLOCKED_RAW_CONSTITUTIVE_SCALE_NONPOSITIVE"}


def test_dbtt_boundary_is_raw_law_domain_not_runtime_clamp():
    dbtt = load_material_rows(".")["DBTT"]
    assert dbtt.domain(950)["domain_status"] == "ELIGIBLE"
    assert dbtt.domain(1000)["domain_status"] == "QUALIFICATION_REQUIRED_NEAR_SCALE_ZERO"
    state = dbtt.domain(1050)
    assert state["emit_sigc_GPa_raw"] < 0.0
    assert state["domain_status"] == "BLOCKED_RAW_CONSTITUTIVE_SCALE_NONPOSITIVE"


def test_physical_rate_is_changed_through_time_not_displacement_increment():
    cases = [case for case in build_cases(".") if case.block == "P1" and case.family == "peak"
             and case.temperature_K == 900 and case.arm == "accommodation"]
    dU = {case.native_args[case.native_args.index("--dU") + 1] for case in cases}
    dt = {case.rate_factor: float(case.native_args[case.native_args.index("--dt") + 1]) for case in cases}
    assert dU == {"1.9999999999999999e-07"}
    assert dt[0.01] == 840.0
    assert dt[1.0] == 8.4
    assert dt[100.0] == 0.084


def test_primary_branch_and_void_cases_remain_gated_by_physics_not_fixtures():
    cases = build_cases(".")
    branch = [case for case in cases if case.block == "P6" and case.crack_path == "branching_multifront"]
    void = [case for case in cases if case.block == "P7" and case.cavity_mode != "disabled"]
    assert branch and void
    assert {case.gate for case in branch} == {"recursive_branch_birth_and_daughter_qualification"}
    assert all(case.command_status == "UNRESOLVED_ADAPTER_GATE" and not case.native_args for case in branch)
    assert "candidate_rate_map_author_decision" in {case.gate for case in void}
    assert all(case.target_extension_um >= 250 for case in branch + void)


def test_only_resolved_native_rows_carry_commands():
    cases = build_cases(".")
    unresolved = [case for case in cases if case.command_status == "UNRESOLVED_ADAPTER_GATE"]
    assert unresolved and all(not case.native_args for case in unresolved)
    resolved = [case for case in cases if case.command_status == "RESOLVED"]
    assert resolved and all(case.native_args for case in resolved)
    fem = [case for case in cases if case.backend == "full_fem_czm"]
    assert fem and all(case.command_status == "UNRESOLVED_ADAPTER_GATE" for case in fem)
    assert all("mode_i_first_passage" in case.native_entrypoint for case in fem)
