"""Typed, disabled v3 prospective campaign.

V3 preserves every v2 record, binds the prospective v10.2.32 Taylor options,
adds a deliberate cohesive-law comparison block, and separates readiness axes.
It is a construction/audit artifact: production permission is always false.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .prospective_campaign_v2 import build_cases_v2


SCHEMA = "prospective-common-fracture-campaign/3"
REGISTRY = "arrhenius_fracture/data/materials/v10_2_32_taylor_domain/v10_2_32_taylor_domain_registry.csv"
FIT_REGISTRY = "arrhenius_fracture/data/materials/v10_2_32_bilinear_opening/v10_2_32_bilinear_opening_fit_registry.json"
ALL_GOALS = tuple(f"FG-{index:02d}" for index in range(1, 19))
STATUS_AXES = (
    "material_domain", "fit_quality", "implementation", "native_qualification",
    "kernel_validity", "numerical_convergence", "input_availability",
    "resource_admission", "production_permission",
)


def _readiness_options(root: Path) -> dict[str, str]:
    with (root / REGISTRY).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = {
        row["material_class"]: row["option_key"]
        for row in rows if row["option_key"].endswith("taylor_margin_0p10eV")
    }
    if set(selected) != {"peak", "DBTT", "weakT", "ceramic"}:
        raise ValueError("v10.2.32 readiness option set is incomplete")
    return selected


def _replace_arg(args: list[str], flag: str, value: str) -> list[str]:
    result = list(args)
    if flag in result:
        result[result.index(flag) + 1] = value
    return result


def _statuses(row: dict[str, Any], *, cohesive: bool = False) -> dict[str, str]:
    unresolved = row.get("command_status") != "RESOLVED_DISABLED"
    fem = row.get("backend") == "full_fem_czm"
    return {
        "material_domain": (
            "CONSTRUCTED_PROSPECTIVE_NOT_PRODUCTION_SELECTED"
            if row.get("eligibility") == "ELIGIBLE" else str(row.get("eligibility"))
        ),
        "fit_quality": (
            "QUALIFIED_FINITE_DOMAIN_ABSOLUTE_SUPPORT_UNASSIGNED" if cohesive
            else "NOT_APPLICABLE_OR_NATIVE_OPENING"
        ),
        "implementation": (
            "IMPLEMENTED_CONSTITUTIVE_SPECIMEN_ADAPTER_PENDING" if cohesive
            else "UNRESOLVED_ADAPTER" if unresolved else "IMPLEMENTED"
        ),
        "native_qualification": (
            "PENDING_SPECIMEN_LEVEL" if cohesive or unresolved else "COMPACT_PATH_QUALIFIED"
        ),
        "kernel_validity": (
            "PENDING_THREE_LEVEL_COMMON_OBSERVER" if fem else "BOUND_NATIVE_KERNEL"
        ),
        "numerical_convergence": (
            "PENDING_THREE_LEVEL" if fem else "CASE_LEVEL_PRODUCTION_CONVERGENCE_PENDING"
        ),
        "input_availability": (
            "BLOCKED_PHYSICAL_COHESIVE_SUPPORT" if cohesive else "AVAILABLE_VERSIONED_INPUTS"
        ),
        "resource_admission": "PENDING_AUTHORIZED_TARGET_MACHINE_AND_OUTPUT_ROOT",
        "production_permission": "DENIED_CURRENT_MACHINE",
    }


def _execution(row: dict[str, Any]) -> dict[str, Any]:
    status = row["readiness"]
    executable = (
        status["implementation"] == "IMPLEMENTED"
        and status["input_availability"] == "AVAILABLE_VERSIONED_INPUTS"
    )
    return {
        "launcher": [
            "python", "-m", "arrhenius_fracture.shared_execution_worker",
            "{case_request_json}",
        ],
        "request_schema": "shared-execution-request/1",
        "native_entrypoint": row.get("native_entrypoint"),
        "native_args": list(row.get("native_args", [])),
        "output_root": None,
        "target_machine": None,
        "restart_from": None,
        "dry_run_only": True,
        "executable_after_axes_resolve": executable,
        "launch_disabled_reasons": [
            axis for axis, value in status.items()
            if value.startswith(("PENDING", "BLOCKED", "DENIED", "UNRESOLVED"))
        ],
    }


def _convert_v2(root: Path, row: dict[str, Any], options: dict[str, str]) -> dict[str, Any]:
    converted = dict(row)
    converted["schema"] = SCHEMA
    family = converted.get("family")
    if family in options:
        converted["parameter_option"] = options[family]
        args = list(converted.get("native_args", []))
        args = _replace_arg(args, "--parameter-registry", REGISTRY)
        args = _replace_arg(args, "--parameter-option", options[family])
        converted["native_args"] = args
    converted["parameter_registry"] = REGISTRY
    converted["cohesive_law"] = {
        "kind": "original_activated_opening",
        "temperature_policy": "native_temperature_matched",
        "fit_registry": None,
        "physical_support_status": "native_parent_law",
    }
    retained = float(converted.get("storage_estimate_GiB", 0.0))
    converted["resource_estimate"] = {
        "retained_output_GiB": retained,
        "peak_transient_GiB": (
            18.0 if converted.get("backend") == "full_fem_czm"
            else 24.0 if converted.get("block") in {"P6", "P7"}
            else 8.0 if converted.get("target_extension_um", 0.0) >= 1000.0
            else 3.0
        ),
        "reserve_GiB": 12.0,
        "basis": "qualification-manifest engineering estimate; target-machine preflight required",
    }
    converted["output_validation"] = {
        "required": ["source identity", "effective configuration", "scalar history", "accepted state"],
        "conditional": ["restart checkpoint", "selected native fields", "topology graph"],
        "dense_movie_required": False,
    }
    converted["readiness"] = _statuses(converted)
    converted["execution"] = _execution(converted)
    converted["production_execution_authorized"] = False
    return converted


def _fit_by_family(root: Path) -> dict[str, dict[str, Any]]:
    registry = json.loads((root / FIT_REGISTRY).read_text())
    return {item["material_class"]: item for item in registry["fits"]}


def _cohesive_cases(root: Path, options: dict[str, str]) -> list[dict[str, Any]]:
    fits = _fit_by_family(root)
    conditions = (
        ("cold_nominal_accommodation", 300.0, 1.0, "accommodation"),
        ("transition_slow_opening", 950.0, 0.01, "opening_only"),
        ("hot_fast_accommodation", 1200.0, 100.0, "accommodation"),
    )
    laws = (
        ("original_activated_opening", "native_temperature_matched"),
        ("activated_bilinear", "fit_temperature_matched"),
        ("autonomous_irreversible_bilinear", "fit_temperature_matched"),
        ("autonomous_irreversible_bilinear", "reference_temperature_frozen"),
    )
    rows: list[dict[str, Any]] = []
    for family, option in options.items():
        fit = fits[family]
        for condition, temperature, rate, arm in conditions:
            for law, temperature_policy in laws:
                case_id = (
                    f"P9_cohesive_{family}_{condition}_{law}_{temperature_policy}"
                ).replace(".", "p")
                cohesive_fit = None
                physical_support = "native_parent_law"
                if law != "original_activated_opening":
                    cohesive_fit = {
                        "registry": FIT_REGISTRY,
                        "parent_option_key": fit["parent_option_key"],
                        "identifiable_quantities": ["B0_A_times_Gamma_eV", "peak_traction_Pa"],
                        "unassigned_physical_quantities": [
                            "activation_area_m2", "work_of_separation_J_per_m2",
                            "initial_stiffness_Pa_per_m", "peak_separation_m",
                            "final_separation_m", "eta",
                        ],
                    }
                    physical_support = "UNASSIGNED_NO_FABRICATED_SPECIMEN_VALUES"
                row = {
                    "schema": SCHEMA, "case_id": case_id, "block": "P9",
                    "backend": "full_fem_czm", "study": "cohesive_law_comparison",
                    "goal_ids": ["FG-04", "FG-11", "FG-12", "FG-18"],
                    "family": family, "parameter_option": option,
                    "parameter_registry": REGISTRY, "temperature_K": temperature,
                    "rate_factor": rate, "arm": arm, "seed": 8666,
                    "orientation_deg": 0.0, "initial_crack_length_m": 5e-4,
                    "target_extension_um": 10.0,
                    "physical_endpoint": {
                        "kind": "first_accepted_fracture_event_or_declared_opening_censor",
                        "target_extension_m": 1e-5,
                        "event_stop_conditions": ["accepted cohesive damage", "accepted emission"],
                    },
                    "mechanics_state": {
                        "plane_state": "plane_strain", "constraint_control": "prescribed_eps_zz",
                        "eps_zz": 0.0, "in_plane_mode": "mode_I", "unit_thickness_m": 1.0,
                    },
                    "cohesive_law": {
                        "kind": law, "temperature_policy": temperature_policy,
                        "fit": cohesive_fit, "physical_support_status": physical_support,
                    },
                    "native_entrypoint": (
                        "arrhenius_fracture.mode_i_first_passage_v10_0_5_18_four_class_stochastic_emission"
                    ),
                    "native_args": [], "output_profile": "publication_restart",
                    "resource_estimate": {
                        "retained_output_GiB": 0.06, "peak_transient_GiB": 6.0,
                        "reserve_GiB": 12.0,
                    },
                    "output_validation": {
                        "required": [
                            "source identity", "effective cohesive parameters", "reaction history",
                            "damage/event history", "energy ledger", "accepted restart state",
                        ],
                        "conditional": ["selected native fields"],
                        "dense_movie_required": False,
                    },
                    "production_execution_authorized": False,
                }
                row["readiness"] = _statuses(row, cohesive=law != "original_activated_opening")
                if law == "original_activated_opening":
                    row["readiness"]["implementation"] = "UNRESOLVED_GENERIC_FEM_ADAPTER"
                    row["readiness"]["native_qualification"] = "PENDING_THREE_LEVEL_SPECIMEN"
                row["execution"] = _execution(row)
                rows.append(row)
    return rows


def build_cases_v3(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    options = _readiness_options(root)
    rows = [_convert_v2(root, row, options) for row in build_cases_v2(root)]
    rows.extend(_cohesive_cases(root, options))
    ids = [row["case_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("v3 contains duplicate case identifiers")
    return rows


def validate_plan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    for row in rows:
        missing_axes = set(STATUS_AXES) - set(row.get("readiness", {}))
        if missing_axes:
            errors.append(f"{row.get('case_id')}: missing axes {sorted(missing_axes)}")
        if row.get("production_execution_authorized") is not False:
            errors.append(f"{row.get('case_id')}: production permission is not false")
        if not row.get("execution", {}).get("launcher"):
            errors.append(f"{row.get('case_id')}: launcher is empty")
        if not row.get("physical_endpoint"):
            errors.append(f"{row.get('case_id')}: physical endpoint is absent")
    return {"valid": not errors, "errors": errors}


def summary(rows: list[dict[str, Any]], raw_bytes: bytes | None = None) -> dict[str, Any]:
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    by_block: dict[str, int] = {}
    for row in rows:
        by_block[row["block"]] = by_block.get(row["block"], 0) + 1
    readiness_counts: dict[str, dict[str, int]] = {axis: {} for axis in STATUS_AXES}
    for row in rows:
        for axis, value in row["readiness"].items():
            readiness_counts[axis][value] = readiness_counts[axis].get(value, 0) + 1
    goal_records = [
        {
            "goal_id": goal,
            "case_count": sum(goal in row.get("goal_ids", []) for row in rows),
            "status": "REPRESENTED_IN_DISABLED_CAMPAIGN" if any(
                goal in row.get("goal_ids", []) for row in rows
            ) else "FRAMEWORK_LEVEL_NO_DEDICATED_CASE",
        }
        for goal in ALL_GOALS
    ]
    return {
        "schema": SCHEMA, "case_count": len(rows), "by_block": by_block,
        "cohesive_comparison_case_count": by_block.get("P9", 0),
        "status_axes": list(STATUS_AXES), "readiness_counts": readiness_counts,
        "resource_estimates": {
            "summed_retained_output_GiB": sum(
                row["resource_estimate"]["retained_output_GiB"] for row in rows
            ),
            "maximum_single_case_peak_transient_GiB": max(
                row["resource_estimate"]["peak_transient_GiB"] for row in rows
            ),
            "reserve_GiB": max(row["resource_estimate"]["reserve_GiB"] for row in rows),
            "admission_rule": "target free >= selected concurrent peak transient + reserve",
            "maximum_substantial_workers_current_plan": 1,
        },
        "goal_records": goal_records,
        "canonical_semantic_sha256": hashlib.sha256(canonical).hexdigest(),
        "raw_jsonl_sha256": None if raw_bytes is None else hashlib.sha256(raw_bytes).hexdigest(),
        "production_execution_authorized": False,
        "target_machine": None, "output_root": None,
        "validation": validate_plan(rows),
    }


__all__ = ["ALL_GOALS", "SCHEMA", "STATUS_AXES", "build_cases_v3", "summary", "validate_plan"]
