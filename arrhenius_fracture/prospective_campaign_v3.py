"""Typed, disabled v3 prospective campaign.

V3 preserves every v2 record, binds the prospective v10.2.32 Taylor options,
adds a deliberate cohesive-law comparison block, and separates readiness axes.
It is a construction/audit artifact: production permission is always false.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any

from .prospective_campaign import _base_args, build_cases, load_material_rows
from .prospective_campaign_v2 import build_cases_v2
from .shared_options import PRESETS


SCHEMA = "prospective-common-fracture-campaign/3"
REGISTRY = "arrhenius_fracture/data/materials/v10_2_32_taylor_domain/v10_2_32_taylor_domain_registry.csv"
FIT_REGISTRY = "arrhenius_fracture/data/materials/v10_2_32_bilinear_opening/v10_2_32_bilinear_opening_fit_registry.json"
DBTT_QUALIFICATION_REGISTRY = (
    "arrhenius_fracture/data/materials/v10_2_33_dbtt_qualification/"
    "dbtt_paper_minimal_qualification_registry.csv"
)
DBTT_QUALIFICATION_OPTION = (
    "dbtt_paper_minimal_sigc_taylor_0p10eV_qualification"
)
SIGNED_KERNEL_FAMILY = (
    "/Volumes/Data/Data/Nanopillar_calculation/"
    "PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/"
    "v10_2_28_kernel_cache/"
    "4fa015d77f1aadf05f77f550366f64cd611f537ae716bbd47870bf9e6fe2f873/"
    "family.json"
)
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
    qualification_material = row.get("material_selection_scope") == "QUALIFICATION_ONLY"
    fem = row.get("backend") == "full_fem_czm"
    return {
        "material_domain": (
            "CONSTRUCTED_QUALIFICATION_ADMISSIBLE_NOT_PRODUCTION_SELECTED"
            if qualification_material else
            "CONSTRUCTED_PROSPECTIVE_NOT_PRODUCTION_SELECTED"
            if row.get("eligibility") == "ELIGIBLE" else str(row.get("eligibility"))
        ),
        "fit_quality": (
            "QUALIFIED_FINITE_DOMAIN_ABSOLUTE_SUPPORT_UNASSIGNED" if cohesive
            else "NOT_APPLICABLE_OR_NATIVE_OPENING"
        ),
        "implementation": (
            "IMPLEMENTED_CONSTITUTIVE_SPECIMEN_ADAPTER_PENDING" if cohesive
            else "IMPLEMENTED" if qualification_material
            else "UNRESOLVED_ADAPTER" if unresolved else "IMPLEMENTED"
        ),
        "native_qualification": (
            "PENDING_CASE_LEVEL_QUALIFICATION" if qualification_material
            else "PENDING_SPECIMEN_LEVEL" if cohesive or unresolved
            else "COMPACT_PATH_QUALIFIED"
        ),
        "kernel_validity": (
            "PENDING_THREE_LEVEL_COMMON_OBSERVER" if fem else "BOUND_NATIVE_KERNEL"
        ),
        "numerical_convergence": (
            "PENDING_THREE_LEVEL" if fem else "CASE_LEVEL_PRODUCTION_CONVERGENCE_PENDING"
        ),
        "input_availability": (
            "BLOCKED_PHYSICAL_COHESIVE_SUPPORT" if cohesive
            else "AVAILABLE_QUALIFICATION_INPUT_PRODUCTION_SELECTION_PENDING"
            if qualification_material else "AVAILABLE_VERSIONED_INPUTS"
        ),
        "resource_admission": "PENDING_AUTHORIZED_TARGET_MACHINE_AND_OUTPUT_ROOT",
        "production_permission": "DENIED_CURRENT_MACHINE",
    }


def _execution(row: dict[str, Any]) -> dict[str, Any]:
    status = row["readiness"]
    executable = (
        status["implementation"] == "IMPLEMENTED"
        and status["input_availability"].startswith("AVAILABLE_")
    )
    execution = {
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
    if row.get("shared_options") is not None:
        execution["shared_options"] = row["shared_options"]
    return execution


def _fem_native_args(
    root: Path, row: dict[str, Any], *, registry: str, option: str,
) -> list[str]:
    """Construct the already-qualified generic native FEM/CZM request shape."""
    target_um = float(row["target_extension_um"])
    rate = float(row["rate_factor"])
    accommodation = (
        "active_tip" if row["arm"] == "accommodation" else "opening_only"
    )
    return [
        "--material-registry", str((root / registry).resolve()),
        "--parameter-option", option,
        "--constraint", "plane_strain",
        "--source-scope", "tip_only",
        "--accommodation-mode", accommodation,
        "--opening-model", "exp_floor_activated_native",
        "--signed-kernel-family", SIGNED_KERNEL_FAMILY,
        "--mode", "2d",
        "--temperatures", f"{float(row['temperature_K']):g}",
        "--steps", "200000" if target_um >= 1000.0 else "1000",
        "--nx", "36", "--ny", "72",
        "--tip-h-fine", "1e-6", "--tip-ratio", "1.20",
        "--dU", "1e-7", "--dt", f"{4.2 / rate:.17g}",
        "--n-stagger", "2", "--print-every", "200",
        "--adaptive-events", "--adaptive-event-target", "0.15",
        "--adaptive-min-frac", "1e-8", "--adaptive-grow", "4",
        "--da-phys", "5e-6",
        "--target-crack-extension-um", f"{target_um:g}",
        "--crystal-aniso", "--crystal-compete",
        "--crystal-theta-deg", f"{float(row['orientation_deg']):g}",
        "--min-global-forward", "0.05",
        "--crystal-C11", "523e9", "--crystal-C12", "203e9",
        "--crystal-C44", "160e9", "--cleave-gamma-aniso", "0.3",
        "--crystal-material", "w", "--max-fronts", "1",
        "--crack-backend", "adaptive_czm",
        "--czm-max-angle-error-deg", "35", "--j-decomposition", "cluster",
        "--mpz-length-m", "50e-6", "--mpz-n-bins", "80",
    ]


def _bind_generic_fem(root: Path, converted: dict[str, Any]) -> None:
    """Bind P8 to current generic FEM source without authorizing production."""
    registry = str(converted["parameter_registry"])
    option = str(converted["parameter_option"])
    if converted["family"] == "DBTT" and converted.get("eligibility") != "ELIGIBLE":
        converted["requested_parameter_option"] = option
        registry = DBTT_QUALIFICATION_REGISTRY
        option = DBTT_QUALIFICATION_OPTION
        converted["parameter_registry"] = registry
        converted["parameter_option"] = option
        converted["material_selection_scope"] = "QUALIFICATION_ONLY"
        converted["production_parameter_option"] = None
        converted["qualification_material_provenance"] = (
            "arrhenius_fracture/data/materials/v10_2_33_dbtt_qualification/"
            "qualification_report.json"
        )
    converted["native_entrypoint"] = (
        "arrhenius_fracture.generic_current_pf_companion"
    )
    converted["native_args"] = _fem_native_args(
        root, converted, registry=registry, option=option,
    )
    converted["shared_options"] = replace(
        PRESETS["fem_czm_generic_current"], random_seed=int(converted["seed"]),
    ).to_dict()
    converted["command_status"] = "RESOLVED_DISABLED"


def _bind_p4_tip_surface(
    converted: dict[str, Any], original, *, registry: str, option: str,
) -> None:
    """Connect qualified orientation/tip-support cells to the native PF entry."""
    args = list(original.native_args)
    if not args:
        # The v1 gate suppressed emission of arguments, but the original case
        # remains fully specified; reconstruct through the same native builder.
        args = list(_base_args(
            SimpleNamespace(option_key=original.parameter_option),
            float(converted["temperature_K"]),
            float(converted["rate_factor"]), str(converted["arm"]),
            target_um=float(converted["target_extension_um"]),
            theta=float(converted["orientation_deg"]),
            a0_m=float(converted["initial_crack_length_m"]),
        ))
    args = _replace_arg(args, "--parameter-registry", registry)
    args = _replace_arg(args, "--parameter-option", option)
    converted["native_args"] = args
    converted["native_entrypoint"] = (
        "arrhenius_fracture.sharp_front_v10_2_27_audited"
    )
    converted["shared_options"] = replace(
        PRESETS["pf_monotonic_tip_persistent"],
        random_seed=int(converted["seed"]),
    ).to_dict()
    converted["command_status"] = "RESOLVED_DISABLED"


def _bind_p4_bulk(
    root: Path, converted: dict[str, Any], original, *, registry: str, option: str,
) -> None:
    """Connect current exact material rows to the qualified native bulk route."""
    args = list(_base_args(
        SimpleNamespace(option_key=original.parameter_option),
        float(converted["temperature_K"]), float(converted["rate_factor"]),
        str(converted["arm"]), target_um=float(converted["target_extension_um"]),
        theta=float(converted["orientation_deg"]),
        a0_m=float(converted["initial_crack_length_m"]),
    ))
    args = _replace_arg(args, "--parameter-registry", registry)
    args = _replace_arg(args, "--parameter-option", option)
    args = _replace_arg(args, "--bulk-plasticity-mode", "full_field")
    if "--signed-kernel-family" not in args:
        args[0:0] = ["--signed-kernel-family", SIGNED_KERNEL_FAMILY]
    geography = str(converted["source_location"])
    preset = {
        "bulk_uniform": "pf_monotonic_bulk",
        "tip_plus_bulk": "pf_monotonic_tip_plus_bulk",
    }[geography]
    if geography == "bulk_uniform":
        args = [token for token in args if token not in {"--tip-plasticity"}]
        if "--no-tip-plasticity" not in args:
            args.append("--no-tip-plasticity")
    manifest = (
        root / "arrhenius_fracture/data/materials/v10_4_current_bulk"
        / f"{converted['family']}_selected_bulk.csv"
    ).resolve()
    converted["native_args"] = args
    converted["native_entrypoint"] = (
        "arrhenius_fracture.sharp_front_v10_4_bulk_peierls_taylor_audited"
    )
    converted["bulk_material_manifest"] = str(manifest)
    converted["shared_options"] = replace(
        PRESETS[preset], random_seed=int(converted["seed"]),
        material_manifest=str(manifest),
    ).to_dict()
    converted["command_status"] = "RESOLVED_DISABLED"


def _bind_p6_branch_pair(
    converted: dict[str, Any], original, *, registry: str, option: str,
) -> None:
    """Bind persistent single-front controls and native V11 branch arms."""
    args = list(original.native_args)
    if not args:
        args = list(_base_args(
            SimpleNamespace(option_key=original.parameter_option),
            float(converted["temperature_K"]), float(converted["rate_factor"]),
            str(converted["arm"]), target_um=float(converted["target_extension_um"]),
            theta=float(converted["orientation_deg"]),
            a0_m=float(converted["initial_crack_length_m"]),
        ))
    args = _replace_arg(args, "--parameter-registry", registry)
    args = _replace_arg(args, "--parameter-option", option)
    branch = converted["crack_path"] == "branching_multifront"
    output = str(converted["output_profile"])
    if branch:
        index = 0
        while index < len(args):
            if args[index] == "--max-fronts":
                del args[index:index + 2]
            elif args[index].startswith("--max-fronts="):
                del args[index]
            else:
                index += 1
        args.extend(["--mechanistic-branching", "--maximum-fronts", "16"])
        options = replace(
            PRESETS["pf_monotonic_tip_persistent"],
            crack_path="branching_multifront", output=output,
            random_seed=int(converted["seed"]),
        )
        converted["native_entrypoint"] = (
            "arrhenius_fracture.sharp_front_v11_branching_audited"
        )
    else:
        options = replace(
            PRESETS["pf_monotonic_tip_persistent"], output=output,
            random_seed=int(converted["seed"]),
        )
        converted["native_entrypoint"] = (
            "arrhenius_fracture.sharp_front_v10_2_27_audited"
        )
    converted["native_args"] = args
    converted["shared_options"] = options.to_dict()
    converted["command_status"] = "RESOLVED_DISABLED"


def _p7_execution_contract(root: Path, converted: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return complete-case and compact-qualification arguments for P7."""
    interval_s = 0.2
    endpoint = converted["physical_endpoint"]
    exposure_s = float(endpoint["no_event_exposure_ceiling_s"])
    target_um = float(converted["target_extension_um"])
    common = [
        "--duration-s", f"{interval_s:g}",
        "--transport-refinement-min-duration-s", "1e-8",
        "--opening-rate-m-per-s", f"{1.0e-6 * float(converted['rate_factor']):.17g}",
        "--material-class", str(converted["family"]),
        "--material-registry", str((root / converted["parameter_registry"]).resolve()),
        "--parameter-option", str(converted["parameter_option"]),
        "--temperature-K", f"{float(converted['temperature_K']):g}",
        "--vacancy-energy-variant", "reference",
    ]
    complete = [
        *common,
        "--target-extension-um", f"{target_um:g}",
        "--exposure-limit-s", f"{exposure_s:g}",
        "--endpoint-max-intervals", str(int(exposure_s / interval_s) + 1),
    ]
    qualification = [*common, "--steps", "1"]
    return complete, qualification


def _finish_p7_binding(converted: dict[str, Any], qualification_args: list[str]) -> None:
    pair_identity = {
        "native_args": converted["native_args"],
        "seed": converted["seed"],
    }
    digest = hashlib.sha256(
        json.dumps(pair_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    converted["paired_execution_id"] = f"P7-paired-{digest}"
    arm_flag = (
        "--cavity-enabled"
        if converted["cavity_mode"] != "disabled" else "--cavity-disabled"
    )
    converted["qualification_execution"] = {
        "repository_role": "current_pf_shared_framework",
        "native_entrypoint": "arrhenius_fracture.shared_v5_cavity_adapter",
        "launcher": [
            "python", "-m", "arrhenius_fracture.shared_v5_cavity_adapter",
            *qualification_args, arm_flag, "--out", "{qualification_output_dir}",
        ],
        "physical_exposure_s": 0.2,
        "production_case_substitution": False,
    }


def _bind_p7_native_void_pair(root: Path, converted: dict[str, Any]) -> None:
    """Bind native vacancy nucleation to the paired accepted-state V5 scheduler."""
    enabled = converted["cavity_mode"] == "nucleation_growth"
    preset = (
        "pf_cavity_v5_single_front" if enabled
        else "pf_cavity_v5_disabled_control"
    )
    converted["native_args"], qualification_args = _p7_execution_contract(root, converted)
    converted["native_entrypoint"] = (
        "arrhenius_fracture.shared_v5_cavity_adapter"
    )
    converted["shared_options"] = replace(
        PRESETS[preset], random_seed=int(converted["seed"]),
    ).to_dict()
    converted["command_status"] = "RESOLVED_DISABLED"
    converted["paired_arms_evolved_in_one_native_execution"] = True
    converted["qualification_interval_s"] = 0.2
    _finish_p7_binding(converted, qualification_args)


def _bind_p7_preexisting_pair(root: Path, converted: dict[str, Any]) -> None:
    """Bind a declared conditional cavity geometry to the native V5 owner."""
    match = re.search(r"_d([0-9.]+)_r([0-9.]+)", str(converted["study"]))
    qualification_geometry = False
    if match:
        separation_pz, radius_pz = map(float, match.groups())
    else:
        # Central member of the already-declared geometry grid, used only to
        # exercise the independent rate axis; it is not a production default.
        separation_pz, radius_pz = 1.0, 0.25
        qualification_geometry = True
    enabled = converted["cavity_mode"] == "preexisting_equilibrated"
    preset = (
        "pf_cavity_v5_single_front" if enabled
        else "pf_cavity_v5_disabled_control"
    )
    converted["native_args"], qualification_args = _p7_execution_contract(root, converted)
    geometry_args = [
        "--preexisting-cavity-separation-pz", f"{separation_pz:g}",
        "--preexisting-cavity-radius-pz", f"{radius_pz:g}",
        "--process-zone-length-um", "50",
        "--initial-crack-length-um", f"{float(converted['initial_crack_length_m']) * 1e6:g}",
    ]
    converted["native_args"].extend(geometry_args)
    qualification_args.extend(geometry_args)
    converted["native_entrypoint"] = "arrhenius_fracture.shared_v5_cavity_adapter"
    converted["shared_options"] = replace(
        PRESETS[preset], random_seed=int(converted["seed"]),
    ).to_dict()
    converted["command_status"] = "RESOLVED_DISABLED"
    converted["paired_arms_evolved_in_one_native_execution"] = True
    converted["cavity_geometry"] = {
        "radius": radius_pz,
        "surface_to_tip_distance": separation_pz,
        "unit": "process_zone_length",
        "reference_scale": "declared L_pz=50 um",
        "preparation": "native_preexisting_resolved_zero_vacancy_debit",
        "qualification_geometry_selection": qualification_geometry,
        "production_geometry_selection": None if qualification_geometry else "declared_grid_cell",
    }
    converted["cavity_material_bundle"] = (
        f"current_registry:{converted['parameter_option']}"
    )
    converted["qualification_interval_s"] = 0.2
    _finish_p7_binding(converted, qualification_args)


def _bind_p3_diagnostic(
    converted: dict[str, Any], original, *, registry: str, option: str,
) -> None:
    """Bind native and populated-state P3 consumers; leave history reset honest."""
    intervention = str(converted["study"])
    args = list(_base_args(
        SimpleNamespace(option_key=original.parameter_option),
        float(converted["temperature_K"]), float(converted["rate_factor"]),
        str(converted["arm"]), target_um=float(converted["target_extension_um"]),
        theta=float(converted["orientation_deg"]),
        a0_m=float(converted["initial_crack_length_m"]),
    ))
    args = _replace_arg(args, "--parameter-registry", registry)
    args = _replace_arg(args, "--parameter-option", option)
    if intervention != "native":
        args.extend(["--diagnostic-intervention", intervention])
        entry = "arrhenius_fracture.shared_diagnostic_intervention_adapter"
    else:
        entry = "arrhenius_fracture.sharp_front_v10_2_27_audited"
    converted["native_args"] = args
    converted["native_entrypoint"] = entry
    converted["shared_options"] = replace(
        PRESETS["pf_monotonic_tip_persistent"],
        diagnostic_intervention=intervention,
        random_seed=int(converted["seed"]),
    ).to_dict()
    converted["command_status"] = "RESOLVED_DISABLED"
    if intervention == "history_reset_reequilibrated":
        converted["requires_populated_restart_checkpoint"] = True
        converted["restart_checkpoint_requirement"] = (
            "same-case accepted combined native checkpoint with nonzero MPZ population"
        )


def _convert_v2(
    root: Path, row: dict[str, Any], options: dict[str, str], original,
) -> dict[str, Any]:
    converted = dict(row)
    converted["schema"] = SCHEMA
    family = converted.get("family")
    if family in options:
        converted["parameter_option"] = options[family]
        args = list(converted.get("native_args", []))
        args = _replace_arg(args, "--parameter-registry", REGISTRY)
        args = _replace_arg(args, "--parameter-option", options[family])
        converted["native_args"] = args
    needs_dbtt_qualification = (
        family == "DBTT" and converted.get("eligibility") != "ELIGIBLE"
        and converted.get("block") in {"P1", "P2", "P4", "P5"}
    )
    if needs_dbtt_qualification:
        # Preserve the original scientific cell, but make its separately
        # authorized numerical-qualification route complete.  This is not a
        # production material selection.
        converted["requested_parameter_option"] = options["DBTT"]
        converted["parameter_option"] = DBTT_QUALIFICATION_OPTION
        converted["parameter_registry"] = DBTT_QUALIFICATION_REGISTRY
        args = list(original.native_args)
        args = _replace_arg(args, "--parameter-registry", DBTT_QUALIFICATION_REGISTRY)
        args = _replace_arg(args, "--parameter-option", DBTT_QUALIFICATION_OPTION)
        converted["native_args"] = args
        converted["command_status"] = "RESOLVED_QUALIFICATION_DISABLED"
        converted["material_selection_scope"] = "QUALIFICATION_ONLY"
        converted["production_parameter_option"] = None
        converted["qualification_material_provenance"] = (
            "arrhenius_fracture/data/materials/v10_2_33_dbtt_qualification/"
            "qualification_report.json"
        )
    else:
        converted["material_selection_scope"] = "PROSPECTIVE_OR_PARENT"
        converted["parameter_registry"] = REGISTRY
    if converted.get("block") == "P8":
        _bind_generic_fem(root, converted)
    if converted.get("block") == "P3":
        _bind_p3_diagnostic(
            converted, original,
            registry=str(converted["parameter_registry"]),
            option=str(converted["parameter_option"]),
        )
    if (converted.get("block") == "P4"
            and converted.get("source_location") == "tip_surface"):
        _bind_p4_tip_surface(
            converted, original,
            registry=str(converted["parameter_registry"]),
            option=str(converted["parameter_option"]),
        )
    if (converted.get("block") == "P4"
            and converted.get("source_location") in {"bulk_uniform", "tip_plus_bulk"}):
        _bind_p4_bulk(
            root, converted, original,
            registry=str(converted["parameter_registry"]),
            option=str(converted["parameter_option"]),
        )
    if converted.get("block") == "P6":
        _bind_p6_branch_pair(
            converted, original,
            registry=str(converted["parameter_registry"]),
            option=str(converted["parameter_option"]),
        )
    if (converted.get("block") == "P7"
            and converted.get("study") == "native_void_nucleation"):
        _bind_p7_native_void_pair(root, converted)
    if (converted.get("block") == "P7"
            and converted.get("study") != "native_void_nucleation"):
        _bind_p7_preexisting_pair(root, converted)
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
    if converted.get("block") == "P8":
        converted["readiness"].update({
            "implementation": "IMPLEMENTED",
            "native_qualification": "GENERIC_ENTRY_DRY_RUN_QUALIFIED_CASE_LEVEL_PENDING",
            "kernel_validity": "BOUND_NATIVE_KERNEL_COMMON_OBSERVER_CONVERGENCE_PENDING",
            "numerical_convergence": "PENDING_THREE_LEVEL_CASE_GRID",
            "input_availability": (
                "AVAILABLE_QUALIFICATION_INPUT_PRODUCTION_SELECTION_PENDING"
                if converted.get("material_selection_scope") == "QUALIFICATION_ONLY"
                else "AVAILABLE_VERSIONED_INPUTS"
            ),
        })
    if converted.get("block") == "P3":
        history_reset = converted["study"] == "history_reset_reequilibrated"
        converted["readiness"].update({
            "implementation": "IMPLEMENTED",
            "native_qualification": (
                "PERSISTED_RESET_REEQUILIBRATION_QUALIFIED_MATCHED_CASE_CHECKPOINT_PENDING"
                if history_reset else
                "POPULATED_STATE_CONSUMER_QUALIFIED_CASE_LEVEL_PENDING"
            ),
            "kernel_validity": "BOUND_NATIVE_SIGNED_KERNEL_FAMILY",
            "input_availability": (
                "PENDING_MATCHED_POPULATED_CHECKPOINT"
                if history_reset else "AVAILABLE_VERSIONED_INPUTS"
            ),
        })
    if (converted.get("block") == "P4"
            and converted.get("source_location") == "tip_surface"):
        converted["readiness"].update({
            "implementation": "IMPLEMENTED",
            "native_qualification": "ORIENTATION_AND_TIP_SUPPORT_NATIVE_QUALIFIED",
            "kernel_validity": "BOUND_NATIVE_ORIENTATION_RESOLVER",
            "input_availability": (
                "AVAILABLE_QUALIFICATION_INPUT_PRODUCTION_SELECTION_PENDING"
                if converted.get("material_selection_scope") == "QUALIFICATION_ONLY"
                else "AVAILABLE_VERSIONED_INPUTS"
            ),
        })
    if converted.get("block") == "P6":
        converted["readiness"].update({
            "implementation": "IMPLEMENTED",
            "native_qualification": (
                "NATIVE_FIRST_BIRTH_DAUGHTER_RESTART_QUALIFIED_"
                "SECOND_GENERATION_OWNER_GATE_PENDING"
                if converted["crack_path"] == "branching_multifront"
                else "NATIVE_SINGLE_FRONT_CONTROL_QUALIFIED"
            ),
            "kernel_validity": "BOUND_NATIVE_ORIENTATION_RESOLVER",
            "input_availability": "AVAILABLE_VERSIONED_INPUTS",
        })
    if (converted.get("block") == "P7"
            and converted.get("study") == "native_void_nucleation"):
        converted["readiness"].update({
            "implementation": "IMPLEMENTED",
            "native_qualification": (
                "NATIVE_PAIRED_V5_CONNECTED_STATE_QUALIFIED_"
                "CASE_CONTINUATION_PENDING"
            ),
            "kernel_validity": "BOUND_NATIVE_V5_DIRECTIONAL_PROVIDER",
            "numerical_convergence": "ACCEPTED_PREFIX_QUALIFIED_CASE_ENDPOINT_PENDING",
            "input_availability": "AVAILABLE_VERSIONED_INPUTS",
        })
    if (converted.get("block") == "P7"
            and converted.get("study") != "native_void_nucleation"):
        converted["readiness"].update({
            "implementation": "IMPLEMENTED",
            "native_qualification": (
                "PREEXISTING_GEOMETRY_CONSTRUCTOR_QUALIFIED_"
                "CASE_CONTINUATION_PENDING"
            ),
            "kernel_validity": "BOUND_NATIVE_V5_DIRECTIONAL_PROVIDER",
            "numerical_convergence": "CONSTRUCTOR_EQUILIBRIUM_PASS_CASE_ENDPOINT_PENDING",
            "input_availability": "AVAILABLE_CONSTRUCTED_QUALIFICATION_INPUTS",
        })
    if (converted.get("block") == "P4"
            and converted.get("source_location") in {"bulk_uniform", "tip_plus_bulk"}):
        converted["readiness"].update({
            "implementation": "IMPLEMENTED",
            "native_qualification": "FULL_FIELD_GEOGRAPHY_RESTART_QUALIFIED",
            "kernel_validity": "BOUND_NATIVE_SIGNED_KERNEL_FAMILY",
            "input_availability": "AVAILABLE_VERSIONED_INPUTS",
        })
    converted["execution"] = _execution(converted)
    converted["production_execution_authorized"] = False
    return converted


def _fit_by_family(root: Path) -> dict[str, dict[str, Any]]:
    registry = json.loads((root / FIT_REGISTRY).read_text())
    return {item["material_class"]: item for item in registry["fits"]}


def _cohesive_cases(root: Path, options: dict[str, str]) -> list[dict[str, Any]]:
    fits = _fit_by_family(root)
    domain_rows = load_material_rows(root)
    conditions = (
        ("cold_nominal_accommodation", 300.0, 1.0, "accommodation"),
        ("transition_slow_opening", 950.0, 0.01, "opening_only"),
        ("hot_fast_accommodation", 1200.0, 100.0, "accommodation"),
    )
    laws = (
        ("original_activated_opening", "native_temperature_matched"),
        ("bilinear_barrier_activated", "fit_temperature_matched"),
        ("bilinear_czm_classical", "fit_temperature_matched"),
        ("bilinear_czm_classical", "reference_temperature_frozen"),
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
                    "eligibility": domain_rows[family].domain(temperature)["domain_status"],
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
                    # Filled below only through the generic native consumer
                    # appropriate to the selected opening description.
                    "native_entrypoint": None,
                    "native_args": [], "output_profile": "publication_restart",
                    "resource_estimate": {
                        "retained_output_GiB": 0.06, "peak_transient_GiB": 6.0,
                        "reserve_GiB": 12.0,
                        "basis": (
                            "engineering allocation estimate; not measured RSS, "
                            "swap, or transient-disk peak"
                        ),
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
                row["material_selection_scope"] = "PROSPECTIVE_OR_PARENT"
                if law == "original_activated_opening":
                    _bind_generic_fem(root, row)
                row["readiness"] = _statuses(row, cohesive=law != "original_activated_opening")
                if law == "original_activated_opening":
                    row["readiness"].update({
                        "implementation": "IMPLEMENTED",
                        "native_qualification": (
                            "GENERIC_ENTRY_DRY_RUN_QUALIFIED_CASE_LEVEL_PENDING"
                        ),
                        "kernel_validity": (
                            "BOUND_NATIVE_KERNEL_COMMON_OBSERVER_CONVERGENCE_PENDING"
                        ),
                        "numerical_convergence": "PENDING_THREE_LEVEL_CASE_GRID",
                        "input_availability": (
                            "AVAILABLE_QUALIFICATION_INPUT_PRODUCTION_SELECTION_PENDING"
                            if row.get("material_selection_scope") == "QUALIFICATION_ONLY"
                            else "AVAILABLE_VERSIONED_INPUTS"
                        ),
                    })
                else:
                    qualification_args = [
                        "--bilinear-fit-registry", str((root / FIT_REGISTRY).resolve()),
                        "--opening-model", law,
                        "--temperature-K", f"{temperature:g}",
                        "--temperature-policy", temperature_policy,
                        "--constraint", "plane_strain",
                        "--constructed-normal-stiffness-Pa-per-m",
                        "2e17" if law == "bilinear_barrier_activated" else "2e19",
                    ]
                    if law == "bilinear_czm_classical":
                        qualification_args.extend([
                            "--constructed-separation-work-J-per-m2", "20",
                        ])
                    _bind_generic_fem(root, row)
                    row["native_args"] = _replace_arg(
                        row["native_args"], "--opening-model", law,
                    )
                    row["native_args"].extend(qualification_args)
                    row["qualification_execution"] = {
                        "repository_role": "current_fem_companion",
                        "native_entrypoint": (
                            "arrhenius_fracture.generic_current_pf_companion"
                        ),
                        "native_driver_binding": {
                            "specimen_driver": (
                                "arrhenius_fracture.mode_i_first_passage_"
                                "v10_0_5_18_3_9_atomic_path_corridor"
                            ),
                            "cohesive_adapter": (
                                "arrhenius_fracture.native_bilinear_stepper_v1"
                            ),
                            "accepted_state_owner": (
                                "NativeBilinearFrontProvider on the current "
                                "CalibratedMixedModeMPZV911Engine lineage"
                            ),
                            "actual_interface_backend": (
                                "NativeBilinearTrial_StochasticLengthBackend"
                            ),
                            "cohesive_residual_tangent_assembled": True,
                            "runtime_requires_actual_trial_interface": True,
                            "opening_clock_replaced": "inherited_EXP_floor",
                            "persistent_source_owner_preserved": True,
                            "restart_owner": "fem-czm-specimen-checkpoint/1",
                        },
                        "launcher": [
                            "python", "-m",
                            "arrhenius_fracture.generic_current_pf_companion",
                            *row["native_args"],
                            "--out", "{qualification_output_dir}",
                        ],
                        "constructed_input_identity": (
                            "CONSTRUCTED_NATIVE_MOVING_TIP_BILINEAR_NOT_PRODUCTION"
                        ),
                        "production_case_substitution": False,
                    }
                    row["readiness"].update({
                        "implementation": (
                            "IMPLEMENTED_ACTUAL_SOLVED_INTERFACE_"
                            "BILINEAR_PROVIDER"
                        ),
                        "native_qualification": (
                            "PUBLIC_ACTUAL_INTERFACE_RESIDUAL_TANGENT_WORK_"
                            "CLASSICAL_AND_ACTIVATED_EVENTS_ACTIVE_RESTART_"
                            "QUALIFIED_CONSTRUCTED_SUPPORT"
                        ),
                    })
                row["execution"] = _execution(row)
                rows.append(row)
    return rows


def build_cases_v3(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    options = _readiness_options(root)
    originals = {row.case_id: row for row in build_cases(root)}
    rows = [
        _convert_v2(root, row, options, originals[row["case_id"]])
        for row in build_cases_v2(root)
    ]
    rows.extend(_cohesive_cases(root, options))
    # Add only the two discriminating FEM full-field controls needed to expose
    # bulk-only versus tip-plus-bulk behavior through the generic entry.
    for arm, location, plasticity in (
        ("accommodation", "tip_plus_bulk", "tip_plus_full_field"),
        ("opening_only", "bulk_uniform", "full_field_evolving_rho"),
    ):
        source_id = (
            "P8_fem_czm_common_peak_T900_r1_" + arm
            + "_s1733_th0_a500_tip_surface_single_front_disabled"
        )
        source = next(row for row in rows if row["case_id"] == source_id)
        row = copy.deepcopy(source)
        row["case_id"] = source_id.replace(
            "tip_surface_single_front", f"{location}_single_front"
        )
        row["source_location"] = location
        row["plasticity_scope"] = plasticity
        row["native_args"][row["native_args"].index("--source-scope") + 1] = (
            "full_field"
        )
        row["shared_options"]["source_location"] = location
        row["shared_options"]["plasticity"] = plasticity
        row["readiness"].update({
            "native_qualification": "GENERIC_FULL_FIELD_NATIVE_RESTART_QUALIFIED",
            "kernel_validity": "BOUND_NATIVE_SIGNED_KERNEL_FAMILY",
        })
        row["execution"] = _execution(row)
        rows.append(row)

    # Selective (not Cartesian) native FEM controls ensure that constraint and
    # signed grip mixity are present in the executable campaign itself.  These
    # remain disabled qualification cells and preserve all original P8 rows.
    comparison_source_id = (
        "P8_fem_czm_common_peak_T900_r1_accommodation_"
        "s1733_th0_a500_tip_surface_single_front_disabled"
    )
    comparison_source = next(
        row for row in rows if row["case_id"] == comparison_source_id
    )
    comparisons = (
        ("plane_stress_mode_I", "plane_stress", None, 0.0, 0.0, "mode_I"),
        ("gps_eps_mode_I", "generalized_plane_strain", 1.0e-4, None, 0.0, "mode_I"),
        ("gps_sigma_mode_I", "generalized_plane_strain", None, 5.0e7, 0.0, "mode_I"),
        ("plane_strain_mode_II", "plane_strain", 0.0, None, 90.0, "mode_II"),
        ("plane_strain_mixed_I_II", "plane_strain", 0.0, None, 45.0, "mixed_I_II"),
    )
    for label, constraint, eps_zz, sigma_zz, angle, in_plane_mode in comparisons:
        row = copy.deepcopy(comparison_source)
        row["case_id"] = f"P8_fem_selective_{label}_peak_T900_s1733"
        args = list(row["native_args"])
        for flag in ("--eps-zz", "--sigma-zz-Pa"):
            while flag in args:
                index = args.index(flag); del args[index:index + 2]
        args = _replace_arg(args, "--constraint", constraint)
        args = _replace_arg(args, "--grip-mode-angle-deg", f"{angle:g}")
        if "--grip-mode-angle-deg" not in args:
            args.extend(["--grip-mode-angle-deg", f"{angle:g}"])
        if eps_zz is not None:
            args.extend(["--eps-zz", f"{eps_zz:g}"])
        if sigma_zz is not None:
            args.extend(["--sigma-zz-Pa", f"{sigma_zz:g}"])
        row["native_args"] = args
        row["mechanics_state"] = {
            "plane_state": constraint,
            "constraint_control": (
                "prescribed_sigma_zz" if sigma_zz is not None
                else "prescribed_eps_zz" if eps_zz is not None
                else "free_eps_zz"
            ),
            "eps_zz": eps_zz, "sigma_zz_Pa": sigma_zz,
            "in_plane_mode": in_plane_mode,
            "grip_mode_angle_deg": angle,
            "unit_thickness_m": 1.0,
        }
        row["shared_options"]["plane_state"] = constraint
        row["shared_options"]["in_plane_mode"] = in_plane_mode
        row["shared_options"]["grip_mode_angle_deg"] = angle
        row["selective_native_comparison"] = True
        row["readiness"]["native_qualification"] = (
            "NATIVE_CONSTRAINT_AND_SIGNED_MIXITY_PATH_QUALIFIED_"
            "CASE_LEVEL_PENDING"
        )
        row["execution"] = _execution(row)
        rows.append(row)
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
