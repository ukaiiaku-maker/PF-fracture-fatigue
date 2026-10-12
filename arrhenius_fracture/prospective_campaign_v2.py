"""Disabled v2 prospective plan with independent dependencies and typed inputs.

V1 is imported read-only and remains reproducible.  V2 repairs planning
semantics; it does not make an unresolved case runnable or authorize a solve.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .prospective_campaign import CampaignCase, build_cases


SCHEMA = "prospective-common-fracture-campaign/2"


GOALS = {
    "P1": ["FG-02", "FG-03", "FG-05", "FG-15"],
    "P2": ["FG-04", "FG-06", "FG-07"],
    "P3": ["FG-04", "FG-07"],
    "P4": ["FG-08", "FG-09", "FG-10"],
    "P5": ["FG-06", "FG-07"],
    "P6": ["FG-13", "FG-14", "FG-16"],
    "P7": ["FG-13", "FG-16"],
    "P8": ["FG-11", "FG-12"],
}


def _adapter_dependency(case: CampaignCase) -> dict[str, str]:
    if case.command_status == "UNRESOLVED_ADAPTER_GATE":
        return {"kind": "native_adapter", "status": "UNRESOLVED",
                "detail": case.gate}
    return {"kind": "native_adapter", "status": "RESOLVED",
            "detail": "v1 source-bound native arguments"}


def _domain_dependency(case: CampaignCase) -> dict[str, str]:
    ok = case.eligibility == "ELIGIBLE"
    return {"kind": "material_domain", "status": "RESOLVED" if ok else "BLOCKED",
            "detail": case.eligibility}


def _cavity_geometry(case: CampaignCase) -> dict[str, Any] | None:
    if case.block != "P7" or case.cavity_mode == "disabled":
        return None
    match = re.search(r"_d([0-9.]+)_r([0-9.]+)", case.study)
    if match:
        distance, radius = map(float, match.groups())
        return {"radius": radius, "surface_to_tip_distance": distance,
                "unit": "process_zone_length", "reference_scale": "declared L_pz",
                "preparation": "preexisting_equilibrated"}
    return {"radius": None, "surface_to_tip_distance": None, "unit": "m",
            "reference_scale": "physical SI", "preparation": "native_nucleation_growth",
            "status": "PHYSICAL_INPUT_REQUIRED"}


def convert_case(case: CampaignCase) -> dict[str, Any]:
    dependencies = [_domain_dependency(case), _adapter_dependency(case)]
    if case.block == "P7":
        dependencies.append({"kind": "cavity_kinetic_interpretation", "status": "DEVELOPMENT_CANDIDATE",
                             "detail": "local_incremental_front; author production choice not approved"})
    if case.block == "P8":
        dependencies.append({"kind": "production_mesh_observer_convergence", "status": "UNRESOLVED_ADAPTER",
                             "detail": "generic current-companion adapter/common observer, then three independent mesh levels plus active-state/restart"})
    if case.block == "P6":
        dependencies.append({"kind": "recursive_branch_birth", "status": "UNRESOLVED_ADAPTER",
                             "detail": "native second-generation birth qualification"})
    runnable = all(d["status"] == "RESOLVED" for d in dependencies)
    # Preserve v1 arguments only where all v2 dependencies are resolved.  This
    # deliberately makes v2 more conservative than v1.
    args = list(case.native_args) if runnable else []
    physical_endpoint = {
        "kind": "first_accepted_fracture_event" if case.target_extension_um <= 10.0
                else "developed_extension_or_declared_physical_censor",
        "target_extension_m": None if case.target_extension_um <= 10.0 else case.target_extension_um * 1e-6,
        "no_event_exposure_ceiling_s": 8400.0 / max(float(case.rate_factor or 1.0), 1e-30),
        "event_stop_conditions": ["accepted cleavage", "accepted emission"],
        "continuation_condition": "continue only while physical endpoint and numerical guards are both open",
        "restart_policy": "atomic accepted state; populated physical marks and RNG state retained",
    }
    return {
        **case.to_dict(), "schema": SCHEMA, "goal_ids": GOALS[case.block],
        "dependencies": dependencies, "gate": None,
        "physical_endpoint": physical_endpoint,
        "numerical_guards": {"maximum_steps": 200000 if case.target_extension_um >= 1000 else 1000,
                             "capacity_preflight_required": True, "one_substantial_worker": True},
        "cavity_geometry": _cavity_geometry(case),
        "mechanics_state": {"plane_state": "plane_strain", "constraint_control": "prescribed_eps_zz",
                            "eps_zz": 0.0, "in_plane_mode": "mode_I",
                            "orientation_deg": case.orientation_deg,
                            "crack_direction": "global_positive_x"},
        "physical_contracts": {"kinetics": "native_signed", "integration": "adaptive_transactional",
                               "event_size_m": 5e-6, "mark_law": "dimensionless exponential threshold",
                               "material_identity": case.parameter_option,
                               "kernel_identity": "declared by source-bound launcher"},
        "output_consumer": {"profile": case.output_profile,
                            "selected_fields": "native selected-state writer",
                            "restart": case.output_profile == "publication_restart"},
        "command_status": "RESOLVED_DISABLED" if runnable else "UNRESOLVED_DEPENDENCIES",
        "native_args": args, "production_execution_authorized": False,
    }


def build_cases_v2(root: str | Path) -> list[dict[str, Any]]:
    return [convert_case(c) for c in build_cases(root)]


def validate_plan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [r["case_id"] for r in rows]
    errors = []
    if len(ids) != len(set(ids)):
        errors.append("duplicate case IDs")
    for row in rows:
        resolved = row["command_status"] == "RESOLVED_DISABLED"
        if resolved != bool(row["native_args"]):
            errors.append(f"{row['case_id']}: command/argument mismatch")
        if row["production_execution_authorized"]:
            errors.append(f"{row['case_id']}: production authorization true")
        if not row["goal_ids"] or not row["dependencies"]:
            errors.append(f"{row['case_id']}: missing goal/dependency")
    return {"valid": not errors, "errors": errors,
            "construction_check": "typed v2 construction only; native parser execution remains source-family qualification",
            "distinct_resolved_families": sorted({(r["backend"], r["source_location"], r["loading_mode"])
                                                   for r in rows if r["command_status"] == "RESOLVED_DISABLED"})}


def summary(rows: list[dict[str, Any]], raw_bytes: bytes | None = None) -> dict[str, Any]:
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["command_status"]] = counts.get(r["command_status"], 0) + 1
    return {"schema": SCHEMA, "case_count": len(rows), "by_command_status": counts,
            "canonicalization": "SHA-256 of sorted-key compact-JSON serialization of the entire record list",
            "canonical_semantic_sha256": hashlib.sha256(canonical).hexdigest(),
            "raw_jsonl_sha256": None if raw_bytes is None else hashlib.sha256(raw_bytes).hexdigest(),
            "production_execution_authorized": False, "validation": validate_plan(rows)}


__all__ = ["build_cases_v2", "convert_case", "summary", "validate_plan"]
