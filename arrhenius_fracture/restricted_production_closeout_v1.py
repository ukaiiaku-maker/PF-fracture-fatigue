"""Small structured validator for restricted-production closeout evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .resolved_sigma_tt_certificate_v1 import (
    INVALID_COMPARISON, NOT_EVALUATED, PASS, SCIENTIFICALLY_UNAVAILABLE,
)

SCHEMA = "unified-2d.restricted-production-closeout/1"
VACANCY_OPERATIONS = (
    "AVAILABLE_SITE", "VACANCY_INVENTORY_BIRTH", "EMBRYO_STABILIZED",
    "PROMOTION", "RESOLVED_VACANCY_GROWTH",
    "LIGAMENT_CONNECTION", "DOWNSTREAM_SOURCE_REFINEMENT",
    "DOWNSTREAM_CHILD", "CHILD_CONTINUATION",
)
DIRECT_OPERATIONS = (
    "AVAILABLE_SITE", "DIRECT_INTERACTION_BIRTH", "EMBRYO_STABILIZED",
    "STABLE_SUBGRID_GROWTH", "PROMOTION",
    "RESOLVED_SOURCE_UNAVAILABLE_ENFORCED",
)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str,
    ).encode()).hexdigest()


def authenticate_stage(stage: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(stage)
    result["authentication_sha256"] = _sha(result)
    return result


def authenticate_operation_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate the measured result of an executed production operation."""
    result = dict(evidence)
    result["authentication_sha256"] = _sha(result)
    return result


def _authentic(stage: Mapping[str, Any]) -> bool:
    body = {key: value for key, value in stage.items() if key != "authentication_sha256"}
    return stage.get("authentication_sha256") == _sha(body)


@dataclass(frozen=True)
class LifecycleValidation:
    path: str
    valid: bool
    predicates: dict[str, bool]
    failed_predicates: tuple[str, ...]
    evidence_reference: str | None
    stages: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]: return asdict(self)


def validate_lifecycle_trace(path: str, stages: Sequence[Mapping[str, Any]] | None) -> LifecycleValidation:
    expected = VACANCY_OPERATIONS if path == "VACANCY_MEDIATED" else (
        DIRECT_OPERATIONS if path == "DIRECT_DISLOCATION" else ())
    rows = tuple(dict(stage) for stage in (stages or ()))
    required = {
        "run_id", "source_identity", "physical_time_s", "operation",
        "input_state_fingerprint", "output_state_fingerprint", "site_id",
        "embryo_id", "cavity_id", "front_id", "active_mechanism",
        "production_api", "inventory_balance", "representation", "topology_state",
        "owner_replaced", "phase_assignment_only", "evidence_kind",
        "authentication_sha256", "operation_evidence",
    }
    shapes = bool(rows) and all(required.issubset(row) for row in rows)
    operations = tuple(row.get("operation") for row in rows)
    same_run = bool(rows) and len({row.get("run_id") for row in rows}) == 1
    same_source = bool(rows) and len({_sha(row.get("source_identity")) for row in rows}) == 1
    times = [row.get("physical_time_s") for row in rows]
    finite_monotone_time = bool(rows) and all(
        isinstance(value, (int, float)) and math.isfinite(float(value)) for value in times
    ) and all(float(after) >= float(before) for before, after in zip(times, times[1:]))
    handoffs = bool(rows) and all(
        before.get("output_state_fingerprint") == after.get("input_state_fingerprint")
        for before, after in zip(rows, rows[1:]))
    identity_fields = ("site_id", "embryo_id", "cavity_id")
    lineage = bool(rows) and all(len({row.get(field) for row in rows if row.get(field) is not None}) <= 1
                                 for field in identity_fields)
    no_owner_replacement = bool(rows) and all(row.get("owner_replaced") is False for row in rows)
    physical_transitions = bool(rows) and all(row.get("phase_assignment_only") is False for row in rows)
    production_apis = bool(rows) and all(
        isinstance(row.get("production_api"), str)
        and row["production_api"].startswith("arrhenius_fracture.") for row in rows)
    inventory = bool(rows) and all(
        isinstance(row.get("inventory_balance"), Mapping)
        and row["inventory_balance"].get("nonnegative") is True
        and abs(float(row["inventory_balance"].get("residual_m2", math.inf))) <=
            float(row["inventory_balance"].get("roundoff_budget_m2", -1.0))
        for row in rows)
    operation_receipts = bool(rows) and all(
        isinstance(row.get("operation_evidence"), Mapping)
        and _authentic(row["operation_evidence"])
        and row["operation_evidence"].get("executed") is True
        and row["operation_evidence"].get("operation") == row.get("operation")
        for row in rows)

    def row_for(operation: str) -> Mapping[str, Any]:
        return next((row for row in rows if row.get("operation") == operation), {})

    def scheduler_audits(proof: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        candidates = []
        single = proof.get("scheduler_audit")
        if isinstance(single, Mapping): candidates.append(single)
        many = proof.get("scheduler_audits", ())
        if isinstance(many, Sequence) and not isinstance(many, (str, bytes)):
            candidates.extend(item for item in many if isinstance(item, Mapping))
        return tuple(candidates)

    def scheduler_stage(operation: str, selected_event: str | None = None) -> bool:
        proof = row_for(operation).get("operation_evidence", {})
        audits = scheduler_audits(proof)
        if not audits:
            events = proof.get("scheduler_events", ())
            return bool(events) and (selected_event is None or any(
                item.get("selected_event") == selected_event for item in events
                if isinstance(item, Mapping)))
        structurally_bound = all(
            audit.get("schema") == "unified-2d.global-physical-event-scheduler/1"
            and isinstance(audit.get("events"), Sequence)
            and audit.get("preview_nonmutating") is True
            and audit.get("no_owner_advanced_twice") is True
            for audit in audits)
        if not structurally_bound: return False
        return selected_event is None or any(
            event.get("selected_event") == selected_event
            for audit in audits for event in audit["events"]
            if isinstance(event, Mapping))

    if path == "VACANCY_MEDIATED":
        scheduler_bound = all((
            scheduler_stage("VACANCY_INVENTORY_BIRTH", "VACANCY_BIRTH_HIT"),
            scheduler_stage("EMBRYO_STABILIZED", "STABILIZATION"),
            scheduler_stage("PROMOTION", "GROWTH_PROMOTION"),
            scheduler_stage("RESOLVED_VACANCY_GROWTH"),
            scheduler_stage("LIGAMENT_CONNECTION", "LIGAMENT_RUPTURE"),
            scheduler_stage("DOWNSTREAM_CHILD", "DOWNSTREAM_FIRST_PASSAGE"),
            scheduler_stage("CHILD_CONTINUATION", "CHILD_CONTINUATION"),
        ))
    elif path == "DIRECT_DISLOCATION":
        scheduler_bound = all((
            scheduler_stage("DIRECT_INTERACTION_BIRTH", "DISLOCATION_INTERACTION_HIT"),
            scheduler_stage("EMBRYO_STABILIZED", "STABILIZATION"),
            scheduler_stage("STABLE_SUBGRID_GROWTH"),
            scheduler_stage("PROMOTION", "GROWTH_PROMOTION"),
        ))
    else:
        scheduler_bound = False

    def positive_growth(operation: str) -> bool:
        row = row_for(operation); proof = row.get("operation_evidence", {})
        delta = proof.get("measured_growth_delta", {})
        mechanisms = delta.get("mechanism_area_increments_m2", {})
        try:
            area = float(delta["area_after_m2"])-float(delta["area_before_m2"])
            radius = float(delta["radius_after_m"])-float(delta["radius_before_m"])
            duration = float(proof["committed_duration_s"])
            resolved = math.fsum(float(value) for value in mechanisms.values())
            residual = float(delta["conservation_residual_m2"])
            budget = float(delta["conservation_budget_m2"])
        except (KeyError, TypeError, ValueError):
            return False
        return (duration > 0.0 and area > 0.0 and radius > 0.0 and bool(mechanisms)
                and math.isclose(resolved, area, rel_tol=2e-9, abs_tol=budget)
                and abs(residual) <= budget
                and proof.get("committed_event_type") not in {None, "EQUILIBRATION"})

    growth_operations = (("RESOLVED_VACANCY_GROWTH",)
                         if path == "VACANCY_MEDIATED" else
                         ("STABLE_SUBGRID_GROWTH",))
    measured_growth = bool(rows) and all(positive_growth(name) for name in growth_operations)
    refinement = row_for("DOWNSTREAM_SOURCE_REFINEMENT")
    refinement_proof = refinement.get("operation_evidence", {})
    refinement_handoff = path != "VACANCY_MEDIATED" or bool(refinement) and (
        refinement.get("input_state_fingerprint") != refinement.get("output_state_fingerprint")
        and refinement_proof.get("committed_event_type") == "REPRESENTATION_TRANSFER"
        and refinement_proof.get("physical_identity_preserved") is True
        and refinement_proof.get("kinetic_owners_preserved") is True
        and refinement_proof.get("transfer_checks_passed") is True)
    vacancy_semantics = path != "VACANCY_MEDIATED" or bool(rows) and all(
        row.get("active_mechanism") in {"VACANCY_CLUSTERING", "VACANCY_TRANSPORT_GROWTH", "CANONICAL_FRACTURE"}
        for row in rows)
    direct_semantics = path != "DIRECT_DISLOCATION" or bool(rows) and (
        rows[1].get("active_mechanism") == "DIRECT_DISLOCATION_INTERACTION"
        and rows[1]["inventory_balance"].get("vacancy_capture_required") is False
        and row_for("STABLE_SUBGRID_GROWTH").get("inventory_balance", {}).get(
            "direct_birth_hits_are_cavity_loops") is False
        and row_for("RESOLVED_SOURCE_UNAVAILABLE_ENFORCED").get(
            "active_mechanism") == "CAVITY_SURFACE_LOOP_EMISSION"
    )
    downstream_owner = path != "VACANCY_MEDIATED" or bool(rows) and all(
        row.get("production_api") == "arrhenius_fracture.voiding_production_v5.downstream_front_transaction"
        for row in rows if row.get("operation") in {"DOWNSTREAM_CHILD", "CHILD_CONTINUATION"})
    unavailable = row_for("RESOLVED_SOURCE_UNAVAILABLE_ENFORCED")
    consumer = unavailable.get("operation_evidence", {}).get("consumer_result", {})
    unavailable_enforced = path != "DIRECT_DISLOCATION" or bool(unavailable) and (
        isinstance(consumer, Mapping) and _authentic(consumer)
        and consumer.get("mechanism_requested") is True
        and consumer.get("source_status") == "UNAVAILABLE_NO_RATE"
        and consumer.get("production_rate_s") is None
        and bool(consumer.get("actual_reason"))
        and consumer.get("accepted_state_unchanged") is True
        and unavailable.get("input_state_fingerprint") == unavailable.get("output_state_fingerprint")
        and unavailable["inventory_balance"].get("emission_rate_supplied") is False)
    predicates = {
        "nonempty_complete_stage_records": shapes,
        "authenticated_stage_records": bool(rows) and all(_authentic(row) for row in rows),
        "exact_required_operation_sequence": operations == expected,
        "common_run_identity": same_run,
        "common_source_identity": same_source,
        "finite_monotone_physical_time": finite_monotone_time,
        "accepted_state_handoffs": handoffs,
        "site_embryo_cavity_lineage": lineage,
        "no_mechanism_owner_replacement": no_owner_replacement,
        "physical_transition_not_phase_assignment": physical_transitions,
        "production_api_ownership": production_apis,
        "inventory_conservation": inventory,
        "authenticated_executed_operation_receipts": operation_receipts,
        "global_scheduler_common_clock_receipts": scheduler_bound,
        "growth_stages_have_measured_positive_mechanism_resolved_delta": measured_growth,
        "downstream_refinement_is_explicit_accepted_handoff": refinement_handoff,
        "vacancy_mechanism_semantics": vacancy_semantics,
        "direct_and_cavity_loop_semantics_distinct": direct_semantics,
        "canonical_downstream_fracture_owner": downstream_owner,
        "unavailable_resolved_source_cannot_supply_rate": unavailable_enforced,
    }
    failed = tuple(name for name, value in predicates.items() if not value)
    reference = _sha(rows) if rows else None
    return LifecycleValidation(path, not failed, predicates, failed, reference, rows)


@dataclass(frozen=True)
class RestrictedCloseoutDecision:
    gates: dict[str, str]
    framework: str
    evidence_references: dict[str, str | None]
    schema: str = SCHEMA

    def to_dict(self) -> dict[str, Any]: return asdict(self)


def aggregate_restricted_closeout(certificate: Any, vacancy: LifecycleValidation,
                                  direct: LifecycleValidation) -> RestrictedCloseoutDecision:
    status = getattr(certificate, "terminal_scientific_availability", NOT_EVALUATED)
    comparison_valid = status in {PASS, SCIENTIFICALLY_UNAVAILABLE} and all(
        getattr(certificate, "comparison_prerequisites", {}).values())
    science_distinct = status in {PASS, SCIENTIFICALLY_UNAVAILABLE}
    restricted = (status != SCIENTIFICALLY_UNAVAILABLE or (
        direct.predicates.get("unavailable_resolved_source_cannot_supply_rate", False)))
    gates = {
        "COMPARISON_VALIDITY": PASS if comparison_valid else status,
        "INVALID_COMPARISONS_REJECTED": PASS,
        "SCIENTIFIC_UNAVAILABILITY_DISTINCT_FROM_INVALIDITY": PASS if science_distinct else "FAIL",
        "VACANCY_LIFECYCLE_EVIDENCE_BINDING": PASS if vacancy.valid else "FAIL",
        "DIRECT_LIFECYCLE_EVIDENCE_BINDING": PASS if direct.valid else "FAIL",
        "RESTRICTED_CAPABILITY_ENFORCEMENT": PASS if restricted else "FAIL",
    }
    complete = all(value == PASS for value in gates.values())
    framework = (PASS if complete and status == PASS else
        "PASS_WITH_RESOLVED_LOOP_EMISSION_SCIENTIFICALLY_UNAVAILABLE"
        if complete and status == SCIENTIFICALLY_UNAVAILABLE else "PENDING_OR_FAILED_CLOSEOUT_GATES")
    return RestrictedCloseoutDecision(gates, framework, {
        "comparison": getattr(certificate, "source_coordinate_fingerprint", None),
        "vacancy_lifecycle": vacancy.evidence_reference,
        "direct_lifecycle": direct.evidence_reference,
    })


__all__ = ["VACANCY_OPERATIONS", "DIRECT_OPERATIONS", "LifecycleValidation",
    "RestrictedCloseoutDecision", "authenticate_stage", "authenticate_operation_evidence",
    "validate_lifecycle_trace",
    "aggregate_restricted_closeout"]
