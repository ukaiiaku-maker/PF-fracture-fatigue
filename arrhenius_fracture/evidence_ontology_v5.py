"""Fail-closed provenance validation for V5 scientific evidence rows."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable, Mapping, Any, Callable

REQUIRED = (
    "case_id", "execution_id", "input_configuration", "input_hash", "actual_realized_geometry",
    "actual_geometry_fingerprint",
    "actual_operation_trace", "initial_fingerprint", "terminal_fingerprint",
    "measurement_source", "predicate_name", "predicate_inputs",
    "predicate_result", "source_row_ids", "implementation_sha",
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


PREDICATES: dict[str, Callable[[Mapping[str, Any]], bool]] = {
    "source_boolean": lambda values: bool(values["measurement"]),
    "strictly_increasing": lambda values: all(right > left for left, right in zip(values["values"], values["values"][1:])),
    "expected_sign": lambda values: values["delta"] > 0 if values["sign"] == "positive" else values["delta"] < 0,
    "finite_positive_rate": lambda values: values["rate_s"] > 0.0 and values["crossing_time_s"] > 0.0,
    "fixed_candidate_causal_step": lambda values: (
        values["candidate_before"] == values["candidate_after"]
        and values["operator_before"] == values["operator_after"]
        and values["opening_after_Pa"] > values["opening_before_Pa"]
        and values["barrier_after_J"] < values["barrier_before_J"]
        and values["rate_after_s"] > values["rate_before_s"]
        and values["crossing_after_s"] < values["crossing_before_s"]
    ),
    "inventory_balance_within_tolerance": lambda values: (
        abs((values["cavity_after_m2"] - values["cavity_before_m2"])
            + (values["available_after_m2"] - values["available_before_m2"])) <= values["tolerance_m2"]
        and abs((values["cavity_after_m2"] - values["cavity_before_m2"])
                - (values["consumed_after_m2"] - values["consumed_before_m2"])) <= values["tolerance_m2"]
    ),
    "combined_crack_void_topology_certified": lambda values: all((
        values["endpoint_matches_intersection"], values["endpoint_on_cavity_boundary"],
        values["no_surviving_solid_ligament_bridge"], values["crack_graph_outside_cavity"],
        values["wake_support_outside_cavity"], values["closed_cycle_passed"],
        values["combined_incidence_component_count"] == 1,
        not values["bridge_search_uncovered_sample_indices"],
        not values["support_triangle_cavity_overlap_element_ids"],
        not any(row["intersects_cavity_open_disk"]
                for row in values["crack_segment_cavity_intersections"]),
        len(values["combined_components"]) == 1,
        len(values["combined_incidence_edges"]) >= 1,
    )),
    "single_active_front_ownership": lambda values: (
        values["connected_graph_active_ids"] == []
        and values["connected_support_active_ids"] == []
        and values["downstream_graph_active_ids"] == ["void-front-1"]
        and values["downstream_support_active_ids"] == ["void-front-1"]
    ),
}


def _source_value(sources, binding):
    value = sources[binding["source_row_id"]]
    for component in binding["path"]:
        value = value[int(component)] if isinstance(value, (list, tuple)) else value[component]
    return value


def running_head(root: Path | None = None) -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=root or Path(__file__).resolve().parents[1], text=True).strip()


def validate_evidence_rows(rows: Iterable[Mapping[str, Any]], *,
                           source_rows: Mapping[str, Mapping[str, Any]] | None = None,
                           implementation_sha: str | None = None,
                           predicate_registry: Mapping[str, Callable[[Mapping[str, Any]], bool]] | None = None) -> tuple[dict, ...]:
    values = tuple(dict(row) for row in rows)
    errors = []
    ids = [row.get("case_id") for row in values]
    if len(ids) != len(set(ids)):
        errors.append({"code": "DUPLICATE_CASE_ID"})
    signatures = defaultdict(list)
    executions = defaultdict(list)
    sources = dict(source_rows or {str(row.get("case_id")): row for row in values})
    predicates = dict(PREDICATES if predicate_registry is None else predicate_registry)
    expected_sha = implementation_sha or running_head()
    for row in values:
        missing = [field for field in REQUIRED if field not in row]
        if missing:
            errors.append({"case_id": row.get("case_id"), "code": "MISSING_PROVENANCE", "fields": missing})
            continue
        case_id = row["case_id"]
        if canonical_hash(row["input_configuration"]) != row["input_hash"]:
            errors.append({"case_id": case_id, "code": "INPUT_HASH_MISMATCH"})
        if canonical_hash(row["actual_realized_geometry"]) != row["actual_geometry_fingerprint"]:
            errors.append({"case_id": case_id, "code": "GEOMETRY_FINGERPRINT_MISMATCH"})
        if row["implementation_sha"] != expected_sha:
            errors.append({"case_id": case_id, "code": "IMPLEMENTATION_SHA_MISMATCH"})
        missing_sources = [source for source in row["source_row_ids"] if source not in sources]
        if not row["measurement_source"] or not row["source_row_ids"] or missing_sources:
            errors.append({"case_id": case_id, "code": "UNTRACEABLE_MEASUREMENT", "missing": missing_sources})
        predicate = predicates.get(row["predicate_name"])
        if predicate is None or not isinstance(row["predicate_inputs"], Mapping):
            errors.append({"case_id": row["case_id"], "code": "LITERAL_OR_UNNAMED_PREDICATE"})
        else:
            try:
                inputs = dict(row["predicate_inputs"])
                for alias, binding in inputs.pop("source_bindings", {}).items():
                    inputs[alias] = _source_value(sources, binding)
                recomputed = bool(predicate(inputs))
            except Exception as error:
                errors.append({"case_id": case_id, "code": "PREDICATE_RECOMPUTE_ERROR",
                               "error": type(error).__name__, "message": str(error)})
            else:
                if recomputed != bool(row["predicate_result"]):
                    errors.append({"case_id": case_id, "code": "PREDICATE_RESULT_MISMATCH"})
        claimed = row.get("claimed_transition")
        if claimed and claimed not in row["actual_operation_trace"]:
            errors.append({"case_id": row["case_id"], "code": "TRANSITION_NOT_EXECUTED", "transition": claimed})
        stage = row.get("restart_stage")
        if stage and row.get("checkpoint_stage") != stage:
            errors.append({"case_id": row["case_id"], "code": "RESTART_STAGE_ALIASED"})
        peers = tuple(row.get("independent_peer_execution_ids", ()))
        if row.get("evidence_type") == "neutrality" and (len(set(peers)) != 2 or row["execution_id"] in peers):
            errors.append({"case_id": row["case_id"], "code": "NEUTRALITY_PEERS_NOT_EXECUTED"})
        if row.get("evidence_type") == "derivative" and len(row["source_row_ids"]) < 2:
            errors.append({"case_id": row["case_id"], "code": "DERIVATIVE_WITHOUT_SOURCE_ROWS"})
        signature = (row["input_hash"], row["initial_fingerprint"], row["terminal_fingerprint"])
        signatures[signature].append(row)
        executions[row["execution_id"]].append(row)
    for execution_id, group in executions.items():
        if len({row.get("input_hash") for row in group}) > 1:
            errors.append({"execution_id": execution_id, "code": "EXECUTION_RELABELED_WITH_DIFFERENT_INPUTS"})
    known_executions = set(executions)
    for row in values:
        if row.get("evidence_type") == "neutrality":
            missing_peers = set(row.get("independent_peer_execution_ids", ())) - known_executions
            if missing_peers:
                errors.append({"case_id": row.get("case_id"), "code": "NEUTRALITY_PEERS_NOT_ACCESSIBLE",
                               "missing": sorted(missing_peers)})
    for group in signatures.values():
        if len(group) > 1:
            if not all(row.get("equality_classification") == "EXPECTED_EQUIVALENCE" for row in group):
                errors.append({"case_ids": [row["case_id"] for row in group], "code": "ALIASED_VARIANT_EXECUTION"})
            elif len({row["execution_id"] for row in group}) != len(group):
                errors.append({"case_ids": [row["case_id"] for row in group], "code": "EXPECTED_EQUIVALENCE_NOT_INDEPENDENT"})
    if errors:
        raise ValueError(errors)
    return values


__all__ = ["PREDICATES", "REQUIRED", "canonical_hash", "running_head", "validate_evidence_rows"]
