"""Prospective physical-evidence contract for the V3 closure campaign."""
from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from .finalization_v3_schema import (
    FROZEN_CASE_REGISTRY, REGISTERED_SCIENTIFIC_PREDICATES, canonical_hash,
    validate_evidence_rows,
)

SCHEMA = "v12.voiding-v5-finalization-v3-closure/1"
PARTITIONS = (1, 2, 4, 8, 16)
PHYSICAL_INPUT_FIELDS = (
    "dataset", "case_identity", "transition_identity", "partition_count", "seed",
    "loading_history", "initial_stage", "expected_terminal_classification",
    "physical_geometry", "candidate_event_identity",
)
TRANSITION_TRACE_TOKENS = {
    "birth_hit_1": ("hazard_accumulation", "threshold_crossing", "threshold_renewal", "BIRTH_HIT"),
    "birth_hit_2": ("hazard_accumulation", "threshold_crossing", "EMBRYO"),
    "stabilization": ("hazard_accumulation", "threshold_crossing", "STABILIZED"),
    "healing": ("hazard_accumulation", "threshold_crossing", "HEALED"),
    "subgrid_growth": ("state_owned_growth", "inventory_debit", "SUBGRID_GROWTH"),
    "promotion": ("promotion_criterion", "explicit_cavity_geometry", "remesh", "field_projection", "equilibrium", "ownership_update", "GEOMETRIC_PROMOTION", "transaction_commit"),
    "ligament": ("cleavage_hazard", "threshold_crossing", "exact_ray_polygon_intersection", "graph_edit", "cavity_connection_update", "remesh", "field_projection", "support_rebuild", "equilibrium", "topology_certificate", "transaction_commit"),
    "downstream_child": ("fixed_cavity_surface_candidate", "hazard_accumulation", "threshold_crossing", "child_r_tip_initialization", "graph_edit", "support_rebuild", "equilibrium", "transaction_commit"),
    "child_continuation": ("child_tip_source", "child_r_tip_state", "threshold_crossing", "graph_edit", "remesh", "equilibrium", "transaction_commit"),
}
CONTROLLED_TRACE_TOKENS = {
    "centered": ("GEOMETRIC_PROMOTION", "ligament", "downstream_child", "child_continuation"),
    "positive_offset": ("ligament", "topology_certificate"),
    "negative_offset": ("ligament", "topology_certificate"),
    "short_ligament": ("ligament", "topology_certificate"),
    "long_ligament": ("ligament", "topology_certificate"),
    "diffusion_limited": ("production_integrator", "state_owned_growth", "inventory_debit"),
    "accommodation_limited": ("production_integrator", "state_owned_growth", "inventory_debit"),
    "embryo_healing": ("EMBRYO", "HEALED"),
    "downstream_zero_drive": ("ligament", "ZERO_DOWNSTREAM_DRIVE"),
    "delayed_downstream": ("ZERO_DOWNSTREAM_DRIVE", "tensile_reload", "threshold_crossing", "downstream_child"),
    "fixed_mesh_oblique": ("ligament", "downstream_child", "child_continuation"),
    "local_remesh_refinement": ("remesh", "field_projection", "equilibrium"),
}


def expected_registry_keys() -> Mapping[str, set[tuple[str, int | None]]]:
    return {
        "transitions": {(case, partition) for case in FROZEN_CASE_REGISTRY["transitions"] for partition in PARTITIONS},
        "restarts": {(case, None) for case in FROZEN_CASE_REGISTRY["restarts"]},
        "controlled": {(case, None) for case in FROZEN_CASE_REGISTRY["controlled"]},
        "neutrality": {(case, None) for case in FROZEN_CASE_REGISTRY["neutrality"]},
    }


def validate_closure_evidence(rows: Sequence[Mapping], source_rows: Mapping[str, Mapping],
                              *, executed_code_sha: str):
    validate_evidence_rows(rows, source_rows, executed_code_sha=executed_code_sha)
    errors = []
    observed = {name: set() for name in expected_registry_keys()}
    multiplicity = {name: Counter() for name in expected_registry_keys()}
    hashes: dict[str, list[Mapping]] = {}
    executions = {row.get("execution_id"): row for row in rows}
    for row in rows:
        case_id = row.get("case_id", "<missing>")
        configuration = row.get("input_configuration", {})
        missing = [field for field in PHYSICAL_INPUT_FIELDS if field not in configuration]
        if missing: errors.append((case_id, "missing_physical_input", missing))
        if row.get("input_hash") != canonical_hash(configuration):
            errors.append((case_id, "input_hash"))
        hashes.setdefault(row.get("input_hash", ""), []).append(row)
        initial = row.get("initial_state")
        if initial is None or row.get("initial_fingerprint") != canonical_hash(initial):
            errors.append((case_id, "initial_state_fingerprint"))
        terminal = row.get("terminal_state")
        if terminal is None or row.get("terminal_fingerprint") != canonical_hash(terminal):
            errors.append((case_id, "terminal_state_fingerprint"))
        if row.get("executed_code_sha") != executed_code_sha:
            errors.append((case_id, "executed_code_sha"))
        if not row.get("source_row_ids") or any(source not in source_rows for source in row.get("source_row_ids", ())):
            errors.append((case_id, "source_rows"))
        captures = [source_rows[source] for source in row.get("source_row_ids", ()) if source in source_rows]
        if not any(source.get("initial_accepted_state_fingerprint") == row.get("initial_fingerprint") and
                   source.get("terminal_accepted_state_fingerprint") == row.get("terminal_fingerprint") and
                   source.get("fingerprint_method") == "complete_accepted_state_fingerprint" for source in captures):
            errors.append((case_id, "accepted_state_source_binding"))
        dataset = configuration.get("dataset")
        if dataset in observed:
            key = (configuration.get("case_identity"), configuration.get("partition_count"))
            observed[dataset].add(key); multiplicity[dataset][key] += 1
        if dataset in {"transitions", "controlled"}:
            claimed = row.get("claimed_transition")
            if not claimed or claimed != configuration.get("transition_identity"):
                errors.append((case_id, "claimed_transition"))
            required = (TRANSITION_TRACE_TOKENS if dataset == "transitions" else CONTROLLED_TRACE_TOKENS).get(configuration.get("case_identity"), ())
            trace = tuple(row.get("actual_operation_trace", ()))
            if claimed not in trace or any(token not in trace for token in required): errors.append((case_id, "operation_trace"))
        observed_terminal = row.get("observed_terminal_classification")
        if observed_terminal != configuration.get("expected_terminal_classification"):
            errors.append((case_id, "terminal_classification"))
        predicate = REGISTERED_SCIENTIFIC_PREDICATES.get(row.get("predicate_name"))
        if predicate is None or bool(predicate(row.get("predicate_inputs", {}))) != bool(row.get("predicate_result")):
            errors.append((case_id, "predicate_recomputation"))
    for digest, group in hashes.items():
        if len(group) > 1:
            bases = {row.get("shared_base_execution_id") for row in group}
            if len(bases) != 1 or None in bases or next(iter(bases)) not in executions:
                errors.append((digest, "aliased_physical_input", [row.get("case_id") for row in group]))
            elif any(not row.get("derived_predicate") for row in group):
                errors.append((digest, "shared_base_not_derived"))
    for dataset, expected in expected_registry_keys().items():
        if observed[dataset] != expected:
            errors.append((dataset, "registry", {"missing": sorted(expected-observed[dataset]),
                                                   "extra": sorted(observed[dataset]-expected)}))
        duplicates = [key for key, count in multiplicity[dataset].items() if count != 1]
        if duplicates: errors.append((dataset, "registry_multiplicity", duplicates))
    if errors: raise ValueError(errors)
    return tuple(rows)


__all__ = ["CONTROLLED_TRACE_TOKENS", "PARTITIONS", "PHYSICAL_INPUT_FIELDS", "SCHEMA",
           "TRANSITION_TRACE_TOKENS", "expected_registry_keys",
           "validate_closure_evidence"]
