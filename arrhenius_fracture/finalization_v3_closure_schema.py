"""Prospective physical-evidence contract for the V3 closure campaign."""
from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from .finalization_v3_schema import (
    FROZEN_CASE_REGISTRY, REGISTERED_SCIENTIFIC_PREDICATES, canonical_hash,
)

SCHEMA = "v12.voiding-v5-finalization-v3-closure/1"
PARTITIONS = (1, 2, 4, 8, 16)
PHYSICAL_INPUT_FIELDS = (
    "dataset", "case_identity", "transition_identity", "partition_count", "seed",
    "loading_history", "initial_stage", "expected_terminal_classification",
    "physical_geometry", "candidate_event_identity",
)


def expected_registry_keys() -> Mapping[str, set[tuple[str, int | None]]]:
    return {
        "transitions": {(case, partition) for case in FROZEN_CASE_REGISTRY["transitions"] for partition in PARTITIONS},
        "restarts": {(case, None) for case in FROZEN_CASE_REGISTRY["restarts"]},
        "controlled": {(case, None) for case in FROZEN_CASE_REGISTRY["controlled"]},
        "neutrality": {(case, None) for case in FROZEN_CASE_REGISTRY["neutrality"]},
    }


def validate_closure_evidence(rows: Sequence[Mapping], source_rows: Mapping[str, Mapping],
                              *, executed_code_sha: str):
    errors = []
    observed = {name: set() for name in expected_registry_keys()}
    hashes: dict[str, list[Mapping]] = {}
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
        dataset = configuration.get("dataset")
        if dataset in observed:
            observed[dataset].add((configuration.get("case_identity"), configuration.get("partition_count")))
        if dataset in {"transitions", "controlled"}:
            claimed = row.get("claimed_transition")
            if not claimed or claimed != configuration.get("transition_identity"):
                errors.append((case_id, "claimed_transition"))
            if claimed not in row.get("actual_operation_trace", ()):
                errors.append((case_id, "operation_trace"))
        predicate = REGISTERED_SCIENTIFIC_PREDICATES.get(row.get("predicate_name"))
        if predicate is None or bool(predicate(row.get("predicate_inputs", {}))) != bool(row.get("predicate_result")):
            errors.append((case_id, "predicate_recomputation"))
    for digest, group in hashes.items():
        if len(group) > 1 and not all(row.get("shared_base_execution_id") for row in group):
            errors.append((digest, "aliased_physical_input", [row.get("case_id") for row in group]))
    for dataset, expected in expected_registry_keys().items():
        if observed[dataset] != expected:
            errors.append((dataset, "registry", {"missing": sorted(expected-observed[dataset]),
                                                   "extra": sorted(observed[dataset]-expected)}))
    if errors: raise ValueError(errors)
    return tuple(rows)


__all__ = ["PARTITIONS", "PHYSICAL_INPUT_FIELDS", "SCHEMA", "expected_registry_keys",
           "validate_closure_evidence"]
