"""Front- and owner-resolved canonical V12 output rows."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .general_multifront_v12 import FrontCandidateObservation, MultiFrontRuntimeState


FRONT_FIELDS = (
    "front_id", "parent_front_id", "junction_lineage", "process_owner_id",
    "process_region_member_ids", "candidate_id", "selected_event_tip_id",
    "controlling_scalar_K_tip_id", "tensor_probe_tip_id", "accepted_state_id",
    "stress_field_state_id", "local_process_coordinate_m",
    "maximum_network_forward_reach_m", "front_projected_extension_m",
    "front_arclength_m", "daughter_length_from_birth_m",
)
OWNER_FIELDS = (
    "owner_id", "member_front_ids", "controlling_front_id",
    "process_update_count", "event_renewal_count", "local_process_coordinate_m",
    "active_ledgers", "wake_ledgers", "reservoir_transfers",
)


def _lineage(state: MultiFrontRuntimeState, front_id: str) -> list[str]:
    branch = state.crack_network.branch(front_id)
    junction_by_parent = {
        item.parent_branch_id: item.junction_id for item in state.junctions.values()
    }
    result = []
    while branch.parent_branch_id is not None:
        parent = state.crack_network.branch(branch.parent_branch_id)
        if parent.branch_id in junction_by_parent:
            result.append(junction_by_parent[parent.branch_id])
        branch = parent
    return list(reversed(result))


def front_observation_row(
    state: MultiFrontRuntimeState, observation: FrontCandidateObservation, *,
    selected_event_tip_id: str | None,
) -> dict[str, Any]:
    front = state.crack_network.branch(observation.front_id)
    owner_id = state.owner_by_front[observation.front_id]
    region = state.process_regions[owner_id]
    root_x = state.crack_network.branch(state.crack_network.primary_branch_id).root[0]
    maximum_reach = max(
        state.crack_network.branch(item).tip[0] - root_x
        for item in state.active_front_ids
    )
    return {
        "front_id": front.branch_id,
        "parent_front_id": front.parent_branch_id,
        "junction_lineage": _lineage(state, front.branch_id),
        "process_owner_id": owner_id,
        "process_region_member_ids": sorted(region.member_front_ids),
        "candidate_id": observation.candidate_id,
        "selected_event_tip_id": selected_event_tip_id,
        "controlling_scalar_K_tip_id": observation.controlling_scalar_K_tip_id,
        "tensor_probe_tip_id": observation.tensor_probe_tip_id,
        "accepted_state_id": observation.accepted_state_id,
        "stress_field_state_id": observation.stress_field_state_id,
        "local_process_coordinate_m": region.cumulative_process_advance_m,
        "maximum_network_forward_reach_m": maximum_reach,
        "front_projected_extension_m": front.projected_extension_m,
        "front_arclength_m": front.physical_path_length_m,
        "daughter_length_from_birth_m": (
            0.0 if front.parent_branch_id is None else front.physical_path_length_m
        ),
    }


def owner_row(
    state: MultiFrontRuntimeState, owner_id: str, *, controlling_front_id: str,
) -> dict[str, Any]:
    region = state.process_regions[owner_id]
    engine = state.process_engines[region.process_engine_id]
    transfers = (
        [] if region.source_reservoir_id is None
        else [region.source_reservoir_id]
    )
    return {
        "owner_id": owner_id,
        "member_front_ids": sorted(region.member_front_ids),
        "controlling_front_id": controlling_front_id,
        "process_update_count": engine.update_count,
        "event_renewal_count": engine.event_renewal_count,
        "local_process_coordinate_m": engine.local_process_coordinate_m,
        "active_ledgers": dict(engine.active_ledgers),
        "wake_ledgers": dict(engine.wake_ledgers),
        "reservoir_transfers": sorted(transfers),
    }


def canonical_csv_bytes(fields: Iterable[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    import io
    stream = io.StringIO(newline="")
    fieldnames = tuple(fields)
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    normalized = []
    for source in rows:
        row = {}
        for field in fieldnames:
            value = source.get(field)
            row[field] = (
                json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
                if isinstance(value, (dict, list, tuple)) else value
            )
        normalized.append(row)
    for row in sorted(
        normalized,
        key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False),
    ):
        writer.writerow(row)
    return stream.getvalue().encode()


def write_canonical_csv(
    path: str | Path, fields: Iterable[str], rows: Iterable[Mapping[str, Any]],
) -> None:
    Path(path).write_bytes(canonical_csv_bytes(fields, rows))


__all__ = [
    "FRONT_FIELDS", "OWNER_FIELDS", "canonical_csv_bytes", "front_observation_row",
    "owner_row", "write_canonical_csv",
]
