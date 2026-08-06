"""Stable, restart-safe outputs for v11 mechanistic branching campaigns."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


TRIAL_FIELDS = (
    "step", "physical_time_s", "accepted_state_id", "trial_id", "action_type",
    "participating_front_ids", "candidate_ids", "pending_event_ids", "completion_times_s",
    "correlation_time_difference_s", "signed_directional_J_J_per_m2",
    "positive_directional_J_J_per_m2", "directional_K_Pa_sqrt_m",
    "proposed_arm_lengths_m", "realized_arm_lengths_m", "pretrial_potential_energy_J_per_m",
    "posttrial_potential_energy_J_per_m", "released_energy_J_per_m",
    "hazard_derived_cost_per_arm_J_per_m", "total_dissipative_cost_J_per_m",
    "net_energy_margin_J_per_m", "relative_energy_residual", "geometry_status",
    "equilibrium_status", "provider_identity", "accepted", "veto_reason",
    "reservation_result", "consumption_result", "pretrial_state_hash", "postrollback_state_hash",
)
BRANCH_EVENT_FIELDS = (
    "event_record_id", "step", "branch_junction", "parent_front", "arm_front_ids",
    "arm_directions", "plane_identities", "event_ids_consumed", "completion_time_difference_s",
    "arm_lengths_m", "tip_positions_m", "tip_separation_m", "shared_cluster_id",
    "shared_cluster_state_hash", "topology_fingerprint", "released_energy_J_per_m",
    "total_cost_J_per_m", "energy_margin_J_per_m",
)
CLUSTER_FIELDS = ("step", "cluster_id", "state_hash", "unresolved", "tip_separation_m", "handoff_required")
FRONT_FIELDS = ("step", "front_id", "parent_front_id", "status", "termination_reason", "tip_x_m", "tip_y_m", "arclength_m")
PROVIDER_FIELDS = ("step", "from_provider", "to_provider", "state_hash", "topology_fingerprint", "parity_passed", "residuals")
ENERGY_FIELDS = ("step", "accepted_state_id", "stored_energy_J_per_m", "released_energy_J_per_m", "dissipative_cost_J_per_m", "residual_J_per_m")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return value


class BranchOutputWriter:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def append_trial(self, record: Mapping[str, Any]) -> None:
        missing = set(TRIAL_FIELDS).difference(record)
        unknown = set(record).difference(TRIAL_FIELDS)
        if missing or unknown:
            raise ValueError(f"invalid trial record; missing={sorted(missing)} unknown={sorted(unknown)}")
        path = self.root / "branch_action_trials.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def append_csv(self, name: str, fields: tuple[str, ...], record: Mapping[str, Any], *, unique_key: str | None = None) -> bool:
        if set(record) != set(fields):
            raise ValueError(f"{name} record does not match stable header")
        path = self.root / name
        if unique_key and path.exists():
            with path.open(newline="") as stream:
                if any(row[unique_key] == str(record[unique_key]) for row in csv.DictReader(stream)):
                    return False
        new = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if new:
                writer.writeheader()
            writer.writerow({key: _cell(record[key]) for key in fields})
            stream.flush()
            os.fsync(stream.fileno())
        return True

    def branch_event(self, record: Mapping[str, Any]) -> bool:
        return self.append_csv("branch_events.csv", BRANCH_EVENT_FIELDS, record, unique_key="event_record_id")

    def cluster(self, record: Mapping[str, Any]) -> bool:
        return self.append_csv("branch_clusters.csv", CLUSTER_FIELDS, record)

    def front(self, record: Mapping[str, Any]) -> bool:
        return self.append_csv("fronts.csv", FRONT_FIELDS, record)

    def provider_transition(self, record: Mapping[str, Any]) -> bool:
        return self.append_csv("provider_transitions.csv", PROVIDER_FIELDS, record)

    def energy(self, record: Mapping[str, Any]) -> bool:
        return self.append_csv("energy_ledger.csv", ENERGY_FIELDS, record)

    def complete(self, *, status: str, final_checkpoint: str, validation: Mapping[str, Any]) -> None:
        allowed = {
            "target_reached", "branch_cluster_independent_tip_handoff_required",
            "physical_veto_no_branch", "numerical_failure",
        }
        if status not in allowed:
            raise ValueError("unsupported v11 completion status")
        _atomic_json(self.root / "run_complete.json", {
            "schema": "v11.branching-run-complete/1", "status": status,
            "final_checkpoint": final_checkpoint, "validation": dict(validation),
        })


__all__ = [
    "BRANCH_EVENT_FIELDS", "CLUSTER_FIELDS", "ENERGY_FIELDS", "FRONT_FIELDS",
    "PROVIDER_FIELDS", "TRIAL_FIELDS", "BranchOutputWriter",
]
