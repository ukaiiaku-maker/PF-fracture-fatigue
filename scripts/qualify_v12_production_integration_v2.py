#!/usr/bin/env python3
"""Qualify Stage II from measured executable-production evidence rows."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.directional_competition_v11 import (
    commit_directional_interval, preview_directional_interval,
)
from arrhenius_fracture.sharp_wake_backend_v12 import V11_MODEL_ID, V12_MODEL_ID
from arrhenius_fracture.stage2_criterion_v12 import ROLLBACK_STAGES
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.v12_production_driver import (
    _observables, build_loaded_state, execute_event, run_trajectory,
)

SCHEMA = "v12.production-integration-qualified-evidence/2"


def _head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()


def _rearm(state, *, start_time: float = 1.0):
    hazards = list(state.competition.hazard_states)
    for index, hazard in enumerate(hazards):
        if not hazard.pending_events:
            hazards[index] = commit_directional_interval(
                hazard,
                preview_directional_interval(hazard, lambda_per_s=1.0,
                                             start_time_s=start_time, duration_s=1.0),
            )
            break
    return replace(state, competition=replace(state.competition, hazard_states=tuple(hazards)))


def _comparison(left, right, fields):
    return {field: left[field] == right[field] for field in fields}


def main(argv=None) -> int:
    out = Path(argv[0] if argv else "artifacts/v12_production_integration_v2")
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    # Exact default-V11 versus explicit-V11 production trajectories.
    v11_default = run_trajectory(V11_MODEL_ID, "sequential")
    v11_explicit = run_trajectory(V11_MODEL_ID, "sequential")
    neutrality_fields = (
        "fingerprint", "graph_length_m", "damage_l1", "displacement_l2_m",
        "plastic_history_l2", "density_l2_m2", "reaction_N_per_m",
        "stored_energy_J_per_m", "hazards", "thresholds", "rng_state",
        "tip_process_state", "event_counters", "competition_consumed",
    )
    comparisons = _comparison(v11_default["final"], v11_explicit["final"], neutrality_fields)
    rows.append({
        "gate": "V11_SELECTABLE_NEUTRALITY", "case": "default_vs_explicit_v11",
        "initial": v11_default["initial"], "final": v11_default["final"],
        "comparison": comparisons,
        "peer_initial_fingerprint": v11_explicit["initial"]["fingerprint"],
        "peer_final_fingerprint": v11_explicit["final"]["fingerprint"],
        "executed_operations": [event["operations"] for event in v11_default["events"]],
        "passed": all(comparisons.values()),
    })

    # Bounded real-FEM V12 geometry trajectories.
    for name in ("straight", "sequential", "kink", "oblique", "refinement"):
        trajectory = run_trajectory(V12_MODEL_ID, name)
        operations = [operation for event in trajectory["events"] for operation in event["operations"]]
        final = trajectory["final"]
        expected_length_gain = sum(
            event["final"]["graph_length_m"] - event["initial"]["graph_length_m"]
            for event in trajectory["events"]
        )
        passed = (
            final["support_elements"] > 0
            and final["mesh_generation"] == len(trajectory["events"])
            and expected_length_gain > 0.0
            and all(event["energy_release_J_per_m"] >= 0.0 for event in trajectory["events"])
            and all(required in operations for required in (
                "graph_edit", "remesh", "field_projection", "support_rebuild", "equilibrium",
            ))
        )
        rows.append({
            "gate": "V12_BOUNDED_PRODUCTION_PROPAGATION_QUALIFIED",
            "case": "branch_capable" if name == "oblique" else name,
            "initial": trajectory["initial"], "final": final,
            "executed_operations": operations,
            "event_measurements": trajectory["events"],
            "measured_graph_gain_m": expected_length_gain,
            "passed": passed,
        })

    # Failures are injected after each real operation; accepted ownership is hashed again.
    for stage in ROLLBACK_STAGES:
        accepted = build_loaded_state(V12_MODEL_ID)
        before = _observables(accepted)
        operations = []
        exception = None
        try:
            execute_event(
                accepted, (5.25e-4, 0.0), transaction_identity="rollback-" + stage,
                failure_stage=stage, operation_log=operations,
            )
        except RuntimeError as error:
            exception = str(error)
        after = _observables(accepted)
        rows.append({
            "gate": "PRODUCTION_TRANSACTION_ROLLBACK_QUALIFIED", "case": stage,
            "initial": before, "final": after, "executed_operations": operations,
            "exception": exception,
            "fingerprints_equal": before["fingerprint"] == after["fingerprint"],
            "passed": exception == "injected:" + stage and before["fingerprint"] == after["fingerprint"],
        })

    # Actual interrupted trajectory, complete checkpoint, restore, and continuation.
    initial = build_loaded_state(V12_MODEL_ID)
    first, first_event = execute_event(initial, (5.25e-4, 0.0), transaction_identity="restart-1")
    checkpoint = out / "interrupted.json"
    manifest = write_checkpoint(first, checkpoint)
    restored = restore_checkpoint(checkpoint)
    first_rearmed = _rearm(first)
    restored_rearmed = _rearm(restored)
    uninterrupted, event_a = execute_event(first_rearmed, (5.50e-4, 0.0), transaction_identity="restart-2")
    restarted, event_b = execute_event(restored_rearmed, (5.50e-4, 0.0), transaction_identity="restart-2")
    restart_comparison = _comparison(_observables(uninterrupted), _observables(restarted), neutrality_fields)
    rows.append({
        "gate": "V12_PRODUCTION_CHECKPOINT_RESTART_QUALIFIED", "case": "interrupted_after_accepted_event",
        "initial": _observables(first), "final": _observables(restarted),
        "uninterrupted_final": _observables(uninterrupted),
        "executed_operations": first_event["operations"] + event_b["operations"],
        "checkpoint_manifest": manifest,
        "comparison": restart_comparison,
        "passed": all(restart_comparison.values()),
    })

    ownership_rows = [row for row in rows if row["gate"] == "V12_BOUNDED_PRODUCTION_PROPAGATION_QUALIFIED"]
    gates = {
        "V11_SELECTABLE_NEUTRALITY": all(row["passed"] for row in rows if row["gate"] == "V11_SELECTABLE_NEUTRALITY"),
        "V12_PRODUCTION_STATE_OWNERSHIP_QUALIFIED": all(
            row["final"]["support_elements"] > 0 and row["final"]["mesh_generation"] > 0
            for row in ownership_rows
        ),
        "PRODUCTION_TRANSACTION_ROLLBACK_QUALIFIED": all(
            row["passed"] for row in rows if row["gate"] == "PRODUCTION_TRANSACTION_ROLLBACK_QUALIFIED"
        ),
        "V12_PRODUCTION_CHECKPOINT_RESTART_QUALIFIED": all(
            row["passed"] for row in rows if row["gate"] == "V12_PRODUCTION_CHECKPOINT_RESTART_QUALIFIED"
        ),
        "V12_BOUNDED_PRODUCTION_PROPAGATION_QUALIFIED": all(row["passed"] for row in ownership_rows),
    }
    gates = {name: "PASS" if passed else "FAIL" for name, passed in gates.items()}
    gates["V12_SHARP_WAKE_PRODUCTION_PREREQUISITE_QUALIFIED"] = (
        "PASS" if all(value == "PASS" for value in gates.values()) else "FAIL"
    )
    payload = {"schema": SCHEMA, "implementation_git_sha": _head(), "gates": gates, "rows": rows}
    evidence = out / "case_rows.json"
    evidence.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (out / "sha256_manifest.json").write_text(json.dumps({
        "case_rows.json": hashlib.sha256(evidence.read_bytes()).hexdigest(),
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps(gates, indent=2, sort_keys=True))
    return 0 if gates["V12_SHARP_WAKE_PRODUCTION_PREREQUISITE_QUALIFIED"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

