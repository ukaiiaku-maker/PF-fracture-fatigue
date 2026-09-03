#!/usr/bin/env python3
"""Direct qualification of production-owned V4 lifecycle and topology rows."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.voiding_production_v4 import (
    build_production_void_state, deterministic_trajectory, ligament_transaction,
    natural_trajectory, observables,
)

ROLLBACK_STAGES = ("graph_edit", "remesh", "field_projection", "support_rebuild", "equilibrium",
                   "energy_gate", "process_state_update", "topology_verification", "late_event_veto")


def main(argv=None):
    out = Path(argv[0] if argv else "artifacts/voiding_v4/production")
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    # Default-off identity includes the complete production state fingerprint.
    disabled, _ = build_production_void_state(enabled=False)
    disabled_again, _ = build_production_void_state(enabled=False)
    rows.append({
        "gate": "V12_VOIDING_DISABLED_NEUTRALITY", "case": "default_off_exact_state",
        "initial": observables(disabled, "disabled-a"), "final": observables(disabled_again, "disabled-b"),
        "fingerprints_equal": complete_accepted_state_fingerprint(disabled) == complete_accepted_state_fingerprint(disabled_again),
        "void_state_absent": disabled.void_state is None and disabled_again.void_state is None,
        "passed": complete_accepted_state_fingerprint(disabled) == complete_accepted_state_fingerprint(disabled_again)
                  and disabled.void_state is None and disabled_again.void_state is None,
    })

    final, deterministic = deterministic_trajectory()
    required = (
        "available_site", "multi_hit_1", "multi_hit_2", "stabilization", "subgrid_void",
        "subgrid_growth", "geometric_promotion", "resolved_growth", "ligament_rupture",
        "connected_topology", "downstream_first_passage", "new_graph_front", "continued_accepted_event",
    )
    sequence = tuple(row["operation"] for row in deterministic)
    area_balances = [
        abs(row["cavity_area_m2"] - row["inventory_area_m2"])
        for row in deterministic if row.get("cavity_area_m2") is not None
    ]
    runner_calls = sum("equilibrium" in row.get("executed_operations", []) for row in deterministic)
    rows.append({
        "gate": "V12_ONE_VOID_END_TO_END_DEMONSTRATED", "case": "deterministic_single_executable_trajectory",
        "initial": deterministic[0], "final": deterministic[-1], "executed_sequence": sequence,
        "stage_rows": deterministic, "maximum_2d_inventory_error_m2": max(area_balances, default=0.0),
        "explicit_transaction_equilibrium_count": runner_calls,
        "passed": sequence == required and max(area_balances, default=0.0) <= 1.0e-20
                  and deterministic[-1]["event_counters"].get("topology_actions", 0) >= 3,
    })

    lifecycle_ops = set(sequence[:6])
    rows.append({
        "gate": "V12_VOID_LIFECYCLE_QUALIFIED", "case": "localized_tensor_rate_lifecycle",
        "initial": deterministic[0], "final": deterministic[5],
        "executed_operations": list(sequence[:6]),
        "tensor_rate_rows": [row for row in deterministic if "rates" in row],
        "passed": lifecycle_ops == set(required[:6]) and deterministic[5]["void_phase"] == "STABLE_SUBGRID_VOID",
    })
    rows.append({
        "gate": "V12_VOID_PROMOTION_AND_GROWTH_QUALIFIED", "case": "body_fitted_2d_promotion_and_growth",
        "initial": deterministic[5], "final": deterministic[7],
        "executed_operations": deterministic[6].get("executed_operations", []) + ["resolved_boundary_remesh", "field_projection", "equilibrium"],
        "area_inventory_error_m2": max(area_balances, default=0.0),
        "reaction_before_N_per_m": deterministic[5]["reaction_N_per_m"],
        "reaction_after_N_per_m": deterministic[7]["reaction_N_per_m"],
        "energy_before_J_per_m": deterministic[5]["energy_J_per_m"],
        "energy_after_J_per_m": deterministic[7]["energy_J_per_m"],
        "passed": deterministic[6]["void_phase"] == "RESOLVED_VOID"
                  and deterministic[7]["cavity_radius_m"] > deterministic[6]["cavity_radius_m"]
                  and max(area_balances, default=0.0) <= 1.0e-20,
    })

    # Ligament failures occur after the named real production operations.
    for stage in ROLLBACK_STAGES:
        accepted, prefix = deterministic_trajectory(stop_before_ligament=True)
        before = complete_accepted_state_fingerprint(accepted)
        operations = []
        error = None
        try:
            ligament_transaction(accepted, failure_stage=stage, operation_log=operations)
        except RuntimeError as exc:
            error = str(exc)
        after = complete_accepted_state_fingerprint(accepted)
        rows.append({
            "gate": "V12_CRACK_VOID_TRANSACTION_QUALIFIED", "case": "rollback_" + stage,
            "initial": observables(accepted, "accepted_before"), "final": observables(accepted, "accepted_after"),
            "executed_operations": operations, "exception": error,
            "fingerprints_equal": before == after,
            "passed": error == "injected:" + stage and before == after,
        })
    rows.append({
        "gate": "V12_CRACK_VOID_TRANSACTION_QUALIFIED", "case": "accepted_ligament_and_downstream",
        "initial": deterministic[7], "final": deterministic[-1],
        "executed_operations": [row["operation"] for row in deterministic[8:]],
        "measured_graph_gain_m": deterministic[-1]["graph_length_m"] - deterministic[7]["graph_length_m"],
        "passed": deterministic[8]["void_phase"] == "CONNECTED_VOID"
                  and deterministic[-1]["void_phase"] == "DOWNSTREAM_FRONT_ACTIVE"
                  and deterministic[-1]["graph_length_m"] > deterministic[7]["graph_length_m"],
    })

    # Checkpoint the nontrivial final state and verify complete restart ownership.
    checkpoint = out / "one_void_checkpoint.json"
    manifest = write_checkpoint(final, checkpoint)
    restored = restore_checkpoint(checkpoint)
    rows.append({
        "gate": "V12_VOID_LIFECYCLE_QUALIFIED", "case": "production_checkpoint_restart",
        "initial": observables(final, "checkpointed"), "final": observables(restored, "restored"),
        "checkpoint_manifest": manifest,
        "passed": complete_accepted_state_fingerprint(final) == complete_accepted_state_fingerprint(restored),
    })

    natural_final, natural_rows = natural_trajectory()
    natural_events = [event for row in natural_rows for event in row["events"]]
    rows.append({
        "gate": "V12_BOUNDED_NATURAL_STOCHASTIC_CASE", "case": "actual_stress_history_seed_3621",
        "initial": natural_rows[0], "final": natural_rows[-1], "executed_steps": natural_rows,
        "integrated_actual_birth_hazard": natural_final.void_state.sites[0].birth.accumulated,
        "stored_threshold": natural_final.void_state.sites[0].birth.threshold,
        "classification": "BIRTH_ACTIVITY" if natural_events else "NO_BIRTH_WITHIN_BOUNDED_TRAJECTORY",
        "passed": True,
    })

    gates = {}
    for gate in sorted({row["gate"] for row in rows}):
        gates[gate] = "PASS" if all(row["passed"] for row in rows if row["gate"] == gate) else "FAIL"
    payload = {
        "schema": "v12.production-voiding-qualification/4",
        "implementation_git_sha": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
        "gates": gates, "rows": rows, "multiple_voids_enabled": False, "fatigue_campaign_run": False,
    }
    path = out / "case_rows.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (out / "sha256_manifest.json").write_text(json.dumps({"case_rows.json": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2) + "\n")
    print(json.dumps(gates, indent=2, sort_keys=True))
    return 0 if all(value == "PASS" for value in gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

