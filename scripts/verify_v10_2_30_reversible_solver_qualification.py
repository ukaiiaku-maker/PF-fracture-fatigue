#!/usr/bin/env python3
"""Verify decisive real-run gates for the v10.2.30 reversible solver."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _load(path: Path):
    return json.loads(path.read_text())


def _ledger_sum(ledgers: dict, token: str) -> float:
    return sum(float(value) for key, value in ledgers.items() if token in key)


def _relative(a: float, b: float, floor: float = 1.0e-18) -> float:
    return abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), floor)


def verify(positive: Path, negative: Path, event: Path, multi: Path,
           explicit: Path, accelerated: Path) -> dict:
    pos = _load(positive / "high_cycle_live_checkpoint.json")
    neg = _load(negative / "high_cycle_live_checkpoint.json")
    pos_l = pos["ledgers"]
    neg_l = neg["ledgers"]
    pos_return = _ledger_sum(pos_l, "cumulative_physical_returned_mobile[")
    neg_return = _ledger_sum(neg_l, "cumulative_physical_returned_mobile[")
    neg_cancel = _ledger_sum(neg_l, "cumulative_cancelled_source_slip[")
    neg_source = float(neg_l["mpz.cumulative_gross_source_activity"])

    one_events = _load(event / "hazard_energy_gated_events_v10_2_30.json")
    multi_events = _load(multi / "hazard_energy_gated_events_v10_2_30.json")
    checks = {
        "positive_R_physical_return_zero": abs(pos_return) <= 1.0e-18,
        "negative_R_physical_return_positive": neg_return > 0.0,
        "negative_R_return_equals_source_cancellation": math.isclose(
            neg_return, neg_cancel, rel_tol=1.0e-12, abs_tol=1.0e-18
        ),
        "negative_R_return_bounded_by_source": neg_return <= neg_source,
        "one_event_present": len(one_events) == 1,
        "multi_event_count_at_least_three": len(multi_events) >= 3,
    }
    transaction_rows = []
    for row in one_events + multi_events:
        audit = row["event_transaction_audit"]
        distances = [
            float(audit["energy_admissible_advance_m"]),
            float(audit["geometry_committed_advance_m"]),
            float(audit["mpz_translated_advance_m"]),
        ]
        complete = all(
            bool(audit[name]["complete_active_state"])
            for name in (
                "pre_event_state",
                "pre_geometry_commit_state",
                "post_event_state",
            )
        )
        action_closed = math.isclose(
            float(audit["hazard_action_completed"]),
            float(audit["threshold_action"]),
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        )
        distance_closed = max(distances) - min(distances) <= 1.0e-15
        transaction_rows.append(
            {"complete_snapshots": complete, "action_closed": action_closed,
             "distance_closed": distance_closed, "distances_m": distances}
        )
    checks["all_transaction_snapshots_complete"] = all(
        row["complete_snapshots"] for row in transaction_rows
    )
    checks["all_first_passage_actions_close"] = all(
        row["action_closed"] for row in transaction_rows
    )
    checks["all_geometry_mpz_distances_atomic"] = all(
        row["distance_closed"] for row in transaction_rows
    )

    history = [
        json.loads(line)
        for line in (multi / "high_cycle_live_history.jsonl").read_text().splitlines()
        if line.strip()
    ]
    rebuilds = sum(
        row.get("high_cycle_cache", {}).get("invalidated_reason")
        == "first_passage_event"
        for row in history
        if row.get("reason") == "outer_driver_geometry_committed"
    )
    checks["cache_invalidated_after_every_multi_event"] = rebuilds >= len(multi_events)
    explicit_cp = _load(explicit / "high_cycle_live_checkpoint.json")
    accelerated_cp = _load(accelerated / "high_cycle_live_checkpoint.json")
    explicit_vector = np.load(explicit / "high_cycle_live_state.npz")["active_vector"]
    accelerated_vector = np.load(accelerated / "high_cycle_live_state.npz")["active_vector"]
    vector_error = float(np.linalg.norm(explicit_vector - accelerated_vector)) / max(
        float(np.linalg.norm(explicit_vector)),
        float(np.linalg.norm(accelerated_vector)), 1.0)
    diagnostic_errors = {
        key: _relative(explicit_cp["diagnostics"][key], accelerated_cp["diagnostics"][key])
        for key in (
            "active_K_shield_Pa_sqrt_m", "emission_hazard_s", "mobile_count",
            "retained_count", "sigma_back_Pa", "tip_radius_m")
    }
    hazard_error = _relative(
        explicit_cp["stochastic"]["hazard_action_current"],
        accelerated_cp["stochastic"]["hazard_action_current"])
    source_ledger_error = _relative(
        explicit_cp["ledgers"]["mpz.cumulative_gross_source_activity"],
        accelerated_cp["ledgers"]["mpz.cumulative_gross_source_activity"])
    escape_abs_error = abs(
        float(explicit_cp["ledgers"]["mpz.escaped_total"])
        - float(accelerated_cp["ledgers"]["mpz.escaped_total"])
    )
    explicit_modes = _load(explicit / "high_cycle_summary.json")["mode_counts"]
    accelerated_modes = _load(accelerated / "high_cycle_summary.json")["mode_counts"]
    checks.update({
        "explicit_reference_uses_only_exact_cycles": set(explicit_modes) == {"exact_cycle_burst"},
        "accelerated_trajectory_uses_projective_state": accelerated_modes.get("slow_projective", 0) > 0,
        "accelerated_active_vector_matches_exact": vector_error <= 1.0e-8,
        "accelerated_diagnostics_match_exact": max(diagnostic_errors.values()) <= 5.0e-4,
        "accelerated_hazard_matches_exact": hazard_error <= 5.0e-4,
        "accelerated_source_ledger_matches_exact": source_ledger_error <= 5.0e-4,
        "accelerated_escape_ledger_matches_exact_absolute": escape_abs_error <= 5.0e-11,
    })
    return {
        "schema": "v10.2.30_reversible_solver_real_run_qualification_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "positive_R_physical_return": pos_return,
            "negative_R_physical_return": neg_return,
            "negative_R_source_cancellation": neg_cancel,
            "negative_R_gross_source_activity": neg_source,
            "multi_event_count": len(multi_events),
            "multi_event_cache_rebuilds": rebuilds,
            "accelerated_active_vector_relative_error": vector_error,
            "accelerated_diagnostic_relative_errors": diagnostic_errors,
            "accelerated_hazard_relative_error": hazard_error,
            "accelerated_source_ledger_relative_error": source_ledger_error,
            "accelerated_escape_ledger_absolute_error": escape_abs_error,
        },
        "transactions": transaction_rows,
        "inputs": {k: str(v.resolve()) for k, v in {
            "positive": positive, "negative": negative,
            "one_event": event, "multi_event": multi,
            "explicit": explicit, "accelerated": accelerated}.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("positive", "negative", "event", "multi", "explicit", "accelerated"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.positive, args.negative, args.event, args.multi,
                    args.explicit, args.accelerated)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
