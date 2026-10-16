#!/usr/bin/env python3
"""Consolidate the decisive 300/450 K Family-A joint fracture screen."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CASES = (
    ("R10A_WEAKT_P40_OPEN_P036MEV_V1", "open"),
    ("R10A_WEAKT_P40_EMIT_M060MEV_V1", "emit_m060"),
    ("R10A_WEAKT_P40_EMIT_P060MEV_V1", "emit_p060"),
)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for candidate, directory in CASES:
        cold_path = args.root / directory / "T_0300K.json"
        hot_path = args.root / directory / "T_0450K.json"
        cold = json.loads(cold_path.read_text())
        hot = json.loads(hot_path.read_text())
        if cold["candidate_id"] != candidate or hot["candidate_id"] != candidate:
            raise RuntimeError(f"candidate identity mismatch for {candidate}")
        if cold["status"] != "complete" or cold["K_init_MPa_sqrt_m"] is None:
            raise RuntimeError(f"cold fracture case did not complete for {candidate}")
        endpoint = float(hot["final_K_MPa_sqrt_m"])
        physical_endpoint = (
            hot["status"] == "KMAX_OR_STEP_CENSOR"
            and int(hot["event_count"]) == 0
            and abs(endpoint - float(hot["Kmax_MPa_sqrt_m"])) <= 1e-12
        )
        if not physical_endpoint:
            raise RuntimeError(f"hot case was not the expected physical endpoint censor: {candidate}")
        terminal = hot["terminal_hazard_bearing_state"]
        rows.append({
            "candidate_id": candidate,
            "family": "A_ORIGINAL_SINGLE_EXP_FLOOR",
            "Kinit_300_MPa_sqrt_m": float(cold["K_init_MPa_sqrt_m"]),
            "Kendpoint_450_MPa_sqrt_m": endpoint,
            "fracture_span_lower_bound": endpoint / float(cold["K_init_MPa_sqrt_m"]),
            "event_count_300": int(cold["event_count"]),
            "event_count_450": int(hot["event_count"]),
            "hot_hazard_action_current": float(terminal["hazard_action_current"]),
            "hot_pending_mark_action": float(terminal["remaining_action_to_mark"]),
            "fracture_gate": "FAIL_PHYSICAL_450K_ENDPOINT_NO_EVENT",
            "fatigue_action": "NOT_AUTHORIZED_AFTER_DECISIVE_JOINT_FRACTURE_REJECTION",
            "spatial_action": "NOT_AUTHORIZED_BEFORE_JOINT_1D_PASS",
            "cold_artifact": str(cold_path.resolve()),
            "hot_artifact": str(hot_path.resolve()),
        })
    write_csv(args.out_root / "family_a_joint_fracture_screen.csv", rows)

    decision = {
        "schema": "r10.family-a-joint-execution/1",
        "decision": "ALL_THREE_ACQUIRED_FAMILY_A_ROWS_REJECTED_BY_450K_FRACTURE_GATE",
        "executed_rows": rows,
        "joint_consequence": {
            "fatigue": "not run: decisive same-row fracture rejection",
            "same_row_2D": "not run: gated behind joint 1D fracture/fatigue pass",
            "stopped_temperatures": "600--1200 K omitted after the decisive 300/450 K result",
        },
        "updated_feasible_set": {
            "Family-B Weak-T benchmark": (
                "P25_TJBSV2_S_026704 retains completed three-point developed fatigue plus historical "
                "fracture/2D receipts; benchmark only, not strict-Family-A success"
            ),
            "strict-Family-A Weak-T": "no jointly feasible row established",
        },
        "closed_findings": [
            "candidate eligibility and endpoint-censor semantics remain closed",
            "the 3.339 startup secant remains excluded",
            "original thermal slopes and transport entropies were preserved in all executed candidates",
            "no endpoint fitting, new physics, or alternative mechanics was introduced",
        ],
        "next_acquisition": (
            "return to the permitted complete-law analytical bank and require 450 K opening-action "
            "accessibility before another expensive current-state run; vary lawful coupled thermal and "
            "nucleation/motion coordinates, not independent temperature endpoints"
        ),
    }
    (args.out_root / "joint_execution_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
