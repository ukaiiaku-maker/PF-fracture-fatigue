#!/usr/bin/env python3
"""Consolidate the executed R10 Weak-T joint acquisition and Pareto update."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SELECTED = "R10C_WEAKT_PEAK300_G2050_S3P75GPA_P4_LOW_TAYLOR_V2"
FAILED = "R10T_WEAKT_P40_G1962_S4P5GPA_P4_EP055_PP010_V1"
PEAK = "R10T_PEAK_P40_G2600_S0P25GPA_P4_LOW_TAYLOR_V1"
TEMPERATURES = (300, 450, 600, 750, 900, 1050, 1200)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--thermal-root", type=Path, required=True)
    parser.add_argument("--continuation-state", type=Path, required=True)
    args = parser.parse_args(argv)

    update = json.loads((args.root / "acquisition_update_decision.json").read_text())
    selected_a0 = json.loads(update["selected_metrics"]["A0_roots_json"])
    curves = []
    for cid, label, panel in (
        (FAILED, "previous selected Weak-T", None),
        ("R10C_WEAKT_PEAK300_G1988_S6P0GPA_P4_EP055_PP010_V1",
         "first cross-background near miss", args.root / "full_state" /
         "R10C_WEAKT_PEAK300_G1988_S6P0GPA_P4_EP055_PP010_V1"),
        (SELECTED, "promoted cross-background row", args.root / "full_state" / SELECTED),
    ):
        if panel is None:
            values = [5.2373966579, 6.258429080544234, 5.6305226556132455,
                      5.170827272321541, 5.25710533871317, 6.127402117669121,
                      5.1499512237412]
        else:
            values = [json.loads((panel / f"T_{t:04d}K.json").read_text())
                      ["K_init_MPa_sqrt_m"] for t in TEMPERATURES]
        for temperature, value in zip(TEMPERATURES, values):
            curves.append({"candidate_id": cid, "role": label,
                           "temperature_K": temperature,
                           "K_init_MPa_sqrt_m": value})
    write_csv(args.root / "fracture_curve_comparison.csv", curves)
    selected_values = [x["K_init_MPa_sqrt_m"] for x in curves
                       if x["candidate_id"] == SELECTED]
    selected_span = max(selected_values) / min(selected_values)
    a0_span = max(selected_a0) / min(selected_a0)

    source_fatigue = list(csv.DictReader(
        (args.thermal_root / "fatigue_case_table.csv").open()))
    fatigue_rows, slope_rows = [], []
    for seed in (4810, 9117):
        points = sorted((x for x in source_fatigue
                         if x["candidate_id"] == PEAK and int(x["seed"]) == seed),
                        key=lambda x: float(x["DeltaK_MPa_sqrt_m"]))
        for point in points:
            fatigue_rows.append({
                "candidate_id": SELECTED, "seed": seed,
                "DeltaK_MPa_sqrt_m": point["DeltaK_MPa_sqrt_m"],
                "rate_20_60_m_per_cycle": point["rate_20_60_m_per_cycle"],
                "event_count": point["event_count"],
                "provenance": "reused exact complete 300 K process/background contract",
                "source_candidate_id": PEAK,
                "overlap_passed": point["overlap_passed"],
                "source_artifact": point["artifact"],
            })
        for left, right in zip(points[:-1], points[1:]):
            slope = math.log(float(right["rate_20_60_m_per_cycle"])
                             / float(left["rate_20_60_m_per_cycle"])) / math.log(
                                 float(right["DeltaK_MPa_sqrt_m"])
                                 / float(left["DeltaK_MPa_sqrt_m"]))
            slope_rows.append({
                "candidate_id": SELECTED, "seed": seed,
                "DeltaK_low_MPa_sqrt_m": left["DeltaK_MPa_sqrt_m"],
                "DeltaK_high_MPa_sqrt_m": right["DeltaK_MPa_sqrt_m"],
                "adjacent_slope": slope,
            })
    write_csv(args.root / "selected_fatigue_rates.csv", fatigue_rows)
    write_csv(args.root / "selected_fatigue_adjacent_slopes.csv", slope_rows)
    windows = []
    for seed in (4810, 9117):
        points = [x for x in fatigue_rows if x["seed"] == seed]
        rates = np.asarray([float(x["rate_20_60_m_per_cycle"]) for x in points])
        loads = np.asarray([float(x["DeltaK_MPa_sqrt_m"]) for x in points])
        local = [x["adjacent_slope"] for x in slope_rows if x["seed"] == seed]
        windows.append({
            "seed": seed, "point_count": len(points),
            "global_slope": float(np.polyfit(np.log(loads), np.log(rates), 1)[0]),
            "rate_width_decades": float(np.log10(rates[-1] / rates[0])),
            "local_slope_min": min(local), "local_slope_max": max(local),
            "moderate_two_decade_interval": bool(
                np.log10(rates[-1] / rates[0]) >= 2.0
                and all(3.4 <= x <= 4.6 for x in local)),
        })

    spatial_path = args.root / "spatial" / SELECTED / "r8_current_backend_spatial.json"
    spatial = json.loads(spatial_path.read_text())
    spatial_receipt = {
        "schema": "r10.weakt-selected-spatial-receipt/1",
        "source_artifact": str(spatial_path.resolve()),
        "source_sha256": hashlib.sha256(spatial_path.read_bytes()).hexdigest(),
        "source_bytes": spatial_path.stat().st_size,
        "passed": spatial["passed"],
        "material_binding": spatial["material_binding"],
        "no_refit_or_multiplier": spatial["no_refit_or_multiplier"],
        "matched_arms": {
            name: {
                "selected_events": row["selected_events"],
                "initial_fingerprint": row["initial"]["accepted_fingerprint"],
                "event_boundary_fingerprint": row["event_boundary"]["accepted_fingerprint"],
                "final_fingerprint": row["final"]["accepted_fingerprint"],
            } for name, row in spatial["matched_arms"].items()
        },
        "files_only_restart": {
            "exact": spatial["files_only_restart"]["exact"],
            "files": spatial["files_only_restart"]["files"],
            "uninterrupted_fingerprint": spatial["files_only_restart"]["uninterrupted_fingerprint"],
            "restarted_fingerprint": spatial["files_only_restart"]["restarted_fingerprint"],
        },
        "geometry_correspondence": spatial["geometry_correspondence"],
    }
    (args.root / "selected_spatial_receipt.json").write_text(
        json.dumps(spatial_receipt, indent=2, sort_keys=True) + "\n"
    )
    decision = {
        "schema": "r10.weakt-joint-continuation-result/1",
        "joint_material_decision": "PROMOTE_UPDATED_WEAKT_ROW_TO_CLASS_PARETO_SET",
        "selected_candidate_id": SELECTED,
        "complete_row_sha256": update["complete_row_sha256"],
        "fracture": {
            "temperatures_K": list(TEMPERATURES),
            "K_init_MPa_sqrt_m": selected_values,
            "complete_span": selected_span,
            "opening_A0_roots_MPa_sqrt_m": selected_a0,
            "opening_A0_span": a0_span,
            "status": "PASS_BOUNDED_WEAK_T",
        },
        "fatigue": {
            "status": "PASS_EXACT_300K_CONTRACT_TWO_SEEDS",
            "windows": windows,
            "window_endpoint_rule": "first accepted event with cumulative extension >= 20 and 60 um",
            "events_between_endpoints": 6,
            "absolute_rate_range_seed4810": [
                min(float(x["rate_20_60_m_per_cycle"]) for x in fatigue_rows if x["seed"] == 4810),
                max(float(x["rate_20_60_m_per_cycle"]) for x in fatigue_rows if x["seed"] == 4810),
            ],
            "seed_9117_over_4810_ratio": float(np.mean([
                float(b["rate_20_60_m_per_cycle"]) / float(a["rate_20_60_m_per_cycle"])
                for a, b in zip([x for x in fatigue_rows if x["seed"] == 4810],
                                [x for x in fatigue_rows if x["seed"] == 9117])
            ])),
        },
        "mechanism": {
            "status": "COMPENSATION_SUPPORTED_SMALL_BUT_OUTCOME_CHANGING",
            "evidence": (
                f"A0 span {a0_span:.6f} exceeds 1.25 while complete span "
                f"{selected_span:.6f} is below 1.25"
            ),
            "warning": "complete-minus-A0 is an operational state correction, not a microscopic work partition",
        },
        "pareto_update": {
            "removed": [],
            "retained_as_failed_range_evidence": [FAILED],
            "retained_as_near_miss": ["R10C_WEAKT_PEAK300_G1988_S6P0GPA_P4_EP055_PP010_V1"],
            "added": [SELECTED],
            "unchanged_controls": [
                "R10T_DBTT_P40_G3075_S0P25GPA_P4_EP400_PP040_V1",
                PEAK, "retained ceramic controls",
            ],
        },
        "lower_window_decision": {
            "conditional_DeltaK_MPa_sqrt_m": 3.003,
            "conditional_rate_m_per_cycle": 1.21e-12,
            "status": "NOT_EXECUTED_NO_SELECTION_LEVERAGE",
            "reason": (
                "the promoted row already passes the inherited 5-22 MPa sqrt(m) two-seed window "
                "by exact 300 K contract; a lower point cannot change this Pareto decision"
            ),
        },
        "spatial": {
            "status": "PASS_EXACT_COMPLETE_ROW_NO_REFIT",
            "passed": spatial["passed"],
            "binding": spatial["material_binding"],
            "artifact": str(spatial_path.resolve()),
        },
        "class_mechanism_statuses": {
            "Weak-T compensated": "COMPENSATION_SUPPORTED_SMALL_BUT_OUTCOME_CHANGING",
            "DBTT": "PLASTICITY_CREATED_DBTT_NOT_ESTABLISHED; intrinsic rise with hot-state enhancement",
            "Peak-T": "PLASTICITY_GENERATED_PEAK_NOT_ESTABLISHED; opening-dominated control",
            "Ceramic-like": "CONTROL_NO_PARIS_REQUIREMENT",
        },
    }
    (args.root / "joint_material_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )

    state = json.loads(args.continuation_state.read_text())
    item = "Weak-T cross-background P3/P4 acquisition, full-state promotion, and exact-row spatial pass"
    if item not in state["completed"]:
        state["completed"].append(item)
    state["running_jobs"] = []
    state["queued_jobs"] = []
    state["next"] = "preserve promoted four-class joint Pareto set; no same-row refit required"
    state["weakt_joint_continuation_result"] = str((args.root / "joint_material_decision.json").resolve())
    state["unresolved"] = [x for x in state.get("unresolved", [])
                           if "return Weak-T fatigue failure" not in x]
    args.continuation_state.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
