#!/usr/bin/env python3
"""Consolidate the R10 thermal-accessibility continuation without promotion by label."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.design_r10_joint_level_iteration import frozen_onset
from scripts.design_r10_thermal_accessibility_iteration import exp_surface


TEMPERATURES = (300.0, 450.0, 600.0, 750.0, 900.0, 1050.0, 1200.0)
PEAK = "R10T_PEAK_P40_G2600_S0P25GPA_P4_LOW_TAYLOR_V1"
DBTT = "R10T_DBTT_P40_G3075_S0P25GPA_P4_EP400_PP040_V1"
WEAK = "R10T_WEAKT_P40_G1962_S4P5GPA_P4_EP055_PP010_V1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def developed_rate(events: list[dict], low: float = 20e-6, high: float = 60e-6):
    rows = [{"cycles": 0.0, "extension_m": 0.0}, *events]
    try:
        left = next(x for x in rows if x["extension_m"] >= low)
        right = next(x for x in rows if x["extension_m"] >= high)
    except StopIteration:
        return None
    return (right["extension_m"] - left["extension_m"]) / (right["cycles"] - left["cycles"])


def fatigue_rows(root: Path) -> tuple[list[dict], list[dict]]:
    cases = []
    campaigns = [
        ("Peak-T", root / "fatigue" / "peak_seed4810", "fresh"),
        ("Peak-T", root / "fatigue" / "peak_seed9117", "fresh"),
        ("Weak-T compensated", root / "fatigue" / "weakt_seed4810",
         "fresh_native_phase_resolved"),
        ("DBTT", root.parents[0] / "bridges" / "r10j_dbtt_p40_lshift_036mev_seed4810", "reused_exact_300K_contract"),
        ("DBTT", root.parents[0] / "bridges" / "r10j_dbtt_p40_lshift_036mev_seed9117", "reused_exact_300K_contract"),
    ]
    for class_name, campaign, evidence in campaigns:
        for path in sorted(campaign.glob("Kmax_*/result.json")):
            result = read_json(path); rate = developed_rate(result["events"])
            cases.append({
                "class": class_name, "candidate_id": result["candidate_id"],
                "seed": result["seed"], "evidence": evidence,
                "DeltaK_MPa_sqrt_m": result["DeltaK_MPa_sqrt_m"],
                "rate_20_60_m_per_cycle": rate, "event_count": len(result["events"]),
                "overlap_passed": result["overlap"]["passed"],
                "overlap_state_error": result["overlap"]["state_relative_max_error"],
                "overlap_hazard_error": result["overlap"]["hazard_relative_error"],
                "artifact": str(path.resolve()),
            })
    weak_progress = root / "fatigue" / "weakt_seed4810" / "Kmax_5p55555556" / "checkpoint" / "trajectory_progress.json"
    weak_results = list((root / "fatigue" / "weakt_seed4810").glob("Kmax_*/result.json"))
    if weak_progress.is_file() and not weak_results:
        result = read_json(weak_progress)
        cases.append({
            "class": "Weak-T compensated", "candidate_id": WEAK,
            "seed": result["identity"]["seed"], "evidence": "fresh_overlap_censor",
            "DeltaK_MPa_sqrt_m": 0.9 * result["identity"]["Kmax_MPa_sqrt_m"],
            "rate_20_60_m_per_cycle": None, "event_count": len(result["events"]),
            "overlap_passed": result["overlap"]["passed"],
            "overlap_state_error": result["overlap"]["state_relative_max_error"],
            "overlap_hazard_error": result["overlap"]["hazard_relative_error"],
            "artifact": str(weak_progress.resolve()),
        })
    windows = []
    for key in sorted({(x["class"], x["candidate_id"], x["seed"]) for x in cases}):
        points = sorted(
            (x for x in cases if (x["class"], x["candidate_id"], x["seed"]) == key
             and x["rate_20_60_m_per_cycle"] is not None),
            key=lambda x: x["DeltaK_MPa_sqrt_m"],
        )
        if len(points) < 2:
            continue
        x = np.log([p["DeltaK_MPa_sqrt_m"] for p in points])
        y = np.log([p["rate_20_60_m_per_cycle"] for p in points])
        slopes = [float((y[i+1]-y[i])/(x[i+1]-x[i])) for i in range(len(x)-1)]
        width = float((y[-1]-y[0])/math.log(10.0))
        windows.append({
            "class": key[0], "candidate_id": key[1], "seed": key[2],
            "point_count": len(points), "global_slope": float(np.polyfit(x, y, 1)[0]),
            "growth_rate_width_decades": width, "local_slope_min": min(slopes),
            "local_slope_max": max(slopes),
            "moderate_two_decade_interval": width >= 2.0 and all(3.4 <= m <= 4.6 for m in slopes),
        })
    return cases, windows


def two_window_rows(cases: list[dict]) -> list[dict]:
    rows = []
    keys = sorted({(x["class"], x["candidate_id"], x["seed"]) for x in cases})
    for key in keys:
        points = sorted(
            (x for x in cases if (x["class"], x["candidate_id"], x["seed"]) == key
             and x["rate_20_60_m_per_cycle"] is not None),
            key=lambda x: x["DeltaK_MPa_sqrt_m"],
        )
        if len(points) != 5:
            continue
        for label, selected in (("low_3", points[:3]), ("high_3", points[-3:])):
            x = np.log([p["DeltaK_MPa_sqrt_m"] for p in selected])
            y = np.log([p["rate_20_60_m_per_cycle"] for p in selected])
            local = [float((y[i+1]-y[i])/(x[i+1]-x[i]))
                     for i in range(len(x)-1)]
            rows.append({
                "class": key[0], "candidate_id": key[1], "seed": key[2],
                "window": label,
                "DeltaK_min_MPa_sqrt_m": selected[0]["DeltaK_MPa_sqrt_m"],
                "DeltaK_max_MPa_sqrt_m": selected[-1]["DeltaK_MPa_sqrt_m"],
                "global_slope": float(np.polyfit(x, y, 1)[0]),
                "growth_rate_width_decades": float((y[-1]-y[0])/math.log(10.0)),
                "local_slope_min": min(local), "local_slope_max": max(local),
            })
    return rows


def full_state_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "full_state").glob("*/T_*K.json")):
        result = read_json(path); events = result.get("events", [])
        state = events[0].get("hazard_bearing_state", {}) if events else result.get("terminal_hazard_bearing_state", {})
        rows.append({
            "candidate_id": result["candidate_id"], "temperature_K": result["temperature_K"],
            "status": result["status"], "K_init_MPa_sqrt_m": result["K_init_MPa_sqrt_m"],
            "event_count": result["event_count"],
            "first_event_K_shield_MPa_sqrt_m": events[0]["K_shield_MPa_sqrt_m"] if events else None,
            "first_or_terminal_radius_over_r0": state.get("radius_over_r0"),
            "first_or_terminal_source_activity": state.get("gross_source_activity"),
            "artifact": str(path.resolve()),
        })
    return rows


def selected_decomposition(root: Path, registry: list[dict], full: list[dict]) -> list[dict]:
    by_id = {x["candidate_id"]: x for x in registry}; rows = []
    reused_300 = {DBTT: 5.2181914331, WEAK: 5.2373966579}
    for class_name, candidate in (("Peak-T", PEAK), ("DBTT", DBTT), ("Weak-T compensated", WEAK)):
        record = by_id[candidate]; row = json.loads(record["complete_parameter_vector"])
        surface = exp_surface(row, "cleave", 1.0e12)
        q = float(row["physics__cleavage_hits"]); tau = float(row["physics__cleavage_correlation_time_s"])
        coupled = {x["temperature_K"]: x for x in full if x["candidate_id"] == candidate}
        for temperature in TEMPERATURES:
            opening = frozen_onset(surface, q, tau, temperature, dK=0.01)
            complete = coupled.get(temperature, {}).get("K_init_MPa_sqrt_m")
            provenance = "fresh_full_state"
            if complete is None and temperature == 300.0 and candidate in reused_300:
                complete = reused_300[candidate]; provenance = "reused_exact_300K_contract"
            shift = None if complete is None or opening is None else complete-opening
            rows.append({
                "class": class_name, "candidate_id": candidate,
                "temperature_K": temperature, "opening_A0_Kinit_MPa_sqrt_m": opening,
                "complete_state_Kinit_MPa_sqrt_m": complete,
                "complete_minus_opening_MPa_sqrt_m": shift,
                "relative_state_contribution": None if shift is None else shift/opening,
                "complete_state_provenance": provenance,
            })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--continuation-state", type=Path, required=True)
    args = parser.parse_args(argv); args.root.mkdir(parents=True, exist_ok=True)
    registry = list(csv.DictReader((args.root / "thermal_state_candidates.csv").open()))
    full = full_state_rows(args.root); fatigue, windows = fatigue_rows(args.root)
    decomposition = selected_decomposition(args.root, registry, full)
    two_windows = two_window_rows(fatigue)
    write_csv(args.root / "full_state_case_table.csv", full)
    write_csv(args.root / "causal_decomposition.csv", decomposition)
    write_csv(args.root / "fatigue_case_table.csv", fatigue)
    write_csv(args.root / "fatigue_windows.csv", windows)
    write_csv(args.root / "fatigue_two_windows.csv", two_windows)

    peak_curve = [x for x in decomposition if x["candidate_id"] == PEAK]
    dbtt_curve = [x for x in decomposition if x["candidate_id"] == DBTT]
    weak_curve = [x for x in decomposition if x["candidate_id"] == WEAK]
    peak_windows = [x for x in windows if x["class"] == "Peak-T"]
    weak_windows = [x for x in windows if x["class"] == "Weak-T compensated"]
    weak_two_windows = [x for x in two_windows if x["class"] == "Weak-T compensated"]
    spatial_path = args.root / "spatial" / "dbtt_900K_current_backend" / "r8_current_backend_spatial.json"
    spatial_result = read_json(spatial_path) if spatial_path.is_file() else None
    decision = {
        "schema": "r10.thermal-accessibility-result/1",
        "selected": {
            "Peak-T": {
                "candidate_id": PEAK, "fracture_status": "PASS_RESOLVED_INTERIOR_MAXIMUM_BOTH_LIMBS",
                "fatigue_status": ("PASS_TWO_SEEDS" if len(peak_windows) == 2 and all(x["moderate_two_decade_interval"] for x in peak_windows) else "PENDING_SECOND_SEED"),
                "Kinit_by_temperature": [x["complete_state_Kinit_MPa_sqrt_m"] for x in peak_curve],
            },
            "DBTT": {
                "candidate_id": DBTT, "fracture_status": "PASS_RISING_WITH_HOT_MECHANICAL_CONTRIBUTION",
                "fatigue_status": "PASS_REUSED_EXACT_300K_CONTRACT_TWO_SEEDS",
                "Kinit_by_temperature": [x["complete_state_Kinit_MPa_sqrt_m"] for x in dbtt_curve],
            },
            "Weak-T compensated": {
                "candidate_id": WEAK, "fracture_status": "PASS_BOUNDED_COMPENSATED_CURVE",
                "fatigue_status": (
                    "PASS_FIVE_POINT_DEVELOPED_INTERVAL"
                    if len(weak_windows) == 1 and weak_windows[0]["moderate_two_decade_interval"]
                    else ("FAIL_DEVELOPED_SLOPE_WIDTH_OBJECTIVE_RETURN_TO_SELECTION"
                          if len(weak_windows) == 1 and weak_windows[0]["point_count"] == 5
                          else "UNRESOLVED_NATIVE_FATIGUE")
                ),
                "Kinit_by_temperature": [x["complete_state_Kinit_MPa_sqrt_m"] for x in weak_curve],
                "numerical_status": "MEAN_CYCLE_REJECTED_NATIVE_PHASE_RESOLVED_COMPLETED",
                "five_point_window": weak_windows[0] if weak_windows else None,
                "two_window_diagnostic": weak_two_windows,
                "independent_stream_status": (
                    "NOT_TRIGGERED_OBJECTIVE_FAILED_IN_FIRST_STREAM"
                    if weak_windows and not weak_windows[0]["moderate_two_decade_interval"]
                    else "REQUIRED_IF_PROMOTED"
                ),
            },
            "Ceramic-like": {"candidate_id": "retained ceramic controls", "status": "CONTROL_NO_PARIS_REQUIREMENT"},
        },
        "spatial": {
            "status": ("PASS_EXACT_COMPLETE_ROW_CURRENT_BACKEND" if spatial_result and spatial_result["passed"] else "PENDING_EXACT_COMPLETE_ROW_CURRENT_BACKEND"),
            "selected_candidate_id": DBTT,
            "artifact": str(spatial_path.resolve()),
            "binding": None if spatial_result is None else spatial_result["material_binding"],
            "reused_controls": "runs/r10_joint_search/reused_sealed_2d_terminal_summary.csv",
        },
        "controls_preserved": ["R10J_DBTT_P40_LSHIFT_036MEV_V1", "R10_DBTT_P40_THERMAL_ANCHOR_V1"],
        "mechanism_classification": {
            "DBTT": {
                "achieved_topology": "rising full-state initiation curve",
                "opening_A0_role": "dominant intrinsic rise: 15.24 MPa sqrt(m) of the tabulated 17.943907 end-to-end rise",
                "state_role": "important hot-end enhancement; operational correction change 2.703907 MPa sqrt(m) and preserves the 1050-to-1200 rise",
                "mechanism_status": "PLASTICITY_CREATED_DBTT_NOT_ESTABLISHED; retain as intrinsically rising hot-state-enhanced successful-fatigue control",
                "root_difference_warning": "K_full-K_A0 is an operational state correction, not explicit K_shield or microscopic work partition",
            },
            "Peak-T": {
                "achieved_topology": "interior sampled maximum at 600 K",
                "opening_A0_role": "opening-dominated; A0 already peaks at 600 K",
                "state_role": "modest correction, maximum relative full/A0 about 2.27 percent",
                "mechanism_status": "PLASTICITY_GENERATED_PEAK_NOT_ESTABLISHED",
                "resolution": "600-to-750 full-root difference 0.0249997 MPa sqrt(m), below the executed dK=0.05 MPa sqrt(m) scan step; precise peak location unresolved",
            },
            "Weak-T compensated": {
                "achieved_topology": "bounded compensated curve, not perfectly flat",
                "opening_A0_role": "A0 span 1.34945 and 300-to-1200 change -0.670 MPa sqrt(m)",
                "state_role": "1050 K operational increase 31.49 percent; correction change +0.58255 MPa sqrt(m) opposes the intrinsic decline",
                "mechanism_status": "COMPENSATION_SUPPORTED; cancellation is interval-dependent",
            },
            "Ceramic-like": {
                "achieved_topology": "retained control",
                "mechanism_status": "CONTROL_NO_PARIS_REQUIREMENT",
            },
        },
        "counts": {"analytical_rows": len(list(csv.DictReader((args.root / "thermal_accessibility_scan.csv").open()))), "full_state_cases": len(full), "fatigue_cases": len(fatigue)},
    }
    (args.root / "thermal_accessibility_result.json").write_text(json.dumps(decision, indent=2, sort_keys=True)+"\n")

    state = read_json(args.continuation_state)
    for item in ("certified F0 opening accessibility bound", "P3/P4 thermal-accessibility analytical search", "class-specific thermal full-state iteration"):
        if item not in state["completed"]: state["completed"].append(item)
    if decision["selected"]["Peak-T"]["fatigue_status"] == "PASS_TWO_SEEDS":
        item = "Peak-T developed fatigue confirmation with two independent seeds"
        if item not in state["completed"]: state["completed"].append(item)
    state["running_jobs"] = []
    state["thermal_accessibility_result"] = str((args.root / "thermal_accessibility_result.json").resolve())
    weak_failed = bool(
        weak_windows and not weak_windows[0]["moderate_two_decade_interval"]
    )
    completed_item = "Weak-T five-point native phase-resolved developed fatigue screen"
    if weak_windows and completed_item not in state["completed"]:
        state["completed"].append(completed_item)
    state["queued_jobs"] = ([
        "return Weak-T fatigue failure to existing class-specific P3/P4 Pareto selection using retained bank"
    ] if weak_failed else [
        "run Weak-T independent stochastic stream for the promising developed interval"
    ])
    if not spatial_result or not spatial_result["passed"]:
        state["queued_jobs"].append("complete exact-row DBTT current-backend spatial execution")
    state["next"] = state["queued_jobs"][0]
    old_censor = "resolve Weak-T phase-exact fatigue overlap censor without changing the law"
    state["unresolved"] = list(dict.fromkeys([
        *(x for x in state.get("unresolved", []) if x != old_censor),
        *state["queued_jobs"],
    ]))
    temp = args.continuation_state.with_suffix(".json.tmp")
    temp.write_text(json.dumps(state, indent=2, sort_keys=True)+"\n"); temp.replace(args.continuation_state)
    print(json.dumps(decision, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
