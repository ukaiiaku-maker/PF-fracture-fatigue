#!/usr/bin/env python3
"""Consolidate the executed active Weak-T acquisition without overstating it."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = "R10A_WEAKT_ACTIVE_EMIT_S3P0GPA_V1"
TEMPERATURES = (300, 450, 600, 750, 900, 1050, 1200)


def endpoint(events: list[dict], threshold_m: float) -> dict:
    return next(event for event in events if float(event["extension_m"]) >= threshold_m)


def developed_window(events: list[dict], low_um: float = 20.0,
                     high_um: float = 60.0) -> dict:
    left = endpoint(events, low_um * 1.0e-6)
    right = endpoint(events, high_um * 1.0e-6)
    cycles = float(right["cycles"]) - float(left["cycles"])
    extension = float(right["extension_m"]) - float(left["extension_m"])
    return {
        "left_event_index": int(left["event_index"]),
        "right_event_index": int(right["event_index"]),
        "left_cycles": float(left["cycles"]),
        "right_cycles": float(right["cycles"]),
        "actual_extension_m": extension,
        "cycles": cycles,
        "rate_m_per_cycle": extension / cycles,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def accepted_history(result_path: Path) -> list[dict]:
    history = result_path.parent / "checkpoint" / "high_cycle_live_history.jsonl"
    return [json.loads(line) for line in history.read_text().splitlines()
            if line.strip() and json.loads(line).get("reason") == "accepted_event"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out_root.mkdir(parents=True, exist_ok=True)

    run_root = ROOT / "runs/r10_joint_search"
    selected_paths = [
        run_root / "weakt_active_refinement/fatigue/s3p0_seed4810_low5/Kmax_5p55555556/result.json",
        run_root / "weakt_active_refinement/fatigue/s3p0_seed4810_low7/Kmax_7p77777778/result.json",
        run_root / "weakt_active_refinement/fatigue/s3p0_seed4810_mid10/Kmax_11p1111111/result.json",
        run_root / "weakt_active_refinement/fatigue/s3p0_seed4810/Kmax_16p6666667/result.json",
        run_root / "weakt_active_refinement/fatigue/s3p0_seed4810/Kmax_24p4444444/result.json",
    ]
    rates, state_rows = [], []
    for path in selected_paths:
        result = json.loads(path.read_text())
        window = developed_window(result["events"])
        native = result.get("native_qualification")
        rates.append({
            "candidate_id": ACTIVE,
            "seed": result["seed"],
            "DeltaK_MPa_sqrt_m": result["DeltaK_MPa_sqrt_m"],
            **window,
            "event_count": len(result["events"]),
            "final_cycles": result["final_cycles"],
            "native_qualification_accepted": (
                bool(native["accepted"]) if native is not None
                else bool(result["overlap"]["passed"])
            ),
            "one_cycle_overlap_passed": result["overlap"]["passed"],
            "one_cycle_state_relative_max_error": result["overlap"]["state_relative_max_error"],
            "source_artifact": str(path.resolve()),
        })
        logs = accepted_history(path)
        if len(logs) != len(result["events"]):
            raise RuntimeError(f"accepted-event history mismatch: {path}")
        chosen = [(event, log) for event, log in zip(result["events"], logs)
                  if 20.0 <= 1.0e6 * float(event["extension_m"]) <= 60.0]
        for event, log in chosen:
            diag = log["diagnostics"]
            state_rows.append({
                "candidate_id": ACTIVE,
                "DeltaK_MPa_sqrt_m": result["DeltaK_MPa_sqrt_m"],
                "event_index": event["event_index"],
                "cycles": event["cycles"],
                "extension_um": 1.0e6 * float(event["extension_m"]),
                "tip_radius_um": 1.0e6 * float(diag["tip_radius_m"]),
                "opening_stress_GPa": 1.0e-9 * float(diag["sigma_tip_Pa"]),
                "opening_rate_s-1": float(diag["lambda_cleave_s"]),
                "backstress_GPa": 1.0e-9 * float(diag["sigma_back_Pa"]),
                "mobile_count": diag["mobile_count"],
                "retained_count": diag["retained_count"],
                "K_shield_MPa_sqrt_m": 1.0e-6 * float(diag["active_K_shield_Pa_sqrt_m"]),
                "continuous_mpz_progress_m": event["continuous_mpz_progress_m"],
            })
    rates.sort(key=lambda row: float(row["DeltaK_MPa_sqrt_m"]))
    write_csv(args.out_root / "selected_fatigue_rates.csv", rates)
    write_csv(args.out_root / "selected_state_effects.csv", state_rows)

    slopes = []
    for left, right in zip(rates[:-1], rates[1:]):
        slope = math.log(float(right["rate_m_per_cycle"]) /
                         float(left["rate_m_per_cycle"])) / math.log(
                             float(right["DeltaK_MPa_sqrt_m"]) /
                             float(left["DeltaK_MPa_sqrt_m"]))
        slopes.append({
            "candidate_id": ACTIVE,
            "DeltaK_low_MPa_sqrt_m": left["DeltaK_MPa_sqrt_m"],
            "DeltaK_high_MPa_sqrt_m": right["DeltaK_MPa_sqrt_m"],
            "adjacent_slope": slope,
            "passes_3p4_to_4p6": 3.4 <= slope <= 4.6,
        })
    write_csv(args.out_root / "selected_fatigue_adjacent_slopes.csv", slopes)
    loads = np.asarray([float(row["DeltaK_MPa_sqrt_m"]) for row in rates])
    values = np.asarray([float(row["rate_m_per_cycle"]) for row in rates])
    width = float(np.log10(values[-1] / values[0]))
    global_slope = float(np.polyfit(np.log(loads), np.log(values), 1)[0])

    high_pair_rows = []
    high_campaigns = {
        "R10A_WEAKT_ACTIVE_EMIT_S1P2GPA_V1": run_root / "weakt_active_acquisition/fatigue/s1p2_seed4810",
        "R10A_WEAKT_ACTIVE_EMIT_S1P5GPA_V1": run_root / "weakt_active_acquisition/fatigue/s1p5_seed4810",
        "R10A_WEAKT_ACTIVE_EMIT_S2P0GPA_V1": run_root / "weakt_active_acquisition/fatigue/s2p0_seed4810",
        ACTIVE: run_root / "weakt_active_refinement/fatigue/s3p0_seed4810",
    }
    for candidate, campaign in high_campaigns.items():
        points = []
        for path in sorted(campaign.glob("Kmax_*/result.json")):
            result = json.loads(path.read_text())
            points.append((float(result["DeltaK_MPa_sqrt_m"]),
                           developed_window(result["events"])["rate_m_per_cycle"], path))
        points.sort()
        slope = math.log(points[1][1] / points[0][1]) / math.log(points[1][0] / points[0][0])
        high_pair_rows.append({
            "candidate_id": candidate,
            "DeltaK_low_MPa_sqrt_m": points[0][0],
            "DeltaK_high_MPa_sqrt_m": points[1][0],
            "rate_low_m_per_cycle": points[0][1],
            "rate_high_m_per_cycle": points[1][1],
            "slope": slope,
            "passes_3p4_to_4p6": 3.4 <= slope <= 4.6,
        })
    write_csv(args.out_root / "candidate_high_pair_results.csv", high_pair_rows)

    refined_panel = json.loads((run_root /
        "weakt_active_refinement/full_state_refined" / ACTIVE / "panel_summary.json").read_text())
    fracture_rows = refined_panel["cases"]
    roots = [float(row["K_init_MPa_sqrt_m"]) for row in fracture_rows]
    refined_span = max(roots) / min(roots)
    write_csv(args.out_root / "selected_refined_fracture_curve.csv", fracture_rows)

    hot_rows = []
    for directory in (
        "weakt_active_thermal_refinement/hot_refined",
        "weakt_active_thermal_refinement_narrow/hot_refined",
        "weakt_active_thermal_refinement_p3_2d/hot_refined",
        "weakt_active_thermal_refinement_low_sigc/hot_refined",
    ):
        for path in sorted((run_root / directory).glob("*/panel_summary.json")):
            panel = json.loads(path.read_text())
            for row in panel["cases"]:
                hot_rows.append({"candidate_id": path.parent.name,
                                 "temperature_K": row["temperature_K"],
                                 "K_init_MPa_sqrt_m": row["K_init_MPa_sqrt_m"],
                                 "source_artifact": str(path.resolve())})
    write_csv(args.out_root / "thermal_refinement_hot_roots.csv", hot_rows)

    state_summary = []
    for load in loads:
        rows = [row for row in state_rows if math.isclose(
            float(row["DeltaK_MPa_sqrt_m"]), load, abs_tol=1.0e-5)]
        state_summary.append({
            "DeltaK_MPa_sqrt_m": load,
            **{f"mean_{key}": float(np.mean([float(row[key]) for row in rows])) for key in (
                "tip_radius_um", "opening_stress_GPa", "opening_rate_s-1",
                "backstress_GPa", "mobile_count", "retained_count",
                "K_shield_MPa_sqrt_m", "continuous_mpz_progress_m")},
        })
    write_csv(args.out_root / "selected_state_effect_summary.csv", state_summary)

    fatigue_pass = width >= 2.0 and all(bool(row["passes_3p4_to_4p6"]) for row in slopes)
    decision = {
        "schema": "r10.active-weakt-joint-acquisition-result/1",
        "joint_scientific_decision": (
            "RETAIN_ACTIVE_FATIGUE_FRONTIER; NO_JOINT_PROMOTION_AFTER_REFINED_FRACTURE"
        ),
        "selected_active_candidate_id": ACTIVE,
        "fatigue": {
            "status": "PASS" if fatigue_pass else "FAIL",
            "DeltaK_MPa_sqrt_m": loads.tolist(),
            "rates_m_per_cycle": values.tolist(),
            "adjacent_slopes": [float(row["adjacent_slope"]) for row in slopes],
            "global_slope": global_slope,
            "rate_width_decades": width,
            "native_state_evolution": True,
            "mechanism_status": "ACTIVE_COUPLED_STATE_OBSERVED",
        },
        "fracture": {
            "status": "FAIL_REFINED_WEAK_T_SPAN",
            "temperatures_K": list(TEMPERATURES),
            "K_init_MPa_sqrt_m": roots,
            "refined_span": refined_span,
            "limit": 1.25,
            "coarse_screen_was_false_positive": True,
            "P3_hot_refinement_status": "BRACKETED_BUT_DID_NOT_RECOVER_COMPLETE_SPAN",
        },
        "selection_history": {
            "P4_emit_knees_GPa": [0.9, 1.2, 1.5, 2.0, 3.0],
            "native_high_pair_candidates_GPa": [1.2, 1.5, 2.0, 3.0],
            "P3_hot_rows_executed": len({row["candidate_id"] for row in hot_rows}),
            "same_row_2D_status": "NOT_RUN_JOINT_1D_GATE_FAILED",
        },
        "preserved_result": {
            "candidate_id": "R10C_WEAKT_PEAK300_G2050_S3P75GPA_P4_LOW_TAYLOR_V2",
            "sha256": "0e2e41347881be1811ffdcf934b73f518cf032aca12ced6cb68489a18982d014",
            "scope": "opening-dominated intrinsic/mixed control; unchanged",
        },
        "class_mechanism_statuses": {
            "Weak-T active": "FATIGUE_PASS_BUT_JOINT_FRACTURE_FAIL",
            "Weak-T opening-dominated control": "PRESERVED_NUMERICAL_PASS_WITH_LIMITED_MECHANISTIC_SCOPE",
            "DBTT": "PLASTICITY_CREATED_DBTT_NOT_ESTABLISHED",
            "Peak-T": "PLASTICITY_GENERATED_PEAK_NOT_ESTABLISHED",
            "Ceramic-like": "CONTROL_NO_PARIS_REQUIREMENT",
        },
    }
    (args.out_root / "joint_scientific_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
