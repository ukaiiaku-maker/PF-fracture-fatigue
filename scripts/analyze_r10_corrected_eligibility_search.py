#!/usr/bin/env python3
"""Correct R10 eligibility semantics and consolidate new multilevel progress."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from arrhenius_fracture.material_manifest import ExpFloorBarrier


WEAK_V2 = "P25_TJBSV2_S_026704"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def progress(path: Path, label: str, delta_k: float) -> dict:
    data = json.loads(path.read_text())
    events = data["events"]
    last = events[-1]
    return {
        "candidate_id": WEAK_V2,
        "case": label,
        "DeltaK_MPa_sqrt_m": delta_k,
        "status": "RESTARTABLE_ACCEPTED_PROGRESS",
        "accepted_events": len(events),
        "accepted_cycles": float(last["cycles"]),
        "accepted_extension_m": float(last["extension_m"]),
        "startup_average_rate_m_per_cycle": float(last["extension_m"]) / float(last["cycles"]),
        "remaining_events_to_target": 12 - len(events),
        "remaining_extension_m_to_70um": 70.0e-6 - float(last["extension_m"]),
        "checkpoint": str(path.resolve()),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--original-registry", type=Path, required=True)
    p.add_argument("--controls-registry", type=Path, required=True)
    p.add_argument("--v2-rows", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    a = p.parse_args(argv)
    a.out_root.mkdir(parents=True, exist_ok=True)

    original = load_csv(a.original_registry)
    controls = load_csv(a.controls_registry)
    v2 = load_csv(a.v2_rows)
    projection = load_csv(a.run_root / "projection_frontier/strict_projection_frontier.csv")
    level = load_csv(a.run_root / "joint_level_iteration/joint_level_candidates.csv")
    thermal = load_csv(a.run_root / "thermal_accessibility_iteration/thermal_state_candidates.csv")

    eligibility = []
    for row in original:
        eligibility.append({
            "candidate_id": row["candidate_id"], "group": "original_four",
            "family": "A_ORIGINAL_SINGLE_EXP_FLOOR", "eligible_role": "complete_material_parent",
            "thermal_final_status": "requires_joint_performance",
            "reason": "original complete record; alias alone does not guarantee current performance",
        })
    for row in controls:
        cid = row["candidate_id"]
        eligibility.append({
            "candidate_id": cid, "group": "P25_P40_controls",
            "family": "A_ORIGINAL_SINGLE_EXP_FLOOR", "eligible_role": "isothermal_control_and_design_seed",
            "thermal_final_status": (
                "joint_thermal_performance_failed" if cid == "P40_TRANSFER_CALIBRATED_GEN2"
                else "not_automatically_qualified"
            ),
            "reason": (
                "zero explicit barrier-temperature slopes remain lawful constant-Q Arrhenius kinetics; "
                "final eligibility is decided by constraints and joint performance"
            ),
        })
    for row in v2:
        eligibility.append({
            "candidate_id": row["candidate_id"], "group": "refined_V2",
            "family": "B_INHERITED_V2_GUARD_ENTROPY_EXTENSION",
            "eligible_role": "exact_extension_candidate",
            "thermal_final_status": "eligible_within_family_B_not_strict_family_A_success",
            "reason": "documented guard/entropy extension identity retained exactly",
        })
    for row in projection:
        eligibility.append({
            "candidate_id": row["candidate_id"], "group": "inverse_projection",
            "family": row["family"], "eligible_role": "300K_shape_level_hypothesis_and_fixed_T_evidence",
            "thermal_final_status": "thermal_anchor_unpromoted",
            "reason": (
                "inverse-generated shape/level coefficients are permissible hypotheses; the independent "
                "1200 K anchor construction is not authorized as the final thermal continuation"
            ),
        })
    for row in level:
        eligibility.append({
            "candidate_id": row["candidate_id"], "group": "level_shift",
            "family": row["family"], "eligible_role": "shape_level_sensitivity_and_fixed_T_evidence",
            "thermal_final_status": "inherits_unpromoted_thermal_anchor",
            "reason": "permitted level coordinate does not cure the inherited thermal construction",
        })
    for row in thermal:
        eligibility.append({
            "candidate_id": row["candidate_id"], "group": "withdrawn_endpoint_rows",
            "family": row["family"], "eligible_role": "fixed_T_diagnostic_only",
            "thermal_final_status": "non_promotable_endpoint_construction",
            "reason": "independently selected cold/hot endpoint coordinates remain withdrawn",
        })
    eligibility.append({
        "candidate_id": "R10K_WEAKT_COUPLED_EMIT_SCALE_S3P0_V1", "group": "tested_endpoint_descendant",
        "family": "A_ORIGINAL_SINGLE_EXP_FLOOR", "eligible_role": "preserved_fixed_T_and_sensitivity_evidence",
        "thermal_final_status": "non_promotable_inherited_endpoint_construction",
        "reason": "coherent newest scale does not cure its parent ancestry",
    })
    write_csv(a.out_root / "corrected_eligibility.csv", eligibility)

    # Regression receipt: constant explicit barrier coordinates do not remove
    # the elementary Arrhenius temperature dependence.
    constant = ExpFloorBarrier(1.0, 0.0, 2.0e9, 0.0, 1.0, 1.0, 0.1)
    rate_300 = float(constant.rate(0.0, 300.0))
    rate_450 = float(constant.rate(0.0, 450.0))
    arrhenius = {
        "formula": "lambda=nu0*exp[-Q/(kB*T)]",
        "explicit_gT": 0.0, "explicit_sT": 0.0,
        "rate_300_s-1": rate_300, "rate_450_s-1": rate_450,
        "rate_increases_with_temperature": rate_450 > rate_300,
        "interpretation": "zero explicit barrier derivative is not a temperature-independent rate",
    }

    comparison = next(x for x in load_csv(a.run_root / "retained_material_comparison.csv")
                      if x["candidate_id"] == WEAK_V2)
    low_result = json.loads((a.run_root /
        "bridges/weakt_v2_p25_seed4810/Kmax_4p60545451/result.json").read_text())
    left = next(x for x in low_result["events"] if float(x["extension_m"]) >= 20e-6)
    right = next(x for x in low_result["events"] if float(x["extension_m"]) >= 60e-6)
    low_rate = (float(right["extension_m"]) - float(left["extension_m"])) / (
        float(right["cycles"]) - float(left["cycles"])
    )
    mid = progress(a.run_root /
        "bridges/weakt_v2_p25_seed4810/Kmax_10p730645/checkpoint/trajectory_progress.json",
        "A0_band_low", 9.657580494899715)
    high = progress(a.run_root /
        "bridges/weakt_v2_p25_high_seed4810/Kmax_20p2367527/checkpoint/trajectory_progress.json",
        "A0_band_high", 18.213077452291675)
    partial_slope = math.log(high["startup_average_rate_m_per_cycle"] /
                             mid["startup_average_rate_m_per_cycle"]) / math.log(
        high["DeltaK_MPa_sqrt_m"] / mid["DeltaK_MPa_sqrt_m"]
    )
    progress_rows = [{
        "candidate_id": WEAK_V2, "case": "developed_low",
        "DeltaK_MPa_sqrt_m": float(low_result["DeltaK_MPa_sqrt_m"]),
        "status": "DEVELOPED_TARGET_COMPLETED", "accepted_events": len(low_result["events"]),
        "accepted_cycles": float(low_result["final_cycles"]),
        "accepted_extension_m": float(low_result["final_extension_m"]),
        "startup_average_rate_m_per_cycle": "", "developed_20_60_rate_m_per_cycle": low_rate,
        "remaining_events_to_target": 0, "remaining_extension_m_to_70um": 0.0,
        "checkpoint": str(Path(low_result["checkpoint_directory"]).resolve()),
    }]
    for row in (mid, high):
        row["developed_20_60_rate_m_per_cycle"] = ""
        progress_rows.append(row)
    write_csv(a.out_root / "weakt_v2_p25_progress.csv", progress_rows)

    original_450 = json.loads((a.run_root /
        "fracture/original_weakt_ancestry_qualified_refined/T_0450K.json").read_text())
    terminal = original_450["terminal_hazard_bearing_state"]
    stop_receipt = {
        "candidate_id": original_450["candidate_id"],
        "reported_status": original_450["status"],
        "resolved_stop_reason": "PHYSICAL_K_ENDPOINT_REACHED_NO_EVENT",
        "final_accepted_K_MPa_sqrt_m": original_450["final_K_MPa_sqrt_m"],
        "accepted_steps": original_450["history"][-1]["step"],
        "accepted_time_s": original_450["final_K_MPa_sqrt_m"] / original_450["Kdot_MPa_sqrt_m_s"],
        "hazard_action_current": terminal["hazard_action_current"],
        "hazard_threshold_action": terminal["hazard_threshold_action"],
        "pending_mark_action": terminal["remaining_action_to_mark"],
        "event_count": original_450["event_count"],
        "work_budget_reached": False,
    }
    fracture_span = float(comparison["historical_1d_K_max"]) / float(comparison["historical_1d_K_min"])
    selected = [{
        "candidate_id": WEAK_V2,
        "family": "B_INHERITED_V2_GUARD_ENTROPY_EXTENSION",
        "selection": "CONTINUE_RESTARTABLE_CURRENT_FATIGUE",
        "analytical_reason": (
            f"A0 moderate band {comparison['A0_moderate_slope_width_decades']} decades; "
            f"historical full-state fracture span {fracture_span:.6f}; exact sealed 2D at 300/1200 K"
        ),
        "joint_status": "fracture_and_2D_passed_family_B; fatigue_low_complete_mid_high_restartable",
    }]
    write_csv(a.out_root / "selected_candidates.csv", selected)

    causal_path = a.run_root / "thermal_accessibility_iteration/thermal_accessibility_result.json"
    causal = json.loads(causal_path.read_text())
    result = {
        "schema": "r10.corrected-eligibility-search-continuation/1",
        "decision": "MULTILEVEL_SEARCH_RESUMED_WITH_RESTARTABLE_ACCEPTED_PROGRESS",
        "arrhenius_zero_slope_regression": arrhenius,
        "original_weakt_450_stop_receipt": stop_receipt,
        "selected_candidates": selected,
        "new_full_state_progress": {
            "candidate_id": WEAK_V2,
            "complete_row_sha256": comparison["complete_row_sha256"],
            "developed_low_rate_m_per_cycle": low_rate,
            "mid": mid, "high": high,
            "startup_mid_to_high_apparent_slope": partial_slope,
            "startup_slope_is_not_developed_evidence": True,
            "restart_contract": "resume the existing atomic checkpoints; do not restart from cycle zero",
        },
        "joint_gates": {
            "historical_1d_fracture_span": fracture_span,
            "weakT_span_limit": 1.25,
            "fracture_pass": fracture_span <= 1.25,
            "same_row_2D_cases": int(comparison["sealed_2d_cases"]),
            "same_row_2D_all_reached_1000um": comparison["sealed_2d_all_reached_1000um"] == "True",
            "fatigue_status": "IN_PROGRESS_TWO_RESTARTABLE_POINTS_PLUS_ONE_DEVELOPED_LOW_POINT",
        },
        "class_specific_feasible_pareto": {
            "Ceramic-like": "retained control; no Paris requirement",
            "Weak-T": (
                "Family-B V2-P25 is current joint frontier: fracture and sealed 2D pass; fatigue "
                "completion pending. It cannot count as strict-family-A success."
            ),
            "DBTT": (
                "developed moderate fatigue evidence preserved; endpoint-descended thermal rows remain "
                "diagnostic/non-promotable; original/refined parents remain in acquisition pool"
            ),
            "Peak-T": (
                "developed fatigue evidence preserved; opening-dominated peak classification retained; "
                "endpoint-descended final row remains non-promotable"
            ),
        },
        "preserved_mechanism_classifications": {
            "DBTT": causal["mechanism_classification"]["DBTT"],
            "Peak-T": causal["mechanism_classification"]["Peak-T"],
        },
        "next_acquisition": [
            "resume V2-P25 mid and high cases from their accepted atomic checkpoints to developed 20-60 um rates",
            "use the measured A0-to-current discrepancy to rank strict-family-A reference-state opening plus nucleation/motion proposals",
            "retain direct Arrhenius coefficients and inherited cross-process couplings; no independent endpoint coordinates",
            "send no new row to 2D until the joint 1D gates pass; reuse the exact sealed V2-P25 2D receipts meanwhile",
        ],
        "sources": {
            "original_registry": {"path": str(a.original_registry.resolve()), "sha256": sha256(a.original_registry)},
            "controls_registry": {"path": str(a.controls_registry.resolve()), "sha256": sha256(a.controls_registry)},
            "v2_rows": {"path": str(a.v2_rows.resolve()), "sha256": sha256(a.v2_rows)},
        },
    }
    (a.out_root / "joint_search_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
