#!/usr/bin/env python3
"""Consolidate the post-reconciliation coupled-coordinate execution."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


CANDIDATE = "R10K_WEAKT_COUPLED_EMIT_SCALE_S3P0_V1"


def endpoint(events: list[dict], threshold_m: float) -> dict:
    return next(x for x in events if float(x["extension_m"]) >= threshold_m)


def developed(result: dict) -> dict:
    left = endpoint(result["events"], 20.0e-6)
    right = endpoint(result["events"], 60.0e-6)
    cycles = float(right["cycles"]) - float(left["cycles"])
    extension = float(right["extension_m"]) - float(left["extension_m"])
    return {
        "DeltaK_MPa_sqrt_m": float(result["DeltaK_MPa_sqrt_m"]),
        "left_event_index": int(left["event_index"]),
        "right_event_index": int(right["event_index"]),
        "cycles": cycles,
        "extension_m": extension,
        "rate_m_per_cycle": extension / cycles,
        "event_count": len(result["events"]),
        "final_cycles": float(result["final_cycles"]),
        "status": result["status"],
        "native_qualification_accepted": bool(result["native_qualification"]["accepted"]),
        "one_cycle_overlap_passed": bool(result["overlap"]["passed"]),
        "one_cycle_state_relative_max_error": float(result["overlap"]["state_relative_max_error"]),
    }


def panel(path: Path) -> list[dict]:
    return json.loads(path.read_text())["cases"]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    a = p.parse_args(argv)
    a.out_root.mkdir(parents=True, exist_ok=True)
    main_root = a.run_root / "weakt_coupled_scale_continuation"
    fracture_path = main_root / "full_state_refined" / CANDIDATE / "panel_summary.json"
    fracture = panel(fracture_path)
    roots = [float(x["K_init_MPa_sqrt_m"]) for x in fracture]
    fracture_span = max(roots) / min(roots)
    fracture_rows = [{**x, "source_artifact": str(fracture_path.resolve())} for x in fracture]
    write_csv(a.out_root / "corrected_fracture_curve.csv", fracture_rows)

    fatigue_root = main_root / "fatigue" / "s3p0_seed4810_high_pair"
    preflight = json.loads((fatigue_root / "production_preflight.json").read_text())
    bound_hash = preflight["binding"]["complete_row_sha256"]
    if preflight["binding"]["candidate_id"] != CANDIDATE:
        raise RuntimeError("fatigue preflight was not bound to the corrected row")
    rates = []
    for result_path in sorted(fatigue_root.glob("Kmax_*/result.json")):
        result = json.loads(result_path.read_text())
        if result["candidate_id"] != CANDIDATE:
            raise RuntimeError("fatigue result was not bound to the corrected row")
        rates.append({
            "candidate_id": CANDIDATE,
            "complete_row_sha256": bound_hash,
            **developed(result),
            "source_artifact": str(result_path.resolve()),
        })
    if len(rates) != 2:
        raise RuntimeError("corrected same-row high fatigue pair is incomplete")
    rates.sort(key=lambda x: x["DeltaK_MPa_sqrt_m"])
    slope = math.log(rates[1]["rate_m_per_cycle"] / rates[0]["rate_m_per_cycle"]) / math.log(
        rates[1]["DeltaK_MPa_sqrt_m"] / rates[0]["DeltaK_MPa_sqrt_m"]
    )
    for row in rates:
        row["high_pair_slope"] = slope
        row["passes_slope_3p4_to_4p6"] = 3.4 <= slope <= 4.6
    write_csv(a.out_root / "corrected_fatigue_high_pair.csv", rates)

    response_rows = []
    response_panels = [
        a.run_root / "weakt_coupled_scale_bracket/hot_refined/R10K_WEAKT_COUPLED_EMIT_SCALE_S2P4_V1/panel_summary.json",
        a.run_root / "weakt_coupled_opening_scale/hot_refined/R10K_WEAKT_COUPLED_S3P0_OPEN_SCALE_0P7_V1/panel_summary.json",
    ]
    for path in response_panels:
        for row in panel(path):
            response_rows.append({
                "candidate_id": row["candidate_id"],
                "temperature_K": row["temperature_K"],
                "K_init_MPa_sqrt_m": row["K_init_MPa_sqrt_m"],
                "status": row["status"],
                "source_artifact": str(path.resolve()),
            })
    write_csv(a.out_root / "coupled_coordinate_response.csv", response_rows)

    lineage = json.loads((a.run_root /
        "constitutive_temperature_reconciliation/reconciliation.json").read_text())
    decision = {
        "schema": "r10.constitutive-coupled-continuation-result/1",
        "candidate_id": CANDIDATE,
        "complete_row_sha256": rates[0]["complete_row_sha256"],
        "constitutive_reconciliation": {
            "decision": lineage["decision"],
            "portable_source_sha256": lineage["portable_source"]["sha256"],
            "historical_endpoint_rows": "PRESERVED_NON_PROMOTABLE",
            "independent_endpoint_optimization_withdrawn": True,
            "empirical_temperature_law_added": False,
            "activation_entropy_zeroed": False,
        },
        "same_complete_material_execution": {
            "fracture": "fresh refined seven-temperature current-backend panel",
            "fatigue": "fresh candidate-bound native 300 K 15/22 DeltaK developed pair",
            "shared_complete_row_sha256": rates[0]["complete_row_sha256"],
        },
        "fracture": {
            "status": "FAIL_REFINED_WEAK_T_SPAN",
            "temperatures_K": [float(x["temperature_K"]) for x in fracture],
            "K_init_MPa_sqrt_m": roots,
            "span": fracture_span,
            "limit": 1.25,
        },
        "fatigue": {
            "status": "PASS_HIGH_PAIR" if 3.4 <= slope <= 4.6 else "FAIL_HIGH_PAIR",
            "DeltaK_MPa_sqrt_m": [x["DeltaK_MPa_sqrt_m"] for x in rates],
            "rates_m_per_cycle": [x["rate_m_per_cycle"] for x in rates],
            "slope": slope,
            "new_execution_scope": "two-point 15/22 MPa sqrt(m) secant",
            "new_rate_span_decades": math.log10(
                rates[1]["rate_m_per_cycle"] / rates[0]["rate_m_per_cycle"]
            ),
            "full_historical_five_point_curve": (
                "preserved as transferred fixed-300-K numerical evidence only; "
                "not newly executed or promoted as five-point final-row qualification"
            ),
        },
        "joint_decision": "NO_PROMOTION_FRACTURE_GATE_FAILED",
        "same_row_2D_status": "NOT_RUN_JOINT_1D_GATE_FAILED",
        "unevaluated_predicted_uncompetitive_rows": [
            "R10K_WEAKT_COUPLED_EMIT_SCALE_S2P6_V1",
            "R10K_WEAKT_COUPLED_EMIT_SCALE_S2P8_V1",
            "R10K_WEAKT_COUPLED_S3P0_OPEN_SCALE_0P6_V1",
            "R10K_WEAKT_COUPLED_S3P0_OPEN_SCALE_0P8_V1",
        ],
        "next_search_constraint": (
            "select from ancestry-qualified original/refined/retained-bank complete rows and "
            "use every physical parameter direction authorized by that row's source family; "
            "exclude unsupported endpoint degrees of freedom without imposing a P1/P2-only "
            "or opening-only restriction"
        ),
        "preserved_results": [
            "all pre-reconciliation runs and checkpoints",
            "historical S3.0 five-point fatigue curve as non-promotable diagnostic evidence",
            "opening-dominated common-barrier positive control",
            "DBTT and Peak fatigue evidence with their existing causal classifications",
        ],
    }
    (a.out_root / "joint_scientific_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
