#!/usr/bin/env python3
"""Analyze the broad moderate-window native batch and its restartable paths."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from scripts.analyze_r10_window_observables import finite_window_reference, full_window

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "integration_evidence/r10_global_mechanism_search"
PLAN_DIR = BASE / "moderate_window_batch"
RUN = BASE / "native_batch_v9_moderate"
OUT = PLAN_DIR / "analysis"


def csv_out(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def slope(a: dict, b: dict, field: str) -> float:
    return math.log(b[field] / a[field]) / math.log(b["DeltaK_MPa_sqrt_m"] / a["DeltaK_MPa_sqrt_m"])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    registry = {r["candidate_id"]: r for r in csv.DictReader((PLAN_DIR / "registry.csv").open(newline=""))}
    descriptors = {r["candidate_id"]: r for r in csv.DictReader((PLAN_DIR / "descriptors.csv").open(newline=""))}
    plan = json.loads((PLAN_DIR / "load_plan.json").read_text())["candidates"]
    rows = []
    for candidate, p in plan.items():
        ref = {round(float(K), 8): float(rate) for K, rate in zip(
            p["Kmax_MPa_sqrt_m"], p["A0_physical_m_per_cycle"])}
        for result_path in sorted((RUN / candidate).glob("fatigue_Kmax_*/result.json")):
            result = json.loads(result_path.read_text())
            full = full_window(result); finite = finite_window_reference(
                result, registry[candidate], PLAN_DIR / "registry.csv")
            row = {
                "candidate_id": candidate,
                "mechanism_acquisition_stratum": descriptors[candidate]["mechanism_acquisition_stratum"],
                "thermal_proxy": descriptors[candidate]["thermal_acquisition_stratum"],
                "temperature_K": result["temperature_K"], "Kmax_MPa_sqrt_m": result["Kmax_MPa_sqrt_m"],
                "DeltaK_MPa_sqrt_m": result["DeltaK_MPa_sqrt_m"], "status": result["status"],
                "event_count": len(result["events"]), "final_cycles": result["final_cycles"],
                "final_extension_um": 1e6 * result["final_extension_m"],
                "stationary_A0_rate_m_per_cycle": ref[round(float(result["Kmax_MPa_sqrt_m"]), 8)],
                "artifact": str(result_path.resolve()),
            }
            if full:
                for key, value in full.items(): row[f"full_{key}"] = value
            if finite:
                for key, value in finite.items(): row[f"finite_A0_{key}"] = value
            if full and finite:
                row["full_over_finite_A0_rate"] = full["rate_m_per_cycle"] / finite["rate_m_per_cycle"]
            rows.append(row)
    csv_out(OUT / "developed_points.csv", rows)

    pending = []
    for candidate, p in plan.items():
        for K in p["Kmax_MPa_sqrt_m"]:
            slug = "fatigue_Kmax_" + format(float(K), ".9g").replace(".", "p")
            case = RUN / candidate / slug
            if (case / "result.json").exists(): continue
            checkpoint_path = case / "checkpoint/high_cycle_live_checkpoint.json"
            if checkpoint_path.exists():
                checkpoint = json.loads(checkpoint_path.read_text())
                completed_advances = checkpoint["stochastic"].get("avalanche_event_length_history", [])
                pending.append({
                    "candidate_id": candidate, "Kmax_MPa_sqrt_m": K,
                    "status": "RESOURCE_BOUNDED_RESTARTABLE_ACCEPTED_EVENT_CHECKPOINT",
                    "cycles": checkpoint["cycles_from_engine_time"],
                    "events": checkpoint["stochastic"]["hazard_event_index"],
                    "extension_um": 1e6 * sum(completed_advances),
                    "checkpoint_reason": checkpoint["reason"],
                    "next_action_mark": checkpoint["stochastic"]["hazard_threshold_action"],
                    "checkpoint": str(checkpoint_path.resolve()),
                })
            else:
                pending.append({"candidate_id": candidate, "Kmax_MPa_sqrt_m": K,
                                "status": "QUEUED_NOT_STARTED"})
    csv_out(OUT / "pending_points.csv", pending)

    fracture = []
    for path in sorted(RUN.glob("R10A_GLOBAL_*/fracture_T_*.json")):
        d = json.loads(path.read_text())
        fracture.append({"candidate_id": d["candidate_id"], "temperature_K": d["temperature_K"],
                         "status": d["status"], "K_init_MPa_sqrt_m": d["K_init_MPa_sqrt_m"],
                         "event_count": d["event_count"], "artifact": str(path.resolve())})
    csv_out(OUT / "fracture_points.csv", fracture)

    opening = sorted((r for r in rows if r["candidate_id"] == "R10A_GLOBAL_002081"),
                     key=lambda r: r["DeltaK_MPa_sqrt_m"])
    full_endpoint = slope(opening[0], opening[-1], "full_rate_m_per_cycle")
    stationary_endpoint = slope(opening[0], opening[-1], "stationary_A0_rate_m_per_cycle")
    finite_endpoint = slope(opening[0], opening[-1], "finite_A0_rate_m_per_cycle")
    decision = {
        "schema": "r10.moderate-window-native-batch/1",
        "status": "ONE_COMPLETE_MODERATE_LADDER_PLUS_ONE_MIXED_DEVELOPED_POINT_AND_RESTARTABLE_REMAINDER",
        "family": "A_ORIGINAL_SINGLE_EXP_FLOOR",
        "completed_opening_candidate": "R10A_GLOBAL_002081",
        "completed_opening_ladder": {
            "developed_points": len(opening),
            "DeltaK_ratio": opening[-1]["DeltaK_MPa_sqrt_m"] / opening[0]["DeltaK_MPa_sqrt_m"],
            "rate_decades": math.log10(opening[-1]["full_rate_m_per_cycle"] / opening[0]["full_rate_m_per_cycle"]),
            "full_endpoint_slope": full_endpoint,
            "stationary_A0_endpoint_slope": stationary_endpoint,
            "finite_window_A0_endpoint_slope": finite_endpoint,
            "state_slope_correction_vs_finite_A0": full_endpoint - finite_endpoint,
            "minimum_window_DeltaN": min(r["full_DeltaN"] for r in opening),
            "all_windows_recurrent": all(not r["full_single_excursion_window"] for r in opening),
            "classification": "MODERATE_RECURRENT_OPENING_DOMINATED_CANDIDATE",
        },
        "mixed": {"developed_points": sum(r["candidate_id"] == "R10A_GLOBAL_011750" for r in rows),
                  "remaining_points_restartable_or_queued": sum(r["candidate_id"] == "R10A_GLOBAL_011750" for r in pending)},
        "plasticity": {"developed_points": sum(r["candidate_id"] == "R10A_GLOBAL_014670" for r in rows),
                       "remaining_points_restartable_or_queued": sum(r["candidate_id"] == "R10A_GLOBAL_014670" for r in pending)},
        "guard": "no pending checkpoint or startup segment is used as a developed rate",
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
