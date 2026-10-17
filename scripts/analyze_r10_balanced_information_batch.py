#!/usr/bin/env python3
"""Consolidate the first completed mechanism-balanced R10 information block."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "integration_evidence/r10_global_mechanism_search"
OUT = BASE / "balanced_information_batch"

CASES = {
    "OPENING_DIRECTED": {
        "candidate_id": "R10A_GLOBAL_018756", "temperature_K": 300.0,
        "fatigue": BASE / "native_batch_v2/R10A_GLOBAL_018756",
        "fracture": BASE / "native_batch_v2/R10A_GLOBAL_018756",
    },
    "MIXED_DIRECTED": {
        "candidate_id": "R10A_GLOBAL_119268", "temperature_K": 600.0,
        "fatigue": BASE / "native_batch_v5/R10A_GLOBAL_119268",
        "replicate": BASE / "native_batch_v8/R10A_GLOBAL_119268",
        "fracture": BASE / "native_batch_v2/R10A_GLOBAL_119268",
    },
    "PLASTICITY_DIRECTED": {
        "candidate_id": "R10A_GLOBAL_064416", "temperature_K": 900.0,
        "fatigue": BASE / "native_batch_v6/R10A_GLOBAL_064416",
        "fracture": BASE / "native_batch_v4/R10A_GLOBAL_064416",
    },
}


def developed(result: dict):
    lo = next((e for e in result["events"] if e["extension_m"] >= 20e-6), None)
    hi = next((e for e in result["events"] if e["extension_m"] >= 60e-6), None)
    if lo is None or hi is None or hi["cycles"] <= lo["cycles"]:
        return None
    return (hi["extension_m"]-lo["extension_m"])/(hi["cycles"]-lo["cycles"])


def refs(directory: Path) -> dict[float, dict]:
    payload = json.loads((directory / "opening_reference.json").read_text())
    return {round(float(x["Kmax_MPa_sqrt_m"]), 8): x for x in payload["references"]}


def rows(directory: Path, lane: str, stream: str) -> list[dict]:
    reference = refs(directory)
    out = []
    for path in sorted(directory.glob("fatigue_Kmax_*/result.json")):
        result = json.loads(path.read_text())
        ref = reference[round(float(result["Kmax_MPa_sqrt_m"]), 8)]
        rate = developed(result)
        cycles = float(result["final_cycles"])
        zero_event_upper = (-math.log(0.05) * 5e-6 / cycles
                            if not result["events"] and cycles > 0 else None)
        out.append({
            "lane": lane, "stream": stream, "candidate_id": result["candidate_id"],
            "temperature_K": result["temperature_K"], "Kmax_MPa_sqrt_m": result["Kmax_MPa_sqrt_m"],
            "DeltaK_MPa_sqrt_m": result["DeltaK_MPa_sqrt_m"], "status": result["status"],
            "events": len(result["events"]), "final_cycles": cycles,
            "final_extension_um": 1e6 * float(result["final_extension_m"]),
            "developed_20_60_rate_m_per_cycle": rate,
            "A0_physical_rate_m_per_cycle": float(ref["A0_physical_m_per_cycle"]),
            "C_rate_full_over_A0": (rate / float(ref["A0_physical_m_per_cycle"]) if rate else None),
            "zero_event_95pct_rate_upper_m_per_cycle": zero_event_upper,
            "zero_event_bound_scope": ("MODEL_CONDITIONAL_HOMOGENEOUS_POISSON_SURROGATE"
                                        if zero_event_upper else None),
            "C_rate_95pct_upper": (zero_event_upper / float(ref["A0_physical_m_per_cycle"])
                                   if zero_event_upper else None),
            "mean_cycle_overlap_passed": bool(result["overlap"]["passed"]),
            "artifact": str(path.resolve()),
        })
    return out


def csv_out(path: Path, records: list[dict]) -> None:
    keys = []
    for row in records:
        for key in row:
            if key not in keys: keys.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n"); writer.writeheader(); writer.writerows(records)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for lane, case in CASES.items():
        all_rows.extend(rows(case["fatigue"], lane, "PRIMARY"))
        if "replicate" in case:
            all_rows.extend(rows(case["replicate"], lane, "INDEPENDENT_MARK_STREAM"))
    csv_out(OUT / "fatigue_outcomes.csv", all_rows)
    fracture = []
    for lane, case in CASES.items():
        for path in sorted(case["fracture"].glob("fracture_T_*.json")):
            result = json.loads(path.read_text())
            fracture.append({"lane": lane, "candidate_id": result["candidate_id"],
                "temperature_K": result["temperature_K"], "status": result["status"],
                "K_init_MPa_sqrt_m": result["K_init_MPa_sqrt_m"],
                "final_extension_um": result["final_extension_um"], "event_count": result["event_count"],
                "artifact": str(path.resolve())})
    csv_out(OUT / "fracture_outcomes.csv", fracture)

    opening = sorted([r for r in all_rows if r["lane"] == "OPENING_DIRECTED" and r["developed_20_60_rate_m_per_cycle"]],
                     key=lambda r: r["DeltaK_MPa_sqrt_m"])
    mixed = sorted([r for r in all_rows if r["lane"] == "MIXED_DIRECTED" and r["stream"] == "PRIMARY"
                    and r["developed_20_60_rate_m_per_cycle"]], key=lambda r: r["DeltaK_MPa_sqrt_m"])
    def adjacent(a, b, field):
        return math.log(b[field]/a[field])/math.log(b["DeltaK_MPa_sqrt_m"]/a["DeltaK_MPa_sqrt_m"])
    opening_effect = [adjacent(a, b, "developed_20_60_rate_m_per_cycle")-
                      adjacent(a, b, "A0_physical_rate_m_per_cycle") for a, b in zip(opening, opening[1:])]
    opening_full_slopes = [adjacent(a, b, "developed_20_60_rate_m_per_cycle")
                           for a, b in zip(opening, opening[1:])]
    opening_a0_slopes = [adjacent(a, b, "A0_physical_rate_m_per_cycle")
                         for a, b in zip(opening, opening[1:])]
    mixed_full_slope = adjacent(mixed[0], mixed[1], "developed_20_60_rate_m_per_cycle")
    mixed_a0_slope = adjacent(mixed[0], mixed[1], "A0_physical_rate_m_per_cycle")
    plastic = [r for r in all_rows if r["lane"] == "PLASTICITY_DIRECTED"]
    decision = {
        "schema": "r10.mechanism-balanced-information-batch/1",
        "status": "COMPLETE_INFORMATION_BATCH_WITH_DEVELOPED_RATES_AND_PHYSICAL_CENSORS",
        "family": "A_ORIGINAL_SINGLE_EXP_FLOOR",
        "operational_thresholds": {"mixed_abs_Delta_m": 0.5, "strong_abs_Delta_m": 1.0,
                                   "rate_factor": 1.5, "two_percent_is_not_material": True},
        "lanes": {
            "OPENING_DIRECTED": {"candidate_id": CASES["OPENING_DIRECTED"]["candidate_id"],
                "developed_points": len(opening), "max_abs_Delta_m_state": max(map(abs, opening_effect)),
                "C_rate_range": [min(r["C_rate_full_over_A0"] for r in opening),
                                 max(r["C_rate_full_over_A0"] for r in opening)],
                "absolute_full_slopes": opening_full_slopes,
                "absolute_A0_slopes": opening_a0_slopes,
                "result": "OPENING_DOMINATED_BUT_STEEP; NOT_A_MODERATE_PARIS_WINDOW"},
            "MIXED_DIRECTED": {"candidate_id": CASES["MIXED_DIRECTED"]["candidate_id"],
                "primary_developed_points": len(mixed), "full_slope": mixed_full_slope,
                "A0_slope": mixed_a0_slope, "Delta_m_state": mixed_full_slope-mixed_a0_slope,
                "C_rate_range": [min(r["C_rate_full_over_A0"] for r in mixed),
                                 max(r["C_rate_full_over_A0"] for r in mixed)],
                "upper_point_is_single_excursion_burst": True,
                "independent_stream_outcomes": {r["Kmax_MPa_sqrt_m"]: r["status"]
                    for r in all_rows if r["lane"] == "MIXED_DIRECTED" and r["stream"] != "PRIMARY"},
                "result": "STRONG_AMPLITUDE_DEPENDENT_EFFECT_WITH_STREAM_SENSITIVITY; UPPER POINT IS A FINITE-WINDOW BURST; FINAL LABEL WITHHELD"},
            "PLASTICITY_DIRECTED": {"candidate_id": CASES["PLASTICITY_DIRECTED"]["candidate_id"],
                "fatigue_points": len(plastic), "zero_event_points": sum(r["events"] == 0 for r in plastic),
                "cycles_per_point": sorted(set(r["final_cycles"] for r in plastic)),
                "maximum_C_rate_95pct_upper": max(r["C_rate_95pct_upper"] for r in plastic),
                "zero_event_bound_scope": "MODEL_CONDITIONAL_HOMOGENEOUS_POISSON_SURROGATE_NOT_MODEL_FREE",
                "result": "FINITE_ZERO_EVENT_CENSOR_DESPITE_A0_GROWTH; NOT ASYMPTOTIC ARREST; NO DEVELOPED RATE OR FINAL LABEL"},
        },
        "classification_guard": "acquisition strata are not final mechanism labels; mixed/plastic final labels require frozen-feedback and normalized block sensitivities",
        "same_material_fracture_reachability": {
            "opening": "complete at 300 K", "mixed": "complete at 300 and 600 K",
            "plasticity": "complete at 300 and 900 K",
        },
        "remaining_48_row_allocation": "ALLOCATION_ONLY; see information_scope_audit for executed/queued dispositions",
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
