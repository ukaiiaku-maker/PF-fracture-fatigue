#!/usr/bin/env python3
"""Extract matched developed-window effects from the R10 native information batch."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def developed(result: dict) -> tuple[float | None, dict]:
    events = result.get("events", [])
    lo = next((e for e in events if float(e["extension_m"]) >= 20e-6), None)
    hi = next((e for e in events if float(e["extension_m"]) >= 60e-6), None)
    if lo is None or hi is None or float(hi["cycles"]) <= float(lo["cycles"]):
        return None, {"window_status": "INSUFFICIENT_20_60UM_COVERAGE"}
    rate = (float(hi["extension_m"]) - float(lo["extension_m"])) / (
        float(hi["cycles"]) - float(lo["cycles"]))
    return rate, {"window_status": "COMPLETE", "lo_event": int(lo["event_index"]),
                  "hi_event": int(hi["event_index"]), "lo_extension_um": 1e6 * float(lo["extension_m"]),
                  "hi_extension_um": 1e6 * float(hi["extension_m"]),
                  "lo_cycles": float(lo["cycles"]), "hi_cycles": float(hi["cycles"])}


def slope(a: dict, b: dict, field: str) -> float:
    return math.log(float(b[field]) / float(a[field])) / math.log(
        float(b["DeltaK_MPa_sqrt_m"]) / float(a["DeltaK_MPa_sqrt_m"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--native-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv); args.out.mkdir(parents=True, exist_ok=True)
    registry = list(csv.DictReader(args.registry.open(newline="")))
    cases: list[dict] = []; effects: list[dict] = []
    for record in registry:
        cid = record["candidate_id"]; root = args.native_root / cid
        if not (root / "opening_reference.json").exists():
            effects.append({"candidate_id": cid, "native_status": "NOT_STARTED"}); continue
        reference = json.loads((root / "opening_reference.json").read_text())
        refs = {float(r["Kmax_MPa_sqrt_m"]): r for r in reference["references"]}
        candidate_rows = []
        for directory in sorted(root.glob("fatigue_Kmax_*")):
            path = directory / "result.json"
            if not path.exists(): continue
            result = json.loads(path.read_text()); rate, window = developed(result)
            Kmax = float(result["Kmax_MPa_sqrt_m"]); ref = refs[Kmax]
            row = {"candidate_id": cid, "family": record["family"],
                   "acquisition_stratum": record["mechanism_acquisition_stratum"],
                   "thermal_stratum": record["thermal_acquisition_stratum"],
                   "temperature_K": float(result["temperature_K"]), "Kmax_MPa_sqrt_m": Kmax,
                   "DeltaK_MPa_sqrt_m": float(result["DeltaK_MPa_sqrt_m"]),
                   "native_status": result["status"], "event_count": len(result["events"]),
                   "final_extension_um": 1e6 * float(result["final_extension_m"]),
                   "full_developed_rate_m_per_cycle": rate,
                   "A0_physical_rate_m_per_cycle": float(ref["A0_physical_m_per_cycle"]),
                   "C_rate_full_over_A0": (rate / float(ref["A0_physical_m_per_cycle"])) if rate else None,
                   **window, "artifact": str(path.resolve())}
            cases.append(row); candidate_rows.append(row)
        complete = sorted([r for r in candidate_rows if r["full_developed_rate_m_per_cycle"]],
                          key=lambda r: r["DeltaK_MPa_sqrt_m"])
        if len(complete) < 3:
            effects.append({"candidate_id": cid, "native_status": "INCOMPLETE_DEVELOPED_LADDER",
                            "completed_points": len(complete)}); continue
        full_slopes = [slope(a, b, "full_developed_rate_m_per_cycle") for a, b in zip(complete, complete[1:])]
        a0_slopes = [slope(a, b, "A0_physical_rate_m_per_cycle") for a, b in zip(complete, complete[1:])]
        deltas = [x-y for x, y in zip(full_slopes, a0_slopes)]
        c = [float(r["C_rate_full_over_A0"]) for r in complete]
        max_dm = max(abs(x) for x in deltas)
        rate_factor = max(max(c), 1/min(c))
        c_amplitude_factor = max(c) / min(c)
        if max_dm >= 1.0:
            observed = "STRONG_STATE_SLOPE_EFFECT_REQUIRES_CAUSAL_CONFIRMATION"
        elif max_dm >= 0.5 or (rate_factor >= 1.5 and c_amplitude_factor >= 1.5):
            observed = "MIXED_EFFECT_REQUIRES_CAUSAL_CONFIRMATION"
        else:
            observed = "OPENING_DOMINATED_AT_OPERATIONAL_RESOLUTION"
        effects.append({"candidate_id": cid, "native_status": "COMPLETE_DEVELOPED_LADDER",
            "temperature_K": complete[0]["temperature_K"], "completed_points": len(complete),
            "full_adjacent_slopes_json": json.dumps(full_slopes),
            "A0_adjacent_slopes_json": json.dumps(a0_slopes),
            "Delta_m_state_json": json.dumps(deltas), "max_abs_Delta_m_state": max_dm,
            "C_rate_min": min(c), "C_rate_max": max(c), "rate_level_factor_from_unity": rate_factor,
            "C_rate_amplitude_factor": c_amplitude_factor, "observed_effect_class": observed,
            "final_mechanism_class": ("OPENING_DOMINATED" if observed.startswith("OPENING") else "UNCLASSIFIED_PENDING_COUNTERFACTUAL_AND_SENSITIVITY")})
    write_csv(args.out / "native_developed_cases.csv", cases)
    write_csv(args.out / "native_mechanism_effects.csv", effects)
    summary = {"schema": "r10.mechanism-balanced-native-analysis/1",
        "thresholds": {"mixed_abs_Delta_m": 0.5, "strong_abs_Delta_m": 1.0,
                       "rate_factor": 1.5, "two_percent_is_not_material": True},
        "candidate_count": len(registry),
        "complete_developed_ladders": sum(r.get("native_status") == "COMPLETE_DEVELOPED_LADDER" for r in effects),
        "effect_counts": {}, "classification_guard": "non-opening effects remain unclassified until causal freeze and normalized block sensitivity"}
    for row in effects:
        key = row.get("observed_effect_class", row["native_status"])
        summary["effect_counts"][key] = summary["effect_counts"].get(key, 0) + 1
    (args.out / "native_batch_decision.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
