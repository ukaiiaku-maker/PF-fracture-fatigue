#!/usr/bin/env python3
"""Analyze the bounded A-D high-DeltaK scout and spatial extension."""
from __future__ import annotations

import csv
import json
from pathlib import Path

CLASSES = {
    "v914_endurance_knee_0462": "A",
    "v914_endurance_knee_0658": "B",
    "v914_endurance_knee_0554": "C",
    "v914_endurance_knee_0133": "D",
}
SELECTED = {
    ("A", 3.0): "H1_intermediate_high", ("A", 10.0): "H2_few_cycle",
    ("B", 1.3): "H1_intermediate_high", ("B", 2.0): "H2_few_cycle",
    ("C", 3.0): "H1_intermediate_high", ("C", 10.0): "H2_few_cycle",
    ("D", 4.0): "H1_intermediate_high", ("D", 10.0): "H2_few_cycle",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    root = Path("runs/v10_2_31_endurance_knee_ABCD_high_deltaK_v1")
    records = []
    for path in sorted((root / "one_d_scout_raw").glob("**/fatigue_result.json")):
        data = json.loads(path.read_text())
        cls = CLASSES[data["candidate_id"]]
        events = data.get("events", [])
        cycles = float(data["final_cycles"])
        fraction = float(data["fraction"])
        rate = float(data["developed_da_dN_m_per_cycle"])
        intervals = [float(e["interval_cycles"]) for e in events]
        near = cycles <= 1.0 and bool(intervals) and sum(x < 1 for x in intervals) / len(intervals) >= .5
        category = "near_monotonic" if near else "few_cycle" if cycles <= 10 else "plateau_or_fatigue"
        records.append({
            "class": cls, "candidate": data["candidate_id"],
            "deltaK_MPa_sqrt_m": float(data["loading"]["deltaK_MPa_sqrt_m"]),
            "f": fraction, "one_d_da_dN_m_per_cycle": rate,
            "total_cycles_to_target": cycles, "fraction_of_cycle": cycles if cycles < 1 else "",
            "event_count": len(events), "stopping_reason": data["status"],
            "target_rate_category": category, "near_monotonic": near,
            "selected_2D_role": SELECTED.get((cls, fraction), "scout_only"),
            "minimum_event_interval_cycles": min(intervals) if intervals else "",
            "median_event_interval_cycles": sorted(intervals)[len(intervals)//2] if intervals else "",
        })
    records.sort(key=lambda row: (row["class"], row["f"]))
    write_csv(root / "analysis" / "abcd_high_deltaK_1D_scout.csv", records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
