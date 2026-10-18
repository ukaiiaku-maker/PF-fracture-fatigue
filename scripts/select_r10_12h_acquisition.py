#!/usr/bin/env python3
"""Select diverse native/HCF cases from the current full-state discrepancy map."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, rows):
    rows = list(rows)
    fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def boolean(value):
    return str(value).lower() == "true"


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def category(row):
    if boolean(row["unsupported"]):
        return "UNRESOLVED_MIXED"
    if boolean(row["one_cycle_event_reachable"]) or boolean(row["second_cycle_event_reachable"]):
        return "FAST_EVENT_BOUNDARY"
    action = number(row["action_timescale_cycles"])
    state = number(row["successor_state_timescale_cycles"])
    times = [x for x in (action, state) if x is not None and 1e2 <= x <= 1e9]
    if times:
        return "FINITE_HCF_TIMESCALE"
    return "QUIET_BONDED_BOUNDARY"


def exposure(row):
    values = [number(row["action_timescale_cycles"]),
              number(row["successor_state_timescale_cycles"])]
    values = [x for x in values if x is not None and x > 0]
    coordinate = min(values, key=lambda x: abs(math.log10(x) - 5.)) if values else 1e4
    levels = (1e2, 1e4, 1e6, 1e8)
    return min(levels, key=lambda x: abs(math.log10(x) - math.log10(coordinate))), coordinate


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-map", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--exclude-summary", type=Path, action="append", default=[])
    parser.add_argument("--exclude-source-ids", default="")
    parser.add_argument("--mode", choices=("cohesive_athermal", "cohesive_mixed"),
                        help="Optionally acquire within one opening owner only.")
    args = parser.parse_args(argv)
    mapped = read(args.state_map)
    if args.mode:
        mapped = [row for row in mapped if row["mode"] == args.mode]
    registry = {r["candidate_id"]: r for r in read(args.registry)}
    excluded_sources = set()
    excluded_sources.update(x.strip() for x in args.exclude_source_ids.split(",") if x.strip())
    for path in args.exclude_summary:
        if path.is_file():
            excluded_sources.update(r.get("source_material_candidate_id", "") for r in read(path))

    by_source = {}
    for row in mapped:
        if row["source_material_candidate_id"] in excluded_sources:
            continue
        by_source.setdefault(row["source_material_candidate_id"], []).append(row)
    candidates = []
    for source, arm in by_source.items():
        for identity in sorted({(r["mode"],r["shape"]) for r in arm}):
            identity_arm=[r for r in arm if (r["mode"],r["shape"])==identity]
            bounded = [r for r in identity_arm if category(r) not in
                       {"FAST_EVENT_BOUNDARY", "UNRESOLVED_MIXED"}]
            event_factors = [float(r["reference_factor"]) for r in identity_arm
                             if category(r) == "FAST_EVENT_BOUNDARY"]
            if bounded:
                # Prefer the highest supported sub-event load with a finite physical timescale.
                def key(r):
                    target, coordinate = exposure(r)
                    finite = category(r) == "FINITE_HCF_TIMESCALE"
                    bracket = bool(event_factors and float(r["reference_factor"]) < min(event_factors))
                    return (not finite, not bracket, abs(math.log10(coordinate) - 5.),
                            -float(r["reference_factor"]))
                chosen = min(bounded, key=key)
            else:
                chosen = min(identity_arm,key=lambda r:abs(float(r["reference_factor"])-.97))
            chosen = dict(chosen)
            chosen["acquisition_category"] = category(chosen)
            chosen["has_higher_event_boundary"] = bool(
                event_factors and float(chosen["reference_factor"]) < max(event_factors))
            chosen["event_boundary_factors"] = json.dumps(sorted(event_factors))
            candidates.append(chosen)

    # Interleave every physical stratum rather than allowing one easy response to dominate.
    groups = {}
    for row in candidates:
        key = (row["search_design"], row["mechanism_prediction"],
               row["mode"], row["shape"], row["acquisition_category"])
        groups.setdefault(key, []).append(row)
    for group in groups.values():
        group.sort(key=lambda r: (not r["has_higher_event_boundary"],
                                  abs(math.log10(exposure(r)[1]) - 5.),
                                  r["source_material_candidate_id"]))
    selected = [];selected_sources=set()
    while len(selected) < min(args.count, len(candidates)):
        advanced = False
        for key in sorted(groups):
            while groups[key] and groups[key][0]["source_material_candidate_id"] in selected_sources:
                groups[key].pop(0)
            if groups[key]:
                picked=groups[key].pop(0);selected.append(picked)
                selected_sources.add(picked["source_material_candidate_id"]);advanced = True
                if len(selected) >= args.count:
                    break
        if not advanced:
            break

    native = []; hcf = []
    for rank, row in enumerate(selected, 1):
        material = registry[row["candidate_id"]]
        native.append({"acquisition_rank": rank, "candidate_id": row["candidate_id"],
            "source_material_candidate_id": row["source_material_candidate_id"],
            "search_design": row["search_design"], "shape": row["shape"], "mode": row["mode"],
            "mechanism_prediction": row["mechanism_prediction"],
            "thermal_prediction": row["thermal_prediction"],
            "reference_factor": row["reference_factor"],
            "Kmax_MPa_sqrt_m": row["Kmax_MPa_sqrt_m"],
            "acquisition_category": row["acquisition_category"],
            "has_higher_event_boundary": row["has_higher_event_boundary"],
            "event_boundary_factors": row["event_boundary_factors"],
            "complete_row_sha256": material["complete_row_sha256"],
            "plasticity_physics_sha256": material["plasticity_physics_sha256"],
            "cohesive_parameter_sha256": material["cohesive_parameter_sha256"]})
        target, coordinate = exposure(row)
        if row["acquisition_category"] not in {"FAST_EVENT_BOUNDARY", "UNRESOLVED_MIXED"}:
            hcf.append({"candidate_id": row["candidate_id"],
                "source_material_candidate_id": row["source_material_candidate_id"],
                "search_design": row["search_design"], "shape": row["shape"], "mode": row["mode"],
                "reference_factor": row["reference_factor"],
                "temperature_K": row["temperature_K"], "R": row["R"],
                "frequency_Hz": row["frequency_Hz"],
                "exposure_target_cycles": target,
                "selection_coordinate_cycles": coordinate,
                "selection_basis": row["acquisition_category"],
                "has_higher_event_boundary": row["has_higher_event_boundary"]})
    write(args.out / "native_material_selection.csv", native)
    write(args.out / "hcf_case_selection.csv", hcf)
    summary = {"selected_physical_vectors": len(selected), "hcf_cases": len(hcf),
               "excluded_previously_executed_sources": len(excluded_sources),
               "categories": {x: sum(r["acquisition_category"] == x for r in selected)
                              for x in sorted({r["acquisition_category"] for r in selected})}}
    (args.out / "selection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
