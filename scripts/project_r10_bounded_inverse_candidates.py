#!/usr/bin/env python3
"""Project bounded inverse optima into complete opt-in research materials.

The inverse table deliberately contains only analytical diagnostics.  This
bridge joins an admissible optimum back to its unchanged EXP_floor source row,
evaluates the complete moving-boundary cycle integral, and emits a registry
that the existing state-informed/native consumers can execute.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.cohesive_arrhenius import COMPACT_INDEPENDENT
from scripts.run_r10_finite_compliance_inverse_shape import (
    SCHEMA,
    _cyclic_registry_point,
)
from scripts.run_r10_global_mechanism_search import canonical_hash


def atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def choose(rows: list[dict], count: int) -> list[dict]:
    valid = [row for row in rows if row.get("domain_pass") == "True"]
    valid.sort(key=lambda row: (float(row.get("objective") or math.inf),
                                row["candidate_id"]))
    selected: list[dict] = []
    seen_plasticity: set[str] = set()
    # First preserve independent EXP_floor backgrounds, then fill from the
    # bounded objective frontier.  Shape is fixed by construction.
    for distinct in (True, False):
        for row in valid:
            if row in selected:
                continue
            plasticity = str(row.get("source_plasticity_sha256") or "")
            if distinct and plasticity in seen_plasticity:
                continue
            selected.append(row)
            seen_plasticity.add(plasticity)
            if len(selected) >= count:
                return selected
    return selected


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inverse-table", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    inverse_rows = read_csv(args.inverse_table)
    sources = read_csv(args.source_registry)
    source_by_id = {row["candidate_id"]: row for row in sources}
    selected = choose(inverse_rows, args.count)
    projected: list[dict] = []
    projection_metrics: list[dict] = []
    for metric in selected:
        source_id = metric["source_material_candidate_id"]
        if source_id not in source_by_id:
            raise ValueError(f"inverse source missing from registry: {source_id}")
        source = dict(source_by_id[source_id])
        cohesive = json.loads(metric["optimized_cohesive_json"])
        cyclic = _cyclic_registry_point((metric, cohesive, COMPACT_INDEPENDENT))
        chosen = cyclic.get("chosen")
        if chosen is None:
            hcf_K = float(metric["metastable_birth_K_300K"])
            hcf_action = 0.0
            hcf_slope = None
            accessible_count = 0
            accessible_fraction = 0.0
            basis = "PEAK_FALLBACK_NO_CYCLIC_ACCESS"
        else:
            hcf_K = float(chosen["Kmax_MPa_sqrt_m"])
            hcf_action = float(chosen["opening_action_per_cycle"])
            hcf_slope = cyclic.get("total_local_slope")
            accessible_count = int(chosen["accessible_phase_count"])
            accessible_fraction = float(chosen.get("accessible_phase_fraction") or 0.0)
            basis = ("BOUNDED_INVERSE_MOVING_BOUNDARY_GAUSS_32_ORDER__" +
                     str(cyclic.get("selection_support", "UNDECLARED_SUPPORT")))
        candidate_id = metric["candidate_id"] + "__CM"
        peak_curve = json.loads(metric["curve_json"])
        peak_at_hcf = min(peak_curve,
                          key=lambda point: abs(float(point["K_MPa_sqrt_m"]) - hcf_K))
        row = dict(source)
        row.update({
            "candidate_id": candidate_id,
            "source_material_candidate_id": source_id,
            "selected_shape": COMPACT_INDEPENDENT,
            "selected_mode": "cohesive_mixed",
            "search_design": "FINITE_COMPLIANCE_BOUNDED_INVERSE_SHAPE",
            "research_status": "OPT_IN_RESEARCH_CANDIDATE",
            "cohesive_parameter_vector": json.dumps(cohesive, sort_keys=True,
                                                       separators=(",", ":")),
            "predicted_fold_K_300K": metric["fold_K_300K"],
            "predicted_Kaccess_300K": metric["fold_K_300K"],
            "predicted_hcf_K_300K": hcf_K,
            "predicted_hcf_factor_300K": hcf_K / float(metric["fold_K_300K"]),
            "predicted_hcf_cyclic_action_per_cycle": hcf_action,
            "predicted_hcf_peak_action_per_cycle":
                float(peak_at_hcf["events_per_cycle_peak_rate"]),
            "predicted_hcf_local_slope": hcf_slope,
            "predicted_hcf_accessible_phases": accessible_count,
            "predicted_hcf_accessible_phase_fraction": accessible_fraction,
            "predicted_hcf_selection_basis": basis,
            "thermal_prediction": metric["thermal_prediction"],
            "mechanism_prediction": metric["mechanism_prediction"],
            "inverse_objective": metric["objective"],
            "inverse_moderate_width_log10rate": metric["moderate_connected_width_log10rate"],
            "inverse_strict_width_log10rate": metric["strict_connected_width_log10rate"],
        })
        row["cohesive_parameter_sha256"] = canonical_hash(cohesive)
        row["complete_row_sha256"] = canonical_hash({
            "source": source.get("complete_row_sha256"),
            "cohesive": cohesive,
            "shape": COMPACT_INDEPENDENT,
            "mode": "cohesive_mixed",
            "family": SCHEMA,
        })
        projected.append(row)
        projection_metrics.append({
            "candidate_id": candidate_id,
            "source_material_candidate_id": source_id,
            "complete_row_sha256": row["complete_row_sha256"],
            "inverse_objective": float(metric["objective"]),
            "peak_hcf_slope": float(metric["hcf_slope_median"])
                if metric.get("hcf_slope_median") not in (None, "", "nan") else None,
            "moderate_width_log10rate": float(metric["moderate_connected_width_log10rate"]),
            "cyclic_action_per_cycle": hcf_action,
            "cyclic_local_slope": hcf_slope,
            "accessible_phase_fraction": accessible_fraction,
            "selection_basis": basis,
        })

    registry = args.out / "projected_registry.csv"
    metrics = args.out / "projection_metrics.csv"
    write_csv(registry, projected)
    write_csv(metrics, projection_metrics)
    summary = {
        "schema": "r10.finite-compliance-bounded-inverse-projection/1",
        "inverse_table": str(args.inverse_table),
        "inverse_table_sha256": hashlib.sha256(args.inverse_table.read_bytes()).hexdigest(),
        "source_registry": str(args.source_registry),
        "source_registry_sha256": hashlib.sha256(args.source_registry.read_bytes()).hexdigest(),
        "eligible_bounded_rows": sum(row.get("domain_pass") == "True" for row in inverse_rows),
        "projected_count": len(projected),
        "projected_unique_plasticity_vectors": len({row.get("plasticity_physics_sha256")
                                                      for row in projected}),
        "algebraic_tail_registered": False,
        "candidate_ids": [row["candidate_id"] for row in projected],
    }
    atomic(args.out / "projection_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
