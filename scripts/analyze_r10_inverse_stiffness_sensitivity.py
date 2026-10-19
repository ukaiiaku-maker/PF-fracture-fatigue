#!/usr/bin/env python3
"""Diagnostic stiffness sensitivity for finite-compliance inverse candidates.

The factor is a declared sensitivity coordinate around the material-derived
environment, never an acquisition or native fitting parameter.
"""
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

from arrhenius_fracture.finite_compliance_inverse_shape_r10 import (
    K_from_imposed, cyclic_rate_curve, metastable_window,
)
from scripts.run_r10_finite_compliance_inverse_shape import _thermal
from scripts.run_r10_opening_alternatives import mechanics


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def evaluate(row: dict, factors: list[float], points: int, phases: int) -> list[dict]:
    cohesive = json.loads(row["cohesive_parameter_vector"])
    thermal = _thermal(cohesive)
    _, derived_stiffness = mechanics(thermal, 300.)
    shape = row["selected_shape"]
    output = []
    for factor in factors:
        stiffness = factor*derived_stiffness
        window = metastable_window(shape, thermal, 300., stiffness)
        base = {"candidate_id": row["candidate_id"],
            "complete_row_sha256": row["complete_row_sha256"],
            "stiffness_factor": factor,
            "derived_environment_stiffness_Pa_per_m": derived_stiffness,
            "evaluated_environment_stiffness_Pa_per_m": stiffness,
            "sensitivity_scope": "DIAGNOSTIC_ONLY_NOT_NATIVE_SUPPORT"}
        if window["state"] != "METASTABLE_WINDOW":
            output.append({**base, "status": window["state"]}); continue
        birth = K_from_imposed(float(window["birth_load_Pa"])/stiffness, stiffness)
        fold = K_from_imposed(float(window["fold_load_Pa"])/stiffness, stiffness)
        K = np.geomspace(birth*1.0001, fold*.9999, points)
        curve = cyclic_rate_curve(K, stiffness, thermal, 300., shape,
            float(cohesive["activation_area_m2"]),
            float(cohesive["attempt_frequency_s"]), R=.1,
            frequency_Hz=1000., phases=phases)
        supported = [point for point in curve["rows"]
            if point["opening_action_per_cycle"] > 0. and
               point["local_log_slope"] is not None]
        if not supported:
            output.append({**base, "status": "NO_SUPPORTED_CYCLIC_DERIVATIVE",
                "metastable_birth_K_MPa_sqrt_m": birth,
                "metastable_fold_K_MPa_sqrt_m": fold}); continue
        minimum = min(supported, key=lambda point: float(point["local_log_slope"]))
        output.append({**base, "status": "RESOLVED",
            "metastable_birth_K_MPa_sqrt_m": birth,
            "metastable_fold_K_MPa_sqrt_m": fold,
            "metastable_K_ratio": fold/birth,
            "minimum_local_slope": minimum["local_log_slope"],
            "minimum_slope_K_MPa_sqrt_m": minimum["Kmax_MPa_sqrt_m"],
            "minimum_slope_action_per_cycle": minimum["opening_action_per_cycle"],
            "moderate_connected_width_log10K": curve["moderate_connected_width_log10K"],
            "moderate_connected_width_log10rate": curve["moderate_connected_width_log10rate"],
            "strict_connected_width_log10K": curve["strict_connected_width_log10K"],
            "strict_connected_width_log10rate": curve["strict_connected_width_log10rate"],
            "strict_internal_reference_pass": curve["strict_internal_reference_pass"],
            "resolved_cyclic_points": len(supported)})
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate-ids", default="")
    parser.add_argument("--stiffness-factors", default="0.25,0.5,1,2,4")
    parser.add_argument("--points", type=int, default=101)
    parser.add_argument("--phases", type=int, default=32)
    args = parser.parse_args(argv)
    factors = [float(value) for value in args.stiffness_factors.split(",")]
    if any(not math.isfinite(value) or value <= 0. for value in factors):
        parser.error("stiffness factors must be finite and positive")
    with args.registry.open(newline="") as stream:
        registry = list(csv.DictReader(stream))
    wanted = {value for value in args.candidate_ids.split(",") if value}
    if wanted:
        registry = [row for row in registry if row["candidate_id"] in wanted]
        missing = wanted-{row["candidate_id"] for row in registry}
        if missing:
            parser.error(f"candidate IDs absent from registry: {sorted(missing)}")
    rows = [result for row in registry
            for result in evaluate(row, factors, args.points, args.phases)]
    write_csv(args.out/"stiffness_sensitivity.csv", rows)
    payload = {"schema": "r10.inverse-shape-stiffness-sensitivity/1",
        "materials": len(registry), "stiffness_factors": factors,
        "amplitude_points": args.points, "phases": args.phases,
        "resolved": sum(row["status"] == "RESOLVED" for row in rows),
        "strict_passes": sum(row.get("strict_internal_reference_pass") is True
                             for row in rows),
        "scope": "DIAGNOSTIC_ONLY; ACTUAL_NATIVE_ENVIRONMENT_IS_FACTOR_1"}
    (args.out/"summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
