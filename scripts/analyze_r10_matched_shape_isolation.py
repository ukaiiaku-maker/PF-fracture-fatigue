#!/usr/bin/env python3
"""Same-vector finite-compliance comparison of the three cohesive shapes."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.cohesive_arrhenius import (
    BILINEAR, COMPACT_INDEPENDENT, SMOOTH_EXPONENTIAL,
)
from arrhenius_fracture.finite_compliance_inverse_shape_r10 import (
    K_from_imposed, cyclic_rate_curve, metastable_window, shape_coordinates,
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--points", type=int, default=101)
    parser.add_argument("--phases", type=int, default=32)
    args = parser.parse_args(argv)
    with args.registry.open(newline="") as stream:
        selected = next((row for row in csv.DictReader(stream)
                         if row["candidate_id"] == args.candidate_id), None)
    if selected is None:
        parser.error("candidate absent from registry")
    cohesive = json.loads(selected["cohesive_parameter_vector"])
    thermal = _thermal(cohesive); _, stiffness = mechanics(thermal, 300.)
    rows = []
    for shape in (SMOOTH_EXPONENTIAL, BILINEAR, COMPACT_INDEPENDENT):
        coordinates = shape_coordinates(shape, thermal, 300.)
        window = metastable_window(shape, thermal, 300., stiffness)
        base = {"source_candidate_id": args.candidate_id,
            "complete_row_sha256": selected["complete_row_sha256"],
            "shape": shape, "Gamma_300K_J_m2": thermal.gamma(300.),
            "peak_traction_300K_Pa": thermal.t_hat(300.),
            "activation_area_m2": cohesive["activation_area_m2"],
            "attempt_frequency_s": cohesive["attempt_frequency_s"],
            "environment_stiffness_Pa_per_m": stiffness,
            "initial_stiffness_Pa_per_m": coordinates["initial_stiffness_Pa_per_m"],
            "peak_separation_m": coordinates["peak_separation_m"],
            "completion_cutoff_m": coordinates["completion_cutoff_m"],
            "comparison_scope": "SAME_COHESIVE_SCALE_AND_PLASTICITY_VECTOR; FROZEN_OPENING"}
        if window["state"] != "METASTABLE_WINDOW":
            rows.append({**base, "status": window["state"]}); continue
        birth = K_from_imposed(float(window["birth_load_Pa"])/stiffness, stiffness)
        fold = K_from_imposed(float(window["fold_load_Pa"])/stiffness, stiffness)
        K = np.geomspace(birth*1.0001, fold*.9999, args.points)
        curve = cyclic_rate_curve(K, stiffness, thermal, 300., shape,
            float(cohesive["activation_area_m2"]),
            float(cohesive["attempt_frequency_s"]), R=.1,
            frequency_Hz=1000., phases=args.phases)
        supported = [point for point in curve["rows"]
            if point["opening_action_per_cycle"] > 0. and
               point["local_log_slope"] is not None]
        if not supported:
            rows.append({**base, "status": "NO_SUPPORTED_CYCLIC_DERIVATIVE",
                "metastable_birth_K_MPa_sqrt_m": birth,
                "metastable_fold_K_MPa_sqrt_m": fold}); continue
        minimum = min(supported, key=lambda point: float(point["local_log_slope"]))
        hcf = [point for point in supported
               if 1e-12 <= float(point["opening_action_per_cycle"]) <= 1e-4]
        rows.append({**base, "status": "RESOLVED",
            "metastable_birth_K_MPa_sqrt_m": birth,
            "metastable_fold_K_MPa_sqrt_m": fold,
            "metastable_K_ratio": fold/birth,
            "minimum_local_slope": minimum["local_log_slope"],
            "minimum_slope_action_per_cycle": minimum["opening_action_per_cycle"],
            "hcf_accessible_points": len(hcf),
            "hcf_minimum_local_slope": (min(float(point["local_log_slope"])
                for point in hcf) if hcf else None),
            "hcf_maximum_local_slope": (max(float(point["local_log_slope"])
                for point in hcf) if hcf else None),
            "moderate_connected_width_log10rate": curve["moderate_connected_width_log10rate"],
            "strict_connected_width_log10rate": curve["strict_connected_width_log10rate"],
            "strict_internal_reference_pass": curve["strict_internal_reference_pass"],
            "resolved_cyclic_points": len(supported)})
    write_csv(args.out/"matched_shape_isolation.csv", rows)
    payload = {"schema": "r10.matched-shape-isolation/1",
        "source_candidate_id": args.candidate_id, "materials": 1,
        "shape_arms": len(rows), "resolved": sum(row["status"] == "RESOLVED" for row in rows),
        "strict_passes": sum(row.get("strict_internal_reference_pass") is True for row in rows),
        "same_complete_plasticity_background": True,
        "initial_stiffness_is_shape_output_not_forced_match": True,
        "points": args.points, "phases": args.phases}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
