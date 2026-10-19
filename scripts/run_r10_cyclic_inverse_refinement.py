#!/usr/bin/env python3
"""Refine admissible compact optima against the complete cyclic consumer."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.cohesive_arrhenius import COMPACT_INDEPENDENT
from arrhenius_fracture.finite_compliance_inverse_shape_r10 import cyclic_rate_curve
from scripts.run_r10_finite_compliance_inverse_shape import (
    _evaluate,
    _inverse_bounds_pass,
    _inverse_coordinates,
    _source_material_class,
    _thermal,
)
from scripts.run_r10_opening_alternatives import mechanics


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
        writer.writeheader(); writer.writerows(rows)


def vector_from(cohesive: dict) -> np.ndarray:
    rise = float(cohesive["compact_rise_fraction"])
    drop = float(cohesive["compact_drop_fraction"])
    coordinate = math.log((drop-rise)/(1.-drop))
    return np.asarray([math.log(rise), coordinate,
                       math.log(float(cohesive["compact_tail_level"])),
                       math.log(float(cohesive["activation_area_m2"]))])


def cyclic_evaluate(vector: np.ndarray, base: dict) -> dict:
    cohesive0 = base["cohesive"]
    gamma = float(cohesive0["Gamma_ref_J_m2"])
    if not _inverse_bounds_pass(vector, gamma):
        return {"domain_pass": False, "rejection": "INVERSE_RESEARCH_BOUNDS"}
    rise, drop, tail, area = _inverse_coordinates(vector)
    cohesive = dict(cohesive0)
    cohesive.update({"compact_variant":"two_scale_tail",
        "compact_rise_fraction":rise, "compact_drop_fraction":drop,
        "compact_tail_level":tail, "activation_area_m2":area})
    metric = _evaluate({"candidate_id":base["candidate_id"],
        "shape":COMPACT_INDEPENDENT, "cohesive":cohesive,
        "source":base["source"]})
    if not metric.get("domain_pass"):
        return metric
    effective = json.loads(metric["evaluated_cohesive_json"])
    peak_curve = json.loads(metric["curve_json"])
    # Thirteen log-spaced operating points are sufficient for optimization;
    # promoted optima receive a denser held-out qualification separately.
    indices = np.unique(np.linspace(0, len(peak_curve)-1, 13).round().astype(int))
    K = [float(peak_curve[index]["K_MPa_sqrt_m"]) for index in indices]
    thermal = _thermal(effective); _, kenv = mechanics(thermal, 300.)
    cyclic = cyclic_rate_curve(K, kenv, thermal, 300., COMPACT_INDEPENDENT,
        float(effective["activation_area_m2"]),
        float(effective["attempt_frequency_s"]), R=.1,
        frequency_Hz=1000., phases=16)
    supported = [row for row in cyclic["rows"]
                 if row["opening_action_per_cycle"] > 0. and
                 row["local_log_slope"] is not None]
    if not supported:
        return {**metric, "domain_pass":False,
                "rejection":"NO_TWO_SIDED_CYCLIC_POINTS"}
    target_action = float(base.get("target_action_per_cycle", 1e-6))
    target_log_action = math.log10(target_action)
    target = min(supported, key=lambda row: abs(math.log10(max(
        float(row["opening_action_per_cycle"]), 1e-300)) - target_log_action))
    target_K = float(target["Kmax_MPa_sqrt_m"])
    # The sparse curve locates an operating region economically, but its broad
    # central secants are not local slopes.  Qualify the objective with a
    # dedicated two-sided perturbation of the complete moving-boundary cycle
    # integral, matching the promotion convention.
    local = cyclic_rate_curve([.99*target_K, target_K, 1.01*target_K],
        kenv, thermal, 300., COMPACT_INDEPENDENT,
        float(effective["activation_area_m2"]),
        float(effective["attempt_frequency_s"]), R=.1,
        frequency_Hz=1000., phases=16)
    local_target = local["rows"][1]
    if (local_target["opening_action_per_cycle"] <= 0. or
            local_target["local_log_slope"] is None):
        return {**metric, "domain_pass":False,
                "rejection":"NO_TWO_SIDED_LOCAL_CYCLIC_DERIVATIVE"}
    return {**metric,
        "evaluated_cohesive_json":json.dumps(effective,sort_keys=True,
                                               separators=(",",":")),
        "cyclic_curve_json":json.dumps(cyclic["rows"],sort_keys=True,
                                        separators=(",",":")),
        "cyclic_local_derivative_curve_json":json.dumps(local["rows"],sort_keys=True,
                                                           separators=(",",":")),
        "cyclic_target_K_MPa_sqrt_m":local_target["Kmax_MPa_sqrt_m"],
        "cyclic_target_action_per_cycle":local_target["opening_action_per_cycle"],
        "cyclic_requested_target_action_per_cycle":target_action,
        "cyclic_target_local_slope":local_target["local_log_slope"],
        "cyclic_local_slope_perturbation_fraction":.01,
        "cyclic_moderate_connected_width_log10K":
            cyclic["moderate_connected_width_log10K"],
        "cyclic_moderate_connected_width_log10rate":
            cyclic["moderate_connected_width_log10rate"],
        "cyclic_strict_connected_width_log10K":
            cyclic["strict_connected_width_log10K"],
        "cyclic_strict_connected_width_log10rate":
            cyclic["strict_connected_width_log10rate"],
        "cyclic_strict_internal_reference_pass":
            cyclic["strict_internal_reference_pass"]}


def objective(vector: np.ndarray, base: dict) -> float:
    row = cyclic_evaluate(vector, base)
    if not row.get("domain_pass"):
        return 1e6
    slope = float(row["cyclic_target_local_slope"])
    action = max(float(row["cyclic_target_action_per_cycle"]), 1e-300)
    target_action = float(base.get("target_action_per_cycle", 1e-6))
    strict = float(row["cyclic_strict_connected_width_log10rate"])
    moderate = float(row["cyclic_moderate_connected_width_log10rate"])
    rate_weight=float(base.get("rate_weight",1.0))
    return (abs(slope-4.) +
            rate_weight*abs(math.log10(action)-math.log10(target_action)) -
            4.*strict - moderate)


def one(task: tuple[int,dict]) -> dict:
    index, base = task
    start = vector_from(base["cohesive"])
    result = minimize(objective, start, args=(base,), method="Nelder-Mead",
        options={"maxiter":48, "xatol":2e-3, "fatol":2e-3})
    evaluated = cyclic_evaluate(result.x, {**base,
        "candidate_id":f"R10_CYCLIC_OPT_{index:03d}"})
    evaluated = {
        "candidate_id": f"R10_CYCLIC_OPT_{index:03d}",
        "source_material_candidate_id": base["source"]["candidate_id"],
        "source_material_class": _source_material_class(base["source"]),
        "source_plasticity_sha256": base["source"].get(
            "plasticity_physics_sha256",
            base["source"].get("parent_complete_row_sha256")),
        **evaluated,
    }
    evaluated.update({"optimizer_success":bool(result.success),
        "optimizer_message":str(result.message),
        "optimizer_evaluations":int(result.nfev),
        "objective":float(result.fun),
        "optimized_cohesive_json":evaluated.get("evaluated_cohesive_json", "")})
    return evaluated


def main(argv=None) -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--inverse-table",type=Path,required=True)
    parser.add_argument("--source-registry",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--starts",type=int,default=16)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--rate-weight",type=float,default=1.0)
    parser.add_argument("--target-action-per-cycle",type=float,default=1e-6)
    args=parser.parse_args(argv)
    if args.target_action_per_cycle <= 0.:
        parser.error("--target-action-per-cycle must be positive")
    args.out.mkdir(parents=True,exist_ok=True)
    inverse=[row for row in read_csv(args.inverse_table)
             if row.get("domain_pass")=="True"]
    inverse.sort(key=lambda row:(float(row.get("objective") or math.inf),
                                 row["candidate_id"]))
    sources={row["candidate_id"]:row for row in read_csv(args.source_registry)}
    bases=[]
    for index,row in enumerate(inverse[:args.starts]):
        source_id=row["source_material_candidate_id"]
        bases.append({"candidate_id":f"R10_CYCLIC_OPT_{index:03d}",
            "cohesive":json.loads(row["optimized_cohesive_json"]),
            "source":sources[source_id],"rate_weight":args.rate_weight,
            "target_action_per_cycle":args.target_action_per_cycle})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(one,enumerate(bases),chunksize=1))
    rows.sort(key=lambda row:(float(row.get("objective",math.inf)),
                              row.get("candidate_id","")))
    write_csv(args.out/"cyclic_inverse_starts.csv",rows)
    payload={"schema":"r10.finite-compliance-cyclic-inverse-refinement/1",
        "stage":"CYCLIC_INVERSE_STARTS_COMPLETE","starts":len(rows),
        "domain_valid":sum(bool(row.get("domain_pass")) for row in rows),
        "optimizer_numerical_convergences":sum(
            bool(row.get("optimizer_success")) for row in rows),
        "optimizer_successes":sum(bool(row.get("optimizer_success")) and
                                  bool(row.get("domain_pass")) for row in rows),
        "strict_passes":sum(row.get("cyclic_strict_internal_reference_pass") is True
                            for row in rows),
        "best_candidate":rows[0].get("candidate_id") if rows else None,
        "best_objective":rows[0].get("objective") if rows else None,
        "hcf_log_rate_penalty_weight":args.rate_weight,
        "target_action_per_cycle":args.target_action_per_cycle,
        "local_slope_perturbation_fraction":.01,
        "algebraic_tail_registered":False}
    atomic(args.out/"cyclic_inverse_summary.json",payload)
    print(json.dumps(payload,indent=2));return 0


if __name__=="__main__":
    raise SystemExit(main())
