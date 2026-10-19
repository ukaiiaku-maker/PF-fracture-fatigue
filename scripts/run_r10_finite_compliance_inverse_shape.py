#!/usr/bin/env python3
"""Analytical and inverse stages of the R10 finite-compliance shape search."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.cohesive_arrhenius import (
    BILINEAR, COMPACT_INDEPENDENT, SMOOTH_EXPONENTIAL,
    CohesiveThermalLaw, EV_TO_J,
)
from arrhenius_fracture.finite_compliance_inverse_shape_r10 import (
    K_from_imposed, branch_loss, cyclic_rate, metastable_window, rate_curve,
    shape_coordinates,
)
from scripts.run_r10_global_mechanism_search import canonical_hash
from scripts.run_r10_opening_alternatives import mechanics

SCHEMA = "r10.finite-compliance-inverse-shape-search/1"
SEED = 20261018
SHAPES = (SMOOTH_EXPONENTIAL, BILINEAR, COMPACT_INDEPENDENT)
TEMPERATURES = (300., 600., 900., 1200.)
K_FRACTIONS = np.geomspace(.01, .985, 25)


def _atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                    default=str)+"\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict], compressed: bool = False) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    opener = gzip.open if compressed else open
    with opener(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _log(u: float, low: float, high: float) -> float:
    return math.exp(math.log(low)+float(u)*math.log(high/low))


def _load_sources(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _source_material_class(source: dict) -> str:
    """Recover the retained EXP_floor class without relabeling cohesive trend."""
    direct = source.get("source_material_class") or source.get("material_class")
    if direct:
        return str(direct)
    payload = source.get("parent_complete_parameter_vector")
    if payload:
        try:
            return str(json.loads(payload).get("material_class") or "UNDECLARED")
        except (TypeError, json.JSONDecodeError):
            pass
    return "UNDECLARED"


def _candidate(index: int, point: np.ndarray, sources: list[dict]) -> dict:
    source = sources[min(int(point[0]*len(sources)), len(sources)-1)]
    shape = SHAPES[index % len(SHAPES)]
    gamma = _log(point[1], .5, 8.)
    peak = _log(point[2], 8e9, 80e9)
    zero_eV = _log(point[3], .25, 12.)
    area = zero_eV*EV_TO_J/gamma
    span_fraction = -.25+point[4]*1.0
    entropy = span_fraction*gamma/900.
    b = _log(point[5], 1., 24.)
    c = -.8+point[6]*12.8
    rise = _log(point[7], .0005, .05)
    fall = .6+.39*point[8]
    if fall <= rise+.05: fall=rise+.05
    drop = _log(point[9], .005, .3)
    drop_end = rise+(1.-rise)*(.001+.079*point[10])
    tail_level = _log(point[11], .005, .3)
    compact_variant=("two_scale_tail" if shape==COMPACT_INDEPENDENT and
                     (index//len(SHAPES))%2 else "rounded_plateau")
    cohesive = {"Gamma_ref_J_m2": gamma, "t_hat_ref_Pa": peak,
        "S_Gamma_J_m2_K": entropy, "T_ref_K": 300.,
        "activation_area_m2": area, "attempt_frequency_s": 1e12,
        "delta0_m": gamma/(math.e*peak),
        "zero_traction_barrier_eV_300K": zero_eV,
        "thermal_subfamily": "MINIMAL_COUPLED_COHESIVE_ENTROPY",
        "compact_b": b, "compact_c": c,
        "compact_variant": compact_variant if shape == COMPACT_INDEPENDENT else "beta_polynomial",
        "compact_rise_fraction": rise, "compact_fall_fraction": fall,
        "compact_plateau_drop": drop,"compact_drop_fraction":drop_end,
        "compact_tail_level":tail_level}
    return {"candidate_id": f"R10_INV_{index:05d}", "shape": shape,
            "cohesive": cohesive, "source": source}


def _thermal(cohesive: dict) -> CohesiveThermalLaw:
    return CohesiveThermalLaw(cohesive["Gamma_ref_J_m2"],
        cohesive["t_hat_ref_Pa"], cohesive["S_Gamma_J_m2_K"],
        cohesive.get("T_ref_K", 300.), cohesive.get("compact_b", 2.),
        cohesive.get("compact_c", 0.), cohesive.get("compact_variant", "beta_polynomial"),
        cohesive.get("compact_rise_fraction", .1),
        cohesive.get("compact_fall_fraction", .85),
        cohesive.get("compact_plateau_drop", .05),
        cohesive.get("compact_drop_fraction", .2),
        cohesive.get("compact_tail_level", .05))


def _thermal_class(Kfold: list[float]) -> str:
    values = np.asarray(Kfold, float)
    if not np.all(np.isfinite(values)):
        return "THERMAL_ACCESS_CENSORED"
    if np.max(values)/max(np.min(values), 1e-300) <= 1.25:
        return "WEAK_T_PREDICTED"
    if np.argmax(values) not in {0, len(values)-1} and np.max(values) > 1.1*max(values[0], values[-1]):
        return "PEAK_T_PREDICTED"
    if values[-1] > 1.25*values[0]:
        return "DBTT_PREDICTED"
    return "OTHER_T_PREDICTED"


def _evaluate(item: dict) -> dict:
    cohesive, shape = item["cohesive"], item["shape"]
    base = {"candidate_id": item["candidate_id"], "cohesive_shape": shape,
            "source_material_candidate_id": item["source"]["candidate_id"],
            "source_material_class": _source_material_class(item["source"]),
            "source_plasticity_sha256": item["source"].get(
                "plasticity_physics_sha256",item["source"].get("parent_complete_row_sha256"))}
    try:
        thermal = _thermal(cohesive)
        if min(thermal.gamma(T) for T in TEMPERATURES) <= .2:
            return {**base, "domain_pass": False, "rejection": "GAMMA_THERMAL_DOMAIN"}
        coords = shape_coordinates(shape, thermal, 300.)
        if coords["completion_cutoff_m"] > 1e-7:
            return {**base,"domain_pass":False,
                "rejection":"COHESIVE_SEPARATION_EXCEEDS_100NM_RESEARCH_BOUND"}
        if coords["peak_separation_m"] < 1e-13:
            return {**base,"domain_pass":False,
                "rejection":"COHESIVE_PEAK_SEPARATION_BELOW_0P1PM_BOUND"}
        folds = []
        for temperature in TEMPERATURES:
            _, kenv = mechanics(thermal, temperature)
            fold = branch_loss(shape, thermal, temperature, kenv)
            imposed = fold.get("fold_imposed_m")
            folds.append(math.nan if imposed is None else K_from_imposed(imposed, kenv))
        if not math.isfinite(folds[0]):
            return {**base, "domain_pass": False, "rejection": "NO_BONDED_BRANCH_FOLD"}
        _, kenv = mechanics(thermal, 300.)
        window = metastable_window(shape, thermal, 300., kenv)
        if window["state"] != "METASTABLE_WINDOW":
            return {**base, "domain_pass": False, "rejection": "NO_METASTABLE_WINDOW"}
        Kbirth = float(window["birth_load_Pa"])*math.sqrt(2*math.pi*1e-6)/1e6
        Kfold = float(window["fold_load_Pa"])*math.sqrt(2*math.pi*1e-6)/1e6
        Kvalues = np.geomspace(Kbirth*1.0001, Kfold*.9999, len(K_FRACTIONS))
        curve = rate_curve(Kvalues, kenv, thermal, 300., shape,
                           cohesive["activation_area_m2"],
                           cohesive["attempt_frequency_s"])
        if (shape == COMPACT_INDEPENDENT and
                cohesive.get("compact_variant") == "two_scale_tail"):
            # Geometry-first inverse step.  mu/(G/kBT)=P*gap/barrier_density is
            # independent of activation area.  Locate the geometry nearest the
            # m=4, G=30 kBT target, then set the one physical area consistently;
            # it remains bounded by the declared zero-load 0.25--12 eV range.
            eligible=[point for point in curve["rows"] if point.get("barrier_kBT") and
                      point.get("sensitivity_identity") is not None]
            if eligible:
                target=min(eligible,key=lambda point:abs(
                    float(point["sensitivity_identity"])/float(point["barrier_kBT"])-4./30.))
                scaled=float(cohesive["activation_area_m2"])*30./float(target["barrier_kBT"])
                lower=.25*EV_TO_J/float(cohesive["Gamma_ref_J_m2"])
                upper=12.*EV_TO_J/float(cohesive["Gamma_ref_J_m2"])
                cohesive["activation_area_m2"]=float(np.clip(scaled,lower,upper))
                cohesive["zero_traction_barrier_eV_300K"]=(
                    cohesive["activation_area_m2"]*cohesive["Gamma_ref_J_m2"]/EV_TO_J)
                curve=rate_curve(Kvalues,kenv,thermal,300.,shape,
                    cohesive["activation_area_m2"],cohesive["attempt_frequency_s"])
        rows = curve["rows"]
        slopes = np.asarray([np.nan if row["local_log_slope"] is None
                             else row["local_log_slope"] for row in rows])
        finite = slopes[np.isfinite(slopes)]
        hcf_slopes = np.asarray([float(row["local_log_slope"]) for row in rows
            if row["local_log_slope"] is not None and
            1e-12 <= float(row["events_per_cycle_peak_rate"]) <= 1e-4])
        source = item["source"]
        mechanism = source.get("mechanism_prediction", "UNRESOLVED")
        return {**base, "domain_pass": True, "rejection": "",
            "thermal_prediction": _thermal_class(folds),
            "cohesive_fold_thermal_prediction": _thermal_class(folds),
            "mechanism_prediction": mechanism,
            "moderate_connected_width_log10K": curve["moderate_connected_width_log10K"],
            "moderate_connected_width_log10rate": curve["moderate_connected_width_log10rate"],
            "moderate_connected_bounds_MPa_sqrt_m": json.dumps(
                curve["moderate_connected_bounds_MPa_sqrt_m"]),
            "moderate_points": curve["max_moderate_points"],
            "strict_connected_width_log10K": curve["strict_connected_width_log10K"],
            "strict_connected_width_log10rate": curve["strict_connected_width_log10rate"],
            "strict_internal_reference_pass": curve["strict_internal_reference_pass"],
            "slope_min": float(np.min(finite)) if len(finite) else math.nan,
            "slope_median": float(np.median(finite)) if len(finite) else math.nan,
            "slope_max": float(np.max(finite)) if len(finite) else math.nan,
            "hcf_slope_median": float(np.median(hcf_slopes)) if len(hcf_slopes) else math.nan,
            "hcf_slope_min": float(np.min(hcf_slopes)) if len(hcf_slopes) else math.nan,
            "hcf_points": int(len(hcf_slopes)),
            "min_events_per_cycle": min(row["events_per_cycle_peak_rate"] for row in rows),
            "max_events_per_cycle": max(row["events_per_cycle_peak_rate"] for row in rows),
            "fold_K_300K": folds[0], "fold_K_600K": folds[1],
            "fold_K_900K": folds[2], "fold_K_1200K": folds[3],
            "initial_stiffness_Pa_per_m": coords["initial_stiffness_Pa_per_m"],
            "completion_cutoff_m": coords["completion_cutoff_m"],
            "compact_b": cohesive["compact_b"], "compact_c": cohesive["compact_c"],
            "Gamma_ref_J_m2": cohesive["Gamma_ref_J_m2"],
            "t_hat_ref_Pa": cohesive["t_hat_ref_Pa"],
            "S_Gamma_J_m2_K": cohesive["S_Gamma_J_m2_K"],
            "activation_area_m2": cohesive["activation_area_m2"],
            "metastable_birth_K_300K": Kbirth,
            "metastable_load_ratio_300K": float(window["load_ratio"]),
            "zero_traction_barrier_eV_300K": cohesive["zero_traction_barrier_eV_300K"],
            "evaluated_cohesive_json":json.dumps(cohesive,sort_keys=True,separators=(",",":")),
            "curve_json": json.dumps([{key:row.get(key) for key in (
                "K_MPa_sqrt_m","elementary_rate_s-1","events_per_cycle_peak_rate",
                "barrier_kBT","root_gap_m","sensitivity_identity",
                "local_log_slope","local_log_curvature","state")} for row in rows],
                sort_keys=True, separators=(",", ":"))}
    except (ValueError, OverflowError, FloatingPointError) as error:
        return {**base, "domain_pass": False,
                "rejection": f"{type(error).__name__}:{error}"}


def _rank(rows: list[dict], count: int) -> list[dict]:
    valid = [row for row in rows if row.get("domain_pass")]
    chosen: list[dict] = []
    # Preserve shape, thermal, and inherited plasticity diversity first.
    for shape in SHAPES:
        for thermal in ("WEAK_T_PREDICTED", "PEAK_T_PREDICTED", "DBTT_PREDICTED",
                        "OTHER_T_PREDICTED"):
            pool = [r for r in valid if r["cohesive_shape"] == shape and
                    r["thermal_prediction"] == thermal]
            pool.sort(key=lambda r: (-float(r["strict_connected_width_log10rate"]),
                -float(r["moderate_connected_width_log10rate"]),
                -int(r["hcf_points"]),
                abs(float(r["hcf_slope_median"])-4.) if math.isfinite(float(r["hcf_slope_median"])) else 1e9,
                abs(math.log10(max(float(r["max_events_per_cycle"]), 1e-300))+7.),
                r["candidate_id"]))
            chosen.extend(pool[:max(count//16, 1)])
    ordered=chosen+sorted(valid, key=lambda r: (
            -float(r["strict_connected_width_log10rate"]),
            -float(r["moderate_connected_width_log10rate"]),
            -int(r["hcf_points"]),
            abs(float(r["hcf_slope_median"])-4.) if math.isfinite(float(r["hcf_slope_median"])) else 1e9,
            r["candidate_id"]))
    seen = set(); seen_plastic=set(); unique = []
    # First protect consumed EXP_floor-vector diversity; then fill remaining
    # slots with distinct complete cohesive vectors.
    for diversity_pass in (True,False):
      for row in ordered:
        if row["candidate_id"] not in seen:
            plastic=row.get("source_plasticity_sha256")
            if diversity_pass and plastic in seen_plastic:continue
            unique.append(row); seen.add(row["candidate_id"])
            if plastic is not None:seen_plastic.add(plastic)
        if len(unique) >= count:
            return unique
    return unique


def _cyclic_registry_point(task: tuple[dict, dict, str]) -> dict:
    """Choose an HCF load from the waveform integral, never peak action."""
    metric, cohesive, shape = task
    thermal = _thermal(cohesive)
    _, kenv = mechanics(thermal, 300.)
    peak_curve = json.loads(metric["curve_json"])
    points = []
    for point in peak_curve:
        cyclic = cyclic_rate(float(point["K_MPa_sqrt_m"]), kenv, thermal,
            300., shape, float(cohesive["activation_area_m2"]),
            float(cohesive["attempt_frequency_s"]), R=.1, frequency_Hz=1000.,
            phases=32)
        points.append(cyclic)
    positive = [point for point in points
                if float(point["opening_action_per_cycle"]) > 0.]
    if not positive:
        return {"cyclic_supported": False, "cyclic_curve": points}
    # An HCF action immediately above metastable birth is not a two-sided
    # cyclic response point: its lower neighbor can have no occupied escape
    # branch.  Prefer the closest target-rate point whose complete cycle is
    # supported at both derivative neighbors.  Retain an onset-only fallback
    # explicitly when the sampled curve contains no such point.
    ranked = sorted(positive, key=lambda point: abs(math.log10(max(
        float(point["opening_action_per_cycle"]), 1e-300)) + 6.))
    chosen = ranked[0]
    adjacent = []
    slope = None
    selection_support = "CYCLIC_ONSET_ONE_SIDED"
    for candidate in ranked:
        K = float(candidate["Kmax_MPa_sqrt_m"])
        trial = [cyclic_rate(K*factor, kenv, thermal, 300., shape,
                    float(cohesive["activation_area_m2"]),
                    float(cohesive["attempt_frequency_s"]), R=.1,
                    frequency_Hz=1000., phases=64)
                 for factor in (.99, 1.01)]
        actions = [float(point["opening_action_per_cycle"]) for point in trial]
        if min(actions) > 0.:
            chosen = candidate
            adjacent = trial
            slope = math.log(actions[1]/actions[0])/math.log(1.01/.99)
            selection_support = "TWO_SIDED_CYCLIC_DERIVATIVE"
            break
    return {"cyclic_supported": True, "cyclic_curve": points,
        "chosen": chosen, "total_local_slope": slope,
        "selection_support": selection_support,
        "target_action_per_cycle": 1e-6}


def analytical(args) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    sources = _load_sources(args.source_registry)
    points = qmc.Sobol(d=12, scramble=True, seed=SEED).random_base2(args.log2_bank)
    items = [_candidate(index, point, sources) for index, point in enumerate(points)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_evaluate, items, chunksize=16))
    selected = _rank(rows, args.state_count)
    item_by_id = {item["candidate_id"]: item for item in items}
    cyclic_tasks=[]
    for metric in selected:
        item=item_by_id[metric["candidate_id"]]
        cohesive=json.loads(metric.get("evaluated_cohesive_json") or
                            json.dumps(item["cohesive"]))
        cyclic_tasks.append((metric,cohesive,item["shape"]))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        cyclic_points=list(pool.map(_cyclic_registry_point,cyclic_tasks,chunksize=1))
    registry = []
    for metric,cyclic_point in zip(selected,cyclic_points):
        item = item_by_id[metric["candidate_id"]]
        source = dict(item["source"])
        cohesive = json.loads(metric.get("evaluated_cohesive_json") or
                              json.dumps(item["cohesive"]))
        analytical_curve=json.loads(metric["curve_json"])
        peak_hcf_point=min(analytical_curve,key=lambda point:abs(math.log10(max(
            float(point["events_per_cycle_peak_rate"]),1e-300))+6.))
        hcf_point=cyclic_point.get("chosen")
        if hcf_point is None:
            hcf_K=float(peak_hcf_point["K_MPa_sqrt_m"])
            hcf_action=0.;hcf_slope=None;accessible_phases=0
            selection_basis="PEAK_FALLBACK_NO_CYCLIC_ACCESS"
        else:
            hcf_K=float(hcf_point["Kmax_MPa_sqrt_m"])
            hcf_action=float(hcf_point["opening_action_per_cycle"])
            hcf_slope=cyclic_point.get("total_local_slope")
            accessible_phases=int(hcf_point["accessible_phase_count"])
            accessible_fraction=float(hcf_point.get("accessible_phase_fraction") or 0.)
            selection_basis=("FROZEN_MOVING_BOUNDARY_GAUSS_32_ORDER__"+
                str(cyclic_point.get("selection_support","UNDECLARED_SUPPORT")))
        if hcf_point is None:
            accessible_fraction=0.
        cid = metric["candidate_id"]+"__"+({SMOOTH_EXPONENTIAL:"S",
            BILINEAR:"B", COMPACT_INDEPENDENT:"C"}[item["shape"]])+"M"
        row = dict(source)
        row.update({"candidate_id": cid, "source_material_candidate_id":
            source["candidate_id"], "selected_shape": item["shape"],
            "selected_mode": "cohesive_mixed", "search_design":
            "FINITE_COMPLIANCE_INVERSE_SHAPE", "cohesive_parameter_vector":
            json.dumps(cohesive, sort_keys=True, separators=(",", ":")),
            "predicted_fold_K_300K": metric["fold_K_300K"],
            "predicted_Kaccess_300K": metric["fold_K_300K"],
            "predicted_hcf_K_300K": hcf_K,
            "predicted_hcf_factor_300K": (hcf_K/
                float(metric["fold_K_300K"])),
            "predicted_hcf_cyclic_action_per_cycle":hcf_action,
            "predicted_hcf_peak_action_per_cycle":peak_hcf_point["events_per_cycle_peak_rate"],
            "predicted_hcf_local_slope":hcf_slope,
            "predicted_hcf_accessible_phases":accessible_phases,
            "predicted_hcf_accessible_phase_fraction":accessible_fraction,
            "predicted_hcf_selection_basis":selection_basis,
            "thermal_prediction": metric["thermal_prediction"],
            "cohesive_fold_thermal_prediction": metric["thermal_prediction"],
            "source_material_class": metric.get("source_material_class", "UNDECLARED"),
            "mechanism_prediction": metric["mechanism_prediction"]})
        row["cohesive_parameter_sha256"] = canonical_hash(cohesive)
        row["complete_row_sha256"] = canonical_hash({"source":
            source.get("complete_row_sha256"), "cohesive": cohesive,
            "shape": item["shape"], "mode": "cohesive_mixed",
            "family": SCHEMA})
        registry.append(row)
    selected_ids = {row["candidate_id"] for row in selected}
    for row in rows:
        row["selected_state_informed"] = row["candidate_id"] in selected_ids
    _write_csv(args.out/"analytical_bank_32768.csv.gz", rows, True)
    _write_csv(args.out/"state_informed_256_registry.csv", registry)
    counts = {}
    for row in rows:
        key = (row["cohesive_shape"], str(row.get("domain_pass")),
               row.get("thermal_prediction", row.get("rejection")))
        counts["__".join(key)] = counts.get("__".join(key), 0)+1
    payload = {"schema": SCHEMA, "stage": "ANALYTICAL_SCREEN_COMPLETE",
        "allocated_vectors": len(items), "domain_valid": sum(bool(r.get("domain_pass")) for r in rows),
        "state_informed_selected": len(registry), "shapes": list(SHAPES),
        "counts": counts, "algebraic_tail_registered": False,
        "source_registry": str(args.source_registry),
        "source_registry_sha256": hashlib.sha256(args.source_registry.read_bytes()).hexdigest()}
    _atomic(args.out/"analytical_summary.json", payload)
    print(json.dumps(payload, indent=2)); return 0


def _inverse_coordinates(vector: np.ndarray) -> tuple[float,float,float,float]:
    log_rise, drop_coordinate, log_tail, logarea = map(float, vector)
    rise=math.exp(log_rise)
    drop_fraction=rise+(1.-rise)/(1.+math.exp(-max(min(drop_coordinate,50.),-50.)))
    return rise,drop_fraction,math.exp(log_tail),math.exp(logarea)


def _inverse_bounds_pass(vector: np.ndarray, gamma_J_m2: float) -> bool:
    rise,drop_fraction,tail,area=_inverse_coordinates(vector)
    normalized_drop=(drop_fraction-rise)/(1.-rise)
    zero_eV=area*float(gamma_J_m2)/EV_TO_J
    return (.0005 <= rise <= .05 and .001 <= normalized_drop <= .08 and
            .005 <= tail <= .3 and .25 <= zero_eV <= 12.)


def _inverse_objective(vector: np.ndarray, base: dict) -> float:
    if not _inverse_bounds_pass(vector,base["cohesive"]["Gamma_ref_J_m2"]):
        return 1e6
    rise,drop_fraction,tail,area=_inverse_coordinates(vector)
    cohesive = dict(base["cohesive"])
    cohesive.update({"compact_variant":"two_scale_tail",
                     "compact_rise_fraction":rise,
                     "compact_drop_fraction":drop_fraction,
                     "compact_tail_level":tail,
                     "activation_area_m2":area})
    item = {**base, "shape": COMPACT_INDEPENDENT, "cohesive": cohesive}
    row = _evaluate(item)
    if not row.get("domain_pass"):
        return 1e6
    curve=json.loads(row["curve_json"])
    supported=[point for point in curve if float(point["events_per_cycle_peak_rate"])>0.
               and point["local_log_slope"] is not None]
    if not supported:return 1e5
    target=min(supported,key=lambda point:abs(
        math.log10(max(float(point["events_per_cycle_peak_rate"]),1e-300))+7.))
    slope = float(target["local_log_slope"])
    width = float(row["strict_connected_width_log10rate"])
    rate = max(float(target["events_per_cycle_peak_rate"]), 1e-300)
    slope_penalty = abs(slope-4.) if math.isfinite(slope) else 100.
    rate_penalty = abs(math.log10(rate)+7.)
    return slope_penalty-4.*width+rate_penalty


def _one_inverse(task: tuple[int, dict]) -> dict:
    index, base = task
    c = base["cohesive"]
    rise=float(c["compact_rise_fraction"]);drop=float(c["compact_drop_fraction"])
    drop_coordinate=math.log((drop-rise)/(1.-drop))
    start = np.array([math.log(rise),drop_coordinate,
                      math.log(c["compact_tail_level"]),
                      math.log(c["activation_area_m2"])])
    result = minimize(_inverse_objective, start, args=(base,), method="Nelder-Mead",
        options={"maxiter": 80, "xatol": 2e-3, "fatol": 2e-3})
    optimized_rise,optimized_drop,optimized_tail,optimized_area=_inverse_coordinates(result.x)
    final = dict(c); final.update({"compact_variant":"two_scale_tail",
        "compact_rise_fraction":optimized_rise,
        "compact_drop_fraction":optimized_drop,
        "compact_tail_level":optimized_tail,
        "activation_area_m2":optimized_area})
    evaluated = _evaluate({**base, "candidate_id": f"R10_INVERSE_OPT_{index:03d}",
                           "shape": COMPACT_INDEPENDENT, "cohesive": final})
    if not _inverse_bounds_pass(result.x,c["Gamma_ref_J_m2"]):
        evaluated.update({"domain_pass":False,"rejection":"INVERSE_RESEARCH_BOUNDS"})
    evaluated.update({"optimizer_success": bool(result.success),
                      "optimizer_message": str(result.message),
                      "optimizer_evaluations": int(result.nfev),
                      "objective": float(result.fun),
                      "optimized_cohesive_json": json.dumps(final, sort_keys=True,
                                                             separators=(",", ":"))})
    return evaluated


def inverse(args) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    sources = _load_sources(args.source_registry)
    points = qmc.Sobol(d=12, scramble=True, seed=SEED+1).random(args.starts)
    bases = [_candidate(index, point, sources) for index, point in enumerate(points)]
    for base in bases:
        base["shape"] = COMPACT_INDEPENDENT
        base["cohesive"]["compact_variant"]="two_scale_tail"
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_one_inverse, enumerate(bases), chunksize=1))
    rows.sort(key=lambda row: (float(row.get("objective", 1e9)), row["candidate_id"]))
    _write_csv(args.out/"inverse_starts.csv", rows)
    payload = {"schema": SCHEMA, "stage": "INVERSE_STARTS_COMPLETE",
        "starts": len(rows), "optimizer_successes": sum(bool(r["optimizer_success"]) for r in rows),
        "domain_valid": sum(bool(r.get("domain_pass")) for r in rows),
        "best_candidate": rows[0]["candidate_id"], "best_objective": rows[0]["objective"],
        "algebraic_tail_registered": False}
    _atomic(args.out/"inverse_summary.json", payload)
    print(json.dumps(payload, indent=2)); return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("analytical", "inverse"):
        child = sub.add_parser(name)
        child.add_argument("--source-registry", type=Path, required=True)
        child.add_argument("--out", type=Path, required=True)
        child.add_argument("--workers", type=int, default=8)
    sub.choices["analytical"].add_argument("--log2-bank", type=int, default=15)
    sub.choices["analytical"].add_argument("--state-count", type=int, default=256)
    sub.choices["inverse"].add_argument("--starts", type=int, default=64)
    args = parser.parse_args(argv)
    return analytical(args) if args.command == "analytical" else inverse(args)


if __name__ == "__main__":
    raise SystemExit(main())
