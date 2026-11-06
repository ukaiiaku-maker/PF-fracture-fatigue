"""Shared thermal Ast fit with the qualified nominal-bending native 1-D path.

All eight early resistance curves use one material, one representative beam,
the same published continuous-J observer, and equal weight by curve.  The
four adjustable coordinates are activation entropies and floor heat capacity;
the fitted 294.15 K material and Arrhenius clocks remain fixed.
"""
from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .config import AstAssumptions
from .mechanics import observe_like_ast, post_event_J_curve
from .objective import _quadrature_weights, physically_admissible
from .opening_surface import KB_EV_PER_K
from .shared_material_v9 import seed_evaluator
from .shared_material_v10 import NAMES, material_from_coordinates
from .single_temp_bending_fit import SOURCE_MATERIAL, material_at, _jsonable
from .spatial import predict_spatial
from .surfaces import surface_diagnostics

ROOT = Path("runs/ast2018_multi_temperature_bending_v1")
FIT_21 = Path("runs/ast2018_21c_rate0p5_bending_fit_v1")
SHAPE = Path("runs/ast2018_cantilever_elastic_2d/nominal_opening_shape.json")
CONDITIONS = (
    ("F3b_T21_r0.2", "Tp21C_rate0p2"),
    ("F3c_T21_r0.5", "Tp21C_rate0p5"),
    ("F3b_T100_r0.2", "Tp100C_rate0p2"),
    ("F3c_T100_r0.5", "Tp100C_rate0p5"),
    ("F3d_T100_r1.6", "Tp100C_rate1p6"),
    ("F3b_T250_r0.2", "Tp250C_rate0p2"),
    ("F3c_T250_r0.5", "Tp250C_rate0p5"),
    ("F3d_T500_r1.6", "Tp500C_rate1p6"),
)
T0 = 294.15
NAMES_THERMAL = ("S0_open_kB", "Sf_open_kB", "Cp_floor_kB", "S_emit_kB")
BOUNDS = np.array([[-35., -2.], [-17.5, -2.], [0., 15.], [-48., -5.]])


def base_material(primary=None, seed=None):
    if primary is None or seed is None:
        primary, seed = seed_evaluator()
    selected = json.loads(SOURCE_MATERIAL.read_text())
    original = material_from_coordinates(
        seed, [selected["coordinates"][name] for name in NAMES])
    center = (selected["coordinates"]["Gf_open_294_eV"],
              selected["coordinates"]["G0_emit_294_eV"],
              original.peierls.H0_eV)
    profile = json.loads((FIT_21 / "stress_profile/profile_summary.json").read_text())
    material = material_at(original, center, (
        profile["best_floor_G_294_eV"],
        profile["fixed_emission_G0_294_eV"],
        profile["fixed_peierls_H0_eV"]))
    opening = replace(material.cleavage,
        G0_ref_eV=material.cleavage.G0_ref_eV +
            profile["fixed_opening_G0_294_eV"] -
            selected["coordinates"]["G0_open_294_eV"],
        sigc_ref_Pa=material.cleavage.sigc_ref_Pa +
            (profile["best_opening_sigc_294_GPa"] -
             selected["coordinates"]["sigc_open_294_GPa"]) * 1e9)
    material = replace(material, cleavage=opening,
                       candidate_id="ast2018_21C_bending_stress_profile")
    saved = json.loads((FIT_21 / "stress_profile/best_material.json").read_text())
    if asdict(material) != saved:
        raise ArithmeticError("294 K native bending material reconstruction differs")
    return primary, seed, material


def thermal_material(base, coordinates):
    x = np.asarray(coordinates, float)
    if x.shape != (4,) or not np.all(np.isfinite(x)) or np.any(x < BOUNDS[:, 0]) or np.any(x > BOUNDS[:, 1]):
        raise ValueError("thermal coordinates outside declared bounds")
    c, e = base.cleavage, base.emission
    G0, Gf, _ = c.endpoints(T0)
    Gemit = e.G00_eV + e.gT_eV_per_K * (T0 - e.Tref_K)
    opening = replace(c,
        G0_ref_eV=G0 - x[0] * KB_EV_PER_K * (c.Tref_K - T0),
        Gf_ref_eV=Gf - x[1] * KB_EV_PER_K * (c.Tref_K - T0),
        S0_over_kB=x[0], Sf_over_kB=x[1], Cp_floor_over_kB=x[2])
    emission = replace(e,
        G00_eV=Gemit - x[3] * KB_EV_PER_K * (e.Tref_K - T0),
        gT_eV_per_K=-x[3] * KB_EV_PER_K)
    material = replace(base, cleavage=opening, emission=emission,
                       candidate_id="ast2018_shared_bending_thermal")
    opening.validate()
    if not physically_admissible(material):
        raise ValueError("thermal material violates physical surface bounds")
    for T in (T0, 373.15, 523.15, 773.15):
        for stress in (0., 1e9, 2e9, 4e9, 8e9, 12e9):
            d = opening.derivatives(stress, T)
            ed = surface_diagnostics(emission, stress, T, 1e11)
            if d["S_over_kB"] > 1e-9 or ed["S_over_kB"] > 1e-9:
                raise ValueError("barrier decreases with temperature at fixed stress")
            if d["H_eV"] <= 0. or ed["H_eV"] <= 0.:
                raise ValueError("nonpositive activation enthalpy")
    return material


def target(data, curve_id):
    rows = sorted((r for r in data.J_curves if r["curve_id"] == curve_id
        and r["sample_for_fit"].lower() in ("true", "1", "yes")
        and 0. < float(r["crack_extension_um"]) <= .5),
        key=lambda r: float(r["crack_extension_um"]))
    if len(rows) < 5:
        raise ValueError(f"insufficient early points: {curve_id}")
    x = np.array([float(r["crack_extension_um"]) * 1e-6 for r in rows])
    y = np.array([float(r["J_N_per_m"]) for r in rows])
    if np.any(np.diff(x) < 0.):
        raise ValueError(f"nonmonotone measured extension: {curve_id}")
    unique, inverse, counts = np.unique(
        x, return_inverse=True, return_counts=True)
    if len(unique) < 2:
        raise ValueError(f"insufficient distinct extensions: {curve_id}")
    # Retain measured vertical segments: a single-valued J(a) prediction is
    # compared to both observations, with one abscissa's weight split evenly.
    weights = _quadrature_weights(unique)[inverse] / counts[inverse]
    return x, y, weights


def score(observed, x, y, weights):
    xx, yy = post_event_J_curve(observed)
    supported = x <= xx[-1] + 1e-15
    fraction = float(np.sum(weights[supported]))
    scale = max(20., .15 * float(np.ptp(y)))
    row = {"supported_points": int(np.sum(supported)),
           "total_points": len(x), "supported_weight": fraction,
           "maximum_supported_extension_um": float(xx[-1] * 1e6),
           "support_shortfall_um": float(max(x[-1] - xx[-1], 0.) * 1e6),
           "normalization_N_per_m": scale,
           "weighted_rms_N_per_m": None, "level_N_per_m": None,
           "shape_rms_N_per_m": None, "normalized_objective": None}
    if np.any(supported):
        p = np.interp(x[supported], xx, yy)
        w = weights[supported] / fraction
        error = p - y[supported]
        level = float(np.dot(w, error))
        shape = float(np.sqrt(np.dot(w, (error - level) ** 2)))
        row.update({"weighted_rms_N_per_m": float(np.sqrt(np.dot(w, error ** 2))),
                    "level_N_per_m": level, "shape_rms_N_per_m": shape,
                    "normalized_objective": float(np.sqrt(level ** 2 + 2 * shape ** 2) / scale),
                    "predicted_supported_J_N_per_m": p.tolist(),
                    "measured_supported_J_N_per_m": y[supported].tolist(),
                    "supported_extension_um": (x[supported] * 1e6).tolist()})
    return row


def one_condition(primary, material, curve_id, condition_id, *, step=.1):
    condition = primary.ev.data.condition(condition_id)
    assumptions = AstAssumptions(target_extension_m=.51e-6,
        nominal_K_step_MPa_sqrt_m=step,
        maximum_nominal_K_MPa_sqrt_m=20.,
        loading_mode="nominal_bending_shared_temperature",
        bending_opening_shape_path=str(SHAPE))
    specimen = assumptions.specimen(condition)
    physics = assumptions.physics(primary.ev.base_physics, specimen)
    T = float(condition["temperature_K"])
    rate = float(condition["initial_elastic_Kdot_MPa_sqrt_m_per_s"])
    x, y, weights = target(primary.ev.data, curve_id)
    start = time.monotonic()
    pred = predict_spatial(primary.ev.spatial, material, physics, specimen,
                           T, rate, assumptions)
    observed = observe_like_ast(pred.history, specimen)
    metric = score(observed, x, y, weights)
    record = {"curve_id": curve_id, "condition_id": condition_id,
              "temperature_K": T, "initial_K_rate_MPa_sqrt_m_per_s": rate,
              "status": pred.status, "numerical_failure": pred.numerical_failure,
              "events": len(pred.events), "load_steps": pred.load_steps,
              "final_extension_um": (pred.history[-1].a_m - specimen.a0_m) * 1e6,
              "wall_seconds": time.monotonic() - start, **metric}
    history = {"record": record,
               "observations": [asdict(o) for o in observed],
               "events": pred.events}
    return record, history


def baseline(output: Path = ROOT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    primary, _, base = base_material()
    x0 = np.array((base.cleavage.S0_over_kB,
                   base.cleavage.Sf_over_kB,
                   base.cleavage.Cp_floor_over_kB,
                   -base.emission.gT_eV_per_K / KB_EV_PER_K))
    material = thermal_material(base, x0)
    cfg = {"schema": "ast2018_multi_temperature_bending_v1",
           "source": str(FIT_21 / "stress_profile/best_material.json"),
           "source_sha256": hashlib.sha256((FIT_21 / "stress_profile/best_material.json").read_bytes()).hexdigest(),
           "elastic_shape": str(SHAPE),
           "elastic_shape_sha256": hashlib.sha256(SHAPE.read_bytes()).hexdigest(),
           "native_commit": primary.ev.receipt["commit"],
           "conditions": [list(item) for item in CONDITIONS],
           "temperature_coordinates": list(NAMES_THERMAL),
           "bounds": BOUNDS.tolist(), "initial_coordinates": x0.tolist(),
           "fit_window_um": [0., .5], "step_MPa_sqrt_m": .1,
           "target_extension_um": .51, "computational_max_nominal_K_MPa_sqrt_m": 20.,
           "observation": "published evolving continuous J; F3c loading is unlinked",
           "selection": "equal curve weight; maximize supported fraction, then minimize normalized level and shape"}
    path = output / "configuration.json"
    if path.is_file() and json.loads(path.read_text()) != cfg:
        raise RuntimeError("existing thermal run configuration differs")
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    records = []
    for curve_id, condition_id in CONDITIONS:
        case_file = output / f"baseline_{curve_id}.json"
        if case_file.is_file():
            history = json.loads(case_file.read_text())
            record = history["record"]
            if record["curve_id"] != curve_id or record["condition_id"] != condition_id:
                raise RuntimeError("saved baseline case identity differs")
        else:
            record, history = one_condition(primary, material, curve_id, condition_id)
            case_file.write_text(json.dumps(history, indent=2,
                                            default=_jsonable) + "\n")
        records.append(record)
        (output / "baseline_summary.json").write_text(
            json.dumps({"coordinates": x0.tolist(), "conditions": records},
                       indent=2, default=_jsonable) + "\n")
        print(curve_id, record["status"], record["supported_points"],
              "/", record["total_points"], record["weighted_rms_N_per_m"],
              round(record["wall_seconds"], 1), flush=True)
    return records


_WORKER_PRIMARY = None
_WORKER_BASE = None


def _initialize_worker():
    global _WORKER_PRIMARY, _WORKER_BASE
    _WORKER_PRIMARY, _, _WORKER_BASE = base_material()


def _candidate_case(job):
    coordinates, curve_id, condition_id = job
    started = time.monotonic()
    try:
        material = thermal_material(_WORKER_BASE, coordinates)
        record, _ = one_condition(
            _WORKER_PRIMARY, material, curve_id, condition_id)
        return record
    except Exception as exc:
        return {"curve_id": curve_id, "condition_id": condition_id,
                "status": "evaluation_error", "error": f"{type(exc).__name__}: {exc}",
                "supported_points": 0, "total_points": 0,
                "supported_weight": 0., "normalized_objective": None,
                "wall_seconds": time.monotonic() - started}


def _candidate_summary(records):
    if len(records) != len(CONDITIONS):
        raise ValueError("candidate has missing condition records")
    unsupported = float(np.mean([1. - r["supported_weight"] for r in records]))
    costs = [r["normalized_objective"] for r in records
             if r["normalized_objective"] is not None]
    cost = float(np.mean(np.square(costs))) if costs else None
    numerical = sum(r["status"].startswith(("numerical", "evaluation_error"))
                    for r in records)
    full_support = (numerical == 0 and unsupported < 1e-9)
    return {"full_support": full_support,
            "mean_uncovered_curve_weight": unsupported,
            "mean_normalized_squared_error_on_supported_portions": cost,
            "numerical_errors": numerical,
            "physical_failures": sum(r["status"] == "physical_ligament_failure"
                                     for r in records)}


def _preference(row):
    s = row["summary"]
    return (s["numerical_errors"], s["mean_uncovered_curve_weight"],
            float("inf") if s["mean_normalized_squared_error_on_supported_portions"] is None
            else s["mean_normalized_squared_error_on_supported_portions"])


def search(output: Path = ROOT):
    output = Path(output)
    baseline_path = output / "baseline_summary.json"
    if not baseline_path.is_file():
        raise FileNotFoundError("run the eight-condition baseline first")
    baseline_data = json.loads(baseline_path.read_text())
    x0 = np.asarray(baseline_data["coordinates"], float)
    proposals = [("baseline", x0)]
    for name, changes in (
            ("floor_entropy_m16p5_Cp0", {1: -16.5, 2: 0.}),
            ("floor_entropy_m17p2_Cp0", {1: -17.2, 2: 0.}),
            ("floor_entropy_m16p5_Cp4", {1: -16.5, 2: 4.}),
            ("floor_entropy_m15p5_Cp0", {1: -15.5, 2: 0.}),
            ("opening_entropy_m25_floor_m16p5_Cp0", {0: -25., 1: -16.5, 2: 0.}),
            ("emission_entropy_m35_floor_m16p5_Cp0", {1: -16.5, 2: 0., 3: -35.})):
        candidate = x0.copy()
        for index, value in changes.items():
            candidate[index] = value
        proposals.append((name, candidate))
    primary, _, base = base_material()
    for _, x in proposals:
        thermal_material(base, x)
    config = {"schema": "ast2018_shared_temperature_bending_search_v1",
              "baseline": str(baseline_path),
              "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
              "coordinates": list(NAMES_THERMAL), "bounds": BOUNDS.tolist(),
              "proposals": [{"name": name, "vector": x.tolist()}
                            for name, x in proposals],
              "workers": 4, "full_native_1d_cases_per_candidate": len(CONDITIONS),
              "selection": "numerical errors, unsupported curve fraction, then equal-curve normalized squared level+shape error",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "material_anchor_sha256": hashlib.sha256((FIT_21 / "stress_profile/best_material.json").read_bytes()).hexdigest()}
    cfg_path = output / "search_configuration.json"
    if cfg_path.is_file() and json.loads(cfg_path.read_text()) != config:
        raise RuntimeError("existing thermal search configuration differs")
    cfg_path.write_text(json.dumps(config, indent=2) + "\n")
    log = output / "search_evaluations.jsonl"
    existing = [json.loads(line) for line in log.read_text().splitlines()] if log.is_file() else []
    done = {row["name"] for row in existing}
    with ProcessPoolExecutor(max_workers=4, initializer=_initialize_worker) as pool:
        for name, x in proposals:
            if name in done:
                continue
            started = time.monotonic()
            if name == "baseline":
                records = baseline_data["conditions"]
            else:
                futures = {pool.submit(_candidate_case,
                    (x.tolist(), curve_id, cid)): curve_id
                    for curve_id, cid in CONDITIONS}
                records = [future.result() for future in as_completed(futures)]
                records.sort(key=lambda r: [c[0] for c in CONDITIONS].index(r["curve_id"]))
            row = {"name": name, "coordinates": dict(zip(NAMES_THERMAL, x.tolist())),
                   "conditions": records, "summary": _candidate_summary(records),
                   "stage_wall_seconds": time.monotonic() - started}
            with log.open("a") as stream:
                stream.write(json.dumps(row, default=_jsonable, allow_nan=False) + "\n")
                stream.flush()
            existing.append(row)
            done.add(name)
            best = min(existing, key=_preference)
            (output / "search_summary.json").write_text(
                json.dumps({"best_name": best["name"],
                            "best_summary": best["summary"],
                            "candidates": [{"name": r["name"],
                                            "summary": r["summary"]}
                                           for r in existing]}, indent=2) + "\n")
            print(name, row["summary"], round(row["stage_wall_seconds"], 1),
                  "best", best["name"], flush=True)
    return min(existing, key=_preference)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("baseline", "search"),
                        default="baseline")
    args = parser.parse_args()
    baseline() if args.phase == "baseline" else search()
