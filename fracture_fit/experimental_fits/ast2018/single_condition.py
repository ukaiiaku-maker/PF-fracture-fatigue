"""Bounded, isothermal Ast 21 C / rate 0.5 resistance-curve capability test.

The loading endpoint is a declared cross-figure horizon, never an optimized
parameter. The measured J curve alone determines the search score.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import csv
import json
import time

import numpy as np

from .config import AstAssumptions
from .data import AstData
from .mechanics import observe_like_ast, post_event_J_curve, continuous_onset_K
from .native import historical_start, load_native
from .objective import WorkingObjective, _quadrature_weights, _select_quantiles, physically_admissible
from .opening_surface import IndependentFloorSurface, KB_EV_PER_K
from .parameterization import Parameterization
from .reduced import predict_reduced
from .spatial import predict_spatial

TSTAR = 294.15
CONDITION = "Tp21C_rate0p5"
HORIZON_M = 1.165284e-6
BOUNDS = ((.3, 7.), (1.5e9, 15e9), (.2, 5.), (1.5e9, 15e9),
          (.3, 5.), (.2, 5.))
SEED_PATHS = {
    "native_rise": ("runs/ast2018_shape_fit_v5/restart_all8/full_final_256/full_candidates.jsonl",
                    "4bd0da3a670d1b95ac6516f09d79da3afd2fa41dced99f82d8541d29bd19c783"),
    "floor_flat": ("runs/ast2018_shape_fit_v5/independent_floor_focused_restart/restart_candidates.jsonl",
                   "26ce47e9121a89010ce8117bcb2c14305b3c6c302bd52f6bf75b3fcb44264575"),
    "floor_transfer": ("runs/ast2018_shape_fit_v5/independent_floor_focused_restart/restart_candidates.jsonl",
                       "acbe4054150fb123f4cb2283c4238b73bcef9af129c038bb44916b50ac3bb023"),
}


def seed_row(name):
    path, key = SEED_PATHS[name]
    for line in Path(path).open():
        row = json.loads(line)
        if row.get("key") == key:
            return row
    raise KeyError(key)


def seed_material(base, name):
    row = seed_row(name)
    family = Parameterization(opening_kind="independent_floor" if name.startswith("floor")
                              else "native_fractional_floor",
                              opening_stress_slope="inherited", extended_mobility=True)
    return family.material(base, row["x"])


def thermal_completion(material):
    c, e = material.cleavage, material.emission
    return {"opening_gT": (-c.S0_over_kB * KB_EV_PER_K if isinstance(c, IndependentFloorSurface)
                            else c.gT_eV_per_K),
            "opening_sT": (c.d_sigc_dT_Pa_per_K if isinstance(c, IndependentFloorSurface)
                            else c.sT_Pa_per_K),
            "floor_Sf": c.Sf_over_kB if isinstance(c, IndependentFloorSurface) else None,
            "emission_gT": e.gT_eV_per_K, "emission_sT": e.sT_Pa_per_K,
            "peierls_entropy": material.peierls.activation_entropy_kB,
            "taylor_entropy": material.taylor.activation_entropy_kB}


def isothermal_coordinates(material):
    c, e = material.cleavage, material.emission
    if isinstance(c, IndependentFloorSurface):
        g, gf, sc = c.endpoints(TSTAR)
    else:
        dt = TSTAR - c.Tref_K
        g, sc = c.G00_eV + c.gT_eV_per_K * dt, c.sigc0_Pa + c.sT_Pa_per_K * dt
        gf = None
    dt = TSTAR - e.Tref_K
    pe = material.peierls.surface(e)
    ta = material.taylor.surface(e)
    x = [g, sc, e.G00_eV + e.gT_eV_per_K * dt,
         e.sigc0_Pa + e.sT_Pa_per_K * dt,
         pe.G00_eV + pe.gT_eV_per_K * dt,
         ta.G00_eV + ta.gT_eV_per_K * dt]
    return np.asarray(x + ([gf] if gf is not None else []), dtype=float)


def from_isothermal(base, x, completion):
    """Change reference values only, preserving the fixed branch slopes."""
    x = np.asarray(x, dtype=float)
    if len(x) not in (6, 7) or not np.all(np.isfinite(x)):
        raise ValueError("six or seven finite coordinates required")
    dt = TSTAR - base.emission.Tref_K
    if len(x) == 7:
        c = base.cleavage
        if not isinstance(c, IndependentFloorSurface):
            raise ValueError("floor branch required")
        if not 0 <= x[6] <= x[0]:
            raise ValueError("invalid isothermal floor")
        opening = replace(c, G0_ref_eV=x[0]-completion["opening_gT"]*dt,
                          S0_over_kB=-completion["opening_gT"]/KB_EV_PER_K,
                          Gf_ref_eV=x[6]+c.Sf_over_kB*KB_EV_PER_K*dt,
                          sigc_ref_Pa=x[1]-completion["opening_sT"]*dt,
                          d_sigc_dT_Pa_per_K=completion["opening_sT"])
    else:
        c = base.cleavage
        opening = replace(c, G00_eV=x[0]-completion["opening_gT"]*dt,
                          gT_eV_per_K=completion["opening_gT"],
                          sigc0_Pa=x[1]-completion["opening_sT"]*dt,
                          sT_Pa_per_K=completion["opening_sT"])
    emission = replace(base.emission,
                       G00_eV=x[2]-completion["emission_gT"]*dt,
                       gT_eV_per_K=completion["emission_gT"],
                       sigc0_Pa=x[3]-completion["emission_sT"]*dt,
                       sT_Pa_per_K=completion["emission_sT"])
    p = replace(base.peierls,
                H0_eV=x[4]-(-completion["peierls_entropy"]*KB_EV_PER_K)*dt,
                activation_entropy_kB=completion["peierls_entropy"])
    t = replace(base.taylor,
                H0_eV=x[5]-(-completion["taylor_entropy"]*KB_EV_PER_K)*dt,
                activation_entropy_kB=completion["taylor_entropy"])
    return replace(base, cleavage=opening, emission=emission, peierls=p, taylor=t,
                   candidate_id="ast2018_single_21C")


def observations(data, n=25):
    raw = sorted((r for r in data.J_curves if r["curve_id"] == "F3c_T21_r0.5"
                  and r["sample_for_fit"].lower() in ("true", "1", "yes")
                  and float(r["crack_extension_nm"]) > 0),
                 key=lambda r: float(r["crack_extension_nm"]))
    selected = _select_quantiles(raw, n, "crack_extension_nm")
    a = np.array([float(r["crack_extension_nm"]) * 1e-9 for r in selected])
    J = np.array([float(r["J_N_per_m"]) for r in selected])
    q = _quadrature_weights(a)
    # The v5 working scale is a tolerance, not an experimental SD.
    scale = max(20., .05 * np.ptp(J), np.median([
        float(r.get("J_digitization_pm_N_per_m", 0) or 0) for r in selected]))
    return selected, a, J, q, float(scale)


def curve_metrics(model_a, model_J, target_a, target_J, q, scale):
    if len(model_a) == 0:
        return {"support_nm": 0., "shortfall_nm": float(target_a[-1] * 1e9),
                "loss": 1e6, "raw_rms": None, "level": None, "rise_error": None}
    pred = np.interp(target_a, model_a, model_J)
    err = pred - target_J
    level = float(np.dot(q, err))
    centered = err - level
    # A single transformed error vector; no duplicate onset/slope observations.
    transformed = (np.sqrt(2.) * centered + level) / scale
    robust = 2 * (np.hypot(1, transformed) - 1)
    shortfall = max(float(target_a[-1] - model_a[-1]), 0.)
    rise = (pred[-1] - pred[0]) - (target_J[-1] - target_J[0])
    edges = np.linspace(target_a[0], target_a[-1], 5)
    mp, op = np.interp(edges, model_a, model_J), np.interp(edges, target_a, target_J)
    windows = np.diff(mp) - np.diff(op)
    return {"support_nm": float(model_a[-1] * 1e9),
            "shortfall_nm": shortfall * 1e9,
            "loss": float(np.dot(q, robust) + (shortfall / 50e-9) ** 2),
            "raw_rms": float(np.sqrt(np.dot(q, err ** 2))),
            "level": level, "shape_rms": float(np.sqrt(np.dot(q, centered ** 2))),
            "rise_error": float(rise), "window_errors": windows.tolist(),
            "predicted_rise": float(pred[-1] - pred[0])}


def rank(row):
    if row.get("error") or row.get("metrics") is None:
        return (float("inf"), float("inf"))
    m = row["metrics"]
    return (max(m["shortfall_nm"] - 50., 0.), m["loss"])


def diverse(rows, n):
    """Interleave quality, level, evolution and coverage without list starvation."""
    valid = [r for r in rows if r.get("metrics") and not r.get("error")]
    orders = [sorted(valid, key=rank),
              sorted(valid, key=lambda r: (abs(r["metrics"]["level"]) if r["metrics"]["level"] is not None else 1e9, rank(r))),
              sorted(valid, key=lambda r: (abs(r["metrics"]["rise_error"]) if r["metrics"]["rise_error"] is not None else 1e9, rank(r))),
              sorted(valid, key=lambda r: (r["metrics"]["shortfall_nm"], rank(r)))]
    result, seen = [], set()
    for i in range(len(valid)):
        for order in orders:
            if len(result) >= n:
                return result
            r = order[i]
            if r["key"] not in seen:
                seen.add(r["key"]); result.append(r)
    return result


class Evaluator:
    def __init__(self, source, data_dir, branch, step=.05,
                 protocol="borrowed_fig2b_horizon_v1"):
        mods, receipt = load_native(Path(source))
        self.zero, self.spatial = mods[:2]
        self.base, self.base_physics, _ = historical_start(Path(source), mods[2], mods[3].CommonPhysics)
        self.data = AstData.load(Path(data_dir))
        self.condition = self.data.condition(CONDITION)
        self.assumptions = replace(AstAssumptions(), nominal_K_step_MPa_sqrt_m=step)
        self.specimen = self.assumptions.specimen(self.condition)
        self.physics = self.assumptions.physics(self.base_physics, self.specimen)
        self.branch = branch
        self.seed_name = "native_rise" if branch == "native" else "floor_flat"
        self.material = seed_material(self.base, self.seed_name)
        self.completion = thermal_completion(self.material)
        self.selected, self.a, self.J, self.q, self.scale = observations(self.data)
        self.receipt = receipt
        self.protocol = protocol
        if protocol not in ("borrowed_fig2b_horizon_v1", "measured_extension_stop_v2",
                            "conservative_work_replay_v3", "line_content_reconciliation_v4"):
            raise ValueError("unknown observation protocol")

    def key_for(self, x, level="reduced"):
        parameter_key = self.branch + ":" + ",".join(f"{v:.12g}" for v in x)
        return (self.protocol + ":" + self.receipt["commit"] + ":" + level + ":" +
                f"step={self.assumptions.nominal_K_step_MPa_sqrt_m:.12g}:" + parameter_key)

    def run(self, x, level="reduced", save=None):
        t0 = time.monotonic()
        x = np.asarray(x, dtype=float)
        parameter_key = self.branch + ":" + ",".join(f"{v:.12g}" for v in x)
        key = self.key_for(x, level)
        row = {"key": key, "parameter_key": parameter_key,
               "protocol": self.protocol, "source_commit": self.receipt["commit"],
               "branch": self.branch, "x": x.tolist(), "level": level}
        try:
            candidate = from_isothermal(self.material, x, self.completion)
            if not physically_admissible(candidate):
                raise ValueError("whole-temperature material inadmissible")
            predictor = predict_reduced if level == "reduced" else predict_spatial
            module = self.zero if level == "reduced" else self.spatial
            required_u, required_a = ((HORIZON_M, 0.) if self.protocol ==
                                      "borrowed_fig2b_horizon_v1" else (0., 1.58e-6))
            result = predictor(module, candidate, self.physics, self.specimen, TSTAR, .5,
                               self.assumptions, deterministic_threshold=True,
                               required_u_m=required_u, required_extension_m=required_a)
            observed = observe_like_ast(result.history, self.specimen)
            a, J = post_event_J_curve(observed) if observed else ([], [])
            m = curve_metrics(a, J, self.a, self.J, self.q, self.scale)
            last_obs = observed[-1]
            J_el = (last_obs.K_el_current_MPa_sqrt_m*1e6)**2/self.specimen.Eprime_Pa
            radius = [float(e["tip_radius_pre_m"]) for e in result.events
                      if "tip_radius_pre_m" in e]
            activations = [float(e["cumulative_activations"]) for e in result.events
                           if "cumulative_activations" in e]
            shielding = [float(e["shielding_pre_MPa_sqrt_m"]) for e in result.events
                         if "shielding_pre_MPa_sqrt_m" in e]
            row.update(status=result.status, numerical_failure=result.numerical_failure,
                       events=len(result.events), load_steps=result.load_steps, metrics=m,
                       onset_K=continuous_onset_K(observed, self.specimen),
                       final_u_m=result.history[-1].imposed_u_m,
                       final_t_s=result.history[-1].time_s,
                       final_extension_m=result.history[-1].a_m-self.specimen.a0_m,
                       support_complete=bool(len(a) and a[-1]>=self.a[-1]-1e-15),
                       final_J_el_N_per_m=J_el,
                       final_J_pl_N_per_m=last_obs.J_continuous_N_per_m-J_el,
                       maximum_tip_radius_pre_m=max(radius) if radius else None,
                       maximum_cumulative_activations=max(activations) if activations else None,
                       maximum_abs_K_shield_MPa_sqrt_m=max(map(abs,shielding)) if shielding else None)
            if save:
                save = Path(save); save.mkdir(parents=True, exist_ok=True)
                for filename, items in (("mechanical_history.csv", [asdict(v) for v in result.history]),
                                        ("ast_observer.csv", [asdict(v) for v in observed])):
                    if items:
                        with (save / filename).open("w", newline="") as f:
                            w = csv.DictWriter(f, fieldnames=list(items[0])); w.writeheader(); w.writerows(items)
                (save / "events.json").write_text(json.dumps(result.events, indent=2) + "\n")
                (save / "parameters.json").write_text(json.dumps({"isothermal": x.tolist(),
                    "completion": self.completion, "material": asdict(candidate)}, indent=2) + "\n")
        except Exception as exc:
            row["error"] = type(exc).__name__ + ": " + str(exc)
        row["runtime_s"] = time.monotonic() - t0
        return row
