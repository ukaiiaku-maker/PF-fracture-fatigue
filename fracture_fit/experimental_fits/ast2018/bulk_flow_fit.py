"""Per-temperature bulk Peierls barrier from Ast loading curves.

Plastic flow is separated from fracture. The beam is driven at the protocol
speed with only its elastic compliance and full-section bulk flow (native
Peierls glide and Taylor capture/release; no bulk nucleation, forest-floor
carriers). On linked rate-0.2 specimens the crack length is the measured one,
inverted from Fig. 3a stiffness through the adapter's own compliance law, so
cleavage kinetics never enter. Unlinked rate-0.5 traces keep a=a0 and are
scored only in an early window.

At each temperature the Peierls surface keeps the native EXP-floor form with
its exponents and floor rules; only its value G0(T) and stress scale sc(T) at
that temperature are fitted (T slopes zero). Consistency of the fitted
G0(T), sc(T) with the shared linear-in-T Arrhenius surface is then a test.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import brentq, least_squares

from .config import AstAssumptions
from .multi_temperature_bending_fit import base_material
from .section_plasticity import (SectionPlasticity, SectionPlasticityConfig,
                                 SectionSaturation)
from .section_plasticity_replay import selected_material

DATA = Path("runs/ast2018_shape_fit_v5/production/input/Ast2018_joint_fit_data/data")
OUTPUT = Path("runs/ast2018_bulk_flow_per_temperature_v1")
TEMPERATURES = {21: "Tp21C", 100: "Tp100C", 250: "Tp250C"}
EARLY_WINDOW_UM = 1.0
K_SCALE = 0.1  # MPa sqrt(m), declared working tolerance for nominal K traces
STEP_UM = 0.01


def _rows(name):
    with (DATA / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def crack_length_from_stiffness(specimen, u_um, S, peak_window_um=0.6):
    """Monotone a(u) from S=C0/C(a); S is renormalized by its pre-growth peak."""
    u = np.asarray(u_um, float)
    S = np.convolve(np.asarray(S, float), np.ones(5) / 5, mode="same")
    S[:2], S[-2:] = S[2], S[-3]
    peak_index = int(np.argmax(np.where(u <= peak_window_um, S, -np.inf)))
    S = np.minimum(S / S[peak_index], 1.0)
    C0 = specimen.elastic_compliance(specimen.a0_m)
    a = np.full_like(u, specimen.a0_m)
    for i in range(peak_index, u.size):
        target = C0 / S[i]
        if target <= C0 * (1 + 1e-12):
            continue
        hi = specimen.W_m - 0.5e-6
        if specimen.elastic_compliance(hi) < target:
            a[i] = hi
            continue
        a[i] = brentq(lambda x: specimen.elastic_compliance(x) - target,
                      specimen.a0_m, hi, xtol=1e-12)
    return u * 1e-6, np.maximum.accumulate(a)


def traces(temperature_C):
    loading, stiffness = _rows("K_displacement_curves.csv"), _rows("relative_stiffness.csv")
    out = []
    for figure, rate, linked in (("2a", "0.2", True), ("2b", "0.5", False)):
        curve = f"Ast2018_F{figure}_T{temperature_C}_rate{rate}"
        rows = [r for r in loading if r["curve_id"] == curve and r["branch"] == "loading"]
        u = np.array([float(r["displacement_um"]) for r in rows])
        K = np.array([float(r["K_MPa_sqrt_m"]) for r in rows])
        order = np.argsort(u)
        u, K = u[order], K[order]
        stiff = ([(float(r["displacement_um"]), float(r["relative_stiffness"]))
                  for r in stiffness if r["curve_id"] == f"F3a_T{temperature_C}_r0.2"]
                 if linked else [])
        if not linked:
            keep = u <= EARLY_WINDOW_UM
            u, K = u[keep], K[keep]
        out.append({"curve_id": curve, "rate": float(rate), "linked": linked,
                    "condition_id": f"{TEMPERATURES[temperature_C]}_rate{rate.replace('.', 'p')}",
                    "u_um": u, "K": K, "stiffness": stiff})
    return out


class Evaluator:
    def __init__(self, temperature_C):
        primary, _, base = base_material()
        self.native = primary.ev.spatial.EmergentGNDState
        self.material = selected_material(base)
        self.assumptions = AstAssumptions()
        self.cases = []
        for trace in traces(temperature_C):
            condition = primary.ev.data.condition(trace["condition_id"])
            specimen = self.assumptions.specimen(condition)
            physics = self.assumptions.physics(primary.ev.base_physics, specimen)
            if trace["linked"]:
                su, sa = crack_length_from_stiffness(
                    specimen, *zip(*trace["stiffness"]))
                a_of_u = (lambda uu, su=su, sa=sa, a0=specimen.a0_m:
                          float(np.interp(uu, su, sa, left=a0, right=sa[-1])))
            else:
                a_of_u = (lambda uu, a0=specimen.a0_m: a0)
            weights = np.gradient(trace["u_um"])
            self.cases.append({**trace, "specimen": specimen, "physics": physics,
                               "T": float(condition["temperature_K"]),
                               "speed": specimen.initial_displacement_rate(trace["rate"]),
                               "a_of_u": a_of_u,
                               "weights": weights / np.sum(weights)})
        self.T = self.cases[0]["T"]
        self.inherited = self.material.peierls.surface(self.material.emission)
        self.calls = 0

    def surface(self, G0_eV, sc_GPa):
        return replace(self.inherited, G00_eV=float(G0_eV), gT_eV_per_K=0.0,
                       sigc0_Pa=float(sc_GPa) * 1e9, sT_Pa_per_K=0.0)

    def simulate(self, case, surface):
        sp = case["specimen"]
        section = SectionPlasticity(self.material, case["physics"], sp,
                                    SectionPlasticityConfig(mode="full_section"),
                                    self.native, peierls_surface=surface)
        u_end = float(case["u_um"][-1]) * 1e-6
        n = max(int(np.ceil(u_end / (STEP_UM * 1e-6))), 2)
        u_grid = np.linspace(0, u_end, n + 1)
        dt = u_grid[1] / case["speed"]
        K = np.zeros_like(u_grid)
        upl = force = 0.0
        for i in range(1, u_grid.size):
            u = u_grid[i]
            a = case["a_of_u"](u)
            C = sp.elastic_compliance(a)

            def trial(F):
                s = section.clone()
                try:
                    du = s.advance(dt, force, F, a, case["T"], du_limit_m=u - upl)
                except SectionSaturation:
                    return s, u - upl + 1e-6
                return s, du

            F_el = max((u - upl) / C, 0.0)
            s, du = trial(F_el)
            if du > 1e-4 * (u_grid[1]):
                residual = lambda F: F * C + upl + trial(F)[1] - u
                F_el = brentq(residual, 0.0, F_el, xtol=1e-12 * max(F_el, 1e-9), rtol=1e-8)
                s, du = trial(F_el)
                if du > u - upl:
                    raise SectionSaturation("bulk root saturated")
            section, upl, force = s, upl + du, F_el
            K[i] = sp.nominal_K_MPa_sqrt_m(force)
        return u_grid * 1e6, K, section

    def residuals(self, x):
        self.calls += 1
        surface = self.surface(*x)
        out = []
        scored = [case for case in self.cases if case["linked"]]
        for case in scored:
            try:
                u, K, _ = self.simulate(case, surface)
                model = np.interp(case["u_um"], u, K)
            except (SectionSaturation, ArithmeticError, ValueError):
                model = np.zeros_like(case["K"])
            # Equal weight per curve; trapezoid-like weights within a curve.
            out.append((model - case["K"]) / K_SCALE * np.sqrt(case["weights"] / len(scored)))
        return np.concatenate(out)


def fit_temperature(temperature_C, parameters="G0", output=OUTPUT):
    """Fit G0(T) only (stress scale inherited) or both G0(T) and sc(T).

    Only the linked rate-0.2 specimen is scored. Unlinked rate-0.5 traces
    differ by specimen scatter larger than their rate effect and are reported
    as predictions.
    """
    if parameters not in ("G0", "G0_sc"):
        raise ValueError("parameters must be 'G0' or 'G0_sc'")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ev = Evaluator(temperature_C)
    full0 = np.array([ev.inherited.zero_stress_eV(ev.T),
                      ev.inherited.characteristic_stress_Pa(ev.T) / 1e9])
    expand = ((lambda x: np.array([x[0], full0[1]])) if parameters == "G0"
              else (lambda x: np.asarray(x, float)))
    x0 = full0[:1] if parameters == "G0" else full0
    lower, upper = np.array([0.2, 0.2]), np.array([4.0, 30.0])
    n = x0.size
    start = time.monotonic()
    log = []

    def tracked(x):
        full = expand(x)
        r = ev.residuals(full)
        log.append({"G0_eV": float(full[0]), "sc_GPa": float(full[1]),
                    "cost": float(.5 * np.dot(r, r))})
        return r

    result = least_squares(tracked, x0, bounds=(lower[:n], upper[:n]),
                           x_scale=[0.1, 1.0][:n], diff_step=[1e-3, 1e-2][:n],
                           max_nfev=40)
    fitted = expand(result.x)
    best = ev.surface(*fitted)
    curves = []
    for case in ev.cases:
        u, K, section = ev.simulate(case, best)
        model = np.interp(case["u_um"], u, K)
        curves.append({"curve_id": case["curve_id"], "rate": case["rate"],
                       "linked": case["linked"], "T_K": case["T"],
                       "measured_u_um": case["u_um"].tolist(),
                       "measured_K": case["K"].tolist(),
                       "model_u_um": u.tolist(), "model_K": K.tolist(),
                       "rms_K_MPa_sqrt_m": float(np.sqrt(np.sum(case["weights"] *
                                                                (model - case["K"]) ** 2))),
                       "final_bulk": section.snapshot(
                           K[-1] * 1e6 / case["specimen"].geometry_factor(case["specimen"].a0_m),
                           case["a_of_u"](u[-1] * 1e-6))})
    summary = {"temperature_C": temperature_C, "T_K": ev.T, "parameters": parameters,
               "scored": "linked rate-0.2 only; rate-0.5 is a prediction",
               "fitted": {"G0_eV": float(fitted[0]), "sigma_c_GPa": float(fitted[1])},
               "inherited_at_T": {"G0_eV": float(full0[0]), "sigma_c_GPa": float(full0[1])},
               "cost": float(result.cost), "status": int(result.status),
               "message": result.message, "forward_calls": ev.calls,
               "wall_seconds": time.monotonic() - start,
               "surface": asdict(best),
               "conventions": SectionPlasticityConfig(mode="full_section").as_dict(),
               "K_scale_MPa_sqrt_m": K_SCALE, "early_window_um": EARLY_WINDOW_UM,
               "step_um": STEP_UM, "curves": curves, "log": log}
    (output / f"peierls_{parameters}_T{temperature_C}.json").write_text(
        json.dumps(summary, indent=1) + "\n")
    return summary


if __name__ == "__main__":
    s = fit_temperature(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "G0")
    print(json.dumps({k: s[k] for k in ("temperature_C", "fitted", "inherited_at_T", "cost",
                                        "forward_calls", "wall_seconds")}))
