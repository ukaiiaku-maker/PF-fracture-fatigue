"""Joint opening/emission/glide fit at each temperature separately.

At temperature T the material is isothermal: each fitted barrier has its
value at T set directly and its temperature slopes zeroed, so no entropy
constraint enters. S* is read afterwards from how the per-temperature values
change with T. Attempt frequencies, Taylor law, source/blunting/shielding
conventions and event statistics are unchanged.

Two opening-barrier families are fitted in parallel with the same emission
and glide coordinates:
  A  EXP floor   G = Gf + (G0 - Gf) exp[-a (s/sc)^n]           (native form)
  B  Model B     G = (1 - s/sc)^(3/2) [F + B exp(-s/s_B)]       (common
     three-halves envelope, Model_B_Expanded_OMML.docx; G0 = F + B, and the
     floor F slopes to zero at the cutoff sc)

Forward model: native 1-D crack-tip state (emission/opening competition,
blunting, shielding, MPZ transport) plus the full-section bulk channel with
forest-floor carriers and no bulk nucleation. Both use the same Peierls
surface, so the fit decides how much plasticity occurs away from the tip.

Glide has its own stress scale. Natively the Peierls and Taylor scales are
fixed multiples of the emission sc, so easier tip emission also softened the
whole beam and the hinge yielded before the tip could cleave: the search could
only reach fits with tip emission switched off. Here the Peierls sc is fitted
directly and Taylor keeps its native ratio to Peierls.

Score at each temperature: every Fig. 3 J curve at that temperature up to
0.98 um, and for the linked rate-0.2 specimen its Fig. 2a nominal-K and Fig.
3a stiffness traces over the displacement the model reaches, and only while
the measured crack is within the 0.98 um the elastic field table represents
(measured stiffness above the elastic stiffness of that crack). Unsupported J
points carry a declared censoring penalty (no support-first ranking).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from .config import AstAssumptions
from .mechanics import observe_like_ast, post_event_J_curve
from .multi_temperature_bending_fit import SHAPE, base_material
from .objective import physically_admissible
from .opening_surface import FiniteCutoffOpeningSurface
from .section_plasticity import SectionPlasticityConfig
from .section_plasticity_replay import selected_material
from .single_temp_bending_fit import _jsonable
from .spatial import TIP_INTERACTIONS, predict_spatial

DATA = Path("runs/ast2018_shape_fit_v5/production/input/Ast2018_joint_fit_data/data")
OUTPUT = Path("runs/ast2018_per_temperature_joint_v5")
PREVIOUS = Path("runs/ast2018_per_temperature_joint_v4")
GROUPS = {
    21: (("F3b_T21_r0.2", "Tp21C_rate0p2", "Ast2018_F2a_T21_rate0.2", "F3a_T21_r0.2"),
         ("F3c_T21_r0.5", "Tp21C_rate0p5", None, None)),
    100: (("F3b_T100_r0.2", "Tp100C_rate0p2", "Ast2018_F2a_T100_rate0.2", "F3a_T100_r0.2"),
          ("F3c_T100_r0.5", "Tp100C_rate0p5", None, None),
          ("F3d_T100_r1.6", "Tp100C_rate1p6", None, None)),
    250: (("F3b_T250_r0.2", "Tp250C_rate0p2", "Ast2018_F2a_T250_rate0.2", "F3a_T250_r0.2"),
          ("F3c_T250_r0.5", "Tp250C_rate0p5", None, None)),
    500: (("F3d_T500_r1.6", "Tp500C_rate1p6", None, None),),
}
SHARED = ("G0_emit_eV", "sigc_emit_GPa", "G0_peierls_eV", "sigc_peierls_GPa")
FAMILIES = {
    "A": ("G0_open_eV", "Gf_open_eV", "sigc_open_GPa") + SHARED,
    "B": ("F_open_eV", "B_open_eV", "log10_sB_open_GPa", "log10_sigc_open_GPa") + SHARED,
}
NAMES = FAMILIES["A"]
MAX_EXTENSION_UM = 0.98
J_SCALE_MIN, J_SCALE_FRACTION = 20.0, 0.05
K_SCALE, S_SCALE = 0.2, 0.03
FAMILY_WEIGHTS = {"J": 1.0, "load": 1.0, "stiffness": 0.5}
CENSOR_PENALTY = 1.0  # working-scale units added per unsupported J point
# The longest measured test reached ~3.5 um; a run uncracked by then is censored.
HORIZON_U_M = 3.5e-6
SECTION = SectionPlasticityConfig(mode="full_section", stations_outside_hinge=6)
BULK_FLOW_SEEDS = Path("runs/ast2018_bulk_flow_per_temperature_v1")


def _rows(name):
    with (DATA / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _weights(x):
    x = np.asarray(x, float)
    if x.size == 1:
        return np.ones(1)
    edges = np.concatenate([[x[0]], .5 * (x[1:] + x[:-1]), [x[-1]]])
    w = np.diff(edges)
    return w / np.sum(w)


def targets():
    J, load, stiff = (_rows("J_R_curves.csv"), _rows("K_displacement_curves.csv"),
                      _rows("relative_stiffness.csv"))
    out = {}
    for T, group in GROUPS.items():
        cases = []
        for curve, condition, load_id, stiff_id in group:
            rows = sorted(((float(r["crack_extension_um"]), float(r["J_N_per_m"]))
                           for r in J if r["curve_id"] == curve
                           and 0 < float(r["crack_extension_um"]) <= MAX_EXTENSION_UM))
            x = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
            case = {"curve_id": curve, "condition_id": condition, "J_da_um": x, "J": y,
                    "J_w": _weights(x), "J_scale": max(J_SCALE_MIN, J_SCALE_FRACTION * float(np.ptp(y)))}
            if load_id:
                lr = sorted((float(r["displacement_um"]), float(r["K_MPa_sqrt_m"]))
                            for r in load if r["curve_id"] == load_id and r["branch"] == "loading")
                sr = sorted((float(r["displacement_um"]), float(r["relative_stiffness"]))
                            for r in stiff if r["curve_id"] == stiff_id)
                su = np.array([r[0] for r in sr]); sv = np.array([r[1] for r in sr])
                # Normalize by the pre-growth peak, as the model's S=C0/C(a).
                sv = sv / np.max(sv[su <= 0.6])
                case.update({"load_u_um": np.array([r[0] for r in lr]),
                             "load_K": np.array([r[1] for r in lr]),
                             "stiff_u_um": su, "stiff": sv})
            cases.append(case)
        out[T] = cases
    return out


def isothermal_material(base, T_K, x, family="A"):
    """Set fitted barrier values at T_K; no temperature dependence is imposed."""
    x = np.asarray(x, float)
    if x.size != len(FAMILIES[family]):
        raise RuntimeError(f"family {family} expects {len(FAMILIES[family])} coordinates")
    ge, sc_emit, gp, sc_glide = map(float, x[-4:])
    if family == "A":
        G0_open, gf, sc_open = map(float, x[:3])
        if not 0 < gf < G0_open:
            raise ValueError("opening floor outside (0, G0)")
        cleavage = replace(base.cleavage, G0_ref_eV=G0_open, S0_over_kB=0.0, Gf_ref_eV=gf,
                           Sf_over_kB=0.0, sigc_ref_Pa=sc_open * 1e9,
                           d_sigc_dT_Pa_per_K=0.0, Cp_floor_over_kB=0.0)
    else:
        F, B, log_sB, log_sc = map(float, x[:4])
        cleavage = FiniteCutoffOpeningSurface(F, B, 10 ** log_sB * 1e9, 10 ** log_sc * 1e9,
                                              Tref_K=float(T_K))
        try:
            cleavage.validate()
        except ValueError as exc:
            raise ValueError(f"Model B opening outside its domain: {exc}") from None
    emission = replace(base.emission, G00_eV=ge, gT_eV_per_K=0.0,
                       sigc0_Pa=sc_emit * 1e9, sT_Pa_per_K=0.0)
    # Native glide surfaces are stress_scale x emission sc; rescale so the
    # Peierls sc is the fitted value and Taylor keeps its native ratio to it.
    peierls = replace(base.peierls, H0_eV=gp, activation_entropy_kB=0.0,
                      stress_scale=sc_glide / sc_emit)
    taylor = replace(base.taylor, stress_scale=base.taylor.stress_scale / base.peierls.stress_scale
                     * sc_glide / sc_emit)
    material = replace(base, cleavage=cleavage, emission=emission, peierls=peierls, taylor=taylor,
                       candidate_id=f"ast2018_isothermal_{family}_{T_K:.2f}K")
    if not physically_admissible(material):
        raise ValueError("isothermal material outside admissible surfaces")
    return material


# Model B start: the v8 21 C finite-cutoff selection (F, B, s_B, sc=16 GPa).
MODEL_B_SEED = (0.847384, 0.313376, float(np.log10(3.494103)), float(np.log10(16.0)))


def seed_coordinates(base, T_K, family="A"):
    c, e = base.cleavage, base.emission
    G0, Gf, sc = c.endpoints(T_K)
    ge = e.G00_eV + e.gT_eV_per_K * (T_K - e.Tref_K)
    se = e.sigc0_Pa + e.sT_Pa_per_K * (T_K - e.Tref_K)
    gp = base.peierls.surface(e).zero_stress_eV(T_K)
    opening = [G0, Gf, sc / 1e9] if family == "A" else list(MODEL_B_SEED)
    return np.array(opening + [ge, se / 1e9, gp, base.peierls.stress_scale * se / 1e9])


# One absolute box per family at every temperature: bounds centred on an
# inherited material's value at T would impose its temperature dependence.
_SHARED_LO, _SHARED_HI = [0.5, 0.5, 0.4, 0.3], [3.0, 4.5, 2.6, 5.0]
BOUNDS = {
    "A": (np.array([2.0, 0.05, 0.5] + _SHARED_LO), np.array([6.5, 1.3, 4.0] + _SHARED_HI)),
    "B": (np.array([0.0, 0.0, -0.5, 0.6] + _SHARED_LO),
          np.array([4.0, 6.0, 1.3, 1.7] + _SHARED_HI)),
}


# Optional box that keeps tip emission competitive with opening: at the
# ~2.75 GPa resolved tip stress the emission barrier is at most ~1.0 eV
# (~40 kT at 21 C) instead of the ~1.1 eV, emission-off v3/v4 optimum.
EMISSION_ACTIVE_HI = {"G0_emit_eV": 1.6, "sigc_emit_GPa": 1.3}


def bounds_around(x0, family="A", emission_active=False):
    lo, hi = BOUNDS[family][0].copy(), BOUNDS[family][1].copy()
    if emission_active:
        for name, value in EMISSION_ACTIVE_HI.items():
            hi[FAMILIES[family].index(name)] = value
    return lo, hi


def previous_best(temperature_C, family="A", previous=PREVIOUS, base=None, T_K=None, n=3):
    """Best fully evaluated earlier points as extra starts.

    v3 tied the Peierls sc to the emission sc; the implied Peierls sc is
    appended so the carried point is the same material.
    """
    path = Path(previous) / f"model_{family}" / f"T{temperature_C}_evaluations.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if all(c["status"] not in ("numerical_failure", "evaluation_error")
                                   for c in r["results"])]
    rows.sort(key=lambda r: r["objective"])
    k = FAMILIES[family].index("sigc_emit_GPa")
    return [np.asarray(r["x"], float) if len(r["x"]) == len(FAMILIES[family])
            else np.append(r["x"], base.peierls.stress_scale * r["x"][k]) for r in rows[:n]]


_WORKER = {}


def _init_worker():
    primary, _, base = base_material()
    _WORKER.update(primary=primary, base=selected_material(base))


def run_case(job):
    """One forward condition; returns compact observations for scoring."""
    T_K, x, curve, condition, family, tip_interaction = job
    primary, base = _WORKER["primary"], _WORKER["base"]
    start = time.monotonic()
    try:
        material = isothermal_material(base, T_K, x, family)
    except ValueError as exc:  # physically inadmissible trial: a scored outcome
        return {"curve_id": curve, "status": "inadmissible_material",
                "numerical_failure": f"{type(exc).__name__}: {exc}",
                "wall_seconds": time.monotonic() - start}
    try:
        cond = primary.ev.data.condition(condition)
        probe = AstAssumptions(target_extension_m=MAX_EXTENSION_UM * 1e-6,
                               loading_mode="nominal_bending_shared_temperature",
                               bending_opening_shape_path=str(SHAPE)).specimen(cond)
        # The loader caps imposed u at an elastic-equivalent nominal K.
        K_horizon = HORIZON_U_M * probe.geometry_factor(probe.a0_m) / (
            probe.initial_compliance_m_per_N + probe.instrument_compliance_m_per_N) / 1e6
        assumptions = AstAssumptions(target_extension_m=MAX_EXTENSION_UM * 1e-6,
            nominal_K_step_MPa_sqrt_m=.1, maximum_nominal_K_MPa_sqrt_m=K_horizon,
            loading_mode="nominal_bending_shared_temperature",
            bending_opening_shape_path=str(SHAPE))
        specimen = assumptions.specimen(cond)
        physics = assumptions.physics(primary.ev.base_physics, specimen)
        pred = predict_spatial(primary.ev.spatial, material, physics, specimen,
                               float(cond["temperature_K"]),
                               float(cond["initial_elastic_Kdot_MPa_sqrt_m_per_s"]),
                               assumptions, section=SECTION, tip_interaction=tip_interaction)
        observed = observe_like_ast(pred.history, specimen)
        C0 = specimen.elastic_compliance(specimen.a0_m)
        stiffness_limit = C0 / specimen.elastic_compliance(specimen.a0_m + MAX_EXTENSION_UM * 1e-6)
        tip = pred.events[-1] if pred.events else {}
        da, J = post_event_J_curve(observed)
        return {"curve_id": curve, "status": pred.status,
                "numerical_failure": pred.numerical_failure,
                "events": len(pred.events), "wall_seconds": time.monotonic() - start,
                "J_da_um": (np.asarray(da) * 1e6).tolist(), "J": np.asarray(J).tolist(),
                "u_um": [o.u_m * 1e6 for o in observed],
                "K_nominal": [o.K_nominal_MPa_sqrt_m for o in observed],
                "stiffness": [o.relative_stiffness for o in observed],
                "bulk_upl_um": (pred.events[-1].get("bulk_plastic_u_m", 0.0) * 1e6
                                if pred.events else None),
                "tip_radius_nm": (pred.events[-1]["tip_radius_pre_m"] * 1e9
                                  if pred.events else None),
                "tip_activations": tip.get("cumulative_activations"),
                "tip_shielding_MPa_sqrt_m": tip.get("shielding_pre_MPa_sqrt_m"),
                "tip_retained_density_m2": tip.get("rho_retained_sum_m2"),
                "tip_emitted_line_per_m": tip.get("cumulative_emitted_line_per_m"),
                "tip_escaped_line_per_m": tip.get("cumulative_escaped_line_per_m"),
                "stiffness_at_extension_limit": stiffness_limit}
    except Exception as exc:  # adapter/code error: the caller must fail closed
        return {"curve_id": curve, "status": "evaluation_error",
                "numerical_failure": f"{type(exc).__name__}: {exc}",
                "wall_seconds": time.monotonic() - start}


def measured_extension_limit_u(target, result):
    """Displacement where the measured crack passes the modelled extension range.

    The elastic field table stops at MAX_EXTENSION_UM; beyond it the model
    cannot represent the specimen, so later load and stiffness are not scored.
    The limit is where the measured stiffness first falls (after the initial
    elastic rise) to the elastic stiffness of that crack length.
    """
    limit = result.get("stiffness_at_extension_limit")
    if limit is None:
        return np.inf
    su, sv = target["stiff_u_um"], target["stiff"]
    peak = int(np.argmax(sv))
    past = np.nonzero(sv[peak:] <= limit)[0]
    return float(su[peak + past[0]]) if past.size else np.inf


def score_case(target, result):
    """Weighted mean squared standardized residual per family."""
    families = {}
    ok = result["status"] in ("complete", "model_limit_nominal_K_window", "physical_ligament_failure")
    x, y, w, s = target["J_da_um"], target["J"], target["J_w"], target["J_scale"]
    if not ok or len(result.get("J_da_um", [])) < 2:
        families["J"] = float(np.sum(w * (CENSOR_PENALTY + y / s) ** 2))
    else:
        mx, my = np.asarray(result["J_da_um"]), np.asarray(result["J"])
        supported = x <= mx[-1] + 1e-9
        r = np.zeros_like(y)
        r[supported] = (np.interp(x[supported], mx, my) - y[supported]) / s
        # Censored: the model never reached this extension within the tested
        # displacement, so its J there is held at its last value (both signs).
        r[~supported] = np.abs(my[-1] - y[~supported]) / s + CENSOR_PENALTY
        families["J"] = float(np.sum(w * r * r))
    if "load_u_um" in target:
        if not ok or len(result.get("u_um", [])) < 3:
            families["load"] = families["stiffness"] = (CENSOR_PENALTY + 10.0) ** 2
        else:
            u = np.asarray(result["u_um"])
            u_end = min(u[-1], measured_extension_limit_u(target, result))
            for key, tu, tv, scale, model in (
                    ("load", target["load_u_um"], target["load_K"], K_SCALE, result["K_nominal"]),
                    ("stiffness", target["stiff_u_um"], target["stiff"], S_SCALE, result["stiffness"])):
                keep = tu <= u_end
                if np.sum(keep) < 3:
                    families[key] = (CENSOR_PENALTY + 10.0) ** 2
                    continue
                r = (np.interp(tu[keep], u, model) - tv[keep]) / scale
                families[key] = float(np.sum(_weights(tu[keep]) * r * r))
    return families


def objective(cases, results):
    total = weight = 0.0
    detail = []
    for target, result in zip(cases, results):
        fam = score_case(target, result)
        for key, value in fam.items():
            total += FAMILY_WEIGHTS[key] * value
            weight += FAMILY_WEIGHTS[key]
        detail.append({"curve_id": target["curve_id"], "status": result["status"],
                       "families": fam, "wall_seconds": result.get("wall_seconds")})
    return total / weight, detail


class TemperatureFit:
    def __init__(self, temperature_C, pool, output=OUTPUT, family="A", emission_active=False,
                 tip_interaction="native"):
        if family not in FAMILIES:
            raise ValueError(f"unknown opening family {family}")
        self.C, self.family, self.names = temperature_C, family, FAMILIES[family]
        self.cases = targets()[temperature_C]
        primary, _, base = base_material()
        self.base = selected_material(base)
        cond = primary.ev.data.condition(self.cases[0]["condition_id"])
        self.T_K = float(cond["temperature_K"])
        self.x0 = seed_coordinates(self.base, self.T_K, family)
        self.lo, self.hi = bounds_around(self.x0, family, emission_active)
        bulk = BULK_FLOW_SEEDS / f"peierls_G0_T{temperature_C}.json"
        if bulk.is_file():  # start glide from the plasticity-consistent value
            self.x0[self.names.index("G0_peierls_eV")] = json.loads(bulk.read_text())["fitted"]["G0_eV"]
        self.pool = pool
        if tip_interaction not in TIP_INTERACTIONS:
            raise ValueError(f"unknown tip interaction {tip_interaction}")
        self.tip_interaction = tip_interaction
        self.output = Path(output) / (f"model_{family}" + ("_emission_active" if emission_active else "")
                                      + ("" if tip_interaction == "native" else f"_{tip_interaction}"))
        self.output.mkdir(parents=True, exist_ok=True)
        self.log = self.output / f"T{temperature_C}_evaluations.jsonl"
        self.load_cache()

    def load_cache(self):
        self.cache = {}
        if self.log.is_file():
            for line in self.log.read_text().splitlines():
                row = json.loads(line)
                # Stored forward results are re-scored with the current objective.
                row["objective"], row["detail"] = objective(self.cases, row["results"])
                self.cache[tuple(np.round(row["x"], 10))] = row

    def evaluate_many(self, xs, stage):
        xs = [np.clip(np.asarray(x, float), self.lo, self.hi) for x in xs]
        todo = [x for x in xs if tuple(np.round(x, 10)) not in self.cache]
        # Submit every case; record each point as soon as all its cases finish,
        # so partial batches are checkpointed and visible.
        futures = {}
        for i, x in enumerate(todo):
            for k, c in enumerate(self.cases):
                job = (self.T_K, x.tolist(), c["curve_id"], c["condition_id"], self.family,
                       self.tip_interaction)
                futures[self.pool.submit(run_case, job)] = (i, k)
        n = len(self.cases)
        pending = {i: [None] * n for i in range(len(todo))}
        try:
            for future in as_completed(futures):
                i, k = futures[future]
                result = future.result()
                if result["status"] == "evaluation_error":
                    # Never score a code error as a physical or numerical outcome.
                    (self.output / f"T{self.C}_ABORTED.json").write_text(
                        json.dumps(result, indent=1, default=_jsonable) + "\n")
                    raise RuntimeError(f"forward evaluation error: {result['numerical_failure']}")
                pending[i][k] = result
                if all(r is not None for r in pending[i]):
                    x, res = todo[i], pending.pop(i)
                    value, detail = objective(self.cases, res)
                    row = {"x": x.tolist(), "objective": value, "stage": stage,
                           "detail": detail, "results": res}
                    self.cache[tuple(np.round(x, 10))] = row
                    with self.log.open("a") as stream:
                        stream.write(json.dumps(row, default=_jsonable) + "\n")
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        return [self.cache[tuple(np.round(x, 10))]["objective"] for x in xs]

    def run(self, n_space=40, n_simplex_iter=60, seed=2018, restarts=1):
        rng = np.random.default_rng(seed)
        # Stage A: seed plus Latin-hypercube space filling in the declared box.
        d = len(self.names)
        lhs = (np.argsort(rng.random((n_space, d)), axis=0) + rng.random((n_space, d))) / n_space
        starts = [self.x0] + [self.lo + u * (self.hi - self.lo) for u in lhs]
        starts += previous_best(self.C, self.family, base=self.base, T_K=self.T_K)
        extra = self.output / f"T{self.C}_extra_starts.json"
        if extra.is_file():  # declared starts, e.g. B matched to A's best barrier
            starts += [np.asarray(x, float) for x in json.loads(extra.read_text())["starts"]]
        # On resume, the best valid points already evaluated in this run.
        valid = sorted((r for r in self.cache.values()
                        if all(c["status"] not in ("numerical_failure", "evaluation_error")
                               for c in r["results"])), key=lambda r: r["objective"])
        starts += [np.asarray(r["x"]) for r in valid[:3]]
        values = self.evaluate_many(starts, "space_filling")
        best = starts[int(np.argmin(values))]
        # Stage B: parallel Nelder-Mead in normalized coordinates from the best
        # start, restarted from its best point with a fresh simplex.
        for attempt in range(restarts + 1):
            best = self.nelder_mead(best, n_simplex_iter, f"r{attempt}_" if attempt else "")
        return self.write_summary()

    def nelder_mead(self, best, n_iter, prefix=""):
        d = len(self.names)
        span = self.hi - self.lo
        simplex = [best] + [np.clip(best + .1 * span * np.eye(d)[k], self.lo, self.hi)
                            for k in range(d)]
        f = self.evaluate_many(simplex, f"{prefix}simplex_init")
        for it in range(n_iter):
            idx = np.argsort(f); simplex = [simplex[i] for i in idx]; f = [f[i] for i in idx]
            centroid = np.mean(simplex[:-1], axis=0)
            trials = [centroid + a * (centroid - simplex[-1]) for a in (1., 2., .5, -.5)]
            ft = self.evaluate_many(trials, f"{prefix}simplex_{it}")
            k = int(np.argmin(ft))
            if ft[k] < f[-2] or (k in (2, 3) and ft[k] < f[-1]):
                simplex[-1], f[-1] = np.clip(trials[k], self.lo, self.hi), ft[k]
            else:  # shrink toward best
                simplex = [simplex[0]] + [simplex[0] + .5 * (s - simplex[0]) for s in simplex[1:]]
                f = [f[0]] + self.evaluate_many(simplex[1:], f"{prefix}shrink_{it}")
            if np.max(np.abs(np.asarray(simplex) - simplex[0]) / span) < 2e-3:
                break
        return simplex[int(np.argmin(f))]

    def write_summary(self):
        rows = sorted(self.cache.values(), key=lambda r: r["objective"])
        names = self.names
        summary = {"temperature_C": self.C, "T_K": self.T_K, "family": self.family,
                   "names": names, "seed": dict(zip(names, self.x0.tolist())),
                   "bounds": {"lower": self.lo.tolist(), "upper": self.hi.tolist()},
                   "best": dict(zip(names, rows[0]["x"])), "best_objective": rows[0]["objective"],
                   "best_detail": rows[0]["detail"], "evaluations": len(rows),
                   "scales": {"J_min": J_SCALE_MIN, "J_fraction": J_SCALE_FRACTION,
                              "K": K_SCALE, "stiffness": S_SCALE},
                   "family_weights": FAMILY_WEIGHTS, "censor_penalty": CENSOR_PENALTY,
                   "section": SECTION.as_dict(), "horizon_u_m": HORIZON_U_M,
                   "tip_interaction": self.tip_interaction,
                   "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (self.output / f"T{self.C}_summary.json").write_text(
            json.dumps(summary, indent=1, default=_jsonable) + "\n")
        return summary


if __name__ == "__main__":
    # python -m ...per_temperature_joint_fit T workers [A|B] [--emission-active]
    #     [--signed-pileup] [--time]
    temperature = int(sys.argv[1])
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    family = next((a for a in sys.argv[3:] if a in FAMILIES), "A")
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        fit = TemperatureFit(temperature, pool, family=family,
                             emission_active="--emission-active" in sys.argv[3:],
                             tip_interaction=("signed_pileup" if "--signed-pileup" in sys.argv[3:]
                                              else "native"))
        if "--time" in sys.argv[3:]:
            t = time.monotonic()
            fit.evaluate_many([fit.x0], "timing")
            row = fit.cache[tuple(np.round(np.clip(fit.x0, fit.lo, fit.hi), 10))]
            print(json.dumps({"objective": row["objective"], "detail": row["detail"],
                              "wall": time.monotonic() - t}, default=_jsonable))
        else:
            print(json.dumps(fit.run(), default=_jsonable)[:2000])
