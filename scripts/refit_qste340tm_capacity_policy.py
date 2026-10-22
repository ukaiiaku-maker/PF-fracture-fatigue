#!/usr/bin/env python3
"""Capacity-policy refits for the QSTE340TM analytical EXP-floor reducer.

This is deliberately additive.  The accepted 63beec13 fit and its files are
read-only controls.  The three new policies are: no empirical cap, fixed 1000,
and fixed 10000 retained-count units.  Physical source occupancy and the
emission/Peierls/Taylor/recovery/opening laws are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.special import gammainc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.config import EV_TO_J
from arrhenius_fracture.material_manifest import KB_EV_PER_K
from scripts.calibrate_qste340tm_stateful_exp_floor import (
    ATTEMPT_FREQUENCY_S, BETA_BACK, BURGERS_M, C_BLUNT,
    COOPERATIVE_HITS, COOPERATIVE_TAU_S, EVENT_LENGTH_M,
    NOMINAL_FREQUENCY_HZ, PHASE_STEPS, POISSON_RATIO, PROCESS_ZONE_M,
    RHO0_M2, SHEAR_MODULUS_PA, SMOOTHING_RATIO, TEMPERATURE_K,
    TIP_RADIUS_M, V_EMB_B3, build_records, exp_floor_barrier,
)
from scripts.fatigue_informed_joint_barrier_v3_v10230 import SmoothBroadBasis

DATA = ROOT / "literature_data/fatigue_recovered_review"
CONTROL = DATA / "model_conditional/stateful_exp_floor/fit.json"
OUT = DATA / "model_conditional/stateful_exp_floor_capacity_policy"
POLICIES = {"uncapped": None, "fixed_1000": 1_000.0, "fixed_10000": 10_000.0}
PARAMETER_NAMES = (
    "opening_floor_over_kBT", "opening_amplitude_over_kBT",
    "log_opening_sigma_b_GPa", "log_opening_exponent",
    "emission_G0_298_eV", "log_plastic_sigc_298_GPa",
    "escape_G0_298_eV", "cleavage_backstress_fraction",
    "log_recovery_per_s", "log1p_initial_retained_count",
)
LOWER = np.array([8., 0., math.log(.1), math.log(.1), .30, math.log(.1), .30, 0., math.log(1e-8), 0.])
UPPER = np.array([25., 100., math.log(100.), math.log(10.), 3., math.log(20.), 5., 2., math.log(1e5), math.log1p(500.)])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unpack(x: np.ndarray, opening_model: str = "single_smooth_broad") -> dict[str, float]:
    out = {
        "opening_floor_over_kBT": float(x[0]),
        "opening_amplitude_over_kBT": float(x[1]),
        "opening_sigma_b_Pa": float(math.exp(x[2]) * 1e9),
        "opening_exponent": float(math.exp(x[3])),
        "emission_G0_298_eV": float(x[4]),
        "plastic_sigc_298_Pa": float(math.exp(x[5]) * 1e9),
        "escape_G0_298_eV": float(x[6]),
        "cleavage_backstress_fraction": float(x[7]),
        "recovery_per_s": float(math.exp(x[8])),
        "initial_retained_count": float(math.expm1(x[9])),
    }
    if opening_model == "two_scale_smooth_broad":
        out.update({
            "opening_secondary_amplitude_over_kBT": float(x[10]),
            "opening_secondary_sigma_b_Pa": float(math.exp(x[11]) * 1e9),
            "opening_secondary_exponent": float(math.exp(x[12])),
        })
        out["plastic_EXP_floor_n"] = float(math.exp(x[13])) if len(x) > 13 else 1.0
    else:
        out["plastic_EXP_floor_n"] = float(math.exp(x[10])) if len(x) > 10 else 1.0
    return out


def pack(p: dict[str, float], initial: float = 0.) -> np.ndarray:
    return np.array([
        p["opening_floor_over_kBT"], p["opening_amplitude_over_kBT"],
        math.log(p["opening_sigma_b_Pa"] / 1e9), math.log(p["opening_exponent"]),
        p["emission_G0_298_eV"], math.log(p["plastic_sigc_298_Pa"] / 1e9),
        p["escape_G0_298_eV"], p["cleavage_backstress_fraction"],
        math.log(p["recovery_per_s"]), math.log1p(initial),
    ])


def exact_retained_step(n: float, source: float, recovery: float,
                        cycles: float, capacity: float | None) -> tuple[float, float, float]:
    """Exact frozen-coefficient update and source/recovery transfers."""
    if cycles <= 0:
        return n, 0., 0.
    loss = recovery + (source / capacity if capacity is not None else 0.)
    if loss <= 1e-14:
        new = n + source * cycles
    else:
        decay = math.exp(-min(loss * cycles, 700.))
        steady = source / loss
        new = steady + (n - steady) * decay
    # Integrated balance: delta n = source transfer - recovery transfer.
    if capacity is None:
        source_transfer = source * cycles
    else:
        # integral source*(1-n/cap)dN reconstructed from balance and recovery.
        if loss <= 1e-14:
            mean_n_int = n * cycles + .5 * source * cycles**2
        else:
            steady = source / loss
            mean_n_int = steady * cycles + (n - steady) * (-math.expm1(-min(loss * cycles, 700.))) / loss
        source_transfer = source * cycles - source / capacity * mean_n_int
    if recovery <= 0:
        recovery_transfer = 0.
    else:
        recovery_transfer = source_transfer - (new - n)
    return max(float(new), 0.), float(source_transfer), float(recovery_transfer)


def stress_cycle(delta_k: float, r: float, n: float,
                 n_phase: int = PHASE_STEPS) -> tuple[np.ndarray, float, float]:
    phase = 2 * math.pi * (np.arange(n_phase) + .5) / n_phase
    fraction = .5 * (1 + r) + .5 * (1 - r) * np.cos(phase)
    radius = TIP_RADIUS_M + C_BLUNT * BURGERS_M * n
    stress = delta_k / (1 - r) * 1e6 * fraction / math.sqrt(2 * math.pi * radius)
    back = BETA_BACK * SHEAR_MODULUS_PA * BURGERS_M / (
        2 * math.pi * (1 - POISSON_RATIO) * PROCESS_ZONE_M) * n
    return stress, radius, back


def embrittlement_eV(n: float) -> float:
    rho = RHO0_M2 + n / PROCESS_ZONE_M**2
    stored = .5 * SHEAR_MODULUS_PA * BURGERS_M**2 * rho
    return stored * V_EMB_B3 * BURGERS_M**3 / EV_TO_J


def retained_source_per_cycle(lam_emit: np.ndarray, lam_escape: np.ndarray,
                              frequency_hz: float, mode: str) -> tuple[float, float, float]:
    dt = 1.0 / (frequency_hz * len(lam_emit))
    mu_emit = float(np.sum(lam_emit) * dt)
    mu_escape = float(np.sum(lam_escape) * dt)
    if mode == "native_full_cycle":
        source = mu_emit * math.exp(-min(mu_escape, 700.0))
    elif mode == "birth_phase_resolved":
        remaining_escape = np.cumsum((lam_escape * dt)[::-1])[::-1]
        source = float(np.sum(lam_emit * dt * np.exp(-np.minimum(remaining_escape, 700.0))))
    elif mode == "continuous_birth_exact":
        exposure = np.asarray(lam_escape, dtype=float) * dt
        after = np.cumsum(exposure[::-1])[::-1] - exposure
        phi1 = np.ones_like(exposure)
        active = np.abs(exposure) > 1.0e-12
        phi1[active] = -np.expm1(-exposure[active]) / exposure[active]
        source = float(np.sum(lam_emit * dt * phi1 * np.exp(-np.minimum(after, 700.0))))
    else:
        raise ValueError(f"unknown retention timing mode: {mode}")
    return mu_emit, mu_escape, source


def evaluate(x: np.ndarray, records: list[dict], policy: str,
             interval_substeps: int = 1, evolve: bool = True,
             legacy_prefill: bool = False,
             frequency_hz: float = NOMINAL_FREQUENCY_HZ,
             retention_timing: str = "native_full_cycle",
             waveform_steps: int = PHASE_STEPS,
             opening_model: str = "single_smooth_broad") -> dict[str, np.ndarray]:
    p = unpack(x, opening_model); capacity = POLICIES[policy]
    opening = SmoothBroadBasis(p["opening_sigma_b_Pa"], p["opening_exponent"], SMOOTHING_RATIO)
    opening_secondary = None
    if opening_model == "two_scale_smooth_broad":
        opening_secondary = SmoothBroadBasis(
            p["opening_secondary_sigma_b_Pa"],
            p["opening_secondary_exponent"], SMOOTHING_RATIO)
    emit = exp_floor_barrier("qste_emit", p["emission_G0_298_eV"], p["plastic_sigc_298_Pa"], 1e11, exp_n=p["plastic_EXP_floor_n"])
    peierls = exp_floor_barrier("qste_peierls", p["escape_G0_298_eV"], p["plastic_sigc_298_Pa"], 1e12, exp_n=p["plastic_EXP_floor_n"])
    taylor = exp_floor_barrier("qste_taylor", p["escape_G0_298_eV"], p["plastic_sigc_298_Pa"], 1e11, exp_n=p["plastic_EXP_floor_n"])
    keys = ("prediction", "retained", "density_m2", "radius", "backstress",
            "emission_per_cycle", "peierls_per_cycle", "taylor_per_cycle",
            "escape_per_cycle", "storage_fraction", "source_per_cycle",
            "capacity_factor", "recovery_per_cycle", "source_transfer",
            "recovery_transfer", "raw_opening_barrier_over_kBT")
    out = {k: np.empty(len(records)) for k in keys}
    groups: dict[float, list[int]] = {}
    for i, row in enumerate(records): groups.setdefault(float(row["R"]), []).append(i)
    for r, indices in groups.items():
        n = p["initial_retained_count"]
        prior_cycles = 0.; prior_dk = float(records[indices[0]]["DeltaK_MPa_sqrt_m"])
        for pos, i in enumerate(indices):
            row = records[i]; cycles = float(row["cycles"]); dk = float(row["DeltaK_MPa_sqrt_m"])
            delta = max(cycles - prior_cycles, 0.)
            if pos == 0 and not legacy_prefill: delta = 0.
            if evolve and delta > 0:
                for j in range(max(interval_substeps, 1)):
                    frac = (j + .5) / max(interval_substeps, 1)
                    dki = dk if (pos == 0 and legacy_prefill) else prior_dk + frac * (dk - prior_dk)
                    s, _, b = stress_cycle(dki, r, n, waveform_steps)
                    se = np.maximum(s - b, 0.)
                    le = emit.rate(se, TEMPERATURE_K)
                    lp = peierls.rate(se, TEMPERATURE_K); lt = taylor.rate(se, TEMPERATURE_K)
                    lx = 1 / (1 / np.maximum(lp, 1e-300) + 1 / np.maximum(lt, 1e-300))
                    mu_e, mu_x, source = retained_source_per_cycle(le, lx, frequency_hz, retention_timing)
                    n, _, _ = exact_retained_step(n, source, p["recovery_per_s"] / frequency_hz,
                                                  delta / max(interval_substeps, 1), capacity)
            prior_cycles = cycles; prior_dk = dk
            s, radius, b = stress_cycle(dk, r, n, waveform_steps); se = np.maximum(s - b, 0.)
            le = emit.rate(se, TEMPERATURE_K); lp = peierls.rate(se, TEMPERATURE_K)
            lt = taylor.rate(se, TEMPERATURE_K)
            lx = 1 / (1 / np.maximum(lp, 1e-300) + 1 / np.maximum(lt, 1e-300))
            mu_e = float(np.mean(le) / frequency_hz)
            mu_p = float(np.mean(lp) / frequency_hz)
            mu_t = float(np.mean(lt) / frequency_hz)
            mu_x = float(np.mean(lx) / frequency_hz)
            _, _, source = retained_source_per_cycle(le, lx, frequency_hz, retention_timing)
            store = source / max(mu_e, 1e-300)
            cf = 1. if capacity is None else max(1 - n / capacity, 0.)
            open_s = np.maximum(s - p["cleavage_backstress_fraction"] * b, 0.)
            raw = p["opening_floor_over_kBT"] + p["opening_amplitude_over_kBT"] * opening.value(open_s)
            if opening_secondary is not None:
                raw = raw + p["opening_secondary_amplitude_over_kBT"] * opening_secondary.value(open_s)
            raw = raw - embrittlement_eV(n) / (KB_EV_PER_K * TEMPERATURE_K)
            elementary = ATTEMPT_FREQUENCY_S * np.exp(-np.clip(raw, 0., 700.))
            completed = gammainc(COOPERATIVE_HITS, np.minimum(elementary * COOPERATIVE_TAU_S, 1e12)) / COOPERATIVE_TAU_S
            out["prediction"][i] = max(EVENT_LENGTH_M * float(np.mean(completed)) / frequency_hz, 1e-300)
            vals = {"retained": n, "density_m2": RHO0_M2 + n / PROCESS_ZONE_M**2,
                    "radius": radius, "backstress": b, "emission_per_cycle": mu_e,
                    "peierls_per_cycle": mu_p, "taylor_per_cycle": mu_t,
                    "escape_per_cycle": mu_x, "storage_fraction": store,
                    "source_per_cycle": source, "capacity_factor": cf,
                    "recovery_per_cycle": p["recovery_per_s"] / frequency_hz * n,
                    "source_transfer": source * cf, "recovery_transfer": p["recovery_per_s"] / frequency_hz * n,
                    "raw_opening_barrier_over_kBT": float(np.min(raw))}
            for k, v in vals.items(): out[k][i] = v
    return out


def metrics(records: list[dict], state: dict[str, np.ndarray]) -> dict:
    obs = np.array([float(r["observed_dadn_m_per_cycle"]) for r in records]); pred = state["prediction"]
    rv = np.array([float(r["R"]) for r in records]); dk = np.array([float(r["DeltaK_MPa_sqrt_m"]) for r in records])
    err = np.log10(pred / obs); blocks = {}
    for r in sorted(set(rv)):
        m = rv == r
        blocks[str(r)] = {"points": int(m.sum()), "rmse_log10_rate": float(np.sqrt(np.mean(err[m]**2))),
                          "bias_log10_rate": float(np.mean(err[m])),
                          "observed_loglog_slope": float(np.polyfit(np.log10(dk[m]), np.log10(obs[m]), 1)[0]),
                          "predicted_loglog_slope": float(np.polyfit(np.log10(dk[m]), np.log10(pred[m]), 1)[0]),
                          "retained_count_final": float(state["retained"][m][-1]),
                          "density_final_m2": float(state["density_m2"][m][-1])}
    return {"overall_rmse_log10_rate": float(np.sqrt(np.mean(err**2))),
            "overall_bias_log10_rate": float(np.mean(err)), "block_metrics": blocks,
            "min_raw_opening_barrier_over_kBT": float(np.min(state["raw_opening_barrier_over_kBT"])),
            "max_retained_count": float(np.max(state["retained"])), "max_density_m2": float(np.max(state["density_m2"])),
            "max_backstress_Pa": float(np.max(state["backstress"])), "max_tip_radius_m": float(np.max(state["radius"]))}


def starts() -> list[tuple[str, np.ndarray]]:
    control = json.loads(CONTROL.read_text())
    found = [("historical_main", pack(control["fit"]["parameters"]))]
    for item in control["leave_one_R_out"]:
        found.append((f"stored_excluding_R{item['held_out_R']}", pack(item["parameters"])))
    # Explicit stress-dependent opening-tail seed.
    tail = found[1][1].copy(); tail[0] = 14.; tail[1] = 45.; tail[2] = math.log(3.); tail[3] = math.log(1.5)
    found.append(("stress_dependent_tail", tail))
    uncapped = OUT / "fit_uncapped.json"
    if uncapped.exists():
        found.append(("completed_uncapped", np.asarray(json.loads(uncapped.read_text())["parameters_vector"], dtype=float)))
    return found


def residual_factory(records: list[dict], policy: str, substeps: int):
    obs = np.log10(np.array([float(r["observed_dadn_m_per_cycle"]) for r in records]))
    rv = np.array([float(r["R"]) for r in records]); weights = np.empty(len(records))
    for r in sorted(set(rv)): weights[rv == r] = 1 / math.sqrt(np.sum(rv == r))
    def residual(x: np.ndarray) -> np.ndarray:
        state = evaluate(x, records, policy, substeps)
        raw = (np.log10(state["prediction"]) - obs) * weights
        # Keep the activated opening law inside its valid positive-barrier domain.
        validity = max(0., .25 - float(np.min(state["raw_opening_barrier_over_kBT"]))) * 10
        return np.r_[raw, validity]
    return residual


def fit_policy(records: list[dict], policy: str, max_nfev: int, substeps: int) -> dict:
    residual = residual_factory(records, policy, substeps)
    scored = []
    for name, x in starts(): scored.append((float(np.dot(residual(x), residual(x))), name, x))
    selected = sorted(scored, key=lambda z: z[0])[:3]
    runs = []
    with ThreadPool(4) as pool:
        for _, name, x0 in selected:
            print(f"[{policy}] optimizing start {name}", flush=True)
            t0 = time.time()
            sol = least_squares(residual, np.clip(x0, LOWER, UPPER), bounds=(LOWER, UPPER),
                                max_nfev=max_nfev, x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10,
                                workers=pool.map)
            state = evaluate(sol.x, records, policy, substeps); met = metrics(records, state)
            runs.append({"start": name, "x": sol.x, "state": state, "metrics": met,
                         "optimizer": {"success": bool(sol.success), "status": int(sol.status), "message": str(sol.message),
                                       "nfev": int(sol.nfev), "cost": float(sol.cost), "elapsed_s": time.time()-t0,
                                       "optimality": float(sol.optimality)}})
            print(f"[{policy}] {name}: RMSE={met['overall_rmse_log10_rate']:.6f}, nfev={sol.nfev}, success={sol.success}", flush=True)
        best = min(runs, key=lambda z: z["metrics"]["overall_rmse_log10_rate"])
        # One convergence pass from the best multi-start result.
        print(f"[{policy}] refining best start {best['start']}", flush=True)
        sol = least_squares(residual, best["x"], bounds=(LOWER, UPPER), max_nfev=max_nfev,
                            x_scale="jac", ftol=2e-12, xtol=2e-12, gtol=2e-12, workers=pool.map)
    state = evaluate(sol.x, records, policy, substeps)
    frozen = evaluate(sol.x, records, policy, substeps, evolve=False)
    result = {"policy": policy, "fixed_capacity": POLICIES[policy], "parameters_vector": sol.x.tolist(),
              "parameters": unpack(sol.x), "metrics": metrics(records, state), "frozen_metrics": metrics(records, frozen),
              "optimizer": {"success": bool(sol.success), "status": int(sol.status), "message": str(sol.message),
                            "nfev": int(sol.nfev), "cost": float(sol.cost), "optimality": float(sol.optimality)},
              "multi_start": [{"start": z["start"], "metrics": z["metrics"], "optimizer": z["optimizer"]} for z in runs],
              "state": {k: v.tolist() for k, v in state.items()}, "frozen_prediction": frozen["prediction"].tolist()}
    return result


def write_case(records: list[dict], result: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True); policy = result["policy"]
    (OUT / f"fit_{policy}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    fields = list(records[0]) + ["predicted_dadn_m_per_cycle", "frozen_dadn_m_per_cycle", "residual_log10"] + list(result["state"])
    with (OUT / f"predictions_{policy}.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields, lineterminator="\n"); w.writeheader()
        for i, row in enumerate(records):
            pred = result["state"]["prediction"][i]; obs = float(row["observed_dadn_m_per_cycle"])
            extra = {"predicted_dadn_m_per_cycle": pred, "frozen_dadn_m_per_cycle": result["frozen_prediction"][i],
                     "residual_log10": math.log10(pred/obs)}
            extra.update({k: v[i] for k, v in result["state"].items()})
            w.writerow(dict(row) | extra)


def diagnostics(records: list[dict], results: dict[str, dict]) -> dict:
    # Exact-step identities and interval/prehistory sensitivities on finalists.
    checks = []
    for cap in (None, 1000.):
        for gamma in (0., 1e-12, 1e-4):
            n, _, _ = exact_retained_step(3., .02, gamma, 1234., cap)
            checks.append({"capacity": cap, "gamma_per_cycle": gamma, "result": n})
    sens = {}
    for policy, result in results.items():
        x = np.array(result["parameters_vector"])
        sens[policy] = {}
        for sub in (1, 2, 4, 8): sens[policy][f"substeps_{sub}"] = metrics(records, evaluate(x, records, policy, sub))
        sens[policy]["legacy_first_load_prefill"] = metrics(records, evaluate(x, records, policy, 1, legacy_prefill=True))
        sens[policy]["frequency_1Hz"] = metrics(records, evaluate(x, records, policy, 1, frequency_hz=1.0))
        sens[policy]["frequency_20Hz"] = metrics(records, evaluate(x, records, policy, 1, frequency_hz=20.0))
    return {"exact_step_samples": checks, "finalist_sensitivity": sens,
            "full_cycle_retention_identity": "source_per_cycle = mu_emit*exp(-mu_escape), identical to FatigueCycleHazardController.integrate_one_cycle",
            "initialization_primary": "shared fitted retained count; no assignment of pre-first-measurement cycles to first measured DeltaK"}


def make_plots(records: list[dict], results: dict[str, dict]) -> None:
    colors = {"uncapped":"#0072B2", "fixed_1000":"#D55E00", "fixed_10000":"#009E73"}
    rv=np.array([float(r["R"]) for r in records]); dk=np.array([float(r["DeltaK_MPa_sqrt_m"]) for r in records]); obs=np.array([float(r["observed_dadn_m_per_cycle"]) for r in records])
    fig, axes=plt.subplots(2,2,figsize=(11,8),sharex=False,sharey=True)
    for ax,r in zip(axes.flat,(.1,.3,.5,.7)):
        m=rv==r; order=np.argsort(dk[m]); ax.scatter(dk[m],obs[m],s=11,c="#333333",alpha=.5,label="data")
        for policy,res in results.items(): ax.plot(dk[m][order],np.array(res["state"]["prediction"])[m][order],lw=1.8,c=colors[policy],label=policy.replace("fixed_","cap "))
        ax.set(xscale="log",yscale="log",xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)",title=f"R={r:.1f}");ax.grid(True,which="both",alpha=.2)
    axes[0,0].set_ylabel("da/dN (m/cycle)"); axes[1,0].set_ylabel("da/dN (m/cycle)")
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,ncol=4,loc="upper center",frameon=False);fig.suptitle("QSTE340TM joint opening/EXP-floor capacity-policy refits",y=1.01);fig.tight_layout()
    fig.savefig(OUT/"qste340tm_capacity_policy_common_axes.png",dpi=240,bbox_inches="tight");fig.savefig(OUT/"qste340tm_capacity_policy_common_axes.svg",bbox_inches="tight");plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharey=True)
    for ax,r in zip(axes.flat,(.1,.3,.5,.7)):
        m=rv==r
        for policy,res in results.items(): ax.plot(dk[m],np.array(res["state"]["retained"])[m],lw=1.7,c=colors[policy],label=policy)
        ax2=ax.twinx(); ax2.tick_params(axis='y',colors='#888888',labelsize=7)
        ax.set(xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)",title=f"R={r:.1f}"); ax.set_ylabel("retained count"); ax.grid(True,alpha=.2)
    fig.suptitle("Retained-count histories (density = 5e12 + count/1 μm²)");fig.tight_layout();fig.savefig(OUT/"qste340tm_retained_histories.png",dpi=240,bbox_inches="tight");plt.close(fig)
    stress=np.linspace(0,25e9,600);fig,ax=plt.subplots(figsize=(8.5,5.5))
    for policy,res in results.items():
        p=res["parameters"]; basis=SmoothBroadBasis(p["opening_sigma_b_Pa"],p["opening_exponent"],SMOOTHING_RATIO)
        g=(p["opening_floor_over_kBT"]+p["opening_amplitude_over_kBT"]*basis.value(stress))*KB_EV_PER_K*TEMPERATURE_K
        ax.plot(stress/1e9,g,lw=2,c=colors[policy],label=policy)
    ax.set(xlabel="local opening stress (GPa)",ylabel=r"$G^*_{open}$ (eV)",title="Joint-fit opening barriers at 298.15 K");ax.grid(True,alpha=.2);ax.legend(frameon=False);fig.tight_layout();fig.savefig(OUT/"qste340tm_Gstar_capacity_policy.png",dpi=240,bbox_inches="tight");plt.close(fig)


def main() -> int:
    ap=argparse.ArgumentParser();ap.add_argument("--policy",choices=[*POLICIES,"all"],default="all");ap.add_argument("--max-nfev",type=int,default=500);ap.add_argument("--substeps",type=int,default=1);ap.add_argument("--summarize",action="store_true");args=ap.parse_args()
    records=build_records(DATA); wanted=list(POLICIES) if args.policy=="all" else [args.policy]
    results={}
    if args.summarize:
        for p in wanted: results[p]=json.loads((OUT/f"fit_{p}.json").read_text())
    else:
        for p in wanted:
            result=fit_policy(records,p,args.max_nfev,args.substeps);write_case(records,result);results[p]=result
            print(json.dumps({"policy":p,"metrics":result["metrics"],"optimizer":result["optimizer"]},indent=2),flush=True)
    if set(results)==set(POLICIES):
        diag=diagnostics(records,results);(OUT/"diagnostics.json").write_text(json.dumps(diag,indent=2,sort_keys=True)+"\n");make_plots(records,results)
        manifest={"schema":"qste340tm-capacity-policy-joint-refit/1","source_commit":"63beec13fe3827f150a955763d28b7b78a049d9b",
                  "temperature_K":TEMPERATURE_K,"frequency_Hz":NOMINAL_FREQUENCY_HZ,"points":len(records),
                  "rate_sha256":sha256(DATA/"normalized/qste340tm_constant_amplitude.csv"),"control_sha256":sha256(CONTROL),
                  "policies":{p:{"capacity":POLICIES[p],"metrics":r["metrics"],"optimizer":r["optimizer"],"parameters":r["parameters"]} for p,r in results.items()}}
        (OUT/"comparison.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    return 0


if __name__ == "__main__": raise SystemExit(main())
