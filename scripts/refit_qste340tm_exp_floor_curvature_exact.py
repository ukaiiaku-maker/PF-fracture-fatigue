#!/usr/bin/env python3
"""Corrected-integral QSTE340TM sensitivity with one shared EXP-floor exponent."""
from __future__ import annotations

import argparse, json, math, time
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import differential_evolution, least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.refit_qste340tm_capacity_policy import (
    DATA, LOWER, OUT, UPPER, build_records, evaluate, metrics, unpack,
)

OUTDIR = OUT / "continuous_birth_exact"
MODE = "continuous_birth_exact"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-iterations", type=int, default=12)
    ap.add_argument("--max-nfev", type=int, default=160)
    args = ap.parse_args()
    records = build_records(DATA)
    observed = np.log10([float(r["observed_dadn_m_per_cycle"]) for r in records])
    rv = np.asarray([float(r["R"]) for r in records])
    weights = np.empty(len(records))
    for r in sorted(set(rv)):
        weights[rv == r] = 1.0 / math.sqrt(np.sum(rv == r))
    base = np.asarray(json.loads((OUTDIR / "fit_uncapped_continuous_birth.json").read_text())["parameters_vector"])
    lo = np.r_[LOWER, math.log(0.1)]
    hi = np.r_[UPPER, math.log(10.0)]
    hi[9] = math.log1p(3000.0)

    def residual(x):
        state = evaluate(x, records, "uncapped", retention_timing=MODE)
        error = (np.log10(state["prediction"]) - observed) * weights
        valid = max(0.0, 0.25 - float(np.min(state["raw_opening_barrier_over_kBT"]))) * 10.0
        return np.r_[error, valid]

    starts = [(f"n_{n:g}", np.r_[base, math.log(n)]) for n in (0.25, 0.5, 1.0, 2.0, 4.0)]
    runs = []
    with ThreadPool(4) as pool:
        print("[curvature] bounded global acquisition", flush=True)
        de = differential_evolution(
            lambda x: float(np.dot(residual(x), residual(x))), list(zip(lo, hi)),
            seed=20261023, popsize=4, maxiter=args.global_iterations, polish=False,
            workers=pool.map, updating="deferred", tol=1e-5,
        )
        starts.append(("bounded_global", de.x))
        scored = sorted((float(np.dot(residual(x), residual(x))), name, x) for name, x in starts)
        for _, name, x0 in scored[:3]:
            print(f"[curvature] {name}", flush=True)
            started = time.time()
            sol = least_squares(
                residual, x0, bounds=(lo, hi), max_nfev=args.max_nfev,
                x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10,
                workers=pool.map,
            )
            state = evaluate(sol.x, records, "uncapped", retention_timing=MODE)
            met = metrics(records, state)
            runs.append((met["overall_rmse_log10_rate"], name, sol, met))
            (OUTDIR / f"checkpoint_curvature_{name}.json").write_text(json.dumps({
                "start": name, "parameters_vector": sol.x.tolist(), "parameters": unpack(sol.x),
                "metrics": met, "optimizer": {"success": bool(sol.success),
                "message": str(sol.message), "nfev": int(sol.nfev),
                "elapsed_s": time.time() - started}}, indent=2, sort_keys=True) + "\n")
            print(f"  RMSE={met['overall_rmse_log10_rate']:.6f} n={unpack(sol.x)['plastic_EXP_floor_n']:.5g} nfev={sol.nfev} success={sol.success}", flush=True)
        _, selected, sol, _ = min(runs, key=lambda x: x[0])
        print(f"[curvature] refine {selected}", flush=True)
        sol = least_squares(residual, sol.x, bounds=(lo, hi), max_nfev=args.max_nfev,
                            x_scale="jac", ftol=2e-12, xtol=2e-12, gtol=2e-12,
                            workers=pool.map)
    state = evaluate(sol.x, records, "uncapped", retention_timing=MODE)
    frozen = evaluate(sol.x, records, "uncapped", evolve=False, retention_timing=MODE)
    convergence = {str(n): metrics(records, evaluate(sol.x, records, "uncapped",
        retention_timing=MODE, waveform_steps=n)) for n in (32, 64, 128, 256, 512)}
    payload = {"schema": "qste340tm-continuous-birth-exp-floor-curvature/1",
        "opening_model": "single_smooth_broad", "retention_timing": MODE, "policy": "uncapped",
        "parameters_vector": sol.x.tolist(), "parameters": unpack(sol.x),
        "metrics": metrics(records, state), "frozen_metrics": metrics(records, frozen),
        "waveform_convergence": convergence, "selected_start": selected,
        "optimizer": {"success": bool(sol.success), "message": str(sol.message),
            "nfev": int(sol.nfev), "cost": float(sol.cost), "optimality": float(sol.optimality)},
        "state": {k: v.tolist() for k, v in state.items()},
        "frozen_prediction": frozen["prediction"].tolist()}
    (OUTDIR / "fit_uncapped_exp_floor_curvature.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"metrics": payload["metrics"], "parameters": payload["parameters"],
                      "optimizer": payload["optimizer"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
