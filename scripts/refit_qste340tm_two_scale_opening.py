#!/usr/bin/env python3
"""Nested QSTE340TM comparison with two additive smooth-broad opening scales."""
from __future__ import annotations

import argparse
import json
import math
import time
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

MODE = "continuous_birth_exact"
MODEL = "two_scale_smooth_broad"
OUTDIR = OUT / "continuous_birth_exact"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-iterations", type=int, default=12)
    parser.add_argument("--max-nfev", type=int, default=160)
    args = parser.parse_args()
    records = build_records(DATA)
    observed = np.log10([float(row["observed_dadn_m_per_cycle"]) for row in records])
    r_values = np.asarray([float(row["R"]) for row in records])
    weights = np.empty(len(records))
    for r_value in sorted(set(r_values)):
        weights[r_values == r_value] = 1.0 / math.sqrt(np.sum(r_values == r_value))

    primary = json.loads((OUTDIR / "fit_uncapped_continuous_birth.json").read_text())
    base = np.asarray(primary["parameters_vector"], dtype=float)
    lower = np.r_[LOWER, 0.0, math.log(0.1), math.log(0.1)]
    upper = np.r_[UPPER, 100.0, math.log(100.0), math.log(10.0)]
    upper[9] = math.log1p(3000.0)

    def residual(vector: np.ndarray) -> np.ndarray:
        state = evaluate(
            vector, records, "uncapped", retention_timing=MODE,
            opening_model=MODEL,
        )
        error = (np.log10(state["prediction"]) - observed) * weights
        validity = max(0.0, 0.25 - float(np.min(state["raw_opening_barrier_over_kBT"]))) * 10.0
        return np.r_[error, validity]

    starts = []
    for amplitude, sigma_gpa, exponent in (
        (0.2, 0.5, 0.5), (2.0, 2.0, 1.0), (10.0, 8.0, 2.0),
        (30.0, 20.0, 0.5),
    ):
        starts.append((
            f"secondary_{amplitude:g}_{sigma_gpa:g}_{exponent:g}",
            np.r_[base, amplitude, math.log(sigma_gpa), math.log(exponent)],
        ))
    runs = []
    with ThreadPool(4) as pool:
        print("[two-scale] bounded global acquisition", flush=True)
        global_solution = differential_evolution(
            lambda x: float(np.dot(residual(x), residual(x))),
            list(zip(lower, upper)), seed=20261022, popsize=4,
            maxiter=args.global_iterations, polish=False, workers=pool.map,
            updating="deferred", tol=1.0e-5,
        )
        starts.append(("bounded_global", global_solution.x))
        scored = sorted((float(np.dot(residual(x), residual(x))), name, x) for name, x in starts)
        for _, name, initial in scored[:3]:
            print(f"[two-scale] {name}", flush=True)
            started = time.time()
            solution = least_squares(
                residual, initial, bounds=(lower, upper), max_nfev=args.max_nfev,
                x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10,
                workers=pool.map,
            )
            state = evaluate(
                solution.x, records, "uncapped", retention_timing=MODE,
                opening_model=MODEL,
            )
            result_metrics = metrics(records, state)
            runs.append((result_metrics["overall_rmse_log10_rate"], name, solution, result_metrics))
            checkpoint = {
                "start": name, "parameters_vector": solution.x.tolist(),
                "parameters": unpack(solution.x, MODEL), "metrics": result_metrics,
                "optimizer": {"success": bool(solution.success), "message": str(solution.message),
                              "nfev": int(solution.nfev), "elapsed_s": time.time() - started},
            }
            (OUTDIR / f"checkpoint_two_scale_{name}.json").write_text(
                json.dumps(checkpoint, indent=2, sort_keys=True) + "\n"
            )
            print(f"  RMSE={runs[-1][0]:.6f} nfev={solution.nfev} success={solution.success}", flush=True)
        _, selected_start, solution, _ = min(runs, key=lambda item: item[0])
        print(f"[two-scale] refine {selected_start}", flush=True)
        solution = least_squares(
            residual, solution.x, bounds=(lower, upper), max_nfev=args.max_nfev,
            x_scale="jac", ftol=2e-12, xtol=2e-12, gtol=2e-12,
            workers=pool.map,
        )

    state = evaluate(solution.x, records, "uncapped", retention_timing=MODE, opening_model=MODEL)
    frozen = evaluate(solution.x, records, "uncapped", evolve=False, retention_timing=MODE, opening_model=MODEL)
    convergence = {str(steps): metrics(records, evaluate(
        solution.x, records, "uncapped", retention_timing=MODE,
        waveform_steps=steps, opening_model=MODEL,
    )) for steps in (32, 64, 128, 256, 512)}
    payload = {
        "schema": "qste340tm-continuous-birth-two-scale-opening/1",
        "opening_model": MODEL, "retention_timing": MODE, "policy": "uncapped",
        "parameters_vector": solution.x.tolist(), "parameters": unpack(solution.x, MODEL),
        "metrics": metrics(records, state), "frozen_metrics": metrics(records, frozen),
        "waveform_convergence": convergence,
        "optimizer": {"success": bool(solution.success), "message": str(solution.message),
                      "nfev": int(solution.nfev), "cost": float(solution.cost),
                      "optimality": float(solution.optimality)},
        "selected_start": selected_start,
        "state": {key: value.tolist() for key, value in state.items()},
        "frozen_prediction": frozen["prediction"].tolist(),
    }
    (OUTDIR / "fit_uncapped_two_scale_opening.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"metrics": payload["metrics"], "parameters": payload["parameters"],
                      "optimizer": payload["optimizer"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
