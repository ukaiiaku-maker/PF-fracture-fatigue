#!/usr/bin/env python3
"""Equal-effort no-bulk control for the signed-bulk QSTE340TM 1-D refit.

The accepted signed-bulk refit compared its fresh 14-coordinate optimum with
the *unrefitted* parent vector evaluated without bulk.  This control repeats
the identical pipeline (same parent, 20-start mechanical bank and seed,
opening profiling, four least-squares refinements, final 48-phase opening
profile) with the established no-bulk state trajectory.  Only the state owner
differs, so the RMSE difference isolates the signed-bulk coordinate.

The accepted signed-bulk evidence is read, never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.refit_qste340tm_signed_bulk_1d as sb
import scripts.refit_qste340tm_emission_owned_blunting as base
import scripts.refit_qste340tm_resolved_sloping_barrier as slope
from scripts.refit_qste340tm_restored_native import build_records

ACCEPTED = sb.OUT / "fit.json"
OUT = base.OUT / "resolved_increment_1d/no_bulk_equal_effort_control_20261028"
# Vector slots in the order documented beside slope.LOWER/UPPER.
VECTOR_NAMES = (
    "opening_floor_over_kBT", "opening_amplitude_over_kBT", "log_opening_sigma_b_GPa",
    "log_opening_exponent", "cleavage_backstress_fraction", "emission_G0_298_eV",
    "log_plastic_sigc_GPa", "transport_G0_298_eV", "log_retained_recovery_per_s",
    "log1p_initial_mobile_count", "log1p_initial_retained_count",
    "log_source_slip_blunting_coefficient", "log_forward_slip_blunting_coefficient",
    "log_blunting_length_um",
)


def no_bulk_trajectory(vector, records, *, phases, advection_step_m):
    return slope.state_trajectory(vector, records, phases, advection_step_m)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    accepted = json.loads(ACCEPTED.read_text())
    parser.add_argument("--support-per-R", type=int,
                        default=int(accepted["support_points"]) // len(slope.R_VALUES))
    parser.add_argument("--fit-phases", type=int, default=int(accepted["fit_phases"]))
    parser.add_argument("--final-phases", type=int, default=int(accepted["final_phases"]))
    parser.add_argument("--max-nfev", type=int, default=90)
    parser.add_argument("--advection-step-um", type=float,
                        default=1.0e6 * float(accepted["advection_step_m"]))
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    # Every sb helper (Residual, profile_opening) resolves the module-global
    # trajectory, so this swaps the state owner and nothing else.
    sb.trajectory = no_bulk_trajectory
    started = time.time()
    parent = json.loads(sb.SOURCE.read_text())
    parent_vector = np.asarray(parent["parameters_vector"], dtype=float)
    records = build_records("index_rank", premeasurement=True)
    support = base.select_support(records, args.support_per_R)
    advection_step_m = float(args.advection_step_um) * 1.0e-6
    residual = sb.Residual(support, args.fit_phases, advection_step_m)

    starts = slope.mechanical_bank(parent_vector, count=20, seed=20261028)
    profiled_starts, profiles = [], []
    for index, (label, start) in enumerate(starts):
        score, profiled = sb.profile_opening(
            start, support, phases=args.fit_phases,
            advection_step_m=advection_step_m, seed=20261028 + index,
        )
        profiles.append({"label": label, "opening_profile_SSE": score})
        profiled_starts.append((score, label, profiled))
        print(f"[profile {label}] SSE={score:.8g}", flush=True)
    fitted = []
    for _, label, profiled in sorted(profiled_starts, key=lambda item: item[0])[:4]:
        solution = least_squares(
            residual, profiled, bounds=(slope.LOWER, slope.UPPER),
            max_nfev=args.max_nfev, x_scale="jac",
            ftol=2e-6, xtol=2e-6, gtol=2e-6,
        )
        score = float(np.dot(residual(solution.x), residual(solution.x)))
        print(f"[{label}] SSE={score:.8g} nfev={solution.nfev}", flush=True)
        fitted.append((score, label, solution))
    score, label, solution = min(fitted, key=lambda item: item[0])
    full_profile_score, final_vector = sb.profile_opening(
        solution.x, records, phases=args.final_phases,
        advection_step_m=advection_step_m, seed=20261128,
    )
    state = no_bulk_trajectory(final_vector, records, phases=args.final_phases,
                               advection_step_m=advection_step_m)
    prediction, _, _ = slope.prediction(final_vector, state)
    metrics = slope.metrics(records, prediction, state)
    at_bound = [
        name for name, value, lo, hi in zip(
            VECTOR_NAMES, final_vector, slope.LOWER, slope.UPPER)
        if np.isclose(value, lo, rtol=0, atol=1e-6 * max(abs(hi - lo), 1e-30))
        or np.isclose(value, hi, rtol=0, atol=1e-6 * max(abs(hi - lo), 1e-30))
    ]
    payload = {
        "schema": "qste340tm-no-bulk-equal-effort-control/1",
        "purpose": "same optimizer and starts as the accepted signed-bulk refit; no-bulk state owner",
        "accepted_signed_bulk_fit": str(ACCEPTED),
        "accepted_signed_bulk_fit_sha256": hashlib.sha256(ACCEPTED.read_bytes()).hexdigest(),
        "source_fit": str(sb.SOURCE),
        "settings": vars(args) | {"bank_count": 20, "bank_seed": 20261028},
        "selected_start": label,
        "parameters_vector": final_vector.tolist(),
        "parameters": slope.unpack(final_vector),
        "coordinates_at_bounds": at_bound,
        "metrics": metrics,
        "comparison_overall_rmse_log10": {
            "unrefit_parent_no_bulk": accepted["no_bulk_metrics"]["overall_rmse_log10_rate"],
            "equal_effort_no_bulk": metrics["overall_rmse_log10_rate"],
            "signed_bulk": accepted["signed_bulk_metrics"]["overall_rmse_log10_rate"],
        },
        "multi_start": [{"label": l, "SSE": s, "nfev": int(x.nfev)}
                        for s, l, x in sorted(fitted, key=lambda item: item[0])],
        "opening_profiles": profiles,
        "full_data_opening_profile_SSE": full_profile_score,
        "elapsed_s": time.time() - started,
    }
    (OUT / "fit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    sb.write_predictions(OUT / "predictions_no_bulk_refit.csv", records, prediction, state)
    print(json.dumps(payload["comparison_overall_rmse_log10"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
