#!/usr/bin/env python3
"""Joint sloping-floor opening/plasticity search with resolved tip advection.

The fixed-shape resolved-increment replay is retained as a control.  This
search releases the established single smooth fractional-EXP opening surface
and profiles it globally over diverse moving-state plasticity backgrounds
before joint local refinement.  No R-specific coordinate is present.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from multiprocessing.dummy import Pool as ThreadPool
from threading import Lock

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy.special import gammainc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.material_manifest import KB_EV_PER_K
from scripts.calibrate_qste340tm_opening_reference import (
    ATTEMPT_FREQUENCY_S, COOPERATIVE_HITS, COOPERATIVE_TAU_S,
    EVENT_LENGTH_M, NOMINAL_FREQUENCY_HZ, SMOOTHING_RATIO,
)
from scripts.fatigue_informed_joint_barrier_v3_v10230 import SmoothBroadBasis
import scripts.refit_qste340tm_emission_owned_blunting as base


OUT = base.OUT / "resolved_increment_1d/released_sloping_barrier_search"
TEMPERATURE_K = base.TEMPERATURE_K
R_VALUES = base.R_VALUES
COLORS = base.COLORS

# floor, amplitude, log sigma_b[GPa], log exponent, backstress fraction,
# emission G0, log plastic sigc[GPa], transport G0, log recovery,
# log1p mobile0, log1p retained0, log source blunt, log forward blunt,
# log blunting/wake length[um].
LOWER = np.array([
    6.0, 0.0, math.log(0.05), math.log(0.08), 0.0,
    0.30, math.log(0.10), 0.30, math.log(1.0e-8),
    0.0, 0.0, math.log(1.0e-3), math.log(1.0e-3), math.log(0.10),
])
UPPER = np.array([
    30.0, 140.0, math.log(100.0), math.log(1.0), 2.0,
    4.0, math.log(30.0), 5.0, math.log(1.0e4),
    math.log1p(5000.0), math.log1p(5000.0),
    math.log(1.0e3), math.log(1.0e3), math.log(30.0),
])


def unpack(vector: np.ndarray) -> dict[str, float]:
    x = np.asarray(vector, dtype=float)
    return {
        "opening_floor_over_kBT": float(x[0]),
        "opening_amplitude_over_kBT": float(x[1]),
        "opening_sigma_b_Pa": float(math.exp(x[2]) * 1.0e9),
        "opening_exponent": float(math.exp(x[3])),
        "opening_smoothing_ratio": SMOOTHING_RATIO,
        "cleavage_backstress_fraction": float(x[4]),
        "emission_G0_298_eV": float(x[5]),
        "plastic_sigc_298_Pa": float(math.exp(x[6]) * 1.0e9),
        "transport_G0_298_eV": float(x[7]),
        "retained_recovery_per_s": float(math.exp(x[8])),
        "initial_mobile_count": float(math.expm1(x[9])),
        "initial_retained_count": float(math.expm1(x[10])),
        "source_slip_blunting_coefficient": float(math.exp(x[11])),
        "forward_slip_blunting_coefficient": float(math.exp(x[12])),
        "blunting_length_m": float(math.exp(x[13]) * 1.0e-6),
    }


def pack(parameters: dict[str, float]) -> np.ndarray:
    reference = base.opening_reference()
    return np.array([
        parameters["opening_floor_over_kBT"],
        parameters.get("opening_amplitude_over_kBT", reference["amplitude"]),
        math.log(parameters.get("opening_sigma_b_Pa", reference["sigma_b_Pa"]) / 1.0e9),
        math.log(parameters.get("opening_exponent", reference["exponent"])),
        parameters["cleavage_backstress_fraction"],
        parameters["emission_G0_298_eV"],
        math.log(parameters["plastic_sigc_298_Pa"] / 1.0e9),
        parameters["transport_G0_298_eV"],
        math.log(parameters["retained_recovery_per_s"]),
        math.log1p(parameters["initial_mobile_count"]),
        math.log1p(parameters["initial_retained_count"]),
        math.log(parameters["source_slip_blunting_coefficient"]),
        math.log(parameters["forward_slip_blunting_coefficient"]),
        math.log(parameters["blunting_length_m"] / 1.0e-6),
    ])


def legacy_vector(vector: np.ndarray) -> np.ndarray:
    """Map released opening coordinates onto the existing state evaluator."""
    x = np.asarray(vector, dtype=float)
    return np.r_[x[0], x[4:]]


def state_trajectory(vector: np.ndarray, records: list[dict], phases: int,
                     advection_step_m: float) -> dict[str, np.ndarray]:
    return base.trajectory(
        legacy_vector(vector), records, phases=phases,
        radius_fraction_per_chunk=0.50, maximum_chunk_cycles=1_000_000,
        resolved_population_advection=True,
        advection_step_m=advection_step_m,
    )


def prediction(vector: np.ndarray, state: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = unpack(vector)
    basis = SmoothBroadBasis(
        p["opening_sigma_b_Pa"], p["opening_exponent"], SMOOTHING_RATIO
    )
    opening_stress = np.maximum(
        state["stress"]
        - p["cleavage_backstress_fraction"] * state["backstress_Pa"][:, None],
        0.0,
    )
    raw = (
        p["opening_floor_over_kBT"]
        + p["opening_amplitude_over_kBT"] * basis.value(opening_stress)
    )
    embrittlement = np.asarray([base.embrittlement_eV(value) for value in state["retained"]])
    raw -= embrittlement[:, None] / (KB_EV_PER_K * TEMPERATURE_K)
    elementary = ATTEMPT_FREQUENCY_S * np.exp(-np.clip(raw, 0.0, 700.0))
    completed = gammainc(
        COOPERATIVE_HITS,
        np.minimum(elementary * COOPERATIVE_TAU_S, 1.0e12),
    ) / COOPERATIVE_TAU_S
    rate = EVENT_LENGTH_M * np.mean(completed, axis=1) / NOMINAL_FREQUENCY_HZ
    return np.maximum(rate, 1.0e-300), raw, opening_stress


def objective_residual(vector: np.ndarray, records: list[dict], state: dict[str, np.ndarray]) -> np.ndarray:
    observed = np.log10([float(row["observed_dadn_m_per_cycle"]) for row in records])
    ratios = np.asarray([float(row["R"]) for row in records])
    delta_k = np.asarray([float(row["DeltaK_MPa_sqrt_m"]) for row in records])
    predicted, raw, _ = prediction(vector, state)
    error = np.log10(predicted) - observed
    pieces = []
    slopes = []
    centered = error.copy()
    for r_value in R_VALUES:
        mask = ratios == r_value
        centered[mask] -= np.mean(error[mask])
        weight = 1.0 / math.sqrt(max(int(np.sum(mask)), 1))
        pieces.extend((error[mask] * weight).tolist())
        pieces.extend((0.50 * centered[mask] * weight).tolist())
        observed_slope = np.polyfit(np.log10(delta_k[mask]), observed[mask], 1)[0]
        predicted_slope = np.polyfit(np.log10(delta_k[mask]), np.log10(predicted[mask]), 1)[0]
        slopes.append(0.50 * (predicted_slope - observed_slope))
    validity = max(0.0, 0.25 - float(np.min(raw))) * 10.0
    return np.r_[pieces, slopes, validity]


def full_residual(vector: np.ndarray, records: list[dict], phases: int,
                  advection_step_m: float) -> np.ndarray:
    try:
        state = state_trajectory(vector, records, phases, advection_step_m)
        residual = objective_residual(vector, records, state)
        radius_penalty = max(0.0, float(np.max(state["radius_m"])) / 25.0e-6 - 1.0) * 5.0
        total = state["mobile"] + state["retained"] + state["wake_mobile"] + state["wake_retained"]
        density_penalty = max(0.0, math.log10(max(float(np.max(total)) / 1.0e-12, 1.0)) - 18.0) * 5.0
        backstress_penalty = max(0.0, float(np.max(state["backstress_Pa"])) / 20.0e9 - 1.0) * 5.0
        return np.r_[residual, radius_penalty, density_penalty, backstress_penalty]
    except (FloatingPointError, OverflowError, RuntimeError, ValueError, np.linalg.LinAlgError):
        return np.full(2 * len(records) + 8, 1.0e6)


class CachedFullResidual:
    """Avoid replaying state when only opening coordinates are perturbed."""
    def __init__(self, records: list[dict], phases: int, advection_step_m: float):
        self.records = records
        self.phases = phases
        self.advection_step_m = advection_step_m
        self.cache: dict[bytes, dict[str, np.ndarray]] = {}
        self.lock = Lock()

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        try:
            values = np.asarray(vector, dtype=float)
            # Coordinates 0:5 affect only the opening consumer, not state evolution.
            key = np.ascontiguousarray(values[5:]).tobytes()
            with self.lock:
                state = self.cache.get(key)
            if state is None:
                state = state_trajectory(values, self.records, self.phases, self.advection_step_m)
                with self.lock:
                    if len(self.cache) > 96:
                        self.cache.clear()
                    self.cache[key] = state
            residual = objective_residual(values, self.records, state)
            radius_penalty = max(0.0, float(np.max(state["radius_m"])) / 25.0e-6 - 1.0) * 5.0
            total = state["mobile"] + state["retained"] + state["wake_mobile"] + state["wake_retained"]
            density_penalty = max(0.0, math.log10(max(float(np.max(total)) / 1.0e-12, 1.0)) - 18.0) * 5.0
            backstress_penalty = max(0.0, float(np.max(state["backstress_Pa"])) / 20.0e9 - 1.0) * 5.0
            return np.r_[residual, radius_penalty, density_penalty, backstress_penalty]
        except (FloatingPointError, OverflowError, RuntimeError, ValueError, np.linalg.LinAlgError):
            return np.full(2 * len(self.records) + 8, 1.0e6)


def mechanical_bank(control: np.ndarray, count: int, seed: int) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    bank = [("fixed_shape_control", control.copy())]
    # The historical generic starts provide mechanism-scale diversity.
    for label, old in base.starts():
        candidate = control.copy()
        candidate[4:] = old[1:]
        bank.append((label, np.clip(candidate, LOWER, UPPER)))
    # Broad log-space state acquisition is deliberate: the corrected moving-tip
    # problem can change regime when recovery, transport, or blunting persistence
    # crosses the crack-advance exposure time.
    scales = np.array([0, 0, 0, 0, .50, .70, 1.00, .70, 2.0, 2.0, 2.0, 2.0, 2.0, 1.2])
    for index in range(max(int(count), 0)):
        candidate = control + rng.normal(size=control.size) * scales
        # Barrier coordinates are globally profiled below; keep their seed neutral.
        candidate[:4] = control[:4]
        bank.append((f"perturb_{index:03d}", np.clip(candidate, LOWER, UPPER)))
    return bank


def profile_barrier(candidate: np.ndarray, records: list[dict], phases: int,
                    advection_step_m: float, iterations: int, seed: int) -> tuple[float, np.ndarray]:
    state = state_trajectory(candidate, records, phases, advection_step_m)
    indices = np.array([0, 1, 2, 3, 4])
    fixed = candidate.copy()

    def assemble(values):
        vector = fixed.copy()
        vector[indices] = values
        return vector

    bounds = list(zip(LOWER[indices], UPPER[indices]))
    result = differential_evolution(
        lambda values: float(np.dot(
            objective_residual(assemble(values), records, state),
            objective_residual(assemble(values), records, state),
        )),
        bounds, seed=seed, popsize=6, maxiter=max(int(iterations), 1),
        polish=False, tol=1.0e-5,
    )
    vector = assemble(result.x)
    residual = objective_residual(vector, records, state)
    return float(np.dot(residual, residual)), vector


def metrics(records: list[dict], predicted: np.ndarray, state: dict[str, np.ndarray]) -> dict:
    return base.metrics(records, predicted, state)


def barrier_diagnostics(vector: np.ndarray, opening_stress: np.ndarray) -> dict:
    p = unpack(vector)
    lo = max(float(np.min(opening_stress)), 0.0)
    hi = float(np.max(opening_stress))
    stress = np.linspace(lo, hi, 1000)
    basis = SmoothBroadBasis(p["opening_sigma_b_Pa"], p["opening_exponent"], SMOOTHING_RATIO)
    raw = p["opening_floor_over_kBT"] + p["opening_amplitude_over_kBT"] * basis.value(stress)
    derivative = p["opening_amplitude_over_kBT"] * basis.derivative(stress)
    upper = stress >= np.quantile(stress, 0.75)
    return {
        "sampled_local_stress_Pa": [lo, hi],
        "barrier_over_kBT_range_without_state_shift": [float(np.min(raw)), float(np.max(raw))],
        "derivative_over_kBT_per_Pa_range": [float(np.min(derivative)), float(np.max(derivative))],
        "upper_quartile_mean_derivative_over_kBT_per_GPa": float(np.mean(derivative[upper]) * 1.0e9),
        "monotone_nonincreasing": bool(np.all(derivative <= 1.0e-15)),
        "long_tail_fraction_at_max_stress": float(basis.value(hi)),
    }


def write_outputs(vector: np.ndarray, records: list[dict], state: dict[str, np.ndarray],
                  predicted: np.ndarray, raw: np.ndarray, opening_stress: np.ndarray,
                  payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    fields = list(records[0]) + ["predicted_dadn_m_per_cycle", "residual_log10"]
    with (OUT / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row, value in zip(records, predicted):
            observed = float(row["observed_dadn_m_per_cycle"])
            writer.writerow(dict(row) | {
                "predicted_dadn_m_per_cycle": value,
                "residual_log10": math.log10(value / observed),
            })

    ratios = np.asarray([float(row["R"]) for row in records])
    delta_k = np.asarray([float(row["DeltaK_MPa_sqrt_m"]) for row in records])
    observed = np.asarray([float(row["observed_dadn_m_per_cycle"]) for row in records])
    figure, axis = plt.subplots(figsize=(9.2, 6.3))
    for r_value in R_VALUES:
        mask = ratios == r_value
        order = np.argsort(delta_k[mask])
        color = COLORS[r_value]
        axis.plot(delta_k[mask][order], observed[mask][order], color=color, lw=2.1, label=f"R={r_value:.1f} experiment")
        axis.scatter(delta_k[mask], predicted[mask], facecolors="white", edgecolors=color, s=22, linewidths=1.0, label=f"R={r_value:.1f} fit")
    axis.set(xscale="log", yscale="log", xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)", ylabel=r"$da/dN$ (m/cycle)", title="Resolved-state joint sloping-floor barrier search")
    axis.grid(False); axis.legend(ncol=2, frameon=False, fontsize=8)
    figure.tight_layout(); figure.savefig(OUT / "combined_data_fit.png", dpi=240, bbox_inches="tight"); plt.close(figure)

    p = unpack(vector)
    maximum = max(float(np.max(opening_stress)), 1.0e9)
    stress = np.linspace(0.0, 1.05 * maximum, 800)
    basis = SmoothBroadBasis(p["opening_sigma_b_Pa"], p["opening_exponent"], SMOOTHING_RATIO)
    barrier = (p["opening_floor_over_kBT"] + p["opening_amplitude_over_kBT"] * basis.value(stress)) * KB_EV_PER_K * TEMPERATURE_K
    figure, axis = plt.subplots(figsize=(8.4, 5.4))
    axis.plot(stress / 1.0e9, barrier, lw=2.2)
    axis.axvspan(float(np.min(opening_stress)) / 1.0e9, float(np.max(opening_stress)) / 1.0e9, color="#999999", alpha=0.15, label="sampled local stress")
    axis.set(xlabel="local opening stress (GPa)", ylabel=r"$G^*_{open}$ (eV)", title="Selected fractional-EXP sloping-floor barrier at 298.15 K")
    axis.grid(False); axis.legend(frameon=False); figure.tight_layout()
    figure.savefig(OUT / "Gstar_vs_stress.png", dpi=240, bbox_inches="tight"); plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-per-R", type=int, default=12)
    parser.add_argument("--fit-phases", type=int, default=6)
    parser.add_argument("--final-phases", type=int, default=32)
    parser.add_argument("--mechanical-perturbations", type=int, default=12)
    parser.add_argument("--barrier-global-iterations", type=int, default=18)
    parser.add_argument("--joint-starts", type=int, default=4)
    parser.add_argument("--joint-max-nfev", type=int, default=24)
    parser.add_argument("--advection-step-um", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    all_records = base.build_records("index_rank", premeasurement=True)
    support = base.select_support(all_records, args.support_per_R)
    control_payload = json.loads((base.OUT / "resolved_increment_1d/fit.json").read_text())
    control = np.clip(pack(control_payload["parameters"]), LOWER, UPPER)
    advection_step_m = args.advection_step_um * 1.0e-6

    profiled = []
    started = time.time()
    for index, (label, candidate) in enumerate(mechanical_bank(control, args.mechanical_perturbations, args.seed)):
        score, vector = profile_barrier(
            candidate, support, args.fit_phases, advection_step_m,
            args.barrier_global_iterations, args.seed + index,
        )
        profiled.append((score, label, vector))
        print(f"[profile] {label} score={score:.8g}", flush=True)
    profiled.sort(key=lambda item: item[0])
    (OUT / "profiled_bank.json").write_text(json.dumps([
        {"profile_score": score, "label": label, "parameters_vector": vector.tolist(), "parameters": unpack(vector)}
        for score, label, vector in profiled
    ], indent=2, sort_keys=True) + "\n")

    refinements = []
    with ThreadPool(max(int(args.workers), 1)) as pool:
        for score, label, initial in profiled[:max(args.joint_starts, 1)]:
            print(f"[joint] {label} profile={score:.8g}", flush=True)
            cached = CachedFullResidual(support, args.fit_phases, advection_step_m)
            solution = least_squares(
                cached, initial, bounds=(LOWER, UPPER), max_nfev=args.joint_max_nfev,
                x_scale="jac", ftol=2.0e-7, xtol=2.0e-7, gtol=2.0e-7,
                workers=pool.map,
            )
            state = state_trajectory(solution.x, support, args.fit_phases, advection_step_m)
            pred, _, _ = prediction(solution.x, state)
            observed = np.asarray([float(row["observed_dadn_m_per_cycle"]) for row in support])
            rmse = float(np.sqrt(np.mean(np.log10(pred / observed) ** 2)))
            refinements.append((rmse, label, solution))
            checkpoint = {
                "label": label, "profile_score": score, "support_RMSE": rmse,
                "parameters_vector": solution.x.tolist(), "parameters": unpack(solution.x),
                "optimizer": {"success": bool(solution.success), "message": str(solution.message), "nfev": int(solution.nfev)},
            }
            (OUT / f"checkpoint_{label}.json").write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")
            print(f"  support_RMSE={rmse:.8g} nfev={solution.nfev}", flush=True)

    # Support is acquisition only.  Select the reported material on a complete
    # same-protocol replay so a sparse support cannot hide curvature between
    # acquisition points.
    complete_refinements = []
    all_observed = np.asarray([float(row["observed_dadn_m_per_cycle"]) for row in all_records])
    for support_rmse, label, solution in refinements:
        complete_state = state_trajectory(solution.x, all_records, args.final_phases, advection_step_m)
        complete_predicted, _, _ = prediction(solution.x, complete_state)
        complete_rmse = float(np.sqrt(np.mean(np.log10(complete_predicted / all_observed) ** 2)))
        complete_refinements.append((complete_rmse, label, solution, support_rmse))
        print(f"[complete] {label} RMSE={complete_rmse:.8g}", flush=True)
    _, selected_label, selected, selected_support_rmse = min(complete_refinements, key=lambda item: item[0])
    final_vector = selected.x.copy()
    final_state = state_trajectory(final_vector, all_records, args.final_phases, advection_step_m)
    opening_indices = np.array([0, 1, 2, 3, 4])
    def assemble_final_opening(values):
        vector = final_vector.copy()
        vector[opening_indices] = values
        return vector
    polish_global = differential_evolution(
        lambda values: float(np.dot(
            objective_residual(assemble_final_opening(values), all_records, final_state),
            objective_residual(assemble_final_opening(values), all_records, final_state),
        )),
        list(zip(LOWER[opening_indices], UPPER[opening_indices])),
        seed=args.seed + 10000, popsize=8, maxiter=32, polish=False, tol=1.0e-7,
    )
    polish_local = least_squares(
        lambda values: objective_residual(assemble_final_opening(values), all_records, final_state),
        polish_global.x, bounds=(LOWER[opening_indices], UPPER[opening_indices]),
        max_nfev=200, x_scale="jac", ftol=1.0e-10, xtol=1.0e-10, gtol=1.0e-10,
    )
    final_vector = assemble_final_opening(polish_local.x)
    predicted, raw, opening_stress = prediction(final_vector, final_state)
    result_metrics = metrics(all_records, predicted, final_state)
    diagnostics = barrier_diagnostics(final_vector, opening_stress)
    payload = {
        "schema": "qste340tm-resolved-increment-sloping-barrier-search/1",
        "status": "executed_global_profile_and_joint_refinement",
        "opening_family": "single_smooth_fractional_EXP_sloping_floor",
        "parameters": unpack(final_vector),
        "parameters_vector": final_vector.tolist(),
        "metrics": result_metrics,
        "barrier_diagnostics": diagnostics,
        "raw_state_shifted_barrier_over_kBT_range": [float(np.min(raw)), float(np.max(raw))],
        "control": {
            "identity": "resolved_increment_fixed_opening_shape",
            "overall_rmse_log10_rate": control_payload["metrics"]["overall_rmse_log10_rate"],
            "path": str(base.OUT / "resolved_increment_1d/fit.json"),
        },
        "search": {
            "support_points": len(support), "fit_phases": args.fit_phases,
            "final_phases": args.final_phases,
            "mechanical_backgrounds_profiled": len(profiled),
            "barrier_global_iterations": args.barrier_global_iterations,
            "joint_refinements": len(refinements),
            "selected_support_RMSE": selected_support_rmse,
            "complete_refinement_RMSE": [
                {"label": label, "complete_RMSE": rmse, "support_RMSE": support_rmse}
                for rmse, label, _solution, support_rmse in complete_refinements
            ],
            "selected_start": selected_label,
            "selected_optimizer": {
                "success": bool(selected.success), "message": str(selected.message),
                "nfev": int(selected.nfev), "cost": float(selected.cost),
            },
            "full_data_barrier_polish": {
                "global_iterations": 32,
                "local_success": bool(polish_local.success),
                "local_message": str(polish_local.message),
                "local_nfev": int(polish_local.nfev),
            },
            "profiled_bank": [
                {"label": label, "profile_score": score, "parameters": unpack(vector)}
                for score, label, vector in profiled
            ],
            "elapsed_s": time.time() - started,
        },
        "constraints": {
            "R_specific_parameters": False,
            "opening_barrier_positive_and_monotone": diagnostics["monotone_nonincreasing"],
            "resolved_population_advection": True,
            "empirical_advection_fraction": False,
            "opening_smoothing_ratio_fixed": SMOOTHING_RATIO,
        },
    }
    write_outputs(final_vector, all_records, final_state, predicted, raw, opening_stress, payload)
    print(json.dumps({"output": str(OUT), "metrics": result_metrics, "parameters": unpack(final_vector), "barrier_diagnostics": diagnostics}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
