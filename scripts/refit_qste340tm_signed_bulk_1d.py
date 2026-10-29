#!/usr/bin/env python3
"""Paired QSTE340TM 1-D refit with and without signed bulk absorption."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.qste340tm_signed_bulk_1d as bulk
import scripts.refit_qste340tm_emission_owned_blunting as base
import scripts.refit_qste340tm_resolved_sloping_barrier as slope
from scripts.refit_qste340tm_restored_native import build_records


SOURCE = (
    base.OUT / "resolved_increment_1d/released_sloping_barrier_search/fit.json"
)
OUT = base.OUT / "resolved_increment_1d/signed_bulk_joint_refit_20261028"


def trajectory(vector: np.ndarray, records: list[dict], *, phases: int,
               advection_step_m: float) -> dict[str, np.ndarray]:
    _, plastic, blunting = base.unpack(slope.legacy_vector(vector))
    names = (
        "radius_m", "backstress_Pa", "mobile", "retained",
        "local_source_slip", "local_forward_slip", "wake_mobile",
        "wake_retained", "bulk_lower_signed", "bulk_upper_signed",
        "bulk_tip_opening", "bulk_opening_absorbed", "adaptive_chunks",
        "advection_substeps", "bulk_advection_substeps",
        "active_to_wake_mobile", "active_to_wake_retained",
        "wake_mobile_discarded", "wake_retained_discarded",
    )
    output = {name: np.zeros(len(records)) for name in names}
    output["stress"] = np.zeros((len(records), int(phases)))
    groups = {
        r: [i for i, row in enumerate(records) if float(row["R"]) == r]
        for r in slope.R_VALUES
    }
    for r_value, indices in groups.items():
        state = bulk.initial_state(plastic)
        prior_cycles = float(records[indices[0]]["cycles"])
        prior_crack = float(records[indices[0]]["crack_length_m"])
        prior_dk = float(records[indices[0]]["DeltaK_MPa_sqrt_m"])
        for position, index in enumerate(indices):
            row = records[index]
            cycles = float(row["cycles"])
            crack = float(row["crack_length_m"])
            dk = float(row["DeltaK_MPa_sqrt_m"])
            diagnostic: dict[str, float] = {"chunks": 0.0}
            if position:
                state, diagnostic = bulk.advance_interval(
                    0.5 * (prior_dk + dk), r_value, state,
                    max(cycles - prior_cycles, 0.0),
                    max(crack - prior_crack, 0.0), plastic, blunting,
                    n_phase=phases, radius_fraction_per_chunk=0.50,
                    maximum_chunk_cycles=1_000_000,
                    advection_step_m=advection_step_m,
                )
            stress, radius, backstress = bulk.stress_cycle(
                dk, r_value, state, blunting, phases
            )
            output["stress"][index] = stress
            output["radius_m"][index] = radius
            output["backstress_Pa"][index] = backstress
            output["mobile"][index] = state[bulk.MOBILE]
            output["retained"][index] = state[bulk.RETAINED]
            output["local_source_slip"][index] = state[bulk.LOCAL_SOURCE_SLIP]
            output["local_forward_slip"][index] = state[bulk.LOCAL_FORWARD_SLIP]
            output["wake_mobile"][index] = state[bulk.WAKE_MOBILE]
            output["wake_retained"][index] = state[bulk.WAKE_RETAINED]
            output["bulk_lower_signed"][index] = state[bulk.BULK_LOWER_SIGNED]
            output["bulk_upper_signed"][index] = state[bulk.BULK_UPPER_SIGNED]
            output["bulk_tip_opening"][index] = state[bulk.BULK_TIP_OPENING]
            for key in names:
                if key not in {
                    "radius_m", "backstress_Pa", "mobile", "retained",
                    "local_source_slip", "local_forward_slip", "wake_mobile",
                    "wake_retained", "bulk_lower_signed", "bulk_upper_signed",
                    "bulk_tip_opening",
                }:
                    output[key][index] = float(diagnostic.get(key, 0.0))
            prior_cycles, prior_crack, prior_dk = cycles, crack, dk
    return output


class Residual:
    def __init__(self, records: list[dict], phases: int, advection_step_m: float):
        self.records = records
        self.phases = phases
        self.advection_step_m = advection_step_m
        self.cache: dict[bytes, dict[str, np.ndarray]] = {}

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        try:
            values = np.asarray(vector, dtype=float)
            key = np.ascontiguousarray(values[5:]).tobytes()
            state = self.cache.get(key)
            if state is None:
                state = trajectory(
                    values, self.records, phases=self.phases,
                    advection_step_m=self.advection_step_m,
                )
                if len(self.cache) > 128:
                    self.cache.clear()
                self.cache[key] = state
            residual = slope.objective_residual(values, self.records, state)
            radius_penalty = max(
                0.0, float(np.max(state["radius_m"])) / 25.0e-6 - 1.0
            ) * 5.0
            density = (
                state["mobile"] + state["retained"]
                + state["wake_mobile"] + state["wake_retained"]
            )
            density_penalty = max(
                0.0, math.log10(max(float(np.max(density)) / 1.0e-12, 1.0)) - 18.0
            ) * 5.0
            return np.r_[residual, radius_penalty, density_penalty]
        except (FloatingPointError, OverflowError, RuntimeError, ValueError,
                np.linalg.LinAlgError):
            return np.full(2 * len(self.records) + 7, 1.0e6)


def profile_opening(candidate: np.ndarray, records: list[dict], *, phases: int,
                    advection_step_m: float, seed: int) -> tuple[float, np.ndarray]:
    """Globally profile opening coordinates on one complete bulk state history."""
    state = trajectory(
        candidate, records, phases=phases, advection_step_m=advection_step_m
    )
    indices = np.arange(5)
    fixed = np.asarray(candidate, dtype=float).copy()

    def assemble(values):
        vector = fixed.copy()
        vector[indices] = values
        return vector

    def objective(values):
        residual = slope.objective_residual(assemble(values), records, state)
        return float(np.dot(residual, residual))

    result = differential_evolution(
        objective, list(zip(slope.LOWER[indices], slope.UPPER[indices])),
        seed=seed, popsize=7, maxiter=32, polish=True, tol=1e-6,
    )
    return float(result.fun), assemble(result.x)


def write_predictions(path: Path, records: list[dict], prediction: np.ndarray,
                      state: dict[str, np.ndarray]) -> None:
    fields = list(records[0]) + ["predicted_dadn_m_per_cycle", "residual_log10"]
    fields += [key for key in state if key != "stress"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(records):
            observed = float(row["observed_dadn_m_per_cycle"])
            extra = {
                "predicted_dadn_m_per_cycle": float(prediction[index]),
                "residual_log10": math.log10(float(prediction[index]) / observed),
            }
            extra.update({
                key: float(value[index]) for key, value in state.items()
                if key != "stress"
            })
            writer.writerow(dict(row) | extra)


def plot(records: list[dict], no_bulk_prediction: np.ndarray,
         bulk_prediction: np.ndarray, bulk_state: dict[str, np.ndarray]) -> None:
    ratios = np.asarray([float(row["R"]) for row in records])
    dk = np.asarray([float(row["DeltaK_MPa_sqrt_m"]) for row in records])
    observed = np.asarray([float(row["observed_dadn_m_per_cycle"]) for row in records])
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), sharex=True, sharey=True)
    for axis, prediction, title in (
        (axes[0], no_bulk_prediction, "Established 1-D: tip plasticity"),
        (axes[1], bulk_prediction, "1-D: signed bulk absorption"),
    ):
        for r_value in slope.R_VALUES:
            mask = ratios == r_value
            order = np.argsort(dk[mask])
            color = slope.COLORS[r_value]
            axis.plot(dk[mask][order], observed[mask][order], color=color, lw=2.0,
                      label=f"R={r_value:.1f} experiment")
            axis.scatter(dk[mask], prediction[mask], facecolors="white",
                         edgecolors=color, s=18, linewidths=0.9,
                         label=f"R={r_value:.1f} model")
        axis.set(xscale="log", yscale="log", xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)",
                 title=title)
    axes[0].set_ylabel(r"$da/dN$ (m/cycle)")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=4, frameon=False, loc="upper center")
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(OUT / "paired_common_axis_fit.png", dpi=240, bbox_inches="tight")
    figure.savefig(OUT / "paired_common_axis_fit.svg", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(9.2, 8.0), sharex=True)
    for r_value in slope.R_VALUES:
        mask = ratios == r_value
        order = np.argsort(dk[mask])
        color = slope.COLORS[r_value]
        axes[0].plot(dk[mask][order], bulk_state["bulk_tip_opening"][mask][order],
                     color=color, lw=2, label=f"R={r_value:.1f}")
        axes[1].plot(dk[mask][order], bulk_state["radius_m"][mask][order] * 1e6,
                     color=color, lw=2)
    axes[0].set(ylabel="opening-effective absorbed count",
                title="Signed-bulk 1-D state histories")
    axes[1].set(xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)",
                ylabel=r"tip radius ($\mu$m)")
    axes[0].legend(frameon=False, ncol=4)
    figure.tight_layout()
    figure.savefig(OUT / "signed_bulk_state_histories.png", dpi=240,
                   bbox_inches="tight")
    figure.savefig(OUT / "signed_bulk_state_histories.svg", bbox_inches="tight")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-per-R", type=int, default=16)
    parser.add_argument("--fit-phases", type=int, default=8)
    parser.add_argument("--final-phases", type=int, default=48)
    parser.add_argument("--max-nfev", type=int, default=90)
    parser.add_argument("--advection-step-um", type=float, default=0.25)
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    parent = json.loads(SOURCE.read_text())
    parent_vector = np.asarray(parent["parameters_vector"], dtype=float)
    records = build_records("index_rank", premeasurement=True)
    support = base.select_support(records, args.support_per_R)
    advection_step_m = float(args.advection_step_um) * 1.0e-6
    residual = Residual(support, args.fit_phases, advection_step_m)

    starts = slope.mechanical_bank(parent_vector, count=20, seed=20261028)
    fitted = []
    profiles = []
    profiled_starts = []
    for index, (label, start) in enumerate(starts):
        profile_score, profiled = profile_opening(
            start, support, phases=args.fit_phases,
            advection_step_m=advection_step_m, seed=20261028 + index,
        )
        profiles.append({"label": label, "opening_profile_SSE": profile_score})
        profiled_starts.append((profile_score, label, profiled))
        print(f"[profile {label}] SSE={profile_score:.8g}", flush=True)
    # Mechanical coordinates are regime-like because the observed histories
    # contain integer cycle/crack increments.  Globally profile the bank first,
    # then spend derivative-based refinement only on the informative front.
    for _, label, profiled in sorted(profiled_starts)[:4]:
        solution = least_squares(
            residual, profiled, bounds=(slope.LOWER, slope.UPPER),
            max_nfev=args.max_nfev, x_scale="jac",
            ftol=2e-6, xtol=2e-6, gtol=2e-6,
        )
        score = float(np.dot(residual(solution.x), residual(solution.x)))
        print(f"[{label}] SSE={score:.8g} nfev={solution.nfev}", flush=True)
        fitted.append((score, label, solution))
    score, label, solution = min(fitted, key=lambda item: item[0])

    full_profile_score, final_vector = profile_opening(
        solution.x, records, phases=args.final_phases,
        advection_step_m=advection_step_m, seed=20261128,
    )
    bulk_state = trajectory(
        final_vector, records, phases=args.final_phases,
        advection_step_m=advection_step_m,
    )
    bulk_prediction, raw, _ = slope.prediction(final_vector, bulk_state)
    no_bulk_state = slope.state_trajectory(
        parent_vector, records, args.final_phases, advection_step_m
    )
    no_bulk_prediction, _, _ = slope.prediction(parent_vector, no_bulk_state)
    bulk_metrics = slope.metrics(records, bulk_prediction, bulk_state)
    no_bulk_metrics = slope.metrics(records, no_bulk_prediction, no_bulk_state)

    payload = {
        "schema": "qste340tm-signed-bulk-1d-joint-refit/1",
        "status": "executed_paired_four_R_refit",
        "source_fit": str(SOURCE),
        "source_fit_parameters": parent["parameters"],
        "selected_start": label,
        "parameters_vector": final_vector.tolist(),
        "parameters": slope.unpack(final_vector),
        "no_bulk_metrics": no_bulk_metrics,
        "signed_bulk_metrics": bulk_metrics,
        "support_points": len(support),
        "total_points": len(records),
        "fit_phases": args.fit_phases,
        "final_phases": args.final_phases,
        "advection_step_m": advection_step_m,
        "elapsed_s": time.time() - started,
        "multi_start": [
            {"label": item_label, "SSE": item_score,
             "nfev": int(item_solution.nfev)}
            for item_score, item_label, item_solution in sorted(fitted)
        ],
        "opening_profiles": profiles,
        "full_data_opening_profile_SSE": full_profile_score,
        "signed_bulk_owner": {
            "birth": "exact increment of existing gross forward-passage ledger",
            "face_signs": "equal-and-opposite Mode-I lower/upper face moments",
            "sweep": "1-exp(-da/process_zone_length)",
            "tip_projection": "positive lower plus negative upper outward-opening content",
            "blunting": "existing source_slip_blunting_coefficient only",
            "free_bulk_efficacy_parameter": False,
            "R_specific_parameters": False,
        },
        "barrier_over_kBT_range": [float(np.min(raw)), float(np.max(raw))],
        "limitations": [
            "two-face 1-D spatial moment, not a replacement for native 2-D GND fields",
            "experimental loading is treated through the reported DeltaK history; absolute CT force requires unresolved W/geometry",
            "frequency remains the user-authorized provisional 10 Hz",
        ],
    }
    (OUT / "fit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_predictions(OUT / "predictions_signed_bulk.csv", records,
                      bulk_prediction, bulk_state)
    write_predictions(OUT / "predictions_no_bulk.csv", records,
                      no_bulk_prediction, no_bulk_state)
    plot(records, no_bulk_prediction, bulk_prediction, bulk_state)
    print(json.dumps({
        "output": str(OUT),
        "no_bulk_RMSE": no_bulk_metrics["overall_rmse_log10_rate"],
        "signed_bulk_RMSE": bulk_metrics["overall_rmse_log10_rate"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

