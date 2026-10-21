#!/usr/bin/env python3
"""Joint QSTE340TM opening/EXP-floor analytical calibration at 25 C.

This is a state-informed analytical reduction, not a native 1-D result.  It
replays each constant-amplitude record in source order using the measured cycle
coordinate from the paired crack history.  The state equations are the scalar
counterparts of ``FrontEngine`` and ``FatigueCycleHazardController``:

* EXP-floor emission and series Peierls/Taylor escape hazards;
* escape-limited retention with a finite retained-count saturation;
* retained-dislocation blunting, back stress, and stored-energy embrittlement;
* the existing smooth-broad opening barrier and cooperative completion law.

Only room-temperature effective plastic barrier levels/stress scale are fitted.
The established EXP-floor temperature slopes are retained but are not identified
by this isothermal dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy.special import gammainc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.config import EV_TO_J
from arrhenius_fracture.fatigue_v1 import ExpFloorBarrierParams, ScaledExpFloorBarrier
from arrhenius_fracture.material_manifest import KB_EV_PER_K
from scripts.calibrate_qste340tm_opening_reference import (
    ATTEMPT_FREQUENCY_S,
    COOPERATIVE_HITS,
    COOPERATIVE_TAU_S,
    EVENT_LENGTH_M,
    NOMINAL_FREQUENCY_HZ,
    PHASE_STEPS,
    SMOOTHING_RATIO,
    TIP_RADIUS_M,
)
from scripts.fatigue_informed_joint_barrier_v3_v10230 import SmoothBroadBasis


TEMPERATURE_C = 25.0
TEMPERATURE_K = 298.15
SHEAR_MODULUS_PA = 80.0e9
POISSON_RATIO = 0.30
BURGERS_M = 2.48e-10
PROCESS_ZONE_M = 1.0e-6
RHO0_M2 = 5.0e12
C_BLUNT = 1.0
BETA_BACK = 1.0
V_EMB_B3 = 500.0
EXP_A = 0.0845685
EXP_N = 1.0
EXP_FLOOR_FRACTION = 0.02
EXP_GT_EV_PER_K = 0.003934
EXP_ST_PA_PER_K = -6.564e5
EXP_TREF_K = 481.33


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_records(data_root: Path) -> list[dict[str, float | int | str]]:
    rate_path = data_root / "normalized/qste340tm_constant_amplitude.csv"
    history_path = data_root / "normalized/qste340tm_crack_histories.csv"
    rates = read_csv(rate_path)
    histories = read_csv(history_path)
    records: list[dict[str, float | int | str]] = []
    for r_value in (0.1, 0.3, 0.5, 0.7):
        block = sorted(
            (row for row in rates if float(row["R"]) == r_value),
            key=lambda row: int(row["source_order"]),
        )
        history = sorted(
            (
                row for row in histories
                if float(row["R"]) == r_value and row["protocol"] == "CONSTANT_AMPLITUDE"
            ),
            key=lambda row: int(row["source_order"]),
        )
        h_fraction = np.linspace(0.0, 1.0, len(history))
        r_fraction = np.linspace(0.0, 1.0, len(block))
        cycles = np.interp(
            r_fraction, h_fraction,
            np.asarray([float(row["cycles"]) for row in history]),
        )
        for index, (row, cycle) in enumerate(zip(block, cycles)):
            records.append({
                "series_id": row["series_id"],
                "source_order": int(row["source_order"]),
                "block_index": index,
                "R": r_value,
                "DeltaK_MPa_sqrt_m": float(row["reported_DeltaK_MPa_sqrt_m"]),
                "observed_dadn_m_per_cycle": float(row["dadn_m_per_cycle"]),
                "cycles": float(cycle),
            })
    return records


def exp_floor_barrier(name: str, g0_298_eV: float, sigc_298_Pa: float,
                      prefactor: float) -> ScaledExpFloorBarrier:
    # Retain the established nonzero temperature couplings.  G00 and sigc0 are
    # merely reparameterized so the fitted coordinates are their 298.15 K values.
    base = ExpFloorBarrierParams(
        name=name,
        G00_eV=g0_298_eV - EXP_GT_EV_PER_K * (TEMPERATURE_K - EXP_TREF_K),
        gT_eV_per_K=EXP_GT_EV_PER_K,
        sigc0_Pa=sigc_298_Pa - EXP_ST_PA_PER_K * (TEMPERATURE_K - EXP_TREF_K),
        sT_Pa_per_K=EXP_ST_PA_PER_K,
        Tref_K=EXP_TREF_K,
        a=EXP_A,
        n=EXP_N,
        Gfloor_fraction=EXP_FLOOR_FRACTION,
    )
    return ScaledExpFloorBarrier(
        base=base, mechanism=name, energy_scale=1.0, entropy_scale=1.0,
        stress_scale=1.0, rate_prefactor=prefactor,
    )


def unpack(parameters: np.ndarray) -> dict[str, float]:
    return {
        "opening_floor_over_kBT": float(parameters[0]),
        "opening_amplitude_over_kBT": float(parameters[1]),
        "opening_sigma_b_Pa": float(math.exp(parameters[2]) * 1.0e9),
        "opening_exponent": float(math.exp(parameters[3])),
        "emission_G0_298_eV": float(parameters[4]),
        "plastic_sigc_298_Pa": float(math.exp(parameters[5]) * 1.0e9),
        "escape_G0_298_eV": float(parameters[6]),
        "retained_count_saturation": float(math.exp(parameters[7])),
        "cleavage_backstress_fraction": float(parameters[8]),
        "recovery_per_s": float(math.exp(parameters[9])),
    }


def _cycle_stress(delta_k: float, r_value: float, retained: float) -> tuple[np.ndarray, float, float]:
    phase = 2.0 * math.pi * (np.arange(PHASE_STEPS) + 0.5) / PHASE_STEPS
    fraction = 0.5 * (1.0 + r_value) + 0.5 * (1.0 - r_value) * np.cos(phase)
    kmax = delta_k / (1.0 - r_value) * 1.0e6
    radius = TIP_RADIUS_M + C_BLUNT * BURGERS_M * retained
    stress = kmax * fraction / math.sqrt(2.0 * math.pi * radius)
    backstress = (
        BETA_BACK * SHEAR_MODULUS_PA * BURGERS_M
        / (2.0 * math.pi * (1.0 - POISSON_RATIO) * PROCESS_ZONE_M)
        * retained
    )
    return stress, radius, backstress


def _embrittlement_eV(retained: float) -> float:
    rho = RHO0_M2 + retained / PROCESS_ZONE_M**2
    stored_J_m3 = 0.5 * SHEAR_MODULUS_PA * BURGERS_M**2 * rho
    return stored_J_m3 * V_EMB_B3 * BURGERS_M**3 / EV_TO_J


def evaluate(parameters: np.ndarray, records: list[dict[str, float | int | str]]) -> dict[str, np.ndarray]:
    p = unpack(parameters)
    opening = SmoothBroadBasis(
        sigma_b_Pa=p["opening_sigma_b_Pa"],
        exponent=p["opening_exponent"],
        smoothing_ratio=SMOOTHING_RATIO,
    )
    emission = exp_floor_barrier(
        "conditional_QSTE340TM_emission", p["emission_G0_298_eV"],
        p["plastic_sigc_298_Pa"], 1.0e11,
    )
    peierls = exp_floor_barrier(
        "conditional_QSTE340TM_peierls", p["escape_G0_298_eV"],
        p["plastic_sigc_298_Pa"], 1.0e12,
    )
    taylor = exp_floor_barrier(
        "conditional_QSTE340TM_taylor", p["escape_G0_298_eV"],
        p["plastic_sigc_298_Pa"], 1.0e11,
    )
    prediction = np.empty(len(records))
    retained_out = np.empty(len(records))
    radius_out = np.empty(len(records))
    backstress_out = np.empty(len(records))
    emit_out = np.empty(len(records))
    store_out = np.empty(len(records))
    escape_out = np.empty(len(records))
    by_r: dict[float, list[int]] = {}
    for index, row in enumerate(records):
        by_r.setdefault(float(row["R"]), []).append(index)
    for r_value, indices in by_r.items():
        retained = 0.0
        prior_cycles = 0.0
        for index in indices:
            row = records[index]
            cycles = float(row["cycles"])
            delta_cycles = max(cycles - prior_cycles, 0.0)
            prior_cycles = cycles
            stress, _, backstress = _cycle_stress(
                float(row["DeltaK_MPa_sqrt_m"]), r_value, retained
            )
            emission_stress = np.maximum(stress - backstress, 0.0)
            lam_emit = emission.rate(emission_stress, TEMPERATURE_K)
            lam_p = peierls.rate(emission_stress, TEMPERATURE_K)
            lam_t = taylor.rate(emission_stress, TEMPERATURE_K)
            lam_escape = 1.0 / (
                1.0 / np.maximum(lam_p, 1.0e-300)
                + 1.0 / np.maximum(lam_t, 1.0e-300)
            )
            mu_emit = float(np.mean(lam_emit) / NOMINAL_FREQUENCY_HZ)
            mu_escape = float(np.mean(lam_escape) / NOMINAL_FREQUENCY_HZ)
            storage_fraction = math.exp(-min(mu_escape, 700.0))
            source_per_cycle = mu_emit * storage_fraction
            saturation = p["retained_count_saturation"]
            loss_per_cycle = (
                source_per_cycle / saturation
                + p["recovery_per_s"] / NOMINAL_FREQUENCY_HZ
            )
            if loss_per_cycle > 1.0e-300:
                steady = source_per_cycle / loss_per_cycle
                retained = steady + (retained - steady) * math.exp(
                    -min(loss_per_cycle * delta_cycles, 700.0)
                )
            else:
                retained += source_per_cycle * delta_cycles
            retained = float(np.clip(retained, 0.0, saturation))

            stress, radius, backstress = _cycle_stress(
                float(row["DeltaK_MPa_sqrt_m"]), r_value, retained
            )
            opening_stress = np.maximum(
                stress - p["cleavage_backstress_fraction"] * backstress, 0.0
            )
            g_over_kbt = (
                p["opening_floor_over_kBT"]
                + p["opening_amplitude_over_kBT"] * opening.value(opening_stress)
                - _embrittlement_eV(retained) / (KB_EV_PER_K * TEMPERATURE_K)
            )
            elementary = ATTEMPT_FREQUENCY_S * np.exp(-np.clip(g_over_kbt, 0.0, 700.0))
            completed = gammainc(
                COOPERATIVE_HITS,
                np.minimum(elementary * COOPERATIVE_TAU_S, 1.0e12),
            ) / COOPERATIVE_TAU_S
            rate = EVENT_LENGTH_M * float(np.mean(completed)) / NOMINAL_FREQUENCY_HZ
            prediction[index] = max(rate, 1.0e-300)
            retained_out[index] = retained
            radius_out[index] = radius
            backstress_out[index] = backstress
            emit_out[index] = mu_emit
            store_out[index] = storage_fraction
            escape_out[index] = mu_escape
    return {
        "prediction": prediction,
        "retained": retained_out,
        "radius": radius_out,
        "backstress": backstress_out,
        "emission_per_cycle": emit_out,
        "storage_fraction": store_out,
        "escape_per_cycle": escape_out,
    }


LOWER = np.asarray([
    8.0, 0.0, math.log(0.1), math.log(0.1),
    0.30, math.log(0.1), 0.30, math.log(1.0), 0.0, math.log(1.0e-8),
])
UPPER = np.asarray([
    25.0, 100.0, math.log(100.0), math.log(10.0),
    3.00, math.log(20.0), 5.00, math.log(1.0e5), 2.0, math.log(1.0e5),
])


def fit(records: list[dict[str, float | int | str]], training_r: set[float],
        seed: int, maxiter: int) -> dict[str, object]:
    observed = np.asarray([float(row["observed_dadn_m_per_cycle"]) for row in records])
    observed_log = np.log10(observed)
    r_values = np.asarray([float(row["R"]) for row in records])
    support = np.zeros(len(records), dtype=bool)
    train = np.isin(r_values, list(training_r))
    for r_value in sorted(training_r):
        indices = np.flatnonzero((r_values == r_value) & train)
        chosen = indices[np.unique(np.linspace(0, len(indices) - 1, min(36, len(indices))).round().astype(int))]
        support[chosen] = True

    def residual(parameters: np.ndarray) -> np.ndarray:
        predicted_log = np.log10(evaluate(parameters, records)["prediction"])
        raw = predicted_log - observed_log
        selected = raw[support]
        selected_r = r_values[support]
        centered = selected.copy()
        for r_value in sorted(training_r):
            block = selected_r == r_value
            centered[block] -= np.mean(selected[block])
        # Equal block weight, preserving the prior opening-reference objective.
        scale = math.sqrt(36.0)
        return np.concatenate((selected / scale, centered / scale))

    opening_reference = json.loads(
        (ROOT / "literature_data/fatigue_recovered_review/model_conditional/conditional_opening_reference.json").read_text()
    )["nominal_fit"]["parameters"]
    initial = np.asarray([
        opening_reference["g_floor_Gstar_over_kBT"],
        opening_reference["g_amplitude_Gstar_over_kBT"],
        math.log(opening_reference["sigma_b_Pa"] / 1.0e9),
        math.log(opening_reference["exponent"]),
        1.20, math.log(2.0), 2.0, math.log(100.0), 0.5, math.log(1.0e-2),
    ])
    population = differential_evolution(
        lambda x: float(np.dot(residual(x), residual(x))),
        list(zip(LOWER, UPPER)), seed=seed, popsize=6, maxiter=maxiter,
        polish=False, workers=1, updating="immediate", x0=initial,
    )
    solution = least_squares(
        residual, population.x, bounds=(LOWER, UPPER), max_nfev=350,
        xtol=1.0e-9, ftol=1.0e-9, gtol=1.0e-9,
    )
    state = evaluate(solution.x, records)
    predicted_log = np.log10(state["prediction"])
    error = predicted_log - observed_log
    block_metrics = {}
    for r_value in sorted(set(r_values)):
        block = r_values == r_value
        x = np.log10(np.asarray([float(records[i]["DeltaK_MPa_sqrt_m"]) for i in np.flatnonzero(block)]))
        block_metrics[str(r_value)] = {
            "points": int(np.sum(block)),
            "training_block": bool(r_value in training_r),
            "observed_loglog_slope": float(np.polyfit(x, observed_log[block], 1)[0]),
            "predicted_loglog_slope": float(np.polyfit(x, predicted_log[block], 1)[0]),
            "rmse_log10_rate": float(np.sqrt(np.mean(error[block] ** 2))),
            "bias_log10_rate": float(np.mean(error[block])),
            "retained_count_final": float(state["retained"][np.flatnonzero(block)[-1]]),
            "tip_radius_final_m": float(state["radius"][np.flatnonzero(block)[-1]]),
            "backstress_final_Pa": float(state["backstress"][np.flatnonzero(block)[-1]]),
        }
    return {
        "parameters_vector": solution.x.tolist(),
        "parameters": unpack(solution.x),
        "optimizer": {
            "success": bool(solution.success), "message": str(solution.message),
            "function_evaluations": int(solution.nfev),
            "de_objective": float(population.fun), "least_squares_cost": float(solution.cost),
            "support_points": int(np.sum(support)), "maxiter": int(maxiter), "seed": int(seed),
        },
        "parameter_boundary_flags": {
            name: bool(abs(solution.x[i] - LOWER[i]) < 1.0e-4 or abs(solution.x[i] - UPPER[i]) < 1.0e-4)
            for i, name in enumerate((
                "opening_floor", "opening_amplitude", "opening_sigma_b", "opening_exponent",
                "emission_G0", "plastic_sigc", "escape_G0", "retained_saturation",
                "cleavage_backstress_fraction", "recovery_per_s",
            ))
        },
        "training_R": sorted(training_r),
        "block_metrics": block_metrics,
        "overall_rmse_log10_rate": float(np.sqrt(np.mean(error**2))),
        "prediction": state["prediction"].tolist(),
        "state": {key: value.tolist() for key, value in state.items() if key != "prediction"},
    }


def write_outputs(data_root: Path, records: list[dict[str, float | int | str]],
                  result: dict[str, object], holdouts: list[dict[str, object]]) -> None:
    output = data_root / "model_conditional/stateful_exp_floor"
    output.mkdir(parents=True, exist_ok=True)
    opening_path = data_root / "model_conditional/conditional_opening_reference.json"
    history_path = data_root / "normalized/qste340tm_crack_histories.csv"
    rate_path = data_root / "normalized/qste340tm_constant_amplitude.csv"
    payload = {
        "schema": "qste340tm-stateful-exp-floor-analytical-fit/1",
        "status": "CONDITIONAL_ISOTHERMAL_STATE_INFORMED_ANALYTICAL_FIT",
        "temperature_C": TEMPERATURE_C,
        "temperature_K": TEMPERATURE_K,
        "frequency_Hz": NOMINAL_FREQUENCY_HZ,
        "inputs": {
            "rate_csv": str(rate_path.relative_to(ROOT)), "rate_sha256": sha256(rate_path),
            "history_csv": str(history_path.relative_to(ROOT)), "history_sha256": sha256(history_path),
            "opening_reference": str(opening_path.relative_to(ROOT)), "opening_reference_sha256": sha256(opening_path),
        },
        "fixed_physics": {
            "EXP_floor_a": EXP_A, "EXP_floor_n": EXP_N,
            "EXP_floor_fraction": EXP_FLOOR_FRACTION,
            "EXP_floor_gT_eV_per_K": EXP_GT_EV_PER_K,
            "EXP_floor_sT_Pa_per_K": EXP_ST_PA_PER_K,
            "EXP_floor_Tref_K": EXP_TREF_K,
            "shear_modulus_Pa": SHEAR_MODULUS_PA, "poisson_ratio": POISSON_RATIO,
            "burgers_m": BURGERS_M, "process_zone_m": PROCESS_ZONE_M,
            "rho0_m-2": RHO0_M2, "c_blunt": C_BLUNT, "beta_back": BETA_BACK,
            "v_emb_b3": V_EMB_B3,
            "phase_steps": PHASE_STEPS,
        },
        "fit": {key: value for key, value in result.items() if key not in {"prediction", "state"}},
        "leave_one_R_out": [
            {key: value for key, value in item.items() if key not in {"prediction", "state", "parameters_vector"}}
            for item in holdouts
        ],
        "limitations": [
            "Analytical scalar-state fit; not a native full-state 1-D execution.",
            "Only 298.15 K is observed; nonzero EXP-floor thermal couplings are retained, not identified.",
            "Steel elastic/geometric state constants are provisional and require protocol/material confirmation.",
            "Opening and plasticity parameters remain correlated; complete-R holdouts bound transferability.",
            "Overload and closure histories are validation targets and were not included in this likelihood.",
        ],
    }
    (output / "fit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(records[0]) + [
            "predicted_dadn_m_per_cycle", "residual_log10", "retained_count",
            "tip_radius_m", "backstress_Pa", "emission_per_cycle",
            "storage_fraction", "escape_per_cycle",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        observed = np.asarray([float(row["observed_dadn_m_per_cycle"]) for row in records])
        prediction = np.asarray(result["prediction"])
        for i, row in enumerate(records):
            writer.writerow(dict(row) | {
                "predicted_dadn_m_per_cycle": prediction[i],
                "residual_log10": math.log10(prediction[i] / observed[i]),
                "retained_count": result["state"]["retained"][i],
                "tip_radius_m": result["state"]["radius"][i],
                "backstress_Pa": result["state"]["backstress"][i],
                "emission_per_cycle": result["state"]["emission_per_cycle"][i],
                "storage_fraction": result["state"]["storage_fraction"][i],
                "escape_per_cycle": result["state"]["escape_per_cycle"][i],
            })

    old = json.loads(opening_path.read_text())
    old_predictions = read_csv(data_root / "model_conditional/opening_reference_predictions.csv")
    colors = {0.1: "#0072B2", 0.3: "#D55E00", 0.5: "#009E73", 0.7: "#CC79A7"}
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.0), sharey=True)
    prediction = np.asarray(result["prediction"])
    for axis, r_value in zip(axes.flat, colors):
        indices = [i for i, row in enumerate(records) if float(row["R"]) == r_value]
        x = np.asarray([float(records[i]["DeltaK_MPa_sqrt_m"]) for i in indices])
        y = np.asarray([float(records[i]["observed_dadn_m_per_cycle"]) for i in indices])
        old_y = np.asarray([
            float(row["predicted_dadn_10Hz_m_per_cycle"])
            for row in old_predictions if float(row["R"]) == r_value
        ])
        order = np.argsort(x)
        axis.scatter(x, y, s=12, color=colors[r_value], alpha=0.70, edgecolors="none", label="data")
        axis.plot(x[order], old_y[order], color="#888888", linestyle="--", linewidth=1.5,
                  label="opening-only")
        axis.plot(x[order], prediction[indices][order], color="#111111", linewidth=2.0,
                  label="stateful EXP_floor")
        metric = result["block_metrics"][str(r_value)]
        axis.text(0.04, 0.95, f"R = {r_value:.1f}\nRMSE = {metric['rmse_log10_rate']:.3f} decades",
                  transform=axis.transAxes, va="top", fontsize=9)
        axis.set_xscale("log"); axis.set_yscale("log")
        ticks = np.geomspace(float(np.min(x)), float(np.max(x)), 3)
        axis.set_xticks(ticks)
        axis.set_xticklabels([f"{value:.1f}" for value in ticks])
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.grid(True, which="both", alpha=0.22)
        axis.set_xlabel(r"$\Delta K$ (MPa$\sqrt{\mathrm{m}}$)")
    axes[0, 0].set_ylabel(r"$da/dN$ (m cycle$^{-1}$)")
    axes[1, 0].set_ylabel(r"$da/dN$ (m cycle$^{-1}$)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("QSTE340TM: opening-only and state-informed EXP_floor fits", y=1.045)
    fig.text(0.5, 0.005, "25 °C, provisional 10 Hz; one shared material across all R histories.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.025, 1, 0.98))
    for ext in ("png", "svg"):
        fig.savefig(output / f"qste340tm_stateful_exp_floor_fit.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    p = result["parameters"]
    stress = np.linspace(0.0, 25.0e9, 600)
    new_basis = SmoothBroadBasis(p["opening_sigma_b_Pa"], p["opening_exponent"], SMOOTHING_RATIO)
    old_p = old["nominal_fit"]["parameters"]
    old_basis = SmoothBroadBasis(old_p["sigma_b_Pa"], old_p["exponent"], SMOOTHING_RATIO)
    new_g = (p["opening_floor_over_kBT"] + p["opening_amplitude_over_kBT"] * new_basis.value(stress)) * KB_EV_PER_K * TEMPERATURE_K
    old_g = (old_p["g_floor_Gstar_over_kBT"] + old_p["g_amplitude_Gstar_over_kBT"] * old_basis.value(stress)) * KB_EV_PER_K * TEMPERATURE_K
    emission = exp_floor_barrier("conditional_QSTE340TM_emission", p["emission_G0_298_eV"], p["plastic_sigc_298_Pa"], 1.0e11)
    escape = exp_floor_barrier("conditional_QSTE340TM_escape", p["escape_G0_298_eV"], p["plastic_sigc_298_Pa"], 1.0e11)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.7))
    ax0.plot(stress / 1e9, old_g, "--", color="#888888", label="opening-only G*")
    ax0.plot(stress / 1e9, new_g, color="#111111", linewidth=2.0, label="joint-fit opening G*")
    ax0.set_xlabel(r"local opening stress $\sigma$ (GPa)")
    ax0.set_ylabel(r"$G^*_{open}$ (eV)")
    ax0.set_title("Opening barrier"); ax0.grid(True, alpha=0.22); ax0.legend(frameon=False)
    ax1.plot(stress / 1e9, emission.deltaG_eV(stress, TEMPERATURE_K), color="#0072B2", linewidth=2.0,
             label="emission EXP_floor")
    ax1.plot(stress / 1e9, escape.deltaG_eV(stress, TEMPERATURE_K), color="#D55E00", linewidth=2.0,
             label="Peierls/Taylor escape EXP_floor")
    ax1.set_xlabel(r"effective plastic stress $\sigma$ (GPa)")
    ax1.set_ylabel(r"$\Delta G_{plastic}$ (eV)")
    ax1.set_title("Fitted room-temperature plastic barriers")
    ax1.grid(True, alpha=0.22); ax1.legend(frameon=False)
    fig.suptitle("Conditional barrier functions at 25 °C")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(output / f"qste340tm_stateful_barriers_298p15K.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.4, 5.5))
    retained = np.asarray(result["state"]["retained"])
    for r_value, color in colors.items():
        indices = [i for i, row in enumerate(records) if float(row["R"]) == r_value]
        x = np.asarray([float(records[i]["DeltaK_MPa_sqrt_m"]) for i in indices])
        axis.plot(x, retained[indices], color=color, linewidth=1.8, label=f"R={r_value:.1f}")
    axis.set_xlabel(r"$\Delta K$ (MPa$\sqrt{\mathrm{m}}$)")
    axis.set_ylabel("retained dislocation count (analytical ledger)")
    axis.set_title("Fitted state evolution along each measured history")
    axis.grid(True, alpha=0.22); axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(output / f"qste340tm_retained_state_histories.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    old_metrics = old["nominal_fit"]["block_metrics"]
    old_points = sum(item["points"] for item in old_metrics.values())
    old_overall = math.sqrt(sum(
        item["points"] * item["rmse_log10_rate"] ** 2
        for item in old_metrics.values()
    ) / old_points)
    comparison_rows = []
    for r_value in (0.1, 0.3, 0.5, 0.7):
        key = str(r_value)
        comparison_rows.append(
            f"| {r_value:.1f} | {old_metrics[key]['rmse_log10_rate']:.3f} | "
            f"{result['block_metrics'][key]['rmse_log10_rate']:.3f} | "
            f"{result['block_metrics'][key]['observed_loglog_slope']:.3f} | "
            f"{result['block_metrics'][key]['predicted_loglog_slope']:.3f} |"
        )
    report = f"""# Conditional QSTE340TM state-informed EXP_floor refinement

Date: 20 October 2026

## Result

The analytical state-informed fit lowers the all-point log-rate RMSE from
{old_overall:.3f} decades for the opening-only reference to
{result['overall_rmse_log10_rate']:.3f} decades. It substantially reduces the
between-R offset, especially at R=0.7, but it does not reproduce every within-R
curve shape. This is a partial refinement, not an accepted complete-material
calibration.

| R | opening-only RMSE | stateful RMSE | observed slope | stateful slope |
|---:|---:|---:|---:|---:|
{chr(10).join(comparison_rows)}

The R=0.5 block becomes worse and the R=0.3, 0.5 and 0.7 fitted slopes flatten.
The model is using a strong transition in retained plastic state to repair the
between-R displacement. That transition is physically interpretable through
the existing emission/escape/retention equations, but it is not yet supported
well enough to call the plasticity vector identified.

## Model scope

- One opening and plasticity material is shared by all four R histories.
- Each block is replayed in source order using cycle coordinates from the paired
  crack history; observations are not reset to a virgin state.
- Opening uses the existing smooth-broad barrier and cooperative completion law.
- Plasticity uses existing EXP_floor emission, series Peierls/Taylor escape,
  escape-limited retention, saturation, recovery, blunting, back stress and
  stored-energy coupling.
- The nonzero EXP_floor temperature slopes are retained. Only their effective
  298.15 K coordinates are fitted.
- Frequency remains the explicitly provisional 10 Hz assumption.

## Validation limit

All four leave-one-R-out least-squares refinements reached their 350-evaluation
limit. Their held-out RMSE values are recorded in `fit.json`; they are useful
bounds, not converged validation optima. R=0.3 transfers best, while R=0.5
remains the principal failure. The overload and closure histories were not
included in the likelihood and remain genuine validation targets.

This analytical result does not authorize a native full-state, thermal, or
complete QSTE340TM material claim.
"""
    (output / "RESULT.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("literature_data/fatigue_recovered_review"))
    parser.add_argument("--maxiter", type=int, default=24)
    parser.add_argument("--holdout-maxiter", type=int, default=12)
    parser.add_argument("--skip-holdouts", action="store_true")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    records = build_records(data_root)
    all_r = {0.1, 0.3, 0.5, 0.7}
    result = fit(records, all_r, seed=20261021, maxiter=args.maxiter)
    holdouts = []
    if not args.skip_holdouts:
        for index, held in enumerate(sorted(all_r)):
            item = fit(records, all_r - {held}, seed=20261100 + index, maxiter=args.holdout_maxiter)
            item["held_out_R"] = held
            holdouts.append(item)
    write_outputs(data_root, records, result, holdouts)
    summary = {
        "parameters": result["parameters"],
        "block_metrics": result["block_metrics"],
        "overall_rmse_log10_rate": result["overall_rmse_log10_rate"],
        "holdout_metrics": {
            str(item["held_out_R"]): item["block_metrics"][str(item["held_out_R"])]
            for item in holdouts
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
