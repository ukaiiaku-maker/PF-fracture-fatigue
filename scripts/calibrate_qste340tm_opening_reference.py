#!/usr/bin/env python3
"""Conditional QSTE340TM opening-reference calibration.

The calculation uses the existing cosine waveform, local K-to-stress mapping,
smooth broad G* basis, cooperative completion rule, and 5 um event reward.  It
fits an effective dimensionless barrier G*/kBT because the experiment's exact
temperature is not source-resolved.  It is not a complete material registry and
does not assign a steel EXP_floor plasticity surface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy.special import gammainc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fatigue_informed_joint_barrier_v3_v10230 import SmoothBroadBasis


ATTEMPT_FREQUENCY_S = 1.0e12
COOPERATIVE_HITS = 3.273241435177624
COOPERATIVE_TAU_S = 6.992153587194454e-7
EVENT_LENGTH_M = 5.0e-6
TIP_RADIUS_M = 1.0e-6
SMOOTHING_RATIO = 1.0e-4
NOMINAL_FREQUENCY_HZ = 10.0
SENSITIVITY_FREQUENCIES_HZ = (1.0, 10.0, 20.0)
PHASE_STEPS = 32


def canonical(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 760:
        raise ValueError(f"expected 760 QSTE340TM constant-amplitude rows, found {len(rows)}")
    return rows


def arrays(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    delta_k = np.asarray([float(row["reported_DeltaK_MPa_sqrt_m"]) for row in rows])
    load_ratio = np.asarray([float(row["R"]) for row in rows])
    observed = np.asarray([float(row["dadn_m_per_cycle"]) for row in rows])
    phase = 2.0 * math.pi * (np.arange(PHASE_STEPS) + 0.5) / PHASE_STEPS
    fraction = (
        0.5 * (1.0 + load_ratio[:, None])
        + 0.5 * (1.0 - load_ratio[:, None]) * np.cos(phase)[None, :]
    )
    kmax = delta_k / (1.0 - load_ratio)
    stress = kmax[:, None] * 1.0e6 * fraction / math.sqrt(2.0 * math.pi * TIP_RADIUS_M)
    return {"delta_k": delta_k, "R": load_ratio, "observed": observed, "stress": stress}


def predict_log10_rate(parameters: np.ndarray, stress: np.ndarray, frequency_hz: float) -> np.ndarray:
    g_floor, amplitude, log_sigma_b_gpa, log_exponent = parameters
    basis = SmoothBroadBasis(
        sigma_b_Pa=math.exp(log_sigma_b_gpa) * 1.0e9,
        exponent=math.exp(log_exponent),
        smoothing_ratio=SMOOTHING_RATIO,
    )
    g_over_kbt = g_floor + amplitude * basis.value(stress)
    elementary_rate = ATTEMPT_FREQUENCY_S * np.exp(-np.clip(g_over_kbt, 0.0, 700.0))
    final_rate = gammainc(
        COOPERATIVE_HITS, np.minimum(elementary_rate * COOPERATIVE_TAU_S, 1.0e12)
    ) / COOPERATIVE_TAU_S
    growth = EVENT_LENGTH_M * np.mean(final_rate, axis=1) / float(frequency_hz)
    return np.log10(np.maximum(growth, 1.0e-300))


def fit(arr: dict[str, np.ndarray], frequency_hz: float, training_mask: np.ndarray) -> dict[str, object]:
    observed_log = np.log10(arr["observed"])
    # The author-reduced curves are densely correlated.  Optimize on an evenly
    # spaced, block-balanced support while evaluating every supplied point.
    optimization_mask = np.zeros(len(training_mask), dtype=bool)
    for r_value in sorted(set(arr["R"])):
        indices = np.flatnonzero((arr["R"] == r_value) & training_mask)
        if indices.size:
            chosen = indices[np.unique(np.linspace(0, indices.size - 1, min(40, indices.size)).round().astype(int))]
            optimization_mask[chosen] = True
    counts = {
        r: int(np.sum((arr["R"] == r) & optimization_mask)) for r in sorted(set(arr["R"]))
    }
    weights = np.asarray(
        [1.0 / math.sqrt(counts[r]) if counts.get(r, 0) else 0.0 for r in arr["R"]]
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        values = predict_log10_rate(parameters, arr["stress"], frequency_hz) - observed_log
        selected = values[optimization_mask]
        selected_r = arr["R"][optimization_mask]
        centered = selected.copy()
        for r_value in sorted(set(selected_r)):
            block = selected_r == r_value
            centered[block] -= np.mean(selected[block])
        # Rate and within-R shape are both objectives.  Dense neighboring points
        # are block-balanced, and shape is not sacrificed merely to match one
        # global rate offset.
        selected_weights = weights[optimization_mask]
        return np.concatenate((0.25 * selected * selected_weights, centered * selected_weights))

    lower = np.asarray([8.0, 0.0, math.log(0.1), math.log(0.1)])
    upper = np.asarray([25.0, 100.0, math.log(100.0), math.log(10.0)])
    global_solution = differential_evolution(
        lambda x: float(np.dot(residual(x), residual(x))),
        list(zip(lower, upper)), seed=20261020, popsize=8, maxiter=45, polish=False,
    )
    solution = least_squares(
        residual, global_solution.x, bounds=(lower, upper), max_nfev=600
    )
    prediction = predict_log10_rate(solution.x, arr["stress"], frequency_hz)
    error = prediction - observed_log
    train_error = error[training_mask]
    test_error = error[~training_mask]
    g_floor, amplitude, log_sigma_b, log_exponent = solution.x
    block_metrics = {}
    for r_value in sorted(set(arr["R"])):
        block = arr["R"] == r_value
        x = np.log10(arr["delta_k"][block])
        observed_block = observed_log[block]
        predicted_block = prediction[block]
        block_error = error[block]
        block_metrics[str(float(r_value))] = {
            "points": int(np.sum(block)),
            "observed_loglog_slope": float(np.polyfit(x, observed_block, 1)[0]),
            "predicted_loglog_slope": float(np.polyfit(x, predicted_block, 1)[0]),
            "rmse_log10_rate": float(np.sqrt(np.mean(block_error**2))),
            "bias_log10_rate": float(np.mean(block_error)),
        }
    return {
        "frequency_Hz": float(frequency_hz),
        "parameters": {
            "g_floor_Gstar_over_kBT": float(g_floor),
            "g_amplitude_Gstar_over_kBT": float(amplitude),
            "sigma_b_Pa": float(math.exp(log_sigma_b) * 1.0e9),
            "exponent": float(math.exp(log_exponent)),
            "smoothing_ratio": SMOOTHING_RATIO,
        },
        "parameter_boundary_flags": {
            "g_floor": bool(abs(g_floor - lower[0]) < 1.0e-5 or abs(g_floor - upper[0]) < 1.0e-5),
            "g_amplitude": bool(abs(amplitude - lower[1]) < 1.0e-5 or abs(amplitude - upper[1]) < 1.0e-5),
            "sigma_b": bool(abs(log_sigma_b - lower[2]) < 1.0e-5 or abs(log_sigma_b - upper[2]) < 1.0e-5),
            "exponent": bool(abs(log_exponent - lower[3]) < 1.0e-5 or abs(log_exponent - upper[3]) < 1.0e-5),
        },
        "block_metrics": block_metrics,
        "optimizer": {
            "success": bool(solution.success),
            "status": int(solution.status),
            "message": str(solution.message),
            "function_evaluations": int(solution.nfev),
            "weighted_cost": float(solution.cost),
            "objective": "block-balanced absolute log-rate (weight 0.25) plus within-R centered shape (weight 1.0)",
        },
        "train_points": int(np.sum(training_mask)),
        "optimizer_support_points": int(np.sum(optimization_mask)),
        "test_points": int(np.sum(~training_mask)),
        "train_rmse_log10_rate": float(np.sqrt(np.mean(train_error**2))),
        "train_bias_log10_rate": float(np.mean(train_error)),
        "test_rmse_log10_rate": None if not test_error.size else float(np.sqrt(np.mean(test_error**2))),
        "test_bias_log10_rate": None if not test_error.size else float(np.mean(test_error)),
        "prediction_log10": prediction,
        "error_log10": error,
    }


def strip_arrays(result: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in result.items() if key not in {"prediction_log10", "error_log10"}}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("literature_data/fatigue_recovered_review")
    )
    args = parser.parse_args()
    root = args.data_root.resolve()
    source = root / "normalized/qste340tm_constant_amplitude.csv"
    output = root / "model_conditional"
    output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(source)
    arr = arrays(rows)
    all_rows = np.ones(len(rows), dtype=bool)

    sensitivity = [fit(arr, frequency, all_rows) for frequency in SENSITIVITY_FREQUENCIES_HZ]
    nominal = next(item for item in sensitivity if item["frequency_Hz"] == NOMINAL_FREQUENCY_HZ)
    holdouts = []
    for held_r in sorted(set(arr["R"])):
        result = fit(arr, NOMINAL_FREQUENCY_HZ, arr["R"] != held_r)
        result["held_out_R"] = float(held_r)
        holdouts.append(result)

    prediction_rows: list[dict[str, object]] = []
    for index, source_row in enumerate(rows):
        row: dict[str, object] = {
            "dataset_id": source_row["dataset_id"],
            "series_id": source_row["series_id"],
            "source_order": int(source_row["source_order"]),
            "R": float(source_row["R"]),
            "DeltaK_MPa_sqrt_m": float(source_row["reported_DeltaK_MPa_sqrt_m"]),
            "observed_dadn_m_per_cycle": float(source_row["dadn_m_per_cycle"]),
        }
        for item in sensitivity:
            label = f"{item['frequency_Hz']:g}Hz"
            row[f"predicted_dadn_{label}_m_per_cycle"] = 10.0 ** float(item["prediction_log10"][index])
            row[f"residual_{label}_log10"] = float(item["error_log10"][index])
        prediction_rows.append(row)
    write_csv(output / "opening_reference_predictions.csv", prediction_rows)

    holdout_rows = []
    for item in holdouts:
        params = item["parameters"]
        holdout_rows.append(
            {
                "held_out_R": item["held_out_R"],
                "train_points": item["train_points"],
                "held_out_points": item["test_points"],
                "train_rmse_log10_rate": item["train_rmse_log10_rate"],
                "held_out_rmse_log10_rate": item["test_rmse_log10_rate"],
                "held_out_bias_log10_rate": item["test_bias_log10_rate"],
                **params,
            }
        )
    write_csv(output / "leave_one_R_out.csv", holdout_rows)

    stress_min = float(np.min(arr["stress"]))
    stress_max = float(np.max(arr["stress"]))
    payload = {
        "schema": "qste340tm-conditional-opening-reference/1",
        "source_csv": str(source),
        "source_csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "status": "CONDITIONAL_OPENING_REFERENCE_NOT_COMPLETE_MATERIAL",
        "assumptions": {
            "frequency_nominal_Hz": NOMINAL_FREQUENCY_HZ,
            "frequency_sensitivity_Hz": list(SENSITIVITY_FREQUENCIES_HZ),
            "frequency_basis": "provisional reasonable laboratory fatigue frequency authorized by user; not source metadata",
            "waveform": "cosine/sinusoidal existing FatigueWaveform convention; provisional because source waveform remains unresolved",
            "temperature_K": None,
            "barrier_representation": "dimensionless effective Gstar/kBT at the unresolved test temperature",
            "local_stress_mapping": "existing K/sqrt(2*pi*r0) mapping",
            "tip_radius_m": TIP_RADIUS_M,
            "event_length_m": EVENT_LENGTH_M,
            "attempt_frequency_s-1": ATTEMPT_FREQUENCY_S,
            "cooperative_hits": COOPERATIVE_HITS,
            "cooperative_tau_s": COOPERATIVE_TAU_S,
            "phase_steps": PHASE_STEPS,
        },
        "data_informed_model_stress_interval_Pa": [stress_min, stress_max],
        "nominal_fit": strip_arrays(nominal),
        "frequency_sensitivity": [strip_arrays(item) for item in sensitivity],
        "leave_one_R_out": [strip_arrays(item) | {"held_out_R": item["held_out_R"]} for item in holdouts],
        "limitations": [
            "No absolute barrier energy or activation volume in SI units is identified without test temperature.",
            "No QSTE340TM EXP_floor plasticity surface is assigned; this is opening-reference only.",
            "The overload and closure histories are not fitted or replayed.",
            "The frequency and waveform are modeling assumptions, not recovered observations.",
        ],
    }
    payload["calibration_sha256"] = canonical(payload)
    (output / "conditional_opening_reference.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
