"""Slope-priority objective and barrier qualification for QSTE340TM.

This module contains no native mechanics.  It applies fixed regression
operators to physical prediction vectors and audits the complete installed
thermodynamic opening surface.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np


R_VALUES = (0.1, 0.3, 0.5, 0.7)
EV_J = 1.602176634e-19


def _slope_operator(x: np.ndarray) -> np.ndarray:
    centered = x - float(np.mean(x))
    denominator = float(centered @ centered)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("slope window has no finite load span")
    return centered / denominator


@dataclass(frozen=True)
class HistoryOperators:
    r_value: float
    indices: np.ndarray
    global_operator: np.ndarray
    regional_indices: tuple[np.ndarray, np.ndarray, np.ndarray]
    regional_operators: tuple[np.ndarray, np.ndarray, np.ndarray]
    observed_global_slope: float
    observed_regional_slopes: tuple[float, float, float]
    global_tolerance: float
    regional_tolerances: tuple[float, float, float]


class SlopePriorityObjective:
    """Composite engineering objective with per-history acceptance gates."""

    def __init__(self, observed_log10: np.ndarray,
                 histories: tuple[HistoryOperators, ...],
                 centered_tolerance: float = 0.05) -> None:
        self.observed_log10 = np.asarray(observed_log10, dtype=float)
        self.histories = histories
        self.centered_tolerance = float(centered_tolerance)
        if self.centered_tolerance <= 0.0:
            raise ValueError("centered tolerance must be positive")

    @classmethod
    def from_records(cls, records: Iterable[dict[str, Any]], *,
                     centered_tolerance: float = 0.05) -> "SlopePriorityObjective":
        rows = list(records)
        if not rows:
            raise ValueError("no observations")
        observed = np.log10(np.asarray(
            [float(row["observed_dadn_m_per_cycle"]) for row in rows], dtype=float,
        ))
        if not np.all(np.isfinite(observed)):
            raise ValueError("observed rates must be finite and positive")
        histories = []
        for r_value in R_VALUES:
            indices = np.asarray(
                [i for i, row in enumerate(rows) if float(row["R"]) == r_value], dtype=int,
            )
            if indices.size < 6:
                raise ValueError(f"insufficient observations for R={r_value}")
            x = np.log10(np.asarray(
                [float(rows[i]["DeltaK_MPa_sqrt_m"]) for i in indices], dtype=float,
            ))
            if not np.all(np.isfinite(x)):
                raise ValueError("DeltaK must be finite and positive")
            global_operator = _slope_operator(x)
            global_slope = float(global_operator @ observed[indices])
            edges = np.linspace(float(np.min(x)), float(np.max(x)), 4)
            regional_indices = []
            regional_operators = []
            regional_slopes = []
            for region in range(3):
                mask = ((x >= edges[region]) &
                        (x <= edges[region + 1] if region == 2 else x < edges[region + 1]))
                local = np.flatnonzero(mask)
                if local.size < 3:
                    raise ValueError(f"regional slope window too small for R={r_value}")
                operator = _slope_operator(x[local])
                regional_indices.append(local)
                regional_operators.append(operator)
                regional_slopes.append(float(operator @ observed[indices[local]]))
            histories.append(HistoryOperators(
                r_value=r_value,
                indices=indices,
                global_operator=global_operator,
                regional_indices=tuple(regional_indices),
                regional_operators=tuple(regional_operators),
                observed_global_slope=global_slope,
                observed_regional_slopes=tuple(regional_slopes),
                global_tolerance=max(0.2, 0.10 * abs(global_slope)),
                regional_tolerances=tuple(max(0.3, 0.15 * abs(v)) for v in regional_slopes),
            ))
        return cls(observed, tuple(histories), centered_tolerance)

    def residuals(self, prediction: np.ndarray, *, offset_weight: float = 0.0) -> np.ndarray:
        prediction = np.asarray(prediction, dtype=float)
        if prediction.shape != self.observed_log10.shape or not np.all(
            np.isfinite(prediction) & (prediction > 0.0)
        ):
            raise ValueError("prediction must be a finite positive vector matching observations")
        if offset_weight < 0.0:
            raise ValueError("offset weight cannot be negative")
        predicted = np.log10(prediction)
        residual = []
        for history in self.histories:
            index = history.indices
            error = predicted[index] - self.observed_log10[index]
            bias = float(np.mean(error))
            centered = error - bias
            global_slope = float(history.global_operator @ predicted[index])
            residual.append(math.sqrt(0.5) * (
                global_slope - history.observed_global_slope
            ) / history.global_tolerance)
            for local, operator, target, tolerance in zip(
                history.regional_indices, history.regional_operators,
                history.observed_regional_slopes, history.regional_tolerances,
            ):
                slope = float(operator @ predicted[index[local]])
                residual.append(math.sqrt(1.0 / 6.0) * (slope - target) / tolerance)
            residual.extend((centered / self.centered_tolerance /
                             math.sqrt(4.0 * centered.size)).tolist())
            if offset_weight:
                residual.append(math.sqrt(offset_weight / 4.0) * bias)
        return np.asarray(residual, dtype=float)

    def evaluate(self, prediction: np.ndarray, *, barrier_pass: bool | None) -> dict[str, Any]:
        prediction = np.asarray(prediction, dtype=float)
        predicted = np.log10(prediction)
        blocks = {}
        primary = float(self.residuals(prediction) @ self.residuals(prediction))
        offset_score = 0.0
        all_shape_pass = True
        worst = {"normalized_violation": -math.inf}
        for history in self.histories:
            index = history.indices
            error = predicted[index] - self.observed_log10[index]
            bias = float(np.mean(error))
            centered_rmse = float(np.sqrt(np.mean((error - bias) ** 2)))
            predicted_global = float(history.global_operator @ predicted[index])
            predicted_regional = tuple(
                float(operator @ predicted[index[local]])
                for local, operator in zip(history.regional_indices, history.regional_operators)
            )
            global_error = predicted_global - history.observed_global_slope
            regional_errors = tuple(
                value - target for value, target in
                zip(predicted_regional, history.observed_regional_slopes)
            )
            normalized = [abs(global_error) / history.global_tolerance]
            normalized.extend(abs(value) / tolerance for value, tolerance in
                              zip(regional_errors, history.regional_tolerances))
            normalized.append(centered_rmse / self.centered_tolerance)
            block_pass = all(value <= 1.0 for value in normalized)
            all_shape_pass = all_shape_pass and block_pass
            local_worst = max(normalized)
            if local_worst > worst["normalized_violation"]:
                worst = {"R": history.r_value, "normalized_violation": local_worst}
            offset_score += bias * bias / 4.0
            blocks[str(history.r_value)] = {
                "points": int(index.size),
                "bias_log10_rate": bias,
                "centered_rmse_log10": centered_rmse,
                "centered_tolerance_log10": self.centered_tolerance,
                "observed_global_slope": history.observed_global_slope,
                "predicted_global_slope": predicted_global,
                "global_slope_error": global_error,
                "global_slope_tolerance": history.global_tolerance,
                "observed_regional_slopes": list(history.observed_regional_slopes),
                "predicted_regional_slopes": list(predicted_regional),
                "regional_slope_errors": list(regional_errors),
                "regional_slope_tolerances": list(history.regional_tolerances),
                "shape_pass": block_pass,
                "worst_normalized_violation": local_worst,
            }
        return {
            "primary_score": primary,
            "offset_score": float(offset_score),
            "barrier_pass": barrier_pass,
            "shape_pass": bool(all_shape_pass),
            "eligible": bool(all_shape_pass and barrier_pass is True),
            "worst_normalized_violation": worst,
            "blocks": blocks,
        }


def select_eligible_candidate(candidates: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [candidate for candidate in candidates if candidate.get("evaluation", {}).get("eligible")]
    if not eligible:
        return None
    return min(eligible, key=lambda candidate: (
        float(candidate["evaluation"]["primary_score"]),
        float(candidate["evaluation"]["offset_score"]),
    ))


def offset_refinement_allowed(reference: dict[str, Any], candidate: dict[str, Any], *,
                              global_budget_fraction: float = 0.02,
                              per_condition_budget_fraction: float = 0.02) -> bool:
    if not reference.get("eligible") or not candidate.get("eligible"):
        return False
    if candidate["primary_score"] > reference["primary_score"] * (1.0 + global_budget_fraction):
        return False
    for r_value in R_VALUES:
        old = reference["blocks"][str(r_value)]["worst_normalized_violation"]
        new = candidate["blocks"][str(r_value)]["worst_normalized_violation"]
        if new > old * (1.0 + per_condition_budget_fraction):
            return False
    return candidate["offset_score"] < reference["offset_score"]


def qualify_opening_surface(surface: Any, *, temperature_K: float,
                            visited_stress_Pa: np.ndarray,
                            stress_max_Pa: float = 30.0e9,
                            grid_points: int = 2001,
                            derivative_tolerance_eV_per_Pa: float = 1.0e-18) -> dict[str, Any]:
    stress = np.linspace(0.0, float(stress_max_Pa), int(grid_points))
    barrier = np.asarray(surface.G_eV(stress, temperature_K), dtype=float)
    derivative = np.asarray(surface.dG_dsigma_eV_per_Pa(stress, temperature_K), dtype=float)
    volume = np.asarray(surface.activation_volume_m3(stress, temperature_K), dtype=float)
    entropy = np.asarray(surface.entropy_kB(stress, temperature_K), dtype=float)
    enthalpy = np.asarray(surface.enthalpy_eV(stress, temperature_K), dtype=float)
    visited = np.asarray(visited_stress_Pa, dtype=float)
    checks = {
        "finite": bool(all(np.all(np.isfinite(value)) for value in
                           (barrier, derivative, volume, entropy, enthalpy))),
        "positive_barrier": bool(np.min(barrier) > 0.0),
        "nonincreasing_barrier": bool(np.max(derivative) <= derivative_tolerance_eV_per_Pa),
        "nonnegative_activation_volume": bool(np.min(volume) >= -derivative_tolerance_eV_per_Pa * EV_J),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "temperature_K": float(temperature_K),
        "stress_domain_Pa": [0.0, float(stress_max_Pa)],
        "visited_stress_range_Pa": [float(np.min(visited)), float(np.max(visited))],
        "minimum_G_eV": float(np.min(barrier)),
        "maximum_G_eV": float(np.max(barrier)),
        "maximum_dG_dsigma_eV_per_Pa": float(np.max(derivative)),
        "minimum_activation_volume_m3": float(np.min(volume)),
        "maximum_activation_volume_m3": float(np.max(volume)),
        "entropy_range_kB": [float(np.min(entropy)), float(np.max(entropy))],
        "enthalpy_range_eV": [float(np.min(enthalpy)), float(np.max(enthalpy))],
        "stress_cap_occupancy_fraction": float(np.mean(visited >= stress_max_Pa * (1.0 - 1.0e-12))),
    }


__all__ = [
    "SlopePriorityObjective", "offset_refinement_allowed", "qualify_opening_surface",
    "select_eligible_candidate",
]
