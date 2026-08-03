"""Hazard-consistent plastic-flow terminal for the v10.4.3 monotonic path.

This module wraps the already transformed physical-progress solver at runtime.
It leaves the mechanics, constitutive update, directional J, cleavage hazard,
and event gate untouched.  Only the terminal decision dictionary is revised:
finite positive J and finite tip stress remain explicit diagnostics, while the
acceptance gate uses a conservative projected cleavage-action increment.
"""
from __future__ import annotations

from functools import wraps
import math
from types import ModuleType

import numpy as np

from .plastic_flow_physical_progress_v1043 import (
    load_transformed_sharp_front as _load_physical_progress_module,
)

MODEL_ID = "v10.4.3_plastic_flow_terminal_projected_cleavage_hazard"
AUDIT_SCHEMA = "v10.4.3_projected_hazard_plastic_flow_terminal_audit_v2"


def _projection_values(window, metrics):
    first = float(metrics["window_first_nominal_progress"])
    last = float(metrics["window_last_nominal_progress"])
    tol = 128.0 * np.finfo(float).eps * max(abs(first), abs(last), 1.0)
    return [
        row for row in window
        if float(row["nominal_progress_end"]) > first + tol
        and float(row["nominal_progress_end"]) <= last + tol
    ]


def _projected_hazard_metrics(window, args, metrics, *, remaining_steps, nominal_dt_s):
    values = _projection_values(window, metrics)
    if len(values) < 2:
        return metrics

    progress = np.asarray(
        [float(row["nominal_progress_end"]) for row in values],
        dtype=float,
    )
    lambdas = np.asarray(
        [max(float(row["lambda_c"]), 0.0) for row in values],
        dtype=float,
    )
    lambda_safe = np.maximum(lambdas, 1.0e-300)
    log_lambda = np.log(lambda_safe)

    span = float(progress[-1] - progress[0])
    if span > 1.0e-15:
        fitted = max(float(np.polyfit(progress, log_lambda, 1)[0]), 0.0)
        secant = max(float((log_lambda[-1] - log_lambda[0]) / span), 0.0)
        local = np.diff(log_lambda) / np.maximum(np.diff(progress), 1.0e-300)
        positive_local = local[local > 0.0]
    else:
        fitted = 0.0
        secant = 0.0
        positive_local = np.asarray([], dtype=float)

    percentile = min(
        max(float(getattr(args, "plastic_flow_hazard_growth_percentile", 95.0) or 95.0), 50.0),
        100.0,
    )
    percentile_growth = (
        float(np.percentile(positive_local, percentile))
        if positive_local.size
        else 0.0
    )
    growth = max(fitted, secant, percentile_growth, 0.0)

    prospective_horizon = max(
        float(getattr(args, "plastic_flow_prospective_horizon_steps", 2000.0) or 2000.0),
        1.0,
    )
    projected_horizon = max(prospective_horizon, max(float(remaining_steps), 0.0))
    dt = max(float(nominal_dt_s), 0.0)
    lambda_reference = max(float(np.max(lambdas)), float(lambda_safe[-1]))

    if projected_horizon <= 0.0 or dt <= 0.0 or lambda_reference <= 0.0:
        log_projected_action = float("-inf")
    elif growth <= 1.0e-15:
        log_projected_action = (
            math.log(max(dt, 1.0e-300))
            + math.log(max(lambda_reference, 1.0e-300))
            + math.log(max(projected_horizon, 1.0e-300))
        )
    else:
        exponent = growth * projected_horizon
        log_expm1 = exponent if exponent > 50.0 else math.log(math.expm1(exponent))
        log_projected_action = (
            math.log(max(dt, 1.0e-300))
            + math.log(max(lambda_reference, 1.0e-300))
            + log_expm1
            - math.log(growth)
        )

    budget = max(1.0 - float(metrics["B_final"]), 1.0e-300)
    log_fraction = log_projected_action - math.log(budget)
    max_fraction = min(
        max(
            float(getattr(args, "plastic_flow_max_projected_hazard_fraction", 0.01) or 0.01),
            1.0e-300,
        ),
        1.0,
    )
    safe = log_fraction <= math.log(max_fraction)

    def exp_or_inf(value):
        if not math.isfinite(value):
            return 0.0 if value < 0.0 else float("inf")
        if value > math.log(np.finfo(float).max):
            return float("inf")
        return math.exp(value)

    criteria = dict(metrics["criteria"])
    legacy_j = bool(criteria.pop("negligible_positive_tip_J", False))
    legacy_sigma = bool(criteria.pop("negligible_tip_stress", False))
    criteria.pop("cleavage_outside_remaining_horizon", None)
    criteria["projected_cleavage_action_safe"] = bool(safe)

    metrics.update(
        {
            # This key is expanded last into plastic_flow_terminal_audit.json,
            # intentionally overriding the base v10.4.2 schema label.
            "schema": AUDIT_SCHEMA,
            "terminal_classifier_model_id": MODEL_ID,
            "criteria": criteria,
            "criteria_pass": all(criteria.values()),
            "legacy_negligible_positive_tip_J": legacy_j,
            "legacy_negligible_tip_stress": legacy_sigma,
            "J_and_sigma_zero_gates_are_diagnostic_only": True,
            "prospective_horizon_nominal_increments": prospective_horizon,
            "projected_horizon_nominal_increments": projected_horizon,
            "projected_horizon_s": projected_horizon * dt,
            "lambda_projection_reference_per_s": lambda_reference,
            "hazard_growth_percentile": percentile,
            "fitted_log_lambda_growth_per_nominal_increment": fitted,
            "secant_log_lambda_growth_per_nominal_increment": secant,
            "percentile_log_lambda_growth_per_nominal_increment": percentile_growth,
            "conservative_log_lambda_growth_per_nominal_increment": growth,
            "projected_cleavage_action_increment": exp_or_inf(log_projected_action),
            "log_projected_cleavage_action_increment": log_projected_action,
            "remaining_first_passage_budget": budget,
            "projected_hazard_fraction_of_remaining_budget": exp_or_inf(log_fraction),
            "log_projected_hazard_fraction_of_remaining_budget": log_fraction,
            "max_projected_hazard_fraction": max_fraction,
        }
    )
    return metrics


def load_transformed_sharp_front() -> ModuleType:
    module = _load_physical_progress_module()
    if getattr(module, "_v1043_hazard_terminal_wrapped", False):
        return module

    original = module._v1042_terminal_metrics

    @wraps(original)
    def wrapped(window, args, **kwargs):
        metrics = original(window, args, **kwargs)
        if metrics is None:
            return None
        return _projected_hazard_metrics(
            window,
            args,
            metrics,
            remaining_steps=kwargs.get("remaining_steps", 0.0),
            nominal_dt_s=kwargs.get("nominal_dt_s", 0.0),
        )

    module._v1042_terminal_metrics = wrapped
    module._v1043_hazard_terminal_wrapped = True
    module._v1043_hazard_terminal_model_id = MODEL_ID
    module._v1043_hazard_terminal_audit_schema = AUDIT_SCHEMA
    return module


__all__ = ["AUDIT_SCHEMA", "MODEL_ID", "load_transformed_sharp_front"]
