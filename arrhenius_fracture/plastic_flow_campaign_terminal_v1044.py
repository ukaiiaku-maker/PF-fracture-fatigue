"""Campaign terminal for full-field bulk-plasticity fracture sweeps.

The fracture physics is unchanged. This overlay changes only campaign
termination and reporting:

* a case may terminate after a prior first-passage event if a later accepted
  loading window has no crack event or meaningful crack extension;
* projected future cleavage action and cleavage-action growth remain recorded
  diagnostics but are not terminal vetoes;
* finite elastic J and tip stress remain diagnostics rather than zero gates;
* the accepted terminal requires sustained plastic accommodation, flat elastic
  storage, and collapsed incremental load-carrying stiffness;
* a second bounded-work terminal recognizes severe adaptive-substep stagnation
  without waiting for a full nominal-loading window.

The legacy ``PLASTIC_FLOW`` marker is retained for existing tooling. A new
``PLASTICITY_DOMINATED`` marker records whether the terminal occurred before
first passage or after partial sharp-fracture growth.
"""
from __future__ import annotations

from functools import wraps
import math
from types import ModuleType

import numpy as np

from . import plastic_flow_terminal_v1042 as _v1042
from .plastic_flow_hazard_terminal_v1043 import (
    load_transformed_sharp_front as _load_hazard_terminal_module,
)

MODEL_ID = "v10.4.4_plasticity_dominated_campaign_terminal"
AUDIT_SCHEMA = "v10.4.4_plasticity_dominated_campaign_terminal_audit_v2"

_ACCEPTANCE_KEYS = (
    "no_crack_event_in_window",
    "negligible_crack_extension",
    "plastic_accommodation_dominant",
    "elastic_storage_flat",
    "load_carrying_capacity_collapsed",
)


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label} changed: expected one occurrence, found {count}"
        )
    return text.replace(old, new, 1)


def _campaign_terminal_block(source: str) -> str:
    """Return the v10.4.2 terminal block with v10.4.4 campaign semantics."""
    text = _replace_once(
        source,
        "and not fatigue_mode and Kc_first is None):",
        "and not fatigue_mode):",
        "plastic terminal after-first-passage condition",
    )
    text = _replace_once(
        text,
        "'classification': 'plastic_flow_no_sharp_fracture',",
        "'classification': (\n"
        "                            'plasticity_dominated_after_partial_fracture'\n"
        "                            if Kc_first is not None\n"
        "                            else 'plasticity_dominated_no_crack_growth'\n"
        "                        ),",
        "plasticity-dominated classification",
    )
    text = _replace_once(
        text,
        "'sharp_fracture_occurred': False,\n"
        "                        'first_passage_recorded': False,",
        "'sharp_fracture_occurred': bool(Kc_first is not None),\n"
        "                        'first_passage_recorded': bool(Kc_first is not None),\n"
        "                        'Kc_first_MPa_sqrt_m': (\n"
        "                            None if Kc_first is None else float(Kc_first)\n"
        "                        ),",
        "dynamic first-passage provenance",
    )
    text = _replace_once(
        text,
        "'failure_regime': 'bulk_plastic_flow',",
        "'failure_regime': (\n"
        "                            'bulk_plasticity_dominated_after_partial_fracture'\n"
        "                            if Kc_first is not None\n"
        "                            else 'bulk_plasticity_dominated_before_first_passage'\n"
        "                        ),\n"
        "                        'Eprime_Pa': float(mat.Eprime),",
        "plasticity-dominated regime provenance",
    )
    text = _replace_once(
        text,
        "with open(os.path.join(args.out, 'PLASTIC_FLOW'), 'w') as _v1042_fp:\n"
        "                        _v1042_fp.write('plastic_flow_no_sharp_fracture\\n')\n"
        "                    plastic_flow_terminal = True",
        "with open(os.path.join(args.out, 'PLASTIC_FLOW'), 'w') as _v1042_fp:\n"
        "                        _v1042_fp.write('plastic_flow_no_sharp_fracture\\n')\n"
        "                    with open(\n"
        "                        os.path.join(args.out, 'PLASTICITY_DOMINATED'), 'w'\n"
        "                    ) as _v1042_fp:\n"
        "                        _v1042_fp.write(\n"
        "                            plastic_flow_terminal_audit['classification'] + '\\n'\n"
        "                        )\n"
        "                    plastic_flow_terminal = True",
        "plasticity-dominated marker",
    )
    return text


def _window_energy_metrics(values, *, peak_force: float, stiffness_reference: float):
    first = values[0]
    last = values[-1]

    dWext = max(float(last["W_ext"] - first["W_ext"]), 0.0)
    dUel = abs(float(last["U_el"] - first["U_el"]))
    dWp = max(float(last["W_p"] - first["W_p"]), 0.0)
    dWemit = max(float(last["W_emit"] - first["W_emit"]), 0.0)
    scale = max(dWext, dWp + dUel + dWemit, 1.0e-30)

    U_values = np.asarray([float(row["Uapp"]) for row in values], dtype=float)
    F_values = np.asarray([float(row["Ftop"]) for row in values], dtype=float)
    if np.ptp(U_values) > 1.0e-30 and len(values) >= 3:
        tangent = abs(float(np.polyfit(U_values, F_values, 1)[0]))
    else:
        tangent = 0.0

    force_fraction = abs(float(last["Ftop"])) / max(
        abs(float(peak_force)), 1.0e-30
    )
    normalized_tangent = tangent / max(
        abs(float(stiffness_reference)), 1.0e-30
    )
    force_span_fraction = float(np.ptp(F_values)) / max(
        float(np.max(np.abs(F_values))), 1.0e-30
    )

    return {
        "dWext": dWext,
        "dUel": dUel,
        "dWp": dWp,
        "dWemit": dWemit,
        "plastic_fraction": dWp / scale,
        "elastic_fraction": dUel / scale,
        "force_fraction": force_fraction,
        "normalized_tangent": normalized_tangent,
        "force_span_fraction": force_span_fraction,
    }


def _substep_stagnation_metrics(window, args, **kwargs):
    """Recognize a plastic plateau after bounded severe adaptive subdivision."""
    required = max(
        int(getattr(args, "plastic_flow_stagnation_substeps", 128) or 128),
        16,
    )
    if len(window) < required:
        return None

    values = list(window)[-required:]
    first = values[0]
    last = values[-1]

    progress_start = float(first.get("nominal_progress_start", 0.0))
    progress_end = float(last.get("nominal_progress_end", progress_start))
    nominal_span = max(progress_end - progress_start, 0.0)
    fractions = np.asarray(
        [
            max(
                float(row.get("nominal_progress_end", 0.0))
                - float(row.get("nominal_progress_start", 0.0)),
                0.0,
            )
            for row in values
        ],
        dtype=float,
    )
    max_fraction = float(np.max(fractions)) if fractions.size else float("inf")
    median_fraction = (
        float(np.median(fractions)) if fractions.size else float("inf")
    )
    fraction_limit = max(
        float(
            getattr(
                args,
                "plastic_flow_stagnation_max_trial_fraction",
                1.0e-6,
            )
            or 1.0e-6
        ),
        np.finfo(float).tiny,
    )

    a_values = np.asarray([float(row["a_tip"]) for row in values], dtype=float)
    j_values = np.asarray(
        [max(float(row["J_positive"]), 0.0) for row in values], dtype=float
    )
    B_values = np.asarray([float(row["B"]) for row in values], dtype=float)
    lambda_values = np.asarray(
        [max(float(row["lambda_c"]), 0.0) for row in values], dtype=float
    )
    fire_values = np.asarray(
        [float(row["n_fire"]) for row in values], dtype=float
    )

    crack_span = float(np.ptp(a_values))
    j_max = float(np.max(j_values)) if j_values.size else 0.0
    j_span_fraction = float(np.ptp(j_values)) / max(j_max, 1.0e-30)
    positive_dB = float(np.sum(np.maximum(np.diff(B_values), 0.0)))
    n_fire = int(np.count_nonzero(fire_values > 0.0))

    energy = _window_energy_metrics(
        values,
        peak_force=float(kwargs.get("peak_force", 0.0)),
        stiffness_reference=float(kwargs.get("stiffness_reference", 0.0)),
    )

    cumulative_Wp = max(float(kwargs.get("cumulative_Wp", 0.0)), 0.0)
    cumulative_Uel = max(float(kwargs.get("cumulative_Uel", 0.0)), 0.0)
    cumulative_Wemit = max(float(kwargs.get("cumulative_Wemit", 0.0)), 0.0)
    cumulative_scale = max(
        cumulative_Wp + cumulative_Uel + cumulative_Wemit,
        1.0e-30,
    )
    cumulative_plastic_fraction = cumulative_Wp / cumulative_scale

    min_plastic = float(
        getattr(args, "plastic_flow_min_plastic_fraction", 0.90) or 0.90
    )
    min_cumulative = float(
        getattr(
            args,
            "plastic_flow_min_cumulative_plastic_fraction",
            0.90,
        )
        or 0.90
    )
    max_elastic = float(
        getattr(args, "plastic_flow_max_elastic_fraction", 0.05) or 0.05
    )
    max_tangent = float(
        getattr(args, "plastic_flow_max_tangent_fraction", 0.05) or 0.05
    )
    max_force = float(
        getattr(args, "plastic_flow_max_force_fraction", 0.10) or 0.10
    )
    plateau_tol = max(
        float(
            getattr(args, "plastic_flow_stagnation_plateau_rel_tol", 1.0e-3)
            or 1.0e-3
        ),
        0.0,
    )

    severe_substepping = max_fraction <= fraction_limit
    capacity_collapsed = (
        energy["force_fraction"] <= max_force
        or energy["normalized_tangent"] <= max_tangent
        or (
            j_span_fraction <= plateau_tol
            and energy["force_span_fraction"] <= plateau_tol
        )
    )

    criteria = {
        "no_crack_event_in_window": n_fire == 0,
        "negligible_crack_extension": crack_span
        < (
            float(
                getattr(args, "plastic_flow_max_da_fraction", 0.1) or 0.1
            )
            * max(float(kwargs.get("da_phys", 0.0)), 1.0e-30)
        ),
        "plastic_accommodation_dominant": (
            energy["plastic_fraction"] >= min_plastic
            or cumulative_plastic_fraction >= min_cumulative
        ),
        "elastic_storage_flat": energy["elastic_fraction"] <= max_elastic,
        "load_carrying_capacity_collapsed": capacity_collapsed,
        "adaptive_substep_stagnation": severe_substepping,
    }

    return {
        "schema": AUDIT_SCHEMA,
        "terminal_classifier_model_id": MODEL_ID,
        "terminal_basis": "collapsed_adaptive_substep_stagnation",
        "criteria": criteria,
        "criteria_pass": all(criteria.values()),
        "window_first_step": int(first["step"]),
        "window_last_step": int(last["step"]),
        "classification_window_steps": len(values),
        "classification_window_nominal_increment_span": nominal_span,
        "window_first_nominal_progress": progress_start,
        "window_last_nominal_progress": progress_end,
        "crack_extension_window_m": crack_span,
        "cleavage_event_count_window": n_fire,
        "J_tip_positive_max_window_J_per_m2": j_max,
        "J_tip_positive_relative_span_window": j_span_fraction,
        "cleavage_action_increment_window": positive_dB,
        "lambda_cleave_max_window_per_s": (
            float(np.max(lambda_values)) if lambda_values.size else 0.0
        ),
        "B_final": float(last["B"]),
        "W_external_increment_window_J_per_m": energy["dWext"],
        "U_elastic_change_window_J_per_m": energy["dUel"],
        "W_bulk_plastic_increment_window_J_per_m": energy["dWp"],
        "W_tip_emit_increment_window_J_per_m": energy["dWemit"],
        "plastic_work_fraction_window": energy["plastic_fraction"],
        "elastic_storage_fraction_window": energy["elastic_fraction"],
        "cumulative_plastic_fraction": cumulative_plastic_fraction,
        "reaction_force_fraction_of_peak": energy["force_fraction"],
        "normalized_tangent_stiffness": energy["normalized_tangent"],
        "reaction_force_relative_span_window": energy["force_span_fraction"],
        "maximum_accepted_trial_fraction_window": max_fraction,
        "median_accepted_trial_fraction_window": median_fraction,
        "stagnation_trial_fraction_limit": fraction_limit,
        "stagnation_substep_count_required": required,
        "projected_cleavage_action_is_diagnostic_only": True,
        "cleavage_action_growth_is_diagnostic_only": True,
        "finite_J_and_tip_stress_are_diagnostic_only": True,
        "terminal_requires_future_cleavage_inaccessibility": False,
        "terminal_requires_cleavage_clock_stall": False,
        "terminal_allows_prior_first_passage": True,
        "terminal_policy": (
            "bounded_severe_adaptive_substep_stagnation_plus_no_recent_"
            "crack_growth_plus_dominant_bulk_plastic_work_plus_flat_elastic_"
            "storage_plus_collapsed_load_carrying_response"
        ),
    }


def _campaign_metrics(window, metrics, *, Eprime: float):
    criteria_source = dict(metrics.get("criteria", {}))
    required_keys = list(_ACCEPTANCE_KEYS)
    if metrics.get("terminal_basis") == "collapsed_adaptive_substep_stagnation":
        required_keys.append("adaptive_substep_stagnation")

    missing = [key for key in required_keys if key not in criteria_source]
    if missing:
        raise RuntimeError(
            "plasticity-dominated terminal lost required criteria: "
            + ", ".join(missing)
        )

    acceptance = {key: bool(criteria_source[key]) for key in required_keys}
    values = list(window)
    last = values[-1]
    J_elastic = max(float(last.get("J_positive", 0.0)), 0.0)
    Eprime_safe = max(float(Eprime), 1.0e-30)
    K_elastic = math.sqrt(J_elastic * Eprime_safe) / 1.0e6

    metrics.update(
        {
            "schema": AUDIT_SCHEMA,
            "terminal_classifier_model_id": MODEL_ID,
            "criteria": acceptance,
            "criteria_pass": all(acceptance.values()),
            "diagnostic_criteria_before_campaign_override": criteria_source,
            "projected_cleavage_action_is_diagnostic_only": True,
            "cleavage_action_growth_is_diagnostic_only": True,
            "finite_J_and_tip_stress_are_diagnostic_only": True,
            "terminal_requires_future_cleavage_inaccessibility": False,
            "terminal_requires_cleavage_clock_stall": False,
            "terminal_allows_prior_first_passage": True,
            "terminal_policy": metrics.get(
                "terminal_policy",
                "no_recent_crack_growth_plus_dominant_bulk_plastic_work_"
                "plus_flat_elastic_storage_plus_collapsed_tangent_stiffness",
            ),
            "J_elastic_positive_terminal_J_per_m2": J_elastic,
            "K_elastic_equivalent_terminal_MPa_sqrt_m": K_elastic,
            "terminal_applied_opening_m": float(last.get("Uapp", 0.0)),
            "terminal_reaction_force_N": float(last.get("Ftop", 0.0)),
            "terminal_cleavage_action_B": float(last.get("B", 0.0)),
            "terminal_cleavage_rate_per_s": max(
                float(last.get("lambda_c", 0.0)), 0.0
            ),
        }
    )
    if "nominal_progress_end" in last:
        metrics["terminal_nominal_progress"] = float(
            last["nominal_progress_end"]
        )
    return metrics


def load_transformed_sharp_front() -> ModuleType:
    original_block = _v1042._TERMINAL_BLOCK
    _v1042._TERMINAL_BLOCK = _campaign_terminal_block(original_block)
    try:
        module = _load_hazard_terminal_module()
    finally:
        _v1042._TERMINAL_BLOCK = original_block

    if getattr(module, "_v1044_campaign_terminal_wrapped", False):
        return module

    original_metrics = module._v1042_terminal_metrics

    @wraps(original_metrics)
    def wrapped(window, args, **kwargs):
        metrics = original_metrics(window, args, **kwargs)
        if metrics is None:
            metrics = _substep_stagnation_metrics(
                window,
                args,
                **kwargs,
            )
        if metrics is None:
            return None
        return _campaign_metrics(
            window,
            metrics,
            Eprime=float(kwargs.get("Eprime", 0.0)),
        )

    module._v1042_terminal_metrics = wrapped
    module._v1044_campaign_terminal_wrapped = True
    module._v1044_campaign_terminal_model_id = MODEL_ID
    module._v1044_campaign_terminal_audit_schema = AUDIT_SCHEMA
    return module


__all__ = [
    "AUDIT_SCHEMA",
    "MODEL_ID",
    "_campaign_metrics",
    "_campaign_terminal_block",
    "_substep_stagnation_metrics",
    "load_transformed_sharp_front",
]
