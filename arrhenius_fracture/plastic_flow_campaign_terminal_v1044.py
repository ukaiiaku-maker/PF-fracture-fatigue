"""Campaign terminal for full-field bulk-plasticity fracture sweeps.

The fracture physics is unchanged.  This overlay changes only campaign
termination and reporting:

* a case may terminate after a prior first-passage event if a later accepted
  loading window has no crack event or meaningful crack extension;
* projected future cleavage action and cleavage-action growth remain recorded
  diagnostics but are not terminal vetoes;
* finite elastic J and tip stress remain diagnostics rather than zero gates;
* the accepted terminal requires sustained plastic accommodation, flat elastic
  storage, and collapsed incremental load-carrying stiffness.

The legacy ``PLASTIC_FLOW`` marker is retained for existing tooling.  A new
``PLASTICITY_DOMINATED`` marker records whether the terminal occurred before
first passage or after partial sharp-fracture growth.
"""
from __future__ import annotations

from functools import wraps
import math
from types import ModuleType

from . import plastic_flow_terminal_v1042 as _v1042
from .plastic_flow_hazard_terminal_v1043 import (
    load_transformed_sharp_front as _load_hazard_terminal_module,
)

MODEL_ID = "v10.4.4_plasticity_dominated_campaign_terminal"
AUDIT_SCHEMA = "v10.4.4_plasticity_dominated_campaign_terminal_audit_v1"

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


def _campaign_metrics(window, metrics, *, Eprime: float):
    criteria_source = dict(metrics.get("criteria", {}))
    missing = [key for key in _ACCEPTANCE_KEYS if key not in criteria_source]
    if missing:
        raise RuntimeError(
            "plasticity-dominated terminal lost required criteria: "
            + ", ".join(missing)
        )

    acceptance = {key: bool(criteria_source[key]) for key in _ACCEPTANCE_KEYS}
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
            "terminal_policy": (
                "no_recent_crack_growth_plus_dominant_bulk_plastic_work_"
                "plus_flat_elastic_storage_plus_collapsed_tangent_stiffness"
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
    # The v10.4.3 chain ultimately invokes v10.4.2.transform_source.  Patch only
    # the terminal block during that one source transformation, then restore the
    # module global so importing v10.4.4 cannot alter other public entries.
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
    "load_transformed_sharp_front",
]
