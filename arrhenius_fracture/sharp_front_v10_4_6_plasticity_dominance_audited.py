"""v10.4.6 campaign entry with a bounded, physically gated plastic terminal.

The constitutive model, FEM equilibrium, directional configurational J,
Arrhenius first-passage law, and fracture event-energy gate are unchanged.
This entry changes only the severe-adaptive-substep campaign terminal.

The fallback is accepted only when all of the following hold over the bounded
window:

* severe adaptive subdivision;
* no cleavage event and negligible crack extension;
* a plateau in directional J and reaction-force response; and
* cumulative bulk-plastic work contributes at least the configured minimum
  cumulative plastic fraction (default 0.90) of cumulative accepted internal
  work partition.

A merely positive cumulative plastic work is not sufficient.
"""
from __future__ import annotations

from functools import wraps
import json
import math
from pathlib import Path
import sys

from . import plastic_flow_campaign_terminal_v1044 as _terminal
from . import sharp_front_v10_4_4_plasticity_dominated_audited as _v1044

MODEL_ID = "v10.4.6_cumulative_plastic_dominance_plateau_terminal"
TERMINAL_BASIS = "collapsed_adaptive_substep_stagnation_v1046"

_BASE_SUBSTEP_METRICS = _terminal._substep_stagnation_metrics
_BASE_CAMPAIGN_METRICS = _terminal._campaign_metrics


def _dominance_substep_metrics(window, args, **kwargs):
    metrics = _BASE_SUBSTEP_METRICS(window, args, **kwargs)
    if metrics is None:
        return None

    original = dict(metrics.get("criteria", {}))
    cumulative_fraction = max(
        float(metrics.get("cumulative_plastic_fraction", 0.0)),
        0.0,
    )
    min_cumulative = float(
        getattr(
            args,
            "plastic_flow_min_cumulative_plastic_fraction",
            0.90,
        )
        or 0.90
    )
    cumulative_dominant = (
        math.isfinite(cumulative_fraction)
        and cumulative_fraction >= min_cumulative
    )

    acceptance = {
        "no_crack_event_in_window": bool(
            original.get("no_crack_event_in_window", False)
        ),
        "negligible_crack_extension": bool(
            original.get("negligible_crack_extension", False)
        ),
        "cumulative_bulk_plastic_work_dominant": bool(cumulative_dominant),
        "load_carrying_response_plateau": bool(
            original.get("load_carrying_capacity_collapsed", False)
        ),
        "adaptive_substep_stagnation": bool(
            original.get("adaptive_substep_stagnation", False)
        ),
    }

    metrics.update(
        {
            "terminal_basis": TERMINAL_BASIS,
            "terminal_classifier_model_id": MODEL_ID,
            "criteria": acceptance,
            "criteria_pass": all(acceptance.values()),
            "v10_4_4_stagnation_criteria_diagnostics": original,
            "cumulative_plastic_fraction_acceptance": cumulative_fraction,
            "minimum_cumulative_plastic_fraction_acceptance": min_cumulative,
            "incremental_plastic_fraction_role": (
                "diagnostic_only_for_severe_substep_fallback"
            ),
            "incremental_elastic_fraction_role": (
                "diagnostic_only_for_severe_substep_fallback"
            ),
            "terminal_policy": (
                "bounded_severe_adaptive_substep_stagnation_plus_no_recent_"
                "crack_growth_plus_flat_J_and_force_response_plus_cumulative_"
                "bulk_plastic_work_fraction_at_or_above_configured_threshold"
            ),
        }
    )
    return metrics


def _dominance_campaign_metrics(window, metrics, *, Eprime: float):
    if metrics.get("terminal_basis") != TERMINAL_BASIS:
        return _BASE_CAMPAIGN_METRICS(window, metrics, Eprime=Eprime)

    acceptance = dict(metrics.get("criteria", {}))
    required = (
        "no_crack_event_in_window",
        "negligible_crack_extension",
        "cumulative_bulk_plastic_work_dominant",
        "load_carrying_response_plateau",
        "adaptive_substep_stagnation",
    )
    missing = [key for key in required if key not in acceptance]
    if missing:
        raise RuntimeError(
            "v10.4.6 plateau terminal lost required criteria: "
            + ", ".join(missing)
        )

    surrogate = dict(metrics)
    surrogate["terminal_basis"] = "collapsed_adaptive_substep_stagnation"
    surrogate["criteria"] = {
        "no_crack_event_in_window": acceptance["no_crack_event_in_window"],
        "negligible_crack_extension": acceptance["negligible_crack_extension"],
        "plastic_accommodation_dominant": acceptance[
            "cumulative_bulk_plastic_work_dominant"
        ],
        "elastic_storage_flat": True,
        "load_carrying_capacity_collapsed": acceptance[
            "load_carrying_response_plateau"
        ],
        "adaptive_substep_stagnation": acceptance[
            "adaptive_substep_stagnation"
        ],
    }

    result = _BASE_CAMPAIGN_METRICS(window, surrogate, Eprime=Eprime)
    result.update(
        {
            "terminal_basis": TERMINAL_BASIS,
            "terminal_classifier_model_id": MODEL_ID,
            "criteria": acceptance,
            "criteria_pass": all(bool(value) for value in acceptance.values()),
            "terminal_policy": metrics["terminal_policy"],
            "incremental_plastic_fraction_role": metrics[
                "incremental_plastic_fraction_role"
            ],
            "incremental_elastic_fraction_role": metrics[
                "incremental_elastic_fraction_role"
            ],
            "v10_4_4_stagnation_criteria_diagnostics": metrics[
                "v10_4_4_stagnation_criteria_diagnostics"
            ],
            "cumulative_plastic_fraction_acceptance": metrics[
                "cumulative_plastic_fraction_acceptance"
            ],
            "minimum_cumulative_plastic_fraction_acceptance": metrics[
                "minimum_cumulative_plastic_fraction_acceptance"
            ],
        }
    )
    return result


def _load_transformed_sharp_front_v1046():
    """Evaluate the physical fallback after an absent or failed primary terminal."""
    module = _terminal.load_transformed_sharp_front()
    if getattr(module, "_v1046_plastic_dominance_fallback_wrapped", False):
        return module

    primary_metrics = module._v1042_terminal_metrics

    @wraps(primary_metrics)
    def wrapped(window, args, **kwargs):
        primary = primary_metrics(window, args, **kwargs)
        if primary is not None and bool(primary.get("criteria_pass", False)):
            return primary

        fallback = _dominance_substep_metrics(window, args, **kwargs)
        if fallback is not None and bool(fallback.get("criteria_pass", False)):
            return _dominance_campaign_metrics(
                window,
                fallback,
                Eprime=float(kwargs.get("Eprime", 0.0)),
            )

        return primary

    module._v1042_terminal_metrics = wrapped
    module._v1046_plastic_dominance_fallback_wrapped = True
    module._v1046_plastic_dominance_terminal_model_id = MODEL_ID
    return module


def _rewrite_v1046_audits(root: Path) -> None:
    model_path = root / "v10_4_bulk_coupled_model_audit.json"
    payload = json.loads(model_path.read_text()) if model_path.is_file() else {}
    payload.update(
        {
            "schema": MODEL_ID,
            "plasticity_plateau_terminal_model": MODEL_ID,
            "severe_substep_fallback_window_steps": 128,
            "severe_substep_fallback_acceptance": [
                "no_crack_event_in_window",
                "negligible_crack_extension",
                "cumulative_bulk_plastic_work_fraction_at_or_above_threshold",
                "flat_directional_J_and_reaction_force_response",
                "accepted_trial_fraction_at_or_below_1e-6_for_bounded_window",
            ],
            "severe_substep_positive_plastic_work_is_sufficient": False,
            "severe_substep_incremental_energy_ratios_role": "diagnostic_only",
            "severe_substep_cumulative_plastic_fraction_default": 0.90,
            "severe_substep_fallback_arbitration": (
                "evaluate_when_standard_terminal_is_none_or_criteria_pass_false"
            ),
            "fracture_hazard_unchanged": True,
            "fracture_event_energy_gate_unchanged": True,
            "bulk_plastic_work_enters_fracture_hazard": False,
            "bulk_plastic_work_enters_fracture_energy_gate": False,
        }
    )
    model_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    terminal_path = root / "plastic_flow_terminal_audit.json"
    if terminal_path.is_file():
        terminal = json.loads(terminal_path.read_text())
        terminal.update(
            {
                "campaign_model_id": MODEL_ID,
                "campaign_terminal_model_id": MODEL_ID,
                "severe_substep_positive_plastic_work_is_sufficient": False,
                "severe_substep_incremental_energy_ratios_role": "diagnostic_only",
                "severe_substep_fallback_arbitration": (
                    "standard_terminal_failed_then_physical_dominance_fallback_accepted"
                ),
            }
        )
        terminal_path.write_text(
            json.dumps(terminal, indent=2, sort_keys=True) + "\n"
        )


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)

    old_loader = _v1044.load_transformed_sharp_front
    _v1044.load_transformed_sharp_front = _load_transformed_sharp_front_v1046
    try:
        print(
            "  v10.4.6 physical plateau terminal: unchanged full-field plasticity "
            "and sharp-fracture physics; after 128 accepted substeps at severe "
            "adaptive subdivision, a case terminates only when crack growth is "
            "absent, directional J and reaction force have plateaued, and the "
            "cumulative bulk-plastic work fraction meets the configured threshold "
            "(default 0.90); merely positive plastic work is insufficient"
        )
        result = _v1044.main(args)
    finally:
        _v1044.load_transformed_sharp_front = old_loader

    out = _v1044._option_value(args, "--out")
    if out:
        _rewrite_v1046_audits(Path(out))
    return result


if __name__ == "__main__":
    main()
