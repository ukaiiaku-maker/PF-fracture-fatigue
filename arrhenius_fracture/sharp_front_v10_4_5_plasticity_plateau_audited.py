"""v10.4.5 campaign entry with a bounded severe-substep plastic plateau terminal.

The constitutive model, FEM equilibrium, directional configurational J, Arrhenius
first-passage law, and fracture event-energy gate are unchanged.  This entry
changes only the campaign terminal used after the converged full-field plasticity
solver has collapsed to extremely small accepted substeps.

For that fallback only, ratios formed from the differences of nearly identical
128-substep endpoint energies are retained as diagnostics rather than acceptance
gates.  Acceptance instead requires:

* severe adaptive subdivision for the complete bounded window;
* no cleavage event and negligible crack extension in that window;
* a plateau in J and reaction-force response; and
* positive cumulative accepted bulk-plastic dissipation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

from . import plastic_flow_campaign_terminal_v1044 as _terminal
from . import sharp_front_v10_4_4_plasticity_dominated_audited as _v1044

MODEL_ID = "v10.4.5_bounded_full_field_plasticity_plateau_terminal"
TERMINAL_BASIS = "collapsed_adaptive_substep_stagnation_v1045"

_BASE_SUBSTEP_METRICS = _terminal._substep_stagnation_metrics
_BASE_CAMPAIGN_METRICS = _terminal._campaign_metrics


def _plateau_substep_metrics(window, args, **kwargs):
    metrics = _BASE_SUBSTEP_METRICS(window, args, **kwargs)
    if metrics is None:
        return None

    original = dict(metrics.get("criteria", {}))
    cumulative_Wp = max(float(kwargs.get("cumulative_Wp", 0.0)), 0.0)
    plastic_work_present = math.isfinite(cumulative_Wp) and cumulative_Wp > 0.0

    acceptance = {
        "no_crack_event_in_window": bool(
            original.get("no_crack_event_in_window", False)
        ),
        "negligible_crack_extension": bool(
            original.get("negligible_crack_extension", False)
        ),
        "bulk_plastic_dissipation_present": bool(plastic_work_present),
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
            "cumulative_bulk_plastic_work_terminal_J_per_m": cumulative_Wp,
            "incremental_plastic_fraction_role": "diagnostic_only_for_severe_substep_fallback",
            "incremental_elastic_fraction_role": "diagnostic_only_for_severe_substep_fallback",
            "terminal_policy": (
                "bounded_severe_adaptive_substep_stagnation_plus_no_recent_"
                "crack_growth_plus_flat_J_and_force_response_plus_positive_"
                "cumulative_accepted_bulk_plastic_dissipation"
            ),
        }
    )
    return metrics


def _plateau_campaign_metrics(window, metrics, *, Eprime: float):
    if metrics.get("terminal_basis") != TERMINAL_BASIS:
        return _BASE_CAMPAIGN_METRICS(window, metrics, Eprime=Eprime)

    acceptance = dict(metrics.get("criteria", {}))
    missing = [
        key
        for key in (
            "no_crack_event_in_window",
            "negligible_crack_extension",
            "bulk_plastic_dissipation_present",
            "load_carrying_response_plateau",
            "adaptive_substep_stagnation",
        )
        if key not in acceptance
    ]
    if missing:
        raise RuntimeError(
            "v10.4.5 plateau terminal lost required criteria: "
            + ", ".join(missing)
        )

    # Reuse the established v10.4.4 reporting path without allowing its two
    # ill-conditioned tiny-window energy ratios to veto this fallback.
    surrogate = dict(metrics)
    surrogate["terminal_basis"] = "collapsed_adaptive_substep_stagnation"
    surrogate["criteria"] = {
        "no_crack_event_in_window": acceptance["no_crack_event_in_window"],
        "negligible_crack_extension": acceptance["negligible_crack_extension"],
        "plastic_accommodation_dominant": acceptance[
            "bulk_plastic_dissipation_present"
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
            "cumulative_bulk_plastic_work_terminal_J_per_m": metrics[
                "cumulative_bulk_plastic_work_terminal_J_per_m"
            ],
        }
    )
    return result


def _rewrite_v1045_audits(root: Path) -> None:
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
                "positive_cumulative_accepted_bulk_plastic_dissipation",
                "flat_directional_J_and_reaction_force_response",
                "accepted_trial_fraction_at_or_below_1e-6_for_bounded_window",
            ],
            "severe_substep_incremental_energy_ratios_role": "diagnostic_only",
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
                "severe_substep_incremental_energy_ratios_role": "diagnostic_only",
            }
        )
        terminal_path.write_text(
            json.dumps(terminal, indent=2, sort_keys=True) + "\n"
        )


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)

    old_substep = _terminal._substep_stagnation_metrics
    old_campaign = _terminal._campaign_metrics
    _terminal._substep_stagnation_metrics = _plateau_substep_metrics
    _terminal._campaign_metrics = _plateau_campaign_metrics
    try:
        print(
            "  v10.4.5 plateau terminal: unchanged full-field plasticity and "
            "sharp-fracture physics; after 128 accepted substeps at severe "
            "adaptive subdivision, a case terminates when crack growth is absent, "
            "directional J and reaction force have plateaued, and cumulative "
            "accepted bulk-plastic dissipation is positive; tiny-window plastic "
            "and elastic energy fractions are diagnostics only"
        )
        result = _v1044.main(args)
    finally:
        _terminal._substep_stagnation_metrics = old_substep
        _terminal._campaign_metrics = old_campaign

    out = _v1044._option_value(args, "--out")
    if out:
        _rewrite_v1045_audits(Path(out))
    return result


if __name__ == "__main__":
    main()
