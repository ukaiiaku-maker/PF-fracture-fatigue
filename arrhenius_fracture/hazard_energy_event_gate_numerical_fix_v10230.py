"""Numerical correction for the v10.2.30 event-length energy bracket.

The original coarse scan stopped when its first finite trial failed.  That is a
valid cumulative-arrest signal, but a smaller positive length can still lie
between zero and the first failed trial.  This adapter bisects that initial
interval using the exact same fixed-opening mechanics and hazard-derived
resistance.  It changes no material quantity.
"""
from __future__ import annotations

import copy
from typing import Any, Callable

from . import hazard_energy_event_gate_v10230 as _gate


MODEL_ID = "v10.2.30_zero_bracket_energy_length_refinement"


def make_zero_bracket_refined_length(
    original: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """Wrap ``energy_gate_event_length`` with a [0, first-fail] refinement."""

    def refined(*, kwargs: dict[str, Any], descriptor: dict[str, Any]) -> dict[str, Any]:
        result = original(kwargs=kwargs, descriptor=descriptor)
        proposal = max(float(descriptor.get("event_advance_m", 0.0)), 0.0)
        if proposal <= 0.0 or float(result.get("committed_event_length_m", 0.0)) > 0.0:
            result["zero_bracket_refinement_used"] = False
            return result

        rows = list(result.get("trial_rows", []) or [])
        positive_trials = [
            float(row.get("trial_length_m", 0.0))
            for row in rows
            if float(row.get("trial_length_m", 0.0)) > 0.0
        ]
        if not positive_trials:
            result["zero_bracket_refinement_used"] = False
            return result

        lo = 0.0
        hi = min(positive_trials)
        best: dict[str, Any] | None = None
        refinement_rows: list[dict[str, Any]] = []
        cfg = _gate.OBSERVER.config
        old_fraction = float(cfg.trial_fraction)
        try:
            cfg.trial_fraction = 1.0
            for _ in range(int(cfg.bisection_iterations)):
                mid = 0.5 * (lo + hi)
                trial_descriptor = copy.deepcopy(descriptor)
                trial_descriptor["event_advance_m"] = mid
                trial = original(kwargs=kwargs, descriptor=trial_descriptor)
                accepted = float(trial.get("committed_event_length_m", 0.0))
                admissible = accepted >= mid * (1.0 - 1.0e-12)
                refinement_rows.append(
                    {
                        "zero_bracket_midpoint_m": mid,
                        "zero_bracket_admissible": bool(admissible),
                        "zero_bracket_trial": {
                            key: value
                            for key, value in trial.items()
                            if key not in {"equilibrated_displacement", "trial_rows"}
                        },
                    }
                )
                if admissible:
                    lo = mid
                    best = trial
                else:
                    hi = mid
        finally:
            cfg.trial_fraction = old_fraction

        if best is None or lo <= 0.0:
            result["zero_bracket_refinement_used"] = True
            result["zero_bracket_refinement_rows"] = refinement_rows
            result["zero_bracket_upper_bound_m"] = hi
            return result

        displacement = best.get("equilibrated_displacement")
        merged = dict(best)
        merged.update(
            {
                "energy_gate_model_id": result.get("energy_gate_model_id"),
                "stochastic_proposed_event_length_m": proposal,
                "energy_admissible_event_length_m": lo,
                "committed_event_length_m": lo,
                "arrest_reason": "hazard_derived_energy_arrest",
                "zero_bracket_refinement_used": True,
                "zero_bracket_refinement_rows": refinement_rows,
                "zero_bracket_upper_bound_m": hi,
                "trial_rows": rows + list(best.get("trial_rows", []) or []),
            }
        )
        if displacement is not None:
            merged["equilibrated_displacement"] = displacement
        return merged

    refined.__name__ = "energy_gate_event_length_zero_bracket_refined"
    refined.__doc__ = original.__doc__
    return refined


__all__ = ["MODEL_ID", "make_zero_bracket_refined_length"]
