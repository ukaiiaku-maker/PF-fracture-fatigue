"""Predictor-only outer block proposal for the v10.2.30 forward marcher.

The forward committer is authoritative and may return partial consumed cycles.
This selector therefore performs no private constitutive integration and imposes
no raw population-count target.  It supplies a bounded proposal only.
"""
from __future__ import annotations

import math
import os
from typing import Any

from .persistent_site_vhcf_selector_v10229 import attach_prediction_context


MODEL_ID = "v10.2.30_forward_marcher_predictor_only_block_v1"


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
    return max(value, minimum)


def select_forward_block(
    controller,
    prediction,
    user_block_cycles: float | None,
    linear_diagnostic: dict[str, Any],
) -> dict[str, Any]:
    engine = getattr(prediction, "_v10229_vhcf_engine", None)
    if engine is None or not bool(
        getattr(engine, "persistent_site_cyclic_v10229", False)
    ):
        return dict(linear_diagnostic)

    cfg = controller.cfg
    requested = float(
        user_block_cycles
        if user_block_cycles is not None
        else getattr(cfg, "block_cycles", 1.0)
    )
    max_block = float(getattr(cfg, "max_block_cycles", requested))
    minimum = max(float(getattr(cfg, "min_block_cycles", 0.0)), 0.0)
    mode = str(
        getattr(cfg, "cycle_block_mode", "requested_cap") or "requested_cap"
    ).lower()
    physical_cap = (
        max_block
        if mode in ("hazard", "hazard_limited", "rate", "auto")
        else min(requested, max_block)
    )
    proposal_cap = _env_float(
        "V10230_FORWARD_OUTER_PROPOSAL_CYCLES", 1.0e6, minimum
    )
    selected = max(min(physical_cap, proposal_cap), minimum)
    if not math.isfinite(selected) or selected <= 0.0:
        return dict(linear_diagnostic)

    audit = {
        "schema": MODEL_ID,
        "search_strategy": "predictor_only_forward_marcher_authoritative",
        "cycles": float(selected),
        "limiter": "forward_marcher_proposal_cap",
        "physical_cap_cycles": float(physical_cap),
        "proposal_cap_cycles": float(proposal_cap),
        "private_trial_evaluations": 0,
        "raw_population_targets_used": False,
        "committer_authoritative": True,
        "partial_return_allowed": True,
    }
    engine._v10229_last_vhcf_block_audit = audit
    return {
        "cycles": float(selected),
        "limiter": "forward_marcher_proposal_cap",
        "unlimited_cycles": float(physical_cap),
        "candidate_limits": {
            "physical_cap_cycles": float(physical_cap),
            "forward_marcher_proposal_cap": float(proposal_cap),
            "forward_marcher_selected_cycles": float(selected),
        },
    }


select_nonlinear_block = select_forward_block


__all__ = [
    "MODEL_ID",
    "attach_prediction_context",
    "select_forward_block",
    "select_nonlinear_block",
]
