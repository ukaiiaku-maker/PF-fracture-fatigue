"""Feedback-controlled VHCF block routing for the v10.2.30 fatigue campaign.

The persistent-site source integrates emission with an implicit backstress-limited
solve. Raw mobile/retained counts are physical state and remain fully evolved and
audited, but an arbitrary count increment is not an appropriate cycle-step error
measure. This runtime patch therefore lets the exact state-coupled trial determine
that state while limiting blocks by cleavage action and by the existing
state-dependent cleavage stress/rate quadrature.

For low cleavage hazard the inherited integrator historically accepted a segment
without checking stress/rate variation whenever dB was below an absolute cutoff.
The launcher paired with this patch sets that cutoff to zero, so every positive
hazard interval must satisfy the existing sigma and log-lambda tolerances.
"""
from __future__ import annotations

import math
from typing import Any

from . import active_state_block_control_v10230 as _active
from . import persistent_site_coupled_hazard_v10229 as _coupled
from . import persistent_site_vhcf_selector_v10230 as _selector


MODEL_ID = "v10.2.30_feedback_state_vhcf_block_control_v1"
_INSTALLED = False
_ORIGINAL_STATE_TARGETS = None
_ORIGINAL_BLOCK_TARGETS = None


def _finite_positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0.0 else None


def feedback_state_targets(controller) -> dict[str, float]:
    """Return no raw-count subdivision targets.

    Mobile and retained populations still evolve in every provisional and
    committed integration. Their effects on blunting, shielding, tip stress and
    cleavage rate are resolved by the coupled quadrature tolerances.
    """
    return {}


def feedback_block_targets(controller, B_pre: float) -> dict[str, float]:
    """Limit the nonlinear outer block by the cleavage first-passage clock."""
    target = _finite_positive(getattr(controller.cfg, "target_dB", None))
    if target is None:
        return {}
    return {
        "cleavage_clock": min(target, max(1.0 - float(B_pre), 0.0)),
    }


def install_feedback_state_block_control() -> None:
    """Install active-state diagnostics plus feedback-based block controls."""
    global _INSTALLED
    global _ORIGINAL_STATE_TARGETS
    global _ORIGINAL_BLOCK_TARGETS
    if _INSTALLED:
        return
    _active.install_active_state_block_control()
    _ORIGINAL_STATE_TARGETS = _coupled._state_targets
    _ORIGINAL_BLOCK_TARGETS = _selector._targets
    _coupled._state_targets = feedback_state_targets
    _selector._targets = feedback_block_targets
    _INSTALLED = True


def restore_feedback_state_block_control() -> None:
    global _INSTALLED
    if not _INSTALLED:
        return
    _coupled._state_targets = _ORIGINAL_STATE_TARGETS
    _selector._targets = _ORIGINAL_BLOCK_TARGETS
    _INSTALLED = False


def audit_payload() -> dict[str, Any]:
    return {
        "schema": MODEL_ID,
        "installed": bool(_INSTALLED),
        "outer_block_metrics": ["cleavage_clock"],
        "inner_raw_population_targets": [],
        "mobile_retained_state_evolved": True,
        "mobile_retained_state_reported": True,
        "raw_population_counts_are_block_limiters": False,
        "coupled_sigma_tolerance_active": True,
        "coupled_log_lambda_tolerance_active": True,
        "low_hazard_absolute_dB_cutoff_must_be_zero": True,
        "persistent_source_physics_changed": False,
        "hazard_rate_changed": False,
        "event_length_law_changed": False,
        "energy_gate_changed": False,
    }


__all__ = [
    "MODEL_ID",
    "audit_payload",
    "feedback_block_targets",
    "feedback_state_targets",
    "install_feedback_state_block_control",
    "restore_feedback_state_block_control",
]
