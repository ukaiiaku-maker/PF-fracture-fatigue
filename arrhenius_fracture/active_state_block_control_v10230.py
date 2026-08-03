"""Active-state-only VHCF block controls for the v10.2.30 fatigue campaign.

The persistent-source model carries cumulative emitted and escaped totals as audit
ledgers. Those ledgers are not active state variables and must not limit adaptive
cycle blocks. This additive runtime patch keeps them in diagnostics while using
only the cleavage clock and the mobile/retained MPZ populations as block limits.
"""
from __future__ import annotations

import copy
import math
from typing import Any

from . import persistent_site_coupled_hazard_v10229 as _coupled_hazard
from . import persistent_site_vhcf_coupled_selector_v10230 as _coupled_selector
from . import persistent_site_vhcf_selector_v10230 as _selector


MODEL_ID = "v10.2.30_active_state_only_vhcf_block_control_v1"
_INSTALLED = False
_ORIGINAL_STATE_TARGETS = None
_ORIGINAL_BLOCK_TARGETS = None
_ORIGINAL_COUPLED_BLOCK_TRIAL = None


def _finite_positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0.0 else None


def active_state_targets(controller) -> dict[str, float]:
    """Return only populations that feed back on shielding and backstress."""
    cfg = controller.cfg
    targets: dict[str, float] = {}
    for key, attr in (
        ("mobile_count", "target_dN_mobile"),
        ("retained_count", "target_dN_store"),
    ):
        value = _finite_positive(getattr(cfg, attr, None))
        if value is not None:
            targets[key] = value
    return targets


def active_block_targets(controller, B_pre: float) -> dict[str, float]:
    """Return cleavage-clock and active-population nonlinear limits."""
    cfg = controller.cfg
    targets: dict[str, float] = {}
    target_dB = _finite_positive(getattr(cfg, "target_dB", None))
    if target_dB is not None:
        targets["cleavage_clock"] = min(
            target_dB,
            max(1.0 - float(B_pre), 0.0),
        )
    for key, attr in (
        ("mobile_pz", "target_dN_mobile"),
        ("stored_pz", "target_dN_store"),
    ):
        value = _finite_positive(getattr(cfg, attr, None))
        if value is not None:
            targets[key] = value
    return targets


def active_coupled_block_trial(
    controller,
    engine,
    waveform,
    temperature_K: float,
    prediction,
    cycles: float,
) -> dict[str, Any]:
    """Evaluate a private trial while separating active state from flux ledgers."""
    requested = max(float(cycles), 0.0)
    trial = copy.deepcopy(engine)
    mobile_pre = float(trial.mpz.mobile_count)
    retained_pre = float(trial.mpz.retained_count)
    escaped_pre = float(trial.mpz.escaped_total)
    emitted_pre = float(trial.mpz.emitted_total)

    coupled = _coupled_hazard.integrate_state_coupled_waveform(
        trial,
        controller,
        waveform,
        float(temperature_K),
        requested,
    )
    consumed = max(float(coupled.get("dt_consumed", 0.0)), 0.0) * max(
        float(waveform.frequency_Hz), 0.0
    )
    plastic = dict(coupled.get("plastic", {}))
    emitted = max(float(trial.mpz.emitted_total) - emitted_pre, 0.0)
    escaped = max(float(trial.mpz.escaped_total) - escaped_pre, 0.0)
    retained_delta = abs(float(trial.mpz.retained_count) - retained_pre)
    mobile_delta = abs(float(trial.mpz.mobile_count) - mobile_pre)

    return {
        "cycles_requested": requested,
        "cycles_consumed": consumed,
        "fired": bool(coupled.get("fired", False)),
        "metrics": {
            "cleavage_clock": max(float(coupled.get("dB", 0.0)), 0.0),
            "stored_pz": retained_delta,
            "mobile_pz": mobile_delta,
            "emitted_pz": emitted,
            "escape_pz": escaped,
        },
        "active_state_metrics": {
            "stored_pz": retained_delta,
            "mobile_pz": mobile_delta,
        },
        "diagnostic_flux_metrics": {
            "emitted_pz": emitted,
            "escape_pz": escaped,
            "trapped_pz": max(float(plastic.get("dN_trapped", 0.0)), 0.0),
            "released_pz": max(float(plastic.get("dN_released", 0.0)), 0.0),
            "escaped_plastic_pz": max(float(plastic.get("dN_escaped", 0.0)), 0.0),
        },
        "active_state_block_control_model_id": MODEL_ID,
        "cumulative_flux_ledgers_are_block_limiters": False,
        "internal_steps": int(coupled.get("microsteps", 0)),
        "coupled_hazard_accepted_segments": int(
            coupled.get("coupled_hazard_accepted_segments", 0)
        ),
        "coupled_hazard_rejected_splits": int(
            coupled.get("coupled_hazard_rejected_splits", 0)
        ),
        "coupled_hazard_log_lambda_span_decades": float(
            coupled.get("coupled_hazard_log_lambda_span_decades", 0.0)
        ),
    }


def install_active_state_block_control() -> None:
    """Install the additive v10.2.30 runtime routing patch."""
    global _INSTALLED
    global _ORIGINAL_STATE_TARGETS
    global _ORIGINAL_BLOCK_TARGETS
    global _ORIGINAL_COUPLED_BLOCK_TRIAL
    if _INSTALLED:
        return
    _ORIGINAL_STATE_TARGETS = _coupled_hazard._state_targets
    _ORIGINAL_BLOCK_TARGETS = _selector._targets
    _ORIGINAL_COUPLED_BLOCK_TRIAL = _coupled_selector._coupled_block_trial
    _coupled_hazard._state_targets = active_state_targets
    _selector._targets = active_block_targets
    _coupled_selector._coupled_block_trial = active_coupled_block_trial
    _INSTALLED = True


def restore_active_state_block_control() -> None:
    """Restore the inherited implementations for tests or non-v10.2.30 use."""
    global _INSTALLED
    if not _INSTALLED:
        return
    _coupled_hazard._state_targets = _ORIGINAL_STATE_TARGETS
    _selector._targets = _ORIGINAL_BLOCK_TARGETS
    _coupled_selector._coupled_block_trial = _ORIGINAL_COUPLED_BLOCK_TRIAL
    _INSTALLED = False


def audit_payload() -> dict[str, Any]:
    return {
        "schema": MODEL_ID,
        "installed": bool(_INSTALLED),
        "active_state_block_metrics": [
            "cleavage_clock",
            "mobile_pz",
            "stored_pz",
        ],
        "cumulative_flux_ledgers_reported": True,
        "cumulative_flux_ledgers_are_block_limiters": False,
        "persistent_source_physics_changed": False,
        "hazard_rate_changed": False,
        "event_length_law_changed": False,
        "energy_gate_changed": False,
    }


__all__ = [
    "MODEL_ID",
    "active_block_targets",
    "active_coupled_block_trial",
    "active_state_targets",
    "audit_payload",
    "install_active_state_block_control",
    "restore_active_state_block_control",
]
