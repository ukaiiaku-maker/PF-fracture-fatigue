"""Phase-resolved one-cycle Poincare map for the v10.2.30 high-cycle engine.

The map advances only the real persistent-site constitutive state through one
complete loading period.  It uses the engine's installed plastic half-step,
signed transport, shielding, backstress, blunting, and cleavage-rate functions.
It does not translate the crack, consume a stochastic threshold, draw RNG state,
or create a pending geometry event.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from typing import Any

import numpy as np

from .persistent_site_high_cycle_state_v10230 import (
    ActiveStateSnapshot,
    capture_ledgers,
    capture_stochastic_state,
    geometry_signature,
    ledger_delta,
    restore_active_state,
    serialize_active_state,
    stochastic_state_equal,
)
from .persistent_site_reversible_transport_v10230 import (
    resolved_mobile_stress_profiles,
)


MODEL_ID = "v10.2.30_phase_resolved_one_cycle_poincare_map_v1"


@dataclass
class PoincareResult:
    engine: Any
    state_start: ActiveStateSnapshot
    state_end: ActiveStateSnapshot
    hazard_action_per_cycle: float
    normalized_clock_per_cycle: float
    phase_hazard_action: tuple[float, ...]
    ledger_delta_per_cycle: dict[str, float]
    transition_signature_start: tuple[Any, ...]
    transition_signature_end: tuple[Any, ...]
    stochastic_state_preserved: bool
    geometry_preserved: bool
    plastic_totals: dict[str, float]
    phase_trace: tuple[dict[str, Any], ...]


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _sum_numeric(target: dict[str, float], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            target[key] = target.get(key, 0.0) + float(value)


def transition_signature(engine) -> tuple[Any, ...]:
    mpz = engine.mpz
    tau = np.asarray(
        getattr(engine, "_anisotropic_tau_signed_Pa", np.empty(0)), dtype=float
    ).reshape(-1)
    signs = tuple(int(value) for value in np.sign(tau))
    source_hazard = max(
        _finite(getattr(mpz, "continuum_source_last_aggregate_hazard_s", 0.0)),
        0.0,
    )
    return (
        signs,
        bool(source_hazard > 0.0),
        bool(_finite(getattr(mpz, "retained_count", 0.0)) > 1.0e-18),
        bool(_finite(getattr(mpz, "escaped_total", 0.0)) > 0.0),
        int(getattr(engine, "n_adv", 0)),
    )


def one_cycle_map(
    engine,
    controller,
    waveform,
    temperature_K: float,
    *,
    state: ActiveStateSnapshot | None = None,
    commit: bool = False,
) -> PoincareResult:
    """Evaluate one exact waiting cycle without consuming first passage."""
    if getattr(engine, "_energy_gate_pending", None) is not None:
        raise RuntimeError("cannot evaluate a Poincare map with a pending crack event")
    work = engine if commit else copy.deepcopy(engine)
    if state is not None:
        restore_active_state(work, state)

    geometry_before = geometry_signature(work)
    stochastic_before = capture_stochastic_state(work)
    ledgers_before = capture_ledgers(work)
    start = serialize_active_state(
        work, waveform=waveform, temperature_K=temperature_K
    )
    signature_start = transition_signature(work)

    phases = np.asarray(controller._phases(), dtype=float)
    if phases.size < 1:
        raise ValueError("Poincare map requires at least one waveform phase")
    K_values = np.asarray(waveform.K_phase(phases), dtype=float)
    if K_values.shape != phases.shape:
        raise ValueError("waveform phase values do not match controller phases")
    dt_phase = float(waveform.period_s) / float(phases.size)
    if not math.isfinite(dt_phase) or dt_phase <= 0.0:
        raise ValueError("waveform period must be positive and finite")

    action = 0.0
    phase_action: list[float] = []
    phase_trace: list[dict[str, Any]] = []
    plastic_totals: dict[str, float] = {}
    for phase_index, (phase_value, K_value) in enumerate(zip(phases, K_values)):
        K_signed = float(K_value)
        K = max(K_signed, 0.0)
        half = 0.5 * dt_phase
        phase_ledgers_before = capture_ledgers(work)
        radius_start = float(work.r_eff())
        shielding_start = float(work.K_shield()) if hasattr(work, "K_shield") else 0.0
        signed_local_stress = (
            (K_signed-shielding_start) /
            math.sqrt(2.0*math.pi*max(radius_start, 1.0e-300))
        )
        mobile_start = _finite(getattr(work.mpz, "mobile_count", 0.0))
        retained_start = _finite(getattr(work.mpz, "retained_count", 0.0))
        if bool(getattr(work.mpz, "_reversible_transport_installed", False)):
            work.mpz._reversible_transport_K_signed_Pa_sqrt_m = K_signed
            work.mpz._reversible_tip_radius_m = float(work.r_eff())
        sigma_start = max(float(work.sigma_tip(K)), 0.0)
        glide_profiles = (
            resolved_mobile_stress_profiles(work.mpz, K_signed, radius_start)
            if (
                bool(getattr(work.mpz, "_reversible_transport_installed", False))
                and hasattr(work.mpz, "cfg")
                and hasattr(work.mpz, "dx")
            )
            else None
        )
        first = work._plastic_half_step(half, float(temperature_K), sigma_start)
        if bool(getattr(work.mpz, "_reversible_transport_installed", False)):
            work.mpz._reversible_tip_radius_m = float(work.r_eff())
        sigma_mid = max(float(work.sigma_tip(K)), 0.0)
        lam_mid, _raw_mid, _barrier_mid = work.lambda_cleave(
            sigma_mid, float(temperature_K)
        )
        lam_mid = max(_finite(lam_mid), 0.0)
        dH = lam_mid * dt_phase
        second = work._plastic_half_step(half, float(temperature_K), sigma_mid)
        _sum_numeric(plastic_totals, dict(first))
        _sum_numeric(plastic_totals, dict(second))
        action += dH
        phase_action.append(dH)
        work.t = _finite(getattr(work, "t", 0.0)) + dt_phase
        stress_cap = _finite(getattr(getattr(work, "f", None), "sigma_cap", 0.0))
        phase_trace.append({
            "phase_index": int(phase_index),
            "phase_rad": float(phase_value),
            "dt_s": float(dt_phase),
            "native_time_end_s": float(work.t),
            "K_signed_Pa_sqrt_m": K_signed,
            "K_after_contact_clip_Pa_sqrt_m": K,
            "signed_local_stress_before_opening_clip_Pa": float(signed_local_stress),
            "opening_stress_start_Pa": float(sigma_start),
            "opening_stress_mid_Pa": float(sigma_mid),
            "tip_radius_start_m": radius_start,
            "tip_radius_end_m": float(work.r_eff()),
            "K_shield_start_Pa_sqrt_m": shielding_start,
            "K_shield_end_Pa_sqrt_m": (
                float(work.K_shield()) if hasattr(work, "K_shield") else 0.0
            ),
            "mobile_count_start": mobile_start,
            "mobile_count_end": _finite(getattr(work.mpz, "mobile_count", 0.0)),
            "retained_count_start": retained_start,
            "retained_count_end": _finite(getattr(work.mpz, "retained_count", 0.0)),
            "elementary_opening_rate_s": float(max(_finite(_raw_mid), 0.0)),
            "completed_event_rate_s": float(lam_mid),
            "effective_opening_barrier_J": float(_finite(_barrier_mid)),
            "event_activity": float(dH),
            "stress_cap_occupied": bool(
                stress_cap > 0.0 and sigma_mid >= stress_cap*(1.0-1.0e-12)
            ),
            "glide_applied_stress_min_Pa": (
                float(np.min(glide_profiles["applied_Pa"]))
                if glide_profiles is not None else math.nan
            ),
            "glide_applied_stress_max_Pa": (
                float(np.max(glide_profiles["applied_Pa"]))
                if glide_profiles is not None else math.nan
            ),
            "glide_internal_stress_min_Pa": (
                float(np.min(glide_profiles["local_internal_Pa"]))
                if glide_profiles is not None else math.nan
            ),
            "glide_internal_stress_max_Pa": (
                float(np.max(glide_profiles["local_internal_Pa"]))
                if glide_profiles is not None else math.nan
            ),
            "glide_total_stress_min_Pa": (
                float(np.min(glide_profiles["total_Pa"]))
                if glide_profiles is not None else math.nan
            ),
            "glide_total_stress_max_Pa": (
                float(np.max(glide_profiles["total_Pa"]))
                if glide_profiles is not None else math.nan
            ),
            "glide_internal_stress_model": str(getattr(
                work.mpz, "_reversible_mobile_internal_stress_model", "not_installed"
            )),
            "first_half_step": dict(first),
            "second_half_step": dict(second),
            "ledger_delta": ledger_delta(phase_ledgers_before, capture_ledgers(work)),
        })

    if hasattr(work, "K_prev"):
        work.K_prev = float(waveform.Kmax)
    geometry_after = geometry_signature(work)
    stochastic_after = capture_stochastic_state(work)
    stochastic_preserved = stochastic_state_equal(stochastic_before, stochastic_after)
    if not stochastic_preserved:
        raise RuntimeError("one-cycle Poincare map altered stochastic threshold or RNG state")
    if geometry_after != geometry_before:
        raise RuntimeError("one-cycle Poincare map altered crack geometry")
    if getattr(work, "_energy_gate_pending", None) is not None:
        raise RuntimeError("one-cycle Poincare map created a pending crack event")

    end = serialize_active_state(
        work, waveform=waveform, temperature_K=temperature_K
    )
    threshold = max(
        _finite(getattr(work, "hazard_threshold_action", 1.0), 1.0), 1.0e-300
    )
    return PoincareResult(
        engine=work,
        state_start=start,
        state_end=end,
        hazard_action_per_cycle=float(action),
        normalized_clock_per_cycle=float(action / threshold),
        phase_hazard_action=tuple(float(value) for value in phase_action),
        ledger_delta_per_cycle=ledger_delta(ledgers_before, capture_ledgers(work)),
        transition_signature_start=signature_start,
        transition_signature_end=transition_signature(work),
        stochastic_state_preserved=stochastic_preserved,
        geometry_preserved=geometry_after == geometry_before,
        plastic_totals=plastic_totals,
        phase_trace=tuple(phase_trace),
    )


__all__ = [
    "MODEL_ID",
    "PoincareResult",
    "one_cycle_map",
    "transition_signature",
]
