"""v3 explicit-cycle wrapper with corrected reverse-drive diagnostics.

The physical cycle/hazard/event integration is the qualified v2 implementation.
This module replaces only the state class and phase diagnostic so that a
Burgers-sign spatial velocity is not mistaken for cyclic stress reversal.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

import v914_minimal_reversible_explicit as _v2
from v914_minimal_reversible_state_v3 import (
    MODEL_ID,
    MinimalReversibleEmergentGNDState,
)
from v914_reverse_drive_utils import reverse_drive_mask


# The v2 run function resolves these module globals at runtime.  Rebind them to
# the v3 state while preserving the exact cycle/event integration algorithm.
_v2.MinimalReversibleEmergentGNDState = MinimalReversibleEmergentGNDState
_v2.REVERSIBLE_STATE_MODEL_ID = MODEL_ID


def _diagnostic(
    state: MinimalReversibleEmergentGNDState,
    loading,
    cycle: int,
    phase: float,
    action: float,
    threshold: float,
    extension: float,
    record_type: str,
) -> dict[str, Any]:
    K = loading.K_at_phase(phase)
    shield = float(state.K_shield_MPa_sqrt_m())
    K_eff = max(K - shield, 0.0)
    radius = float(state.tip_radius_m())
    sigma = K_eff * 1e6 / math.sqrt(
        2 * math.pi * max(radius, state.c.b_m, 1e-30)
    )
    barrier = float(state.p.cleavage.barrier_eV(sigma, loading.temperature_K))
    rates = state.local_rates(K, loading.temperature_K)
    backstress = state.backstress_state()[2]
    transport_tau = np.asarray(
        rates["reversible_tau_transport_eff_Pa"], dtype=float
    )
    drive_factors = np.asarray(state.emission_drive_factors(), dtype=float)
    reverse_mask = reverse_drive_mask(transport_tau, drive_factors)
    emitted_velocity = np.asarray(
        rates["reversible_emitted_sign_spatial_velocity_m_s"], dtype=float
    )

    reverse_mobile = 0.0
    emitted_mobile = 0.0
    for system, sign in enumerate(state.c.emission_signs):
        q = 1 if sign > 0 else 0
        mobile = np.asarray(state.mobile_m2[system, q], dtype=float)
        emitted_mobile += float(np.sum(mobile))
        reverse_mobile += float(np.sum(mobile[reverse_mask[system]]))

    projected_tau = np.asarray(
        rates["reversible_forward_projected_tau_Pa"], dtype=float
    )
    record = {
        "record_type": record_type,
        "cycle_index": int(cycle),
        "phase": float(phase),
        "cumulative_cycles": float(cycle + phase),
        "time_s": float(state.time_s),
        "K_MPa_sqrt_m": float(K),
        "K_eff_MPa_sqrt_m": float(K_eff),
        "transport_K_signed_MPa_sqrt_m": float(
            rates["reversible_transport_K_signed_MPa_sqrt_m"]
        ),
        "transport_tau_min_Pa": float(np.min(transport_tau)),
        "transport_tau_max_Pa": float(np.max(transport_tau)),
        "forward_projected_transport_tau_min_Pa": float(np.min(projected_tau)),
        "forward_projected_transport_tau_max_Pa": float(np.max(projected_tau)),
        "emitted_sign_velocity_min_m_s": float(np.min(emitted_velocity)),
        "emitted_sign_velocity_max_m_s": float(np.max(emitted_velocity)),
        "negative_spatial_velocity_fraction": float(
            np.mean(emitted_velocity < 0.0)
        ),
        "true_reverse_drive_spatial_fraction": float(np.mean(reverse_mask)),
        "emitted_mobile_present_m2": emitted_mobile,
        "reverse_mobile_present_m2": reverse_mobile,
        "reverse_mobile_fraction_of_emitted_mobile": (
            reverse_mobile / max(emitted_mobile, 1.0e-300)
        ),
        "tip_stress_Pa": float(sigma),
        "effective_barrier_eV": barrier,
        "cleavage_rate_s": float(
            state.cleavage_rate_s(K, loading.temperature_K)
        ),
        "cumulative_hazard_action": float(action),
        "threshold_action": float(threshold),
        "crack_extension_m": float(extension),
        "mobile_total_m2": float(np.sum(state.mobile_m2)),
        "retained_total_m2": float(np.sum(state.retained_m2)),
        "accumulated_source_slip_total_m2": float(
            np.sum(state.accumulated_slip_m2)
        ),
        "returned_slip_total_m2": float(np.sum(state.returned_slip_m2)),
        "net_slip_total_m2": float(np.sum(state.net_slip_m2())),
        "shielding_MPa_sqrt_m": shield,
        "backstress_Pa": float(np.mean(backstress)),
        "tip_radius_m": radius,
        "front_width_m": float(state.source_geometry()["front_width_m"]),
        "emission_rate_peak_s": float(np.max(rates["emission_rate_s"])),
        "peierls_velocity_peak_m_s": float(
            np.max(np.abs(rates["peierls_velocity_m_s"]))
        ),
        "taylor_completion_peak_s": float(
            np.max(rates["taylor_completion_s"])
        ),
        "encounter_rate_peak_s": float(np.max(rates["encounter_s"])),
    }
    record.update(state.reversibility_diagnostics())
    return record


_v2._diagnostic = _diagnostic

MODE = _v2.MODE
SCHEMA = _v2.SCHEMA
run_minimal_reversible_explicit = _v2.run_minimal_reversible_explicit

__all__ = ["MODE", "SCHEMA", "run_minimal_reversible_explicit"]
