"""Detailed-balance correction for v10.4.1 bulk Peierls--Taylor slip.

The v10.4.0 constitutive prototype converted one-way Arrhenius event rates
straight into an Orowan strain rate. A one-way thermal event rate is finite at
zero resolved stress, but the forward and reverse events cancel there and must
not produce directed macroscopic strain. This overlay constructs the reverse
barrier symmetrically about the zero-stress barrier,

    G_reverse(tau) = 2 G(0) - G_forward(tau),

and uses the net rate Gamma_forward - Gamma_reverse. Consequently the net
rate is exactly zero at zero stress, is non-negative for non-negative equivalent
stress, and obeys local detailed balance without introducing a fitted constant.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from .emission_derived_plasticity import ExpFloorSurface

KB_EV_PER_K = 8.617333262145e-5
MODEL_ID = "v10.4.1_bulk_peierls_taylor_detailed_balance"
_ORIGINAL_RATE_METHOD: Callable | None = None


def detailed_balance_rate_s(
    surface: ExpFloorSurface,
    stress_Pa: np.ndarray,
    T_K: float,
) -> np.ndarray:
    stress = np.maximum(np.asarray(stress_Pa, dtype=float), 0.0)
    kT_eV = max(KB_EV_PER_K * float(T_K), 1.0e-30)

    forward_barrier = surface.barrier_eV(stress, T_K)
    zero_barrier = surface.barrier_eV(np.zeros_like(stress), T_K)
    reverse_barrier = np.maximum(2.0 * zero_barrier - forward_barrier, zero_barrier)

    forward = surface.attempt_frequency_s * np.exp(
        np.clip(-forward_barrier / kT_eV, -745.0, 0.0)
    )
    reverse = surface.attempt_frequency_s * np.exp(
        np.clip(-reverse_barrier / kT_eV, -745.0, 0.0)
    )
    net = np.maximum(forward - reverse, 0.0)

    # Enforce the exact thermodynamic fixed point rather than relying on
    # floating-point cancellation of equal exponentials.
    return np.where(stress > 0.0, net, 0.0)


def install_detailed_balance_net_slip() -> Callable:
    global _ORIGINAL_RATE_METHOD
    if _ORIGINAL_RATE_METHOD is None:
        _ORIGINAL_RATE_METHOD = ExpFloorSurface.rate_s
    ExpFloorSurface.rate_s = detailed_balance_rate_s
    return _ORIGINAL_RATE_METHOD


def restore_detailed_balance_net_slip(original: Callable | None = None) -> None:
    global _ORIGINAL_RATE_METHOD
    method = original if original is not None else _ORIGINAL_RATE_METHOD
    if method is not None:
        ExpFloorSurface.rate_s = method
    _ORIGINAL_RATE_METHOD = None


def audit_payload() -> dict:
    return {
        "schema": MODEL_ID,
        "one_way_arrhenius_rate_used_as_net_slip": False,
        "net_slip_rate": "Gamma_forward_minus_Gamma_reverse",
        "forward_reverse_barriers": "symmetric_about_zero_stress_barrier",
        "reverse_barrier": "2*G_zero-G_forward",
        "zero_stress_net_plastic_rate_exactly_zero": True,
        "new_fitted_parameters": 0,
        "v10_4_0_outputs_physics_compatible": False,
    }


def write_detailed_balance_audit(output_root: str | Path) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "v10_4_1_bulk_detailed_balance_audit.json"
    path.write_text(json.dumps(audit_payload(), indent=2, sort_keys=True) + "\n")
    return path


__all__ = [
    "MODEL_ID",
    "audit_payload",
    "detailed_balance_rate_s",
    "install_detailed_balance_net_slip",
    "restore_detailed_balance_net_slip",
    "write_detailed_balance_audit",
]
