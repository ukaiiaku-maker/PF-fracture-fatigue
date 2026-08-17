"""Helpers for the v9.14 minimal reversible-fatigue model.

The boundary/cancellation functions are bookkeeping utilities.  The signed
transport helper implements the one deliberate constitutive extension in the
minimal reversible model: already-mobile dislocations see the signed cyclic
stress intensity, while cleavage and new emission remain on the unchanged
opening-only v9.13/v9.14 laws.
"""
from __future__ import annotations

import math

import numpy as np


def signed_transport_stress_fields(
    K_applied_MPa_sqrt_m: float,
    K_shield_MPa_sqrt_m: float,
    tip_radius_m: float,
    drive_factors: np.ndarray,
    tau_gnd_Pa: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Build signed applied/effective stress fields for mobile transport.

    Unlike the opening/cleavage channel, ``K_applied`` is not clipped at zero.
    The effective transport intensity is ``K_applied - K_shield`` and can
    therefore become negative either under an externally compressive cycle or
    when shielding exceeds the instantaneous positive applied K during
    unloading.  The existing signed GND stress is then added exactly as in the
    parent transport law.
    """
    drive = np.asarray(drive_factors, dtype=float)
    tau_gnd = np.asarray(tau_gnd_Pa, dtype=float)
    if drive.ndim != 1:
        raise ValueError("drive_factors must be one-dimensional")
    if tau_gnd.ndim != 2 or tau_gnd.shape[0] != drive.size:
        raise ValueError("tau_gnd_Pa must have shape (n_systems, n_bins)")

    radius = max(float(tip_radius_m), 1.0e-30)
    scale = 1.0e6 / math.sqrt(2.0 * math.pi * radius)
    K_applied = float(K_applied_MPa_sqrt_m)
    K_transport = K_applied - float(K_shield_MPa_sqrt_m)
    sigma_applied = K_applied * scale
    sigma_transport = K_transport * scale

    tau_applied_column = drive[:, None] * sigma_applied
    tau_transport_column = drive[:, None] * sigma_transport
    tau_applied = np.broadcast_to(tau_applied_column, tau_gnd.shape).copy()
    tau_transport = np.broadcast_to(tau_transport_column, tau_gnd.shape).copy()
    tau_effective = tau_transport + tau_gnd
    return {
        "K_transport_MPa_sqrt_m": K_transport,
        "sigma_applied_Pa": sigma_applied,
        "sigma_transport_Pa": sigma_transport,
        "tau_applied_Pa": tau_applied,
        "tau_transport_Pa": tau_transport,
        "tau_effective_Pa": tau_effective,
    }


def boundary_outflow_per_m(
    mobile_m2: np.ndarray,
    velocity_m_s: np.ndarray,
    dt_s: float,
) -> tuple[float, float]:
    """Return left(surface) and right(far-field) mobile outflow per unit front.

    The mobile density has units m^-2 and velocity has units m/s.  Integrating
    the boundary flux over ``dt_s`` therefore gives line content per unit crack
    front, m^-1.  The values are consistent with a backward-Euler transport
    update when evaluated from the post-solve mobile state.
    """
    mobile = np.maximum(np.asarray(mobile_m2, dtype=float), 0.0)
    velocity = np.asarray(velocity_m_s, dtype=float)
    if mobile.ndim != 1 or velocity.shape != mobile.shape:
        raise ValueError("mobile_m2 and velocity_m_s must be matching 1-D arrays")
    dt = max(float(dt_s), 0.0)
    if mobile.size == 0 or dt <= 0.0:
        return 0.0, 0.0
    returned = max(-float(velocity[0]), 0.0) * float(mobile[0]) * dt
    escaped = max(float(velocity[-1]), 0.0) * float(mobile[-1]) * dt
    return max(returned, 0.0), max(escaped, 0.0)


def proportional_cancellation_density(
    net_source_density_m2: np.ndarray,
    returned_line_content: float,
    cell_area_m2: float,
) -> tuple[np.ndarray, float]:
    """Cancel returned line content from a nonnegative source-slip density.

    Cancellation is distributed in proportion to the currently uncancelled
    source-slip density.  It is bounded by the available net source slip, so
    numerical roundoff or return of mobile content released from a retained
    population can never drive the net-slip field negative.

    Returns ``(density_increment_m2, cancelled_line_content)``.
    """
    net = np.maximum(np.asarray(net_source_density_m2, dtype=float), 0.0)
    area = max(float(cell_area_m2), 0.0)
    requested = max(float(returned_line_content), 0.0)
    if net.ndim != 1:
        raise ValueError("net_source_density_m2 must be one-dimensional")
    if net.size == 0 or area <= 0.0 or requested <= 0.0:
        return np.zeros_like(net), 0.0
    available = float(np.sum(net) * area)
    if available <= 0.0:
        return np.zeros_like(net), 0.0
    cancelled = min(requested, available)
    fraction = cancelled / available
    return net * fraction, cancelled


__all__ = [
    "signed_transport_stress_fields",
    "boundary_outflow_per_m",
    "proportional_cancellation_density",
]
