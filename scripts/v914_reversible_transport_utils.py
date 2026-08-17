"""Pure bookkeeping helpers for the v9.14 minimal reversible-fatigue model.

These functions contain no constitutive physics.  They convert the conservative
upwind boundary fluxes already present in the v9.13/v9.14 mobile-dislocation
operator into explicit surface-return and far-field-escape ledgers, and provide
an exact bounded cancellation of the source-slip ledger when a mobile
population returns to the crack/free surface.
"""
from __future__ import annotations

import numpy as np


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


__all__ = ["boundary_outflow_per_m", "proportional_cancellation_density"]
