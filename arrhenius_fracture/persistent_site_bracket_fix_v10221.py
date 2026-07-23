"""Numerically robust complementarity solver for v10.2.21 emission.

The persistent-site source has a hard mechanical blocking condition:

    sigma_eff = max(sigma_drive - k_back*sqrt(rho), 0).

At sufficiently high aggregate Arrhenius hazard, the backward-Euler rate equation
need not have an interior root before the blocking density is reached. That is
not a failure: the admissible increment is the mechanically blocking increment.
This module installs that complementarity interpretation without changing any
constitutive parameter or adding a source inventory/cap.
"""
from __future__ import annotations

import math
from typing import Callable

from . import persistent_site_source_v10221 as _source


def solve_backstress_limited_activations(
    *,
    multiplicity: float,
    dt_s: float,
    drive_stress_Pa: float,
    rho_initial_m2: float,
    rho_increment_per_activation_m2: float,
    backstress_prefactor_Pa_sqrt_m2: float,
    rate_function: Callable[[float], float],
    tolerance: float = 1.0e-10,
    max_iterations: int = 96,
) -> float:
    """Solve the implicit emission update with a mechanical blocking constraint.

    The admissible increment satisfies ``0 <= dN <= dN_block``. An interior
    backward-Euler root is used when it exists. If the aggregate hazard remains
    larger than the admissible increment all the way to the blocking state, the
    complementarity solution is exactly ``dN_block``.
    """
    M = max(float(multiplicity), 0.0)
    dt = max(float(dt_s), 0.0)
    drive = max(float(drive_stress_Pa), 0.0)
    rho0 = max(float(rho_initial_m2), 0.0)
    rho_per = max(float(rho_increment_per_activation_m2), 0.0)
    kback = max(float(backstress_prefactor_Pa_sqrt_m2), 0.0)
    if M <= 0.0 or dt <= 0.0 or drive <= 0.0:
        return 0.0
    if rho_per <= 0.0 or kback <= 0.0:
        raise RuntimeError(
            "persistent-site emission requires positive backstress coupling"
        )

    sigma0 = drive - kback * math.sqrt(rho0)
    if sigma0 <= 0.0:
        return 0.0
    rate0 = max(float(rate_function(sigma0)), 0.0)
    if not math.isfinite(rate0) or rate0 <= 0.0:
        return 0.0

    rho_block = (drive / kback) ** 2
    upper = max((rho_block - rho0) / rho_per, 0.0)
    if upper <= 0.0:
        return 0.0

    def residual(value: float) -> float:
        rho = rho0 + rho_per * max(value, 0.0)
        sigma_eff = drive - kback * math.sqrt(max(rho, 0.0))
        if sigma_eff <= 0.0:
            rate = 0.0
        else:
            rate = max(float(rate_function(sigma_eff)), 0.0)
            if not math.isfinite(rate):
                rate = 0.0
        return value - M * rate * dt

    lo = 0.0
    # Evaluate immediately inside the admissible interval. At the exact upper
    # endpoint the hard mechanical gate is active by definition.
    hi_inside = math.nextafter(upper, 0.0)
    if hi_inside <= 0.0:
        return upper
    r_hi_inside = residual(hi_inside)

    # No interior root exists: the aggregate hazard would overrun the
    # mechanically admissible state, so the complementarity solution is the
    # blocking increment. This is backstress saturation, not a source cap.
    if r_hi_inside <= 0.0:
        return upper

    hi = hi_inside
    scale = max(upper, 1.0)
    for _ in range(int(max_iterations)):
        mid = 0.5 * (lo + hi)
        value = residual(mid)
        if (
            abs(value) <= float(tolerance) * scale
            or (hi - lo) <= float(tolerance) * scale
        ):
            return min(max(mid, 0.0), upper)
        if value > 0.0:
            hi = mid
        else:
            lo = mid
    return min(max(0.5 * (lo + hi), 0.0), upper)


def install_backstress_complementarity_fix() -> None:
    """Install the corrected solver in the persistent-source module."""
    _source.solve_backstress_limited_activations = (
        solve_backstress_limited_activations
    )


__all__ = [
    "solve_backstress_limited_activations",
    "install_backstress_complementarity_fix",
]
