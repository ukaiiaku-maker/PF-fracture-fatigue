"""Signed mobile transport primitives for the v10.2.30 persistent-site state.

Cleavage and emission remain opening-only. These functions act only on existing
positive/negative Burgers mobile populations and deliberately do not select an
emission/source-drive projection.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

MODEL_ID = "v10.2.30_intrinsic_signed_mobile_transport_v1"


def signed_finite_tip_stress_Pa(K_signed_Pa_sqrt_m: float, radius_m: float) -> float:
    """Map signed applied K to finite-tip stress without an opening-only clip."""
    radius = max(float(radius_m), 1.0e-30)
    return float(K_signed_Pa_sqrt_m) / math.sqrt(2.0 * math.pi * radius)


def signed_arrhenius_rate(surface, stress_Pa, temperature_K: float) -> np.ndarray:
    """Forward-minus-reverse rate using one unchanged barrier surface."""
    stress = np.asarray(stress_Pa, dtype=float)
    magnitude = np.abs(stress)
    driven = np.asarray(surface.rate(magnitude, float(temperature_K)), dtype=float)
    zero = float(np.asarray(surface.rate(0.0, float(temperature_K))))
    return np.sign(stress) * np.maximum(driven - zero, 0.0)


def advect_signed_mobile_populations(
    mpz: Any,
    velocity_by_system_m_s,
    dt_s: float,
    emitted_sign_by_system,
    true_reverse_drive_by_system,
) -> dict[str, np.ndarray]:
    """Advect actual signed-Burgers fields and classify both boundary fates.

    Positive and negative Burgers populations move in opposite spatial
    directions under the same signed glide rate. Only emitted-sign left-boundary
    outflow under true reverse drive is a physical surface return.
    """
    velocity = np.asarray(velocity_by_system_m_s, dtype=float).reshape(-1)
    emitted_sign = np.sign(np.asarray(emitted_sign_by_system, dtype=float).reshape(-1))
    reverse = np.asarray(true_reverse_drive_by_system, dtype=bool).reshape(-1)
    if velocity.shape != (mpz.n_systems,):
        raise ValueError("one signed velocity is required per slip system")
    if emitted_sign.shape != velocity.shape or reverse.shape != velocity.shape:
        raise ValueError("emitted-sign and reverse-drive arrays must match systems")
    duration = max(float(dt_s), 0.0)
    returned = np.zeros((mpz.n_systems, 2), dtype=float)
    escaped = np.zeros((mpz.n_systems, 2), dtype=float)
    physical_return = np.zeros((mpz.n_systems, 2), dtype=float)
    for system in range(mpz.n_systems):
        emitted_q = 1 if emitted_sign[system] > 0.0 else 0
        for q, suffix in enumerate(("negative", "positive")):
            field = np.asarray(getattr(mpz, f"mobile_{suffix}"), dtype=float)
            spatial_velocity = (-1.0 if q == 0 else 1.0) * velocity[system]
            distance = abs(spatial_velocity) * duration
            row = field[system : system + 1]
            if spatial_velocity >= 0.0:
                moved, loss = mpz._advect_forward(row, distance, mpz.dx)
                escaped[system, q] = loss
            else:
                flipped, loss = mpz._advect_forward(row[:, ::-1], distance, mpz.dx)
                moved = flipped[:, ::-1]
                returned[system, q] = loss
                if q == emitted_q and reverse[system]:
                    physical_return[system, q] = loss
            field[system : system + 1] = moved
            setattr(mpz, f"mobile_{suffix}", field)
    mpz.mobile = mpz.mobile_positive + mpz.mobile_negative
    return {
        "returned_mobile": returned,
        "escaped_mobile": escaped,
        "physical_returned_mobile": physical_return,
    }


__all__ = [
    "MODEL_ID",
    "advect_signed_mobile_populations",
    "signed_arrhenius_rate",
    "signed_finite_tip_stress_Pa",
]
