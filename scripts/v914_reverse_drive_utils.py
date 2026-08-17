"""Reference-direction diagnostics for reversible mobile transport.

The signed spatial velocity of a Burgers population is not, by itself, a
reversal diagnostic: opposite Burgers-sign populations move in opposite spatial
directions under the same forward resolved stress.  A true cyclic reversal is
therefore defined relative to the resolved-stress direction produced by a
positive tensile loading on each slip system.
"""
from __future__ import annotations

import numpy as np


def tensile_reference_signs(drive_factors: np.ndarray) -> np.ndarray:
    """Return the resolved-stress sign associated with positive tensile K."""
    drive = np.asarray(drive_factors, dtype=float)
    if drive.ndim != 1:
        raise ValueError("drive_factors must be one-dimensional")
    signs = np.sign(drive)
    # A truly zero projection has no preferred tensile sense.  Treat it as +1
    # for diagnostics only; such a system has zero external forward drive.
    signs[signs == 0.0] = 1.0
    return signs


def forward_projected_transport_stress(
    tau_effective_Pa: np.ndarray,
    drive_factors: np.ndarray,
) -> np.ndarray:
    """Project signed transport stress onto each system's tensile reference."""
    tau = np.asarray(tau_effective_Pa, dtype=float)
    signs = tensile_reference_signs(drive_factors)
    if tau.ndim != 2 or tau.shape[0] != signs.size:
        raise ValueError("tau_effective_Pa must have shape (n_systems, n_bins)")
    return tau * signs[:, None]


def reverse_drive_mask(
    tau_effective_Pa: np.ndarray,
    drive_factors: np.ndarray,
) -> np.ndarray:
    """True where the effective transport stress opposes tensile loading."""
    return forward_projected_transport_stress(
        tau_effective_Pa, drive_factors
    ) < 0.0


__all__ = [
    "tensile_reference_signs",
    "forward_projected_transport_stress",
    "reverse_drive_mask",
]
