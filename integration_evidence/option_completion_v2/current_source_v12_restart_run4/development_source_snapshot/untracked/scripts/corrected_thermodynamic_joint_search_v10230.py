"""Serialization support for the preserved corrected thermodynamic surfaces.

The broader donor analysis module also contains search/reporting helpers.  The
shared runtime needs only the exact, lossless surface reconstruction used by
the canonical V2 registry.
"""
from __future__ import annotations

from typing import Mapping

from scripts.thermodynamic_joint_barrier_v10230 import Basis, ThermodynamicSurface


def deserialize_surface(data: Mapping) -> ThermodynamicSurface:
    payload = dict(data)
    payload["bases"] = tuple(Basis(**item) for item in payload["bases"])
    for key in (
        "reference_amplitudes_eV",
        "entropy_amplitudes_kB",
        "heat_capacity_amplitudes_kB",
    ):
        payload[key] = tuple(payload[key])
    return ThermodynamicSurface(**payload)


__all__ = ["deserialize_surface"]
