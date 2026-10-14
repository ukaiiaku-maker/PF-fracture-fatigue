"""Certified opening-clock accessibility descriptors for the R10 search.

The bound in this module is deliberately channel local.  It applies when an
``ExpFloorBarrier`` is the complete barrier consumed by the cooperative
opening clock.  It says nothing about a separate failure mechanism and is not
an upper bound on an arbitrary coupled model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from scipy.optimize import brentq
from scipy.special import gammainc

from .material_manifest import ExpFloorBarrier, KB_EV_PER_K


@dataclass(frozen=True)
class OpeningAccessibilityBound:
    temperature_K: float
    evaluated_floor_eV: float
    elementary_rate_ceiling_s: float
    cooperative_rate_ceiling_s: float
    exposure_s: float
    action_ceiling: float
    pending_mark: float
    multiplicity: float
    applicable: bool
    necessarily_inaccessible: bool


def raw_coordinates(surface: ExpFloorBarrier, temperature_K: float) -> tuple[float, float]:
    """Return the unclipped zero-stress energy and characteristic stress."""
    dT = float(temperature_K) - float(surface.Tref_K)
    return (
        float(surface.G00_eV + surface.gT_eV_per_K * dT),
        float(surface.sigc0_Pa + surface.sT_Pa_per_K * dT),
    )


def evaluated_floor_eV(surface: ExpFloorBarrier, temperature_K: float) -> float:
    """Reproduce the exact floor used by :meth:`ExpFloorBarrier.values_eV`."""
    raw_G0, _ = raw_coordinates(surface, temperature_K)
    G0 = max(raw_G0, 1.0e-12)
    raw_floor = max(float(surface.floor_min_eV), float(surface.floor_fraction) * G0)
    return float(min(float(surface.floor_max_fraction) * G0, raw_floor))


def cooperative_rate_from_barrier(
    barrier_eV: float, *, temperature_K: float, attempt_frequency_s: float,
    hits: float, tau_s: float,
) -> tuple[float, float]:
    elementary = float(attempt_frequency_s) * math.exp(
        max(-700.0, min(0.0, -float(barrier_eV) / (KB_EV_PER_K * float(temperature_K))))
    )
    argument = min(elementary * float(tau_s), 1.0e12)
    cooperative = float(gammainc(float(hits), argument) / float(tau_s))
    return elementary, cooperative


def opening_floor_bound(
    surface: ExpFloorBarrier, *, temperature_K: float, hits: float, tau_s: float,
    exposure_s: float, pending_mark: float, multiplicity: float = 1.0,
    applicable: bool = True,
) -> OpeningAccessibilityBound:
    """Bound cooperative opening action using the evaluated barrier floor.

    ``pending_mark`` is the action remaining in the actual final consumer;
    callers must also pass its actual multiplicity.  ``applicable=False`` is
    an explicit record that another unbounded barrier contribution prevents
    use of this simple proof.
    """
    if hits <= 0.0 or tau_s <= 0.0 or exposure_s < 0.0 or pending_mark < 0.0:
        raise ValueError("invalid cooperative-clock contract")
    if multiplicity < 0.0:
        raise ValueError("multiplicity must be nonnegative")
    floor = evaluated_floor_eV(surface, temperature_K)
    elementary, cooperative = cooperative_rate_from_barrier(
        floor, temperature_K=temperature_K,
        attempt_frequency_s=surface.attempt_frequency_s, hits=hits, tau_s=tau_s,
    )
    action = float(multiplicity) * float(exposure_s) * cooperative
    return OpeningAccessibilityBound(
        temperature_K=float(temperature_K), evaluated_floor_eV=floor,
        elementary_rate_ceiling_s=elementary,
        cooperative_rate_ceiling_s=cooperative, exposure_s=float(exposure_s),
        action_ceiling=action, pending_mark=float(pending_mark),
        multiplicity=float(multiplicity), applicable=bool(applicable),
        necessarily_inaccessible=bool(applicable and action < pending_mark),
    )


def maximum_floor_for_mark(
    *, temperature_K: float, attempt_frequency_s: float, hits: float, tau_s: float,
    exposure_s: float, pending_mark: float, multiplicity: float = 1.0,
) -> float:
    """Largest floor not excluded by the certified action bound."""
    if pending_mark <= 0.0:
        return math.inf

    def residual(barrier: float) -> float:
        _, rate = cooperative_rate_from_barrier(
            barrier, temperature_K=temperature_K,
            attempt_frequency_s=attempt_frequency_s, hits=hits, tau_s=tau_s,
        )
        return multiplicity * exposure_s * rate - pending_mark

    if residual(0.0) < 0.0:
        raise ValueError("mark is unreachable even at a zero barrier")
    upper = 1.0
    while residual(upper) > 0.0 and upper < 100.0:
        upper *= 2.0
    if upper >= 100.0 and residual(upper) > 0.0:
        raise ValueError("failed to bracket floor threshold")
    return float(brentq(residual, 0.0, upper, xtol=1.0e-14, rtol=1.0e-14))


__all__ = [
    "OpeningAccessibilityBound", "cooperative_rate_from_barrier",
    "evaluated_floor_eV", "maximum_floor_for_mark", "opening_floor_bound",
    "raw_coordinates",
]
