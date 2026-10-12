"""Potential-based cohesive landscapes for constitutive analysis.

These functions evaluate a reversible one-coordinate cohesive potential.  They
do not advance a crack, insert a cohesive element, or replace the native PF or
FEM/CZM specimen solvers.  A specimen calculation must separately define the
activation patch, loading ensemble, kinetic prefactor, and topology update.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import e
from typing import Protocol

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


class CohesiveLandscape(Protocol):
    peak_traction_Pa: float
    separation_work_J_m2: float

    def traction_Pa(self, opening_m: float | np.ndarray) -> float | np.ndarray: ...
    def potential_J_m2(self, opening_m: float) -> float: ...
    def support_end_m(self) -> float: ...


@dataclass(frozen=True)
class BarrierEvaluation:
    applied_traction_Pa: float
    minimum_opening_m: float
    saddle_opening_m: float
    barrier_J: float
    activation_volume_m3: float
    activation_area_m2: float
    ensemble: str = "prescribed_local_traction"


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass(frozen=True)
class TriangularBilinearLandscape:
    """Triangular bilinear envelope with matched peak and total work."""

    peak_traction_Pa: float
    separation_work_J_m2: float
    peak_fraction: float = 0.25

    def __post_init__(self) -> None:
        _positive(self.peak_traction_Pa, "peak_traction_Pa")
        _positive(self.separation_work_J_m2, "separation_work_J_m2")
        if not 0.0 < float(self.peak_fraction) < 1.0:
            raise ValueError("peak_fraction must lie strictly between zero and one")

    def support_end_m(self) -> float:
        return 2.0 * self.separation_work_J_m2 / self.peak_traction_Pa

    @property
    def peak_opening_m(self) -> float:
        return self.peak_fraction * self.support_end_m()

    def traction_Pa(self, opening_m):
        x = np.asarray(opening_m, dtype=float)
        d0, df, tc = self.peak_opening_m, self.support_end_m(), self.peak_traction_Pa
        out = np.where(
            (x >= 0.0) & (x <= d0), tc * x / d0,
            np.where((x > d0) & (x <= df), tc * (df - x) / (df - d0), 0.0),
        )
        return float(out) if out.ndim == 0 else out

    def potential_J_m2(self, opening_m: float) -> float:
        d = float(np.clip(opening_m, 0.0, self.support_end_m()))
        d0, df, tc = self.peak_opening_m, self.support_end_m(), self.peak_traction_Pa
        if d <= d0:
            return 0.5 * tc * d * d / d0
        return 0.5 * tc * d0 + tc * (
            df * (d - d0) - 0.5 * (d * d - d0 * d0)
        ) / (df - d0)

    def exact_barrier(self, applied_traction_Pa: float, activation_area_m2: float) -> BarrierEvaluation:
        t = float(applied_traction_Pa)
        A = _positive(activation_area_m2, "activation_area_m2")
        tc = self.peak_traction_Pa
        if not 0.0 < t < tc:
            raise ValueError("applied traction must lie strictly between zero and peak")
        d0, df = self.peak_opening_m, self.support_end_m()
        dm = t * d0 / tc
        ds = df - t * (df - d0) / tc
        barrier = A * self.separation_work_J_m2 * (1.0 - t / tc) ** 2
        return BarrierEvaluation(t, dm, ds, barrier, A * (ds - dm), A)


@dataclass(frozen=True)
class ExponentialLandscape:
    """Smooth infinite-tail envelope ``t=tc*x*exp(1-x)``."""

    peak_traction_Pa: float
    separation_work_J_m2: float

    def __post_init__(self) -> None:
        _positive(self.peak_traction_Pa, "peak_traction_Pa")
        _positive(self.separation_work_J_m2, "separation_work_J_m2")

    @property
    def scale_m(self) -> float:
        return self.separation_work_J_m2 / (e * self.peak_traction_Pa)

    def support_end_m(self) -> float:
        # Numerical root bracketing only; the constitutive support is infinite.
        return 64.0 * self.scale_m

    def traction_Pa(self, opening_m):
        x = np.maximum(np.asarray(opening_m, dtype=float) / self.scale_m, 0.0)
        out = self.peak_traction_Pa * x * np.exp(1.0 - x)
        return float(out) if out.ndim == 0 else out

    def potential_J_m2(self, opening_m: float) -> float:
        x = max(float(opening_m) / self.scale_m, 0.0)
        return self.separation_work_J_m2 * (1.0 - (1.0 + x) * np.exp(-x))


@dataclass(frozen=True)
class PolynomialLandscape:
    """Finite-support smooth envelope ``t/tc=(27/4)x(1-x)^2``.

    Its maximum is at ``x=1/3`` and its area is matched exactly to the
    requested separation work.  This is a specified polynomial benchmark, not
    a claim that every Tvergaard--Hutchinson implementation uses this shape.
    """

    peak_traction_Pa: float
    separation_work_J_m2: float

    def __post_init__(self) -> None:
        _positive(self.peak_traction_Pa, "peak_traction_Pa")
        _positive(self.separation_work_J_m2, "separation_work_J_m2")

    def support_end_m(self) -> float:
        return 16.0 * self.separation_work_J_m2 / (9.0 * self.peak_traction_Pa)

    def traction_Pa(self, opening_m):
        x = np.asarray(opening_m, dtype=float) / self.support_end_m()
        out = np.where((x >= 0.0) & (x <= 1.0), self.peak_traction_Pa * 27.0 / 4.0 * x * (1.0 - x) ** 2, 0.0)
        return float(out) if out.ndim == 0 else out

    def potential_J_m2(self, opening_m: float) -> float:
        x = float(np.clip(opening_m / self.support_end_m(), 0.0, 1.0))
        return self.peak_traction_Pa * self.support_end_m() * 27.0 / 4.0 * (
            0.5 * x * x - 2.0 * x ** 3 / 3.0 + 0.25 * x ** 4
        )


def evaluate_traction_controlled_barrier(
    landscape: CohesiveLandscape,
    applied_traction_Pa: float,
    activation_area_m2: float,
) -> BarrierEvaluation:
    """Return the metastable-to-saddle barrier of the tilted potential.

    The calculation is valid only for prescribed local traction and a fixed
    reversible landscape.  Finite-body displacement control additionally
    requires the surrounding elastic energy/compliance.
    """

    t = float(applied_traction_Pa)
    A = _positive(activation_area_m2, "activation_area_m2")
    tc = _positive(landscape.peak_traction_Pa, "peak_traction_Pa")
    if not 0.0 < t < tc:
        raise ValueError("applied traction must lie strictly between zero and peak")
    if isinstance(landscape, TriangularBilinearLandscape):
        return landscape.exact_barrier(t, A)
    peak = landscape.scale_m if isinstance(landscape, ExponentialLandscape) else landscape.support_end_m() / 3.0
    end = landscape.support_end_m()
    f = lambda d: float(landscape.traction_Pa(d)) - t
    dm = brentq(f, 0.0, peak)
    ds = brentq(f, peak, end)
    biased = lambda d: landscape.potential_J_m2(d) - t * d
    barrier = A * (biased(ds) - biased(dm))
    return BarrierEvaluation(t, dm, ds, barrier, A * (ds - dm), A)


def numerical_work_J_m2(landscape: CohesiveLandscape) -> float:
    """Independent quadrature check of the envelope's separation work."""

    end = landscape.support_end_m()
    value, _ = quad(lambda d: float(landscape.traction_Pa(d)), 0.0, end, epsabs=1e-12)
    if isinstance(landscape, ExponentialLandscape):
        # Correct the intentionally finite quadrature bracket analytically.
        x = end / landscape.scale_m
        value += landscape.separation_work_J_m2 * (1.0 + x) * np.exp(-x)
    return float(value)


__all__ = [
    "BarrierEvaluation", "CohesiveLandscape", "ExponentialLandscape",
    "PolynomialLandscape", "TriangularBilinearLandscape",
    "evaluate_traction_controlled_barrier", "numerical_work_J_m2",
]
