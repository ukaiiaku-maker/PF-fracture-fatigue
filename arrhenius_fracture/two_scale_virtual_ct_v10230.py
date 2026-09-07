"""Shape-preserving local-rate surfaces and two-scale virtual C(T) integration.

The local constitutive closure is the measured developed rate itself.  No
Paris-law coefficient is introduced: interpolation is PCHIP in ``ln K`` and
``ln(da/dN)`` independently at each measured R ratio.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import PchipInterpolator

from .virtual_ct_v10230 import (
    CompactTensionGeometry,
    ct_geometry_factor,
    ct_load_from_k_pa_sqrt_m,
)


class LocalSurfaceOutOfDomain(ValueError):
    """Raised before a local response surface is evaluated outside its data."""


@dataclass(frozen=True)
class SurfaceEvaluation:
    rate_m_per_cycle: np.ndarray
    local_slope: np.ndarray
    R_interpolated: bool


class LogPchipRateSurface:
    """Positive, monotone, shape-preserving ``g(Kmax, R)`` surface."""

    def __init__(self, rows: Iterable[dict], *, option: str, version: str):
        self.option = str(option)
        self.version = str(version)
        grouped: dict[float, list[tuple[float, float]]] = {}
        for row in rows:
            grouped.setdefault(float(row["R"]), []).append(
                (float(row["Kmax_MPa_sqrt_m"]), float(row["developed_da_dN"]))
            )
        if len(grouped) < 1:
            raise ValueError("rate surface requires source rows")
        self.R_values = np.asarray(sorted(grouped), dtype=float)
        self._nodes: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        self._curves: dict[float, PchipInterpolator] = {}
        for R, pairs in grouped.items():
            pairs = sorted(pairs)
            K = np.asarray([x[0] for x in pairs], dtype=float)
            g = np.asarray([x[1] for x in pairs], dtype=float)
            if len(K) < 3 or np.any(np.diff(K) <= 0) or np.any(g <= 0) or np.any(np.diff(g) < 0):
                raise ValueError(f"invalid monotone positive rate nodes at R={R:g}")
            self._nodes[R] = (K, g)
            self._curves[R] = PchipInterpolator(np.log(K), np.log(g), extrapolate=False)
        lows = {float(x[0].min()) for x in self._nodes.values()}
        highs = {float(x[0].max()) for x in self._nodes.values()}
        if len(lows) != 1 or len(highs) != 1:
            raise ValueError("all R curves must share a common K domain")
        self.K_min = lows.pop()
        self.K_max = highs.pop()
        self.R_min = float(self.R_values.min())
        self.R_max = float(self.R_values.max())

    def nodes(self) -> list[dict]:
        return [
            {"option": self.option, "version": self.version, "R": R,
             "Kmax_MPa_sqrt_m": float(K), "developed_da_dN": float(g)}
            for R in self.R_values for K, g in zip(*self._nodes[float(R)])
        ]

    def _bracket_R(self, R: float) -> tuple[float, float, float]:
        R = float(R)
        if R < self.R_min - 1e-14 or R > self.R_max + 1e-14:
            raise LocalSurfaceOutOfDomain(f"LOCAL_SURFACE_OUT_OF_DOMAIN: R={R:g}")
        exact = self.R_values[np.isclose(self.R_values, R, rtol=0, atol=1e-14)]
        if len(exact):
            value = float(exact[0])
            return value, value, 0.0
        upper_index = int(np.searchsorted(self.R_values, R))
        lo, hi = float(self.R_values[upper_index - 1]), float(self.R_values[upper_index])
        return lo, hi, (R - lo) / (hi - lo)

    def evaluate(self, Kmax_MPa_sqrt_m, R: float) -> SurfaceEvaluation:
        K = np.asarray(Kmax_MPa_sqrt_m, dtype=float)
        if np.any(~np.isfinite(K)) or np.any(K < self.K_min - 1e-12) or np.any(K > self.K_max + 1e-12):
            raise LocalSurfaceOutOfDomain(
                f"LOCAL_SURFACE_OUT_OF_DOMAIN: K in [{np.nanmin(K):g},{np.nanmax(K):g}], "
                f"validated [{self.K_min:g},{self.K_max:g}]"
            )
        u = np.log(K)
        lo, hi, weight = self._bracket_R(R)
        vlo = self._curves[lo](u)
        slo = self._curves[lo].derivative()(u)
        if lo == hi:
            v, slope = vlo, slo
        else:
            vhi = self._curves[hi](u)
            shi = self._curves[hi].derivative()(u)
            v = (1.0 - weight) * vlo + weight * vhi
            slope = (1.0 - weight) * slo + weight * shi
        return SurfaceEvaluation(np.exp(v), np.asarray(slope), lo != hi)

    def prospective_anchor_prediction(self, Kmax_MPa_sqrt_m: float, R: float, *, maximum_extension_K: float = 24.3) -> dict:
        """Freeze a prospective endpoint check without expanding v0's domain.

        Interior anchors use the admissible surface.  The explicitly requested
        24.3 endpoint check uses the terminal PCHIP polynomial continuation and
        is labelled as a prospective extension; it cannot be used by virtual
        integration until a physical 24.3 result is incorporated into v1.
        """
        K = float(Kmax_MPa_sqrt_m)
        lo, hi, weight = self._bracket_R(R)
        prospective_extension = K > self.K_max
        if prospective_extension and K > maximum_extension_K + 1e-12:
            raise LocalSurfaceOutOfDomain("prospective anchor exceeds authorized endpoint")
        if not prospective_extension:
            ev = self.evaluate(K, R)
            rate, slope = float(ev.rate_m_per_cycle), float(ev.local_slope)
        else:
            u = math.log(K)
            curves = []
            for q in (lo, hi):
                curve = PchipInterpolator(
                    np.log(self._nodes[q][0]), np.log(self._nodes[q][1]), extrapolate=True
                )
                curves.append((float(curve(u)), float(curve.derivative()(u))))
            if lo == hi:
                v, slope = curves[0]
            else:
                v = (1-weight)*curves[0][0] + weight*curves[1][0]
                slope = (1-weight)*curves[0][1] + weight*curves[1][1]
            rate = math.exp(v)
        return {"R": float(R), "Kmax_MPa_sqrt_m": K, "predicted_da_dN": rate,
                "predicted_local_slope": slope, "prospective_endpoint_extension": prospective_extension,
                "admitted_for_virtual_integration": not prospective_extension}


def fixed_load_kmax_MPa(x, *, initial_kmax_MPa_sqrt_m: float = 12.0, x0: float = 0.45):
    x = np.asarray(x, dtype=float)
    return initial_kmax_MPa_sqrt_m * np.vectorize(ct_geometry_factor)(x) / ct_geometry_factor(x0)


def shedding_kmax_MPa(x, *, K0: float = 24.0, K1: float = 12.0, x0: float = 0.45, x1: float = 0.65):
    x = np.asarray(x, dtype=float)
    return K0 * np.exp(math.log(K1 / K0) * (x - x0) / (x1 - x0))


def protocol_kmax(protocol: str, x):
    if protocol == "FIXED_LOAD":
        return fixed_load_kmax_MPa(x)
    if protocol == "LOAD_SHEDDING":
        return shedding_kmax_MPa(x)
    if protocol == "CONSTANT_KMAX":
        return np.full_like(np.asarray(x, dtype=float), 18.0)
    raise ValueError(f"unknown protocol {protocol}")


def integrate_virtual_ct(
    surface: LogPchipRateSurface, *, R: float, protocol: str, geometry: CompactTensionGeometry,
    x0: float = 0.45, x1: float = 0.65, output_points: int = 401,
    relative_tolerance: float = 1e-9,
) -> tuple[list[dict], dict]:
    x = np.linspace(x0, x1, int(output_points))
    K = np.asarray(protocol_kmax(protocol, x), dtype=float)
    # Domain validation occurs before any integration output is admitted.
    ev = surface.evaluate(K, R)

    def dNdx(value: float) -> float:
        kval = float(protocol_kmax(protocol, np.asarray([value]))[0])
        rate = float(surface.evaluate(kval, R).rate_m_per_cycle)
        return geometry.width_m / rate

    total, error = quad(dNdx, x0, x1, epsrel=relative_tolerance, epsabs=1e-8, limit=300)
    N = np.zeros_like(x)
    for i in range(1, len(x)):
        N[i] = N[i-1] + quad(dNdx, float(x[i-1]), float(x[i]), epsrel=relative_tolerance, epsabs=1e-8, limit=80)[0]
    dense_x = np.linspace(x0, x1, max(20001, 4*len(x)+1))
    dense_K = protocol_kmax(protocol, dense_x)
    dense_rate = surface.evaluate(dense_K, R).rate_m_per_cycle
    dense_total = float(cumulative_trapezoid(geometry.width_m/dense_rate, dense_x, initial=0.0)[-1])
    half_tol = quad(dNdx, x0, x1, epsrel=relative_tolerance/2, epsabs=5e-9, limit=500)[0]
    convergence = max(abs(total-half_tol), abs(total-dense_total)) / max(abs(total), 1e-300)
    if convergence >= 1e-5 or np.any(np.diff(N) <= 0):
        raise RuntimeError(f"virtual integration convergence/monotonicity failure: {convergence:.3e}")

    if protocol == "FIXED_LOAD":
        Pmax = np.full_like(x, ct_load_from_k_pa_sqrt_m(float(K[0])*1e6, geometry, x0*geometry.width_m))
    else:
        Pmax = np.asarray([ct_load_from_k_pa_sqrt_m(float(k)*1e6, geometry, float(q)*geometry.width_m) for k, q in zip(K, x)])
    Kmin = float(R)*K
    full = K-Kmin
    tensile = K-np.maximum(Kmin, 0.0)
    rows = []
    for values in zip(x, K, Kmin, full, tensile, ev.rate_m_per_cycle, ev.local_slope, N, Pmax):
        xi, kmax, kmin, dk, dkp, rate, slope, cycles, pmax = map(float, values)
        rows.append({"option":surface.option,"surface_version":surface.version,"geometry_W_m":geometry.width_m,
          "geometry_B_m":geometry.thickness_m,"protocol":protocol,"R":float(R),"a_m":xi*geometry.width_m,
          "a_over_W":xi,"Pmax_N":pmax,"Pmin_N":float(R)*pmax,"Kmax_MPa_sqrt_m":kmax,"Kmin_MPa_sqrt_m":kmin,
          "deltaK_full_MPa_sqrt_m":dk,"deltaK_tensile_MPa_sqrt_m":dkp,"local_da_dN":rate,
          "local_effective_slope":slope,"cumulative_cycles":cycles,"elapsed_seconds_1000Hz":cycles/1000.0})
    metadata = {"total_cycles":float(N[-1]),"quad_total_cycles":float(total),"dense_trapezoid_total_cycles":dense_total,
      "half_tolerance_total_cycles":float(half_tol),"maximum_relative_convergence_difference":float(convergence),
      "output_points":len(rows),"relative_tolerance":relative_tolerance,"monotonic_N":bool(np.all(np.diff(N)>0)),
      "K_domain_margin_low":float(K.min()-surface.K_min),"K_domain_margin_high":float(surface.K_max-K.max())}
    return rows, metadata
