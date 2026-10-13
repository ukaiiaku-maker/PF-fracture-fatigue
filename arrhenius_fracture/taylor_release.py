"""Native Taylor release kinetics and quarantined historical reproductions.

The current solver uses the Arrhenius forward-minus-reverse rate divided by
the density-dependent mean completion order. It has no observation window,
implicit clock, or rate/order cap. The former one-second incomplete-gamma
mapping is retained only as an explicitly named historical reproducer and is
not reachable from :func:`runtime_selection` or :func:`completion_rate`.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import gammainc


NATIVE_MEAN_MODE = "mean_net_rate_over_density_order"
HISTORICAL_LEGACY_1S = "historical_legacy_implicit_1s"


def signed_arrhenius_components(surface, signed_stress_Pa, temperature_K: float):
    """Evaluate cancellation-stable forward, reverse, and directed net rates."""
    stress = np.asarray(signed_stress_Pa, dtype=float)
    if np.any(~np.isfinite(stress)):
        raise ValueError("signed_stress_Pa must be finite")
    T = float(temperature_K)
    if not math.isfinite(T) or T <= 0.0:
        raise ValueError("temperature_K must be positive and finite")
    forward = np.asarray(surface.rate(np.maximum(stress, 0.0), T), dtype=float)
    reverse = np.asarray(surface.rate(np.maximum(-stress, 0.0), T), dtype=float)
    high = np.maximum(forward, reverse)
    low = np.minimum(forward, reverse)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(low, high, out=np.ones_like(high), where=high > 0.0)
        magnitude = np.where(high > 0.0, high * (-np.expm1(np.log(ratio))), 0.0)
    net = np.sign(stress) * magnitude
    return {
        "forward_rate_s": forward, "reverse_rate_s": reverse,
        "net_rate_s": np.where(stress == 0.0, 0.0, net),
    }


def density_completion_order(forest_density_m2, *, correlation_scale: float,
                             correlation_density_m2: float) -> np.ndarray:
    """Return ``1 + scale*sqrt(rho/rho_c)`` without an artificial ceiling."""
    rho = np.asarray(forest_density_m2, dtype=float)
    scale = float(correlation_scale)
    rho_c = float(correlation_density_m2)
    if np.any(~np.isfinite(rho)) or np.any(rho < 0.0):
        raise ValueError("forest_density_m2 must be finite and nonnegative")
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("correlation_scale must be finite and nonnegative")
    if not math.isfinite(rho_c) or rho_c <= 0.0:
        raise ValueError("correlation_density_m2 must be positive and finite")
    return 1.0 + scale * np.sqrt(rho / rho_c)


def completion_rate(net_elementary_rate_s, order) -> np.ndarray:
    """Native directed-positive mean Taylor completion rate in s^-1."""
    rate = np.asarray(net_elementary_rate_s, dtype=float)
    hits = np.asarray(order, dtype=float)
    if np.any(~np.isfinite(rate)) or np.any(~np.isfinite(hits)):
        raise ValueError("Taylor rate and order must be finite")
    if np.any(hits < 1.0):
        raise ValueError("Taylor completion order must be at least one")
    return np.maximum(rate, 0.0) / hits


def runtime_selection() -> tuple[str, None, None]:
    """Return the sole current-framework law; clocks and hard caps are absent."""
    return NATIVE_MEAN_MODE, None, None


def historical_legacy_implicit_1s(single_rate_s, order, *, rate_cap_s=1.0e12):
    """Reproduce the superseded one-second mapping for labeled history only."""
    rate = np.asarray(single_rate_s, dtype=float)
    hits = np.asarray(order, dtype=float)
    cap = float(rate_cap_s)
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("rate_cap_s must be positive and finite")
    rate = np.clip(np.nan_to_num(rate, nan=0.0, posinf=cap, neginf=0.0), 0.0, cap)
    hits = np.maximum(np.nan_to_num(hits, nan=1.0, posinf=np.finfo(float).max), 1.0)
    return gammainc(hits, np.minimum(rate, 1.0e12))


__all__ = [
    "HISTORICAL_LEGACY_1S", "NATIVE_MEAN_MODE", "completion_rate",
    "density_completion_order", "historical_legacy_implicit_1s",
    "runtime_selection", "signed_arrhenius_components",
]
