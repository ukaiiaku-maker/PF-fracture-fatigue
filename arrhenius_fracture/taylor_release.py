"""Selectable Taylor-completion laws for bounded implementation sensitivity.

The accepted production path remains legacy_implicit_1s. The finite_window
path is a declared comparison using a physical correlation interval; it is not
selected by numerical timestep.
"""
from __future__ import annotations

import math
import os

import numpy as np
from scipy.special import gammainc


LEGACY_MODE = "legacy_implicit_1s"
FINITE_WINDOW_MODE = "finite_window"
SEQUENTIAL_MEAN_MODE = "sequential_mean"
VALID_MODES = {LEGACY_MODE, FINITE_WINDOW_MODE, SEQUENTIAL_MEAN_MODE}


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _inputs(single_rate_s, order, rate_cap_s: float) -> tuple[np.ndarray, np.ndarray]:
    cap = _positive_finite(rate_cap_s, "rate_cap_s")
    rate = np.asarray(single_rate_s, dtype=float)
    hits = np.asarray(order, dtype=float)
    rate = np.nan_to_num(rate, nan=0.0, posinf=cap, neginf=0.0)
    hits = np.nan_to_num(hits, nan=1.0, posinf=np.finfo(float).max, neginf=1.0)
    return np.clip(rate, 0.0, cap), np.maximum(hits, 1.0)


def finite_window_completion(
    single_rate_per_unit,
    order,
    *,
    window_in_unit: float,
    rate_cap_per_unit: float,
) -> np.ndarray:
    """Return P(m, lambda*tau)/tau in the caller's inverse-time unit."""
    window = _positive_finite(window_in_unit, "window_in_unit")
    rate, hits = _inputs(single_rate_per_unit, order, rate_cap_per_unit)
    argument = np.clip(rate * window, 0.0, 1.0e12)
    return gammainc(hits, argument) / window


def legacy_completion(single_rate_s, order, *, rate_cap_s: float = 1.0e12) -> np.ndarray:
    """Exact dimensional identity for the accepted numerical mapping.

    The implicit physical reference time is one second, so the result is
    bounded by 1 s^-1.
    """
    return finite_window_completion(
        single_rate_s,
        order,
        window_in_unit=1.0,
        rate_cap_per_unit=rate_cap_s,
    )


def completion_rate(
    single_rate_s,
    order,
    *,
    mode: str = LEGACY_MODE,
    window_s: float = 1.0e-9,
    rate_cap_s: float = 1.0e12,
) -> np.ndarray:
    selected = str(mode).strip().lower()
    if selected == LEGACY_MODE:
        return legacy_completion(single_rate_s, order, rate_cap_s=rate_cap_s)
    if selected == FINITE_WINDOW_MODE:
        return finite_window_completion(
            single_rate_s,
            order,
            window_in_unit=window_s,
            rate_cap_per_unit=rate_cap_s,
        )
    if selected == SEQUENTIAL_MEAN_MODE:
        rate, hits = _inputs(single_rate_s, order, rate_cap_s)
        return rate / hits
    raise ValueError(f"unknown Taylor release mode {mode!r}; expected {sorted(VALID_MODES)}")


def runtime_selection() -> tuple[str, float, float]:
    """Return the explicit sensitivity selection; legacy remains the default."""
    mode = os.environ.get("MPZ_TAYLOR_RELEASE_MODE", LEGACY_MODE)
    window = float(os.environ.get("MPZ_TAYLOR_RELEASE_TIME_S", "1e-9"))
    cap = float(os.environ.get("MPZ_TAYLOR_SINGLE_RATE_CAP_S", "1e12"))
    if mode not in VALID_MODES:
        raise ValueError(f"invalid MPZ_TAYLOR_RELEASE_MODE={mode!r}")
    _positive_finite(window, "MPZ_TAYLOR_RELEASE_TIME_S")
    _positive_finite(cap, "MPZ_TAYLOR_SINGLE_RATE_CAP_S")
    return mode, window, cap


__all__ = [
    "FINITE_WINDOW_MODE",
    "LEGACY_MODE",
    "SEQUENTIAL_MEAN_MODE",
    "completion_rate",
    "finite_window_completion",
    "legacy_completion",
    "runtime_selection",
]
