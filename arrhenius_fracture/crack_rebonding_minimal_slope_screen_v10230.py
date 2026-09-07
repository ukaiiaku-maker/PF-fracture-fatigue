"""Minimal local Paris-slope screen (v10.2.30), reversible regime only.

Staged, bounded follow-up to the completed two-seed causal pilot
(codex/v10.2.30-crack-rebonding-causal-pilot-v2 @ 5bc56c9, preserved
immutable). Tests whether the reproducible ~0.042-decade cohesive
waiting-time effect found at Kmax=18 MPa*sqrt(m) is a flat rate offset or
actually changes the local crack-growth slope, by adding two more loads
(Kmax=15, 21 MPa*sqrt(m)) at the SAME two seeds (1720, 1001723), reusing
the ALREADY-QUALIFIED reversible-regime configs verbatim -- no barrier
re-inversion, no Pi_K rescaling at the new loads.

Authorization chain (see docs/v10_2_30_crack_rebonding_minimal_slope_screen
.md): the reversible/persistent regime-equivalence gate (analysis-only, no
new physics, computed against the parent branch's two tracked event
ledgers) passed comfortably for both seeds
(REVERSIBLE_PERSISTENT_EQUIVALENT_FOR_SLOPE_SCREEN), authorizing this
8-trajectory single-regime screen in place of a 16-trajectory two-regime
one.

Explicitly NOT authorized by this screen: the full multi-K/R/frequency/
dwell/passivation Part X campaign, or a merge into the production line.
"""
from __future__ import annotations

import math
from typing import Any

from arrhenius_fracture.crack_rebonding_causal_pilot_v2_v10230 import (
    F_HZ,
    N_PHASE,
    R_REF,
    T_K,
)
from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (
    CrackRebondingControls,
    _integrate_A_off,
    _integrate_A_on,
    two_state_fixed_point,
)

SCREEN_KMAX_GRID_Pa_sqrt_m = (15.0e6, 18.0e6, 21.0e6)


def analytical_predictions_at_Kmax(
    resolved_cfg: CrackRebondingControls,
    *,
    Kmax_Pa_sqrt_m: float,
    reference_contact_radius_m: float,
    Eprime_Pa: float,
) -> dict[str, Any]:
    """A_on(K), A_off(K), p_B*(K), K_rebond(K) at the given Kmax, holding
    every kinetics/cohesion parameter in resolved_cfg fixed (no barrier
    re-inversion, no Pi_K rescaling) -- a pure re-evaluation of the
    already-frozen single-patch model at a different drive amplitude."""
    a_on = _integrate_A_on(
        resolved_cfg.bond_barrier_eV, T_K=T_K, f_Hz=F_HZ, R=R_REF,
        Kmax_Pa_sqrt_m=Kmax_Pa_sqrt_m, s_j_m=0.0,
        r_contact_m=reference_contact_radius_m, cfg=resolved_cfg, n_phase=360,
    )
    a_off = _integrate_A_off(
        resolved_cfg.rupture_barrier_eV, T_K=T_K, f_Hz=F_HZ, R=R_REF,
        Kmax_Pa_sqrt_m=Kmax_Pa_sqrt_m, s_j_m=0.0,
        r_contact_m=reference_contact_radius_m, cfg=resolved_cfg, n_phase=360,
    )
    fixed_point = two_state_fixed_point(a_on, a_off)
    K_rebond_max = resolved_cfg.rebond_K_geometry_factor * math.sqrt(
        max(Eprime_Pa, 0.0) * max(resolved_cfg.restored_work_of_separation_J_m2, 0.0)
    )
    K_rebond_predicted = K_rebond_max * fixed_point["b_star"]
    Pi_K_at_this_load = K_rebond_max / Kmax_Pa_sqrt_m
    return {
        "Kmax_Pa_sqrt_m": Kmax_Pa_sqrt_m,
        "A_on": a_on,
        "A_off": a_off,
        "p_B_star": fixed_point["b_star"],
        "P_survive": fixed_point["P_survive"],
        "K_rebond_max_Pa_sqrt_m": K_rebond_max,
        "K_rebond_predicted_Pa_sqrt_m": K_rebond_predicted,
        "Pi_K_at_this_load": Pi_K_at_this_load,
    }


def three_point_slope_fit(x: list[float], y: list[float]) -> dict[str, float]:
    """Delta m = sum((x-xbar)(y-ybar)) / sum((x-xbar)^2) -- ordinary
    least-squares slope through the 3 (log10 Kmax, S_h) points."""
    n = len(x)
    xbar = sum(x) / n
    ybar = sum(y) / n
    num = sum((xi - xbar) * (yi - ybar) for xi, yi in zip(x, y))
    den = sum((xi - xbar) ** 2 for xi in x)
    slope = num / den if den > 0.0 else float("nan")
    return {"slope": slope, "xbar": xbar, "ybar": ybar}


def adjacent_secant(x0: float, y0: float, x1: float, y1: float) -> float:
    return (y1 - y0) / (x1 - x0)


def classify_slope_effect(
    S_h_by_Kmax: dict[float, float], delta_m: float, *, rate_offset_span_decade: float = 0.01,
    slope_gate: float = 0.25,
) -> str:
    Kmax_lo = min(S_h_by_Kmax)
    Kmax_hi = max(S_h_by_Kmax)
    span = abs(S_h_by_Kmax[Kmax_hi] - S_h_by_Kmax[Kmax_lo])
    if span < rate_offset_span_decade and abs(delta_m) < slope_gate:
        return "REBONDING_RATE_OFFSET_LIKE"
    if delta_m >= slope_gate:
        return "REBONDING_STEEPENS_LOCAL_RESPONSE"
    if delta_m <= -slope_gate:
        return "REBONDING_FLATTENS_LOCAL_RESPONSE"
    return "REBONDING_SLOPE_EFFECT_WEAK_OR_UNRESOLVED"


__all__ = [
    "SCREEN_KMAX_GRID_Pa_sqrt_m",
    "analytical_predictions_at_Kmax",
    "three_point_slope_fit",
    "adjacent_secant",
    "classify_slope_effect",
]
