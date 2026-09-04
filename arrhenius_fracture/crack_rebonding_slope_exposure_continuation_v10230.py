"""Revised slope classification for the exposure-unconditioned continuation.

Extends the minimal slope screen's classification with
REBONDING_PHASE_EXPOSURE_SENSITIVE: unlike the original causal-pilot
qualification question (where compression exposure was a hard admissibility
gate), reduced compression exposure at high Kmax may itself be part of the
physical load dependence under study, so a seed-to-seed disagreement driven
by very different contact exposure is reported as its own class rather than
folded into "weak or unresolved" or forced into a steepens/flattens call.
"""
from __future__ import annotations

from typing import Any


def classify_slope_effect_v2(
    *,
    S_h_by_Kmax: dict[float, dict[int, float]],
    delta_m_by_seed: dict[int, float],
    exposure_by_seed_at_Kmax: dict[int, dict[str, Any]],
    rate_offset_span_decade: float = 0.01,
    slope_gate: float = 0.25,
    exposure_disparity_complete_excursion_ratio: float = 3.0,
) -> dict[str, Any]:
    """S_h_by_Kmax: {Kmax: {seed: S_h}}. delta_m_by_seed: {seed: slope}.
    exposure_by_seed_at_Kmax: {seed: {"n_intervals_complete_excursion": int,
    "n_intervals_partial_excursion": int, "total_negative_K_contact_time_s":
    float}} for the HIGHEST Kmax point only (where exposure sensitivity
    matters most).
    """
    seeds = sorted(delta_m_by_seed)
    if len(seeds) != 2:
        raise ValueError("expects exactly two seeds")
    s0, s1 = seeds
    dm0, dm1 = delta_m_by_seed[s0], delta_m_by_seed[s1]

    Kmax_lo, Kmax_hi = min(S_h_by_Kmax), max(S_h_by_Kmax)
    span_by_seed = {
        seed: abs(S_h_by_Kmax[Kmax_hi][seed] - S_h_by_Kmax[Kmax_lo][seed]) for seed in seeds
    }

    seeds_disagree_in_sign = (dm0 > 0) != (dm1 > 0)
    delta_m_diff = abs(dm0 - dm1)
    seeds_differ_a_lot = delta_m_diff > slope_gate

    exp0 = exposure_by_seed_at_Kmax[s0]
    exp1 = exposure_by_seed_at_Kmax[s1]
    c0 = max(exp0.get("n_intervals_complete_excursion", 0), 0)
    c1 = max(exp1.get("n_intervals_complete_excursion", 0), 0)
    exposure_driven = False
    if seeds_disagree_in_sign or seeds_differ_a_lot:
        lo_c, hi_c = min(c0, c1), max(c0, c1)
        if lo_c == 0 or (hi_c / max(lo_c, 1)) >= exposure_disparity_complete_excursion_ratio:
            exposure_driven = True

    diagnostics = {
        "seeds": seeds,
        "delta_m_by_seed": delta_m_by_seed,
        "delta_m_abs_difference": delta_m_diff,
        "seeds_disagree_in_sign": seeds_disagree_in_sign,
        "seeds_differ_a_lot": seeds_differ_a_lot,
        "endpoint_span_S_h_by_seed_decade": span_by_seed,
        "complete_excursion_count_by_seed_at_Kmax_hi": {s0: c0, s1: c1},
        "high_K_difference_traced_to_exposure_disparity": exposure_driven,
    }

    if (
        all(span_by_seed[seed] < rate_offset_span_decade for seed in seeds)
        and all(abs(delta_m_by_seed[seed]) < slope_gate for seed in seeds)
    ):
        classification = "REBONDING_RATE_OFFSET_LIKE"
    elif seeds_disagree_in_sign or seeds_differ_a_lot or exposure_driven:
        classification = "REBONDING_PHASE_EXPOSURE_SENSITIVE"
    elif all(delta_m_by_seed[seed] >= slope_gate for seed in seeds):
        classification = "REBONDING_STEEPENS_LOCAL_RESPONSE"
    elif all(delta_m_by_seed[seed] <= -slope_gate for seed in seeds):
        classification = "REBONDING_FLATTENS_LOCAL_RESPONSE"
    else:
        classification = "REBONDING_SLOPE_EFFECT_WEAK_OR_UNRESOLVED"

    diagnostics["classification"] = classification
    return diagnostics


__all__ = ["classify_slope_effect_v2"]
