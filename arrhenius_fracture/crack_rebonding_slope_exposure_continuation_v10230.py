"""Revised slope classification for the exposure-unconditioned continuation.

v2 (this revision) fixes a logic bug found on review: the original
``exposure_driven`` check only ever evaluated inside the branch already
guarded by ``seeds_disagree_in_sign or seeds_differ_a_lot``, so it could
never independently change the classification (dead code), and it used
only the binary complete-excursion count rather than a continuous
exposure/action metric. Both are fixed here:

- ``seeds_disagree_in_sign``, ``seeds_differ_a_lot``, and
  ``exposure_disparity_exceeds_threshold`` are now three genuinely
  independent conditions, each evaluated unconditionally and each capable
  of triggering ``REBONDING_PHASE_EXPOSURE_SENSITIVE`` on its own.
- exposure disparity is measured via a continuous quantity
  (``total_negative_K_contact_time_s``, integrated compressive-contact
  duration across the trajectory) rather than the coarse count of
  *complete* negative-K lobes, since partial excursions can carry
  meaningful contact-conditioned formation action too (mission Section 8
  follow-up review).

Reduced compression exposure at high Kmax may itself be part of the
physical load dependence under study (unlike the original causal-pilot
qualification question, where compression exposure was a hard
admissibility gate), so this module reports exposure disparity as an
explanatory/robustness diagnostic, not as a reason to discard a point.
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
    exposure_disparity_ratio_threshold: float = 2.0,
) -> dict[str, Any]:
    """S_h_by_Kmax: {Kmax: {seed: S_h}}. delta_m_by_seed: {seed: slope}.
    exposure_by_seed_at_Kmax: {seed: {"total_negative_K_contact_time_s":
    float, "n_intervals_complete_excursion": int,
    "n_intervals_partial_excursion": int}} for the HIGHEST Kmax point
    (where exposure sensitivity matters most -- fastest opening renewals).
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

    # Condition 1: sign disagreement (independent).
    seeds_disagree_in_sign = (dm0 > 0) != (dm1 > 0)

    # Condition 2: magnitude disagreement (independent).
    delta_m_diff = abs(dm0 - dm1)
    seeds_differ_a_lot = delta_m_diff > slope_gate

    # Condition 3: continuous exposure/action disparity at the highest
    # Kmax point (independent -- evaluated regardless of conditions 1/2,
    # not nested inside them; this is the bug fix).
    exp0 = exposure_by_seed_at_Kmax[s0]
    exp1 = exposure_by_seed_at_Kmax[s1]
    t0 = float(exp0.get("total_negative_K_contact_time_s", 0.0))
    t1 = float(exp1.get("total_negative_K_contact_time_s", 0.0))
    lo_t, hi_t = min(t0, t1), max(t0, t1)
    exposure_disparity_ratio = (hi_t / lo_t) if lo_t > 0.0 else float("inf")
    exposure_disparity_exceeds_threshold = exposure_disparity_ratio >= exposure_disparity_ratio_threshold

    c0 = int(exp0.get("n_intervals_complete_excursion", 0))
    c1 = int(exp1.get("n_intervals_complete_excursion", 0))

    diagnostics = {
        "seeds": seeds,
        "delta_m_by_seed": delta_m_by_seed,
        "delta_m_abs_difference": delta_m_diff,
        "seeds_disagree_in_sign": seeds_disagree_in_sign,
        "seeds_differ_a_lot": seeds_differ_a_lot,
        "endpoint_span_S_h_by_seed_decade": span_by_seed,
        "total_negative_K_contact_time_s_by_seed_at_Kmax_hi": {s0: t0, s1: t1},
        "exposure_disparity_ratio_continuous": exposure_disparity_ratio,
        "exposure_disparity_exceeds_threshold": exposure_disparity_exceeds_threshold,
        "complete_excursion_count_by_seed_at_Kmax_hi": {s0: c0, s1: c1},
        "note": (
            "exposure_disparity_ratio_continuous uses integrated compressive-"
            "contact TIME (a continuous quantity), not the binary complete-"
            "excursion count -- the complete-excursion counts can differ "
            "starkly (e.g. 2 vs 0) while the continuous contact time is "
            "nearly identical, since partial excursions still contribute "
            "real contact duration; this is the metric review found more "
            "physically meaningful for judging whether an outcome is "
            "exposure-driven."
        ),
    }

    # Three genuinely independent triggers this time -- none nested inside
    # another, so exposure disparity CAN change the classification even
    # when the seeds' delta_m happen to agree closely (unlike the v1 bug).
    if (
        all(span_by_seed[seed] < rate_offset_span_decade for seed in seeds)
        and all(abs(delta_m_by_seed[seed]) < slope_gate for seed in seeds)
    ):
        classification = "REBONDING_RATE_OFFSET_LIKE"
    elif seeds_disagree_in_sign or seeds_differ_a_lot or exposure_disparity_exceeds_threshold:
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
