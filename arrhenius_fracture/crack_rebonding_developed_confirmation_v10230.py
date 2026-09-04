"""Developed-response confirmation: the reused developed-growth/
stationarity gate, the Section F terminal classifier, and the Section D
fixed-absolute-shielding decomposition benchmark.

The stationarity gate here is REUSED VERBATIM (same numeric constants,
same formula) from scripts/analyze_v10_2_30_developed_fatigue_growth.py
(lines 322-359), which is the gate scripts/analyze_v10_2_30_
A_native_plus_8PT_final.py cites for the qualified A_NATIVE
~102.668um/18-event campaign ("The 36/36 n80 base trajectories reached
102.668 um and all pass the unchanged stationarity gate."). That script
parses a different run harness's steps_{tag}.csv output; only the
field-name mapping onto the crack-rebonding event schema
(arrhenius_fracture.crack_rebonding_causal_pilot_v2_v10230.run_trajectory)
is new here -- the constants (development_extension_um=20.0,
target_extension_um=100.0, stability_window_um=50.0) and the formula
(>=10 developed events, >=50um of final-window growth, late/early
cumulative rate ratio in [0.5, 2.0]) are unchanged.
"""
from __future__ import annotations

import math
from typing import Any

STABILITY_DEFINITION_STRING = (
    "at_least_10_events_and_50um_developed_growth_and_"
    "late_to_early_cumulative_rate_ratio_between_0p5_and_2"
)

DEVELOPMENT_EXTENSION_UM_DEFAULT = 20.0
TARGET_EXTENSION_UM_DEFAULT = 100.0
STABILITY_WINDOW_UM_DEFAULT = 50.0


def _events_pre_post(events: list[dict]) -> list[dict]:
    """Adapt the crack-rebonding event schema to the (pre, post) extension
    fields the reused gate's interval_rate expects, plus a cycles field
    derived from the fixed cyclic frequency (dN = dt * f is exact for a
    fixed-frequency waveform, not an approximation)."""
    out = []
    for e in events:
        post = float(e["cumulative_extension_m"])
        da = float(e["accepted_length_m"])
        out.append({
            "event_index": e["event_index"],
            "projected_advance_m": da,
            "projected_extension_pre_m": post - da,
            "projected_extension_post_m": post,
            "waiting_time_s_this_event": float(e["waiting_time_s_this_event"]),
        })
    return out


def interval_rate(events: list[dict], start_m: float, stop_m: float, frequency_Hz: float) -> dict[str, Any]:
    selected = [
        row for row in events
        if row["projected_extension_post_m"] > start_m
        and row["projected_extension_pre_m"] < stop_m
    ]
    if not selected:
        return {"event_count": 0, "da_m": 0.0, "dN": 0.0, "da_dN": None}
    da = sum(row["projected_advance_m"] for row in selected)
    dN = sum(row["waiting_time_s_this_event"] * frequency_Hz for row in selected)
    return {
        "event_count": len(selected),
        "da_m": da,
        "dN": dN,
        "da_dN": da / dN if dN > 0.0 else None,
    }


def stable_growth_gate(
    events: list[dict],
    *,
    frequency_Hz: float,
    development_extension_um: float = DEVELOPMENT_EXTENSION_UM_DEFAULT,
    target_extension_um: float = TARGET_EXTENSION_UM_DEFAULT,
    stability_window_um: float = STABILITY_WINDOW_UM_DEFAULT,
) -> dict[str, Any]:
    """Reused verbatim (formula + constants) from
    analyze_v10_2_30_developed_fatigue_growth.py:main, adapted to the
    crack-rebonding event schema via _events_pre_post."""
    mapped = _events_pre_post(events)
    development_m = development_extension_um * 1.0e-6
    target_m = target_extension_um * 1.0e-6
    stability_m = stability_window_um * 1.0e-6

    final_extension = float(mapped[-1]["projected_extension_post_m"]) if mapped else 0.0
    developed = interval_rate(mapped, development_m, max(final_extension, development_m), frequency_Hz)
    stability_start = max(final_extension - stability_m, development_m)
    stability_mid = 0.5 * (stability_start + final_extension)
    early = interval_rate(mapped, stability_start, stability_mid, frequency_Hz)
    late = interval_rate(mapped, stability_mid, final_extension, frequency_Hz)

    ratio = None
    if early.get("da_dN") and late.get("da_dN"):
        ratio = float(late["da_dN"]) / float(early["da_dN"])

    stable = bool(
        developed["event_count"] >= 10
        and (final_extension - development_m) >= stability_m
        and ratio is not None
        and 0.5 <= ratio <= 2.0
    )

    return {
        "development_extension_um": development_extension_um,
        "target_extension_um": target_extension_um,
        "stability_window_um": stability_window_um,
        "final_extension_um": final_extension * 1.0e6,
        "target_reached": final_extension >= target_m,
        "developed_interval": developed,
        "stability_early_interval": early,
        "stability_late_interval": late,
        "late_to_early_rate_ratio": ratio,
        "stable_growth_provisional": stable,
        "stability_definition": STABILITY_DEFINITION_STRING,
        "developed_da_dN_m_per_cycle": developed["da_dN"] if stable else None,
    }


def true_final_half_indices(n_events: int) -> list[int]:
    """The actual last ceil(n/2) event_index values of a trajectory with
    n_events accepted events -- NOT the fixed [9..17] window that was only
    correct for a nominal 18-event trajectory. For n_events=30 this is
    indices 15-29."""
    half = math.ceil(n_events / 2)
    return list(range(max(n_events - half, 0), n_events))


def true_final_six_indices(n_events: int) -> list[int]:
    """The actual last six event_index values -- NOT the fixed [12..17]
    window. For n_events=30 this is indices 24-29."""
    return list(range(max(n_events - 6, 0), n_events))


def extension_window_rate(events: list[dict], width_um: float, frequency_Hz: float) -> dict[str, Any]:
    """da/dN over the last ``width_um`` of PROJECTED EXTENSION (not event
    count) -- robust even if event lengths later cease to be fixed."""
    mapped = _events_pre_post(events)
    if not mapped:
        return {"event_count": 0, "da_m": 0.0, "dN": 0.0, "da_dN": None}
    final_extension = float(mapped[-1]["projected_extension_post_m"])
    width_m = width_um * 1.0e-6
    start_m = max(final_extension - width_m, 0.0)
    return interval_rate(mapped, start_m, final_extension, frequency_Hz)


def effective_horizon_censored(
    traj: dict, *, max_accepted_events: int, max_projected_extension_m: float,
) -> dict[str, Any]:
    """run_trajectory's own ``censored`` flag is set ONLY for a wall-time or
    no-fire failure mid-event; simply exhausting the event-count or
    extension budget between events reports censored=False even though the
    trajectory may never have reached the developed/stationarity target.
    Per Section B ('a horizon-limited trajectory is a censor, never
    da/dN=0'), this function layers that case in: a trajectory is
    EFFECTIVELY horizon-censored for this campaign if run_trajectory's own
    flag is True, OR it stopped because it hit max_accepted_events /
    max_projected_extension_m, in EITHER case without having satisfied the
    developed/stationarity gate (checked by the caller and combined with
    this flag, since a trajectory that reaches the budget AFTER already
    satisfying the gate is not horizon-limited)."""
    n = traj["n_accepted_events"]
    ext = traj["cumulative_extension_m"]
    hit_event_budget = n >= max_accepted_events
    hit_extension_budget = ext >= max_projected_extension_m
    return {
        "raw_censored": traj["censored"],
        "raw_censor_reason": traj["censor_reason"],
        "hit_max_accepted_events_budget": hit_event_budget,
        "hit_max_projected_extension_budget": hit_extension_budget,
        "stopped_by_budget_exhaustion": hit_event_budget or hit_extension_budget,
    }


def event_index_window_rate(events: list[dict], indices: list[int], frequency_Hz: float) -> dict[str, Any]:
    by_index = {e["event_index"]: e for e in events}
    idx = sorted(i for i in indices if i in by_index)
    if not idx:
        return {"event_count": 0, "da_m": 0.0, "dN": 0.0, "da_dN": None}
    da = sum(float(by_index[i]["accepted_length_m"]) for i in idx)
    dN = sum(float(by_index[i]["waiting_time_s_this_event"]) * frequency_Hz for i in idx)
    return {"event_count": len(idx), "da_m": da, "dN": dN, "da_dN": da / dN if dN > 0.0 else None}


def classify_developed_confirmation(
    *,
    both_seeds_pass_gate_uncensored: bool,
    delta_m_developed_by_seed: dict[int, float] | None,
    S_h_developed_by_Kmax_by_seed: dict[float, dict[int, float]] | None,
    rate_offset_span_decade: float = 0.01,
    slope_gate: float = 0.25,
) -> dict[str, Any]:
    """Section F's five terminal classifications.

    If ``both_seeds_pass_gate_uncensored`` is False, only
    FINITE_WINDOW_STEEPENING_NOT_PERSISTENT, DEVELOPED_REBONDING_SEED_
    SENSITIVE, or DEVELOPED_REBONDING_SLOPE_EFFECT_WEAK_OR_UNRESOLVED can
    result (never STEEPENING_CONFIRMED or RATE_OFFSET_LIKE, both of
    which require a hardened two-seed developed-window comparison).
    """
    diagnostics: dict[str, Any] = {
        "both_seeds_pass_gate_uncensored": both_seeds_pass_gate_uncensored,
        "delta_m_developed_by_seed": delta_m_developed_by_seed,
    }

    if not both_seeds_pass_gate_uncensored or delta_m_developed_by_seed is None:
        classification = "FINITE_WINDOW_STEEPENING_NOT_PERSISTENT"
        diagnostics["reason"] = (
            "developed/stationarity gate not satisfied (uncensored + "
            "stable_growth_provisional) by both seeds -- Section E's "
            "conditional gate did not pass, or only one seed has been run"
        )
        diagnostics["classification"] = classification
        return diagnostics

    seeds = sorted(delta_m_developed_by_seed)
    if len(seeds) != 2:
        raise ValueError("expects exactly two seeds once both-seeds gate is satisfied")
    s0, s1 = seeds
    dm0, dm1 = delta_m_developed_by_seed[s0], delta_m_developed_by_seed[s1]

    seeds_disagree_in_sign = (dm0 > 0) != (dm1 > 0)
    delta_m_diff = abs(dm0 - dm1)
    seeds_differ_a_lot = delta_m_diff > slope_gate
    diagnostics.update({
        "seeds_disagree_in_sign": seeds_disagree_in_sign,
        "delta_m_abs_difference": delta_m_diff,
        "seeds_differ_a_lot": seeds_differ_a_lot,
    })

    if seeds_disagree_in_sign or seeds_differ_a_lot:
        classification = "DEVELOPED_REBONDING_SEED_SENSITIVE"
        diagnostics["classification"] = classification
        return diagnostics

    if all(delta_m_developed_by_seed[s] >= slope_gate for s in seeds):
        classification = "DEVELOPED_REBONDING_STEEPENING_CONFIRMED"
        diagnostics["classification"] = classification
        return diagnostics

    # Rate-offset check needs the endpoint S_h span per seed.
    rate_offset_like = False
    if S_h_developed_by_Kmax_by_seed is not None:
        Kmax_lo, Kmax_hi = min(S_h_developed_by_Kmax_by_seed), max(S_h_developed_by_Kmax_by_seed)
        span_by_seed = {
            s: abs(S_h_developed_by_Kmax_by_seed[Kmax_hi][s] - S_h_developed_by_Kmax_by_seed[Kmax_lo][s])
            for s in seeds
        }
        diagnostics["endpoint_span_S_h_by_seed_decade"] = span_by_seed
        rate_offset_like = (
            all(span_by_seed[s] < rate_offset_span_decade for s in seeds)
            and all(abs(delta_m_developed_by_seed[s]) < slope_gate for s in seeds)
        )

    if rate_offset_like:
        classification = "DEVELOPED_REBONDING_RATE_OFFSET_LIKE"
    else:
        classification = "DEVELOPED_REBONDING_SLOPE_EFFECT_WEAK_OR_UNRESOLVED"

    diagnostics["classification"] = classification
    return diagnostics


def apply_tail_sensitivity_gate(
    primary_classification: dict[str, Any],
    *,
    true_terminal_delta_m_by_seed: dict[int, float] | None,
    slope_gate: float = 0.25,
) -> dict[str, Any]:
    """Evidence-hardening pass (post-review): the original Section F
    classification used fixed event-index windows (matched_18_event_half/
    tail = [9..17]/[12..17]) that were only valid for a nominal 18-event
    trajectory. The actual campaign trajectories ran to 30 events, so
    those windows terminate at event 17 and never see the final 12 events
    (60um) of growth. This function re-tests DEVELOPED_REBONDING_
    STEEPENING_CONFIRMED against the TRUE terminal window (the trajectory's
    actual last-half events, true_final_half_indices) and downgrades to
    DEVELOPED_REBONDING_TAIL_SENSITIVE if the true terminal response does
    not independently corroborate the developed-window result.

    Only ever downgrades STEEPENING_CONFIRMED; every other primary
    classification is passed through unchanged (a result that was not
    already confirmed cannot become tail-sensitive -- it is already one of
    the other fail-closed categories)."""
    result = dict(primary_classification)
    result["tail_sensitivity_check"] = {
        "true_terminal_delta_m_by_seed": true_terminal_delta_m_by_seed,
        "applies_only_to": "DEVELOPED_REBONDING_STEEPENING_CONFIRMED",
    }
    if primary_classification["classification"] != "DEVELOPED_REBONDING_STEEPENING_CONFIRMED":
        result["tail_sensitivity_check"]["evaluated"] = False
        return result

    if not true_terminal_delta_m_by_seed or len(true_terminal_delta_m_by_seed) != 2:
        result["classification"] = "DEVELOPED_REBONDING_TAIL_SENSITIVE"
        result["tail_sensitivity_check"]["evaluated"] = True
        result["tail_sensitivity_check"]["reason"] = "true terminal window delta_m unavailable for both seeds"
        return result

    seeds = sorted(true_terminal_delta_m_by_seed)
    s0, s1 = seeds
    dm0, dm1 = true_terminal_delta_m_by_seed[s0], true_terminal_delta_m_by_seed[s1]
    both_positive = dm0 > 0.0 and dm1 > 0.0
    both_ge_gate = dm0 >= slope_gate and dm1 >= slope_gate
    disagree_in_sign = (dm0 > 0) != (dm1 > 0)
    diff = abs(dm0 - dm1)
    within_agreement = diff <= slope_gate

    passes = both_positive and both_ge_gate and not disagree_in_sign and within_agreement
    result["tail_sensitivity_check"].update({
        "evaluated": True,
        "both_positive": both_positive,
        "both_ge_slope_gate": both_ge_gate,
        "seeds_disagree_in_sign": disagree_in_sign,
        "delta_m_abs_difference": diff,
        "within_agreement_le_slope_gate": within_agreement,
        "passes": passes,
    })
    if not passes:
        result["classification"] = "DEVELOPED_REBONDING_TAIL_SENSITIVE"
    return result


def action_weighted_K_rebond_means(events: list[dict]) -> dict[str, Any]:
    """Section D fix: report BOTH the unconditional mean (over ALL events)
    and the conditional mean (over nonzero-K_rebond events only), plus the
    nonzero-event fraction -- the prior campaign's 'conditional mean'
    silently excluded zero-K_rebond events without reporting that fraction."""
    vals = [float(e.get("action_weighted_K_rebond_Pa_sqrt_m") or 0.0) for e in events]
    nonzero = [v for v in vals if v > 0.0]
    n = len(vals)
    return {
        "n_events": n,
        "n_nonzero_events": len(nonzero),
        "nonzero_event_fraction": (len(nonzero) / n) if n else float("nan"),
        "unconditional_mean_Pa_sqrt_m": (sum(vals) / n) if n else 0.0,
        "conditional_mean_nonzero_only_Pa_sqrt_m": (sum(nonzero) / len(nonzero)) if nonzero else 0.0,
    }


def action_weighted_K_rebond_true_inter_event_weighted(events: list[dict]) -> dict[str, Any]:
    """Evidence-hardening pass (post-review): ``action_weighted_K_rebond_
    means`` above reports a SIMPLE per-event arithmetic mean of each
    event's own (already intra-event action-weighted)
    action_weighted_K_rebond_Pa_sqrt_m value -- every event counts
    equally regardless of how much cleavage action it actually carried.
    This function instead computes the genuinely INTER-EVENT
    action-weighted mean:

        <K_b> = sum_j( A_c,j * Kbar_b,j ) / sum_j( A_c,j )

    where A_c,j is event j's own total ``cleavage_action`` and Kbar_b,j is
    its action_weighted_K_rebond_Pa_sqrt_m. The two means can differ
    substantially (observed ~3x in this campaign's data) whenever
    cleavage action is unevenly distributed across events -- they are
    reported as separate, clearly labeled fields, never conflated."""
    weighted_num = 0.0
    weight_den = 0.0
    nonzero_weighted_num = 0.0
    nonzero_weight_den = 0.0
    n_nonzero = 0
    for e in events:
        A_c = float(e.get("cleavage_action") or 0.0)
        K_b = float(e.get("action_weighted_K_rebond_Pa_sqrt_m") or 0.0)
        weighted_num += A_c * K_b
        weight_den += A_c
        if K_b > 0.0:
            nonzero_weighted_num += A_c * K_b
            nonzero_weight_den += A_c
            n_nonzero += 1
    return {
        "formula": "sum(cleavage_action_j * action_weighted_K_rebond_j) / sum(cleavage_action_j)",
        "unconditional_action_weighted_mean_Pa_sqrt_m": (
            weighted_num / weight_den if weight_den > 0.0 else 0.0
        ),
        "conditional_action_weighted_mean_nonzero_only_Pa_sqrt_m": (
            nonzero_weighted_num / nonzero_weight_den if nonzero_weight_den > 0.0 else 0.0
        ),
        "sum_cleavage_action_all_events": weight_den,
        "sum_cleavage_action_nonzero_events": nonzero_weight_den,
        "n_nonzero_events": n_nonzero,
    }


def S_abs_shape_preserving(K: float, K_b: float, g_zero_points: dict[float, float]) -> float | None:
    """S_abs(K) = log10( g_zero(K - K_b) / g_zero(K) ) via shape-preserving
    (monotone piecewise-cubic, falling back to linear with <=2 points)
    interpolation of the discrete zero-cohesion rate points g_zero_points
    = {K: rate}. Returns None if K - K_b falls outside the interpolable
    range (extrapolation is refused rather than silently guessed)."""
    Ks = sorted(g_zero_points)
    if K not in g_zero_points:
        return None
    K_shift = K - K_b
    if K_shift < Ks[0] or K_shift > Ks[-1]:
        return None
    g_at_K = g_zero_points[K]
    g_at_shift = _interp_shape_preserving(Ks, [g_zero_points[k] for k in Ks], K_shift)
    if g_at_shift is None or g_at_shift <= 0.0 or g_at_K <= 0.0:
        return None
    return math.log10(g_at_shift / g_at_K)


def _interp_shape_preserving(xs: list[float], ys: list[float], x: float) -> float | None:
    """Monotone log-linear interpolation (log10(y) linear in x between
    bracketing points) -- shape-preserving in the sense of never
    overshooting between two positive rate samples, and reducing to exact
    endpoints. Used only for the 3-point (Kmax=15/18/21) zero-cohesion
    grid this campaign has; not a general-purpose spline."""
    if x in xs:
        return ys[xs.index(x)]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[i], ys[i + 1]
            if y0 <= 0.0 or y1 <= 0.0:
                return None
            t = (x - x0) / (x1 - x0)
            return 10.0 ** (math.log10(y0) * (1.0 - t) + math.log10(y1) * t)
    return None


def S_abs_local_power_law(K: float, K_b: float, m_zero_local: float) -> float | None:
    """S_abs(K) ~= m_zero(K) * log10(1 - K_b/K) -- the local power-law
    fallback, used only when shape-preserving interpolation is not
    admissible (e.g. K - K_b falls outside the 3-point zero-cohesion
    grid)."""
    if K <= 0.0 or K_b >= K:
        return None
    return m_zero_local * math.log10(1.0 - K_b / K)


__all__ = [
    "STABILITY_DEFINITION_STRING",
    "stable_growth_gate",
    "effective_horizon_censored",
    "interval_rate",
    "event_index_window_rate",
    "true_final_half_indices",
    "true_final_six_indices",
    "extension_window_rate",
    "classify_developed_confirmation",
    "apply_tail_sensitivity_gate",
    "action_weighted_K_rebond_means",
    "action_weighted_K_rebond_true_inter_event_weighted",
    "S_abs_shape_preserving",
    "S_abs_local_power_law",
]
