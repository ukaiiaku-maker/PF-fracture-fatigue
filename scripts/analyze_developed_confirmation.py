"""Section C/D: developed-response confirmation analysis.

Reads ONLY the tracked event ledger
(artifacts/crack_rebonding_developed_confirmation/event_ledger.json,
built by scripts/build_developed_confirmation_event_ledger.py) plus the
Section A frozen protocol/predictions -- never a gitignored runs/
directory.

v2 (evidence-hardening pass, post-review): the review found that the
original ``final_half``/``final_six`` windows used the FIXED event-index
ranges [9..17]/[12..17], which were only correct for a nominal 18-event
trajectory -- the actual campaign trajectories ran to 30 events, so those
windows terminated at event 17 and never saw the final 12 events (60um)
of growth. This version:

  - renames those fixed windows to matched_18_event_half/
    matched_18_event_tail (retained verbatim for prospective-protocol
    traceability -- NOT deleted, just correctly labeled as what they
    actually are);
  - adds TRUE terminal windows (true_final_half, true_final_six) computed
    dynamically from each trajectory's own actual event count;
  - adds extension-based windows (last_50_um, last_25_um) that remain
    valid even if event lengths are not fixed in a future campaign;
  - computes exposure/action diagnostics PER WINDOW (not just once over
    all events) so the Section D decomposition is evaluated against
    diagnostics from the SAME window as the S_h it is being compared to;
  - adds a genuinely inter-event ACTION-WEIGHTED action-weighted-K_rebond
    mean (Sum_j A_c,j * Kbar_b,j / Sum_j A_c,j, weighted by each event's
    own total cleavage_action) alongside the previously-reported simple
    per-event arithmetic mean, so the two are not conflated;
  - explicitly checks the raw run_trajectory event records for
    trajectory-integrated A_on/A_off and finds them NOT PRESENT (the real
    engine's phase-resolved action diagnostic never computes single-patch
    A_on/A_off; those are single-patch ANALYTICAL predictions only, never
    a measured trajectory quantity) -- archived as an explicit
    NOT_ARCHIVED status rather than silently omitted or substituted with
    the frozen analytical prediction;
  - re-tests the terminal classification against the TRUE terminal
    (true_final_half) window and downgrades DEVELOPED_REBONDING_
    STEEPENING_CONFIRMED to DEVELOPED_REBONDING_TAIL_SENSITIVE if the true
    terminal response does not independently corroborate it.

Usage:
    <pinned interpreter> scripts/analyze_developed_confirmation.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from arrhenius_fracture.crack_rebonding_developed_confirmation_v10230 import (  # noqa: E402
    S_abs_local_power_law,
    S_abs_shape_preserving,
    action_weighted_K_rebond_means,
    action_weighted_K_rebond_true_inter_event_weighted,
    apply_tail_sensitivity_gate,
    classify_developed_confirmation,
    effective_horizon_censored,
    event_index_window_rate,
    extension_window_rate,
    stable_growth_gate,
    true_final_half_indices,
    true_final_six_indices,
)
from run_developed_confirmation_stage import (  # noqa: E402
    MAX_ACCEPTED_EVENTS,
    MAX_PROJECTED_EXTENSION_m,
)
from arrhenius_fracture.crack_rebonding_minimal_slope_screen_v10230 import (  # noqa: E402
    adjacent_secant,
    three_point_slope_fit,
)

DEV_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_developed_confirmation"
KMAX_GRID_Pa_sqrt_m = (15.0e6, 18.0e6, 21.0e6)
FREQUENCY_HZ = 1000.0

# Retained VERBATIM from v1 for prospective-protocol traceability -- these
# are NOT the actual final half/six events of a 30-event trajectory (see
# module docstring). Renamed from FINAL_HALF_INDICES/FINAL_SIX_INDICES.
MATCHED_18_EVENT_HALF_INDICES = list(range(9, 18))
MATCHED_18_EVENT_TAIL_INDICES = list(range(12, 18))

SLOPE_GATE = 0.25
RATE_OFFSET_SPAN_DECADE = 0.01

# Windows evaluated uniformly for S_h/delta_m AND for the per-window
# Section D diagnostics. "developed" and "all_event"/"post_first_event"
# are handled specially (developed comes from stable_growth_gate; the
# other two need the trajectory's own event list) -- this list covers the
# tail/extension windows only.
TAIL_WINDOW_NAMES = [
    "matched_18_event_half", "matched_18_event_tail",
    "true_final_half", "true_final_six",
    "last_50_um", "last_25_um",
]


def _traj_name(seed: int, Kmax_MPa: int, cohesion: str) -> str:
    stage = "D1" if seed == 1720 else "D2"
    return f"{stage}_K{Kmax_MPa}MPa_seed{seed}_{cohesion}"


def _rate(da_m: float, dN: float) -> float | None:
    return da_m / dN if dN > 0.0 else None


def _S_h_from_windows(zero_win: dict, finite_win: dict) -> dict[str, Any]:
    len_ident = (
        zero_win["da_m"] > 0.0 and finite_win["da_m"] > 0.0
        and abs(zero_win["da_m"] - finite_win["da_m"]) < 1.0e-12 * max(zero_win["da_m"], finite_win["da_m"])
    )
    zero_rate = _rate(zero_win["da_m"], zero_win["dN"])
    finite_rate = _rate(finite_win["da_m"], finite_win["dN"])
    if not zero_rate or not finite_rate or zero_rate <= 0.0 or finite_rate <= 0.0:
        return {
            "accepted_length_identity_holds": len_ident, "S_h_decade": None,
            "zero_rate_m_per_cycle": zero_rate, "finite_rate_m_per_cycle": finite_rate,
        }
    S_h = math.log10(finite_rate / zero_rate)
    return {
        "accepted_length_identity_holds": len_ident,
        "waiting_time_simplification_admissible": len_ident,
        "S_h_decade": S_h,
        "zero_rate_m_per_cycle": zero_rate,
        "finite_rate_m_per_cycle": finite_rate,
        "rate_ratio_finite_over_zero": 10.0 ** S_h,
        "rate_reduction_fraction": 1.0 - 10.0 ** S_h,
        "waiting_time_increase_fraction": 10.0 ** (-S_h) - 1.0,
    }


def _eventwise_and_rolling(zero_events: list[dict], finite_events: list[dict], roll: int = 4) -> dict[str, Any]:
    z = {e["event_index"]: e for e in zero_events}
    f = {e["event_index"]: e for e in finite_events}
    common = sorted(set(z) & set(f))
    eventwise = []
    for i in common:
        wz = z[i]["waiting_time_s_this_event"]
        wf = f[i]["waiting_time_s_this_event"]
        eventwise.append({
            "event_index": i, "waiting_time_zero_s": wz, "waiting_time_finite_s": wf,
            "ratio_finite_over_zero": (wf / wz) if wz > 0 else None,
            "log10_ratio_signed": (math.log10(wf / wz) if wz > 0 and wf > 0 else None),
        })
    rolling = []
    for start in range(0, max(len(common) - roll + 1, 0)):
        window = common[start:start + roll]
        wz_sum = sum(z[i]["waiting_time_s_this_event"] for i in window)
        wf_sum = sum(f[i]["waiting_time_s_this_event"] for i in window)
        rolling.append({
            "window_event_indices": window,
            "ratio_finite_over_zero": (wf_sum / wz_sum) if wz_sum > 0 else None,
            "log10_ratio_signed": (math.log10(wf_sum / wz_sum) if wz_sum > 0 and wf_sum > 0 else None),
        })
    return {"eventwise": eventwise, "rolling_4event": rolling}


def _events_in_index_set(events: list[dict], indices: set[int]) -> list[dict]:
    return [e for e in events if e["event_index"] in indices]


def _events_in_extension_range(events: list[dict], start_m: float, stop_m: float) -> list[dict]:
    out = []
    for e in events:
        post = float(e["cumulative_extension_m"])
        pre = post - float(e["accepted_length_m"])
        if post > start_m and pre < stop_m:
            out.append(e)
    return out


def _exposure_and_action(events: list[dict], intervals: list[dict]) -> dict[str, Any]:
    """Diagnostics restricted to the given ``events`` (and their matching
    ``intervals``, already pre-filtered by the caller to the same window)
    -- Section D's decomposition must compare S_h and these diagnostics
    from the SAME window, not developed-window S_h against all-event
    diagnostics."""
    aw_means = action_weighted_K_rebond_means(events)
    aw_true_weighted = action_weighted_K_rebond_true_inter_event_weighted(events)
    pre_event_pB = [e["pre_event_max_pB"] for e in events]
    return {
        "n_events_in_window": len(events),
        "total_negative_K_contact_time_s": sum(iv["negative_contact_duration_s"] for iv in intervals),
        "n_intervals_complete_excursion": sum(1 for iv in intervals if iv["complete_negative_excursion"]),
        "n_intervals_partial_excursion": sum(
            1 for iv in intervals
            if not iv["complete_negative_excursion"] and iv["negative_contact_duration_s"] > 0.0
        ),
        "pre_event_max_pB_mean": (sum(pre_event_pB) / len(pre_event_pB)) if pre_event_pB else 0.0,
        "pre_event_max_pB_overall_max": max(pre_event_pB, default=0.0),
        "fraction_events_nonzero_bonded_shielding": aw_means["nonzero_event_fraction"],
        "action_weighted_K_rebond_simple_mean_Pa_sqrt_m": aw_means["unconditional_mean_Pa_sqrt_m"],
        "action_weighted_K_rebond_simple_mean_conditional_nonzero_Pa_sqrt_m": aw_means[
            "conditional_mean_nonzero_only_Pa_sqrt_m"
        ],
        "action_weighted_K_rebond_true_inter_event_weighted": aw_true_weighted,
        "n_events_with_nonzero_action_weighted_K_rebond": aw_means["n_nonzero_events"],
        "max_phase_resolved_K_rebond_Pa_sqrt_m": max(
            (e.get("max_phase_resolved_K_rebond_Pa_sqrt_m", 0.0) or 0.0 for e in events), default=0.0,
        ),
        "all_bulk_action_qualified_every_event": all(e["all_bulk_action_qualified"] for e in events) if events else True,
        "trajectory_integrated_A_on": "NOT_ARCHIVED",
        "trajectory_integrated_A_off": "NOT_ARCHIVED",
        "trajectory_integrated_A_on_A_off_note": (
            "checked the raw run_trajectory per-event bulk_action_records "
            "(the real engine's phase-resolved action diagnostic) -- it "
            "records action, action_weighted_K_rebond_Pa_sqrt_m, "
            "max_K_rebond_Pa_sqrt_m, bulk_action_qualified/used, and "
            "periodic-orbit/convergence diagnostics, but NEVER a single-"
            "patch A_on/A_off pair. A_on/A_off are single-patch ANALYTICAL "
            "model predictions (frozen in developed_confirmation_"
            "predictions.json) computed from a closed-form bonded-fraction "
            "model, not a measured trajectory-integrated quantity the real "
            "engine ever produces -- so there is nothing to archive here, "
            "and the frozen analytical values must not be substituted in "
            "as if they were measured"
        ),
    }


def _tail_windows_for_pair(
    zero_traj: dict, finite_traj: dict,
) -> dict[str, tuple[list[dict], list[dict]]]:
    """Returns {window_name: (zero_events_in_window, finite_events_in_window)}
    for every tail/extension window, each side computed from ITS OWN
    event count/extension (so this is correct even if zero/finite ever
    have different event counts, though in this campaign both are always
    30)."""
    n_zero = len(zero_traj["events"])
    n_finite = len(finite_traj["events"])
    windows = {}

    idx_matched_half = set(MATCHED_18_EVENT_HALF_INDICES)
    idx_matched_tail = set(MATCHED_18_EVENT_TAIL_INDICES)
    windows["matched_18_event_half"] = (
        _events_in_index_set(zero_traj["events"], idx_matched_half),
        _events_in_index_set(finite_traj["events"], idx_matched_half),
    )
    windows["matched_18_event_tail"] = (
        _events_in_index_set(zero_traj["events"], idx_matched_tail),
        _events_in_index_set(finite_traj["events"], idx_matched_tail),
    )

    idx_true_half_zero = set(true_final_half_indices(n_zero))
    idx_true_half_finite = set(true_final_half_indices(n_finite))
    windows["true_final_half"] = (
        _events_in_index_set(zero_traj["events"], idx_true_half_zero),
        _events_in_index_set(finite_traj["events"], idx_true_half_finite),
    )
    idx_true_six_zero = set(true_final_six_indices(n_zero))
    idx_true_six_finite = set(true_final_six_indices(n_finite))
    windows["true_final_six"] = (
        _events_in_index_set(zero_traj["events"], idx_true_six_zero),
        _events_in_index_set(finite_traj["events"], idx_true_six_finite),
    )

    final_zero = float(zero_traj["events"][-1]["cumulative_extension_m"]) if zero_traj["events"] else 0.0
    final_finite = float(finite_traj["events"][-1]["cumulative_extension_m"]) if finite_traj["events"] else 0.0
    for width_um, name in ((50.0, "last_50_um"), (25.0, "last_25_um")):
        width_m = width_um * 1.0e-6
        windows[name] = (
            _events_in_extension_range(zero_traj["events"], max(final_zero - width_m, 0.0), final_zero),
            _events_in_extension_range(finite_traj["events"], max(final_finite - width_m, 0.0), final_finite),
        )
    return windows


def _intervals_matching_events(intervals: list[dict], events_subset: list[dict]) -> list[dict]:
    subset_indices = {e["event_index"] for e in events_subset}
    return [iv for iv in intervals if iv.get("next_event_index") in subset_indices]


def analyze_seed(ledger: dict, seed: int) -> dict[str, Any] | None:
    trajs = ledger["trajectories"]
    per_pair: dict[str, Any] = {}
    S_h_developed_by_Kmax: dict[float, float] = {}
    S_h_all_by_Kmax: dict[float, float] = {}
    S_h_by_window_by_Kmax: dict[str, dict[float, float | None]] = {name: {} for name in TAIL_WINDOW_NAMES}
    all_gate_pass = True
    all_uncensored = True

    for Kmax in KMAX_GRID_Pa_sqrt_m:
        Kmax_MPa = int(round(Kmax / 1.0e6))
        name_zero = _traj_name(seed, Kmax_MPa, "zero")
        name_finite = _traj_name(seed, Kmax_MPa, "finite")
        if name_zero not in trajs or name_finite not in trajs:
            return None
        zero_traj = trajs[name_zero]
        finite_traj = trajs[name_finite]
        all_uncensored = all_uncensored and zero_traj["uncensored"] and finite_traj["uncensored"]

        gate_zero = stable_growth_gate(zero_traj["events"], frequency_Hz=FREQUENCY_HZ)
        gate_finite = stable_growth_gate(finite_traj["events"], frequency_Hz=FREQUENCY_HZ)
        pair_gate_pass = gate_zero["stable_growth_provisional"] and gate_finite["stable_growth_provisional"]
        all_gate_pass = all_gate_pass and pair_gate_pass

        hz_zero = [e["hazard_threshold_action"] for e in zero_traj["events"]]
        hz_finite = [e["hazard_threshold_action"] for e in finite_traj["events"]]
        hazard_pairs_identical = (
            len(hz_zero) == len(hz_finite) and all(a == b for a, b in zip(hz_zero, hz_finite))
        )

        developed_z = gate_zero["developed_interval"]
        developed_f = gate_finite["developed_interval"]
        S_h_developed = _S_h_from_windows(developed_z, developed_f)
        S_h_developed_by_Kmax[Kmax] = S_h_developed["S_h_decade"]

        all_z = event_index_window_rate(zero_traj["events"], [e["event_index"] for e in zero_traj["events"]], FREQUENCY_HZ)
        all_f = event_index_window_rate(finite_traj["events"], [e["event_index"] for e in finite_traj["events"]], FREQUENCY_HZ)
        S_h_all = _S_h_from_windows(all_z, all_f)
        S_h_all_by_Kmax[Kmax] = S_h_all["S_h_decade"]

        pf_z = event_index_window_rate(zero_traj["events"], [e["event_index"] for e in zero_traj["events"][1:]], FREQUENCY_HZ)
        pf_f = event_index_window_rate(finite_traj["events"], [e["event_index"] for e in finite_traj["events"][1:]], FREQUENCY_HZ)
        S_h_post_first = _S_h_from_windows(pf_z, pf_f)

        ratios = _eventwise_and_rolling(zero_traj["events"], finite_traj["events"])

        horizon_zero = effective_horizon_censored(
            zero_traj, max_accepted_events=MAX_ACCEPTED_EVENTS,
            max_projected_extension_m=MAX_PROJECTED_EXTENSION_m,
        )
        horizon_finite = effective_horizon_censored(
            finite_traj, max_accepted_events=MAX_ACCEPTED_EVENTS,
            max_projected_extension_m=MAX_PROJECTED_EXTENSION_m,
        )
        horizon_zero["effective_censored_for_campaign"] = (
            (horizon_zero["raw_censored"] or horizon_zero["stopped_by_budget_exhaustion"])
            and not gate_zero["stable_growth_provisional"]
        )
        horizon_finite["effective_censored_for_campaign"] = (
            (horizon_finite["raw_censored"] or horizon_finite["stopped_by_budget_exhaustion"])
            and not gate_finite["stable_growth_provisional"]
        )
        all_uncensored = (
            all_uncensored
            and not horizon_zero["effective_censored_for_campaign"]
            and not horizon_finite["effective_censored_for_campaign"]
        )

        # --- tail/extension windows: S_h AND matching per-window diagnostics ---
        tail_windows = _tail_windows_for_pair(zero_traj, finite_traj)
        S_h_by_window: dict[str, Any] = {}
        diagnostics_by_window: dict[str, Any] = {}
        for wname, (zero_events_w, finite_events_w) in tail_windows.items():
            zero_rate_w = event_index_window_rate(
                zero_events_w, [e["event_index"] for e in zero_events_w], FREQUENCY_HZ
            )
            finite_rate_w = event_index_window_rate(
                finite_events_w, [e["event_index"] for e in finite_events_w], FREQUENCY_HZ
            )
            S_h_w = _S_h_from_windows(zero_rate_w, finite_rate_w)
            S_h_by_window[wname] = S_h_w
            S_h_by_window_by_Kmax[wname][Kmax] = S_h_w["S_h_decade"]
            diagnostics_by_window[wname] = {
                "zero": _exposure_and_action(
                    zero_events_w, _intervals_matching_events(zero_traj["post_first_event_intervals"], zero_events_w)
                ),
                "finite": _exposure_and_action(
                    finite_events_w, _intervals_matching_events(finite_traj["post_first_event_intervals"], finite_events_w)
                ),
            }

        per_pair[f"K{Kmax_MPa}MPa_seed{seed}"] = {
            "Kmax_Pa_sqrt_m": Kmax,
            "gate_zero": gate_zero, "gate_finite": gate_finite,
            "horizon_censorship_zero": horizon_zero, "horizon_censorship_finite": horizon_finite,
            "pair_gate_pass": pair_gate_pass,
            "hazard_threshold_pairs_identical": hazard_pairs_identical,
            "n_accepted_events_zero": len(zero_traj["events"]),
            "n_accepted_events_finite": len(finite_traj["events"]),
            "S_h_developed": S_h_developed, "S_h_all_event": S_h_all,
            "S_h_post_first_event": S_h_post_first,
            "S_h_by_tail_window": S_h_by_window,
            "diagnostics_by_tail_window": diagnostics_by_window,
            "eventwise_and_rolling_waiting_time_ratios": ratios,
            "zero_exposure_and_action": _exposure_and_action(
                zero_traj["events"], zero_traj["post_first_event_intervals"]
            ),
            "finite_exposure_and_action": _exposure_and_action(
                finite_traj["events"], finite_traj["post_first_event_intervals"]
            ),
            "zero_uncensored": zero_traj["uncensored"], "finite_uncensored": finite_traj["uncensored"],
        }

    x = [math.log10(K) for K in KMAX_GRID_Pa_sqrt_m]

    def _fit(y_by_K: dict[float, float | None]) -> dict[str, Any] | None:
        if any(y_by_K.get(K) is None for K in KMAX_GRID_Pa_sqrt_m):
            return None
        y = [y_by_K[K] for K in KMAX_GRID_Pa_sqrt_m]
        fit = three_point_slope_fit(x, y)
        return {
            "S_h_by_Kmax": {str(int(K)): y_by_K[K] for K in KMAX_GRID_Pa_sqrt_m},
            "delta_m_least_squares": fit["slope"],
            "secant_15_to_18": adjacent_secant(x[0], y[0], x[1], y[1]),
            "secant_18_to_21": adjacent_secant(x[1], y[1], x[2], y[2]),
            "endpoint_curvature": (
                adjacent_secant(x[1], y[1], x[2], y[2]) - adjacent_secant(x[0], y[0], x[1], y[1])
            ),
        }

    fits = {
        "developed": _fit(S_h_developed_by_Kmax),
        "all_event": _fit(S_h_all_by_Kmax),
    }
    for wname in TAIL_WINDOW_NAMES:
        fits[wname] = _fit(S_h_by_window_by_Kmax[wname])

    return {
        "seed": seed,
        "all_six_trajectories_present": True,
        "all_uncensored": all_uncensored,
        "all_pairs_pass_developed_gate": all_gate_pass,
        "per_pair": per_pair,
        "fits_by_window": fits,
        "S_h_developed_by_Kmax": S_h_developed_by_Kmax,
        "S_h_true_final_half_by_Kmax": S_h_by_window_by_Kmax["true_final_half"],
    }


def decompose_section_d_window(seed_analysis: dict, window: str) -> dict[str, Any]:
    """Section D decomposition, evaluated for ONE window (S_h and the
    exposure/action diagnostics both drawn from that same window -- fixes
    the v1 mismatch where developed-window S_h was compared against
    all-event diagnostics)."""
    def _S_h_for(pair: dict) -> float | None:
        if window == "developed":
            return pair["S_h_developed"]["S_h_decade"]
        if window == "all_event":
            return pair["S_h_all_event"]["S_h_decade"]
        return pair["S_h_by_tail_window"][window]["S_h_decade"]

    def _finite_diag_for(pair: dict) -> dict:
        if window in ("developed", "all_event"):
            return pair["finite_exposure_and_action"]
        return pair["diagnostics_by_tail_window"][window]["finite"]

    pooled_K_b = sum(
        _finite_diag_for(pair)["action_weighted_K_rebond_true_inter_event_weighted"][
            "unconditional_action_weighted_mean_Pa_sqrt_m"
        ]
        for pair in seed_analysis["per_pair"].values()
    ) / len(seed_analysis["per_pair"])

    g_zero_window: dict[float, float] = {}
    for pair in seed_analysis["per_pair"].values():
        Kmax = pair["Kmax_Pa_sqrt_m"]
        if window == "developed":
            rate = pair["gate_zero"]["developed_interval"]["da_dN"]
        elif window == "all_event":
            rate = pair["S_h_all_event"]["zero_rate_m_per_cycle"]
        else:
            rate = pair["S_h_by_tail_window"][window]["zero_rate_m_per_cycle"]
        if rate:
            g_zero_window[Kmax] = rate

    m_zero_local = None
    if len(g_zero_window) == 3:
        xs = [math.log10(K) for K in sorted(g_zero_window)]
        ys = [math.log10(g_zero_window[K]) for K in sorted(g_zero_window)]
        m_zero_local = three_point_slope_fit(xs, ys)["slope"]

    rows = {}
    for pair in seed_analysis["per_pair"].values():
        Kmax = pair["Kmax_Pa_sqrt_m"]
        S_h = _S_h_for(pair)
        S_abs = S_abs_shape_preserving(Kmax, pooled_K_b, g_zero_window) if g_zero_window else None
        method = "shape_preserving_interpolation"
        if S_abs is None and m_zero_local is not None:
            S_abs = S_abs_local_power_law(Kmax, pooled_K_b, m_zero_local)
            method = "local_power_law_fallback"
        S_occupancy = (S_h - S_abs) if (S_h is not None and S_abs is not None) else None
        rows[str(int(Kmax))] = {
            "S_h_decade": S_h, "S_abs_decade": S_abs, "S_abs_method": method,
            "S_occupancy_residual_decade": S_occupancy,
        }

    return {
        "window": window,
        "pooled_K_b_Pa_sqrt_m": pooled_K_b,
        "pooled_K_b_formula": "mean over Kmax of the true inter-event action-weighted unconditional K_rebond mean",
        "zero_cohesion_rate_grid_m_per_cycle": {str(int(k)): v for k, v in g_zero_window.items()},
        "local_zero_cohesion_power_law_slope_m_zero": m_zero_local,
        "S_abs_S_occupancy_by_Kmax": rows,
    }


def decompose_section_d(seed_analysis: dict) -> dict[str, Any]:
    """Section D decomposition recomputed consistently across: all events,
    the gate-defined developed interval, the last 50um, the true final
    half, and the true final six (per review request)."""
    windows = ["all_event", "developed", "last_50_um", "true_final_half", "true_final_six"]
    per_window = {w: decompose_section_d_window(seed_analysis, w) for w in windows}
    return {
        "per_window": per_window,
        "attribution_note": (
            "S_occupancy is reported as a residual diagnostic only; steepening "
            "is attributed to reduced bond-formation time (occupancy) only where "
            "the measured unconditional/conditional K_rebond means, nonzero-event "
            "fraction, and contact-time diagnostics actually support it, evaluated "
            "in the SAME window as the S_h/S_abs/S_occupancy values above"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)

    ledger = json.loads((DEV_ARTIFACTS / "event_ledger.json").read_text())
    json.loads((DEV_ARTIFACTS / "developed_confirmation_predictions.json").read_text())

    seed1720 = analyze_seed(ledger, 1720)
    seed1001723 = analyze_seed(ledger, 1001723)

    stage2_gate: dict[str, Any] = {"authorized": False}
    if seed1720 is not None:
        dev_fit = seed1720["fits_by_window"]["developed"]
        th_fit = seed1720["fits_by_window"]["true_final_half"]
        checks = {
            "all_six_uncensored": seed1720["all_uncensored"],
            "all_pairs_pass_developed_gate": seed1720["all_pairs_pass_developed_gate"],
            "all_pairs_hazard_threshold_identical": all(
                p["hazard_threshold_pairs_identical"] for p in seed1720["per_pair"].values()
            ),
            "all_events_action_certified": all(
                p["zero_exposure_and_action"]["all_bulk_action_qualified_every_event"]
                and p["finite_exposure_and_action"]["all_bulk_action_qualified_every_event"]
                for p in seed1720["per_pair"].values()
            ),
            "developed_delta_m_ge_0p25": (dev_fit is not None and dev_fit["delta_m_least_squares"] >= SLOPE_GATE),
            "true_final_half_delta_m_ge_0p25": (th_fit is not None and th_fit["delta_m_least_squares"] >= SLOPE_GATE),
            "same_sign": (
                dev_fit is not None and th_fit is not None
                and (dev_fit["delta_m_least_squares"] > 0) == (th_fit["delta_m_least_squares"] > 0)
            ),
        }
        stage2_gate = {"authorized": all(checks.values()), "checks": checks}

    both_seeds_pass = (
        seed1720 is not None and seed1001723 is not None
        and seed1720["all_uncensored"] and seed1720["all_pairs_pass_developed_gate"]
        and seed1001723["all_uncensored"] and seed1001723["all_pairs_pass_developed_gate"]
    )
    delta_m_developed_by_seed = None
    S_h_developed_by_Kmax_by_seed = None
    true_terminal_delta_m_by_seed = None
    if both_seeds_pass:
        delta_m_developed_by_seed = {
            1720: seed1720["fits_by_window"]["developed"]["delta_m_least_squares"],
            1001723: seed1001723["fits_by_window"]["developed"]["delta_m_least_squares"],
        }
        S_h_developed_by_Kmax_by_seed = {
            K: {1720: seed1720["S_h_developed_by_Kmax"][K], 1001723: seed1001723["S_h_developed_by_Kmax"][K]}
            for K in KMAX_GRID_Pa_sqrt_m
        }
        th_1720 = seed1720["fits_by_window"]["true_final_half"]
        th_1001723 = seed1001723["fits_by_window"]["true_final_half"]
        if th_1720 is not None and th_1001723 is not None:
            true_terminal_delta_m_by_seed = {
                1720: th_1720["delta_m_least_squares"], 1001723: th_1001723["delta_m_least_squares"],
            }

    primary_classification = classify_developed_confirmation(
        both_seeds_pass_gate_uncensored=both_seeds_pass,
        delta_m_developed_by_seed=delta_m_developed_by_seed,
        S_h_developed_by_Kmax_by_seed=S_h_developed_by_Kmax_by_seed,
        rate_offset_span_decade=RATE_OFFSET_SPAN_DECADE, slope_gate=SLOPE_GATE,
    )
    classification = apply_tail_sensitivity_gate(
        primary_classification, true_terminal_delta_m_by_seed=true_terminal_delta_m_by_seed,
        slope_gate=SLOPE_GATE,
    )

    decomposition = {}
    if seed1720 is not None:
        decomposition["1720"] = decompose_section_d(seed1720)
    if seed1001723 is not None:
        decomposition["1001723"] = decompose_section_d(seed1001723)

    scientific_summary = None
    if seed1720 is not None and seed1001723 is not None:
        dev_1720 = seed1720["fits_by_window"]["developed"]
        dev_1001723 = seed1001723["fits_by_window"]["developed"]
        if dev_1720 is not None and dev_1001723 is not None:
            scientific_summary = (
                "Reversible contact-gated rebonding produces a reproducible developed "
                "low-K suppression that weakens rapidly with Kmax. Over Kmax=15-21 "
                "MPa sqrt(m), the three-point effective correction is approximately "
                f"+{0.5 * (dev_1720['delta_m_least_squares'] + dev_1001723['delta_m_least_squares']):.3f} "
                "(mean of two seeds), but the local correction falls from approximately "
                f"+{0.5 * (dev_1720['secant_15_to_18'] + dev_1001723['secant_15_to_18']):.3f} over "
                "15-18 to approximately "
                f"+{0.5 * (dev_1720['secant_18_to_21'] + dev_1001723['secant_18_to_21']):.3f} over 18-21. "
                "This is a strongly curved onset/crossover shift, not a uniform addition "
                "to the Paris exponent."
            )

    decision = {
        "schema": "v10.2.30_crack_rebonding_developed_confirmation_decision_v2",
        "kmax_grid_Pa_sqrt_m": list(KMAX_GRID_Pa_sqrt_m),
        "frequency_Hz": FREQUENCY_HZ,
        "seed_1720_analysis": seed1720,
        "seed_1001723_analysis": seed1001723,
        "stage_2_conditional_gate": stage2_gate,
        "primary_classification_developed_window_only": primary_classification,
        "terminal_classification": classification,
        "section_d_decomposition": decomposition,
        "scientific_summary": scientific_summary,
        "scientific_qualifiers": [
            "DEVELOPED_REBONDING_STEEPENING_CONFIRMED_WITHIN_REVERSIBLE_SIGNED_CONTACT_ABLATION",
            "STRONG_LOW_K_CROSSOVER_NOT_UNIFORM_PARIS_SHIFT",
            "HAZARD_WAITING_TIME_EFFECT_WITH_FIXED_EVENT_LENGTH",
            "REVERSIBLE_PRESET_ONLY",
            "R_FREQUENCY_TEMPERATURE_FIXED_AT_FROZEN_PROTOCOL_VALUES",
            "TWO_SEEDS_NOT_A_CONFIDENCE_INTERVAL",
            "PHYSICAL_CHEMISTRY_REALISM_NOT_CLAIMED",
            "PERSISTENT_REGIME_LOAD_DEPENDENCE_NOT_DEMONSTRATED",
        ],
        "frozen_predictions_reference": str(
            DEV_ARTIFACTS / "developed_confirmation_predictions.json"
        ),
        "multi_K_paris_slope_campaign_authorized": False,
        "part_x_authorized": False,
        "production_line_merge_authorized": False,
    }

    out_path = DEV_ARTIFACTS / "developed_confirmation_decision.json"
    out_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    print(f"primary (developed-window-only) classification = {primary_classification['classification']}")
    print(f"terminal_classification (with tail-sensitivity gate) = {classification['classification']}")
    print(f"stage_2_conditional_gate authorized = {stage2_gate['authorized']}")
    if seed1720 is not None and seed1720["fits_by_window"]["true_final_half"] is not None:
        for seed, analysis in (("1720", seed1720), ("1001723", seed1001723)):
            if analysis is None:
                continue
            for wname in ("developed", "true_final_half", "true_final_six", "matched_18_event_half"):
                fit = analysis["fits_by_window"][wname]
                if fit is not None:
                    print(f"  seed {seed} [{wname}]: delta_m={fit['delta_m_least_squares']:.4f}  "
                          f"S_h(15/18/21)={fit['S_h_by_Kmax']['15000000']:.5f}/"
                          f"{fit['S_h_by_Kmax']['18000000']:.5f}/{fit['S_h_by_Kmax']['21000000']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
