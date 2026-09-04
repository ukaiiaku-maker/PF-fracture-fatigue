"""Section C/D: developed-response confirmation analysis.

Reads ONLY the tracked event ledger
(artifacts/crack_rebonding_developed_confirmation/event_ledger.json,
built by scripts/build_developed_confirmation_event_ledger.py) plus the
Section A frozen protocol/predictions -- never a gitignored runs/
directory.

Primary observable: production developed_da_dN (m/cycle), from the
reused developed-growth/stationarity gate
(arrhenius_fracture.crack_rebonding_developed_confirmation_v10230.
stable_growth_gate). Secondary/diagnostic: all-event, post-first-event,
final-half (event indices 9-17), final-six (event indices 12-17),
eventwise and 4-event-rolling waiting-time ratios. Section D: corrected
unconditional/conditional action-weighted K_rebond means, plus the
S_abs/S_occupancy fixed-absolute-shielding decomposition benchmark.

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
    classify_developed_confirmation,
    effective_horizon_censored,
    event_index_window_rate,
    stable_growth_gate,
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
FINAL_HALF_INDICES = list(range(9, 18))
FINAL_SIX_INDICES = list(range(12, 18))
SLOPE_GATE = 0.25
RATE_OFFSET_SPAN_DECADE = 0.01


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


def _exposure_and_action(events: list[dict], intervals: list[dict]) -> dict[str, Any]:
    aw_means = action_weighted_K_rebond_means(events)
    pre_event_pB = [e["pre_event_max_pB"] for e in events]
    return {
        "total_negative_K_contact_time_s": sum(iv["negative_contact_duration_s"] for iv in intervals),
        "n_intervals_complete_excursion": sum(1 for iv in intervals if iv["complete_negative_excursion"]),
        "n_intervals_partial_excursion": sum(
            1 for iv in intervals
            if not iv["complete_negative_excursion"] and iv["negative_contact_duration_s"] > 0.0
        ),
        "pre_event_max_pB_mean": (sum(pre_event_pB) / len(pre_event_pB)) if pre_event_pB else 0.0,
        "pre_event_max_pB_overall_max": max(pre_event_pB, default=0.0),
        "fraction_events_nonzero_bonded_shielding": aw_means["nonzero_event_fraction"],
        "action_weighted_K_rebond_unconditional_mean_Pa_sqrt_m": aw_means["unconditional_mean_Pa_sqrt_m"],
        "action_weighted_K_rebond_conditional_mean_nonzero_Pa_sqrt_m": aw_means[
            "conditional_mean_nonzero_only_Pa_sqrt_m"
        ],
        "n_events_with_nonzero_action_weighted_K_rebond": aw_means["n_nonzero_events"],
        "max_phase_resolved_K_rebond_Pa_sqrt_m": max(
            (e.get("max_phase_resolved_K_rebond_Pa_sqrt_m", 0.0) or 0.0 for e in events), default=0.0,
        ),
        "all_bulk_action_qualified_every_event": all(e["all_bulk_action_qualified"] for e in events),
    }


def analyze_seed(ledger: dict, seed: int) -> dict[str, Any] | None:
    trajs = ledger["trajectories"]
    per_pair: dict[str, Any] = {}
    S_h_developed_by_Kmax: dict[float, float] = {}
    S_h_all_by_Kmax: dict[float, float] = {}
    S_h_final_half_by_Kmax: dict[float, float] = {}
    S_h_final_six_by_Kmax: dict[float, float] = {}
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

        fh_z = event_index_window_rate(zero_traj["events"], FINAL_HALF_INDICES, FREQUENCY_HZ)
        fh_f = event_index_window_rate(finite_traj["events"], FINAL_HALF_INDICES, FREQUENCY_HZ)
        S_h_final_half = _S_h_from_windows(fh_z, fh_f)
        S_h_final_half_by_Kmax[Kmax] = S_h_final_half["S_h_decade"]

        fs_z = event_index_window_rate(zero_traj["events"], FINAL_SIX_INDICES, FREQUENCY_HZ)
        fs_f = event_index_window_rate(finite_traj["events"], FINAL_SIX_INDICES, FREQUENCY_HZ)
        S_h_final_six = _S_h_from_windows(fs_z, fs_f)
        S_h_final_six_by_Kmax[Kmax] = S_h_final_six["S_h_decade"]

        ratios = _eventwise_and_rolling(zero_traj["events"], finite_traj["events"])

        horizon_zero = effective_horizon_censored(
            zero_traj, max_accepted_events=MAX_ACCEPTED_EVENTS,
            max_projected_extension_m=MAX_PROJECTED_EXTENSION_m,
        )
        horizon_finite = effective_horizon_censored(
            finite_traj, max_accepted_events=MAX_ACCEPTED_EVENTS,
            max_projected_extension_m=MAX_PROJECTED_EXTENSION_m,
        )
        # A trajectory is a genuine horizon-limited censor for THIS campaign
        # if it stopped on budget exhaustion WITHOUT satisfying the
        # developed/stationarity gate (Section B: never coerced to
        # da/dN=0). If the gate is satisfied, budget exhaustion afterward
        # (or simultaneously) is not a censoring concern.
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

        per_pair[f"K{Kmax_MPa}MPa_seed{seed}"] = {
            "Kmax_Pa_sqrt_m": Kmax,
            "gate_zero": gate_zero, "gate_finite": gate_finite,
            "horizon_censorship_zero": horizon_zero, "horizon_censorship_finite": horizon_finite,
            "pair_gate_pass": pair_gate_pass,
            "hazard_threshold_pairs_identical": hazard_pairs_identical,
            "S_h_developed": S_h_developed, "S_h_all_event": S_h_all,
            "S_h_post_first_event": S_h_post_first,
            "S_h_final_half": S_h_final_half, "S_h_final_six": S_h_final_six,
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
        if any(y_by_K[K] is None for K in KMAX_GRID_Pa_sqrt_m):
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
        "final_half": _fit(S_h_final_half_by_Kmax),
        "final_six": _fit(S_h_final_six_by_Kmax),
    }

    return {
        "seed": seed,
        "all_six_trajectories_present": True,
        "all_uncensored": all_uncensored,
        "all_pairs_pass_developed_gate": all_gate_pass,
        "per_pair": per_pair,
        "fits_by_window": fits,
        "S_h_developed_by_Kmax": S_h_developed_by_Kmax,
    }


def decompose_section_d(seed_analysis: dict) -> dict[str, Any]:
    """Section D: unconditional/conditional K_rebond means (already
    reported per-pair by analyze_seed) plus the S_abs/S_occupancy
    fixed-absolute-shielding benchmark, computed against the POOLED
    developed K_b (mean of the finite-cohesion unconditional
    action-weighted K_rebond means across the 3 Kmax points)."""
    pooled_K_b = sum(
        pair["finite_exposure_and_action"]["action_weighted_K_rebond_unconditional_mean_Pa_sqrt_m"]
        for pair in seed_analysis["per_pair"].values()
    ) / len(seed_analysis["per_pair"])

    g_zero_developed = {
        pair["Kmax_Pa_sqrt_m"]: pair["gate_zero"]["developed_interval"]["da_dN"]
        for pair in seed_analysis["per_pair"].values()
        if pair["gate_zero"]["developed_interval"]["da_dN"]
    }

    # local zero-cohesion power-law slope (least-squares over the 3 points,
    # in log10(rate) vs log10(K)) for the fallback S_abs formula.
    m_zero_local = None
    if len(g_zero_developed) == 3:
        xs = [math.log10(K) for K in sorted(g_zero_developed)]
        ys = [math.log10(g_zero_developed[K]) for K in sorted(g_zero_developed)]
        m_zero_local = three_point_slope_fit(xs, ys)["slope"]

    rows = {}
    for Kmax, pair in ((p["Kmax_Pa_sqrt_m"], p) for p in seed_analysis["per_pair"].values()):
        S_h = pair["S_h_developed"]["S_h_decade"]
        S_abs = S_abs_shape_preserving(Kmax, pooled_K_b, g_zero_developed) if g_zero_developed else None
        method = "shape_preserving_interpolation"
        if S_abs is None and m_zero_local is not None:
            S_abs = S_abs_local_power_law(Kmax, pooled_K_b, m_zero_local)
            method = "local_power_law_fallback"
        S_occupancy = (S_h - S_abs) if (S_h is not None and S_abs is not None) else None
        rows[str(int(Kmax))] = {
            "S_h_developed_decade": S_h, "S_abs_decade": S_abs, "S_abs_method": method,
            "S_occupancy_residual_decade": S_occupancy,
        }

    return {
        "pooled_developed_K_b_Pa_sqrt_m": pooled_K_b,
        "zero_cohesion_developed_rate_grid_m_per_cycle": {str(int(k)): v for k, v in g_zero_developed.items()},
        "local_zero_cohesion_power_law_slope_m_zero": m_zero_local,
        "S_abs_S_occupancy_by_Kmax": rows,
        "attribution_note": (
            "S_occupancy is reported as a residual diagnostic only; steepening "
            "is attributed to reduced bond-formation time (occupancy) only where "
            "the measured unconditional/conditional K_rebond means, nonzero-event "
            "fraction, and contact-time diagnostics in per_pair actually support it"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)

    ledger = json.loads((DEV_ARTIFACTS / "event_ledger.json").read_text())
    predictions = json.loads((DEV_ARTIFACTS / "developed_confirmation_predictions.json").read_text())

    seed1720 = analyze_seed(ledger, 1720)
    seed1001723 = analyze_seed(ledger, 1001723)

    stage2_gate: dict[str, Any] = {"authorized": False}
    if seed1720 is not None:
        dev_fit = seed1720["fits_by_window"]["developed"]
        fh_fit = seed1720["fits_by_window"]["final_half"]
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
            "final_half_delta_m_ge_0p25": (fh_fit is not None and fh_fit["delta_m_least_squares"] >= SLOPE_GATE),
            "same_sign": (
                dev_fit is not None and fh_fit is not None
                and (dev_fit["delta_m_least_squares"] > 0) == (fh_fit["delta_m_least_squares"] > 0)
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
    if both_seeds_pass:
        delta_m_developed_by_seed = {
            1720: seed1720["fits_by_window"]["developed"]["delta_m_least_squares"],
            1001723: seed1001723["fits_by_window"]["developed"]["delta_m_least_squares"],
        }
        S_h_developed_by_Kmax_by_seed = {
            K: {1720: seed1720["S_h_developed_by_Kmax"][K], 1001723: seed1001723["S_h_developed_by_Kmax"][K]}
            for K in KMAX_GRID_Pa_sqrt_m
        }

    classification = classify_developed_confirmation(
        both_seeds_pass_gate_uncensored=both_seeds_pass,
        delta_m_developed_by_seed=delta_m_developed_by_seed,
        S_h_developed_by_Kmax_by_seed=S_h_developed_by_Kmax_by_seed,
        rate_offset_span_decade=RATE_OFFSET_SPAN_DECADE, slope_gate=SLOPE_GATE,
    )

    decomposition = {}
    if seed1720 is not None:
        decomposition["1720"] = decompose_section_d(seed1720)
    if seed1001723 is not None:
        decomposition["1001723"] = decompose_section_d(seed1001723)

    decision = {
        "schema": "v10.2.30_crack_rebonding_developed_confirmation_decision_v1",
        "kmax_grid_Pa_sqrt_m": list(KMAX_GRID_Pa_sqrt_m),
        "frequency_Hz": FREQUENCY_HZ,
        "seed_1720_analysis": seed1720,
        "seed_1001723_analysis": seed1001723,
        "stage_2_conditional_gate": stage2_gate,
        "terminal_classification": classification,
        "section_d_decomposition": decomposition,
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
    print(f"terminal_classification = {classification['classification']}")
    print(f"stage_2_conditional_gate authorized = {stage2_gate['authorized']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
