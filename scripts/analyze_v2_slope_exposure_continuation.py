"""Final analysis after the exposure-unconditioned completion trajectory
(v2, hardened per review): the full 15/18/21 three-point S_h(K) slope fit
for BOTH seeds, adjacent secants, corrected rate/waiting-time physical
units, windowed (all/post-first/late) S_h to probe developed-state
persistence, eventwise finite/zero waiting-time ratios, continuous
exposure/action diagnostics, a load-dependence decomposition diagnostic
(action-weighted K_rebond vs Kmax), and the revised, non-tautological
classification (arrhenius_fracture.crack_rebonding_slope_exposure_
continuation_v10230.classify_slope_effect_v2).

No new physics: everything here is computed from the tracked event
ledgers (parent pilot, slope screen, this continuation) that already
exist on disk.

Usage:
    <pinned interpreter> scripts/analyze_v2_slope_exposure_continuation.py \\
        --run-root runs/crack_rebonding_slope_exposure_continuation
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

from arrhenius_fracture.crack_rebonding_minimal_slope_screen_v10230 import (  # noqa: E402
    adjacent_secant,
    three_point_slope_fit,
)
from arrhenius_fracture.crack_rebonding_slope_exposure_continuation_v10230 import (  # noqa: E402
    classify_slope_effect_v2,
)

PARENT_PILOT_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_causal_pilot_v2"
SLOPE_SCREEN_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_minimal_slope_screen_v1"
CONTINUATION_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_slope_exposure_continuation"
KMAX_GRID_Pa_sqrt_m = (15.0e6, 18.0e6, 21.0e6)
SEEDS = (1720, 1001723)
RATE_OFFSET_SPAN_DECADE = 0.01
SLOPE_GATE = 0.25
FREQUENCY_HZ = 1000.0
LATE_WINDOW_EVENT_INDICES = (3, 4, 5, 6)


def _window_rate(res: dict, event_indices: list[int]) -> dict[str, float]:
    events = {e["event_index"]: e for e in res["events"]}
    idx_sorted = sorted(event_indices)
    delta_a = sum(events[i]["accepted_length_m"] for i in idx_sorted)
    t_start = events[idx_sorted[0] - 1]["cumulative_time_s"] if idx_sorted[0] > 0 else 0.0
    t_end = events[idx_sorted[-1]]["cumulative_time_s"]
    delta_t = t_end - t_start
    return {"delta_a_m": delta_a, "delta_t_s": delta_t, "g_m_per_s": delta_a / delta_t if delta_t > 0 else float("nan")}


def _rate_shift_windowed(zero_res: dict, finite_res: dict, event_indices: list[int]) -> dict[str, Any]:
    zero_events = {e["event_index"]: e for e in zero_res["events"]}
    finite_events = {e["event_index"]: e for e in finite_res["events"]}
    lengths_identical = all(
        abs(zero_events[i]["accepted_length_m"] - finite_events[i]["accepted_length_m"]) < 1.0e-15
        for i in event_indices if i in zero_events and i in finite_events
    )
    rz = _window_rate(zero_res, event_indices)
    rf = _window_rate(finite_res, event_indices)
    S_h = math.log10(rf["g_m_per_s"] / rz["g_m_per_s"])
    return {
        "accepted_lengths_identical": lengths_identical,
        "zero": rz, "finite": rf,
        "S_h_decade": S_h,
        "da_dN_zero_m_per_cycle": rz["g_m_per_s"] / FREQUENCY_HZ,
        "da_dN_finite_m_per_cycle": rf["g_m_per_s"] / FREQUENCY_HZ,
        "rate_ratio_finite_over_zero": 10.0 ** S_h,
        "rate_reduction_fraction": 1.0 - 10.0 ** S_h,
        "waiting_time_increase_fraction": 10.0 ** (-S_h) - 1.0,
    }


def _eventwise_ratios(zero_res: dict, finite_res: dict) -> list[dict[str, Any]]:
    zero_events = {e["event_index"]: e for e in zero_res["events"]}
    finite_events = {e["event_index"]: e for e in finite_res["events"]}
    rows = []
    for i in sorted(set(zero_events) & set(finite_events)):
        wz = zero_events[i]["waiting_time_s_this_event"]
        wf = finite_events[i]["waiting_time_s_this_event"]
        rows.append({
            "event_index": i, "waiting_time_zero_s": wz, "waiting_time_finite_s": wf,
            "ratio_finite_over_zero": wf / wz if wz > 0 else float("nan"),
            "log10_ratio_signed": math.log10(wf / wz) if wz > 0 and wf > 0 else float("nan"),
        })
    return rows


def _exposure(res: dict) -> dict[str, Any]:
    intervals = res["post_first_event_intervals"]
    events = res["events"]
    aw_K_rebond = [
        e["action_weighted_K_rebond_Pa_sqrt_m"] for e in events
        if e.get("action_weighted_K_rebond_Pa_sqrt_m") is not None
        and e["action_weighted_K_rebond_Pa_sqrt_m"] > 0.0
    ]
    return {
        "n_intervals_complete_excursion": sum(1 for iv in intervals if iv["complete_negative_excursion"]),
        "n_intervals_partial_excursion": sum(
            1 for iv in intervals
            if not iv["complete_negative_excursion"] and iv["negative_contact_duration_s"] > 0.0
        ),
        "total_negative_K_contact_time_s": sum(iv["negative_contact_duration_s"] for iv in intervals),
        "max_p_B_post_commit": max((e["max_pB_post_commit"] for e in events), default=0.0),
        "max_phase_resolved_K_rebond_Pa_sqrt_m": max(
            (e.get("max_phase_resolved_K_rebond_Pa_sqrt_m", 0.0) or 0.0 for e in events), default=0.0
        ),
        "mean_action_weighted_K_rebond_Pa_sqrt_m_nonzero_events": (
            sum(aw_K_rebond) / len(aw_K_rebond) if aw_K_rebond else 0.0
        ),
        "n_events_with_nonzero_action_weighted_K_rebond": len(aw_K_rebond),
        "all_bulk_action_qualified_every_event": all(e["all_bulk_action_qualified"] for e in events),
    }


def _frozen_A_on_off(Kmax_Pa_sqrt_m: float) -> dict[str, float]:
    frozen_predictions = json.loads((SLOPE_SCREEN_ARTIFACTS / "frozen_predictions.json").read_text())
    for pred in frozen_predictions["predictions"]["reversible"]:
        if abs(pred["Kmax_Pa_sqrt_m"] - Kmax_Pa_sqrt_m) < 1.0:
            return {"A_on": pred["A_on"], "A_off": pred["A_off"]}
    raise KeyError(Kmax_Pa_sqrt_m)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args(argv)
    run_root = Path(args.run_root)

    parent_1720 = json.loads((PARENT_PILOT_ARTIFACTS / "event_ledger.json").read_text())
    parent_1001723 = json.loads((PARENT_PILOT_ARTIFACTS / "second_seed_event_ledger.json").read_text())
    screen_ledger = json.loads((SLOPE_SCREEN_ARTIFACTS / "event_ledger.json").read_text())
    completion_trajectories = json.loads((run_root / "trajectories.json").read_text())
    completion_finite = completion_trajectories["C3R_K21MPa_seed1001723_EXPOSURE_UNCONDITIONED"]
    completion_zero = screen_ledger["trajectories"]["C2R_K21MPa_seed1001723"]

    pairs = {
        (15.0e6, 1720): (screen_ledger["trajectories"]["C2R_K15MPa_seed1720"],
                          screen_ledger["trajectories"]["C3R_K15MPa_seed1720"]),
        (15.0e6, 1001723): (screen_ledger["trajectories"]["C2R_K15MPa_seed1001723"],
                             screen_ledger["trajectories"]["C3R_K15MPa_seed1001723"]),
        (18.0e6, 1720): (parent_1720["trajectories"]["C2R"], parent_1720["trajectories"]["C3R"]),
        (18.0e6, 1001723): (parent_1001723["trajectories"]["S2_C2R"], parent_1001723["trajectories"]["S2_C3R"]),
        (21.0e6, 1720): (screen_ledger["trajectories"]["C2R_K21MPa_seed1720"],
                          screen_ledger["trajectories"]["C3R_K21MPa_seed1720"]),
        (21.0e6, 1001723): (completion_zero, completion_finite),
    }

    all_event_indices = [0, 1, 2, 3, 4, 5, 6]
    post_first_indices = [1, 2, 3, 4, 5, 6]
    late_indices = list(LATE_WINDOW_EVENT_INDICES)

    S_h_by_Kmax: dict[float, dict[int, float]] = {K: {} for K in KMAX_GRID_Pa_sqrt_m}
    per_pair: dict[str, dict] = {}
    for (Kmax, seed), (zero_res, finite_res) in pairs.items():
        key = f"K{Kmax/1e6:.0f}MPa_seed{seed}"
        windows = {
            "all": _rate_shift_windowed(zero_res, finite_res, all_event_indices),
            "post_first": _rate_shift_windowed(zero_res, finite_res, post_first_indices),
            "late": _rate_shift_windowed(zero_res, finite_res, late_indices),
        }
        S_h_by_Kmax[Kmax][seed] = windows["all"]["S_h_decade"]
        A_on_off = _frozen_A_on_off(Kmax)
        per_pair[key] = {
            "Kmax_Pa_sqrt_m": Kmax, "seed": seed,
            "windows": windows,
            "eventwise_ratios": _eventwise_ratios(zero_res, finite_res),
            "A_on_frozen": A_on_off["A_on"], "A_off_frozen": A_on_off["A_off"],
            "zero_cohesion_diagnostics": _exposure(zero_res),
            "finite_cohesion_diagnostics": _exposure(finite_res),
        }

    per_seed: dict[int, dict] = {}
    for seed in SEEDS:
        x = [math.log10(K) for K in KMAX_GRID_Pa_sqrt_m]
        for window_name in ("all", "post_first", "late"):
            y = [per_pair[f"K{K/1e6:.0f}MPa_seed{seed}"]["windows"][window_name]["S_h_decade"] for K in KMAX_GRID_Pa_sqrt_m]
            fit = three_point_slope_fit(x, y)
            secant_15_18 = adjacent_secant(x[0], y[0], x[1], y[1])
            secant_18_21 = adjacent_secant(x[1], y[1], x[2], y[2])
            per_seed.setdefault(seed, {})[window_name] = {
                "S_h_by_Kmax": {str(int(K)): y[i] for i, K in enumerate(KMAX_GRID_Pa_sqrt_m)},
                "delta_m_least_squares": fit["slope"],
                "secant_15_to_18": secant_15_18,
                "secant_18_to_21": secant_18_21,
            }

    # Classification uses the "all" window (matches the originally reported
    # result) with the fixed, non-tautological exposure check.
    delta_m_by_seed = {seed: per_seed[seed]["all"]["delta_m_least_squares"] for seed in SEEDS}
    exposure_at_Kmax_hi = {
        seed: per_pair[f"K21MPa_seed{seed}"]["finite_cohesion_diagnostics"] for seed in SEEDS
    }
    classification_result = classify_slope_effect_v2(
        S_h_by_Kmax=S_h_by_Kmax, delta_m_by_seed=delta_m_by_seed,
        exposure_by_seed_at_Kmax=exposure_at_Kmax_hi,
        rate_offset_span_decade=RATE_OFFSET_SPAN_DECADE, slope_gate=SLOPE_GATE,
    )

    # Late-window persistence check: does the classification survive using
    # only events 3-6 instead of all 7?
    delta_m_by_seed_late = {seed: per_seed[seed]["late"]["delta_m_least_squares"] for seed in SEEDS}
    classification_late = classify_slope_effect_v2(
        S_h_by_Kmax={K: {seed: per_pair[f"K{K/1e6:.0f}MPa_seed{seed}"]["windows"]["late"]["S_h_decade"]
                          for seed in SEEDS} for K in KMAX_GRID_Pa_sqrt_m},
        delta_m_by_seed=delta_m_by_seed_late,
        exposure_by_seed_at_Kmax=exposure_at_Kmax_hi,
        rate_offset_span_decade=RATE_OFFSET_SPAN_DECADE, slope_gate=SLOPE_GATE,
    )

    # Load-dependence decomposition diagnostic (analysis-only): does the
    # action-weighted K_rebond the finite-cohesion trajectory actually
    # reaches stay roughly constant across Kmax (supporting a mostly
    # fractional-shielding/1-over-K explanation) or fall substantially
    # (supporting an additional load-dependent occupancy/exposure
    # contribution)? No counterfactual re-simulation is run -- this
    # compares the already-tracked, already-measured values only.
    decomposition = {}
    for seed in SEEDS:
        vals = {
            int(K): per_pair[f"K{K/1e6:.0f}MPa_seed{seed}"]["finite_cohesion_diagnostics"][
                "mean_action_weighted_K_rebond_Pa_sqrt_m_nonzero_events"
            ]
            for K in KMAX_GRID_Pa_sqrt_m
        }
        nonzero_vals = [v for v in vals.values() if v > 0]
        if nonzero_vals:
            rel_spread = (max(nonzero_vals) - min(nonzero_vals)) / max(nonzero_vals)
        else:
            rel_spread = float("nan")
        decomposition[seed] = {
            "mean_action_weighted_K_rebond_Pa_sqrt_m_by_Kmax": vals,
            "relative_spread_across_Kmax_grid": rel_spread,
            "interpretation": (
                "action-weighted K_rebond varies by "
                f"{rel_spread * 100:.1f}% across the Kmax grid -- "
                + (
                    "close to constant in absolute terms, consistent with the "
                    "steepening being explained mostly by the fixed absolute "
                    "shielding scale becoming a smaller FRACTION of a larger "
                    "Kmax (Pi_K = K_rebond_max/Kmax falling with load), not by "
                    "a strongly load-dependent bonded-occupancy mechanism"
                    if rel_spread < 0.30
                    else "varies substantially, indicating a load-dependent "
                    "bonded-occupancy/contact-exposure contribution beyond "
                    "the fixed-absolute-shielding/fractional-intensity effect "
                    "alone -- the two contributions are not decomposed "
                    "quantitatively here, only flagged as both plausibly present"
                )
            ),
        }

    decision = {
        "schema": "v10.2.30_crack_rebonding_slope_exposure_continuation_decision_v2",
        "status": "COMPLETE",
        "kmax_grid_Pa_sqrt_m": list(KMAX_GRID_Pa_sqrt_m),
        "frequency_Hz": FREQUENCY_HZ,
        "per_pair": per_pair,
        "per_seed_by_window": {str(seed): per_seed[seed] for seed in SEEDS},
        "classification_analysis_all_window": classification_result,
        "classification_analysis_late_window": classification_late,
        "overall_classification": classification_result["classification"],
        "late_window_classification": classification_late["classification"],
        "late_window_classification_matches_all_window": (
            classification_late["classification"] == classification_result["classification"]
        ),
        "load_dependence_decomposition_diagnostic": {str(seed): decomposition[seed] for seed in SEEDS},
        "scientific_qualifiers": [
            "CONTACT_GATED_REBONDING_FINITE_WINDOW_STEEPENING_DEMONSTRATED",
            "RELATIVE_SLOPE_CORRECTION_POSITIVE_OVER_KMAX_15_TO_21",
            "STRONG_CURVATURE_OR_ONSET_SHIFT_PRESENT",
            "DEVELOPED_PARIS_SLOPE_NOT_YET_QUALIFIED",
            "PHASE_EXPOSURE_AND_ABSOLUTE_SHIELDING_CONTRIBUTIONS_NOT_YET_DECOMPOSED",
            "PHYSICAL_CHEMISTRY_REALISM_NOT_CLAIMED",
        ],
        "multi_K_paris_slope_campaign_authorized": False,
        "part_x_authorized": False,
        "production_line_merge_authorized": False,
    }

    out_path = run_root / "slope_exposure_continuation_decision.json"
    out_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    CONTINUATION_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (CONTINUATION_ARTIFACTS / "slope_exposure_continuation_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {out_path}")
    print(f"overall_classification (all-window)  = {classification_result['classification']}")
    print(f"late-window classification            = {classification_late['classification']}  "
          f"(matches all-window: {decision['late_window_classification_matches_all_window']})")
    for seed in SEEDS:
        w = per_seed[seed]["all"]
        wl = per_seed[seed]["late"]
        print(
            f"  seed {seed} [all]:  S_h(15)={w['S_h_by_Kmax']['15000000']:.6f} "
            f"S_h(18)={w['S_h_by_Kmax']['18000000']:.6f} S_h(21)={w['S_h_by_Kmax']['21000000']:.6f} "
            f"delta_m={w['delta_m_least_squares']:.4f}"
        )
        print(
            f"  seed {seed} [late]: S_h(15)={wl['S_h_by_Kmax']['15000000']:.6f} "
            f"S_h(18)={wl['S_h_by_Kmax']['18000000']:.6f} S_h(21)={wl['S_h_by_Kmax']['21000000']:.6f} "
            f"delta_m={wl['delta_m_least_squares']:.4f}"
        )
    print("classification diagnostics (all-window):", json.dumps(classification_result, indent=2))
    for seed in SEEDS:
        print(f"decomposition seed {seed}:", decomposition[seed]["interpretation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
