"""Analysis-only partial calculation (no new physics): freezes and reports
the signed rate shift S_h and the requested archived diagnostics for the
five (Kmax, seed) pairs that already have both a zero- and finite-cohesion
trajectory, BEFORE launching the one missing trajectory
(Kmax=21, seed=1001723, finite cohesion).

Available pairs (all reversible regime):
    S_h(15, 1720), S_h(15, 1001723)   -- from this branch's slope screen
    S_h(18, 1720), S_h(18, 1001723)   -- reused verbatim from the parent
                                          causal pilot's tracked ledgers
    (21, 1720) has both zero+finite  -- S_h(21, 1720) computable
    (21, 1001723) has zero ONLY      -- reported as PENDING, no S_h yet

Uses ONLY tracked artifacts already committed to git (the parent pilot's
event_ledger.json/second_seed_event_ledger.json, and this slope-screen
branch's own event_ledger.json) -- no new trajectory is run.

Usage:
    <pinned interpreter> scripts/analyze_v2_slope_exposure_partial.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_PILOT_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_causal_pilot_v2"
SLOPE_SCREEN_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_minimal_slope_screen_v1"
OUT_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_slope_exposure_continuation"


def _rate_shift(zero_res: dict, finite_res: dict) -> dict[str, Any]:
    zero_lengths = [e["accepted_length_m"] for e in zero_res["events"]]
    finite_lengths = [e["accepted_length_m"] for e in finite_res["events"]]
    lengths_identical = (
        len(zero_lengths) == len(finite_lengths)
        and all(abs(a - b) < 1.0e-15 for a, b in zip(zero_lengths, finite_lengths))
    )
    delta_a_zero = sum(zero_lengths)
    delta_a_finite = sum(finite_lengths)
    t_zero = zero_res["cumulative_time_s"]
    t_finite = finite_res["cumulative_time_s"]
    g_zero = delta_a_zero / t_zero
    g_finite = delta_a_finite / t_finite
    S_h = math.log10(g_finite / g_zero)
    return {
        "accepted_lengths_identical": lengths_identical,
        "delta_a_zero_m": delta_a_zero, "delta_a_finite_m": delta_a_finite,
        "cumulative_time_zero_s": t_zero, "cumulative_time_finite_s": t_finite,
        "g_zero_m_per_s": g_zero, "g_finite_m_per_s": g_finite,
        "S_h_decade": S_h,
    }


def _exposure_diagnostics(res: dict) -> dict[str, Any]:
    intervals = res["post_first_event_intervals"]
    total_negative_contact_s = sum(iv["negative_contact_duration_s"] for iv in intervals)
    n_complete = sum(1 for iv in intervals if iv["complete_negative_excursion"])
    n_partial = sum(
        1 for iv in intervals
        if not iv["complete_negative_excursion"] and iv["negative_contact_duration_s"] > 0.0
    )
    n_none = sum(
        1 for iv in intervals
        if not iv["complete_negative_excursion"] and iv["negative_contact_duration_s"] == 0.0
    )
    events = res["events"]
    aw_K_rebond = [
        e["action_weighted_K_rebond_Pa_sqrt_m"] for e in events
        if e.get("action_weighted_K_rebond_Pa_sqrt_m") is not None
        and e["action_weighted_K_rebond_Pa_sqrt_m"] > 0.0
    ]
    return {
        "n_accepted_events": res["n_accepted_events"],
        "censored": res["censored"],
        "total_negative_K_contact_time_s": total_negative_contact_s,
        "n_intervals_complete_excursion": n_complete,
        "n_intervals_partial_excursion": n_partial,
        "n_intervals_no_excursion": n_none,
        "n_intervals_total": len(intervals),
        "max_p_B_post_commit": max((e["max_pB_post_commit"] for e in events), default=0.0),
        "max_phase_resolved_K_rebond_Pa_sqrt_m": max(
            (e.get("max_phase_resolved_K_rebond_Pa_sqrt_m", 0.0) or 0.0 for e in events), default=0.0
        ),
        "max_action_weighted_K_rebond_Pa_sqrt_m": max(aw_K_rebond, default=0.0),
        "mean_action_weighted_K_rebond_Pa_sqrt_m_nonzero_events": (
            sum(aw_K_rebond) / len(aw_K_rebond) if aw_K_rebond else 0.0
        ),
        "cumulative_extension_m": res["cumulative_extension_m"],
        "cumulative_time_s": res["cumulative_time_s"],
    }


def _frozen_A_on_off(Kmax_Pa_sqrt_m: float) -> dict[str, float]:
    frozen_predictions = json.loads((SLOPE_SCREEN_ARTIFACTS / "frozen_predictions.json").read_text())
    for pred in frozen_predictions["predictions"]["reversible"]:
        if abs(pred["Kmax_Pa_sqrt_m"] - Kmax_Pa_sqrt_m) < 1.0:
            return {"A_on": pred["A_on"], "A_off": pred["A_off"], "Pi_K_at_this_load": pred["Pi_K_at_this_load"]}
    raise KeyError(Kmax_Pa_sqrt_m)


def main() -> int:
    OUT_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    parent_ledger_1720 = json.loads((PARENT_PILOT_ARTIFACTS / "event_ledger.json").read_text())
    parent_ledger_1001723 = json.loads(
        (PARENT_PILOT_ARTIFACTS / "second_seed_event_ledger.json").read_text()
    )
    screen_ledger = json.loads((SLOPE_SCREEN_ARTIFACTS / "event_ledger.json").read_text())

    pairs = {
        (15.0e6, 1720): (
            screen_ledger["trajectories"]["C2R_K15MPa_seed1720"],
            screen_ledger["trajectories"]["C3R_K15MPa_seed1720"],
        ),
        (15.0e6, 1001723): (
            screen_ledger["trajectories"]["C2R_K15MPa_seed1001723"],
            screen_ledger["trajectories"]["C3R_K15MPa_seed1001723"],
        ),
        (18.0e6, 1720): (
            parent_ledger_1720["trajectories"]["C2R"], parent_ledger_1720["trajectories"]["C3R"],
        ),
        (18.0e6, 1001723): (
            parent_ledger_1001723["trajectories"]["S2_C2R"],
            parent_ledger_1001723["trajectories"]["S2_C3R"],
        ),
        (21.0e6, 1720): (
            screen_ledger["trajectories"]["C2R_K21MPa_seed1720"],
            screen_ledger["trajectories"]["C3R_K21MPa_seed1720"],
        ),
    }

    results: dict[str, Any] = {}
    for (Kmax, seed), (zero_res, finite_res) in pairs.items():
        key = f"K{Kmax/1e6:.0f}MPa_seed{seed}"
        rate_shift = _rate_shift(zero_res, finite_res)
        A_on_off = _frozen_A_on_off(Kmax)
        results[key] = {
            "Kmax_Pa_sqrt_m": Kmax, "seed": seed,
            "rate_shift": rate_shift,
            "A_on_frozen": A_on_off["A_on"], "A_off_frozen": A_on_off["A_off"],
            "Pi_K_at_this_load": A_on_off["Pi_K_at_this_load"],
            "zero_cohesion_diagnostics": _exposure_diagnostics(zero_res),
            "finite_cohesion_diagnostics": _exposure_diagnostics(finite_res),
        }

    # The pending pair: (21, 1001723) has zero-cohesion only so far.
    pending_zero = screen_ledger["trajectories"]["C2R_K21MPa_seed1001723"]
    results["K21MPa_seed1001723_PENDING"] = {
        "Kmax_Pa_sqrt_m": 21.0e6, "seed": 1001723,
        "status": "PENDING_FINITE_COHESION_TRAJECTORY_NOT_YET_RUN",
        "A_on_frozen": _frozen_A_on_off(21.0e6)["A_on"],
        "A_off_frozen": _frozen_A_on_off(21.0e6)["A_off"],
        "Pi_K_at_this_load": _frozen_A_on_off(21.0e6)["Pi_K_at_this_load"],
        "zero_cohesion_diagnostics": _exposure_diagnostics(pending_zero),
    }

    payload = {
        "schema": "v10.2.30_crack_rebonding_slope_exposure_continuation_partial_analysis_v1",
        "note": (
            "analysis-only, no new physics -- computed entirely from tracked "
            "artifacts already committed by the parent causal pilot and the "
            "minimal slope screen. S_h uses the actual crack-growth-rate ratio "
            "(sum of accepted lengths / cumulative time), reducing exactly to "
            "the waiting-time simplification only where accepted_lengths_"
            "identical is confirmed True (checked per pair, not assumed)."
        ),
        "results": results,
    }
    out_path = OUT_ARTIFACTS / "partial_analysis.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    for key in ("K15MPa_seed1720", "K15MPa_seed1001723", "K18MPa_seed1720", "K18MPa_seed1001723",
                "K21MPa_seed1720"):
        r = results[key]
        print(
            f"  {key}: S_h={r['rate_shift']['S_h_decade']:.6f} decade  "
            f"lengths_identical={r['rate_shift']['accepted_lengths_identical']}  "
            f"A_on={r['A_on_frozen']:.4f} A_off={r['A_off_frozen']:.4f}"
        )
    pend = results["K21MPa_seed1001723_PENDING"]
    print(
        f"  K21MPa_seed1001723: PENDING -- zero-cohesion only. "
        f"neg_K_contact_time={pend['zero_cohesion_diagnostics']['total_negative_K_contact_time_s']:.3e}s "
        f"complete={pend['zero_cohesion_diagnostics']['n_intervals_complete_excursion']} "
        f"partial={pend['zero_cohesion_diagnostics']['n_intervals_partial_excursion']} "
        f"none={pend['zero_cohesion_diagnostics']['n_intervals_no_excursion']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
