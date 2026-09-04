"""Compute the local Paris-slope correction from the minimal slope screen.

For each seed, computes S_h(Kmax) = log10(g_finite/g_zero) at Kmax in
{15, 18, 21} MPa*sqrt(m) (Kmax=18 reused verbatim from the parent branch's
tracked ledgers; 15/21 from this screen's own new trajectories), using the
ACTUAL crack-growth-rate ratio g = (sum of accepted event lengths) /
(cumulative elapsed time) -- not the waiting-time simplification, unless
accepted event lengths are confirmed identical between the zero/finite
twin at that load (checked, not assumed, at every point).

Fits the 3-point slope Delta m = d S_h / d log10(Kmax) by ordinary least
squares, and reports the two adjacent secants (15->18, 18->21) to separate
a smooth slope shift from curvature. Classifies per the prospectively
declared gates:

    REBONDING_RATE_OFFSET_LIKE          if |S_h(21)-S_h(15)| < 0.01 decade
                                            and |Delta m| < 0.25 for BOTH seeds
    REBONDING_STEEPENS_LOCAL_RESPONSE   if Delta m >= 0.25 for BOTH seeds
    REBONDING_FLATTENS_LOCAL_RESPONSE   if Delta m <= -0.25 for BOTH seeds
    REBONDING_SLOPE_EFFECT_WEAK_OR_UNRESOLVED  otherwise (including seed
                                            disagreement in sign)

Usage:
    <pinned interpreter> scripts/analyze_v2_minimal_slope_screen.py \\
        --run-root runs/crack_rebonding_minimal_slope_screen_v1
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

from arrhenius_fracture.crack_rebonding_minimal_slope_screen_v10230 import (  # noqa: E402
    adjacent_secant,
    classify_slope_effect,
    three_point_slope_fit,
)

PARENT_ARTIFACTS_DIR = REPO_ROOT / "artifacts/crack_rebonding_causal_pilot_v2"
OUT_ARTIFACTS_DIR = REPO_ROOT / "artifacts/crack_rebonding_minimal_slope_screen_v1"
KMAX_GRID_Pa_sqrt_m = (15.0e6, 18.0e6, 21.0e6)
SEEDS = (1720, 1001723)

RATE_OFFSET_SPAN_DECADE = 0.01
SLOPE_GATE = 0.25


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
        "t_zero_s": t_zero, "t_finite_s": t_finite,
        "g_zero_m_per_s": g_zero, "g_finite_m_per_s": g_finite,
        "S_h_decade": S_h,
        "n_events_zero": zero_res["n_accepted_events"], "n_events_finite": finite_res["n_accepted_events"],
        "zero_censored": zero_res["censored"], "finite_censored": finite_res["censored"],
    }


def _reused_kmax18(seed: int) -> dict[str, dict]:
    if seed == 1720:
        ledger = json.loads((PARENT_ARTIFACTS_DIR / "event_ledger.json").read_text())
        return {"zero": ledger["trajectories"]["C2R"], "finite": ledger["trajectories"]["C3R"]}
    ledger = json.loads((PARENT_ARTIFACTS_DIR / "second_seed_event_ledger.json").read_text())
    return {"zero": ledger["trajectories"]["S2_C2R"], "finite": ledger["trajectories"]["S2_C3R"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args(argv)

    run_root = Path(args.run_root)
    trajectories: dict[str, dict] = json.loads((run_root / "trajectories.json").read_text())
    exposure_report = json.loads((run_root / "exposure_gate_report.json").read_text())
    if exposure_report["stopped_early_at"] is not None:
        report = {
            "schema": "v10.2.30_crack_rebonding_minimal_slope_screen_analysis_v1",
            "status": "STOPPED_EARLY_EXPOSURE_GATE_FAILED",
            "stopped_early_at": exposure_report["stopped_early_at"],
            "exposure_gate_report": exposure_report,
        }
        out_path = run_root / "slope_screen_decision.json"
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (OUT_ARTIFACTS_DIR / "slope_screen_decision.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(f"wrote {out_path}: STOPPED_EARLY_EXPOSURE_GATE_FAILED at {exposure_report['stopped_early_at']}")
        return 1

    per_seed: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        reused = _reused_kmax18(seed)
        rate_shift_18 = _rate_shift(reused["zero"], reused["finite"])

        Kmax15_label = "15MPa"
        Kmax21_label = "21MPa"
        rate_shift_15 = _rate_shift(
            trajectories[f"C2R_K{Kmax15_label}_seed{seed}"],
            trajectories[f"C3R_K{Kmax15_label}_seed{seed}"],
        )
        rate_shift_21 = _rate_shift(
            trajectories[f"C2R_K{Kmax21_label}_seed{seed}"],
            trajectories[f"C3R_K{Kmax21_label}_seed{seed}"],
        )

        S_h_by_Kmax = {15.0e6: rate_shift_15["S_h_decade"], 18.0e6: rate_shift_18["S_h_decade"],
                       21.0e6: rate_shift_21["S_h_decade"]}
        x = [math.log10(K) for K in KMAX_GRID_Pa_sqrt_m]
        y = [S_h_by_Kmax[K] for K in KMAX_GRID_Pa_sqrt_m]
        fit = three_point_slope_fit(x, y)
        secant_15_18 = adjacent_secant(
            math.log10(15.0e6), S_h_by_Kmax[15.0e6], math.log10(18.0e6), S_h_by_Kmax[18.0e6]
        )
        secant_18_21 = adjacent_secant(
            math.log10(18.0e6), S_h_by_Kmax[18.0e6], math.log10(21.0e6), S_h_by_Kmax[21.0e6]
        )
        classification = classify_slope_effect(
            S_h_by_Kmax, fit["slope"], rate_offset_span_decade=RATE_OFFSET_SPAN_DECADE,
            slope_gate=SLOPE_GATE,
        )

        per_seed[seed] = {
            "rate_shift_Kmax15": rate_shift_15,
            "rate_shift_Kmax18_reused": rate_shift_18,
            "rate_shift_Kmax21": rate_shift_21,
            "S_h_by_Kmax": {str(int(k)): v for k, v in S_h_by_Kmax.items()},
            "delta_m_least_squares": fit["slope"],
            "secant_15_to_18": secant_15_18,
            "secant_18_to_21": secant_18_21,
            "endpoint_span_S_h_decade": abs(S_h_by_Kmax[21.0e6] - S_h_by_Kmax[15.0e6]),
            "classification_this_seed": classification,
        }

    classifications = {seed: per_seed[seed]["classification_this_seed"] for seed in SEEDS}
    delta_ms = {seed: per_seed[seed]["delta_m_least_squares"] for seed in SEEDS}
    same_sign = (delta_ms[SEEDS[0]] > 0) == (delta_ms[SEEDS[1]] > 0)

    if len(set(classifications.values())) == 1 and same_sign:
        overall_classification = next(iter(classifications.values()))
    else:
        overall_classification = "REBONDING_SLOPE_EFFECT_WEAK_OR_UNRESOLVED"

    decision = {
        "schema": "v10.2.30_crack_rebonding_minimal_slope_screen_analysis_v1",
        "status": "COMPLETE",
        "kmax_grid_Pa_sqrt_m": list(KMAX_GRID_Pa_sqrt_m),
        "rate_offset_span_threshold_decade": RATE_OFFSET_SPAN_DECADE,
        "slope_gate": SLOPE_GATE,
        "per_seed": {str(seed): per_seed[seed] for seed in SEEDS},
        "per_seed_classification": {str(seed): classifications[seed] for seed in SEEDS},
        "seeds_agree_in_sign": same_sign,
        "overall_classification": overall_classification,
        "multi_K_paris_slope_campaign_authorized": False,
        "part_x_authorized": False,
        "production_line_merge_authorized": False,
    }

    out_path = run_root / "slope_screen_decision.json"
    out_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    (OUT_ARTIFACTS_DIR / "slope_screen_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {out_path}")
    print(f"overall_classification={overall_classification}")
    for seed in SEEDS:
        p = per_seed[seed]
        print(
            f"  seed {seed}: S_h(15)={p['S_h_by_Kmax']['15000000']:.6f} "
            f"S_h(18)={p['S_h_by_Kmax']['18000000']:.6f} "
            f"S_h(21)={p['S_h_by_Kmax']['21000000']:.6f} "
            f"delta_m={p['delta_m_least_squares']:.4f} "
            f"secant(15->18)={p['secant_15_to_18']:.4f} "
            f"secant(18->21)={p['secant_18_to_21']:.4f} "
            f"class={p['classification_this_seed']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
