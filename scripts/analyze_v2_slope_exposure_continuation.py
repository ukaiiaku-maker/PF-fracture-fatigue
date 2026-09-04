"""Final analysis after the exposure-unconditioned completion trajectory:
the full 15/18/21 three-point S_h(K) slope fit for BOTH seeds, adjacent
secants, and the revised classification (adds
REBONDING_PHASE_EXPOSURE_SENSITIVE).

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


def _rate_shift(zero_res: dict, finite_res: dict) -> dict:
    zero_lengths = [e["accepted_length_m"] for e in zero_res["events"]]
    finite_lengths = [e["accepted_length_m"] for e in finite_res["events"]]
    lengths_identical = (
        len(zero_lengths) == len(finite_lengths)
        and all(abs(a - b) < 1.0e-15 for a, b in zip(zero_lengths, finite_lengths))
    )
    delta_a_zero, delta_a_finite = sum(zero_lengths), sum(finite_lengths)
    t_zero, t_finite = zero_res["cumulative_time_s"], finite_res["cumulative_time_s"]
    g_zero, g_finite = delta_a_zero / t_zero, delta_a_finite / t_finite
    return {
        "accepted_lengths_identical": lengths_identical,
        "S_h_decade": math.log10(g_finite / g_zero),
        "g_zero_m_per_s": g_zero, "g_finite_m_per_s": g_finite,
    }


def _exposure(res: dict) -> dict:
    intervals = res["post_first_event_intervals"]
    return {
        "n_intervals_complete_excursion": sum(1 for iv in intervals if iv["complete_negative_excursion"]),
        "n_intervals_partial_excursion": sum(
            1 for iv in intervals
            if not iv["complete_negative_excursion"] and iv["negative_contact_duration_s"] > 0.0
        ),
        "total_negative_K_contact_time_s": sum(iv["negative_contact_duration_s"] for iv in intervals),
    }


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

    S_h_by_Kmax: dict[float, dict[int, float]] = {K: {} for K in KMAX_GRID_Pa_sqrt_m}
    rate_shifts: dict[str, dict] = {}
    exposure_records: dict[str, dict] = {}
    for (Kmax, seed), (zero_res, finite_res) in pairs.items():
        rs = _rate_shift(zero_res, finite_res)
        key = f"K{Kmax/1e6:.0f}MPa_seed{seed}"
        rate_shifts[key] = rs
        S_h_by_Kmax[Kmax][seed] = rs["S_h_decade"]
        exposure_records[key] = {
            "zero_cohesion": _exposure(zero_res), "finite_cohesion": _exposure(finite_res),
        }

    per_seed: dict[int, dict] = {}
    for seed in SEEDS:
        x = [math.log10(K) for K in KMAX_GRID_Pa_sqrt_m]
        y = [S_h_by_Kmax[K][seed] for K in KMAX_GRID_Pa_sqrt_m]
        fit = three_point_slope_fit(x, y)
        secant_15_18 = adjacent_secant(math.log10(15.0e6), S_h_by_Kmax[15.0e6][seed],
                                        math.log10(18.0e6), S_h_by_Kmax[18.0e6][seed])
        secant_18_21 = adjacent_secant(math.log10(18.0e6), S_h_by_Kmax[18.0e6][seed],
                                        math.log10(21.0e6), S_h_by_Kmax[21.0e6][seed])
        per_seed[seed] = {
            "S_h_by_Kmax": {str(int(K)): S_h_by_Kmax[K][seed] for K in KMAX_GRID_Pa_sqrt_m},
            "delta_m_least_squares": fit["slope"],
            "secant_15_to_18": secant_15_18,
            "secant_18_to_21": secant_18_21,
        }

    delta_m_by_seed = {seed: per_seed[seed]["delta_m_least_squares"] for seed in SEEDS}
    exposure_at_Kmax_hi = {
        seed: exposure_records[f"K21MPa_seed{seed}"]["finite_cohesion"] for seed in SEEDS
    }
    classification_result = classify_slope_effect_v2(
        S_h_by_Kmax=S_h_by_Kmax, delta_m_by_seed=delta_m_by_seed,
        exposure_by_seed_at_Kmax=exposure_at_Kmax_hi,
        rate_offset_span_decade=RATE_OFFSET_SPAN_DECADE, slope_gate=SLOPE_GATE,
    )

    decision = {
        "schema": "v10.2.30_crack_rebonding_slope_exposure_continuation_decision_v1",
        "status": "COMPLETE",
        "kmax_grid_Pa_sqrt_m": list(KMAX_GRID_Pa_sqrt_m),
        "rate_shifts": rate_shifts,
        "exposure_records": exposure_records,
        "per_seed_slope_fit": {str(seed): per_seed[seed] for seed in SEEDS},
        "classification_analysis": classification_result,
        "overall_classification": classification_result["classification"],
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
    print(f"overall_classification={classification_result['classification']}")
    for seed in SEEDS:
        p = per_seed[seed]
        print(
            f"  seed {seed}: S_h(15)={p['S_h_by_Kmax']['15000000']:.6f} "
            f"S_h(18)={p['S_h_by_Kmax']['18000000']:.6f} "
            f"S_h(21)={p['S_h_by_Kmax']['21000000']:.6f} "
            f"delta_m={p['delta_m_least_squares']:.4f} "
            f"secant(15->18)={p['secant_15_to_18']:.4f} "
            f"secant(18->21)={p['secant_18_to_21']:.4f}"
        )
    print("classification diagnostics:", json.dumps(classification_result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
