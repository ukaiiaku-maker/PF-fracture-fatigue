"""Section D: static-shield attribution comparison.

Reads ONLY tracked ledgers -- artifacts/crack_rebonding_developed_
confirmation/event_ledger.json (for the existing zero-cohesion and
dynamic-rebonding seed=1720/1001723 trajectories, reused verbatim, never
rerun) and artifacts/crack_rebonding_static_shield_attribution/
event_ledger.json (for the new static-shield trajectories this study
adds) -- never a gitignored runs/ directory.

For each Kmax and each of the developed/true_final_half/true_final_six
windows:

    S_dynamic(K) = log10[(da/dN)_dynamic / (da/dN)_zero]
    S_static(K)  = log10[(da/dN)_static  / (da/dN)_zero]
    D_history(K) = S_dynamic(K) - S_static(K)

plus delta_m_dynamic, delta_m_static, delta_m_history, and both adjacent
secants for each. Applies the prospectively frozen terminal-classification
gates from static_shield_attribution_protocol.json.

Usage:
    <pinned interpreter> scripts/analyze_static_shield_attribution.py
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
    event_index_window_rate,
    stable_growth_gate,
    true_final_half_indices,
    true_final_six_indices,
)
from arrhenius_fracture.crack_rebonding_minimal_slope_screen_v10230 import (  # noqa: E402
    adjacent_secant,
    three_point_slope_fit,
)

DEVELOPED_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_developed_confirmation"
STATIC_ARTIFACTS = REPO_ROOT / "artifacts/crack_rebonding_static_shield_attribution"
KMAX_GRID_Pa_sqrt_m = (15.0e6, 18.0e6, 21.0e6)
FREQUENCY_HZ = 1000.0
WINDOWS = ("developed", "true_final_half", "true_final_six")

DOMINANT_D_HISTORY_MAX_DECADE = 0.01
DOMINANT_DELTA_M_MAX_DIFF = 0.10
MIXED_D_HISTORY_MAX_DECADE = 0.03
MIXED_DELTA_M_MAX_DIFF = 0.25


def _developed_traj_name(seed: int, Kmax_MPa: int, cohesion: str) -> str:
    stage = "D1" if seed == 1720 else "D2"
    return f"{stage}_K{Kmax_MPa}MPa_seed{seed}_{cohesion}"


def _static_traj_name(seed: int, Kmax_MPa: int) -> str:
    stage = "S1" if seed == 1720 else "S2"
    return f"{stage}_K{Kmax_MPa}MPa_seed{seed}_static"


def _window_rate(events: list[dict], window: str, frequency_Hz: float) -> dict[str, Any]:
    if window == "developed":
        gate = stable_growth_gate(events, frequency_Hz=frequency_Hz)
        return gate["developed_interval"]
    n = len(events)
    if window == "true_final_half":
        idx = true_final_half_indices(n)
    elif window == "true_final_six":
        idx = true_final_six_indices(n)
    else:
        raise ValueError(window)
    return event_index_window_rate(events, idx, frequency_Hz)


def _accepted_length_identity(a_win: dict, b_win: dict) -> bool:
    return (
        a_win["da_m"] > 0.0 and b_win["da_m"] > 0.0
        and abs(a_win["da_m"] - b_win["da_m"]) < 1.0e-12 * max(a_win["da_m"], b_win["da_m"])
    )


def _S(numerator_win: dict, denominator_win: dict) -> tuple[float | None, bool]:
    len_ident = _accepted_length_identity(numerator_win, denominator_win)
    num_rate = numerator_win["da_dN"]
    den_rate = denominator_win["da_dN"]
    if not num_rate or not den_rate or num_rate <= 0.0 or den_rate <= 0.0:
        return None, len_ident
    return math.log10(num_rate / den_rate), len_ident


def analyze_seed(developed_ledger: dict, static_ledger: dict, seed: int) -> dict[str, Any] | None:
    dev_trajs = developed_ledger["trajectories"]
    static_trajs = static_ledger["trajectories"]

    per_Kmax: dict[str, Any] = {}
    S_dynamic_by_window_by_Kmax: dict[str, dict[float, float | None]] = {w: {} for w in WINDOWS}
    S_static_by_window_by_Kmax: dict[str, dict[float, float | None]] = {w: {} for w in WINDOWS}
    D_history_by_window_by_Kmax: dict[str, dict[float, float | None]] = {w: {} for w in WINDOWS}
    all_present = True
    all_uncensored_and_gate_pass = True

    for Kmax in KMAX_GRID_Pa_sqrt_m:
        Kmax_MPa = int(round(Kmax / 1.0e6))
        name_zero = _developed_traj_name(seed, Kmax_MPa, "zero")
        name_dynamic = _developed_traj_name(seed, Kmax_MPa, "finite")
        name_static = _static_traj_name(seed, Kmax_MPa)
        if name_zero not in dev_trajs or name_dynamic not in dev_trajs or name_static not in static_trajs:
            all_present = False
            continue

        zero_traj = dev_trajs[name_zero]
        dynamic_traj = dev_trajs[name_dynamic]
        static_traj = static_trajs[name_static]

        gate_zero = stable_growth_gate(zero_traj["events"], frequency_Hz=FREQUENCY_HZ)
        gate_dynamic = stable_growth_gate(dynamic_traj["events"], frequency_Hz=FREQUENCY_HZ)
        gate_static = stable_growth_gate(static_traj["events"], frequency_Hz=FREQUENCY_HZ)
        pair_ok = (
            zero_traj["uncensored"] and dynamic_traj["uncensored"] and static_traj["uncensored"]
            and gate_zero["stable_growth_provisional"] and gate_dynamic["stable_growth_provisional"]
            and gate_static["stable_growth_provisional"]
        )
        all_uncensored_and_gate_pass = all_uncensored_and_gate_pass and pair_ok

        # K_b step-function verification (Section C hard requirement).
        k_b_seq = [e["K_b_applied_Pa_sqrt_m"] for e in static_traj["events"]]
        k_b_step_function_valid = (
            len(k_b_seq) >= 1 and k_b_seq[0] == 0.0
            and all(v == 900000.0 for v in k_b_seq[1:])
        )

        # hazard-threshold-stream identity vs the existing zero-cohesion trajectory.
        hz_zero = [e["hazard_threshold_action"] for e in zero_traj["events"]]
        hz_static = [e["hazard_threshold_action"] for e in static_traj["events"]]
        threshold_stream_identical = (
            len(hz_zero) == len(hz_static) and all(a == b for a, b in zip(hz_zero, hz_static))
        )

        windows_out: dict[str, Any] = {}
        for window in WINDOWS:
            win_zero = _window_rate(zero_traj["events"], window, FREQUENCY_HZ)
            win_dynamic = _window_rate(dynamic_traj["events"], window, FREQUENCY_HZ)
            win_static = _window_rate(static_traj["events"], window, FREQUENCY_HZ)

            S_dynamic, len_ident_dyn = _S(win_dynamic, win_zero)
            S_static, len_ident_stat = _S(win_static, win_zero)
            D_history = (S_dynamic - S_static) if (S_dynamic is not None and S_static is not None) else None

            S_dynamic_by_window_by_Kmax[window][Kmax] = S_dynamic
            S_static_by_window_by_Kmax[window][Kmax] = S_static
            D_history_by_window_by_Kmax[window][Kmax] = D_history

            windows_out[window] = {
                "zero": win_zero, "dynamic": win_dynamic, "static": win_static,
                "S_dynamic_decade": S_dynamic, "S_static_decade": S_static,
                "D_history_decade": D_history,
                "accepted_length_identity_dynamic_vs_zero": len_ident_dyn,
                "accepted_length_identity_static_vs_zero": len_ident_stat,
                "waiting_time_simplification_admissible": len_ident_dyn and len_ident_stat,
            }

        per_Kmax[f"K{Kmax_MPa}MPa_seed{seed}"] = {
            "Kmax_Pa_sqrt_m": Kmax,
            "pair_ok": pair_ok,
            "k_b_step_function_valid": k_b_step_function_valid,
            "threshold_stream_identical_static_vs_zero": threshold_stream_identical,
            "windows": windows_out,
        }

    if not all_present:
        return None

    x = [math.log10(K) for K in KMAX_GRID_Pa_sqrt_m]

    def _fit(y_by_K: dict[float, float | None]) -> dict[str, Any] | None:
        if any(y_by_K.get(K) is None for K in KMAX_GRID_Pa_sqrt_m):
            return None
        y = [y_by_K[K] for K in KMAX_GRID_Pa_sqrt_m]
        fit = three_point_slope_fit(x, y)
        return {
            "by_Kmax": {str(int(K)): y_by_K[K] for K in KMAX_GRID_Pa_sqrt_m},
            "delta_m_least_squares": fit["slope"],
            "secant_15_to_18": adjacent_secant(x[0], y[0], x[1], y[1]),
            "secant_18_to_21": adjacent_secant(x[1], y[1], x[2], y[2]),
        }

    fits: dict[str, Any] = {}
    for window in WINDOWS:
        dyn_fit = _fit(S_dynamic_by_window_by_Kmax[window])
        stat_fit = _fit(S_static_by_window_by_Kmax[window])
        delta_m_history = (
            dyn_fit["delta_m_least_squares"] - stat_fit["delta_m_least_squares"]
            if dyn_fit is not None and stat_fit is not None else None
        )
        fits[window] = {
            "S_dynamic_fit": dyn_fit, "S_static_fit": stat_fit,
            "delta_m_history": delta_m_history,
            "D_history_by_Kmax": {str(int(K)): D_history_by_window_by_Kmax[window][K] for K in KMAX_GRID_Pa_sqrt_m},
            "max_abs_D_history_decade": max(
                abs(D_history_by_window_by_Kmax[window][K]) for K in KMAX_GRID_Pa_sqrt_m
            ),
        }

    return {
        "seed": seed,
        "all_present": all_present,
        "all_uncensored_and_gate_pass": all_uncensored_and_gate_pass,
        "all_k_b_step_functions_valid": all(v["k_b_step_function_valid"] for v in per_Kmax.values()),
        "all_threshold_streams_identical": all(
            v["threshold_stream_identical_static_vs_zero"] for v in per_Kmax.values()
        ),
        "per_Kmax": per_Kmax,
        "fits_by_window": fits,
    }


def classify(seed_analysis: dict) -> dict[str, Any]:
    checks_dominant = []
    checks_mixed = []
    for window in ("developed", "true_final_half"):
        fit = seed_analysis["fits_by_window"][window]
        max_d = fit["max_abs_D_history_decade"]
        delta_m_diff = abs(fit["delta_m_history"]) if fit["delta_m_history"] is not None else float("inf")
        checks_dominant.append(max_d <= DOMINANT_D_HISTORY_MAX_DECADE and delta_m_diff <= DOMINANT_DELTA_M_MAX_DIFF)
        checks_mixed.append(max_d <= MIXED_D_HISTORY_MAX_DECADE or delta_m_diff <= MIXED_DELTA_M_MAX_DIFF)

    if all(checks_dominant):
        classification = "FIXED_ABSOLUTE_COHESIVE_SHIELDING_DOMINANT"
    elif all(checks_mixed):
        classification = "MIXED_STATIC_SHIELDING_AND_KINETIC_HISTORY"
    else:
        classification = "DYNAMIC_REBONDING_HISTORY_REQUIRED"

    return {
        "classification": classification,
        "dominant_gate_passed_developed_and_true_final_half": checks_dominant,
        "mixed_gate_passed_developed_and_true_final_half": checks_mixed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)

    developed_ledger = json.loads((DEVELOPED_ARTIFACTS / "event_ledger.json").read_text())
    static_ledger = json.loads((STATIC_ARTIFACTS / "event_ledger.json").read_text())

    seed1720 = analyze_seed(developed_ledger, static_ledger, 1720)
    seed1001723 = analyze_seed(developed_ledger, static_ledger, 1001723)

    stage1_classification = classify(seed1720) if seed1720 is not None else None
    stage2_gate_authorized = (
        stage1_classification is not None
        and stage1_classification["classification"] != "FIXED_ABSOLUTE_COHESIVE_SHIELDING_DOMINANT"
    )

    terminal_classification = stage1_classification
    if seed1001723 is not None:
        stage2_classification = classify(seed1001723)
        both_present = seed1720 is not None
        agree = (
            both_present
            and stage1_classification["classification"] == stage2_classification["classification"]
        )
        terminal_classification = {
            "stage1_classification": stage1_classification["classification"] if stage1_classification else None,
            "stage2_classification": stage2_classification["classification"],
            "agree": agree,
            "classification": (
                stage2_classification["classification"] if agree
                else "STATIC_SHIELD_ATTRIBUTION_SEED_SENSITIVE"
            ),
        }

    parity_audit_path = STATIC_ARTIFACTS / "static_shield_localization_parity_audit.json"
    parity_audit = json.loads(parity_audit_path.read_text()) if parity_audit_path.is_file() else None
    residual_terminology = (
        "dynamic-history-plus-numerical-path residual"
        if parity_audit is not None and parity_audit["classification"] != "PRESCRIBED_STATIC_EVENT_LOCALIZATION_PARITY_QUALIFIED"
        else "dynamic-history residual"
    )

    scientific_wording = None
    if seed1720 is not None:
        max_d = seed1720["fits_by_window"]["developed"]["max_abs_D_history_decade"]
        scientific_wording = {
            "K_b_description": "fixed post-first-event cohesive shield (not an unqualified 'always-on' shield -- K_b=0 for the first accepted event, K_b=0.9 MPa sqrt(m) from the first accepted event onward)",
            "delta_m_description": (
                "delta_m is the three-point least-squares fitted slope INCREMENT that "
                "dynamic (or prescribed-static) rebonding induces in S_h(K) over "
                "Kmax=15-21 MPa sqrt(m) -- it is a rebonding-induced correction to the "
                "local Paris-slope response over this narrow load range, NOT the complete "
                "Paris exponent of the underlying da/dN(K) curve"
            ),
            "residual_terminology": residual_terminology,
            "residual_terminology_note": (
                "the localization-parity audit (scripts/audit_static_shield_localization_parity.py) "
                "did not reach PRESCRIBED_STATIC_EVENT_LOCALIZATION_PARITY_QUALIFIED -- a naive "
                "single-segment model of the static-shield engine's event-time localization showed "
                "up to ~32% relative discrepancy against an exact phase-resolved integrator (the real "
                "adaptive-quadrature implementation is expected to do much better, consistent with the "
                "tiny empirical residual actually observed, but this was not independently proven at "
                "the naive-model level) -- so the small residual below is reported as a combined "
                "dynamic-history-plus-numerical-path quantity, not attributed solely to genuine P/C/B "
                "kinetic history"
                if residual_terminology == "dynamic-history-plus-numerical-path residual"
                else "the localization-parity audit reached PRESCRIBED_STATIC_EVENT_LOCALIZATION_PARITY_QUALIFIED, "
                "supporting attribution of the residual to genuine kinetic history"
            ),
            "p_c_b_kinetics_claim": (
                "This study does NOT establish that P/C/B bond-formation/rupture kinetics are "
                "physically unnecessary. It shows only that, once the near-ceiling cohesive "
                "amplitude K_rebond_max is present, the incremental contribution of P/C/B temporal "
                f"modulation beyond that near-saturated shield amplitude is at most "
                f"{max_d:.5f} decade ({residual_terminology}) for the tested developed regime -- "
                "P/C/B kinetics may still be the mechanism that drives the wake rapidly into, and "
                "maintains it near, that saturated bonded state."
            ),
            "load_dependence_note": (
                "K_b/Kmax = 6.0%, 5.0%, 4.29% at Kmax=15/18/21 MPa sqrt(m) respectively -- a fixed "
                "absolute subtraction is proportionally larger at low Kmax, naturally producing "
                "stronger low-load suppression and a positive delta_m without requiring a strongly "
                "load-dependent bonded fraction"
            ),
        }

    decision = {
        "schema": "v10.2.30_crack_rebonding_static_shield_attribution_decision_v2",
        "kmax_grid_Pa_sqrt_m": list(KMAX_GRID_Pa_sqrt_m),
        "frequency_Hz": FREQUENCY_HZ,
        "K_b_static_Pa_sqrt_m": 900000.0,
        "seed_1720_analysis": seed1720,
        "seed_1001723_analysis": seed1001723,
        "stage1_classification": stage1_classification,
        "stage2_conditional_gate_authorized": stage2_gate_authorized,
        "terminal_classification": terminal_classification,
        "localization_parity_audit_reference": str(parity_audit_path),
        "localization_parity_classification": parity_audit["classification"] if parity_audit else None,
        "scientific_wording": scientific_wording,
        "multi_K_paris_slope_campaign_authorized": False,
        "part_x_authorized": False,
        "production_line_merge_authorized": False,
    }

    out_path = STATIC_ARTIFACTS / "static_shield_attribution_decision.json"
    out_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    if stage1_classification is not None:
        print(f"stage1_classification = {stage1_classification['classification']}")
        print(f"stage2_conditional_gate_authorized = {stage2_gate_authorized}")
        for window in WINDOWS:
            fit = seed1720["fits_by_window"][window]
            print(f"  [{window}] delta_m_dynamic={fit['S_dynamic_fit']['delta_m_least_squares']:.4f} "
                  f"delta_m_static={fit['S_static_fit']['delta_m_least_squares']:.4f} "
                  f"delta_m_history={fit['delta_m_history']:.4f} "
                  f"max|D_history|={fit['max_abs_D_history_decade']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
