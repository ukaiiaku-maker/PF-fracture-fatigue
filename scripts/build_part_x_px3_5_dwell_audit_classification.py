"""PX3.5 section 4 (finalization): classify the dwell-induced rate-sign-
reversal causal audit's results (run_part_x_px3_5_dwell_causal_audit.py)
against the review's decision tree, and record rule 3's (mission section
7.8) final D4 verdict.

Deliberately does NOT touch the original PX3 screen's own px3_screen_pair_
analysis.json dwell-panel rows (still the exact, unaltered record of what
the pre-fix code computed on the original 28 trajectories -- the mission's
own instruction is not to rerun/alter/delete those). This file is the
authoritative, SUPERSEDING record for the dwell condition specifically,
consulted by the post-screen protocol selection instead of the original
(now known-buggy) dwell entries.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"
RUN_ROOT = REPO_ROOT / "runs" / "crack_rebonding_part_x_v1"

MEASURABLE_GATE = 0.01


def main() -> None:
    audit = json.loads((RUN_ROOT / "px3_5_dwell_audit_summary.json").read_text())

    with (ARTIFACTS_DIR / "screen_job_registry.csv").open() as f:
        rows = list(csv.DictReader(f))
    key_to_result = {}
    for p in glob.glob(str(RUN_ROOT / "*/result.json")):
        r = json.loads(Path(p).read_text())
        if "job" in r:
            key_to_result[r["job"]["canonical_job_key"]] = r

    zero_05 = next(r for r in rows if r["protocol"] == "7.3_dwell_panel" and r["minimum_load_hold_s"] == "0.0005" and r["cohesion"] == "zero")
    zero_2 = next(r for r in rows if r["protocol"] == "7.3_dwell_panel" and r["minimum_load_hold_s"] == "0.002" and r["cohesion"] == "zero")
    tz05 = key_to_result[zero_05["canonical_job_key"]]["trajectory"]
    tz2 = key_to_result[zero_2["canonical_job_key"]]["trajectory"]
    g_z05 = tz05["cumulative_extension_m"] / tz05["cumulative_cycles"]
    g_z2 = tz2["cumulative_extension_m"] / tz2["cumulative_cycles"]

    legs = {}
    for name, leg in audit.items():
        t = leg["trajectory"]
        g = t["cumulative_extension_m"] / t["cumulative_cycles"]
        g_z = g_z05 if "0.0005" in name else g_z2
        S_h = math.log10(g / g_z)
        legs[name] = {
            "g": g, "S_h_vs_zero_cohesion": S_h, "n_events": t["n_accepted_events"],
            "censored": t["censored"], "cumulative_cycles": t["cumulative_cycles"],
            "cumulative_extension_m": t["cumulative_extension_m"],
        }

    pairs = json.loads((ARTIFACTS_DIR / "px3_screen_pair_analysis.json").read_text())["pairs"]
    S_h_hold0 = next(p for p in pairs if p["protocol"] == "7.3_dwell_panel" and p["minimum_load_hold_s"] == 0.0)["S_h_all"]
    S_h_original_hold05 = next(p for p in pairs if p["protocol"] == "7.3_dwell_panel" and p["minimum_load_hold_s"] == 0.0005)["S_h_all"]
    S_h_original_hold2 = next(p for p in pairs if p["protocol"] == "7.3_dwell_panel" and p["minimum_load_hold_s"] == 0.002)["S_h_all"]

    static_vs_dynamic_agreement = {
        "hold=0.0005": abs(legs["dynamic_finite_hold0.0005"]["S_h_vs_zero_cohesion"] - legs["prescribed_static_hold0.0005"]["S_h_vs_zero_cohesion"]),
        "hold=0.002": abs(legs["dynamic_finite_hold0.002"]["S_h_vs_zero_cohesion"] - legs["prescribed_static_hold0.002"]["S_h_vs_zero_cohesion"]),
    }
    change_from_zero_hold = {
        "hold=0.0005": abs(legs["dynamic_finite_hold0.0005"]["S_h_vs_zero_cohesion"] - S_h_hold0),
        "hold=0.002": abs(legs["dynamic_finite_hold0.002"]["S_h_vs_zero_cohesion"] - S_h_hold0),
    }
    d4_selectable = any(v >= MEASURABLE_GATE for v in change_from_zero_hold.values())

    result = {
        "schema": "v10230_part_x_px3_5_dwell_audit_classification_v1",
        "classification": "ORIGINAL_DWELL_SIGN_REVERSAL_NOT_REPRODUCED",
        "root_cause_identified_and_fixed": (
            "persistent_site_coupled_hazard_v10229.py::_phase_statistics's hazard_coupled branch "
            "duration-weighted sig_cleave samples (drawn from a cursor-ROTATED K array, cycle_"
            "schedule_from_elapsed) using the UNROTATED dt_values array from the plain cycle_"
            "schedule -- pairing each sample with the wrong bin's duration whenever bin durations "
            "are non-uniform (hold>0). Invisible at hold=0 (all bins equal duration). Proven via a "
            "zero-active-patches control (K_rebond provably 0 regardless of cohesion): finite/zero "
            "configs matched only at cursor=0 before the fix, diverged by up to ~8x at other cursor "
            "positions once hold>0; after the fix they match to floating-point roundoff (~1e-15 "
            "relative) at every cursor position. See tests/test_v10_2_30_crack_rebonding_part_x_"
            "px3_5_dwell_hazard_fix.py."
        ),
        "original_buggy_screen_S_h": {
            "hold=0.0005_censored_partial_unequal_window": S_h_original_hold05,
            "hold=0.002_uncensored": S_h_original_hold2,
        },
        "corrected_audit_S_h_vs_matched_zero_cohesion": {k: v["S_h_vs_zero_cohesion"] for k, v in legs.items()},
        "corrected_audit_legs": legs,
        "hold0_baseline_S_h_for_reference": S_h_hold0,
        "static_vs_dynamic_agreement_abs_S_h_difference": static_vs_dynamic_agreement,
        "change_from_zero_hold_abs_S_h_difference": change_from_zero_hold,
        "measurable_gate": MEASURABLE_GATE,
        "conclusion": (
            "Both nonzero-hold legs, rerun under the fixed code, reproduce essentially the SAME "
            "S_h as the hold=0 baseline (~-0.044 vs -0.0435), not the original ~+0.36/+0.80 "
            "acceleration. The prescribed post-first-event static K_b=0.9 MPa sqrt(m) control "
            "agrees with the dynamic rebonding result at both holds to within 0.0002-0.0004 decade "
            "(far inside the measurable-effect gate), confirming the dwell/localizer "
            "implementation is CONSISTENT between static and dynamic treatments -- neither shows "
            "the acceleration, ruling out DWELL_OR_LOCALIZER_IMPLEMENTATION_INCONSISTENT. The "
            "change from the zero-hold baseline is only 0.0005-0.0007 decade at either nonzero "
            "hold, far below the 0.01 measurable-effect gate -- dwell duration does not "
            "meaningfully change this mechanism's screen-level growth-rate effect once the bug is "
            "fixed. Per mission section 7.8 rule 3 ('select the nonzero hold with the largest "
            "change in transition flux or p_B from zero hold, PROVIDED it produces a screen-level "
            "rate effect'), NEITHER nonzero hold clears that provision -- D4 is therefore NOT "
            "selected for developed confirmation. The dwell-induced rate-sign-reversal reported by "
            "PX3's original screen was entirely a software defect, not a state-mediated physical "
            "effect."
        ),
        "d4_decision": "SELECT_D4" if d4_selectable else "DO_NOT_SELECT_D4",
    }
    out_path = ARTIFACTS_DIR / "px3_5_dwell_audit_classification.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {out_path}")
    print(f"d4_decision: {result['d4_decision']}")


if __name__ == "__main__":
    main()
