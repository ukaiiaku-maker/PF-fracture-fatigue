"""Focused verifier for the PX5 transient-regime analysis (paper-evidence
provenance closure mission, Section 7). Independently recomputes, from
the tracked raw event ledgers ONLY (never trusting px5_transient_regime_
analysis.json/csv at face value), every quantity that file reports:

  - event-0 equality (static-shield controls bit-identical to zero);
  - the first 10 post-event finite/zero and static/zero ratios;
  - the rolling 5-event ratios;
  - the approach-to-developed-regime event index;
  - the D2-vs-D5 comparison (both protocols must agree to within
    floating-point precision at every Kmax, since PX5 already
    established D2 and D5 are physically near-identical).

Requires EXACT agreement with the committed px5_transient_regime_
analysis.{json,csv}. Never touches a gitignored runs/ directory.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from build_px5_transient_regime_analysis import (  # noqa: E402
    _events_by_key, _waiting_times, _rolling_ratio, _approach_to_developed,
    KMAX_GRID_MPa, ROLLING_WINDOW, DEVELOPED_WINDOW_EVENTS, STABILITY_TOL,
)


def main() -> int:
    checks: dict[str, bool] = {}

    committed = json.loads((ARTIFACTS_DIR / "px5_transient_regime_analysis.json").read_text())
    committed_by_key = {(p["protocol"], p["Kmax_MPa_sqrt_m"]): p for p in committed["points"]}

    dev_events = _events_by_key(ARTIFACTS_DIR / "developed_event_ledger.json")
    px5_events = _events_by_key(ARTIFACTS_DIR / "px5_event_ledger.json")

    dev_registry = list(csv.DictReader(open(ARTIFACTS_DIR / "developed_job_registry.csv")))
    dev_authorized = [r for r in dev_registry if r["status"] == "AUTHORIZED_PX4" and r["seed"] == "1720"]
    dev_key_by = {(r["protocol"], round(float(r["Kmax_Pa_sqrt_m"])), r["cohesion"]): r["canonical_job_key"] for r in dev_authorized}

    px5_main = list(csv.DictReader(open(ARTIFACTS_DIR / "px5_static_shield_job_registry.csv")))
    px5_retry = list(csv.DictReader(open(ARTIFACTS_DIR / "px5_static_shield_wall_budget_retry_registry.csv")))
    px5_admitted = [r for r in px5_main if r["status"] == "AUTHORIZED_PX5"] + [r for r in px5_retry if r["status"] == "AUTHORIZED_PX5"]
    px5_key_by = {(r["protocol"], round(float(r["Kmax_Pa_sqrt_m"])), r["cohesion"]): r["canonical_job_key"] for r in px5_admitted}

    d2_d5_first_event = {}
    n_mismatch = 0
    for protocol in ("D2", "D5"):
        for Kmax_MPa in KMAX_GRID_MPa:
            Kmax = round(Kmax_MPa * 1.0e6)
            zero_key = dev_key_by[(protocol, Kmax, "zero")]
            dynamic_key = dev_key_by[(protocol, Kmax, "finite")]
            ceiling_key = px5_key_by[(protocol, Kmax, "ceiling_static")]
            orbit_key = px5_key_by[(protocol, Kmax, "orbit_matched_static")]

            w_zero = _waiting_times(dev_events[zero_key])
            w_dynamic = _waiting_times(dev_events[dynamic_key])
            w_ceiling = _waiting_times(px5_events[ceiling_key])
            w_orbit = _waiting_times(px5_events[orbit_key])

            # Event-0 equality
            ceiling_eq = (w_ceiling[0] / w_zero[0]) == 1.0
            orbit_eq = (w_orbit[0] / w_zero[0]) == 1.0
            checks[f"event0_bit_identical_{protocol}_{Kmax_MPa}"] = ceiling_eq and orbit_eq
            d2_d5_first_event[(protocol, Kmax_MPa)] = w_dynamic[0] / w_zero[0]

            # Rolling ratios + approach index, cross-checked against committed file
            rolling_dynamic = _rolling_ratio(w_dynamic, w_zero, ROLLING_WINDOW)
            rolling_ceiling = _rolling_ratio(w_ceiling, w_zero, ROLLING_WINDOW)
            rolling_orbit = _rolling_ratio(w_orbit, w_zero, ROLLING_WINDOW)
            approach = {
                "dynamic_over_zero": _approach_to_developed(rolling_dynamic, STABILITY_TOL, DEVELOPED_WINDOW_EVENTS),
                "ceiling_over_zero": _approach_to_developed(rolling_ceiling, STABILITY_TOL, DEVELOPED_WINDOW_EVENTS),
                "orbit_matched_over_zero": _approach_to_developed(rolling_orbit, STABILITY_TOL, DEVELOPED_WINDOW_EVENTS),
            }

            committed_point = committed_by_key[(protocol, Kmax_MPa)]
            rolling_match = (
                rolling_dynamic == committed_point["rolling_5event_ratio"]["dynamic_over_zero"]
                and rolling_ceiling == committed_point["rolling_5event_ratio"]["ceiling_over_zero"]
                and rolling_orbit == committed_point["rolling_5event_ratio"]["orbit_matched_over_zero"]
            )
            approach_match = approach == committed_point["approach_to_developed_regime_event_index"]
            checks[f"rolling_ratio_exact_match_{protocol}_{Kmax_MPa}"] = rolling_match
            checks[f"approach_index_exact_match_{protocol}_{Kmax_MPa}"] = approach_match
            if not (rolling_match and approach_match):
                n_mismatch += 1

            # First 10 post-event ratios cross-check
            post_first = {
                "dynamic_over_zero": [w_dynamic[i] / w_zero[i] for i in range(1, min(11, len(w_zero)))],
                "ceiling_over_zero": [w_ceiling[i] / w_zero[i] for i in range(1, min(11, len(w_zero)))],
                "orbit_matched_over_zero": [w_orbit[i] / w_zero[i] for i in range(1, min(11, len(w_zero)))],
            }
            checks[f"post_first_event_ratios_exact_match_{protocol}_{Kmax_MPa}"] = (
                post_first == committed_point["post_first_event_eventwise_ratios"]
            )

    checks["zero_rolling_or_approach_mismatches"] = n_mismatch == 0

    # D2-vs-D5 agreement: both protocols must produce IDENTICAL first-event
    # deviation-from-1 at every shared Kmax (established finding: this
    # deviation is a floating-point integration artifact tied to Kmax
    # alone, not a real cohesion-dependent difference between D2 and D5).
    d2_d5_agree = all(
        d2_d5_first_event[("D2", K)] == d2_d5_first_event[("D5", K)] for K in KMAX_GRID_MPa
    )
    checks["D2_D5_first_event_ratio_identical_at_every_Kmax"] = d2_d5_agree

    overall_pass = all(checks.values())
    verification = {
        "schema": "v1_px5_transient_regime_verification",
        "overall_pass": overall_pass, "checks": checks,
    }
    out_path = ARTIFACTS_DIR / "px5_transient_regime_verification.json"
    out_path.write_text(json.dumps(verification, indent=2, sort_keys=True, default=str) + "\n")
    print(f"wrote {out_path}")
    print(f"overall_pass={overall_pass}")
    for k, v in checks.items():
        if not v:
            print(f"  FAILED: {k}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
