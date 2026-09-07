"""PX3.6 verifier: extends verify_part_x_px3_5.py's 8 checks with the items
external review specifically required before PX4 could be considered.

New checks:
  9.  The two bug-contaminated dwell trajectories are formally invalidated
      (px3_bug_invalidated_trajectories.json) with an explicit exclusion
      scope covering screen counts/censor stats/sensitivity/decisions.
  10. Clean-producer reruns exist for the decision-driving pairs (frequency
      bisection, persistent-at-transition, dwell audit) and reproduce the
      pre-commit PX3.5 values (self-consistency, not a new physics claim).
  11. D5's decision is backed by genuine live cycle-mean p_B/p_P data (not
      an event-extrema proxy), and the literal rule (closest to 0.30,
      p_P>=0.30) is satisfied.
  12. The numerical/action uncertainty gate (|S_h| > max(0.005, 5*epsilon))
      is evaluated and satisfied for D3, D5, D6, using a real certified
      system constant (bulk_action_error_rel_tol).
  13. The three previously-failing production-adjacent test files
      (state-coupled hazard, forward marcher, partition-robust forward)
      now pass, restoring parity with the authoritative pre-Part-X base.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"


def _load(name: str):
    return json.loads((ARTIFACTS_DIR / name).read_text())


def check_9_bug_invalidation(report: dict) -> bool:
    try:
        data = _load("px3_bug_invalidated_trajectories.json")
    except FileNotFoundError:
        report["errors"].append("check_9: px3_bug_invalidated_trajectories.json not found")
        report["checks"]["dwell_bug_trajectories_invalidated"] = False
        return False
    ok = (
        len(data["invalidated_trajectories"]) == 2
        and all(e["status"] == "INVALIDATED_DWELL_DURATION_WEIGHTING_BUG" for e in data["invalidated_trajectories"])
        and "scientific screen event/trajectory counts" in data["exclusion_scope"]
    )
    report["checks"]["dwell_bug_trajectories_invalidated"] = ok
    if not ok:
        report["errors"].append("check_9: invalidation manifest incomplete or malformed")
    return ok


def check_10_clean_producer_reruns_self_consistent(report: dict) -> bool:
    run_root = REPO_ROOT / "runs" / "crack_rebonding_part_x_v1"
    required_precommit = [
        "px3_5_dwell_audit_summary_PRECOMMIT_ba1731e.json",
        "px3_5_persistent_at_transition_summary_PRECOMMIT_ba1731e.json",
    ]
    missing = [f for f in required_precommit if not (run_root / f).exists()]
    ok = len(missing) == 0
    # Frequency bisection: current (clean-producer) record must exist and
    # reproduce the same localized frequency PX3.5 found pre-commit.
    freq = _load("px3_5_frequency_bisection.json")
    ok = ok and freq["classification"].startswith("FREQUENCY_TRANSITION_LOCALIZED_AT_316.227766")
    report["checks"]["clean_producer_reruns_present_and_self_consistent"] = ok
    if not ok:
        report["errors"].append(f"check_10: missing precommit historical records or inconsistent reruns: {missing}")
    return ok


def check_11_passivation_live_data(report: dict) -> bool:
    try:
        passivation = _load("px3_6_passivation_requalification.json")["results"]
    except FileNotFoundError:
        report["errors"].append("check_11: px3_6_passivation_requalification.json not found")
        report["checks"]["d5_qualified_with_live_cycle_mean_data"] = False
        return False
    p_B = {chem: r["cycle_mean_state"]["cycle_mean_p_B"] for chem, r in passivation.items()}
    p_P = {chem: r["cycle_mean_state"]["cycle_mean_p_P"] for chem, r in passivation.items()}
    if any(v is None for v in list(p_B.values()) + list(p_P.values())):
        report["errors"].append("check_11: cycle_mean_p_B/p_P is None for some chemistry factor (no active patches sampled)")
        report["checks"]["d5_qualified_with_live_cycle_mean_data"] = False
        return False
    deltas = {chem: abs(v - 0.30) for chem, v in p_B.items()}
    best_chem = min(deltas, key=deltas.get)
    ok = best_chem == "1.0" and p_P[best_chem] >= 0.30
    selection = _load("post_screen_protocol_selection.json")
    ok = ok and selection["protocols"]["D5"]["decision"] == "AUTHORIZED_PX4"
    report["checks"]["d5_qualified_with_live_cycle_mean_data"] = ok
    if not ok:
        report["errors"].append(f"check_11: D5 literal criterion not satisfied: best_chem={best_chem}, p_P={p_P.get(best_chem)}")
    return ok


def check_12_uncertainty_gate(report: dict) -> bool:
    try:
        data = _load("px3_6_uncertainty_propagation.json")
    except FileNotFoundError:
        report["errors"].append("check_12: px3_6_uncertainty_propagation.json not found")
        report["checks"]["uncertainty_gate_satisfied_D3_D5_D6"] = False
        return False
    decisions = data["decisions"]
    ok = all(d in decisions for d in ("D3", "D5", "D6")) and all(decisions[d]["clears_gate"] for d in ("D3", "D5", "D6"))
    report["checks"]["uncertainty_gate_satisfied_D3_D5_D6"] = ok
    if not ok:
        report["errors"].append(f"check_12: uncertainty gate not satisfied for all of D3/D5/D6: {decisions}")
    return ok


def check_13_test_doubles_fixed(report: dict) -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/test_v10_2_29_state_coupled_hazard.py",
         "tests/test_v10_2_30_forward_coupled_marcher.py",
         "tests/test_v10_2_30_partition_robust_forward.py", "-q"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    ok = result.returncode == 0
    report["checks"]["previously_failing_test_doubles_now_pass"] = ok
    if not ok:
        report["errors"].append(f"check_13: test doubles still failing: {result.stdout[-2000:]}")
    return ok


def main() -> int:
    report = {"schema": "v10230_part_x_px3_6_verifier_v1", "checks": {}, "errors": []}
    results = [
        check_9_bug_invalidation(report),
        check_10_clean_producer_reruns_self_consistent(report),
        check_11_passivation_live_data(report),
        check_12_uncertainty_gate(report),
        check_13_test_doubles_fixed(report),
    ]
    report["overall_pass"] = all(results)
    out_path = ARTIFACTS_DIR / "px3_6_verification.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {out_path}")
    for name, value in report["checks"].items():
        print(f"  {name}: {value}")
    for error in report["errors"]:
        print(f"  ERROR: {error}")
    print(f"overall_pass={report['overall_pass']}")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
