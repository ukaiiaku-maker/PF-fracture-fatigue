"""PX7: strict, portable TERMINAL verifier for the whole signed-K
crack-rebonding compression-conditioned Part X campaign (PX0-PX6).

Depends ONLY on tracked files under artifacts/crack_rebonding_part_x_v1/
plus re-invoking the two stage verifiers as subprocesses (which are
themselves proven portable) -- never touches a gitignored runs/
directory directly. Does not recompute any physics itself: this is a
CLOSURE check that (a) both stage verifiers independently pass, (b) the
synthesis document's claims are consistent with what those verifiers
actually certified, and (c) no artifact in this campaign has smuggled in
a claim the mission explicitly forbids (closure-corrected DeltaK_eff,
production-line merge authorization, or an unscoped PX5 expansion to
D1/D3/D6).

Usage:
    <pinned interpreter> scripts/verify_part_x_terminal.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"

REQUIRED_SCOPE_LABELS = {
    "SURROGATE_SIGNED_K_CONTACT_NOT_RESOLVED_FACE_CONTACT", "HAZARD_ONLY_COHESIVE_FEEDBACK",
    "TOPOLOGICAL_HEALING_NOT_MODELED", "PHYSICAL_CHEMISTRY_PARAMETERIZATION_NOT_CALIBRATED",
}
FORBIDDEN_STRINGS = [
    "closure_corrected_deltaK_eff", "closure-corrected DeltaK_eff computed",
    "production_line_merge_authorized\": true", "part_x_authorized\": true",
]
PX5_SCOPED_PROTOCOLS = {"D2", "D5"}


def _run_verifier(script_name: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script_name)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    rc_px4, out_px4 = _run_verifier("verify_part_x_px4_stage1.py")
    checks["px4_stage1_verifier_exits_zero"] = rc_px4 == 0
    details["px4_stage1_verifier_output_tail"] = out_px4[-2000:]

    rc_px5, out_px5 = _run_verifier("verify_part_x_px5.py")
    checks["px5_verifier_exits_zero"] = rc_px5 == 0
    details["px5_verifier_output_tail"] = out_px5[-2000:]

    px4_v = json.loads((ARTIFACTS_DIR / "px4_stage1_verification.json").read_text())
    px5_v = json.loads((ARTIFACTS_DIR / "px5_verification.json").read_text())
    checks["px4_stage1_classification_complete"] = px4_v["classification"] == "PX4_STAGE1_COMPLETE"
    checks["px5_classification_complete"] = px5_v["classification"] == "PX5_COMPLETE"

    synthesis = json.loads((ARTIFACTS_DIR / "px6_synthesis.json").read_text())
    checks["synthesis_reports_px4_verification_consistent"] = (
        synthesis["px4_findings"]["verification_classification"] == px4_v["classification"]
    )
    checks["synthesis_reports_px5_verification_consistent"] = (
        synthesis["px5_findings"]["verification_classification"] == px5_v["classification"]
    )
    checks["synthesis_reports_all_seed_axes_robust"] = synthesis["px4_findings"]["all_5_axes_seed_robust"] is True
    checks["synthesis_scope_labels_present"] = REQUIRED_SCOPE_LABELS.issubset(set(synthesis["scope_reminder"]))

    px4_decision = json.loads((ARTIFACTS_DIR / "px4_scientific_decision.json").read_text())
    px5_decision = json.loads((ARTIFACTS_DIR / "px5_scientific_decision.json").read_text())
    checks["px4_decision_status_provisional"] = px4_decision["status"] == "PROVISIONAL_PENDING_PX5_PX6_PX7"
    checks["px5_decision_status_provisional"] = px5_decision["status"] == "PROVISIONAL_PENDING_PX6_PX7"
    checks["px5_decision_scope_labels_present"] = REQUIRED_SCOPE_LABELS.issubset(set(px5_decision["scope_reminder"]))
    checks["px4_decision_scope_labels_present"] = REQUIRED_SCOPE_LABELS.issubset(set(px4_decision["scope_reminder"]))

    px5_registry_protocols = set()
    for name in ("px5_static_shield_job_registry.csv", "px5_static_shield_wall_budget_retry_registry.csv"):
        import csv
        for row in csv.DictReader(open(ARTIFACTS_DIR / name)):
            if row["status"] in ("AUTHORIZED_PX5", "SUPERSEDED_WALL_BUDGET_TOO_SMALL"):
                px5_registry_protocols.add(row["protocol"])
    checks["px5_scope_never_expanded_beyond_D2_D5"] = px5_registry_protocols == PX5_SCOPED_PROTOCOLS
    details["px5_registry_protocols_observed"] = sorted(px5_registry_protocols)

    forbidden_hits = []
    for path in ARTIFACTS_DIR.glob("*.json"):
        text = path.read_text()
        for needle in FORBIDDEN_STRINGS:
            if needle in text:
                forbidden_hits.append((path.name, needle))
    checks["no_forbidden_claims_in_any_artifact"] = len(forbidden_hits) == 0
    details["forbidden_claim_hits"] = forbidden_hits

    overall_pass = all(checks.values())
    classification = "PART_X_PX0_THROUGH_PX6_COMPLETE" if overall_pass else "PART_X_INCOMPLETE"

    terminal = {
        "schema": "v10230_part_x_terminal_verification_v1",
        "classification": classification,
        "checks": checks,
        "details": details,
        "unclaimed_scope": [
            "PX5 static-shield attribution for D1/D3/D6 (explicitly out of scope, not tested)",
            "transient (pre-developed-window) explanatory power of static shielding",
            "closure-corrected DeltaK_eff",
            "production-line merge readiness",
        ],
        "overall_pass": overall_pass,
    }
    out_path = ARTIFACTS_DIR / "part_x_terminal_verification.json"
    out_path.write_text(json.dumps(terminal, indent=2, sort_keys=True, default=str) + "\n")
    print(f"wrote {out_path}")
    print(f"classification={classification}")
    print(f"overall_pass={overall_pass}")
    for key, value in checks.items():
        if not value:
            print(f"  FAILED CHECK: {key}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
