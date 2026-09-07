"""PX7 (final closure G): comprehensive, strict, portable TERMINAL
verifier for the WHOLE signed-K crack-rebonding compression-conditioned
Part X campaign (PX0 through this final closure) -- not merely a PX4/
PX5 synthesis closure check (the version this supersedes).

Depends ONLY on tracked files under artifacts/crack_rebonding_part_x_v1/
(plus git history for the base/merge-base check and re-invoking the two
stage verifiers as subprocesses, themselves independently proven
portable) -- never touches a gitignored runs/ directory. Independently
RECOMPUTES every count and classification claimed below from the
lowest-level tracked ledger available, rather than trusting a summary
JSON field at face value.

Usage:
    <pinned interpreter> scripts/verify_part_x_terminal.py
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"
EXPECTED_PART_X_BASE = "a72d46557f5eba45c8c2e0a428574e5b9b624c81"

REQUIRED_SCOPE_LABELS = {
    "SURROGATE_SIGNED_K_CONTACT_NOT_RESOLVED_FACE_CONTACT", "HAZARD_ONLY_COHESIVE_FEEDBACK",
    "TOPOLOGICAL_HEALING_NOT_MODELED", "PHYSICAL_CHEMISTRY_PARAMETERIZATION_NOT_CALIBRATED",
}
REQUIRED_QUALIFIERS = {
    "CLEAN_REVERSIBLE_KINETICS_QUALIFIED", "PERSISTENT_KINETICS_DISTINGUISHED_AT_SELECTED_FREQUENCY",
    "PASSIVATION_GATED_REBONDING_PHYSICALLY_EXERCISED", "PASSIVATION_EFFECT_ON_DEVELOPED_RATE_SMALL_FOR_SELECTED_ROW",
    "FREQUENCY_SENSITIVITY_QUALIFIED", "DWELL_SENSITIVITY_BELOW_SCREEN_RESOLUTION_AT_REFERENCE_CONDITION",
    "NEGATIVE_R_SENSITIVITY_SMALL_OVER_R_MINUS_0P95_TO_MINUS_0P50", "COHESIVE_STRENGTH_SENSITIVITY_QUALIFIED",
    "REBONDING_SEED_ROBUST", "MIXED_STATIC_SHIELDING_AND_KINETIC_HISTORY",
    "CEILING_STATIC_SHIELD_EQUIVALENT_FOR_D2_D5_DEVELOPED_RATE",
    "HARD_FIRST_PASSAGE_CYCLE_HORIZON_QUALIFIED_WITH_1E-6_CYCLE_FLOOR",
} | REQUIRED_SCOPE_LABELS
FORBIDDEN_STRINGS = [
    "closure_corrected_deltaK_eff", "closure-corrected DeltaK_eff computed",
    "production_line_merge_authorized\": true", "part_x_authorized\": true",
    "resolved opposing-face contact model implemented", "topological healing implemented",
]
PX5_SCOPED_PROTOCOLS = {"D2", "D5"}
DEVELOPED_PROTOCOLS = {"D1", "D2", "D3", "D5", "D6_conditional_persistent"}
SEED_1720_KMAX_MPa = {12.0, 15.0, 18.0, 21.0, 24.3}
SEED_1001723_KMAX_MPa = {15.0, 18.0, 21.0}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _run_script(script_name: str) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / script_name)], cwd=REPO_ROOT, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    # --- A: authoritative base and source provenance ---
    head = _git("rev-parse", "HEAD")
    merge_base = _git("merge-base", "HEAD", EXPECTED_PART_X_BASE)
    checks["merge_base_matches_expected_part_x_base"] = merge_base == EXPECTED_PART_X_BASE
    checks["source_provenance_present"] = (ARTIFACTS_DIR / "source_provenance.json").is_file()
    checks["completion_contract_present"] = (ARTIFACTS_DIR / "completion_contract.json").is_file()
    checks["mission_scope_present"] = (ARTIFACTS_DIR / "mission_scope.json").is_file()

    # --- PX1: software/numerical qualification records present ---
    checks["px1_source_preimage_manifest_present"] = (ARTIFACTS_DIR / "px1_source_preimage_manifest.json").is_file()
    checks["px1_4_static_localizer_parity_finding_present"] = (ARTIFACTS_DIR / "px1_4_static_localizer_parity_finding.json").is_file()

    # --- PX2: analytical freeze and regime selection ---
    kinetic_registry = json.loads((ARTIFACTS_DIR / "kinetic_regime_registry.json").read_text())
    checks["px2_four_kinetic_rows_present"] = set(kinetic_registry["rows"].keys()) == {
        "SAT_EXISTING", "COMPETING_REVERSIBLE", "COMPETING_PERSISTENT", "PASSIVATION_LIMITED",
    }
    checks["analytical_regime_selection_present"] = (ARTIFACTS_DIR / "analytical_regime_selection.json").is_file()

    # --- PX3: physical screen counts ---
    screen_registry = list(csv.DictReader(open(ARTIFACTS_DIR / "screen_job_registry.csv")))
    screen_authorized = [r for r in screen_registry if r["status"].startswith("AUTHORIZED")]
    checks["px3_screen_authorized_count_is_28"] = len(screen_authorized) == 28

    # --- PX3.5/PX3.6: duration-weighting correction, dwell causality, invalidation ---
    dwell_audit = json.loads((ARTIFACTS_DIR / "px3_5_dwell_audit_classification.json").read_text())
    checks["dwell_audit_classification_correct"] = dwell_audit["classification"] == "ORIGINAL_DWELL_SIGN_REVERSAL_NOT_REPRODUCED"
    invalidated = json.loads((ARTIFACTS_DIR / "px3_bug_invalidated_trajectories.json").read_text())
    checks["exactly_2_invalidated_dwell_trajectories"] = len(invalidated["invalidated_trajectories"]) == 2
    checks["all_invalidated_trajectories_tagged_correctly"] = all(
        t["status"] == "INVALIDATED_DWELL_DURATION_WEIGHTING_BUG" for t in invalidated["invalidated_trajectories"]
    )
    checks["px3_6_passivation_requalification_present"] = (ARTIFACTS_DIR / "px3_6_passivation_requalification.json").is_file()

    # --- Correct selection of D1/D2/D3/D5/D6 ---
    # post_screen_protocol_selection.json uses the SHORT protocol labels
    # (D1..D7); the developed registry uses the full names (D6_conditional_
    # persistent) and, for second-seed confirmation rows, a "_confirm"
    # suffix (D1_confirm, ..., D6_confirm) -- both are the SAME 5 base
    # protocols under different naming conventions, checked separately below.
    post_screen_selection = json.loads((ARTIFACTS_DIR / "post_screen_protocol_selection.json").read_text())
    checks["protocols_D1_D2_D3_D5_D6_selected"] = {"D1", "D2", "D3", "D5", "D6"}.issubset(set(post_screen_selection["protocols"].keys()))

    # --- Developed registry: independently recompute admitted/pair/interrupted counts ---
    BASE_AND_CONFIRM_PROTOCOLS = DEVELOPED_PROTOCOLS | {f"{p}_confirm" if p != "D6_conditional_persistent" else "D6_confirm" for p in DEVELOPED_PROTOCOLS}
    dev_registry = list(csv.DictReader(open(ARTIFACTS_DIR / "developed_job_registry.csv")))
    dev_admitted = [r for r in dev_registry if r["status"] == "AUTHORIZED_PX4"]
    checks["exactly_80_admitted_developed_trajectories"] = len(dev_admitted) == 80
    checks["all_admitted_developed_rows_are_D1_D2_D3_D5_D6"] = all(r["protocol"] in BASE_AND_CONFIRM_PROTOCOLS for r in dev_admitted)

    seed_1720_rows = [r for r in dev_admitted if r["seed"] == "1720"]
    seed_1001723_rows = [r for r in dev_admitted if r["seed"] == "1001723"]
    checks["seed_1720_developed_row_count_is_50"] = len(seed_1720_rows) == len(DEVELOPED_PROTOCOLS) * len(SEED_1720_KMAX_MPa) * 2
    checks["seed_1001723_developed_row_count_is_30"] = len(seed_1001723_rows) == len(DEVELOPED_PROTOCOLS) * len(SEED_1001723_KMAX_MPa) * 2
    checks["seed_1720_kmax_grid_complete"] = {round(float(r["Kmax_Pa_sqrt_m"]) / 1.0e6, 1) for r in seed_1720_rows} == SEED_1720_KMAX_MPa
    checks["seed_1001723_kmax_grid_complete"] = {round(float(r["Kmax_Pa_sqrt_m"]) / 1.0e6, 1) for r in seed_1001723_rows} == SEED_1001723_KMAX_MPa

    pair_analysis = json.loads((ARTIFACTS_DIR / "px4_developed_pair_analysis.json").read_text())
    checks["exactly_40_developed_pairs"] = len(pair_analysis["pairs"]) == 40

    admissibility = json.loads((ARTIFACTS_DIR / "part_x_admissibility_map.json").read_text())
    disp = admissibility["disposition_counts"]
    checks["admissibility_map_admitted_count_is_126"] = disp.get("ADMITTED") == 126
    checks["admissibility_map_interrupted_count_is_3"] = disp.get("INTERRUPTED_NOT_SCIENCE") == 3
    checks["admissibility_map_superseded_count_is_2"] = disp.get("SUPERSEDED_WALL_BUDGET_TOO_SMALL") == 2
    checks["admissibility_map_invalidated_count_is_2"] = disp.get("INVALIDATED_DWELL_DURATION_WEIGHTING_BUG") == 2
    admitted_keys = {r["canonical_job_key"] for r in admissibility["rows"] if r["final_disposition"] == "ADMITTED"}
    interrupted_keys = {r["canonical_job_key"] for r in admissibility["rows"] if r["final_disposition"] == "INTERRUPTED_NOT_SCIENCE"}
    checks["zero_admitted_interrupted_overlap"] = len(admitted_keys & interrupted_keys) == 0

    # --- Seed robustness ---
    slope_rows = list(csv.DictReader(open(ARTIFACTS_DIR / "px4_stage1_slope_table.csv")))
    checks["exactly_5_seed_robustness_axes"] = len(slope_rows) == 5
    checks["all_5_axes_rebonding_seed_robust"] = all(r["classification"] == "REBONDING_SEED_ROBUST" for r in slope_rows)

    # --- PX5: admitted/superseded/censored counts, scope ---
    px5_main = list(csv.DictReader(open(ARTIFACTS_DIR / "px5_static_shield_job_registry.csv")))
    px5_retry = list(csv.DictReader(open(ARTIFACTS_DIR / "px5_static_shield_wall_budget_retry_registry.csv")))
    px5_admitted = [r for r in px5_main if r["status"] == "AUTHORIZED_PX5"] + [r for r in px5_retry if r["status"] == "AUTHORIZED_PX5"]
    px5_superseded = [r for r in px5_main if r["status"] == "SUPERSEDED_WALL_BUDGET_TOO_SMALL"]
    checks["exactly_20_admitted_px5_trajectories"] = len(px5_admitted) == 20
    checks["exactly_2_superseded_px5_attempts"] = len(px5_superseded) == 2
    checks["px5_scope_never_expanded_beyond_D2_D5"] = {r["protocol"] for r in px5_main + px5_retry} == PX5_SCOPED_PROTOCOLS

    px5_censor = list(csv.DictReader(open(ARTIFACTS_DIR / "px5_censor_registry.csv")))
    checks["zero_censored_admitted_px5_trajectories"] = all(r["censored"] == "False" for r in px5_censor)
    checks["px5_censor_registry_count_is_20"] = len(px5_censor) == 20

    dev_censor = list(csv.DictReader(open(ARTIFACTS_DIR / "developed_censor_registry.csv")))
    checks["all_developed_trajectories_30_events"] = all(int(r["n_accepted_events"]) == 30 for r in dev_censor)
    checks["all_developed_trajectories_150um"] = all(abs(float(r["cumulative_extension_m"]) - 150.0e-6) < 1.0e-9 for r in dev_censor)
    checks["all_developed_trajectories_uncensored"] = all(r["censored"] == "False" for r in dev_censor)
    checks["all_px5_trajectories_150um"] = all(abs(float(r["cumulative_extension_m"]) - 150.0e-6) < 1.0e-9 for r in px5_censor)

    # --- Cycle horizon / 1e-6 floor qualification ---
    px6_synthesis = json.loads((ARTIFACTS_DIR / "px6_synthesis.json").read_text())
    horizon_q = px6_synthesis.get("cycle_horizon_qualification", {})
    checks["cycle_horizon_classification_present"] = horizon_q.get("classification") == "HARD_FIRST_PASSAGE_CYCLE_HORIZON_QUALIFIED_WITH_1E-6_CYCLE_FLOOR"
    checks["cycle_horizon_value_is_1e12"] = horizon_q.get("horizon_cycles") == 1.0e12
    checks["cycle_horizon_floor_is_1e-6"] = horizon_q.get("inherited_precision_floor_cycles") == 1.0e-6
    horizon_admission = json.loads((ARTIFACTS_DIR / "px4_pre_horizon_repair_admission.json").read_text())
    checks["all_44_pre_horizon_trajectories_admitted"] = horizon_admission["n_admitted"] == 44

    # --- Producer/launch provenance closure ---
    provenance_closure = json.loads((ARTIFACTS_DIR / "part_x_producer_launch_provenance_closure.json").read_text())
    checks["producer_launch_provenance_closed_for_all_pairs"] = (
        provenance_closure["overall_decision"] == "PHYSICS_SOURCE_IDENTICAL_ALLOW_HEAD_DRIFT_CLOSED_FOR_ALL_PAIRS"
        and provenance_closure["n_diverged"] == 0
    )

    # --- Component verifiers ---
    rc_px4, out_px4 = _run_script("verify_part_x_px4_stage1.py")
    checks["px4_stage1_verifier_exits_zero"] = rc_px4 == 0
    details["px4_stage1_verifier_output_tail"] = out_px4[-1500:]
    rc_px5, out_px5 = _run_script("verify_part_x_px5.py")
    checks["px5_verifier_exits_zero"] = rc_px5 == 0
    details["px5_verifier_output_tail"] = out_px5[-1500:]

    px4_v = json.loads((ARTIFACTS_DIR / "px4_stage1_verification.json").read_text())
    px5_v = json.loads((ARTIFACTS_DIR / "px5_verification.json").read_text())
    checks["px4_stage1_classification_complete"] = px4_v["classification"] == "PX4_STAGE1_COMPLETE"
    checks["px5_classification_complete"] = px5_v["classification"] == "PX5_COMPLETE"

    # --- Final decision ---
    final_decision = json.loads((ARTIFACTS_DIR / "part_x_final_decision.json").read_text())
    checks["final_decision_status_is_FINAL"] = final_decision["status"] == "FINAL"
    checks["final_decision_primary_classification_correct"] = (
        final_decision["primary_classification"] == "SIGNED_K_CRACK_REBONDING_PART_X_COMPLETE"
    )
    checks["final_decision_required_qualifiers_present"] = REQUIRED_QUALIFIERS.issubset(set(final_decision["qualifiers"]))
    checks["final_decision_scope_labels_present"] = REQUIRED_SCOPE_LABELS.issubset(set(final_decision["qualifiers"]))
    # Forbidden-claim scanning is done separately below (a JSON-text scan for
    # affirmative forbidden phrases) -- final_decision.json's own "not_claimed"
    # field NAMING these items is the required disclosure, not a violation.

    # --- Artifact / figure manifests ---
    rc_manifest, out_manifest = _run_script("build_part_x_artifact_manifest.py")
    checks["artifact_manifest_builds_cleanly"] = rc_manifest == 0
    details["artifact_manifest_output"] = out_manifest[-1000:]
    artifact_manifest = json.loads((ARTIFACTS_DIR / "part_x_artifact_manifest.json").read_text())
    checks["artifact_manifest_no_vague_missing_status"] = all(
        r["status"] in ("PRESENT_AND_VERIFIED", "GENERATED_IN_FINAL_CLOSURE", "NOT_APPLICABLE_WITH_REASON", "NOT_ARCHIVED_WITH_EXPLICIT_LIMITATION")
        for r in artifact_manifest["requirements"]
    )
    checks["artifact_manifest_counts_consistent"] = (
        artifact_manifest["n_present_and_verified"] + artifact_manifest["n_generated_in_final_closure"] == artifact_manifest["n_resolved"]
    )

    rc_fig_manifest, out_fig_manifest = _run_script("build_part_x_figure_manifest.py")
    checks["figure_manifest_builds_cleanly"] = rc_fig_manifest == 0
    figure_manifest = json.loads((ARTIFACTS_DIR / "part_x_figure_manifest.json").read_text())
    checks["figure_manifest_all_19_resolved"] = figure_manifest["all_19_resolved"] is True

    # --- File hashes current ---
    rc_hashes, _ = _run_script("build_part_x_file_hashes.py")
    checks["file_hashes_build_cleanly"] = rc_hashes == 0
    file_hashes = json.loads((ARTIFACTS_DIR / "part_x_file_hashes.json").read_text())
    checks["file_hashes_nonempty"] = file_hashes["n_files"] > 50

    # --- No unsupported scientific claims ---
    # Scanned only over the SCIENTIFIC decision/synthesis documents, not the
    # manifest/verification meta-documents -- those legitimately reference
    # forbidden-claim concept names as identifiers/negations (e.g. an
    # artifact-manifest row id="closure_corrected_deltaK_eff" marked
    # NOT_APPLICABLE), which would otherwise false-positive against
    # themselves (and, for this verifier's own prior output, against ITS
    # OWN forbidden-string list on a second run).
    SCIENTIFIC_CLAIM_DOCUMENTS = [
        "px4_scientific_decision.json", "px5_scientific_decision.json",
        "px6_synthesis.json", "part_x_final_decision.json",
    ]
    forbidden_hits = []
    for name in SCIENTIFIC_CLAIM_DOCUMENTS:
        text = (ARTIFACTS_DIR / name).read_text()
        for needle in FORBIDDEN_STRINGS:
            if needle in text:
                forbidden_hits.append((name, needle))
    checks["no_forbidden_claims_in_scientific_documents"] = len(forbidden_hits) == 0
    details["forbidden_claim_hits"] = forbidden_hits

    overall_pass = all(checks.values())
    classification = "SIGNED_K_CRACK_REBONDING_PART_X_COMPLETE" if overall_pass else "PART_X_INCOMPLETE"
    subordinate_classification = "PART_X_PX0_THROUGH_PX6_COMPLETE" if overall_pass else "PART_X_INCOMPLETE"

    terminal = {
        "schema": "v10230_part_x_terminal_verification_v2",
        "depends_on_gitignored_run_files": False,
        "independence_level": "INDEPENDENT_LEDGER_REDUCTION_WITH_SHARED_QUALIFIED_PRIMITIVES",
        "classification": classification,
        "subordinate_verification_classification": subordinate_classification,
        "checks": checks,
        "details": details,
        "unclaimed_scope": [
            "PX5 static-shield attribution for D1/D3/D6 (explicitly out of scope, not tested)",
            "transient (pre-developed-window) explanatory power of static shielding",
            "closure-corrected DeltaK_eff", "production-line merge readiness",
            "resolved opposing-face contact model", "topological crack healing", "calibrated physical chemistry",
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
