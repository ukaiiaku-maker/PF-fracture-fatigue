"""Final closure D: the Part X artifact manifest -- maps every artifact
category required by the original Part X contract to exactly one
canonical file and a resolution status. Reuses existing files wherever
they already satisfy a requirement rather than creating duplicates.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"

PART_X = "artifacts/crack_rebonding_part_x_v1"

REQUIREMENTS = [
    dict(id="source_provenance", description="Source/provenance records",
         canonical_file=f"{PART_X}/source_provenance.json", status="PRESENT_AND_VERIFIED"),
    dict(id="config_and_bundle_hashes", description="Configuration and source-bundle hashes",
         canonical_file=f"{PART_X}/part_x_producer_launch_provenance_closure.json (+ file_hashes.json, inherited_artifact_hashes.json)",
         status="PRESENT_AND_VERIFIED"),
    dict(id="analytical_kinetic_regime_atlas", description="Analytical kinetic-regime atlas",
         canonical_file=f"{PART_X}/analytical_phase_atlas.csv (+ .parquet, kinetic_regime_registry.json)",
         status="PRESENT_AND_VERIFIED"),
    dict(id="screen_registry_and_ledger", description="Screen registry and screen event ledger",
         canonical_file=f"{PART_X}/screen_job_registry.csv, {PART_X}/screen_event_ledger.json",
         status="PRESENT_AND_VERIFIED"),
    dict(id="developed_registry_and_ledger", description="Developed registry and developed event ledger",
         canonical_file=f"{PART_X}/developed_job_registry.csv, {PART_X}/developed_event_ledger.json",
         status="PRESENT_AND_VERIFIED"),
    dict(id="px5_registry_and_ledger", description="PX5 static-control registry and event ledger",
         canonical_file=f"{PART_X}/px5_static_shield_job_registry.csv (+ wall_budget_retry_registry.csv), {PART_X}/px5_event_ledger.json",
         status="PRESENT_AND_VERIFIED"),
    dict(id="rate_tables", description="Rate tables",
         canonical_file=f"{PART_X}/px4_stage1_rate_table.csv", status="PRESENT_AND_VERIFIED"),
    dict(id="slope_and_curvature_tables", description="Slope and curvature tables",
         canonical_file=f"{PART_X}/px4_stage1_slope_table.csv (global 3-point fit), {PART_X}/px4_local_slope_curvature_table.csv (local secant + curvature)",
         status="GENERATED_IN_FINAL_CLOSURE"),
    dict(id="seed_robustness_table", description="Seed-robustness table",
         canonical_file=f"{PART_X}/px4_stage1_slope_table.csv", status="PRESENT_AND_VERIFIED"),
    dict(id="attempt_registry", description="Attempt registry",
         canonical_file=f"{PART_X}/developed_attempt_registry.csv, {PART_X}/px5_attempt_registry.csv",
         status="PRESENT_AND_VERIFIED"),
    dict(id="censor_registry", description="Censor registry",
         canonical_file=f"{PART_X}/developed_censor_registry.csv, {PART_X}/px5_censor_registry.csv",
         status="PRESENT_AND_VERIFIED"),
    dict(id="invalidated_dwell_bug_records", description="Invalidated dwell-bug records",
         canonical_file=f"{PART_X}/px3_bug_invalidated_trajectories.json", status="PRESENT_AND_VERIFIED"),
    dict(id="transition_action_flux_table", description="Transition-action and transition-flux table",
         canonical_file=f"{PART_X}/analytical_phase_atlas.csv (A_CB/A_BC/A_PC/A_CP, F_CB/F_BC/F_PC/F_CP columns)",
         status="PRESENT_AND_VERIFIED"),
    dict(id="contact_exposure_table", description="Contact-exposure table",
         canonical_file=f"{PART_X}/analytical_phase_atlas.csv (contact_time_s/contact_time_sinusoid_s/contact_time_dwell_s columns)",
         status="PRESENT_AND_VERIFIED"),
    dict(id="R_sensitivity_table", description="R-sensitivity table",
         canonical_file=f"{PART_X}/px3_screen_pair_analysis.csv (protocol=7.1_R_panel) + px4_scientific_decision.json (D1_vs_D2)",
         status="PRESENT_AND_VERIFIED"),
    dict(id="frequency_sensitivity_table", description="Frequency-sensitivity table",
         canonical_file=f"{PART_X}/px3_screen_pair_analysis.csv (protocol=7.2_frequency_panel), {PART_X}/px3_5_frequency_bisection.json + px4_scientific_decision.json (D2_vs_D3)",
         status="PRESENT_AND_VERIFIED"),
    dict(id="dwell_sensitivity_table", description="Dwell-sensitivity table",
         canonical_file=f"{PART_X}/px3_screen_pair_analysis.csv (protocol=7.3_dwell_panel), {PART_X}/px3_5_dwell_audit_classification.json",
         status="PRESENT_AND_VERIFIED"),
    dict(id="passivation_chemistry_table", description="Passivation/chemistry-sensitivity table",
         canonical_file=f"{PART_X}/px3_screen_pair_analysis.csv (protocol=7.5_passivation_chemistry), {PART_X}/px3_6_passivation_requalification.json + px4_scientific_decision.json (D2_vs_D5)",
         status="PRESENT_AND_VERIFIED"),
    dict(id="cohesive_strength_table", description="Cohesive-strength-sensitivity table",
         canonical_file=f"{PART_X}/px3_screen_pair_analysis.csv (protocol=7.6_cohesive_strength)",
         status="PRESENT_AND_VERIFIED"),
    dict(id="reversible_vs_persistent_comparison", description="Reversible-versus-persistent comparison",
         canonical_file=f"{PART_X}/px3_screen_pair_analysis.csv (protocol=7.4_reversible_vs_persistent[_at_transition]) + px4_scientific_decision.json (D3_vs_D6)",
         status="PRESENT_AND_VERIFIED"),
    dict(id="static_vs_dynamic_attribution_table", description="Static-versus-dynamic attribution table",
         canonical_file=f"{PART_X}/px5_static_shield_analysis.csv", status="PRESENT_AND_VERIFIED"),
    dict(id="censor_admissibility_map", description="Censor/admissibility map",
         canonical_file=f"{PART_X}/part_x_admissibility_map.csv", status="GENERATED_IN_FINAL_CLOSURE"),
    dict(id="final_decision", description="Final decision",
         canonical_file=f"{PART_X}/part_x_final_decision.json", status="GENERATED_IN_FINAL_CLOSURE"),
    dict(id="terminal_verification", description="Terminal verification",
         canonical_file=f"{PART_X}/part_x_terminal_verification.json", status="GENERATED_IN_FINAL_CLOSURE"),
    dict(id="final_handoff", description="Final handoff",
         canonical_file=f"{PART_X}/part_x_final_handoff.md", status="GENERATED_IN_FINAL_CLOSURE"),
    dict(id="producer_launch_provenance_closure", description="Producer-versus-launch source-identity closure",
         canonical_file=f"{PART_X}/part_x_producer_launch_provenance_closure.json", status="GENERATED_IN_FINAL_CLOSURE"),
    dict(id="closure_corrected_deltaK_eff", description="Closure-corrected DeltaK_eff",
         canonical_file="N/A", status="NOT_APPLICABLE_WITH_REASON",
         reason="Explicitly out of Part X scope -- the campaign uses a signed-K surrogate contact model "
                "(SURROGATE_SIGNED_K_CONTACT_NOT_RESOLVED_FACE_CONTACT), never resolves opposing-face contact, "
                "and never computes a closure-corrected DeltaK_eff. This is a scope boundary, not a gap."),
    dict(id="resolved_face_contact_model", description="Resolved opposing-face contact model",
         canonical_file="N/A", status="NOT_APPLICABLE_WITH_REASON",
         reason="Out of scope by design (surrogate signed-K compression-conditioning is the qualified contact "
                "model for this campaign); resolving true opposing-face contact would require a different "
                "constitutive model, not an analysis-only closure step."),
    dict(id="topological_healing_model", description="Topological crack-healing model",
         canonical_file="N/A", status="NOT_APPLICABLE_WITH_REASON",
         reason="Out of scope by design (TOPOLOGICAL_HEALING_NOT_MODELED) -- the hazard-only cohesive-feedback "
                "surrogate never models topological reconnection of crack faces."),
    dict(id="calibrated_chemistry_parameterization", description="Calibrated physical chemistry parameterization",
         canonical_file="N/A", status="NOT_APPLICABLE_WITH_REASON",
         reason="Out of scope (PHYSICAL_CHEMISTRY_PARAMETERIZATION_NOT_CALIBRATED) -- chemistry_factor and the "
                "bond/rupture/depassivation/repassivation barriers are analytically constructed reference "
                "values (PX2's zero-fitting kinetic-regime characterization), never calibrated against an "
                "experimental measurement."),
    dict(id="transient_static_shield_explanatory_power", description="Transient (pre-developed-window) explanatory power of static shielding",
         canonical_file="N/A", status="NOT_ARCHIVED_WITH_EXPLICIT_LIMITATION",
         reason="PX5's S_h_static/S_h_dynamic comparison is computed ONLY over the stable/developed window "
                "(stable_growth_gate's developed_interval); no PX5 analysis characterizes the TRANSIENT "
                "(first ~20um) approach to that window. The raw event-by-event data needed to do so IS "
                "archived in px5_event_ledger.json (every event, including the transient ones), so this is a "
                "genuinely available-but-unanalyzed limitation, not an unrecoverable data gap."),
    dict(id="px5_d1_d3_d6_static_attribution", description="PX5 static-shield attribution for D1/D3/D6",
         canonical_file="N/A", status="NOT_APPLICABLE_WITH_REASON",
         reason="Explicitly out of PX5's prescribed scope (D2/D5 only, per the review's own scope-restriction "
                "instruction: 'do NOT expand to D1/D3/D6 without new prospective reason'). No physical "
                "trajectories exist for this comparison and none were launched to manufacture one."),
]


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows_out = []
    for req in REQUIREMENTS:
        row = dict(req)
        row.setdefault("reason", "")
        # Verify PRESENT_AND_VERIFIED/GENERATED_IN_FINAL_CLOSURE canonical files actually exist
        # (the first file named in a comma-separated canonical_file list, when it is a real path).
        first_path_str = row["canonical_file"].split(",")[0].split(" (")[0].strip()
        if row["status"] in ("PRESENT_AND_VERIFIED", "GENERATED_IN_FINAL_CLOSURE") and first_path_str != "N/A":
            full_path = REPO_ROOT / first_path_str
            if not full_path.is_file():
                raise RuntimeError(f"manifest entry '{row['id']}' claims {row['status']} but {full_path} does not exist")
        rows_out.append(row)

    n_present = sum(1 for r in rows_out if r["status"] == "PRESENT_AND_VERIFIED")
    n_generated = sum(1 for r in rows_out if r["status"] == "GENERATED_IN_FINAL_CLOSURE")
    n_na = sum(1 for r in rows_out if r["status"] == "NOT_APPLICABLE_WITH_REASON")
    n_not_archived = sum(1 for r in rows_out if r["status"] == "NOT_ARCHIVED_WITH_EXPLICIT_LIMITATION")
    n_resolved = n_present + n_generated

    with (ARTIFACTS_DIR / "part_x_artifact_manifest.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "description", "canonical_file", "status", "reason"], lineterminator="\n")
        writer.writeheader()
        for r in rows_out:
            writer.writerow({k: r[k] for k in ("id", "description", "canonical_file", "status", "reason")})

    (ARTIFACTS_DIR / "part_x_artifact_manifest.json").write_text(json.dumps({
        "schema": "v10230_part_x_artifact_manifest_v1",
        "n_requirements": len(rows_out), "n_present_and_verified": n_present, "n_generated_in_final_closure": n_generated,
        "n_not_applicable_with_reason": n_na, "n_not_archived_with_explicit_limitation": n_not_archived,
        "n_resolved": n_resolved, "requirements": rows_out,
    }, indent=2, default=str))

    print(f"Wrote part_x_artifact_manifest.{{csv,json}}: {len(rows_out)} requirements")
    print(f"  PRESENT_AND_VERIFIED: {n_present}, GENERATED_IN_FINAL_CLOSURE: {n_generated}, "
          f"NOT_APPLICABLE_WITH_REASON: {n_na}, NOT_ARCHIVED_WITH_EXPLICIT_LIMITATION: {n_not_archived}")
    print(f"  resolved (present+generated): {n_resolved}/{len(rows_out)}")


if __name__ == "__main__":
    main()
