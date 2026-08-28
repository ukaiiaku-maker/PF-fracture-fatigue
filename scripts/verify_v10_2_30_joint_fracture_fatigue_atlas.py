#!/usr/bin/env python3
"""Authoritative fail-closed verifier for the joint fracture-fatigue atlas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/joint_fracture_fatigue_archetype_atlas_v2"
EXPECTED_BRANCH = "codex/v10.2.30-joint-fracture-fatigue-archetype-atlas"
SOLVER_SHA = "c15a957161e8cdb43bf944243d8a18a64839dbfe0518aa2d1c5f0c93773b817b"
REQUIRED = [
    "fracture_source_manifest.json", "fracture_result_inventory.csv",
    "constitutive_lineage_matrix.csv", "candidate_cross_lineage_registry.csv",
    "fracture_state_semantics.csv", "equation_source_catalog.csv", "source_hashes.json",
    "fracture_analytical_input_manifest.json", "fracture_analytical_equation_lineage.md",
    "fracture_analytical_controls.json", "fracture_validation_plan.json",
    "monotonic_F0_predictions.parquet", "monotonic_F1_predictions.parquet",
    "monotonic_F2_predictions.parquet", "monotonic_cross_fidelity_validation.csv",
    "monotonic_common_population_error.csv", "mechanical_transfer_decomposition.csv",
    "mechanical_transfer_assumptions.json",
    "monotonic_state_validation.parquet", "temperature_sensitivity_decomposition.parquet",
    "loading_rate_sensitivity_decomposition.parquet", "parameter_sensitivity_jacobian.parquet",
    "historical_search_bounds.json", "response_atlas.parquet",
    "response_atlas_descriptors.parquet", "atlas_rejection_audit.csv",
    "fatigue_asymptotic_audit.parquet", "fatigue_asymptotic_thresholds.json",
    "binned_variance_screen.csv", "standardized_local_effect_screen.csv",
    "trend_confirmation_audit.csv", "trend_stratified_audit.csv",
    "archetype_definitions.json", "archetype_cluster_membership.csv",
    "archetype_cluster_stability.csv", "archetype_nonuniqueness_manifolds.parquet",
    "joint_fracture_fatigue_metrics.parquet", "joint_fracture_fatigue_correlations.csv",
    "joint_pareto_candidates.csv", "archetype_candidate_registry.csv",
    "archetype_candidate_diff_audit.csv", "archetype_candidate_hashes.json",
    "response_exemplar_registry.csv", "response_exemplar_hashes.json",
    "superseded_candidate_audit.csv",
    "prospective_monotonic_predictions.csv", "prospective_fatigue_predictions.csv",
    "prospective_state_predictions.parquet", "prospective_validation_plan.json",
    "physical_monotonic_validation.csv", "physical_fatigue_validation.csv",
    "physical_state_validation.parquet", "cross_fidelity_archetype_validation.csv",
    "physical_validation_controller_state.json", "F2B_activation_gate.csv",
    "joint_archetype_final_decision.md", "joint_archetype_final_decision.json",
]
FIGURES = [
    "CANONICAL_KINIT_VS_T_BY_FIDELITY", "F0_F1_F2_KINIT_COMPARISON",
    "FRACTURE_ANALYTICAL_PARITY", "PRE_EVENT_RADIUS_SHIELDING_BACKSTRESS",
    "TEMPERATURE_SENSITIVITY_DECOMPOSITION", "LOADING_RATE_SHIFT_DECOMPOSITION",
    "INTRINSIC_VS_STATE_MEDIATED_DBTT", "PEAK_MECHANISM_DECOMPOSITION",
    "BROAD_CLASS_VS_PEAK_CROSS_FIDELITY", "RESPONSE_ATLAS_CLASS_MAP",
    "RESPONSE_ATLAS_MECHANISM_MAP", "BINNED_SCREEN_PARAMETER_IMPORTANCE",
    "TREND_CONFIRMATION_AND_COUNTEREXAMPLES", "JOINT_FRACTURE_FATIGUE_RESPONSE_MAP",
    "MONOTONIC_RESISTANCE_VS_FATIGUE_THRESHOLD", "LOCAL_PARIS_SLOPE_VS_FRACTURE_CLASS",
    "RESPONSE_CLUSTER_MAP", "PROVISIONAL_RESPONSE_CLASS_SUMMARY",
    "UNVALIDATED_RESPONSE_EXEMPLAR_SUMMARY", "PROSPECTIVE_VS_PHYSICAL_MONOTONIC",
    "PROSPECTIVE_VS_PHYSICAL_FATIGUE", "CROSS_FIDELITY_RESPONSE_TRANSFER",
    "FINAL_JOINT_MECHANISM_SUMMARY",
]


def command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def verify(require_clean: bool = True) -> dict:
    checks: dict[str, bool] = {}
    checks["required_artifacts"] = all((OUT / name).is_file() for name in REQUIRED)
    checks["required_figures_png_pdf"] = all(
        (OUT / "figures" / f"{name}.png").is_file()
        and (OUT / "figures" / f"{name}.pdf").is_file() for name in FIGURES
    )
    source = json.loads((OUT / "fracture_source_manifest.json").read_text())
    checks["source_omissions_explained"] = all(
        item.get("available") or item.get("omission_explanation") for item in source["sources"]
    )
    checks["solver_hash"] = source["qualified_solver_sha256"] == SOLVER_SHA
    inventory = pd.read_csv(OUT / "fracture_result_inventory.csv")
    checks["no_duplicate_physical_result"] = not inventory.inventory_id.duplicated().any()
    checks["nonphysical_not_admitted"] = not inventory[
        inventory.completion_state.isin(["NUMERICAL_FAILURE", "RIGHT_CENSORED",
                                         "MECHANICAL_DOMAIN_LIMITATION", "FAILURE_MODE_PREEMPTION"])
    ].physical_admission.fillna(False).any()
    lineage = pd.read_csv(OUT / "constitutive_lineage_matrix.csv")
    checks["current_legacy_separated"] = (
        lineage.lineage.str.contains("LEGACY").any()
        and not lineage[lineage.lineage.str.contains("LEGACY")].used_in_current_analytical_model.any()
    )
    equations = pd.read_csv(OUT / "equation_source_catalog.csv")
    checks["no_empirical_toughness_or_paris"] = (
        (equations.equation == "EMPIRICAL_TOUGHNESS_OR_PARIS").any()
        and not equations[equations.equation == "EMPIRICAL_TOUGHNESS_OR_PARIS"].active.any()
    )
    freeze = json.loads((OUT / "fracture_analytical_input_manifest.json").read_text())
    checks["no_Kinit_fit"] = freeze["K_init_fit_performed"] is False
    state = pd.read_parquet(OUT / "monotonic_state_validation.parquet")
    checks["pre_event_state_semantics"] = set(state.state_semantics) == {"PRE_EVENT_PRE_TRANSLATION"}
    atlas = pd.read_parquet(OUT / "response_atlas.parquet")
    checks["atlas_count"] = len(atlas) == 147456
    checks["initial_sobol_count"] = int((atlas.sampling_stage == "INITIAL_SOBOL").sum()) == 131072
    checks["adaptive_count"] = int((atlas.sampling_stage == "ADAPTIVE_CLASS_BOUNDARY").sum()) == 16384
    checks["extrapolation_classified"] = set(atlas.fatigue_temperature_status) == {
        "ANALYTICAL_EXTRAPOLATION_UNVALIDATED"
    }
    asymptotic = pd.read_parquet(OUT / "fatigue_asymptotic_audit.parquet")
    checks["asymptotic_audit_complete"] = (
        len(asymptotic) == 4 * len(atlas)
        and asymptotic.groupby("atlas_id").Kmax_MPa_sqrt_m.nunique().eq(4).all()
    )
    checks["renewal_ceiling_identity"] = (
        (asymptotic.da_dN_m_per_cycle / asymptotic.da_dN_ceiling_m_per_cycle
         - asymptotic.da_dN_ceiling_fraction).abs().max() < 1e-13
        and (asymptotic.da_dN_ceiling_fraction
             - asymptotic.Lambda_c_over_inverse_tau_cycle_mean).abs().max() < 1e-13
    )
    checks["asymptotic_rejection_precedes_selection"] = (
        atlas.loc[atlas.fatigue_ceiling_dominated,
                  "barrier_or_state_asymptotic_artifact"].all()
        and not atlas.loc[atlas.barrier_or_state_asymptotic_artifact,
                          "candidate_selection_eligible"].any()
    )
    checks["descriptor_defined_archetypes"] = json.loads(
        (OUT / "archetype_definitions.json").read_text()
    )["descriptor_space_not_filename"] is True
    trends = pd.read_csv(OUT / "trend_confirmation_audit.csv")
    checks["trend_failures_not_suppressed"] = len(trends) == 12 and trends.status.notna().all()
    variance = pd.read_csv(OUT / "binned_variance_screen.csv")
    local = pd.read_csv(OUT / "standardized_local_effect_screen.csv")
    checks["screening_not_mislabeled_sobol_morris"] = (
        variance.estimator.str.contains("NOT_SOBOL").all()
        and local.estimator.str.contains("NOT_MORRIS").all()
    )
    current_candidates = pd.read_csv(OUT / "archetype_candidate_registry.csv")
    superseded = pd.read_csv(OUT / "superseded_candidate_audit.csv")
    checks["saturated_v1_candidates_invalidated"] = (
        current_candidates.empty and len(superseded) == 6
        and not superseded.admitted_to_current_candidate_registry.any()
    )
    membership = pd.read_csv(OUT / "archetype_cluster_membership.csv")
    checks["no_unvalidated_archetype_mapping"] = (
        set(membership.archetype_label) == {"MIXED_OR_UNRESOLVED"}
        and (membership.loc[~atlas.candidate_selection_eligible.to_numpy(), "cluster_id"] == -1).all()
    )
    transfer = pd.read_csv(OUT / "mechanical_transfer_decomposition.csv")
    assumptions = json.loads((OUT / "mechanical_transfer_assumptions.json").read_text())
    checks["constitutive_mechanical_transfer_separated"] = (
        (transfer.comparison_domain == "MATCHED_LOCAL_1D").any()
        and not assumptions["FEM_applied_to_local_mapping_qualified"]
        and not assumptions["FEM_transfer_map_fitted"]
    )
    physical_m = pd.read_csv(OUT / "physical_monotonic_validation.csv")
    physical_f = pd.read_csv(OUT / "physical_fatigue_validation.csv")
    checks["no_resumed_trajectory"] = not physical_m.resumed.any() and not physical_f.resumed.any()
    controller = json.loads((OUT / "physical_validation_controller_state.json").read_text())
    checks["controller_terminal"] = controller["state"] == "TERMINAL"
    checks["no_new_physics_launched"] = controller["new_jobs_launched"] == 0
    decision = json.loads((OUT / "joint_archetype_final_decision.json").read_text())
    markdown = (OUT / "joint_archetype_final_decision.md").read_text()
    checks["markdown_json_decision_agree"] = (
        decision["primary_classification"] in markdown
        and decision["markdown_json_classification_agreement_key"] in markdown
    )
    checks["all_29_answers"] = len(decision["answers"]) == 29
    checks["corrected_three_part_classification"] = decision["classifications"] == [
        "LOCAL_MONOTONIC_KERNEL_VALIDATED_FOR_MATCHED_1D",
        "CURRENT_TRANSIENT_STATE_CLOSURE_PARTIAL",
        "CROSS_FIDELITY_MECHANICAL_TRANSFER_UNRESOLVED",
    ]
    checks["zero_joint_archetype_candidates"] = (
        decision["candidate_count"] == 0
        and decision["superseded_candidate_count"] == 6
    )
    checks["two_compartment_class_not_rejected"] = (
        decision["tested_F2B_promoted"] is False
        and decision["two_compartment_model_class_rejected"] is False
    )
    checks["production_solver_unmodified_by_study"] = decision["solver_modified"] is False
    checks["no_new_2D"] = decision["new_2D_trajectories"] == 0
    checks["branch"] = command("git", "branch", "--show-current") == EXPECTED_BRANCH
    checks["git_diff_check"] = subprocess.run(["git", "diff", "--check"], cwd=ROOT).returncode == 0
    status = command("git", "status", "--short")
    checks["clean_worktree"] = (status == "") if require_clean else True
    processes = command("ps", "-axo", "command")
    active = [line for line in processes.splitlines()
              if "joint_fracture_fatigue_archetype_atlas_v2" in line
              and "verify_v10_2_30_joint" not in line]
    checks["no_active_workers"] = len(active) == 0
    checks = {name: bool(value) for name, value in checks.items()}
    passed = all(checks.values())
    result = {
        "schema": "joint_archetype_verification_v2", "status": "PASS" if passed else "FAIL",
        "checks": checks, "branch": command("git", "branch", "--show-current"),
        "HEAD": command("git", "rev-parse", "HEAD"), "active_worker_count": len(active),
        "atlas_rows": len(atlas), "inventory_rows": len(inventory),
        "required_artifact_count": len(REQUIRED), "required_figure_count": len(FIGURES),
    }
    (OUT / "joint_archetype_verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    if not passed:
        failed = [name for name,value in checks.items() if not value]
        raise SystemExit("verification FAIL: " + ", ".join(failed))
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    verify(require_clean=not args.allow_dirty)


if __name__ == "__main__":
    main()
