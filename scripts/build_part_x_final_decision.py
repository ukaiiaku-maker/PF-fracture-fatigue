"""Final closure C: the terminal scientific decision for the whole
signed-K crack-rebonding compression-conditioned Part X campaign.

Reduces already-committed, already-independently-verified tracked
artifacts (px4_scientific_decision.json, px5_scientific_decision.json,
px6_synthesis.json, part_x_admissibility_map.json, part_x_producer_
launch_provenance_closure.json, px4_local_slope_curvature_table.csv,
px3_screen_pair_analysis.csv) into ONE terminal decision document.
Introduces no new physical claim -- every number here traces to a
tracked, independently-verified artifact.

Historical component decisions (px4_scientific_decision.json,
px5_scientific_decision.json) are left in place UNCHANGED in status
(still PROVISIONAL_PENDING_*) -- this document is what supersedes them,
recorded explicitly in the "supersedes" field below, per the review's
own "prefer preserving those historical files unchanged" instruction.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"

PRIMARY_CLASSIFICATION = "SIGNED_K_CRACK_REBONDING_PART_X_COMPLETE"
FINAL_STATUS = "FINAL"

PROTOCOL_DEFINITIONS = {
    "D1": {"row_name": "COMPETING_REVERSIBLE", "R": -0.95, "frequency_Hz": 1000.0, "minimum_load_hold_s": 0.0,
           "chemistry_factor": 1.0, "K_rebond_max_target_Pa_sqrt_m": 900000.0,
           "purpose": "R-sensitivity leg of the COMPETING_REVERSIBLE kinetic row at near-fully-reversed R"},
    "D2": {"row_name": "COMPETING_REVERSIBLE", "R": -0.50, "frequency_Hz": 1000.0, "minimum_load_hold_s": 0.0,
           "chemistry_factor": 1.0, "K_rebond_max_target_Pa_sqrt_m": 900000.0,
           "purpose": "reference/baseline developed protocol (also a PX5-scoped attribution target)"},
    "D3": {"row_name": "COMPETING_REVERSIBLE", "R": -0.50, "frequency_Hz": 316.227766016837952, "minimum_load_hold_s": 0.0,
           "chemistry_factor": 1.0, "K_rebond_max_target_Pa_sqrt_m": 900000.0,
           "purpose": "reduced-frequency leg of COMPETING_REVERSIBLE (frequency-sensitivity)"},
    "D5": {"row_name": "PASSIVATION_LIMITED", "R": -0.50, "frequency_Hz": 1000.0, "minimum_load_hold_s": 0.0,
           "chemistry_factor": 1.0, "K_rebond_max_target_Pa_sqrt_m": 900000.0,
           "purpose": "passivation-gated kinetic row at the reference condition (also a PX5-scoped attribution target)"},
    "D6_conditional_persistent": {"row_name": "COMPETING_PERSISTENT", "R": -0.50, "frequency_Hz": 316.227766016837952,
           "minimum_load_hold_s": 0.0, "chemistry_factor": 1.0, "K_rebond_max_target_Pa_sqrt_m": 900000.0,
           "purpose": "conditional-persistence kinetic row at D3's reduced frequency (persistence-vs-reversibility)"},
}
KMAX_GRID_SEED_1720_MPa = [12.0, 15.0, 18.0, 21.0, 24.3]
KMAX_GRID_SEED_1001723_MPa = [15.0, 18.0, 21.0]
SEEDS = [1720, 1001723]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    px4_decision = json.loads((ARTIFACTS_DIR / "px4_scientific_decision.json").read_text())
    px5_decision = json.loads((ARTIFACTS_DIR / "px5_scientific_decision.json").read_text())
    px6_synthesis = json.loads((ARTIFACTS_DIR / "px6_synthesis.json").read_text())
    admissibility = json.loads((ARTIFACTS_DIR / "part_x_admissibility_map.json").read_text())
    provenance_closure = json.loads((ARTIFACTS_DIR / "part_x_producer_launch_provenance_closure.json").read_text())
    px4_stage1_verification = json.loads((ARTIFACTS_DIR / "px4_stage1_verification.json").read_text())
    px5_verification = json.loads((ARTIFACTS_DIR / "px5_verification.json").read_text())
    screen_pairs = list(csv.DictReader(open(ARTIFACTS_DIR / "px3_screen_pair_analysis.csv")))
    dwell_audit = json.loads((ARTIFACTS_DIR / "px3_5_dwell_audit_classification.json").read_text())
    invalidated = json.loads((ARTIFACTS_DIR / "px3_bug_invalidated_trajectories.json").read_text())

    head = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    merge_base = _git("merge-base", "HEAD", "a72d46557f5eba45c8c2e0a428574e5b9b624c81")

    dispositions = admissibility["disposition_counts"]

    def _screen(protocol: str, **filt) -> list[dict]:
        out = [r for r in screen_pairs if r["protocol"] == protocol and all(r[k] == v for k, v in filt.items())]
        return out

    R_panel = _screen("7.1_R_panel")
    freq_panel = _screen("7.2_frequency_panel")
    dwell_panel = _screen("7.3_dwell_panel")
    passivation_panel = _screen("7.5_passivation_chemistry")
    cohesive_panel = _screen("7.6_cohesive_strength")
    reversible_persistent_panel = _screen("7.4_reversible_vs_persistent") + _screen("7.4_reversible_vs_persistent_at_transition")

    decision = {
        "schema": "v10230_part_x_final_decision_v1",
        "status": FINAL_STATUS,
        "primary_classification": PRIMARY_CLASSIFICATION,
        "subordinate_verification_classification": "PART_X_PX0_THROUGH_PX6_COMPLETE",
        "terminal_provenance": {
            "final_sha": head, "branch": branch, "merge_base_with_expected_part_x_base": merge_base,
            "expected_part_x_base": "a72d46557f5eba45c8c2e0a428574e5b9b624c81",
        },
        "supersedes": {
            "px4_scientific_decision.json": {
                "original_status_preserved_unchanged": px4_decision["status"],
                "superseded_by": "this document (part_x_final_decision.json)",
                "note": "px4_scientific_decision.json is left in place with its original PROVISIONAL status and "
                        "correction_history intact; this document is the terminal scientific classification for "
                        "the whole campaign, incorporating and superseding it.",
            },
            "px5_scientific_decision.json": {
                "original_status_preserved_unchanged": px5_decision["status"],
                "superseded_by": "this document (part_x_final_decision.json)",
                "note": "px5_scientific_decision.json is left in place with its original PROVISIONAL status and "
                        "correction_history intact; this document is the terminal scientific classification for "
                        "the whole campaign, incorporating and superseding it.",
            },
        },
        "campaign_lineage": (
            "PX0 (repo/provenance contract) -> PX1 (exact phase-resolved P/C/B kinetics + one-cycle monodromy "
            "primitives, software/numerical qualification) -> PX2 (zero-fitting analytical characterization of "
            "4 candidate kinetic regimes: SAT_EXISTING, COMPETING_REVERSIBLE, COMPETING_PERSISTENT, "
            "PASSIVATION_LIMITED) -> PX3 (28-job physical screen across R/frequency/dwell/passivation/cohesive-"
            "strength/persistence panels at a Kmax=18 MPa*sqrt(m) reference condition) -> PX3.5/PX3.6 (a real "
            "duration-weighting bug found and fixed, dwell-causality audit, live-sampled passivation "
            "requalification, final developed-protocol selection: D1/D2/D3/D5/D6) -> PX4/PX4.1 (80-trajectory "
            "developed campaign: 5 protocols x 5-point Kmax grid at seed=1720, plus seed=1001723 second-seed "
            "confirmation at 3 Kmax across all 5 protocols, under an exact first-passage 1e12-cycle horizon) -> "
            "PX5 (scoped to D2/D5: 20-trajectory static-shield attribution against two prescribed controls) -> "
            "PX6 (synthesis and figures) -> PX7 (strict portable terminal verifier) -> this final closure "
            "(wording corrections, producer/launch provenance closure, artifact/figure manifests, terminal "
            "decision)."
        ),
        "job_and_attempt_counts": {
            "px3_screen_authorized_jobs": 28,
            "px4_developed_admitted_trajectories": 80,
            "px4_developed_matched_finite_zero_pairs": 40,
            "px5_admitted_static_control_trajectories": 20,
            "total_unique_canonical_job_keys_across_campaign": admissibility["n_total_rows"],
            "disposition_counts": dispositions,
        },
        "kmax_grids": {
            "seed_1720_five_point_grid_MPa_sqrt_m": KMAX_GRID_SEED_1720_MPa,
            "seed_1001723_three_point_confirmation_grid_MPa_sqrt_m": KMAX_GRID_SEED_1001723_MPa,
            "px5_grid_MPa_sqrt_m": KMAX_GRID_SEED_1720_MPa,
            "px3_screen_reference_Kmax_MPa_sqrt_m": 18.0,
        },
        "seeds": {"primary": 1720, "second_seed_confirmation": 1001723},
        "protocol_definitions": PROTOCOL_DEFINITIONS,
        "developed_da_dN_and_S_h_interpretation": (
            "da/dN is computed over the STABLE/DEVELOPED window only (stable_growth_gate's developed_interval: "
            ">=10 events, >=50um final-window growth, late/early rate ratio in [0.5,2.0] -- excludes the first "
            "20um as transient). S_h_developed = log10(da/dN_finite / da/dN_zero); NEGATIVE S_h means cohesive "
            "rebonding SLOWS growth relative to the matched zero-cohesion control. fold_slowdown = 10^(-S_h); "
            "pct_rate_reduction = (1 - 10^S_h) * 100 -- these are DIFFERENT numbers from the same S_h and must "
            "never be conflated as '{fold}x rate reduction'."
        ),
        "local_slopes_and_curvature": (
            "See px4_local_slope_curvature_table.csv: the log10(Kmax)-vs-S_h_developed relationship is strongly "
            "nonlinear for all 5 protocols -- local secant slope decays from ~11 (12-15 MPa*sqrt(m) interval) to "
            "~0.1-0.6 (21-24.3 MPa*sqrt(m) interval), with the largest curvature (second difference) concentrated "
            "at the low-Kmax end. This nonlinearity is why the PX3 screen (run only at the Kmax=18 MPa*sqrt(m) "
            "reference condition, already in the flat high-Kmax region) could not have detected the large "
            "low-Kmax effects PX4's expanded 5-point grid revealed."
        ),
        "load_ratio_result": {
            "screen_level_qualifier": "NEGATIVE_R_SENSITIVITY_SMALL_OVER_R_MINUS_0P95_TO_MINUS_0P50",
            "screen_R_panel_S_h_all": {r["R"]: float(r["S_h_all"]) for r in R_panel},
            "developed_level_result": px4_decision["interpretation"]["D1_vs_D2_R_sensitivity"],
            "interpretation": (
                "At the screen reference condition, S_h changes by only ~0.0006 between R=-0.95 and R=-0.5 "
                "(both compression-conditioned); R=+0.1 collapses the effect to ~0 (no compression phase, no "
                "rebonding possible). At the developed level (D1 vs D2, full Kmax grid), R remains "
                "practically indistinguishable (max|delta S_h|=0.0007)."
            ),
        },
        "frequency_result": {
            "qualifier": "FREQUENCY_SENSITIVITY_QUALIFIED",
            "screen_frequency_panel_S_h_all": {r["frequency_Hz"]: float(r["S_h_all"]) for r in freq_panel},
            "developed_level_result": px4_decision["interpretation"]["D2_vs_D3_frequency"],
            "interpretation": (
                "Screen level: S_h grows from ~-0.0011 at 100Hz to ~-0.0435 at 1000Hz (a ~38x change), then "
                "saturates from 1000Hz to 10000Hz (~-0.0435 vs ~-0.0435) -- frequency sensitivity is strong at "
                "low frequency and saturates above ~1000Hz at the reference Kmax. Developed level (D2 1000Hz vs "
                "D3 316.228Hz, full Kmax grid): frequency has a LARGE effect throughout, largest at low Kmax."
            ),
        },
        "dwell_result": {
            "qualifier": "DWELL_SENSITIVITY_BELOW_SCREEN_RESOLUTION_AT_REFERENCE_CONDITION",
            "invalidated_bug_history": {
                "classification": dwell_audit["classification"],
                "root_cause": dwell_audit["root_cause_identified_and_fixed"],
                "original_buggy_screen_S_h": dwell_audit["original_buggy_screen_S_h"],
                "n_invalidated_trajectories": len(invalidated["invalidated_trajectories"]),
                "invalidated_canonical_job_keys": [t["canonical_job_key"] for t in invalidated["invalidated_trajectories"]],
            },
            "corrected_audit_S_h": dwell_audit["corrected_audit_S_h_vs_matched_zero_cohesion"],
            "interpretation": (
                "The ORIGINAL screen dwell rows (hold=0.0005s, hold=0.002s) reported S_h=+0.355 and +0.799 -- a "
                "spurious, SIGN-REVERSED (positive, i.e. apparent acceleration) large effect caused by a real "
                "duration-weighting bug in _phase_statistics's cursor-rotated hazard sampling (invisible at "
                "hold=0, present whenever hold>0). Those two trajectories are formally INVALIDATED and must "
                "never be used in any scientific count or classification. The corrected dwell-causality audit "
                "(genuine dynamic + prescribed-static legs, both holds) found S_h in the same narrow "
                "~-0.0440-to-0.0442 range as hold=0's own -0.0435 -- i.e. dwell has NO measurable effect beyond "
                "the screen's own resolution once the bug is corrected (classification: "
                f"{dwell_audit['classification']})."
            ),
        },
        "passivation_chemistry_result": {
            "qualifiers": ["PASSIVATION_GATED_REBONDING_PHYSICALLY_EXERCISED", "PASSIVATION_EFFECT_ON_DEVELOPED_RATE_SMALL_FOR_SELECTED_ROW"],
            "screen_passivation_panel_S_h_all": {r["chemistry_factor"]: float(r["S_h_all"]) for r in passivation_panel},
            "developed_level_result": px4_decision["interpretation"]["D2_vs_D5_passivation"],
            "interpretation": (
                "Screen level: S_h magnitude decreases modestly as chemistry_factor decreases (chem=1.0: "
                "-0.0435, chem=0.3: -0.0379, chem=0.1: -0.0211). Developed level (D2 vs D5, the "
                "PASSIVATION_LIMITED row at chemistry_factor=1.0 selected via PX3.6's live duration-weighted "
                "mean-p_B requalification): the passivation-gated kinetic row is materially and physically "
                "exercised (mean_p_B~=0.317, nearest the 0.30 target while retaining mean_p_P>=0.30), yet "
                "produces an almost identical developed-regime slowdown to COMPETING_REVERSIBLE "
                "(max|delta S_h|=0.0001) -- passivation state does not materially change the STABILIZED "
                "developed rate for this selected row, though it may still govern the transient approach."
            ),
        },
        "reversible_vs_persistent_result": {
            "qualifiers": ["CLEAN_REVERSIBLE_KINETICS_QUALIFIED", "PERSISTENT_KINETICS_DISTINGUISHED_AT_SELECTED_FREQUENCY"],
            "screen_panel_S_h_all": {f"{r['row_name']}_f{r['frequency_Hz']}": float(r["S_h_all"]) for r in reversible_persistent_panel},
            "developed_level_result": px4_decision["interpretation"]["D3_vs_D6_persistence"],
            "interpretation": (
                "At the SCREEN reference condition (Kmax=18 MPa*sqrt(m), already in the flat high-Kmax "
                "region), COMPETING_PERSISTENT and COMPETING_REVERSIBLE are nearly indistinguishable "
                "(S_h~=-0.0436 vs -0.0435 at f=1000Hz; -0.0424 at f=100Hz). The developed campaign's expanded "
                "Kmax grid reveals the effect the screen could not: at D3/D6's shared reduced frequency "
                "(316.228Hz), persistence separates from reversibility by ~6.18-fold at the low-K endpoint "
                "(Kmax=12 MPa*sqrt(m)) -- NOT a full order of magnitude -- decaying to ~1.02-fold by Kmax=24.3. "
                "Persistent bonding materially amplifies the reduced-frequency low-K retardation, with the "
                "separation decaying strongly as Kmax increases."
            ),
        },
        "cohesive_strength_result": {
            "qualifier": "COHESIVE_STRENGTH_SENSITIVITY_QUALIFIED",
            "screen_cohesive_panel_S_h_all": {r["K_rebond_max_target_Pa_sqrt_m"]: float(r["S_h_all"]) for r in cohesive_panel},
            "interpretation": (
                "Monotonic, substantial sensitivity to the cohesive-strength ceiling at the screen reference "
                "condition: S_h=-0.0208 at K_target=450000, -0.0435 at 900000 (baseline), -0.0981 at 1800000 "
                "Pa*sqrt(m) -- roughly proportional to K_target over this range."
            ),
        },
        "seed_robustness": {
            "qualifier": "REBONDING_SEED_ROBUST",
            "by_protocol_axis": px6_synthesis["px4_findings"]["seed_robustness_by_protocol_axis"],
            "all_5_axes_robust": px6_synthesis["px4_findings"]["all_5_axes_seed_robust"],
        },
        "px5_static_vs_dynamic_attribution": {
            "scope": "D2 (COMPETING_REVERSIBLE) and D5 (PASSIVATION_LIMITED) only -- explicitly NOT extended to D1/D3/D6",
            "formal_classification": "MIXED_STATIC_SHIELDING_AND_KINETIC_HISTORY",
            "narrower_qualified_result": "CEILING_STATIC_SHIELD_EQUIVALENT_FOR_D2_D5_DEVELOPED_RATE",
            "interpretation": px5_decision["interpretation"],
            "classification_note": px5_decision["classification_note"],
        },
        "numerical_tolerances_and_censor_qualifications": {
            "developed_event_extension_contract": "exactly 30 accepted events, ~150um cumulative extension, whichever is not applicable is the binding constraint (all 80 admitted trajectories reached exactly 30 events AND exactly 150.000000um -- both bounds were reached simultaneously by construction of the campaign's own developed target)",
            "cycle_horizon": px6_synthesis["cycle_horizon_qualification"],
            "bulk_action_error_rel_tol": "1e-3 (certified numerical/action uncertainty basis, actively enforced by the adaptive block search)",
            "stable_growth_gate_thresholds": ">=10 events, >=50um final-window growth, late/early rate ratio in [0.5,2.0]",
        },
        "physical_vs_analytical_vs_postprocessing_distinction": (
            "PHYSICAL SIMULATION: every da/dN, S_h, and event-ledger quantity in PX3/PX4/PX5 comes from the "
            "real A_NATIVE production engine's fresh-process-per-job cyclic integration (part_x_run_one_job.py "
            "-> crack_rebonding_causal_pilot_v2_v10230.run_trajectory). ANALYTICAL PREDICTION: PX2's kinetic-"
            "regime atlas (mean_p_B, transition actions/fluxes, predicted_static_shield_equivalent_K_b) is "
            "computed from the phase-resolved P/C/B monodromy with ZERO fitting to any physical trajectory -- "
            "used only to SELECT candidate rows and to construct PX5's periodic-orbit-matched static control "
            "value, never substituted for a physical measurement. POST-PROCESSING: every rate/slope/curvature "
            "table, seed-robustness classification, and this final decision itself are pure reductions of "
            "already-physical or already-analytical tracked artifacts -- no new physics or analytical model "
            "evaluation happens in any final-closure script."
        ),
        "producer_launch_provenance": {
            "overall_decision": provenance_closure["overall_decision"],
            "n_producer_launch_pairs_closed": provenance_closure["n_producer_launch_pairs"],
            "n_diverged": provenance_closure["n_diverged"],
        },
        "component_verifier_results": {
            "px4_stage1_verification_classification": px4_stage1_verification["classification"],
            "px5_verification_classification": px5_verification["classification"],
        },
        "qualifiers": [
            "CLEAN_REVERSIBLE_KINETICS_QUALIFIED",
            "PERSISTENT_KINETICS_DISTINGUISHED_AT_SELECTED_FREQUENCY",
            "PASSIVATION_GATED_REBONDING_PHYSICALLY_EXERCISED",
            "PASSIVATION_EFFECT_ON_DEVELOPED_RATE_SMALL_FOR_SELECTED_ROW",
            "FREQUENCY_SENSITIVITY_QUALIFIED",
            "DWELL_SENSITIVITY_BELOW_SCREEN_RESOLUTION_AT_REFERENCE_CONDITION",
            "NEGATIVE_R_SENSITIVITY_SMALL_OVER_R_MINUS_0P95_TO_MINUS_0P50",
            "COHESIVE_STRENGTH_SENSITIVITY_QUALIFIED",
            "REBONDING_SEED_ROBUST",
            "MIXED_STATIC_SHIELDING_AND_KINETIC_HISTORY",
            "CEILING_STATIC_SHIELD_EQUIVALENT_FOR_D2_D5_DEVELOPED_RATE",
            "HARD_FIRST_PASSAGE_CYCLE_HORIZON_QUALIFIED_WITH_1E-6_CYCLE_FLOOR",
            "SURROGATE_SIGNED_K_CONTACT_NOT_RESOLVED_FACE_CONTACT",
            "HAZARD_ONLY_COHESIVE_FEEDBACK",
            "TOPOLOGICAL_HEALING_NOT_MODELED",
            "PHYSICAL_CHEMISTRY_PARAMETERIZATION_NOT_CALIBRATED",
        ],
        "explicit_scope_limitations": [
            "SURROGATE_SIGNED_K_CONTACT_NOT_RESOLVED_FACE_CONTACT: the contact model is a signed-K compression-"
            "conditioning proxy, never a resolved opposing-crack-face contact/traction solve.",
            "HAZARD_ONLY_COHESIVE_FEEDBACK: cohesion feeds back only through the cleavage hazard rate, never "
            "through a mechanical stiffness/compliance change.",
            "TOPOLOGICAL_HEALING_NOT_MODELED: no topological reconnection of crack faces is modeled.",
            "PHYSICAL_CHEMISTRY_PARAMETERIZATION_NOT_CALIBRATED: chemistry_factor and the bond/rupture/"
            "depassivation/repassivation barriers are analytically constructed (PX2), never calibrated against "
            "an experimental measurement.",
            "PX5 static-shield attribution covers ONLY D2 and D5 -- not extended to D1/D3/D6.",
            "PX5's S_h comparison covers only the STABLE/DEVELOPED window -- not the transient approach.",
        ],
        "not_claimed": [
            "closure-corrected DeltaK_eff", "resolved opposing-face contact", "topological crack healing",
            "calibrated physical chemistry", "production-line merge readiness", "predictive experimental calibration",
            "PX5 static-shield attribution for D1/D3/D6",
        ],
    }

    out_path = ARTIFACTS_DIR / "part_x_final_decision.json"
    out_path.write_text(json.dumps(decision, indent=2, default=str))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
