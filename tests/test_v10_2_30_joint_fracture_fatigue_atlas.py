import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/joint_fracture_fatigue_archetype_atlas_v1"


def test_source_hashes_and_explained_unavailable_documents():
    manifest = json.loads((OUT / "fracture_source_manifest.json").read_text())
    for source in manifest["sources"]:
        if source["available"]:
            path = Path(source["path"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
        else:
            assert source["omission_explanation"]


def test_no_duplicate_or_nonphysical_physical_admission():
    data = pd.read_csv(OUT / "fracture_result_inventory.csv")
    assert not data.inventory_id.duplicated().any()
    bad = data.completion_state.isin([
        "RIGHT_CENSORED", "NUMERICAL_FAILURE", "MECHANICAL_DOMAIN_LIMITATION",
        "FAILURE_MODE_PREEMPTION",
    ])
    assert not data.loc[bad, "physical_admission"].fillna(False).any()


def test_exact_canonical_and_named_candidate_recovery():
    data = pd.read_csv(OUT / "candidate_cross_lineage_registry.csv")
    assert {
        "v913_zeroD_sobol_0242980", "v913_zeroD_sobol_0202500",
        "v913_zeroD_sobol_0129902", "v913_zeroD_sobol_0077080",
    }.issubset(set(data.candidate_id))
    assert {
        "ceramic_primary", "weakT_primary", "dbtt_primary", "peak_primary",
        "dbtt_broad_shielding", "dbtt_intrinsic_control",
        "dbtt_moderate_shielding_reference",
    }.issubset(set(data.registry_role))
    assert data.row_sha256.str.fullmatch(r"[0-9a-f]{64}").all()


def test_current_and_legacy_lineages_are_separated_and_inactive_legacy_absent():
    lineages = pd.read_csv(OUT / "constitutive_lineage_matrix.csv")
    legacy = lineages[lineages.lineage.str.startswith("LEGACY_")]
    assert len(legacy) >= 2
    assert not legacy.used_in_current_analytical_model.any()
    equations = pd.read_csv(OUT / "equation_source_catalog.csv")
    assert not equations[equations.classification == "LEGACY_INACTIVE"].active.any()


def test_first_passage_censor_and_state_semantics_are_explicit():
    state = pd.read_parquet(OUT / "monotonic_state_validation.parquet")
    assert set(state.state_semantics) == {"PRE_EVENT_PRE_TRANSLATION"}
    for level in ("F0", "F1", "F2"):
        data = pd.read_parquet(OUT / f"monotonic_{level}_predictions.parquet")
        assert ((data.first_passage_reached & ~data.right_censored)
                | (~data.first_passage_reached & data.right_censored)).all()
        assert (data.loc[data.first_passage_reached, "cleavage_action"] == 1.0).all()


def test_endpoint_and_tangent_validation_are_recorded():
    f0 = pd.read_parquet(OUT / "monotonic_F0_predictions.parquet")
    localized = f0[f0.hazard_localization >= 3]
    assert len(localized) > 0
    assert localized.endpoint_relative_error.notna().all()
    temp = pd.read_parquet(OUT / "temperature_sensitivity_decomposition.parquet")
    assert (temp.relative_closure_error.dropna() < .2).mean() > .8
    jac = pd.read_parquet(OUT / "parameter_sensitivity_jacobian.parquet")
    assert set(jac.parameter) == {
        "cleave_G00_eV", "cleave_gT_eV_per_K", "cleave_sigc0_GPa",
        "cleave_sT_GPa_per_K", "cleave_exp_a", "cleave_exp_n",
        "cleave_floor_frac",
    }
    assert (jac.relative_error.dropna() < .25).mean() > .75


def test_historical_bounds_and_sobol_are_reproducible():
    bounds = json.loads((OUT / "historical_search_bounds.json").read_text())
    assert bounds["continuation_of_historical_search"]
    assert len(bounds["historical_search_dimensions"]) == 26
    atlas = pd.read_parquet(OUT / "response_atlas.parquet")
    assert len(atlas) == 147456
    assert (atlas.sampling_stage == "INITIAL_SOBOL").sum() == 131072
    assert (atlas.sampling_stage == "ADAPTIVE_CLASS_BOUNDARY").sum() == 16384
    assert atlas.atlas_id.is_unique


def test_atlas_rejection_classification_and_unvalidated_extrapolation():
    atlas = pd.read_parquet(OUT / "response_atlas.parquet")
    rejection = pd.read_csv(OUT / "atlas_rejection_audit.csv")
    assert set(rejection.atlas_id) == set(atlas.loc[
        atlas.barrier_or_state_asymptotic_artifact, "atlas_id"
    ])
    assert set(atlas.fatigue_temperature_status) == {
        "ANALYTICAL_EXTRAPOLATION_UNVALIDATED"
    }


def test_response_classes_and_peak_detection_are_reproducible():
    atlas = pd.read_parquet(OUT / "response_atlas.parquet")
    assert set(atlas.response_class) == {
        "CERAMIC_LIKE", "WEAK_T", "DBTT_LIKE", "PEAK_LIKE", "MIXED_OR_UNRESOLVED"
    }
    peak = atlas.response_class == "PEAK_LIKE"
    assert (atlas.loc[peak, "peak_temperature_K"].isin([600,700,900,1100])).all()
    assert (atlas.loc[peak, "peak_amplitude_MPa_sqrt_m"] > 0).all()


def test_cluster_stability_and_mechanism_nonuniqueness():
    stability = pd.read_csv(OUT / "archetype_cluster_stability.csv")
    assert stability.stable_under_resampling.mean() >= .75
    manifolds = pd.read_parquet(OUT / "archetype_nonuniqueness_manifolds.parquet")
    assert manifolds.groupby("response_class").mechanism_class.nunique().max() > 1


def test_same_complete_candidate_row_and_prediction_freeze():
    audit = pd.read_csv(OUT / "archetype_candidate_diff_audit.csv")
    assert audit.changed_fields_equal_declared.all()
    assert audit.same_complete_row_for_fracture_and_fatigue.all()
    plan = json.loads((OUT / "prospective_validation_plan.json").read_text())
    assert plan["fresh_uninterrupted_only"] and not plan["resume_allowed"]
    assert (OUT / "prospective_monotonic_predictions.csv").stat().st_mtime_ns <= (
        OUT / "physical_validation_controller_state.json"
    ).stat().st_mtime_ns


def test_no_resumed_duplicate_or_unclassified_physical_trajectory():
    for name in ("physical_monotonic_validation.csv", "physical_fatigue_validation.csv"):
        data = pd.read_csv(OUT / name)
        assert not data.resumed.any()
        assert not data.new_trajectory.any()
    controller = json.loads((OUT / "physical_validation_controller_state.json").read_text())
    assert controller["state"] == "TERMINAL"
    assert controller["new_jobs_launched"] == 0


def test_markdown_json_decision_agreement():
    decision = json.loads((OUT / "joint_archetype_final_decision.json").read_text())
    report = (OUT / "joint_archetype_final_decision.md").read_text()
    assert decision["primary_classification"] in report
    assert decision["markdown_json_classification_agreement_key"] in report
    assert len(decision["answers"]) == 29
