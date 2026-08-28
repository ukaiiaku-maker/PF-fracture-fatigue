import json
from pathlib import Path

import pandas as pd

from scripts.run_v10_2_30_inverse_fatigue_barrier_validation import (
    prephysics_infrastructure_failure,
)


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/inverse_fatigue_barrier_design_v1"


def test_tracks_enforce_zero_and_one_point_archived_access():
    data=json.loads((OUT/"target_design_manifest.json").read_text())
    assert data["archived_rate_access"]["PURE_SYNTHETIC"]==0
    assert data["archived_rate_access"]["REFERENCE_SCALE_ANCHORED"]==1
    assert data["archived_rate_access"]["archived_slope_access"]==0


def test_A2_rate_equality_is_explicit_not_physical_PT_evidence():
    source=json.loads((OUT/"inverse_design_source_manifest.json").read_text())
    assert "g_A2_equals_g_A1_by_construction" in source["A2_rate_semantics"]


def test_exact_inverse_is_saturation_compatible_and_roundtrips():
    exact=pd.read_parquet(OUT/"exact_inverse_barriers.parquet")
    assert exact.saturation_compatible.all()
    gamma=pd.read_csv(OUT/"cooperative_gamma_inverse_validation.csv")
    assert gamma.relative_error.max()<=1e-8


def test_three_candidates_change_only_declared_cleavage_fields():
    audit=pd.read_csv(OUT/"inverse_design_candidate_diff_audit.csv")
    assert len(audit)==3
    assert audit.changed_fields_equal_declared.all()
    assert audit.noncleavage_physics_unchanged.all()


def test_predictions_were_frozen_before_physical_launch():
    freeze=json.loads((OUT/"prospective_prediction_freeze.json").read_text())
    assert freeze["target_frozen_utc"]<=freeze["candidate_frozen_utc"]
    assert freeze["candidate_frozen_utc"]<=freeze["prospective_predictions_frozen_utc"]
    if freeze["physical_launch_utc"] is not None:
        assert freeze["prospective_predictions_frozen_utc"]<=freeze["physical_launch_utc"]


def test_no_empirical_Paris_term_was_added_to_production_solver():
    module=(ROOT/"arrhenius_fracture/inverse_fatigue_barrier_design_v10230.py").read_text()
    assert "analysis-only" in module
    source=json.loads((OUT/"inverse_design_source_manifest.json").read_text())
    assert source["no_physics_modified_by_inverse_analysis"] is True


def test_final_physical_jobs_are_fresh_unique_and_not_resumed():
    path=OUT/"inverse_design_physical_points.csv"
    if not path.exists():return
    data=pd.read_csv(path)
    assert data.result_path.nunique()==len(data)
    assert data.fresh.all()
    assert not data.resume.astype(bool).any()
    assert not data.resume_used.astype(bool).any()


def test_markdown_and_json_decisions_agree_when_finalized():
    path=OUT/"inverse_design_final_decision.json"
    if not path.exists():return
    decision=json.loads(path.read_text())
    markdown=(OUT/"inverse_design_final_decision.md").read_text()
    assert decision["primary_classification"] in markdown
    assert "g_A2 = g_A1" in markdown


def test_prephysics_launch_failure_is_not_a_numerical_trajectory(tmp_path):
    (tmp_path/"run.log").write_text("")
    (tmp_path/"high_cycle_run_manifest.json").write_text("{}")
    (tmp_path/"high_cycle_summary.json").write_text("{}")
    assert prephysics_infrastructure_failure(tmp_path)
    (tmp_path/"kinetic_tip_cell_audit_v101.json").write_text("{}")
    assert not prephysics_infrastructure_failure(tmp_path)


def test_R_reference_launch_is_gated_by_addendum_freeze():
    source=(ROOT/"scripts/run_v10_2_30_inverse_fatigue_barrier_validation.py").read_text()
    assert 'if stage == "R_REFERENCE":' in source
    assert "addendum_preflight(head)" in source
    assert 'freeze["material_barrier_R_invariant"]' in source


def test_multi_R_targets_keep_R_out_of_material_barriers():
    data=json.loads((OUT/"multi_R_target_design_configurations.json").read_text())
    assert data["common"]["R_design_values"]==[.1]
    assert data["common"]["R_prediction_values"]==[-.95,.5]
    assert data["common"]["comparison_mode"]=="BOTH"
    assert data["common"]["negative_emission_branch_enabled"] is False
    assert data["common"]["negative_fracture_branch_enabled"] is False


def test_multi_R_derivative_identity_and_barrier_hashes():
    identity=pd.read_csv(OUT/"R_fixed_Kmax_vs_fixed_deltaK.csv")
    identity=identity[identity.comparison_mode=="DERIVATIVE_IDENTITY"]
    assert identity.closure_residual.abs().max()<2e-6
    cross=pd.read_csv(OUT/"cross_R_inverse_barrier_consistency.csv")
    assert (cross.barrier_parameter_hash_i==cross.barrier_parameter_hash_j).all()
    assert cross.E_RR_max_barrier_eV.max()==0


def test_reverse_transport_is_not_reverse_emission():
    audit=pd.read_csv(OUT/"reverse_transport_vs_reverse_emission_audit.csv")
    assert not audit.double_count_detected.any()
    assert not audit.negative_emission_branch_enabled.any()
    assert audit.loc[audit.R<0,"reverse_transport_fraction"].min()>0
