import json
from pathlib import Path

import numpy as np
import pandas as pd

from arrhenius_fracture.physical_slope_transfer_v10230 import (
    interval_log_slope,
    local_log_slopes,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/physical_slope_transfer_v1"


def test_log_slope_recovers_an_exact_power():
    x=np.array([1.,2.,3.,5.,8.])
    assert np.allclose(local_log_slopes(x,3.7*x**4.25),4.25,rtol=0,atol=1e-12)
    assert abs(interval_log_slope(2,8,7*2**3,7*8**3)-3)<1e-12


def test_archive_audit_fails_closed_on_missing_phase_state():
    audit=pd.read_csv(OUT/"archive_resolution_audit.csv")
    missing=audit[(audit.resolution=="cycle/phase")&audit.quantity.isin(["K_shield","r_eff","cleavage_stress"])]
    assert len(missing)==3
    assert missing.availability.eq("NOT_ARCHIVED").all()
    status=pd.read_csv(OUT/"counterfactual_replay_status.csv").set_index("replay")
    assert status.loc[["T1","T2","T3"],"status"].eq("NOT_EXACTLY_IDENTIFIABLE").all()
    assert status.loc["T4","status"]=="EXACT_ARCHIVED_IDENTITY"


def test_all_21_trajectories_and_378_events_are_reconstructed():
    events=pd.read_parquet(OUT/"archived_event_stage_history.parquet")
    assert len(events)==378
    assert len(events[["option_key","Kmax_MPa_sqrt_m"]].drop_duplicates())==21
    assert events.groupby(["option_key","Kmax_MPa_sqrt_m"]).size().eq(18).all()
    assert not events.event_phase_archived.any()
    assert events.archived_raw_barrier_eV.isna().all()
    assert events.archived_raw_cleavage_rate_s.isna().all()
    assert events.csv_raw_barrier_is_zero_placeholder.all()
    assert events.csv_shield_is_zero_placeholder.all()


def test_event_transactions_and_common_random_numbers_are_preserved():
    events=pd.read_parquet(OUT/"archived_event_stage_history.parquet")
    assert events.geometry_commit_inserted.all()
    assert events.first_passage_action_closes_to_threshold.all()
    assert events.proposal_matches_gate.all()
    assert events.gate_matches_path_commit.all()
    thresholds=events.groupby(["option_key","Kmax_MPa_sqrt_m"]).first_passage_threshold.apply(tuple)
    rewards=events.groupby(["option_key","Kmax_MPa_sqrt_m"]).proposed_event_length_m.apply(tuple)
    assert len(set(thresholds))==1
    assert len(set(rewards))==1


def test_transfer_is_cross_target_universal_on_predeclared_gate():
    decision=json.loads((OUT/"physical_transfer_operator.json").read_text())
    fit=pd.read_csv(OUT/"cross_target_transfer_fit.csv")
    assert decision["universal_acceptance"] is True
    assert fit.A_phys_spread.max()<=.03
    assert np.sqrt(np.mean(fit.M4_heldout_residual**2))<=.05
    assert fit.M4_heldout_residual.abs().max()<.02


def test_flattening_starts_in_K_to_sigma_mapping():
    points=pd.read_csv(OUT/"stagewise_slope_transmission.csv")
    m4=points[points.M_target==4]
    assert m4.m_K_to_sigma.iloc[0]<1
    assert m4.m_K_to_sigma.is_monotonic_decreasing
    assert m4.m_K_to_sigma.iloc[-1]<.45
    assert np.max(abs(m4.m_archived_action_per_cycle-m4.m_T4_archived_identity_da_dN))<1e-10
    assert np.max(abs(m4.m_accepted_to_da_dN-1))<1e-10


def test_radius_freeze_restores_A0_slope_and_shielding_is_negligible():
    points=pd.read_csv(OUT/"stagewise_slope_transmission.csv")
    assert np.max(abs(points.m_diagnostic_freeze_r_eff_da_dN-points.m_T0_exact_A0_da_dN))<6e-4
    decision=json.loads((OUT/"physical_transfer_operator.json").read_text())
    assert decision["maximum_absolute_shield_fraction"]<3e-5


def test_energy_gate_and_common_event_reward_add_no_slope_attenuation():
    points=pd.read_csv(OUT/"stagewise_slope_transmission.csv")
    assert np.max(abs(points.m_diagnostic_remove_energy_truncation_da_dN-points.m_T4_archived_identity_da_dN))<1e-10
    assert np.max(abs(points.m_mean_accepted_event_length_m))<1e-10


def test_diagnostic_fit_is_not_a_production_constitutive_law():
    module=(ROOT/"arrhenius_fracture/physical_slope_transfer_v10230.py").read_text()
    decision=json.loads((OUT/"physical_transfer_operator.json").read_text())
    assert "never instantiates or modifies the production solver" in module
    assert decision["diagnostic_fit_is_constitutive_law"] is False


def test_seed_predictions_are_frozen_before_launch():
    freeze=json.loads((OUT/"transfer_freeze.json").read_text())
    predictions=pd.read_csv(OUT/"second_seed_prospective_predictions.csv")
    assert freeze["new_physics_runs_before_freeze"]==0
    assert freeze["physics_launch_utc"] is not None
    assert len(predictions)==2
    assert predictions.seed.eq(1001723).all()


def test_second_seed_launcher_is_fresh_exact_and_freeze_gated():
    source=(ROOT/"scripts/run_v10_2_30_physical_slope_transfer.py").read_text()
    assert 'V10230_HIGH_CYCLE_EXPLICIT_ONLY":"1"' in source
    assert '"seed":1001723' in source
    assert '"HAZARD_SEED":str(job["seed"])' in source
    assert 'env.pop("V10230_RESTART_CHECKPOINT_DIR",None)' in source
    assert 'freeze["analysis_artifact_hashes"]' in source
    assert 'if git("status","--porcelain")' in source


def test_second_seed_prospectively_validates_frozen_transfer_without_refit():
    result=json.loads((OUT/"second_seed_transfer_decision.json").read_text())
    validation=pd.read_csv(OUT/"second_seed_transfer_validation.csv")
    physical=pd.read_csv(OUT/"second_seed_physical_results.csv")
    assert result["result"]=="PASS"
    assert result["operator_refit_performed"] is False
    assert result["maximum_absolute_A_interval_difference"]<.034
    assert validation.acceptance_pass.all()
    assert not validation.operator_refit_performed.any()
    assert len(physical)==3 and physical.target_reached.all()
    assert physical.fresh.all() and not physical.resume_environment_present.any()
    assert physical.seed.eq(1001723).all()


def test_corrected_candidate_changes_only_the_five_cleavage_surface_fields():
    projection=json.loads((OUT/"corrected_candidate_projection.json").read_text())
    audit=pd.read_csv(OUT/"corrected_candidate_diff_audit.csv")
    expected={"cleave_G00_eV","cleave_sigc0_GPa","cleave_exp_a","cleave_exp_n","cleave_floor_frac"}
    assert set(projection["changed_constitutive_fields"])==expected
    assert not audit.unexpected_change.any()
    assert audit[audit.field.isin(expected)].changed.all()
    assert projection["noncleavage_common_physics_unchanged"] is True


def test_corrected_inverse_uses_reduced_operator_and_keeps_fit_diagnostic():
    design=pd.read_csv(OUT/"corrected_candidate_prospective_predictions.csv")
    projection=json.loads((OUT/"corrected_candidate_projection.json").read_text())
    assert len(design)==7
    assert np.allclose(design.reduced_operator_required_A0_slope,
                       4/design.frozen_m_K_to_sigma,rtol=0,atol=1e-12)
    assert "reduced physical operator" in projection["construction_operator"]
    assert "not used" in projection["empirical_fit_role"]
    assert design.reduced_operator_required_A0_slope.is_monotonic_increasing
    assert projection["predicted_reduced_physical_slope_RMSE"]<=.5


def test_corrected_predictions_are_frozen_before_physical_launch():
    freeze=json.loads((OUT/"corrected_candidate_freeze.json").read_text())
    assert freeze["new_corrected_physics_runs_before_freeze"]==0
    assert freeze["physics_launch_utc"] is not None
    assert freeze["operator_refit_after_second_seed"] is False
    assert freeze["loads_MPa_sqrt_m"]==[12.,12.75,13.5,15.,18.,21.,24.3]
    assert freeze["seed"]==1720 and freeze["n_bins"]==80


def test_seven_corrected_physical_trajectories_are_terminal_fresh_and_uncensored():
    points=pd.read_csv(OUT/"corrected_candidate_physical_points.csv",keep_default_na=False)
    state=json.loads((OUT/"corrected_candidate_controller_state.json").read_text())
    assert len(points)==7
    assert points.target_reached.all()
    assert points.fresh.all() and not points.resume.any()
    assert not points.restart_environment_present.any()
    assert points.censor_or_failure_reason.eq("").all()
    assert points.seed.eq(1720).all() and points.n_bins.eq(80).all()
    assert state["phase"]=="CORRECTED_TERMINAL" and state["active_worker_count"]==0


def test_final_decision_matches_frozen_acceptance_and_has_all_answers():
    decision=json.loads((OUT/"physical_slope_transfer_final_decision.json").read_text())
    report=(OUT/"physical_slope_transfer_final_decision.md").read_text()
    comparison=pd.read_csv(OUT/"corrected_candidate_prediction_comparison.csv")
    assert decision["result"]=="PASS"
    assert decision["primary_classification"] in report
    assert decision["all_terminal_uncensored"] is True
    assert decision["all_fresh_without_resume"] is True
    assert decision["operator_refit_after_second_seed"] is False
    assert len(comparison)==7 and np.allclose(comparison.Kmax_MPa_sqrt_m,
        [12.,12.75,13.5,15.,18.,21.,24.3])
    assert all(f"{number}. **" in report for number in range(1,11))
    assert (OUT/"figures"/"PHYSICAL_SLOPE_TRANSFER_ALL_DATA.png").stat().st_size>10000
