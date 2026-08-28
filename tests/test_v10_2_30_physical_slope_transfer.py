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
    assert freeze["physics_launch_utc"] is None
    assert len(predictions)==2
    assert predictions.seed.eq(1001723).all()


def test_second_seed_launcher_is_fresh_exact_and_freeze_gated():
    source=(ROOT/"scripts/run_v10_2_30_physical_slope_transfer.py").read_text()
    assert 'V10230_HIGH_CYCLE_EXPLICIT_ONLY":"1"' in source
    assert '"HAZARD_SEED":"1001723"' in source
    assert 'env.pop("V10230_RESTART_CHECKPOINT_DIR",None)' in source
    assert 'freeze["analysis_artifact_hashes"]' in source
    assert 'if git("status","--porcelain")' in source
