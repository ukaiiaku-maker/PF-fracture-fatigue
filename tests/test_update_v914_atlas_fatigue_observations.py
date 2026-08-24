import importlib.util
from pathlib import Path
import numpy as np,pandas as pd
P=Path(__file__).parents[1]/"scripts/update_v914_atlas_fatigue_observations.py";S=importlib.util.spec_from_file_location("u",P);M=importlib.util.module_from_spec(S);S.loader.exec_module(M)
def test_fit_preserves_complete_inert_class():
 f=pd.DataFrame({"base_candidate_id":["a"]*3,"restart_count":[0]*3,"developed_da_dN_m_per_cycle":[np.nan]*3,"DeltaK_MPa_sqrt_m":[10.8,16.2,21.6],"response_class":["inert/censored"]*3,"descriptor_usable":[False]*3})
 r=M.fits(f).iloc[0];assert r.candidate_response_class=="inert/censored";assert r.initial_descriptor_terminal_load_count==3;assert r.usable_descriptor_load_count==0
def test_two_positive_rates_produce_provisional_fit():
 f=pd.DataFrame({"base_candidate_id":["a"]*3,"restart_count":[0]*3,"developed_da_dN_m_per_cycle":[1e-9,4e-9,np.nan],"DeltaK_MPa_sqrt_m":[10.,20.,30.],"response_class":["sustained low-rate growth"]*2+["inert/censored"],"descriptor_usable":[True,True,False]})
 r=M.fits(f).iloc[0];assert np.isclose(r.provisional_m64,2);assert np.isclose(r.provisional_R2,1);assert r.usable_descriptor_load_count==2
def test_terminal_status_alone_does_not_gate_descriptor_usability():
 # A work_budget_exhausted / cycle_horizon_reached load with a positive
 # developed_da_dN_m_per_cycle must be able to carry descriptor_usable=True;
 # only restart contamination or insufficient post-initiation evidence
 # (enforced upstream by post_initiation_rate) may exclude it.
 f=pd.DataFrame({"base_candidate_id":["a"]*3,"restart_count":[0]*3,"developed_da_dN_m_per_cycle":[1e-9,2e-9,3e-9],"DeltaK_MPa_sqrt_m":[10.,15.,20.],"response_class":["sustained low-rate growth"]*3,"status":["work_budget_exhausted"]*3,"descriptor_usable":[True]*3})
 r=M.fits(f).iloc[0];assert r.usable_descriptor_load_count==3;assert np.isfinite(r.provisional_m64)
def test_resolve_base_candidate_id_is_identity_for_atlas_native_ids():
 assert M.resolve_base_candidate_id("V914_ATLAS_000055173",{})=="V914_ATLAS_000055173"
def test_resolve_base_candidate_id_uses_crosswalk_for_legacy_ids():
 crosswalk={"v914_large_analytical_0055172":"V914_ATLAS_000055173"}
 assert M.resolve_base_candidate_id("v914_large_analytical_0055172",crosswalk)=="V914_ATLAS_000055173"
def test_resolve_base_candidate_id_fails_closed_when_unmatched():
 # Historical A/B/C/D-style candidates have no exact atlas match and must
 # not be fabricated into a V914_ATLAS identity.
 assert M.resolve_base_candidate_id("v914_endurance_knee_0462",{})is None
