import importlib.util
from pathlib import Path
import numpy as np,pandas as pd
P=Path(__file__).parents[1]/"scripts/update_v914_atlas_fatigue_observations.py";S=importlib.util.spec_from_file_location("u",P);M=importlib.util.module_from_spec(S);S.loader.exec_module(M)
def test_fit_preserves_complete_inert_class():
 f=pd.DataFrame({"candidate_id":["a"]*3,"restart_count":[0]*3,"developed_da_dN_m_per_cycle":[np.nan]*3,"DeltaK_MPa_sqrt_m":[10.8,16.2,21.6],"response_class":["inert/censored"]*3})
 r=M.fits(f).iloc[0];assert r.candidate_response_class=="inert/censored";assert r.initial_descriptor_terminal_load_count==3
def test_two_positive_rates_produce_provisional_fit():
 f=pd.DataFrame({"candidate_id":["a"]*3,"restart_count":[0]*3,"developed_da_dN_m_per_cycle":[1e-9,4e-9,np.nan],"DeltaK_MPa_sqrt_m":[10.,20.,30.],"response_class":["sustained low-rate growth"]*2+["inert/censored"]})
 r=M.fits(f).iloc[0];assert np.isclose(r.provisional_m64,2);assert np.isclose(r.provisional_R2,1)
