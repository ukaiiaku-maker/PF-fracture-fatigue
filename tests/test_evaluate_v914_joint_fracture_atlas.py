import importlib.util
from pathlib import Path
import numpy as np,pandas as pd
P=Path(__file__).parents[1]/"scripts/evaluate_v914_joint_fracture_atlas.py";S=importlib.util.spec_from_file_location("e",P);M=importlib.util.module_from_spec(S);S.loader.exec_module(M)
def test_materialization_preserves_300K_base():
 q=pd.DataFrame({"joint_sequence_index":[0],"joint_candidate_id":["j"],"Tref_K":[481.33],"cleave_G00_eV":[4.],"cleave_gT_eV_per_K":[.003],"cleave_sigc0_GPa":[7.],"cleave_sT_GPa_per_K":[-.001],"emit_G00_eV":[3.],"emit_gT_eV_per_K":[-.002],"emit_sigc0_GPa":[6.],"emit_sT_GPa_per_K":[.001]});r=M.materialize(q).iloc[0];assert np.isclose(r.cleave_G00_eV+r.cleave_gT_eV_per_K*(300-r.Tref_K),4.);assert np.isclose(r.emit_sigc0_GPa+r.emit_sT_GPa_per_K*(300-r.Tref_K),6.)
