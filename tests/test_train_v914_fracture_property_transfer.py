import importlib.util
from pathlib import Path
import numpy as np,pandas as pd
P=Path(__file__).parents[1]/"scripts/train_v914_fracture_property_transfer.py";S=importlib.util.spec_from_file_location("m",P);M=importlib.util.module_from_spec(S);S.loader.exec_module(M)
def test_base300_preserves_surface_values():
 q=pd.DataFrame({"Tref_K":[481.33],"cleave_G00_eV":[4.],"cleave_gT_eV_per_K":[.003],"cleave_sigc0_GPa":[7.],"cleave_sT_GPa_per_K":[-.001],"emit_G00_eV":[3.],"emit_gT_eV_per_K":[-.002],"emit_sigc0_GPa":[6.],"emit_sT_GPa_per_K":[.001]});r=M.base300(q).iloc[0];assert np.isclose(r.cleave_G00_eV,4+.003*(300-481.33));assert np.isclose(r.emit_sigc0_GPa,6+.001*(300-481.33))
def test_ridge_transfer_cross_validation_is_finite():
 x=np.arange(60,dtype=float).reshape(20,3);y=2+x[:,0]-.5*x[:,1];r=M.train_one(x,y,np.arange(20));assert r["cv_R2"]>.99
