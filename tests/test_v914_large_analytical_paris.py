import numpy as np
import pandas as pd
from scripts.search_v914_large_analytical_paris import best_window, transform, COORDINATES, cleavage_extension_bounds


def test_transform_covers_exact_22d_historical_domain():
    d=transform(np.vstack([np.zeros(22),np.ones(22)]))
    assert len(COORDINATES)==22
    assert all(f"u__{x}" in d for x in COORDINATES)
    assert d.cleave_G00_eV.tolist()==[1.0,5.5]


def test_best_window_searches_contiguous_feasible_subsets():
    cfg={"minimum_Paris_points":4,"minimum_relative_DeltaK_span":.2,"minimum_fit_R2":.9,"preferred_da_dN_span_decades":1.,"target_m_min":2.,"target_m_max":10.}
    k=np.array([8.,10.,12.,15.,18.,22.]); y=-12+4*np.log10(k/8)
    got=best_window(k,y,np.array([False,True,True,True,True,True]),cfg)
    assert got["n_points"]==5
    assert 3.99 < got["m"] < 4.01
    assert got["DeltaK_min"]==10.


def test_cleavage_extension_uses_only_valid_shell_rows(tmp_path):
    rows=[]
    for valid,shift in ((True,0.),(True,1.),(False,100.)):
        row={name:float(i+1+shift) for i,name in enumerate(COORDINATES[:5])}; row["barrier_surface_valid"]=valid; rows.append(row)
    path=tmp_path/"shell.csv"; pd.DataFrame(rows).to_csv(path,index=False)
    got=cleavage_extension_bounds(path)
    assert got["cleave_G00_eV"][:2]==(1.,2.)
