from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts/analyze_v913_barrier_temperature_fracture_morphology.py"
SOURCE=Path("/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v9_13_dbtt_temperature_shelf")
spec=importlib.util.spec_from_file_location("v913_morphology",SCRIPT)
module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module
assert spec.loader is not None; spec.loader.exec_module(module)


def test_exact_historical_surface_and_entropy_sign():
    ExpFloorSurface,_=module.load_production_types(SOURCE)
    candidates,_,_,paths=module.load_population(SOURCE)
    row=candidates[candidates.candidate_id.eq("v913_zeroD_sobol_0202500")].iloc[0]
    surface=module.make_surface(row,"cleave",ExpFloorSurface)
    assert np.isfinite(surface.barrier_eV(5e9,900.0))
    assert np.isclose(-row.cleave_gT_eV_per_K/module.KB_EV_K,
                      -float(row.cleave_gT_eV_per_K)/module.KB_EV_K,rtol=0,atol=0)
    computed=module.validate_canonical_selection(candidates,paths["selection"],paths["registry"])
    assert len(computed)==64


def test_entropy_and_stress_scale_counterfactuals_are_distinct():
    ExpFloorSurface,_=module.load_production_types(SOURCE)
    candidates,_,_,_=module.load_population(SOURCE)
    row=candidates[candidates.candidate_id.eq("v913_zeroD_sobol_0242980")].iloc[0]
    full=module.make_surface(row,"cleave",ExpFloorSurface)
    no_g=module.make_surface(row,"cleave",ExpFloorSurface,zero_gT=True)
    no_s=module.make_surface(row,"cleave",ExpFloorSurface,zero_sT=True)
    def slope(s): return (s.barrier_eV(5e9,901)-s.barrier_eV(5e9,899))/2
    assert not np.isclose(slope(full),slope(no_g),rtol=0,atol=1e-10)
    assert not np.isclose(slope(full),slope(no_s),rtol=0,atol=1e-10)


def test_response_builder_never_uses_censor_or_extrapolates_300K():
    cases=pd.DataFrame([
        {"candidate_id":"x","temperature_K":500,"status":"complete","authoritative_response_MPa_sqrt_m":10,"response_target_um":50,"source_dataset":"test"},
        {"candidate_id":"x","temperature_K":700,"status":"complete","authoritative_response_MPa_sqrt_m":11,"response_target_um":50,"source_dataset":"test"},
        {"candidate_id":"x","temperature_K":900,"status":"complete","authoritative_response_MPa_sqrt_m":12,"response_target_um":50,"source_dataset":"test"},
        {"candidate_id":"x","temperature_K":1100,"status":"right_censored_maximum_substeps","authoritative_response_MPa_sqrt_m":999,"response_target_um":50,"source_dataset":"test"},
    ])
    desc,points=module.response_descriptors(cases)
    assert len(points)==3 and 999 not in points.K_response_MPa_sqrt_m.to_list()
    assert np.isnan(desc.iloc[0].K_300_MPa_sqrt_m)
    assert desc.iloc[0].temperature_max_K==900


def test_fdr_and_small_n_fail_closed():
    values=pd.Series([.01,.02,np.nan,.5])
    q=module.fdr_bh(values)
    assert np.isnan(q[2]) and np.all(q[[0,1,3]]>=values[[0,1,3]])
    frame=pd.DataFrame({"candidate_id":["a","b"],"S_low_MPa_sqrt_m_per_K":[1,2],"cleavage_entropy_kB":[2,3],
                        "canonical_family":[None,None],"historical_response_class":["x","x"],"is_canonical_holdout":[False,False]})
    corr,_=module.correlation_tables(frame)
    row=corr[(corr.subset.eq("DISCOVERY_NONCANONICAL"))&corr.predictor.eq("cleavage_entropy_kB")].iloc[0]
    assert row.test_status=="INSUFFICIENT_N_OR_VARIATION" and np.isnan(row.spearman_p)


def test_scatter_keeps_unlabeled_discovery_population():
    import matplotlib.pyplot as plt
    frame=pd.DataFrame({"candidate_id":["a","b","c"],"canonical_family":[None,None,"DBTT"],"x":[1,2,3],"y":[2,3,4]})
    fig,ax=plt.subplots(); plotted=module.scatter(ax,frame,"x","y")
    assert len(plotted)==3
    assert sum(len(collection.get_offsets()) for collection in ax.collections)==3
    plt.close(fig)
