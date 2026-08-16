from __future__ import annotations
import importlib.util,sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; path=ROOT/"scripts/analyze_v913_expanded_barrier_temperature_morphology.py"
spec=importlib.util.spec_from_file_location("v913_expanded",path); m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m)

def test_normalized_transition_geometry_is_ordered_and_exact():
    v1=m.load_v1_module(); E,_=v1.load_production_types(m.SOURCE); candidates,_,_,_=v1.load_population(m.SOURCE)
    row=candidates[candidates.candidate_id.eq("v913_zeroD_sobol_0202500")].iloc[0]; surface=v1.make_surface(row,"cleave",E); K=np.linspace(0,200,2001)
    *_,phi,_,_=m.surface_arrays(surface,900.,K); positions=[m.level_position(K,phi,x) for x in m.LEVELS]
    assert positions==sorted(positions)
    for level,pos in zip(m.LEVELS,positions): assert np.isclose(np.interp(pos,K,phi),level,atol=2e-3)

def test_dimensionless_and_serial_timescale_descriptors_are_finite():
    v1=m.load_v1_module(); E,P=v1.load_production_types(m.SOURCE); candidates,cases,events,_=v1.load_population(m.SOURCE)
    case=cases[(cases.candidate_id.eq("v913_zeroD_sobol_0202500"))&cases.temperature_K.eq(900)].iloc[0]
    state=events[(events.candidate_id.eq(case.candidate_id))&events.temperature_K.eq(900)&events.event_index.eq(0)].iloc[0]
    rec=m.barrier_geometry_row(case.candidate_id,900,float(case.authoritative_response_MPa_sqrt_m),candidates[candidates.candidate_id.eq(case.candidate_id)].iloc[0],state,v1,E,P)
    for key in ["delta_g_ec_at_K075","log10_tau_c_over_tau_p_at_K075","delta_K50_MPa_sqrt_m","relative_curvature_at_closest"]: assert np.isfinite(rec[key])
    assert rec["frozen_path_proxy_not_evolved"] is True
    assert np.isnan(rec["state_K_shield_over_K_applied"])

def test_mutual_information_detects_nonmonotonic_signal():
    rng=np.random.default_rng(4); x=np.linspace(-1,1,396); y=x*x+rng.normal(0,.02,len(x))
    frame=pd.DataFrame({"candidate_id":[str(i) for i in range(400)],"x":np.r_[x,[0,0,0,0]],"y":np.r_[y,[0,0,0,0]],"is_canonical_holdout":[False]*396+[True]*4})
    table=m.mutual_information_table(frame,["x"],["y"],seed=3)
    assert abs(pd.Series(x).corr(pd.Series(y),method="spearman"))<.05
    assert table.iloc[0].bias_corrected_MI_nats>.5 and table.iloc[0].permutation_p<.05

def test_shallow_tree_depth_is_bounded():
    x=np.arange(120,dtype=float); X=np.column_stack([x,np.sin(x)]); y=np.where(x<40,"a",np.where(x<80,"b","c")); tree=m.build_tree(X,y,["x","s"],max_depth=3,min_leaf=10)
    def depth(node): return 0 if "feature" not in node else 1+max(depth(node["left"]),depth(node["right"]))
    assert depth(tree)<=3

def test_feature_family_routing_prevents_name_based_misclassification():
    assert m.feature_family("log10_rate_ratio_at_K75__span_T")=="KINETIC_TIMESCALE_COMPETITION"
    assert m.feature_family("state_front_width_dT_m_per_K")=="EVOLVED_STATE_PROXY"
    assert m.feature_family("delta_K50_MPa_sqrt_m")=="RELATIVE_BARRIER_POSITION"

def test_barrier_and_rate_topology_use_opposite_positive_sign_conventions():
    positive=np.array([2.,3.,4.])
    assert m.topology(positive,0)=="NO_CROSSING_CLEAVAGE_EASIER"
    assert m.topology(positive,0,positive_means_cleavage=False)=="NO_CROSSING_EMISSION_EASIER"

def test_phase_axis_guard_rejects_singular_tail_without_altering_values():
    ordinary=pd.Series(np.r_[np.linspace(-2,2,399),5.])
    singular=ordinary.copy(); singular.iloc[-1]=1e12
    original=singular.copy()
    assert m.robust_phase_axis(ordinary)
    assert not m.robust_phase_axis(singular)
    pd.testing.assert_series_equal(singular,original)
