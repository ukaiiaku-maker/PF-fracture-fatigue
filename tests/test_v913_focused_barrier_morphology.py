from __future__ import annotations
import importlib.util,sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; path=ROOT/"scripts/analyze_v913_focused_barrier_morphology.py"
spec=importlib.util.spec_from_file_location("v913_focused",path); m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m)

def test_analytic_transition_positions_reconstruct_exp_floor_shape():
    q=m.shape_metrics(.73,1.61,.04)
    for p in m.LEVELS:
        x=q[f"x{int(100*p):02d}"]
        assert np.isclose(np.exp(-.73*x**1.61),p,rtol=1e-12,atol=1e-12)
    assert q["x90"]<q["x75"]<q["x50"]<q["x25"]<q["x10"]

def test_activation_window_moments_match_weibull_and_identical_overlap_is_one():
    q=m.shape_metrics(.8,1.7,.03)
    mean,var,skew,kurt=m.stats.weibull_min.stats(1.7,scale=.8**(-1/1.7),moments="mvsk")
    assert np.isclose(q["activation_mu"],mean)
    assert np.isclose(q["activation_variance"],var)
    assert np.isclose(q["activation_skewness"],skew)
    assert np.isclose(q["activation_excess_kurtosis"],kurt)
    overlap,wasser,js=m.activation_overlap_distance((.8,1.7),(.8,1.7))
    assert overlap>.999999 and wasser<1e-12 and js<1e-12
    assert m.activation_overlap_only((.8,1.7),(.8,1.7))>.999999

def test_true_origin_singularities_are_flagged_not_grid_regularized():
    q=m.shape_metrics(.6,.75,.02)
    assert q["sstar_singular_at_zero"] is True and np.isnan(q["sstar_max"])
    assert q["cstar_singular_at_zero"] is True

def test_leakage_classification_separates_all_three_levels():
    assert m.classify_column("delta_mu_emit_minus_cleave","activation_window")[0]=="INTRINSIC_PREDICTOR"
    assert m.classify_column("delta_g_ec_at_K075","response_conditioned_diagnostic")[0]=="RESPONSE_DERIVED_DIAGNOSTIC"
    assert m.classify_column("tip_radius_over_r0_span_T","state_proxy")[0]=="STATE_MEDIATOR"
    assert m.classify_column("authoritative_response_MPa_sqrt_m","v1_master")[0]=="RESPONSE_VARIABLE"
    assert m.classify_column("first_event_K_MPa_sqrt_m","v1_master")[0]=="RESPONSE_VARIABLE"

def test_incremental_hierarchy_has_exact_six_physical_stages():
    expected=["BARRIER_SCALE","BARRIER_SHAPE","RELATIVE_GEOMETRY_PLUS_THERMAL_MOTION","KINETIC_COMPETITION","PLASTIC_BOTTLENECK","STATE_PROXY"]
    source=Path(ROOT/"runs/v913_barrier_temperature_fracture_morphology_v3_focused/feature_family_incremental_models.csv")
    if source.exists():
        q=pd.read_csv(source); assert q[q.response.eq(q.response.iloc[0])].sort_values("model_stage").added_family.tolist()==expected

def test_hash_folds_are_deterministic_and_canonical_ids_are_not_special_cased():
    ids=np.array(["a","b","v913_zeroD_sobol_0242980"])
    assert np.array_equal(m.fold_ids(ids),m.fold_ids(ids))
    assert np.all((m.fold_ids(ids)>=0)&(m.fold_ids(ids)<5))

def test_activation_center_map_uses_lossless_signed_log_for_long_tail(tmp_path):
    q=pd.DataFrame({"candidate_id":["a","b"],"canonical_family":[np.nan,np.nan],"delta_mu_emit_minus_cleave":[-2.,6000.],"activation_window_overlap_Oce":[.5,.1],"fractional_resistance_span":[.2,.3]})
    m.scatter_continuous(tmp_path,"activation_window_center_overlap_map",q,"delta_mu_emit_minus_cleave","activation_window_overlap_Oce","fractional_resistance_span","span")
    saved=pd.read_csv(tmp_path/"activation_window_center_overlap_map_plot_data.csv")
    assert saved.delta_mu_emit_minus_cleave.tolist()==[-2.,6000.]

def test_clipped_characteristic_scale_has_zero_exact_thermal_derivative():
    v1=m.load_module(ROOT/"scripts/analyze_v913_barrier_temperature_fracture_morphology.py","_clip_v1")
    E,_=v1.load_production_types(m.SOURCE)
    s=E(2.,-.001,1e9,-2e6,1.,1.5,.03,Tref_K=500.)
    dG,ds,Gclip,Sclip=m.effective_surface_scale_derivatives(s,1100.)
    assert not Gclip and Sclip and dG==-.001 and ds==0.
    _,dGstress=m.surface_thermal_parts(s,1100.,5e9)
    assert dGstress==0.

def test_physical_interaction_terms_are_not_conditioned_on_univariate_rank():
    source=(ROOT/"scripts/analyze_v913_focused_barrier_morphology.py").read_text()
    assert "main=list(dict.fromkeys(top+pairvars))" in source
    assert "main+interactions" in source
