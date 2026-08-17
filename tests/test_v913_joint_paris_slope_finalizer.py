from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
analysis_spec=importlib.util.spec_from_file_location("fatigue_analysis",ROOT/"scripts/analyze_v914_prospective_joint_fatigue.py")
fatigue_analysis=importlib.util.module_from_spec(analysis_spec);assert analysis_spec.loader is not None;analysis_spec.loader.exec_module(fatigue_analysis)
spec=importlib.util.spec_from_file_location("finalizer",ROOT/"scripts/finalize_v913_joint_paris_slope_study.py")
module=importlib.util.module_from_spec(spec);assert spec.loader is not None;spec.loader.exec_module(module)


def test_segment_fit_reports_uncertainty_and_exact_log_slope():
    frame=pd.DataFrame({"deltaK_MPa_sqrt_m":[10.,12.,15.,20.],"developed_da_dN_m_per_cycle":[1e-12,1.2**4*1e-12,1.5**4*1e-12,2**4*1e-12]})
    result=module.segment(frame)
    assert abs(result["m"]-4.)<1e-12
    assert result["n_points"]==4 and result["m_r2"]>0.999999
    assert result["fit_quality"]=="QUALIFIED_OLS"


def test_pareto_is_raw_multiobjective_not_scalar_score():
    frame=pd.DataFrame({"K300_relative_error":[.01,.02,.005],"m_HCF_m_r2":[.99,.98,.97],"m_HCF_deltaK_span_MPa_sqrt_m":[3.,2.,1.],"partial_or_numerical_unresolved":[0,0,1]})
    keep=module.nondominated(frame)
    assert keep.tolist()==[True,False,True]


def test_finalizer_declares_censor_and_experiment_semantics():
    source=(ROOT/"scripts/finalize_v913_joint_paris_slope_study.py").read_text()
    assert "NO_DEFENSIBLE_QUANTITATIVE_LOCAL_ENVELOPE" in source
    assert "cycle_or_hazard_censors" in source
    assert "partial_or_numerical_unresolved" in source


def test_explicit_cycle_limit_is_a_true_censor_not_partial():
    assert fatigue_analysis.classify_status("explicit_cycle_limit", 16.0) == (
        "cycle_or_hazard_censor", "downward_triangle"
    )


def test_explicit_loader_uses_run_contract_fraction(tmp_path):
    case=tmp_path/"case";case.mkdir()
    (case/"run_contract.json").write_text('{"normalized_f": 1.3}')
    (case/"result.json").write_text(
        '{"schema":"v10.2.32_explicit_cycle_lcf_result_v1",'
        '"candidate_id":"c","seed":1720,"loading":{"deltaK_MPa_sqrt_m":13.0},'
        '"status":"explicit_cycle_limit","final_extension_m":0.0,"final_cycles":20,'
        '"events":[]}'
    )
    loads=pd.DataFrame({"candidate_id":["c"],"normalized_f":[1.0],"deltaK_MPa_sqrt_m":[10.0]})
    rates,_=fatigue_analysis.load_explicit(tmp_path,loads)
    assert rates.iloc[0].normalized_f == 1.3
    assert rates.iloc[0].status_class == "cycle_or_hazard_censor"


def test_finalizer_uses_current_fatigue_predictor_schema():
    source=(ROOT/"scripts/finalize_v913_joint_paris_slope_study.py").read_text()
    assert "m_bare_cleavage" in source
    assert "m_cycle_hazard_frozen" in source
    assert "m_evolved_state_predictor" in source
    assert "predicted_dm_dlnDeltaK_bare" in source
    assert "instantaneous_barrier_predictor_m" not in source
    assert ".to_markdown()" not in source
    verifier=(ROOT/"scripts/verify_v913_joint_paris_slope_study.py").read_text()
    assert 're.findall(r"(?m)^(\\d+)\\. \\*\\*", report)' in verifier
