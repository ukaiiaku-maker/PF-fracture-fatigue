from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
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
