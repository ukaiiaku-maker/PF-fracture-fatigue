import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

PATH=Path(__file__).parents[1]/"scripts"/"finalize_v914_candidate_atlas.py"
SPEC=importlib.util.spec_from_file_location("finalize_atlas",PATH);MOD=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MOD)


def test_robust_scale_imputes_and_scales():
    x=np.array([[1.,np.nan],[2.,4.],[3.,8.]])
    z,med,scale=MOD.robust_scale(x)
    assert np.isfinite(z).all()
    assert med.tolist()==[2.,6.]
    assert (scale>0).all()


def test_queue_roles_are_unique_and_complete():
    n=600
    f=pd.DataFrame({
        "candidate_id":[f"c{i}" for i in range(n)],
        "best_broad_window_m":np.linspace(.5,10,n),
        "number_valid_windows":np.ones(n),
        "maximum_valid_window_width":np.linspace(0,1,n),
        "sustained_growth_probability":np.linspace(0,1,n),
        "property_cluster_id":np.arange(n)%48,
        "parameter_cluster_id":np.arange(n)%64,
        "property_cluster_distance":np.linspace(0,2,n),
        "parameter_cluster_distance":np.linspace(2,0,n),
    })
    q=MOD.select_queue(f,250)
    assert len(q)==250
    assert q.candidate_id.is_unique
    assert set(q.atlas_sampling_role) >= {"realistic_paris_exploitation","distinct_low_m_basin","property_space_coverage","analytical_uncertainty","maximin_global_exploration"}


def test_v7_fixed_launch_constants_match_qualified_registry_contract():
    source=PATH.read_text()
    assert '"peierls_nu0_s": 1.0e12' in source
    assert '"taylor_nu0_s": 1.0e11' in source
    assert 'queue["paris_design_role"] = queue["atlas_sampling_role"]' in source
