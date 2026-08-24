import numpy as np
from scripts.build_v914_candidate_atlas import sha,windows
from scripts.search_v914_large_analytical_paris import transform

def test_stable_sha_depends_on_domain_and_sobol_index():
    r=transform(np.zeros((1,22))).iloc[0]
    assert sha(r,"HISTORICAL_DOMAIN",0)==sha(r,"HISTORICAL_DOMAIN",0)
    assert sha(r,"HISTORICAL_DOMAIN",0)!=sha(r,"CLEAVAGE_EXTENSION",0)

def test_window_catalog_can_preserve_disconnected_regimes():
    k=np.geomspace(8,32,16);y=4*np.log10(k)-10;f=np.ones(16,bool);f[7]=False
    cfg={"minimum_Paris_points":4,"minimum_relative_DeltaK_span":.2,"minimum_fit_R2":.9}
    got=windows(k,y,f,cfg)
    assert len(got)>=2
