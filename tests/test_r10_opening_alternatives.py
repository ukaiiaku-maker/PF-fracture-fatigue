import math

import numpy as np

from arrhenius_fracture.cohesive_arrhenius import (
    BILINEAR, SMOOTH_EXPONENTIAL, BilinearCohesiveLaw, CohesiveThermalLaw,
    cohesive_shape_response, mechanical_completion,
)
from arrhenius_fracture.opening_alternatives_r10 import (
    OpeningAlternative, evaluate_opening,
)


def law():
    return CohesiveThermalLaw(2.5, 20e9, 2e-4)


def test_bilinear_matches_exponential_work_peak_and_initial_stiffness():
    thermal=law();bilinear=BilinearCohesiveLaw(thermal);T=300.0
    dp=bilinear.peak_separation(T);df=bilinear.final_separation(T)
    assert math.isclose(float(bilinear.traction(dp,T)),thermal.t_hat(T),rel_tol=1e-12)
    assert math.isclose(float(bilinear.potential(df,T)),thermal.gamma(T),rel_tol=1e-12)
    exp_k=thermal.initial_stiffness(T)
    bilinear_k=float(bilinear.tangent(0.0,T))
    assert math.isclose(bilinear_k,exp_k,rel_tol=1e-12)


def test_shape_work_is_monotone_and_bounded():
    thermal=law()
    for shape in (SMOOTH_EXPONENTIAL,BILINEAR):
        d=np.linspace(0,20*thermal.delta0_m,1000)
        work=np.asarray(cohesive_shape_response(shape,d,thermal,300.)[2])
        assert np.all(np.diff(work)>=-1e-12)
        assert work.min()>=0.0
        assert work.max()<=thermal.gamma(300.)*(1+1e-12)


def test_athermal_contains_no_rate_and_mixed_respects_validity_scope():
    thermal=law();kenv=0.04*thermal.initial_stiffness(300.)
    athermal=OpeningAlternative("cohesive_athermal",SMOOTH_EXPONENTIAL,
                                "quasistatic_mechanical")
    mixed=OpeningAlternative("cohesive_mixed",SMOOTH_EXPONENTIAL,
                             "qualified_high_barrier_hazard",1e-18,1e12)
    a=evaluate_opening(athermal,2*thermal.delta0_m,kenv,thermal,300.)
    m=evaluate_opening(mixed,2*thermal.delta0_m,kenv,thermal,300.)
    assert a["elementary_rate_s-1"] is None
    assert a["kinetic_validity"]=="NOT_APPLICABLE_ATHERMAL"
    assert m["kinetic_validity"] in {"QUALIFIED_HIGH_BARRIER",
        "CROSSOVER_REQUIRES_FIRST_PASSAGE","NO_METASTABLE_ESCAPE_PATH",
        "MECHANICAL_CONTINUATION"}


def test_completion_is_shape_specific_and_geometry_owned():
    thermal=law();bilinear=BilinearCohesiveLaw(thermal)
    assert mechanical_completion(BILINEAR,bilinear.final_separation(300.),thermal,300.)["complete"]
    assert not mechanical_completion(SMOOTH_EXPONENTIAL,thermal.delta0_m,thermal,300.)["complete"]
