import math

import numpy as np

from arrhenius_fracture.cohesive import CohesiveElement,CohesiveNetwork,cohesive_element_diagnostics

from arrhenius_fracture.cohesive_arrhenius import (
    BILINEAR, COMPACT_INDEPENDENT, SMOOTH_EXPONENTIAL, BilinearCohesiveLaw,
    CompactIndependentCohesiveLaw, CohesiveThermalLaw,
    cohesive_shape_response, mechanical_completion,
)
from arrhenius_fracture.opening_alternatives_r10 import (
    OpeningAlternative, evaluate_opening, frozen_commitment_density_reference,
)


def law():
    return CohesiveThermalLaw(2.5, 20e9, 2e-4)


def test_compact_research_law_is_one_normalized_potential():
    thermal = CohesiveThermalLaw(2.5, 20e9, 2e-4, compact_b=4.5,
                                compact_c=2.0)
    compact = CompactIndependentCohesiveLaw(thermal)
    cutoff = compact.final_separation(300.)
    grid = np.linspace(0., cutoff, 10001)
    traction, tangent, work = cohesive_shape_response(
        COMPACT_INDEPENDENT, grid, thermal, 300.)
    assert np.isclose(np.max(traction), 20e9, rtol=2e-7)
    assert np.isclose(np.trapezoid(traction, grid), 2.5, rtol=2e-7)
    assert np.isclose(work[-1], 2.5, rtol=1e-13)
    assert np.count_nonzero(np.diff(np.sign(tangent[:-1]))) == 1
    completion = mechanical_completion(COMPACT_INDEPENDENT, cutoff, thermal, 300.)
    assert completion["complete"]
    assert completion["cutoff_m"] == cutoff


def test_compact_rounded_plateau_is_normalized_smooth_and_single_peak():
    thermal = CohesiveThermalLaw(2.5, 20e9, 2e-4,
        compact_variant="rounded_plateau", compact_rise_fraction=.05,
        compact_fall_fraction=.9, compact_plateau_drop=.08)
    compact = CompactIndependentCohesiveLaw(thermal)
    cutoff = compact.final_separation(300.)
    grid = np.linspace(0., cutoff, 20001)
    traction, tangent, work = cohesive_shape_response(
        COMPACT_INDEPENDENT, grid, thermal, 300.)
    assert np.isclose(np.max(traction), 20e9, rtol=2e-7)
    assert np.isclose(np.trapezoid(traction, grid), 2.5, rtol=2e-7)
    assert np.isclose(work[-1], 2.5, rtol=1e-13)
    assert np.all(np.diff(traction[np.argmax(traction):]) <= 1e-3)
    assert tangent[0] > 0.


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


def test_crossover_first_passage_uses_declared_attempt_scale_without_fit():
    thermal=law();kenv=0.04*thermal.initial_stiffness(300.)
    mixed=OpeningAlternative("cohesive_mixed",SMOOTH_EXPONENTIAL,
        "time_dependent_first_passage",1e-18,1e12,high_barrier_min_kBT=1e6)
    found=None
    for factor in np.linspace(1.0,8.0,80):
        trial=evaluate_opening(mixed,factor*thermal.delta0_m,kenv,thermal,300.,
                               phase_duration_s=1e-5)
        if trial["kinetic_validity"]=="STATIC_SADDLE_MFPT_HAZARD_APPROXIMATION":
            found=trial;break
    assert found is not None
    assert found["mfpt_s"]>0.0
    assert found["elementary_rate_s-1"]==found["mfpt_rate_s-1"]
    expected=(1e-18*thermal.initial_stiffness(300.)/1e12)
    assert math.isclose(found["friction_J_s_m-2"],expected,rel_tol=1e-12)


def test_frozen_density_uses_completion_not_saddle_as_absorber():
    thermal=law();kenv=0.04*thermal.initial_stiffness(300.)
    mixed=OpeningAlternative("cohesive_mixed",SMOOTH_EXPONENTIAL,
        "time_dependent_first_passage",1e-18,1e12,high_barrier_min_kBT=1e6)
    opening=None
    for factor in np.linspace(1.,8.,80):
        trial=evaluate_opening(mixed,factor*thermal.delta0_m,kenv,thermal,300.,
                               phase_duration_s=1e-5)
        if trial["kinetic_validity"]=="STATIC_SADDLE_MFPT_HAZARD_APPROXIMATION":
            opening=trial;break
    assert opening is not None
    reference=frozen_commitment_density_reference(
        mixed,factor*thermal.delta0_m,kenv,thermal,300.,opening,1e-9,65)
    saddle=next(p["delta_m"] for p in opening["stationary"] if p["kind"]=="SADDLE")
    assert reference["absorbing_commitment_boundary_m"]>saddle
    assert not reference["saddle_is_absorbing"]
    assert 0.<=reference["committed_probability"]<=1.
    assert (reference["mean_commitment_time_s"] is None or
            reference["mean_commitment_time_s"]>0.)


def test_spatial_plastic_closure_is_a_signed_mechanical_feedback():
    thermal=law();elem=CohesiveElement((0,1),(2,3),np.array([0.,1.]),
        np.array([1.,0.]),1e-6,thermal)
    u=np.zeros(8);u[[1,3]]=thermal.delta0_m/2;u[[5,7]]=-thermal.delta0_m/2
    zero=CohesiveNetwork([elem],plastic_closure_Pa=0.)
    base=cohesive_element_diagnostics(elem,u,zero)["normal_traction_Pa"]
    coupled=CohesiveNetwork([elem],plastic_closure_Pa=2e8)
    changed=cohesive_element_diagnostics(elem,u,coupled)
    assert math.isclose(changed["normal_traction_Pa"]-base,2e8,rel_tol=1e-12)
    assert changed["plastic_closure_traction_Pa"]==2e8


def test_intact_plastic_closure_is_continuous_into_contact():
    thermal=law();elem=CohesiveElement((0,1),(2,3),np.array([0.,1.]),
        np.array([1.,0.]),1e-6,thermal)
    closure=2e8;network=CohesiveNetwork([elem],contact_penalty_Pa_per_m=1e23,
                                        plastic_closure_Pa=closure)
    values=[]
    for opening in (-1e-18,0.,1e-18):
        u=np.zeros(8);u[[1,3]]=opening/2;u[[5,7]]=-opening/2
        values.append(cohesive_element_diagnostics(elem,u,network)["normal_traction_Pa"])
    assert values[0] < values[1] < values[2]
    assert max(abs(value-closure) for value in values) < 2e5
