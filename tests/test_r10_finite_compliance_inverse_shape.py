import math

import numpy as np

from arrhenius_fracture.cohesive_arrhenius import (
    COMPACT_INDEPENDENT, SMOOTH_EXPONENTIAL, CohesiveThermalLaw,
)
from arrhenius_fracture.finite_compliance_inverse_shape_r10 import (
    branch_loss, cyclic_rate, metastable_window, opening_rate, shape_coordinates,
)
from arrhenius_fracture.opening_alternatives_r10 import OpeningAlternative, evaluate_opening


def test_frozen_22871_reference_is_reproduced_under_finite_compliance():
    thermal = CohesiveThermalLaw(1.3911304644424378, 13192485005.3693,
                                0.0004716842664772423)
    kenv = 6.255403970563228e19
    result = opening_rate(31.035017113039576, kenv, thermal, 300.,
                          SMOOTH_EXPONENTIAL, 6.409066161100495e-19, 1e12,
                          tip_radius_m=1.0000983276769692e-6)
    assert result["state"] == "METASTABLE"
    assert math.isclose(result["barrier_kBT"], 31.129215497112707,
                        rel_tol=2e-7)
    assert math.isclose(result["root_gap_m"], 1.0725246606618489e-10,
                        rel_tol=2e-7)
    fold = branch_loss(SMOOTH_EXPONENTIAL, thermal, 300., kenv)
    fold_stress = kenv * float(fold["fold_imposed_m"])
    fold_K = fold_stress * math.sqrt(2*math.pi*1e-6)/1e6
    assert math.isclose(fold_K, 39.79889989669268, rel_tol=2e-7)
    cycle = cyclic_rate(31.305705774243552, kenv, thermal, 300.,
        SMOOTH_EXPONENTIAL, 6.409066161100495e-19, 1e12, phases=32)
    assert math.isclose(cycle["ensemble_rate_m_per_cycle"],
                        7.1002786035099876e-12, rel_tol=2e-5)


def test_compact_shape_has_finite_work_support_and_a_unique_fold_when_admissible():
    thermal = CohesiveThermalLaw(2.0, 15e9, 1e-4,
                                compact_b=5., compact_c=3.)
    coordinates = shape_coordinates(COMPACT_INDEPENDENT, thermal, 300.)
    assert coordinates["completion_cutoff_m"] > coordinates["peak_separation_m"] > 0.
    kenv = .02 * coordinates["initial_stiffness_Pa_per_m"]
    fold = branch_loss(COMPACT_INDEPENDENT, thermal, 300., kenv)
    assert fold["state"] == "SMOOTH_BRANCH_LOSS"
    assert float(fold["fold_imposed_m"]) > float(fold["fold_separation_m"])


def test_rounded_plateau_analytical_and_native_landscapes_agree():
    thermal = CohesiveThermalLaw(2., 15e9, 0., compact_variant="rounded_plateau",
        compact_rise_fraction=.02, compact_fall_fraction=.95,
        compact_plateau_drop=.03)
    kenv = 15e9**2/(2.*2.)
    window = metastable_window(COMPACT_INDEPENDENT, thermal, 300., kenv)
    load = math.sqrt(float(window["birth_load_Pa"])*float(window["fold_load_Pa"]))
    imposed = load/kenv
    area = 1e-19
    analytical = opening_rate(load*math.sqrt(2*math.pi*1e-6)/1e6, kenv,
        thermal, 300., COMPACT_INDEPENDENT, area, 1e12)
    alternative = OpeningAlternative("cohesive_mixed", COMPACT_INDEPENDENT,
        "qualified_high_barrier_hazard", area, 1e12)
    native = evaluate_opening(alternative, imposed, kenv, thermal, 300.)
    assert analytical["state"] == "METASTABLE"
    assert native["state"] == "BONDED_METASTABLE"
    assert math.isclose(analytical["barrier_J_m2"], native["barrier_J_m2"],
                        rel_tol=2e-6)
