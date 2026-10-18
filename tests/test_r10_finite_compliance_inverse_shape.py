import math

import numpy as np

from arrhenius_fracture.cohesive_arrhenius import (
    COMPACT_INDEPENDENT, SMOOTH_EXPONENTIAL, CohesiveThermalLaw,
)
from arrhenius_fracture.finite_compliance_inverse_shape_r10 import (
    branch_loss, opening_rate, shape_coordinates,
)


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


def test_compact_shape_has_finite_work_support_and_a_unique_fold_when_admissible():
    thermal = CohesiveThermalLaw(2.0, 15e9, 1e-4,
                                compact_b=5., compact_c=3.)
    coordinates = shape_coordinates(COMPACT_INDEPENDENT, thermal, 300.)
    assert coordinates["completion_cutoff_m"] > coordinates["peak_separation_m"] > 0.
    kenv = .02 * coordinates["initial_stiffness_Pa_per_m"]
    fold = branch_loss(COMPACT_INDEPENDENT, thermal, 300., kenv)
    assert fold["state"] == "SMOOTH_BRANCH_LOSS"
    assert float(fold["fold_imposed_m"]) > float(fold["fold_separation_m"])
