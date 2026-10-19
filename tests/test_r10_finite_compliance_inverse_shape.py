import math

import numpy as np

from arrhenius_fracture.cohesive_arrhenius import (
    COMPACT_INDEPENDENT, SMOOTH_EXPONENTIAL, CohesiveThermalLaw,
)
from arrhenius_fracture.finite_compliance_inverse_shape_r10 import (
    branch_loss, connected_barrier, cyclic_rate, metastable_window, opening_rate,
    shape_coordinates,
)
from arrhenius_fracture.opening_alternatives_r10 import OpeningAlternative, evaluate_opening
from scripts import run_r10_finite_compliance_inverse_shape as search


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
        SMOOTH_EXPONENTIAL, 6.409066161100495e-19, 1e12, phases=32,
        integration="midpoint")
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
    checked = connected_barrier(imposed, kenv, thermal, 300.,
                                COMPACT_INDEPENDENT, verify_quadrature=True)
    assert checked["barrier_quadrature_relative_error"] < 1e-8


def test_two_scale_tail_releases_a_broad_finite_compliance_window():
    thermal = CohesiveThermalLaw(2., 15e9, 0., compact_variant="two_scale_tail",
        compact_rise_fraction=.001, compact_drop_fraction=.003,
        compact_tail_level=.035)
    kenv=15e9**2/(2.*2.)
    window=metastable_window(COMPACT_INDEPENDENT,thermal,300.,kenv)
    assert window["state"] == "METASTABLE_WINDOW"
    assert float(window["load_ratio"]) > 5.


def test_interval_aware_cycle_integral_resolves_narrow_peak_accessibility():
    thermal = CohesiveThermalLaw(2., 15e9, 0., compact_variant="two_scale_tail",
        compact_rise_fraction=.001, compact_drop_fraction=.003,
        compact_tail_level=.035)
    kenv=15e9**2/(2.*2.)
    window=metastable_window(COMPACT_INDEPENDENT,thermal,300.,kenv)
    birth_K=(float(window["birth_load_Pa"])*math.sqrt(2.*math.pi*1e-6)/1e6)
    Kmax=birth_K*1.001
    midpoint=cyclic_rate(Kmax,kenv,thermal,300.,COMPACT_INDEPENDENT,
        1e-21,1e12,phases=32,integration="midpoint")
    adaptive=cyclic_rate(Kmax,kenv,thermal,300.,COMPACT_INDEPENDENT,
        1e-21,1e12,phases=32)
    assert midpoint["opening_action_per_cycle"] == 0.
    assert adaptive["opening_action_per_cycle"] > 0.
    assert 0. < adaptive["accessible_phase_fraction"] < 1./32.
    assert adaptive["integration_method"] == "finite_interval_gauss"


def test_registry_hcf_point_is_selected_from_cycle_integral(monkeypatch):
    def fake_cycle(K, *args, **kwargs):
        action = 10.0 ** (3.0 * (float(K) - 2.0) - 6.0)
        return {"Kmax_MPa_sqrt_m": float(K),
                "opening_action_per_cycle": action,
                "accessible_phase_count": 17}

    monkeypatch.setattr(search, "cyclic_rate", fake_cycle)
    metric = {"curve_json": "["
        "{\"K_MPa_sqrt_m\":1.0},{\"K_MPa_sqrt_m\":2.0},"
        "{\"K_MPa_sqrt_m\":3.0}]"}
    cohesive = {"Gamma_ref_J_m2": 2.0, "t_hat_ref_Pa": 15e9,
        "S_Gamma_J_m2_K": 0.0, "T_ref_K": 300.0,
        "activation_area_m2": 1e-19, "attempt_frequency_s": 1e12}
    chosen = search._cyclic_registry_point(
        (metric, cohesive, SMOOTH_EXPONENTIAL))
    assert chosen["chosen"]["Kmax_MPa_sqrt_m"] == 2.0
    assert chosen["chosen"]["opening_action_per_cycle"] == 1e-6
    assert chosen["target_action_per_cycle"] == 1e-6
    assert chosen["total_local_slope"] is not None


def test_registry_hcf_point_skips_one_sided_metastable_onset(monkeypatch):
    def fake_cycle(K, *args, **kwargs):
        value = float(K)
        action = 0.0 if value < .995 else 1e-6 * 100.0 ** (value-1.0)
        return {"Kmax_MPa_sqrt_m": value,
                "opening_action_per_cycle": action,
                "accessible_phase_count": 1,
                "accessible_phase_fraction": .01}

    monkeypatch.setattr(search, "cyclic_rate", fake_cycle)
    metric = {"curve_json": "["
        "{\"K_MPa_sqrt_m\":1.0},{\"K_MPa_sqrt_m\":2.0}]"}
    cohesive = {"Gamma_ref_J_m2": 2.0, "t_hat_ref_Pa": 15e9,
        "S_Gamma_J_m2_K": 0.0, "T_ref_K": 300.0,
        "activation_area_m2": 1e-19, "attempt_frequency_s": 1e12}
    chosen = search._cyclic_registry_point(
        (metric, cohesive, SMOOTH_EXPONENTIAL))
    assert chosen["chosen"]["Kmax_MPa_sqrt_m"] == 2.0
    assert chosen["selection_support"] == "TWO_SIDED_CYCLIC_DERIVATIVE"
    assert chosen["total_local_slope"] is not None


def test_inverse_coordinates_cannot_leave_declared_research_bounds():
    gamma=2.0
    area=2.0e-19
    valid=np.asarray([math.log(.01),math.log(.02/.98),math.log(.02),math.log(area)])
    assert search._inverse_bounds_pass(valid,gamma)
    too_small_rise=valid.copy();too_small_rise[0]=math.log(1e-5)
    too_long_drop=valid.copy();too_long_drop[1]=math.log(.2/.8)
    too_low_tail=valid.copy();too_low_tail[2]=math.log(1e-4)
    assert not search._inverse_bounds_pass(too_small_rise,gamma)
    assert not search._inverse_bounds_pass(too_long_drop,gamma)
    assert not search._inverse_bounds_pass(too_low_tail,gamma)
