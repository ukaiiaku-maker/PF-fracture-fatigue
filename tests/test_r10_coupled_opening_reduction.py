import math
from arrhenius_fracture.cohesive_arrhenius import CohesiveThermalLaw
from arrhenius_fracture.coupled_opening_reduction_r10 import bilinear_accessibility_coordinates
from scripts.analyze_r10_coupled_opening_search import native_disposition


def test_inactive_bilinear_reference_separates_cut_loss_and_discrete_screen():
    thermal=CohesiveThermalLaw(1.895736714987287,1.0332740562155006e10,
                              0.,300.)
    got=bilinear_accessibility_coordinates(thermal,300.,2.8159376426256364e19,1e-6,8)
    assert math.isclose(got["static_completed_branch_K_MPa_sqrt_m"],25.900339647524365,rel_tol=2e-15)
    assert math.isclose(got["occupied_bonded_branch_loss_K_MPa_sqrt_m"],27.65295454858545,rel_tol=2e-15)
    assert math.isclose(got["discrete_waveform_Kmax_for_branch_loss_MPa_sqrt_m"],28.63378214589751,rel_tol=2e-15)
    assert got["static_completed_branch_K_MPa_sqrt_m"] < got["occupied_bonded_branch_loss_K_MPa_sqrt_m"]


def test_subcycle_first_passage_is_not_a_developed_rate():
    result = {
        "unsupported_crossover_encountered": False,
        "stop_category": "OBSERVATION_PATH_ENDPOINT_DURING_CASCADE",
        "events": [
            {"cycle": 9.0e-6, "extension_m": 20e-6},
            {"cycle": 2.9e-5, "extension_m": 60e-6},
        ],
    }
    assert native_disposition(result) == "TARGET_TRUNCATED_SAME_CYCLE_CASCADE"


def test_unsupported_mixed_nonadvance_remains_unresolved():
    result = {
        "unsupported_crossover_encountered": True,
        "stop_category": "SUPPORTED_CYCLIC_SHAKEDOWN_NO_ADVANCE",
        "events": [],
    }
    assert native_disposition(result) == "UNSUPPORTED_CROSSOVER_UNRESOLVED"
