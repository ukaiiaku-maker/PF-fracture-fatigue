import numpy as np

from scripts.plot_qste340tm_conditional_fit import TEMPERATURE_K, barrier_eV


def test_room_temperature_binding_is_25_celsius():
    assert TEMPERATURE_K == 298.15


def test_barrier_conversion_is_positive_and_stress_decreasing():
    parameters = {
        "g_floor_Gstar_over_kBT": 16.0,
        "g_amplitude_Gstar_over_kBT": 30.0,
        "sigma_b_Pa": 2.0e9,
        "exponent": 0.5,
    }
    barrier = barrier_eV(parameters, np.asarray([0.0, 1.0e9, 10.0e9]))
    assert np.all(barrier > 0.0)
    assert np.all(np.diff(barrier) < 0.0)
