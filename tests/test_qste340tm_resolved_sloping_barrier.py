import numpy as np

from scripts.fatigue_informed_joint_barrier_v3_v10230 import SmoothBroadBasis
from scripts.refit_qste340tm_resolved_sloping_barrier import (
    LOWER, UPPER, legacy_vector, pack, unpack,
)


def _parameters():
    return {
        "opening_floor_over_kBT": 12.0,
        "opening_amplitude_over_kBT": 20.0,
        "opening_sigma_b_Pa": 8.0e9,
        "opening_exponent": 0.4,
        "cleavage_backstress_fraction": 0.8,
        "emission_G0_298_eV": 0.9,
        "plastic_sigc_298_Pa": 1.8e9,
        "transport_G0_298_eV": 0.8,
        "retained_recovery_per_s": 0.4,
        "initial_mobile_count": 380.0,
        "initial_retained_count": 0.7,
        "source_slip_blunting_coefficient": 100.0,
        "forward_slip_blunting_coefficient": 160.0,
        "blunting_length_m": 5.5e-6,
    }


def test_released_barrier_vector_round_trip_and_bounds():
    vector = pack(_parameters())
    restored = unpack(vector)
    assert np.all(vector >= LOWER)
    assert np.all(vector <= UPPER)
    for key, value in _parameters().items():
        assert np.isclose(restored[key], value)


def test_opening_shape_coordinates_do_not_mutate_plastic_state_binding():
    vector = pack(_parameters())
    changed = vector.copy()
    changed[:4] = [8.0, 80.0, np.log(50.0), np.log(0.9)]
    assert np.array_equal(legacy_vector(vector)[1:], legacy_vector(changed)[1:])


def test_fractional_exp_sloping_floor_is_positive_and_monotone():
    vector = pack(_parameters())
    p = unpack(vector)
    stress = np.linspace(0.0, 30.0e9, 1000)
    basis = SmoothBroadBasis(p["opening_sigma_b_Pa"], p["opening_exponent"])
    barrier = p["opening_floor_over_kBT"] + p["opening_amplitude_over_kBT"] * basis.value(stress)
    assert np.all(barrier > 0.0)
    assert np.all(np.diff(barrier) <= 0.0)
    assert barrier[-1] < barrier[0]
