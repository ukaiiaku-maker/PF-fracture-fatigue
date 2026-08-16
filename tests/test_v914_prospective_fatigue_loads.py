import numpy as np

from scripts.select_v914_prospective_fatigue_loads import interpolate_fraction


def test_log_cycle_interpolation_recovers_monotone_target():
    fraction = np.array([0.8, 1.0, 1.2])
    cycles = np.array([100.0, 10.0, 1.0])
    assert abs(interpolate_fraction(fraction, cycles, 10.0) - 1.0) < 1e-12
    assert 1.0 < interpolate_fraction(fraction, cycles, 3.0) < 1.2
