import numpy as np

from scripts.calibrate_qste340tm_opening_reference import predict_log10_rate


def test_frequency_changes_cycle_rate_by_inverse_frequency_for_fixed_barrier():
    stress = np.full((3, 16), 5.0e9)
    parameters = np.asarray([18.0, 16.0, 0.0, 0.0])
    one_hz = predict_log10_rate(parameters, stress, 1.0)
    ten_hz = predict_log10_rate(parameters, stress, 10.0)
    assert np.allclose(one_hz - ten_hz, 1.0, rtol=0.0, atol=1.0e-12)


def test_stress_sensitive_barrier_produces_increasing_growth_rate():
    parameters = np.asarray([18.0, 16.0, 0.0, 0.0])
    low = predict_log10_rate(parameters, np.full((1, 64), 2.0e9), 10.0)
    high = predict_log10_rate(parameters, np.full((1, 64), 8.0e9), 10.0)
    assert high[0] > low[0]
