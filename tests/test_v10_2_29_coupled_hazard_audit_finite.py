import math

from arrhenius_fracture.persistent_site_coupled_hazard_v10229 import _log_span


def test_zero_to_positive_log_span_remains_finite():
    span = _log_span([0.0, 1.0e-20, 1.0e-5])
    assert math.isfinite(span)
    assert span > 100.0
