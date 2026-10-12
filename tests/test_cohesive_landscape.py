import numpy as np
import pytest

from arrhenius_fracture.cohesive_landscape import (
    ExponentialLandscape,
    PolynomialLandscape,
    TriangularBilinearLandscape,
    evaluate_traction_controlled_barrier,
    numerical_work_J_m2,
)


TC = 8.0e9
GAMMA = 2.4
AREA = (0.286e-9) ** 2


@pytest.mark.parametrize("kind", [
    TriangularBilinearLandscape(TC, GAMMA),
    ExponentialLandscape(TC, GAMMA),
    PolynomialLandscape(TC, GAMMA),
])
def test_peak_and_work_are_matched(kind):
    peak_opening = (
        kind.peak_opening_m if isinstance(kind, TriangularBilinearLandscape)
        else kind.scale_m if isinstance(kind, ExponentialLandscape)
        else kind.support_end_m() / 3.0
    )
    assert kind.traction_Pa(peak_opening) == pytest.approx(TC, rel=2e-14)
    assert numerical_work_J_m2(kind) == pytest.approx(GAMMA, rel=2e-11)


def test_bilinear_exact_kocks_p1_q2_identity_and_activation_volume():
    law = TriangularBilinearLandscape(TC, GAMMA, peak_fraction=0.31)
    for fraction in (0.1, 0.3, 0.7, 0.95):
        result = evaluate_traction_controlled_barrier(law, fraction * TC, AREA)
        expected = AREA * GAMMA * (1.0 - fraction) ** 2
        assert result.barrier_J == pytest.approx(expected, rel=1e-13)
        step = TC * 1e-6
        plus = evaluate_traction_controlled_barrier(law, fraction * TC + step, AREA).barrier_J
        minus = evaluate_traction_controlled_barrier(law, fraction * TC - step, AREA).barrier_J
        assert -(plus - minus) / (2 * step) == pytest.approx(
            result.activation_volume_m3, rel=2e-6
        )


@pytest.mark.parametrize("law", [ExponentialLandscape(TC, GAMMA), PolynomialLandscape(TC, GAMMA)])
def test_native_barrier_and_activation_volume_match_derivative(law):
    for fraction in (0.15, 0.5, 0.85):
        result = evaluate_traction_controlled_barrier(law, fraction * TC, AREA)
        step = TC * 2e-6
        plus = evaluate_traction_controlled_barrier(law, fraction * TC + step, AREA).barrier_J
        minus = evaluate_traction_controlled_barrier(law, fraction * TC - step, AREA).barrier_J
        assert result.barrier_J > 0.0
        assert result.minimum_opening_m < result.saddle_opening_m
        assert -(plus - minus) / (2 * step) == pytest.approx(
            result.activation_volume_m3, rel=3e-5
        )


def test_barrier_api_rejects_endpoints_and_nonphysical_area():
    law = TriangularBilinearLandscape(TC, GAMMA)
    for traction in (0.0, TC, 1.1 * TC):
        with pytest.raises(ValueError, match="strictly between"):
            evaluate_traction_controlled_barrier(law, traction, AREA)
    with pytest.raises(ValueError, match="activation_area"):
        evaluate_traction_controlled_barrier(law, 0.5 * TC, 0.0)
