import pytest

from arrhenius_fracture.off_axis_void_steering import (
    circular_void_path_offset_m,
    initial_steering_prediction,
    segment_circle_clearance_m,
)


def test_circular_void_predictor_reflects_and_attracts():
    common = dict(cavity_radius_m=5.0e-5, poisson_ratio=0.3)
    above = initial_steering_prediction(
        4.0e-4, cavity_center_m=(6.0e-4, 6.0e-5), **common)
    below = initial_steering_prediction(
        4.0e-4, cavity_center_m=(6.0e-4, -6.0e-5), **common)
    assert above["slope_at_initial_tip"] > 0.0
    assert below["slope_at_initial_tip"] == pytest.approx(
        -above["slope_at_initial_tip"], rel=1.0e-9)
    assert below["offset_at_initial_tip_m"] == pytest.approx(
        -above["offset_at_initial_tip_m"], rel=1.0e-12)


def test_predictor_limits_and_straight_path_clearance():
    assert circular_void_path_offset_m(
        -100.0, cavity_center_m=(0.0, 0.06), cavity_radius_m=0.05,
        poisson_ratio=0.3) == pytest.approx(0.0, abs=1.0e-8)
    assert segment_circle_clearance_m(
        (0.0, 0.0), (1.0, 0.0), cavity_center_m=(0.6, 0.06),
        cavity_radius_m=0.05) == pytest.approx(0.01)
