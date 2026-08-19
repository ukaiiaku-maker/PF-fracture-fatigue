from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
V914 = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/"
    "Arrhenius_FEM_CZM_MPZ_v9_14_cyclic_fatigue_knee_search"
)
for _path in (str(ROOT / "scripts"), str(V914)):
    while _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(V914))
sys.path.insert(0, str(ROOT / "scripts"))

from v914_intrinsic_reverse_glide_v7 import (
    IntrinsicReverseGlideState,
    intrinsic_signed_glide_stress_fields,
    reduced_competing_return_probability,
)


def test_signed_applied_stress_is_forward_reverse_symmetric_without_gnd():
    drive = np.array([0.5, -0.25])
    tau_gnd = np.zeros((2, 3))
    plus = intrinsic_signed_glide_stress_fields(
        10.0, 2.0e-6, drive, tau_gnd
    )
    minus = intrinsic_signed_glide_stress_fields(
        -10.0, 2.0e-6, drive, tau_gnd
    )
    np.testing.assert_allclose(
        minus["tau_applied_Pa"], -plus["tau_applied_Pa"], rtol=1e-14, atol=0.0
    )
    np.testing.assert_allclose(
        minus["tau_effective_Pa"], -plus["tau_effective_Pa"], rtol=1e-14, atol=0.0
    )


def test_zero_applied_load_leaves_only_signed_gnd_drive():
    drive = np.array([0.5, -0.25])
    tau_gnd = np.array([[1.0, -2.0], [-3.0, 4.0]]) * 1.0e6
    fields = intrinsic_signed_glide_stress_fields(
        0.0, 2.0e-6, drive, tau_gnd
    )
    np.testing.assert_allclose(fields["tau_applied_Pa"], 0.0, atol=0.0)
    np.testing.assert_allclose(fields["tau_effective_Pa"], tau_gnd, rtol=0.0, atol=0.0)


def test_finite_tip_radius_bounds_signed_applied_stress():
    drive = np.array([1.0])
    tau_gnd = np.zeros((1, 1))
    small = intrinsic_signed_glide_stress_fields(
        10.0, 1.0e-6, drive, tau_gnd
    )
    large = intrinsic_signed_glide_stress_fields(
        10.0, 4.0e-6, drive, tau_gnd
    )
    ratio = float(small["sigma_applied_signed_Pa"]) / float(
        large["sigma_applied_signed_Pa"]
    )
    assert ratio == pytest.approx(2.0, rel=1e-14)


def test_stress_helper_has_no_nonlocal_shielding_argument():
    import inspect

    names = tuple(inspect.signature(intrinsic_signed_glide_stress_fields).parameters)
    assert "K_shield_MPa_sqrt_m" not in names
    assert names == (
        "K_applied_MPa_sqrt_m",
        "tip_radius_m",
        "drive_factors",
        "tau_gnd_Pa",
    )


def test_reduced_return_probability_has_no_return_fraction_parameter():
    import inspect

    names = tuple(inspect.signature(reduced_competing_return_probability).parameters)
    assert "return_fraction" not in names
    assert "reverse_return_rate_s" in names
    assert "reverse_loss_rate_s" in names


def test_reduced_competing_return_probability_limits():
    assert reduced_competing_return_probability(
        forward_loss_rate_s=1.0,
        reverse_return_rate_s=0.0,
        reverse_loss_rate_s=2.0,
        forward_time_s=1.0,
        reverse_time_s=1.0,
    ) == 0.0

    value = reduced_competing_return_probability(
        forward_loss_rate_s=1.0,
        reverse_return_rate_s=3.0,
        reverse_loss_rate_s=2.0,
        forward_time_s=0.2,
        reverse_time_s=0.4,
    )
    assert 0.0 < value < 1.0


def test_reduced_frequency_response_can_have_interior_maximum():
    # t_forward and t_reverse scale as 1/f.  With finite forward loss and finite
    # reverse-return time, the reduced mechanism vanishes at both sufficiently
    # low and sufficiently high frequency and therefore has an interior peak.
    def p(f):
        return reduced_competing_return_probability(
            forward_loss_rate_s=50.0,
            reverse_return_rate_s=200.0,
            reverse_loss_rate_s=20.0,
            forward_time_s=0.5 / f,
            reverse_time_s=0.5 / f,
        )

    low = p(0.1)
    mid = p(100.0)
    high = p(1.0e8)
    assert mid > low
    assert mid > high


def test_v7_state_adds_no_constructor_return_parameter():
    import inspect

    names = tuple(inspect.signature(IntrinsicReverseGlideState.__init__).parameters)
    assert names == ("self", "candidate", "physics")


def test_intrinsic_stress_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        intrinsic_signed_glide_stress_fields(
            1.0,
            1.0e-6,
            np.array([1.0, 1.0]),
            np.zeros((1, 3)),
        )
