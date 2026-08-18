from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.v914_finite_tip_shielding_state_v5 import finite_tip_shield_kernel


G = 160.15625e9
B = 2.74e-10
NU = 0.28
AREA = 0.625e-6 * 10.0e-6
CORE = 5.0 * B


def sharp_kernel(x: np.ndarray) -> np.ndarray:
    xx = np.maximum(x, CORE)
    prefactor = G * B / (1.0 - NU)
    return prefactor / np.sqrt(2.0 * math.pi * xx) * AREA / 1.0e6


def test_radius_floor_recovers_sharp_kernel_outside_tip_radius() -> None:
    x = np.array([0.5e-6, 1.0e-6, 2.0e-6])
    got = finite_tip_shield_kernel(
        x,
        G_Pa=G,
        b_m=B,
        nu=NU,
        cell_area_m2=AREA,
        tip_radius_m=0.1e-6,
        core_radius_m=CORE,
        mode="radius_floor",
    )
    np.testing.assert_allclose(got, sharp_kernel(x), rtol=1e-14, atol=0.0)


def test_radius_floor_saturates_inside_existing_tip_radius() -> None:
    x = np.array([0.1e-6, 0.5e-6, 1.0e-6, 3.0e-6])
    got = finite_tip_shield_kernel(
        x,
        G_Pa=G,
        b_m=B,
        nu=NU,
        cell_area_m2=AREA,
        tip_radius_m=2.0e-6,
        core_radius_m=CORE,
        mode="radius_floor",
    )
    assert got[0] == pytest.approx(got[1])
    assert got[1] == pytest.approx(got[2])
    assert got[3] < got[2]
    assert np.all(got <= sharp_kernel(x))


def test_radius_shift_is_smooth_and_no_stronger_than_sharp() -> None:
    x = np.array([0.1e-6, 0.5e-6, 1.0e-6, 3.0e-6])
    got = finite_tip_shield_kernel(
        x,
        G_Pa=G,
        b_m=B,
        nu=NU,
        cell_area_m2=AREA,
        tip_radius_m=2.0e-6,
        core_radius_m=CORE,
        mode="radius_shift",
    )
    assert np.all(np.diff(got) < 0.0)
    assert np.all(got <= sharp_kernel(x))


def test_larger_tip_radius_weakens_both_finite_tip_kernels() -> None:
    x = np.array([0.3125e-6, 0.9375e-6, 1.5625e-6])
    for mode in ("radius_floor", "radius_shift"):
        small = finite_tip_shield_kernel(
            x,
            G_Pa=G,
            b_m=B,
            nu=NU,
            cell_area_m2=AREA,
            tip_radius_m=1.0e-6,
            core_radius_m=CORE,
            mode=mode,
        )
        large = finite_tip_shield_kernel(
            x,
            G_Pa=G,
            b_m=B,
            nu=NU,
            cell_area_m2=AREA,
            tip_radius_m=2.0e-6,
            core_radius_m=CORE,
            mode=mode,
        )
        assert np.all(large <= small)


def test_invalid_mode_fails_closed() -> None:
    with pytest.raises(ValueError):
        finite_tip_shield_kernel(
            np.array([1.0e-6]),
            G_Pa=G,
            b_m=B,
            nu=NU,
            cell_area_m2=AREA,
            tip_radius_m=1.0e-6,
            core_radius_m=CORE,
            mode="fit_me",
        )
