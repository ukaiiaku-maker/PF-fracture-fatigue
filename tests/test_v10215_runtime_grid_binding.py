from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from arrhenius_fracture import runtime_grid_binding_v10215  # noqa: F401
from arrhenius_fracture.signed_kernel_family_v10214 import (
    ActiveOnlySigned2DShieldingKernelFamily,
    KernelState,
)


def _family() -> ActiveOnlySigned2DShieldingKernelFamily:
    source_x = (np.arange(40, dtype=float) + 0.5) * (100.0e-6 / 40.0)
    rows_I = np.vstack((1.0 + 2.0e4 * source_x, -0.5 + 1.0e4 * source_x))
    rows_II = 0.25 * rows_I
    states = []
    for state_id, extension, scale in (
        ("E000", 0.0, 1.0),
        ("E200", 200.0e-6, 1.1),
    ):
        states.append(
            KernelState(
                state_id=state_id,
                coordinates=np.asarray([1.0, 0.0, extension]),
                active_I=scale * rows_I,
                wake_I=np.zeros((2, 40)),
                active_II=scale * rows_II,
                wake_II=np.zeros((2, 40)),
                metadata={},
            )
        )
    family = ActiveOnlySigned2DShieldingKernelFamily(
        states=states,
        active_x_m=source_x,
        wake_x_m=source_x.copy(),
        activation_to_line_content=np.ones(2),
        source_capacity_bounds=np.asarray([[0.0, 1.0e5], [0.0, 1.0e5]]),
        fixed_kernel_assessment={"fixed_kernel_accepted": False},
        interpolation={"method": "inverse_distance", "neighbors": 2},
        metadata={"schema": "test"},
        source_path="synthetic",
    )
    family._opening_boundary_policy = {"policy": "strict"}
    family._last_boundary_action = "none"
    family._last_observed_analytical_r_eff_over_r0 = 1.0
    family._last_observed_opening_strength_fraction = 0.0
    family._runtime_grid_binding = None
    return family


def _state(length_m: float, n_bins: int) -> SimpleNamespace:
    dx = length_m / n_bins
    x = (np.arange(n_bins, dtype=float) + 0.5) * dx
    return SimpleNamespace(
        n_systems=2,
        n_bins=n_bins,
        x=x,
        wake_n_bins=n_bins,
        wake_x=x.copy(),
        site_capacity=np.asarray([1000.0, 1000.0]),
    )


def test_ceramic_and_weakt_200_bin_runtime_binding():
    family = _family()
    state = _state(100.0e-6, 200)
    family.validate_state(state)
    assert family.states[0].active_I.shape == (2, 200)
    assert family.states[0].wake_I.shape == (2, 200)
    assert np.allclose(family.active_x_m, state.x)
    assert np.all(family.states[0].wake_I == 0.0)
    expected = 1.0 + 2.0e4 * state.x
    assert np.allclose(family.states[0].active_I[0], expected)


def test_dbtt_and_peak_80_bin_50um_runtime_binding():
    family = _family()
    state = _state(50.0e-6, 80)
    family.validate_state(state)
    assert family.states[0].active_I.shape == (2, 80)
    assert family.states[0].wake_I.shape == (2, 80)
    assert np.allclose(family.active_x_m, state.x)
    expected = 1.0 + 2.0e4 * state.x
    assert np.allclose(family.states[0].active_I[0], expected)
    audit = family.audit_payload()
    assert audit["runtime_grid_binding"]["source_active_bins"] == 40
    assert audit["runtime_grid_binding"]["runtime_active_bins"] == 80


def test_binding_rejects_runtime_grid_beyond_measured_support():
    family = _family()
    state = _state(120.0e-6, 200)
    try:
        family.validate_state(state)
    except ValueError as exc:
        assert "outside the measured active-kernel support" in str(exc)
    else:
        raise AssertionError("grid beyond the measured support must fail closed")
