"""Bind the v10.2.14 physical-x active atlas to each Stage 3 MPZ grid.

The FEM atlas station count is a numerical sampling choice, not a required MPZ
state dimension. Stage 3 uses 200 bins over 100 um for ceramic/weakT and 80 bins
over 50 um for DBTT/peak. This module makes the v10.2.14 family evaluate its
physical spatial operator on the runtime cell-centre grid before the shared
signed-population engine validates and installs it.

The mechanics normalization also reports a dense geometric *possible-position*
range based on 10--100 b spacing. Its lower value is not a constitutive minimum
number of active nucleation sites. Stage 3 preserves the finite source count in
the validated v9.11.1 material row while retaining the mechanics-derived upper
capacity check and activation-to-line-content conversion.
"""
from __future__ import annotations

import copy

import numpy as np

from .signed_kernel_family_v10214 import ActiveOnlySigned2DShieldingKernelFamily

_ORIGINAL_VALIDATE_STATE = ActiveOnlySigned2DShieldingKernelFamily.validate_state


def _preserve_sparse_registry_source_capacity(self, state) -> None:
    capacity = np.asarray(state.site_capacity, dtype=float).reshape(-1)
    if capacity.shape != (int(state.n_systems),):
        raise ValueError("runtime source capacity must have one value per system")
    if np.any(~np.isfinite(capacity)) or np.any(capacity < 0.0):
        raise ValueError("runtime source capacity must be finite and nonnegative")

    bounds = np.asarray(self.source_capacity_bounds, dtype=float).copy()
    if bounds.shape != (int(state.n_systems), 2):
        raise ValueError("atlas source-capacity bounds have the wrong shape")
    original = bounds.copy()
    # A sparse active-source population may be below the dense geometric count.
    # The mechanically derived upper bound remains enforced.
    bounds[:, 0] = 0.0
    self.source_capacity_bounds = bounds
    self.metadata = {
        **copy.deepcopy(self.metadata),
        "runtime_source_capacity_binding": {
            "model_id": "v10.2.15_preserve_validated_registry_source_sites",
            "runtime_source_sites_per_system": capacity.tolist(),
            "original_geometric_possible_position_bounds": original.tolist(),
            "runtime_validation_bounds": bounds.tolist(),
            "geometric_lower_bound_used_as_constitutive_minimum": False,
            "mechanical_upper_capacity_bound_retained": True,
            "activation_to_line_content_unchanged": True,
        },
    }


def _runtime_grid_validate_state(self, state) -> None:
    expected_active = (int(state.n_systems), int(state.n_bins))
    expected_wake = (int(state.n_systems), int(state.wake_n_bins))
    active_matches = (
        self.states[0].active_I.shape == expected_active
        and self.active_x_m.shape == np.asarray(state.x).shape
        and np.allclose(self.active_x_m, state.x, rtol=1.0e-12, atol=1.0e-18)
    )
    wake_matches = (
        self.states[0].wake_I.shape == expected_wake
        and self.wake_x_m.shape == np.asarray(state.wake_x).shape
        and np.allclose(self.wake_x_m, state.wake_x, rtol=1.0e-12, atol=1.0e-18)
    )
    if not (active_matches and wake_matches):
        bound = self.bind_to_state_grid(state)
        self.__dict__.clear()
        self.__dict__.update(bound.__dict__)
    _preserve_sparse_registry_source_capacity(self, state)
    _ORIGINAL_VALIDATE_STATE(self, state)


def install_runtime_grid_binding() -> None:
    current = ActiveOnlySigned2DShieldingKernelFamily.validate_state
    if current is not _runtime_grid_validate_state:
        ActiveOnlySigned2DShieldingKernelFamily.validate_state = _runtime_grid_validate_state


install_runtime_grid_binding()

__all__ = ["install_runtime_grid_binding"]
