"""Bind the v10.2.14 physical-x active atlas to each Stage 3 MPZ grid.

The FEM atlas station count is a numerical sampling choice, not a required MPZ
state dimension.  Stage 3 uses 200 bins over 100 um for ceramic/weakT and 80 bins
over 50 um for DBTT/peak.  This module makes the v10.2.14 family evaluate its
physical spatial operator on the runtime cell-centre grid before the shared
signed-population engine validates and installs it.
"""
from __future__ import annotations

import numpy as np

from .signed_kernel_family_v10214 import ActiveOnlySigned2DShieldingKernelFamily

_ORIGINAL_VALIDATE_STATE = ActiveOnlySigned2DShieldingKernelFamily.validate_state


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
    _ORIGINAL_VALIDATE_STATE(self, state)


def install_runtime_grid_binding() -> None:
    current = ActiveOnlySigned2DShieldingKernelFamily.validate_state
    if current is not _runtime_grid_validate_state:
        ActiveOnlySigned2DShieldingKernelFamily.validate_state = _runtime_grid_validate_state


install_runtime_grid_binding()

__all__ = ["install_runtime_grid_binding"]
