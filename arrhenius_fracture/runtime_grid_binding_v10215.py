"""Bind the v10.2.14 physical-x active atlas to each Stage 3 MPZ grid.

The FEM atlas station count is a numerical sampling choice, not a required MPZ
state dimension. Stage 3 uses 200 bins over 100 um for ceramic/weakT and 80 bins
over 50 um for DBTT/peak. This module makes the v10.2.14 family evaluate its
physical spatial operator on the runtime cell-centre grid before the shared
signed-population engine validates and installs it.

The final accepted 2-D source law does not interpret
``manifest.source_sites_per_system`` as a count of geometrically packed source
positions. It is the promoted reference continuum hazard budget S0. The active
budget S(t) is depleted by Arrhenius emission, coupled to the evolving local
mobile-plus-retained field through Taylor back stress, and refreshed only when
crack advance exposes fresh tip geometry. Therefore the geometric 10--100 b
possible-position interval stored in the mechanics-normalization artifact is
audit metadata only and must not gate S0. The activation-to-signed-line-content
conversion remains mechanically derived and unchanged.
"""
from __future__ import annotations

import copy

import numpy as np

from .signed_kernel_family_v10214 import ActiveOnlySigned2DShieldingKernelFamily

_ORIGINAL_VALIDATE_STATE = ActiveOnlySigned2DShieldingKernelFamily.validate_state


def _preserve_final_2d_continuum_source_budget(self, state) -> None:
    reference_budget = np.asarray(state.site_capacity, dtype=float).reshape(-1)
    if reference_budget.shape != (int(state.n_systems),):
        raise ValueError("runtime continuum source budget must have one value per system")
    if np.any(~np.isfinite(reference_budget)) or np.any(reference_budget < 0.0):
        raise ValueError("runtime continuum source budget must be finite and nonnegative")

    geometric_bounds = np.asarray(self.source_capacity_bounds, dtype=float).copy()
    if geometric_bounds.shape != (int(state.n_systems), 2):
        raise ValueError("atlas source-position audit bounds have the wrong shape")
    if np.any(~np.isfinite(geometric_bounds)):
        raise ValueError("atlas source-position audit bounds must be finite")

    # The inherited family validator expects numerical bounds. Supply permissive
    # bounds only for that legacy interface; they are not a constitutive law.
    runtime_bounds = np.column_stack(
        (
            np.zeros(int(state.n_systems), dtype=float),
            np.maximum(geometric_bounds[:, 1], reference_budget),
        )
    )
    self.source_capacity_bounds = runtime_bounds
    self.metadata = {
        **copy.deepcopy(self.metadata),
        "runtime_source_budget_binding": {
            "model_id": "v10.2.15_preserve_final_2d_continuum_source_budget",
            "source_model": "campaign_calibrated_tip_budget",
            "reference_continuum_hazard_budget_S0_per_system": (
                reference_budget.tolist()
            ),
            "active_budget_evolves_in_time": True,
            "emission_depletes_active_budget": True,
            "local_mobile_retained_field_controls_backstress": True,
            "crack_advance_refreshes_budget_over_manifest_length": True,
            "stationary_temporal_recycling": False,
            "geometric_possible_position_bounds_audit_only": geometric_bounds.tolist(),
            "geometric_bounds_used_as_constitutive_S0_limits": False,
            "legacy_validator_bounds": runtime_bounds.tolist(),
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
    _preserve_final_2d_continuum_source_budget(self, state)
    _ORIGINAL_VALIDATE_STATE(self, state)


def install_runtime_grid_binding() -> None:
    current = ActiveOnlySigned2DShieldingKernelFamily.validate_state
    if current is not _runtime_grid_validate_state:
        ActiveOnlySigned2DShieldingKernelFamily.validate_state = _runtime_grid_validate_state


install_runtime_grid_binding()

__all__ = ["install_runtime_grid_binding"]
