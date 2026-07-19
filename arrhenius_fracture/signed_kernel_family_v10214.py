"""Active-only production loader for the v10.2.14 signed FEM atlas."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .checked_spatial_station_projection_v10212 import (
    KERNEL_RADIUS_COMPATIBILITY_COORDINATE,
)
from .signed_kernel_family_v10212 import (
    RealSigned2DShieldingKernelFamily as _V10212Family,
    SCHEMA as V10212_SCHEMA,
)
from .signed_kernel_family_v1029 import KernelState, STATE_AXES

SCHEMA = "v10.2.14_active_only_real_signed_2d_shielding_atlas"
OPENING_COMPATIBILITY_COORDINATE = 0.0


class ActiveOnlySigned2DShieldingKernelFamily(_V10212Family):
    """Interpolate active FEM kernels by cumulative crack-path extension.

    Wake kernels are required to be exactly zero.  The scalar persistent wake
    remains available to the kinetic state for recovery/bookkeeping but cannot
    contribute a claimed physical 2-D shielding response.
    """

    @classmethod
    def from_json(cls, path: str | Path) -> "ActiveOnlySigned2DShieldingKernelFamily":
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text())
        if payload.get("schema") != SCHEMA:
            raise ValueError(
                f"v10.2.14 kernel-family schema must be {SCHEMA!r}; "
                f"got {payload.get('schema')!r}"
            )
        required = {
            "kernel_radius_axis_policy": "disabled_constant_compatibility",
            "opening_axis_policy": "validation_only_collapsed_constant_compatibility",
            "same_kernel_family_for_monotonic_and_fatigue": True,
            "constitutive_K_shield_cap_present": False,
            "signed_burgers_population_required": True,
            "full_mpz_grid_values_are_spatial_projection": True,
            "direct_fem_measurements_exist_only_at_recorded_station_indices": True,
            "frozen_geometry_load_invariance_passed": True,
            "active_kernel_mechanically_measured": True,
            "wake_kernel_mechanically_measured": False,
            "wake_shielding_supported": False,
        }
        for key, expected in required.items():
            if payload.get(key) != expected:
                raise ValueError(f"v10.2.14 atlas requires {key}={expected!r}")
        opening_coordinate = float(
            payload.get(
                "kernel_opening_compatibility_coordinate",
                OPENING_COMPATIBILITY_COORDINATE,
            )
        )
        if not np.isclose(
            opening_coordinate,
            OPENING_COMPATIBILITY_COORDINATE,
            rtol=0.0,
            atol=1.0e-15,
        ):
            raise ValueError("invalid opening compatibility coordinate")
        openings = {
            float(row["opening_strength_fraction"])
            for row in payload.get("states", [])
        }
        if openings != {OPENING_COMPATIBILITY_COORDINATE}:
            raise ValueError("all v10.2.14 states must use constant opening")
        extensions = {
            float(row["crack_extension_m"])
            for row in payload.get("states", [])
        }
        if len(extensions) < 2:
            raise ValueError("v10.2.14 atlas requires at least two path extensions")

        compatibility = copy.deepcopy(payload)
        compatibility["schema"] = V10212_SCHEMA
        compatibility.setdefault("finite_radius_fem_kernel_claimed", False)
        compatibility.setdefault(
            "kernel_radius_compatibility_coordinate",
            KERNEL_RADIUS_COMPATIBILITY_COORDINATE,
        )
        with tempfile.TemporaryDirectory(prefix="v10214_family_load_") as temp_dir:
            temporary = Path(temp_dir) / "v10212_compatibility.json"
            temporary.write_text(json.dumps(compatibility))
            base = _V10212Family.from_json(temporary)

        for state in base.states:
            if np.any(np.abs(state.wake_I) > 0.0) or np.any(np.abs(state.wake_II) > 0.0):
                raise ValueError(
                    "v10.2.14 active-only atlas requires identically zero wake kernels"
                )

        family = cls(
            states=[
                KernelState(
                    state_id=state.state_id,
                    coordinates=state.coordinates.copy(),
                    active_I=state.active_I.copy(),
                    wake_I=np.zeros_like(state.wake_I),
                    active_II=state.active_II.copy(),
                    wake_II=np.zeros_like(state.wake_II),
                    metadata=copy.deepcopy(state.metadata),
                )
                for state in base.states
            ],
            active_x_m=base.active_x_m.copy(),
            wake_x_m=base.wake_x_m.copy(),
            activation_to_line_content=base.activation_to_line_content.copy(),
            source_capacity_bounds=base.source_capacity_bounds.copy(),
            fixed_kernel_assessment=copy.deepcopy(base.fixed_kernel_assessment),
            interpolation=copy.deepcopy(base.interpolation),
            metadata={key: value for key, value in payload.items() if key != "states"},
            source_path=str(source),
        )
        family._opening_boundary_policy = {"policy": "strict"}
        family._last_boundary_action = "none"
        family._last_observed_analytical_r_eff_over_r0 = 1.0
        family._last_observed_opening_strength_fraction = 0.0
        family._validate_complete_grid()
        return family

    def clone_for_engine(self) -> "ActiveOnlySigned2DShieldingKernelFamily":
        family = copy.deepcopy(self)
        family._last_boundary_action = "none"
        family._last_observed_analytical_r_eff_over_r0 = 1.0
        family._last_observed_opening_strength_fraction = 0.0
        return family

    def _prepare_query(
        self,
        r_eff_over_r0: float,
        opening_strength_fraction: float,
        crack_extension_m: float,
    ) -> np.ndarray:
        observed_r = float(r_eff_over_r0)
        observed_opening = float(opening_strength_fraction)
        if not np.isfinite(observed_r) or observed_r <= 0.0:
            raise ValueError("observed analytical radius must be positive and finite")
        if not np.isfinite(observed_opening) or not 0.0 <= observed_opening <= 1.0:
            raise ValueError("observed opening fraction must lie in [0,1]")
        query = super()._prepare_query(
            KERNEL_RADIUS_COMPATIBILITY_COORDINATE,
            OPENING_COMPATIBILITY_COORDINATE,
            float(crack_extension_m),
        )
        self._last_observed_analytical_r_eff_over_r0 = observed_r
        self._last_observed_opening_strength_fraction = observed_opening
        return query

    def audit_payload(self) -> dict[str, Any]:
        payload = super().audit_payload()
        payload.update(
            {
                "schema": SCHEMA,
                "opening_axis_policy": "validation_only_collapsed_constant_compatibility",
                "kernel_opening_compatibility_coordinate": OPENING_COMPATIBILITY_COORDINATE,
                "active_physical_kernel_axes": ["cumulative_crack_path_extension_m"],
                "crack_extension_m_semantics": "cumulative_crack_path_extension_m",
                "opening_strength_fraction_used_for_interpolation": False,
                "analytical_r_eff_used_for_interpolation": False,
                "active_kernel_mechanically_measured": True,
                "wake_kernel_mechanically_measured": False,
                "wake_shielding_supported": False,
                "wake_kernel_forced_zero": True,
            }
        )
        return payload


StateResolvedSignedShieldingKernelFamily = ActiveOnlySigned2DShieldingKernelFamily

__all__ = [
    "SCHEMA",
    "STATE_AXES",
    "KernelState",
    "OPENING_COMPATIBILITY_COORDINATE",
    "ActiveOnlySigned2DShieldingKernelFamily",
    "StateResolvedSignedShieldingKernelFamily",
]
