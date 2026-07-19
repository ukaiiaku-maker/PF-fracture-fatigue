"""Production loader for the v10.2.12 real signed 2-D shielding atlas."""
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
from .signed_kernel_family_v1029 import (
    KernelState,
    STATE_AXES,
    SCHEMA as V1029_SCHEMA,
    StateResolvedSignedShieldingKernelFamily as _V1029Family,
)

SCHEMA = "v10.2.12_real_signed_state_resolved_2d_shielding_atlas"


class RealSigned2DShieldingKernelFamily(_V1029Family):
    """Use opening and crack extension as the physical atlas coordinates.

    ``r_eff_over_r0`` is retained in the serialized coordinate vector solely for
    backward-compatible shape. The runtime analytical radius is recorded but is
    replaced by the constant compatibility coordinate before interpolation.
    """

    @classmethod
    def from_json(cls, path: str | Path) -> "RealSigned2DShieldingKernelFamily":
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text())
        if payload.get("schema") != SCHEMA:
            raise ValueError(
                f"v10.2.12 kernel-family schema must be {SCHEMA!r}; "
                f"got {payload.get('schema')!r}"
            )
        required = {
            "kernel_radius_axis_policy": "disabled_constant_compatibility",
            "finite_radius_fem_kernel_claimed": False,
            "same_kernel_family_for_monotonic_and_fatigue": True,
            "constitutive_K_shield_cap_present": False,
            "signed_burgers_population_required": True,
            "full_mpz_grid_values_are_spatial_projection": True,
            "direct_fem_measurements_exist_only_at_recorded_station_indices": True,
        }
        for key, expected in required.items():
            if payload.get(key) != expected:
                raise ValueError(f"v10.2.12 atlas requires {key}={expected!r}")
        coordinate = float(
            payload.get(
                "kernel_radius_compatibility_coordinate",
                KERNEL_RADIUS_COMPATIBILITY_COORDINATE,
            )
        )
        if not np.isclose(
            coordinate,
            KERNEL_RADIUS_COMPATIBILITY_COORDINATE,
            rtol=0.0,
            atol=1.0e-15,
        ):
            raise ValueError("invalid radius compatibility coordinate")
        radii = {
            float(row["r_eff_over_r0"]) for row in payload.get("states", [])
        }
        if radii != {KERNEL_RADIUS_COMPATIBILITY_COORDINATE}:
            raise ValueError(
                "all v10.2.12 kernel states must use the constant radius "
                f"coordinate {KERNEL_RADIUS_COMPATIBILITY_COORDINATE}"
            )

        compatibility = copy.deepcopy(payload)
        compatibility["schema"] = V1029_SCHEMA
        with tempfile.TemporaryDirectory(prefix="v10212_family_load_") as temp_dir:
            temporary = Path(temp_dir) / "v1029_compatibility.json"
            temporary.write_text(json.dumps(compatibility))
            base = _V1029Family.from_json(temporary)

        family = cls(
            states=[
                KernelState(
                    state_id=state.state_id,
                    coordinates=state.coordinates.copy(),
                    active_I=state.active_I.copy(),
                    wake_I=state.wake_I.copy(),
                    active_II=state.active_II.copy(),
                    wake_II=state.wake_II.copy(),
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
        family._opening_boundary_policy = copy.deepcopy(
            getattr(base, "_opening_boundary_policy", {"policy": "strict"})
        )
        family._last_boundary_action = "none"
        family._last_observed_analytical_r_eff_over_r0 = 1.0
        family._validate_complete_grid()
        return family

    def clone_for_engine(self) -> "RealSigned2DShieldingKernelFamily":
        family = copy.deepcopy(self)
        family._last_boundary_action = "none"
        family._last_observed_analytical_r_eff_over_r0 = 1.0
        return family

    def _prepare_query(
        self,
        r_eff_over_r0: float,
        opening_strength_fraction: float,
        crack_extension_m: float,
    ) -> np.ndarray:
        observed = float(r_eff_over_r0)
        if not np.isfinite(observed) or observed <= 0.0:
            raise ValueError("observed analytical r_eff_over_r0 must be positive and finite")
        self._last_observed_analytical_r_eff_over_r0 = observed
        return super()._prepare_query(
            KERNEL_RADIUS_COMPATIBILITY_COORDINATE,
            float(opening_strength_fraction),
            float(crack_extension_m),
        )

    def audit_payload(self) -> dict[str, Any]:
        payload = super().audit_payload()
        payload.update(
            {
                "schema": SCHEMA,
                "kernel_radius_axis_policy": "disabled_constant_compatibility",
                "kernel_radius_compatibility_coordinate": (
                    KERNEL_RADIUS_COMPATIBILITY_COORDINATE
                ),
                "active_physical_kernel_axes": [
                    "opening_strength_fraction",
                    "crack_extension_m",
                ],
                "last_observed_analytical_r_eff_over_r0": float(
                    getattr(self, "_last_observed_analytical_r_eff_over_r0", 1.0)
                ),
                "analytical_r_eff_used_for_interpolation": False,
                "finite_radius_fem_geometry_claimed": False,
            }
        )
        return payload


StateResolvedSignedShieldingKernelFamily = RealSigned2DShieldingKernelFamily

__all__ = [
    "SCHEMA",
    "STATE_AXES",
    "KernelState",
    "RealSigned2DShieldingKernelFamily",
    "StateResolvedSignedShieldingKernelFamily",
]
