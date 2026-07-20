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


def _resample_rows(
    source_x: np.ndarray,
    values: np.ndarray,
    target_x: np.ndarray,
) -> np.ndarray:
    """Evaluate one measured spatial operator on another cell-centre grid.

    The atlas stores a physical kernel as a function of distance from the tip;
    the number of atlas stations is therefore not a constitutive state variable.
    Runtime MPZ grids may be finer, or may cover a shorter physical process zone.
    Linear interpolation is used in the measured interval.  A target cell centre
    may extend from the first/last source centre to the corresponding physical
    cell boundary, in which case the nearest measured slope is used.
    """
    sx = np.asarray(source_x, dtype=float).reshape(-1)
    tx = np.asarray(target_x, dtype=float).reshape(-1)
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != sx.size:
        raise ValueError("kernel rows do not match source spatial coordinates")
    if sx.size < 2:
        raise ValueError("runtime grid binding requires at least two atlas stations")
    if tx.size < 1:
        return np.zeros((matrix.shape[0], 0), dtype=float)
    if np.any(~np.isfinite(sx)) or np.any(~np.isfinite(tx)) or np.any(~np.isfinite(matrix)):
        raise ValueError("runtime grid binding requires finite coordinates and kernels")
    if np.any(np.diff(sx) <= 0.0) or np.any(np.diff(tx) <= 0.0):
        raise ValueError("spatial coordinates must be strictly increasing")

    left_dx = float(sx[1] - sx[0])
    right_dx = float(sx[-1] - sx[-2])
    left_boundary = float(sx[0] - 0.5 * left_dx)
    right_boundary = float(sx[-1] + 0.5 * right_dx)
    tolerance = max(1.0e-15, 1.0e-10 * max(abs(left_boundary), abs(right_boundary), 1.0))
    if float(tx[0]) < left_boundary - tolerance or float(tx[-1]) > right_boundary + tolerance:
        raise ValueError(
            "runtime MPZ grid lies outside the measured active-kernel support: "
            f"target=[{tx[0]:.9e},{tx[-1]:.9e}] "
            f"support=[{left_boundary:.9e},{right_boundary:.9e}]"
        )

    output = np.empty((matrix.shape[0], tx.size), dtype=float)
    for system, row in enumerate(matrix):
        interpolated = np.interp(tx, sx, row)
        left = tx < sx[0]
        right = tx > sx[-1]
        if np.any(left):
            slope = float((row[1] - row[0]) / left_dx)
            interpolated[left] = row[0] + slope * (tx[left] - sx[0])
        if np.any(right):
            slope = float((row[-1] - row[-2]) / right_dx)
            interpolated[right] = row[-1] + slope * (tx[right] - sx[-1])
        output[system] = interpolated
    return output


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
        family._runtime_grid_binding = None
        family._validate_complete_grid()
        return family

    def clone_for_engine(self) -> "ActiveOnlySigned2DShieldingKernelFamily":
        family = copy.deepcopy(self)
        family._last_boundary_action = "none"
        family._last_observed_analytical_r_eff_over_r0 = 1.0
        family._last_observed_opening_strength_fraction = 0.0
        return family

    def bind_to_state_grid(self, state) -> "ActiveOnlySigned2DShieldingKernelFamily":
        """Return a per-engine family evaluated on the runtime MPZ discretization."""
        if int(state.n_systems) != self.n_systems:
            raise ValueError(
                f"runtime MPZ systems {state.n_systems} != atlas systems {self.n_systems}"
            )
        target_active = np.asarray(state.x, dtype=float).copy()
        target_wake = np.asarray(state.wake_x, dtype=float).copy()
        source_active = self.active_x_m.copy()
        source_wake = self.wake_x_m.copy()
        bound_states = []
        for kernel_state in self.states:
            bound_states.append(
                KernelState(
                    state_id=kernel_state.state_id,
                    coordinates=kernel_state.coordinates.copy(),
                    active_I=_resample_rows(
                        source_active, kernel_state.active_I, target_active
                    ),
                    wake_I=np.zeros((self.n_systems, target_wake.size), dtype=float),
                    active_II=_resample_rows(
                        source_active, kernel_state.active_II, target_active
                    ),
                    wake_II=np.zeros((self.n_systems, target_wake.size), dtype=float),
                    metadata={
                        **copy.deepcopy(kernel_state.metadata),
                        "runtime_grid_resampled": True,
                        "source_active_bins": int(source_active.size),
                        "runtime_active_bins": int(target_active.size),
                    },
                )
            )
        bound = type(self)(
            states=bound_states,
            active_x_m=target_active,
            wake_x_m=target_wake,
            activation_to_line_content=self.activation_to_line_content.copy(),
            source_capacity_bounds=self.source_capacity_bounds.copy(),
            fixed_kernel_assessment=copy.deepcopy(self.fixed_kernel_assessment),
            interpolation=copy.deepcopy(self.interpolation),
            metadata={
                **copy.deepcopy(self.metadata),
                "runtime_grid_binding": {
                    "model_id": "v10.2.15_physical_x_kernel_to_runtime_mpz_grid",
                    "source_active_bins": int(source_active.size),
                    "runtime_active_bins": int(target_active.size),
                    "source_active_x_min_m": float(source_active[0]),
                    "source_active_x_max_m": float(source_active[-1]),
                    "runtime_active_x_min_m": float(target_active[0]),
                    "runtime_active_x_max_m": float(target_active[-1]),
                    "runtime_wake_bins": int(target_wake.size),
                    "active_interpolation": "piecewise_linear_in_physical_x",
                    "wake_operator": "identically_zero",
                },
            },
            source_path=self.source_path,
        )
        bound._opening_boundary_policy = copy.deepcopy(self._opening_boundary_policy)
        bound._last_boundary_action = "none"
        bound._last_observed_analytical_r_eff_over_r0 = 1.0
        bound._last_observed_opening_strength_fraction = 0.0
        bound._runtime_grid_binding = copy.deepcopy(
            bound.metadata["runtime_grid_binding"]
        )
        bound._validate_complete_grid()
        return bound

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
                "runtime_grid_binding": copy.deepcopy(
                    getattr(self, "_runtime_grid_binding", None)
                ),
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
