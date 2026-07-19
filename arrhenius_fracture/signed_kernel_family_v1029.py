"""v10.2.9 state-resolved signed shielding family with explicit boundaries.

The v10.2.6 resolver used a rectangular min/max envelope.  v10.2.9 retains
fail-closed behavior for tip radius and crack extension and permits opening-axis
clamping only when the artifact explicitly demonstrates boundary stationarity.
A complete Cartesian state grid is required so inverse-distance interpolation
cannot bridge an unsampled corner of the nominal envelope.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from .signed_kernel_family_v1026 import (
    KernelState,
    STATE_AXES,
    StateResolvedSignedShieldingKernelFamily as _BaseFamily,
)

SCHEMA = "v10.2.9_state_resolved_signed_kernel_family"


class StateResolvedSignedShieldingKernelFamily(_BaseFamily):
    """Per-engine state resolver with reviewed saturation and grid coverage."""

    @classmethod
    def from_json(
        cls, path: str | Path
    ) -> "StateResolvedSignedShieldingKernelFamily":
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text())
        if payload.get("schema") != SCHEMA:
            raise ValueError(
                f"v10.2.9 kernel-family schema must be {SCHEMA!r}; "
                f"got {payload.get('schema')!r}"
            )
        required_truth = {
            "candidate_independent": True,
            "counts_are_signed_burgers_lines": True,
            "kernel_from_signed_interaction_integral": True,
            "analytic_auxiliary_gradients": True,
            "hermite_domain_weight": True,
            "positive_and_negative_perturbations": True,
            "multiple_perturbation_magnitudes": True,
            "multi_amplitude_validation_passed": True,
            "normalization_is_mechanically_derived": True,
            "fitted_attenuation_factor": False,
            "constitutive_K_shield_cap": False,
            "complete_cartesian_state_grid": True,
        }
        for key, expected in required_truth.items():
            if payload.get(key) is not expected:
                raise ValueError(f"kernel-family metadata requires {key}={expected}")
        if tuple(payload.get("state_axes", ())) != STATE_AXES:
            raise ValueError(f"kernel-family state axes must be {STATE_AXES}")
        if str(payload.get("normalization_source", "")) not in {
            "2d_unit_slip_to_line_content",
            "plastic_distortion_burgers_integral",
            "process_zone_geometry_and_line_spacing",
            "front_thickness_source_geometry",
        }:
            raise ValueError("normalization source is not a recognized mechanical derivation")

        boundary = dict(payload.get("opening_boundary_policy", {}))
        policy = str(boundary.get("policy", "strict"))
        if policy not in {"strict", "validated_boundary_saturation"}:
            raise ValueError("invalid opening boundary policy")
        if policy == "validated_boundary_saturation":
            if not bool(boundary.get("lower_boundary_validated", False)):
                raise ValueError("lower opening-boundary saturation is not validated")
            if not bool(boundary.get("upper_boundary_validated", False)):
                raise ValueError("upper opening-boundary saturation is not validated")

        active_x = np.asarray(payload["active_x_m"], dtype=float)
        wake_x = np.asarray(payload.get("wake_x_m", []), dtype=float)
        conversion = np.asarray(
            payload["activation_to_line_content_by_system"], dtype=float
        )
        bounds = np.asarray(
            payload["source_capacity_bounds_per_system"], dtype=float
        )
        states: list[KernelState] = []
        for index, row in enumerate(payload.get("states", [])):
            coordinates = np.asarray(
                [float(row[name]) for name in STATE_AXES], dtype=float
            )
            active_I = np.asarray(
                row["active_kernel_I_Pa_sqrt_m_per_signed_line"], dtype=float
            )
            wake_I = np.asarray(
                row.get("wake_kernel_I_Pa_sqrt_m_per_signed_line", []), dtype=float
            )
            active_II = np.asarray(
                row.get(
                    "active_kernel_II_Pa_sqrt_m_per_signed_line",
                    np.zeros_like(active_I),
                ),
                dtype=float,
            )
            wake_II = np.asarray(
                row.get(
                    "wake_kernel_II_Pa_sqrt_m_per_signed_line",
                    np.zeros_like(wake_I),
                ),
                dtype=float,
            )
            if wake_I.size == 0:
                wake_I = np.zeros((active_I.shape[0], 0), dtype=float)
            if wake_II.size == 0:
                wake_II = np.zeros_like(wake_I)
            states.append(
                KernelState(
                    state_id=str(row.get("state_id", f"state_{index:04d}")),
                    coordinates=coordinates,
                    active_I=active_I,
                    wake_I=wake_I,
                    active_II=active_II,
                    wake_II=wake_II,
                    metadata={
                        key: value
                        for key, value in row.items()
                        if key
                        not in {
                            *STATE_AXES,
                            "active_kernel_I_Pa_sqrt_m_per_signed_line",
                            "wake_kernel_I_Pa_sqrt_m_per_signed_line",
                            "active_kernel_II_Pa_sqrt_m_per_signed_line",
                            "wake_kernel_II_Pa_sqrt_m_per_signed_line",
                        }
                    },
                )
            )
        family = cls(
            states=states,
            active_x_m=active_x,
            wake_x_m=wake_x,
            activation_to_line_content=conversion,
            source_capacity_bounds=bounds,
            fixed_kernel_assessment=dict(payload["fixed_kernel_assessment"]),
            interpolation=dict(payload.get("interpolation", {})),
            metadata={key: value for key, value in payload.items() if key != "states"},
            source_path=str(source),
        )
        family._opening_boundary_policy = boundary
        family._last_boundary_action = "none"
        family._validate_complete_grid()
        return family

    def clone_for_engine(self) -> "StateResolvedSignedShieldingKernelFamily":
        family = type(self)(
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
                for state in self.states
            ],
            active_x_m=self.active_x_m.copy(),
            wake_x_m=self.wake_x_m.copy(),
            activation_to_line_content=self.activation_to_line_content.copy(),
            source_capacity_bounds=self.source_capacity_bounds.copy(),
            fixed_kernel_assessment=copy.deepcopy(self.fixed_kernel_assessment),
            interpolation=copy.deepcopy(self.interpolation),
            metadata=copy.deepcopy(self.metadata),
            source_path=self.source_path,
        )
        family._opening_boundary_policy = copy.deepcopy(
            getattr(self, "_opening_boundary_policy", {"policy": "strict"})
        )
        family._last_boundary_action = "none"
        family._validate_complete_grid()
        return family

    def _validate_complete_grid(self) -> None:
        levels = [np.unique(self._coordinates[:, i]) for i in range(3)]
        expected = int(np.prod([len(level) for level in levels]))
        tuples = {tuple(float(v) for v in row) for row in self._coordinates}
        if len(self.states) != expected or len(tuples) != expected:
            raise ValueError(
                "state-resolved family does not contain a complete Cartesian grid"
            )
        for r in levels[0]:
            for opening in levels[1]:
                for extension in levels[2]:
                    if (float(r), float(opening), float(extension)) not in tuples:
                        raise ValueError("state-resolved family has an unsampled grid corner")

    def _prepare_query(
        self,
        r_eff_over_r0: float,
        opening_strength_fraction: float,
        crack_extension_m: float,
    ) -> np.ndarray:
        query = np.asarray(
            [r_eff_over_r0, opening_strength_fraction, crack_extension_m],
            dtype=float,
        )
        if np.any(~np.isfinite(query)):
            raise ValueError("kernel-family state coordinates must be finite")
        tolerance = float(
            self.interpolation.get("envelope_relative_tolerance", 1.0e-10)
        )
        allowance = tolerance * np.maximum(np.abs(self._scale), 1.0)
        # Tip radius and crack extension are never saturated or extrapolated.
        for axis in (0, 2):
            if query[axis] < self._minimum[axis] - allowance[axis] or query[axis] > self._maximum[axis] + allowance[axis]:
                raise RuntimeError(
                    "requested crack-tip state lies outside the validated signed-kernel "
                    f"envelope on {STATE_AXES[axis]}: query={query.tolist()} "
                    f"min={self._minimum.tolist()} max={self._maximum.tolist()}"
                )
        query[0] = np.clip(query[0], self._minimum[0], self._maximum[0])
        query[2] = np.clip(query[2], self._minimum[2], self._maximum[2])

        boundary = getattr(self, "_opening_boundary_policy", {"policy": "strict"})
        policy = str(boundary.get("policy", "strict"))
        self._last_boundary_action = "none"
        if query[1] < self._minimum[1] - allowance[1]:
            if policy != "validated_boundary_saturation" or not bool(
                boundary.get("lower_boundary_validated", False)
            ):
                raise RuntimeError("opening state is below the validated kernel envelope")
            query[1] = self._minimum[1]
            self._last_boundary_action = "lower_saturation"
        elif query[1] > self._maximum[1] + allowance[1]:
            if policy != "validated_boundary_saturation" or not bool(
                boundary.get("upper_boundary_validated", False)
            ):
                raise RuntimeError("opening state is above the validated kernel envelope")
            query[1] = self._maximum[1]
            self._last_boundary_action = "upper_saturation"
        else:
            query[1] = np.clip(query[1], self._minimum[1], self._maximum[1])
        return query

    def resolve(
        self,
        *,
        r_eff_over_r0: float,
        opening_strength_fraction: float,
        crack_extension_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        query = self._prepare_query(
            float(r_eff_over_r0),
            float(opening_strength_fraction),
            float(crack_extension_m),
        )
        # The base resolver sees an already validated/clamped query.
        return super().resolve(
            r_eff_over_r0=float(query[0]),
            opening_strength_fraction=float(query[1]),
            crack_extension_m=float(query[2]),
        )

    def audit_payload(self) -> dict[str, Any]:
        payload = super().audit_payload()
        payload.update(
            {
                "schema": SCHEMA,
                "complete_cartesian_state_grid": True,
                "opening_boundary_policy": copy.deepcopy(
                    getattr(self, "_opening_boundary_policy", {"policy": "strict"})
                ),
                "last_opening_boundary_action": getattr(
                    self, "_last_boundary_action", "none"
                ),
            }
        )
        return payload


__all__ = [
    "SCHEMA",
    "STATE_AXES",
    "KernelState",
    "StateResolvedSignedShieldingKernelFamily",
]
