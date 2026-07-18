"""State-resolved signed shielding-kernel family for v10.2.6.

The family is a candidate-independent mechanical artifact.  It stores signed
mode-I and mode-II unit-response operators at multiple crack-tip states and
interpolates only inside the validated state envelope.  The mode-I operator is
consumed by the current cleavage-driving law; mode II is retained for mixed-mode
audit and future direct coupling.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "v10.2.6_state_resolved_signed_kernel_family"
STATE_AXES = (
    "r_eff_over_r0",
    "opening_strength_fraction",
    "crack_extension_m",
)


@dataclass(frozen=True)
class KernelState:
    state_id: str
    coordinates: np.ndarray
    active_I: np.ndarray
    wake_I: np.ndarray
    active_II: np.ndarray
    wake_II: np.ndarray
    metadata: dict[str, Any]


class StateResolvedSignedShieldingKernelFamily:
    """Mutable per-engine resolver with a validated immutable state atlas.

    The public attributes ``active_kernel`` and ``wake_kernel`` intentionally
    match the v10.2.5 static-kernel interface.  They contain the most recently
    resolved mode-I operators, so the existing signed population and transport
    implementation remains shared between monotonic fracture and fatigue.
    """

    def __init__(
        self,
        *,
        states: list[KernelState],
        active_x_m: np.ndarray,
        wake_x_m: np.ndarray,
        activation_to_line_content: np.ndarray,
        source_capacity_bounds: np.ndarray,
        fixed_kernel_assessment: dict[str, Any],
        interpolation: dict[str, Any],
        metadata: dict[str, Any],
        source_path: str,
    ) -> None:
        if not states:
            raise ValueError("state-resolved kernel family requires at least one state")
        self.states = tuple(states)
        self.active_x_m = np.asarray(active_x_m, dtype=float).copy()
        self.wake_x_m = np.asarray(wake_x_m, dtype=float).copy()
        self.activation_to_line_content = np.asarray(
            activation_to_line_content, dtype=float
        ).reshape(-1)
        self.source_capacity_bounds = np.asarray(
            source_capacity_bounds, dtype=float
        ).copy()
        self.fixed_kernel_assessment = copy.deepcopy(fixed_kernel_assessment)
        self.interpolation = copy.deepcopy(interpolation)
        self.metadata = copy.deepcopy(metadata)
        self.source_path = str(source_path)
        self._coordinates = np.vstack([state.coordinates for state in self.states])
        self._minimum = np.min(self._coordinates, axis=0)
        self._maximum = np.max(self._coordinates, axis=0)
        self._scale = np.maximum(self._maximum - self._minimum, 1.0e-15)
        self._last_query = self.states[0].coordinates.copy()
        self._last_weights = np.zeros(len(self.states), dtype=float)
        self._last_weights[0] = 1.0
        self._last_state_ids = [self.states[0].state_id]
        self.active_kernel = self.states[0].active_I.copy()
        self.wake_kernel = self.states[0].wake_I.copy()
        self.active_kernel_II = self.states[0].active_II.copy()
        self.wake_kernel_II = self.states[0].wake_II.copy()
        self._validate_internal()

    @classmethod
    def from_json(
        cls, path: str | Path
    ) -> "StateResolvedSignedShieldingKernelFamily":
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text())
        if payload.get("schema") != SCHEMA:
            raise ValueError(
                f"state-resolved kernel schema must be {SCHEMA!r}; "
                f"got {payload.get('schema')!r}"
            )
        required_truth = {
            "candidate_independent": True,
            "counts_are_signed_burgers_lines": True,
            "kernel_from_signed_interaction_integral": True,
            "positive_and_negative_perturbations": True,
            "multiple_perturbation_magnitudes": True,
            "multi_amplitude_validation_passed": True,
            "normalization_is_mechanically_derived": True,
            "fitted_attenuation_factor": False,
            "constitutive_K_shield_cap": False,
        }
        for key, expected in required_truth.items():
            if payload.get(key) is not expected:
                raise ValueError(f"kernel-family metadata requires {key}={expected}")
        axes = tuple(payload.get("state_axes", ()))
        if axes != STATE_AXES:
            raise ValueError(
                f"kernel-family state axes must be {STATE_AXES}; got {axes}"
            )
        normalization_source = str(payload.get("normalization_source", ""))
        if normalization_source not in {
            "2d_unit_slip_to_line_content",
            "plastic_distortion_burgers_integral",
            "process_zone_geometry_and_line_spacing",
            "front_thickness_source_geometry",
        }:
            raise ValueError("normalization source is not a recognized mechanical derivation")

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
            state_id = str(row.get("state_id", f"state_{index:04d}"))
            states.append(
                KernelState(
                    state_id=state_id,
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
        return cls(
            states=states,
            active_x_m=active_x,
            wake_x_m=wake_x,
            activation_to_line_content=conversion,
            source_capacity_bounds=bounds,
            fixed_kernel_assessment=dict(payload["fixed_kernel_assessment"]),
            interpolation=dict(payload.get("interpolation", {})),
            metadata={
                key: value
                for key, value in payload.items()
                if key not in {"states"}
            },
            source_path=str(source),
        )

    def clone_for_engine(self) -> "StateResolvedSignedShieldingKernelFamily":
        return StateResolvedSignedShieldingKernelFamily(
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

    @property
    def n_systems(self) -> int:
        return int(self.states[0].active_I.shape[0])

    def _validate_internal(self) -> None:
        reference = self.states[0]
        if reference.active_I.ndim != 2 or reference.wake_I.ndim != 2:
            raise ValueError("kernel-family operators must be two-dimensional")
        if self.active_x_m.shape != (reference.active_I.shape[1],):
            raise ValueError("active coordinates do not match kernel bins")
        if self.wake_x_m.shape != (reference.wake_I.shape[1],):
            raise ValueError("wake coordinates do not match kernel bins")
        if self.activation_to_line_content.shape != (self.n_systems,):
            raise ValueError("activation-to-line conversion must have one value per system")
        if np.any(~np.isfinite(self.activation_to_line_content)) or np.any(
            self.activation_to_line_content <= 0.0
        ):
            raise ValueError("activation-to-line conversion must be positive and finite")
        if self.source_capacity_bounds.shape != (self.n_systems, 2):
            raise ValueError("source-capacity bounds must have shape (n_systems, 2)")
        if np.any(~np.isfinite(self.source_capacity_bounds)) or np.any(
            self.source_capacity_bounds[:, 0] < 0.0
        ) or np.any(
            self.source_capacity_bounds[:, 1]
            < self.source_capacity_bounds[:, 0]
        ):
            raise ValueError("invalid source-capacity bounds")
        for state in self.states:
            for array in (
                state.coordinates,
                state.active_I,
                state.wake_I,
                state.active_II,
                state.wake_II,
            ):
                if np.any(~np.isfinite(array)):
                    raise ValueError(f"non-finite data in kernel state {state.state_id}")
            if state.coordinates.shape != (len(STATE_AXES),):
                raise ValueError("kernel state has the wrong coordinate dimension")
            if state.active_I.shape != reference.active_I.shape:
                raise ValueError("active mode-I kernel shapes differ across states")
            if state.wake_I.shape != reference.wake_I.shape:
                raise ValueError("wake mode-I kernel shapes differ across states")
            if state.active_II.shape != reference.active_I.shape:
                raise ValueError("active mode-II kernel shape mismatch")
            if state.wake_II.shape != reference.wake_I.shape:
                raise ValueError("wake mode-II kernel shape mismatch")
        interpolation_mode = str(self.interpolation.get("method", "")).strip()
        fixed_accepted = bool(
            self.fixed_kernel_assessment.get("fixed_kernel_accepted", False)
        )
        if not fixed_accepted and interpolation_mode != "inverse_distance":
            raise ValueError(
                "state-dependent family requires inverse_distance interpolation"
            )
        if len(self.states) == 1 and not fixed_accepted:
            raise ValueError(
                "one-state kernel cannot claim state dependence without a validated fixed kernel"
            )

    def validate_state(self, state) -> None:
        expected_active = (state.n_systems, state.n_bins)
        expected_wake = (state.n_systems, state.wake_n_bins)
        if self.states[0].active_I.shape != expected_active:
            raise ValueError(
                f"active kernel shape {self.states[0].active_I.shape} != {expected_active}"
            )
        if self.states[0].wake_I.shape != expected_wake:
            raise ValueError(
                f"wake kernel shape {self.states[0].wake_I.shape} != {expected_wake}"
            )
        if not np.allclose(
            self.active_x_m, state.x, rtol=1.0e-12, atol=1.0e-18
        ):
            raise ValueError("active family coordinates do not match production MPZ grid")
        if not np.allclose(
            self.wake_x_m, state.wake_x, rtol=1.0e-12, atol=1.0e-18
        ):
            raise ValueError("wake family coordinates do not match production wake grid")
        capacity = np.asarray(state.site_capacity, dtype=float)
        lo = self.source_capacity_bounds[:, 0]
        hi = self.source_capacity_bounds[:, 1]
        if np.any(capacity < lo) or np.any(capacity > hi):
            raise ValueError(
                "manifest source_sites_per_system lies outside the mechanically "
                "derived source-capacity range; rebuild the search region"
            )

    def resolve(
        self,
        *,
        r_eff_over_r0: float,
        opening_strength_fraction: float,
        crack_extension_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(
            [
                float(r_eff_over_r0),
                float(opening_strength_fraction),
                float(crack_extension_m),
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(query)):
            raise ValueError("kernel-family state coordinates must be finite")
        envelope_tolerance = float(
            self.interpolation.get("envelope_relative_tolerance", 1.0e-10)
        )
        allowance = envelope_tolerance * np.maximum(np.abs(self._scale), 1.0)
        if np.any(query < self._minimum - allowance) or np.any(
            query > self._maximum + allowance
        ):
            raise RuntimeError(
                "requested crack-tip state lies outside the validated signed-kernel "
                f"envelope: query={query.tolist()} min={self._minimum.tolist()} "
                f"max={self._maximum.tolist()}"
            )
        query = np.minimum(np.maximum(query, self._minimum), self._maximum)
        normalized = (self._coordinates - query[None, :]) / self._scale[None, :]
        distance = np.linalg.norm(normalized, axis=1)
        exact = np.flatnonzero(distance <= 1.0e-12)
        if exact.size:
            weights = np.zeros(len(self.states), dtype=float)
            weights[int(exact[0])] = 1.0
        elif bool(self.fixed_kernel_assessment.get("fixed_kernel_accepted", False)):
            reference_id = str(
                self.fixed_kernel_assessment.get(
                    "reference_state_id", self.states[0].state_id
                )
            )
            index = next(
                (
                    i
                    for i, state in enumerate(self.states)
                    if state.state_id == reference_id
                ),
                0,
            )
            weights = np.zeros(len(self.states), dtype=float)
            weights[index] = 1.0
        else:
            neighbors = max(
                2,
                min(
                    int(self.interpolation.get("neighbors", 8)),
                    len(self.states),
                ),
            )
            indices = np.argsort(distance)[:neighbors]
            power = max(float(self.interpolation.get("power", 2.0)), 0.1)
            local = 1.0 / np.maximum(distance[indices], 1.0e-15) ** power
            local /= np.sum(local)
            weights = np.zeros(len(self.states), dtype=float)
            weights[indices] = local

        self.active_kernel = sum(
            weight * state.active_I
            for weight, state in zip(weights, self.states)
        )
        self.wake_kernel = sum(
            weight * state.wake_I
            for weight, state in zip(weights, self.states)
        )
        self.active_kernel_II = sum(
            weight * state.active_II
            for weight, state in zip(weights, self.states)
        )
        self.wake_kernel_II = sum(
            weight * state.wake_II
            for weight, state in zip(weights, self.states)
        )
        self._last_query = query.copy()
        self._last_weights = weights.copy()
        self._last_state_ids = [
            state.state_id
            for weight, state in zip(weights, self.states)
            if weight > 0.0
        ]
        return self.active_kernel, self.wake_kernel

    def audit_payload(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "source_path": self.source_path,
            "state_axes": list(STATE_AXES),
            "n_states": len(self.states),
            "state_envelope_min": self._minimum.tolist(),
            "state_envelope_max": self._maximum.tolist(),
            "fixed_kernel_assessment": copy.deepcopy(
                self.fixed_kernel_assessment
            ),
            "interpolation": copy.deepcopy(self.interpolation),
            "last_query": self._last_query.tolist(),
            "last_state_ids": list(self._last_state_ids),
            "last_weights": self._last_weights.tolist(),
            "active_shape": list(self.active_kernel.shape),
            "wake_shape": list(self.wake_kernel.shape),
            "activation_to_line_content_by_system": (
                self.activation_to_line_content.tolist()
            ),
            "source_capacity_bounds_per_system": (
                self.source_capacity_bounds.tolist()
            ),
            "mode_II_operator_retained": True,
            "constitutive_K_shield_cap": False,
        }


__all__ = [
    "SCHEMA",
    "STATE_AXES",
    "KernelState",
    "StateResolvedSignedShieldingKernelFamily",
]
