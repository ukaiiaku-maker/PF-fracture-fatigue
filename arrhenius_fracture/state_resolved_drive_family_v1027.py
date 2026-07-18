"""Candidate-independent state-resolved signed emission-drive family for v10.2.7.

The reduced campaign does not invent constant Schmid factors. It interpolates the
signed resolved-shear/opening-stress ratios measured from equilibrated 2-D tensor
probes at the same mechanical state coordinates used by the v10.2.6 shielding
kernel family. Extrapolation is forbidden.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .signed_kernel_family_v1026 import STATE_AXES

SCHEMA = "v10.2.7_state_resolved_signed_drive_family"


@dataclass(frozen=True)
class DriveState:
    state_id: str
    coordinates: np.ndarray
    signed_factors: np.ndarray
    metadata: dict[str, Any]


class StateResolvedSignedDriveFamily:
    """Fail-closed interpolation of signed tau/sigma opening factors."""

    def __init__(
        self,
        *,
        states: list[DriveState],
        interpolation: dict[str, Any],
        metadata: dict[str, Any],
        source_path: str,
    ) -> None:
        if not states:
            raise ValueError("signed drive family requires at least one state")
        self.states = tuple(states)
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
        self._last_signed_factors = self.states[0].signed_factors.copy()
        self._validate_internal()

    @classmethod
    def from_json(cls, path: str | Path) -> "StateResolvedSignedDriveFamily":
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text())
        if payload.get("schema") != SCHEMA:
            raise ValueError(
                f"signed drive-family schema must be {SCHEMA!r}; "
                f"got {payload.get('schema')!r}"
            )
        required_truth = {
            "candidate_independent": True,
            "derived_from_2d_tensor_probe": True,
            "signed_resolved_shear": True,
            "normalized_by_local_opening_stress": True,
            "fitted_to_toughness_or_fatigue": False,
        }
        for key, expected in required_truth.items():
            if payload.get(key) is not expected:
                raise ValueError(f"drive-family metadata requires {key}={expected}")
        axes = tuple(payload.get("state_axes", ()))
        if axes != STATE_AXES:
            raise ValueError(f"drive-family axes must be {STATE_AXES}; got {axes}")
        states: list[DriveState] = []
        for index, row in enumerate(payload.get("states", [])):
            coordinates = np.asarray([float(row[name]) for name in STATE_AXES])
            factors = np.asarray(row["signed_tau_over_sigma_by_system"], dtype=float)
            states.append(
                DriveState(
                    state_id=str(row.get("state_id", f"state_{index:04d}")),
                    coordinates=coordinates,
                    signed_factors=factors,
                    metadata={
                        key: value
                        for key, value in row.items()
                        if key not in {*STATE_AXES, "signed_tau_over_sigma_by_system"}
                    },
                )
            )
        return cls(
            states=states,
            interpolation=dict(payload.get("interpolation", {})),
            metadata={key: value for key, value in payload.items() if key != "states"},
            source_path=str(source),
        )

    @property
    def n_systems(self) -> int:
        return int(self.states[0].signed_factors.size)

    def _validate_internal(self) -> None:
        if tuple(self.metadata.get("state_axes", ())) != STATE_AXES:
            raise ValueError("drive-family metadata state axes are inconsistent")
        method = str(self.interpolation.get("method", "inverse_distance"))
        if method != "inverse_distance":
            raise ValueError("signed drive family requires inverse_distance interpolation")
        reference_shape = self.states[0].signed_factors.shape
        if len(reference_shape) != 1 or reference_shape[0] < 1:
            raise ValueError("signed drive factors must be one-dimensional")
        for state in self.states:
            if state.coordinates.shape != (len(STATE_AXES),):
                raise ValueError("drive state has the wrong coordinate dimension")
            if state.signed_factors.shape != reference_shape:
                raise ValueError("drive-factor shapes differ across states")
            if np.any(~np.isfinite(state.coordinates)) or np.any(
                ~np.isfinite(state.signed_factors)
            ):
                raise ValueError(f"non-finite data in drive state {state.state_id}")
            if np.any(np.abs(state.signed_factors) > 1.0 + 1.0e-9):
                raise ValueError("normalized signed shear factors must satisfy |tau/sigma| <= 1")

    def validate_against_kernel_family(self, kernel_family) -> None:
        kernel_by_id = {state.state_id: state.coordinates for state in kernel_family.states}
        drive_by_id = {state.state_id: state.coordinates for state in self.states}
        if set(kernel_by_id) != set(drive_by_id):
            raise ValueError("drive and shielding families must contain identical state IDs")
        for state_id in sorted(kernel_by_id):
            if not np.allclose(
                kernel_by_id[state_id],
                drive_by_id[state_id],
                rtol=1.0e-12,
                atol=1.0e-15,
            ):
                raise ValueError(
                    f"drive and shielding coordinates differ for state {state_id}"
                )
        if self.n_systems != kernel_family.n_systems:
            raise ValueError("drive and shielding families use different system counts")

    def resolve(
        self,
        *,
        r_eff_over_r0: float,
        opening_strength_fraction: float,
        crack_extension_m: float,
    ) -> np.ndarray:
        query = np.asarray(
            [r_eff_over_r0, opening_strength_fraction, crack_extension_m],
            dtype=float,
        )
        if np.any(~np.isfinite(query)):
            raise ValueError("drive-family query must be finite")
        tolerance = float(self.interpolation.get("envelope_relative_tolerance", 1.0e-10))
        allowance = tolerance * np.maximum(np.abs(self._scale), 1.0)
        if np.any(query < self._minimum - allowance) or np.any(
            query > self._maximum + allowance
        ):
            raise RuntimeError(
                "requested state lies outside the validated signed-drive envelope: "
                f"query={query.tolist()} min={self._minimum.tolist()} "
                f"max={self._maximum.tolist()}"
            )
        query = np.minimum(np.maximum(query, self._minimum), self._maximum)
        normalized = (self._coordinates - query[None, :]) / self._scale[None, :]
        distance = np.linalg.norm(normalized, axis=1)
        exact = np.flatnonzero(distance <= 1.0e-12)
        weights = np.zeros(len(self.states), dtype=float)
        if exact.size:
            weights[int(exact[0])] = 1.0
        else:
            neighbors = max(
                2,
                min(int(self.interpolation.get("neighbors", 8)), len(self.states)),
            )
            indices = np.argsort(distance)[:neighbors]
            power = max(float(self.interpolation.get("power", 2.0)), 0.1)
            local = 1.0 / np.maximum(distance[indices], 1.0e-15) ** power
            local /= np.sum(local)
            weights[indices] = local
        factors = sum(
            weight * state.signed_factors
            for weight, state in zip(weights, self.states)
        )
        self._last_query = query.copy()
        self._last_weights = weights.copy()
        self._last_state_ids = [
            state.state_id
            for weight, state in zip(weights, self.states)
            if weight > 0.0
        ]
        self._last_signed_factors = np.asarray(factors, dtype=float)
        return self._last_signed_factors.copy()

    def audit_payload(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "source_path": self.source_path,
            "n_states": len(self.states),
            "n_systems": self.n_systems,
            "state_axes": list(STATE_AXES),
            "state_envelope_min": self._minimum.tolist(),
            "state_envelope_max": self._maximum.tolist(),
            "interpolation": copy.deepcopy(self.interpolation),
            "last_query": self._last_query.tolist(),
            "last_state_ids": list(self._last_state_ids),
            "last_weights": self._last_weights.tolist(),
            "last_signed_tau_over_sigma": self._last_signed_factors.tolist(),
            "production_parameterization_allowed": bool(
                self.metadata.get("production_parameterization_allowed", False)
            ),
        }


__all__ = ["SCHEMA", "DriveState", "StateResolvedSignedDriveFamily"]
