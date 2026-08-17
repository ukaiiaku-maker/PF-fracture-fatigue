"""v3 diagnostic refinement for the minimal reversible v9.14 state.

The v2 state already contains the intended signed mobile-transport physics.
This subclass changes only the meaning and bookkeeping of *reverse* diagnostics:
reversal is defined relative to the tensile reference stress direction of each
slip system, not by the raw sign of a Burgers-population spatial velocity.

Boundary annihilation/cancellation physics is unchanged from v2.  An additional
ledger records the subset of left-boundary return that occurs while the emitted
population is under a true reverse transport drive.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from scipy.linalg import solve_banded

from v914_minimal_reversible_state import (
    MinimalReversibleEmergentGNDState as _V2State,
)
from v914_reversible_transport_utils import boundary_outflow_per_m
from v914_reverse_drive_utils import (
    forward_projected_transport_stress,
    reverse_drive_mask,
)


MODEL_ID = "v9.14_minimal_reversible_mobile_return_v3_reverse_drive_diagnostics"
STATE_EXTENSION_SCHEMA = "v914_minimal_reversible_state_extension_v3"


class MinimalReversibleEmergentGNDState(_V2State):
    """v2 physics with tensile-reference cyclic-reversal diagnostics."""

    def _ensure_reversible_fields(self) -> None:
        super()._ensure_reversible_fields()
        shape = (self.c.n_systems, 2)
        if not hasattr(self, "cumulative_reverse_driven_returned_mobile_per_m"):
            self.cumulative_reverse_driven_returned_mobile_per_m = np.zeros(
                shape, dtype=float
            )
        if not hasattr(self, "interval_reverse_driven_returned_mobile_per_m"):
            self.interval_reverse_driven_returned_mobile_per_m = np.zeros(
                shape, dtype=float
            )

    def begin_diagnostic_interval(self) -> None:
        super().begin_diagnostic_interval()
        self.interval_reverse_driven_returned_mobile_per_m = np.zeros(
            (self.c.n_systems, 2), dtype=float
        )

    def integration_metadata(self) -> dict[str, object]:
        metadata = dict(super().integration_metadata())
        metadata.update(
            {
                "model_id": MODEL_ID,
                "reverse_diagnostic_reference": (
                    "effective_transport_stress_projected_onto_positive_tensile_"
                    "resolved_stress_direction_per_system"
                ),
                "negative_burgers_spatial_velocity_alone_is_reverse": False,
                "reverse_driven_surface_return_ledger": True,
                "transport_physics_changed_from_v2": False,
            }
        )
        return metadata

    def local_rates(self, K_MPa_sqrt_m: float, T_K: float) -> dict[str, np.ndarray]:
        rates = dict(super().local_rates(K_MPa_sqrt_m, T_K))
        tau = np.asarray(rates["reversible_tau_transport_eff_Pa"], dtype=float)
        drive_factors = np.asarray(self.emission_drive_factors(), dtype=float)
        projected = forward_projected_transport_stress(tau, drive_factors)
        rates["reversible_forward_projected_tau_Pa"] = projected
        rates["reversible_true_reverse_drive_mask"] = projected < 0.0
        return rates

    def _update_transport_diagnostics(
        self,
        rates: Mapping[str, np.ndarray],
        dt: float,
    ) -> None:
        """Retain v2 extrema/time bookkeeping but correct the reversal count."""
        self._ensure_reversible_fields()
        duration = max(float(dt), 0.0)
        if duration <= 0.0:
            return
        before_cumulative = float(self.cumulative_reverse_channel_time_s)
        before_interval = float(self.interval_reverse_channel_time_s)
        super()._update_transport_diagnostics(rates, duration)

        mask = np.asarray(rates["reversible_true_reverse_drive_mask"], dtype=bool)
        reverse_time = duration * float(np.count_nonzero(mask))
        self.cumulative_reverse_channel_time_s = before_cumulative + reverse_time
        self.interval_reverse_channel_time_s = before_interval + reverse_time

    def _coupled_mobile_retained(
        self,
        rates: Mapping[str, np.ndarray],
        dt: float,
    ) -> None:
        """v2 stiff operator with corrected reverse-exposure diagnostics."""
        if dt <= 0.0:
            return
        self._ensure_reversible_fields()
        self._update_transport_diagnostics(rates, dt)
        n_substeps = max(int(self.coupled_operator_substeps), 1)
        h = float(dt) / float(n_substeps)
        recovery = float(rates["recovery_rate_s"])
        velocity_base = np.asarray(rates["velocity_m_s"], dtype=float)
        encounter = np.asarray(rates["encounter_s"], dtype=float)
        taylor = np.asarray(rates["taylor_completion_s"], dtype=float)
        reverse_mask = np.asarray(
            rates["reversible_true_reverse_drive_mask"], dtype=bool
        )

        for system in range(self.c.n_systems):
            emitted_q = 1 if self.c.emission_signs[system] > 0 else 0
            for q in range(2):
                burgers_sign = -1.0 if q == 0 else 1.0
                velocity = burgers_sign * velocity_base[system]
                banded = self._coupled_banded_matrix(
                    velocity,
                    encounter[system],
                    taylor[system],
                    recovery,
                    self.dx,
                    h,
                )
                state = np.empty(2 * self.c.n_bins, dtype=float)
                state[0::2] = np.maximum(self.mobile_m2[system, q], 0.0)
                state[1::2] = np.maximum(self.retained_m2[system, q], 0.0)

                for _ in range(n_substeps):
                    state = solve_banded(
                        (2, 2),
                        banded,
                        state,
                        overwrite_ab=False,
                        overwrite_b=False,
                        check_finite=False,
                    )
                    state = np.maximum(state, 0.0)
                    mobile_now = state[0::2]

                    if q == emitted_q:
                        total_exposure = float(np.sum(mobile_now)) * h
                        reverse_exposure = float(
                            np.sum(mobile_now[reverse_mask[system]])
                        ) * h
                        self.cumulative_mobile_exposure_m2_s += total_exposure
                        self.cumulative_reverse_mobile_exposure_m2_s += reverse_exposure
                        self.interval_mobile_exposure_m2_s += total_exposure
                        self.interval_reverse_mobile_exposure_m2_s += reverse_exposure

                    returned, escaped = boundary_outflow_per_m(
                        mobile_now, velocity, h
                    )
                    if returned > 0.0:
                        self.cumulative_returned_mobile_per_m[system, q] += returned
                        self.interval_returned_mobile_per_m[system, q] += returned
                        self._cancel_returned_source_slip(system, q, returned)
                        if q == emitted_q and bool(reverse_mask[system, 0]):
                            self.cumulative_reverse_driven_returned_mobile_per_m[
                                system, q
                            ] += returned
                            self.interval_reverse_driven_returned_mobile_per_m[
                                system, q
                            ] += returned
                    if escaped > 0.0:
                        self.cumulative_escaped_mobile_per_m[system, q] += escaped
                        self.interval_escaped_mobile_per_m[system, q] += escaped

                self.mobile_m2[system, q] = state[0::2]
                self.retained_m2[system, q] = state[1::2]

        if not (
            np.all(np.isfinite(self.mobile_m2))
            and np.all(np.isfinite(self.retained_m2))
            and np.all(np.isfinite(self.returned_slip_m2))
        ):
            raise RuntimeError("minimal reversible v3 operator produced nonfinite state")

    def reversibility_diagnostics(self) -> dict[str, float]:
        data = dict(super().reversibility_diagnostics())
        reverse_returned_per_m = float(
            np.sum(self.cumulative_reverse_driven_returned_mobile_per_m)
        )
        reverse_returned_line = (
            reverse_returned_per_m * max(float(self.state_strip_width_m), 0.0)
        )
        emitted = max(float(np.sum(self.cumulative_line_content)), 1.0e-300)
        data.update(
            {
                "reversible_reverse_driven_returned_mobile_per_m": (
                    reverse_returned_per_m
                ),
                "reversible_reverse_driven_returned_line_content": (
                    reverse_returned_line
                ),
                "reversible_reverse_driven_return_fraction_of_emitted": (
                    reverse_returned_line / emitted
                ),
                "reversible_interval_reverse_driven_returned_mobile_per_m": float(
                    np.sum(self.interval_reverse_driven_returned_mobile_per_m)
                ),
            }
        )
        return data

    def reversible_checkpoint_payload(self) -> dict[str, Any]:
        payload = dict(super().reversible_checkpoint_payload())
        payload["schema"] = STATE_EXTENSION_SCHEMA
        payload["cumulative_reverse_driven_returned_mobile_per_m"] = np.asarray(
            self.cumulative_reverse_driven_returned_mobile_per_m
        ).tolist()
        payload["interval_reverse_driven_returned_mobile_per_m"] = np.asarray(
            self.interval_reverse_driven_returned_mobile_per_m
        ).tolist()
        return payload

    def restore_reversible_checkpoint_payload(
        self, payload: Mapping[str, Any] | None
    ) -> None:
        self._ensure_reversible_fields()
        if not payload:
            return
        if payload.get("schema") == STATE_EXTENSION_SCHEMA:
            parent_payload = dict(payload)
            parent_payload["schema"] = "v914_minimal_reversible_state_extension_v2"
            super().restore_reversible_checkpoint_payload(parent_payload)
            self.cumulative_reverse_driven_returned_mobile_per_m = np.asarray(
                payload["cumulative_reverse_driven_returned_mobile_per_m"],
                dtype=float,
            )
            self.interval_reverse_driven_returned_mobile_per_m = np.asarray(
                payload["interval_reverse_driven_returned_mobile_per_m"],
                dtype=float,
            )
            return
        # Permit promotion of an older v2 checkpoint for diagnostic continuity.
        super().restore_reversible_checkpoint_payload(payload)


__all__ = [
    "MODEL_ID",
    "STATE_EXTENSION_SCHEMA",
    "MinimalReversibleEmergentGNDState",
]
