"""Minimal reversible extension of the v9.13/v9.14 1-D MPZ state.

Physics added in this branch is intentionally narrow:

1. The existing signed Peierls/upwind transport operator remains authoritative.
2. Mobile content that leaves the left MPZ boundary under reverse effective
   drive is counted as return to the crack/free surface and annihilated there.
3. Retained/tangled content is *not* recovered by this rule.
4. Returned mobile line content cancels the corresponding near-tip source-slip
   ledger used for permanent blunting, while the historical cumulative source
   slip ledger is retained unchanged.
5. Right-boundary mobile outflow is recorded separately as far-field escape.

No cleavage barrier, emission barrier, stochastic threshold, event-length law,
non-Schmid term, or empirical recovery fraction is introduced here.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np
from scipy.linalg import solve_banded

from arrhenius_fracture.emergent_gnd_state_v913 import (
    EmergentGNDState as _BaseEmergentGNDState,
)

from v914_reversible_transport_utils import (
    boundary_outflow_per_m,
    proportional_cancellation_density,
)


MODEL_ID = "v9.14_minimal_reversible_mobile_return_v1"
STATE_EXTENSION_SCHEMA = "v914_minimal_reversible_state_extension_v1"


class MinimalReversibleEmergentGNDState(_BaseEmergentGNDState):
    """Persistent-site MPZ state with mobile return/surface annihilation."""

    def __init__(self, candidate, physics):
        super().__init__(candidate, physics)
        self._ensure_reversible_fields()

    def _ensure_reversible_fields(self) -> None:
        if not hasattr(self, "returned_slip_m2"):
            self.returned_slip_m2 = np.zeros_like(self.accumulated_slip_m2)
        shape = (self.c.n_systems, 2)
        if not hasattr(self, "cumulative_returned_mobile_per_m"):
            self.cumulative_returned_mobile_per_m = np.zeros(shape, dtype=float)
        if not hasattr(self, "cumulative_escaped_mobile_per_m"):
            self.cumulative_escaped_mobile_per_m = np.zeros(shape, dtype=float)
        if not hasattr(self, "cumulative_cancelled_slip_line_content"):
            self.cumulative_cancelled_slip_line_content = np.zeros(shape, dtype=float)
        if not hasattr(self, "interval_returned_mobile_per_m"):
            self.interval_returned_mobile_per_m = np.zeros(shape, dtype=float)
        if not hasattr(self, "interval_escaped_mobile_per_m"):
            self.interval_escaped_mobile_per_m = np.zeros(shape, dtype=float)
        if not hasattr(self, "interval_cancelled_slip_line_content"):
            self.interval_cancelled_slip_line_content = np.zeros(shape, dtype=float)

    def begin_diagnostic_interval(self) -> None:
        super().begin_diagnostic_interval()
        shape = (self.c.n_systems, 2)
        self.interval_returned_mobile_per_m = np.zeros(shape, dtype=float)
        self.interval_escaped_mobile_per_m = np.zeros(shape, dtype=float)
        self.interval_cancelled_slip_line_content = np.zeros(shape, dtype=float)

    def integration_metadata(self) -> dict[str, object]:
        metadata = dict(super().integration_metadata())
        metadata.update(
            {
                "model_id": MODEL_ID,
                "minimal_reversible_mobile_return": True,
                "reverse_transport_law": "existing_signed_peierls_upwind_operator",
                "surface_return_boundary": "left_mpz_mobile_outflow",
                "surface_return_applies_to": "mobile_only",
                "retained_recovery_from_surface_return": False,
                "return_fraction_parameterized": False,
                "net_blunting_ledger": "accumulated_source_slip_minus_returned_slip",
                "cumulative_source_slip_ledger_preserved": True,
                "right_boundary_mobile_outflow": "far_field_escape_diagnostic",
                "non_schmid_change": False,
                "cleavage_physics_change": False,
            }
        )
        return metadata

    def net_slip_m2(self) -> np.ndarray:
        self._ensure_reversible_fields()
        return np.maximum(
            np.asarray(self.accumulated_slip_m2, dtype=float)
            - np.asarray(self.returned_slip_m2, dtype=float),
            0.0,
        )

    def cumulative_source_slip_count(self) -> float:
        weights, _ = self._tip_weights()
        line_content = (
            np.maximum(np.asarray(self.accumulated_slip_m2, dtype=float), 0.0)
            * self.cell_area_m2
        )
        return float(
            np.sum(line_content * weights[None, None, :])
            * max(float(self.c.blunting_slip_fraction), 0.0)
        )

    def returned_source_slip_count(self) -> float:
        self._ensure_reversible_fields()
        weights, _ = self._tip_weights()
        line_content = (
            np.maximum(np.asarray(self.returned_slip_m2, dtype=float), 0.0)
            * self.cell_area_m2
        )
        return float(
            np.sum(line_content * weights[None, None, :])
            * max(float(self.c.blunting_slip_fraction), 0.0)
        )

    def local_accumulated_slip_count(self) -> float:
        # During the parent constructor this override can be reached before the
        # reversible extension fields exist.  In that narrow initialization
        # window the state is exactly the baseline state.
        if not hasattr(self, "returned_slip_m2"):
            return super().local_accumulated_slip_count()
        weights, _ = self._tip_weights()
        line_content = self.net_slip_m2() * self.cell_area_m2
        return float(
            np.sum(line_content * weights[None, None, :])
            * max(float(self.c.blunting_slip_fraction), 0.0)
        )

    def _cancel_returned_source_slip(
        self,
        system: int,
        q: int,
        returned_per_m: float,
    ) -> float:
        """Cancel source-slip line content corresponding to surface return."""
        self._ensure_reversible_fields()
        returned_line_content = (
            max(float(returned_per_m), 0.0)
            * max(float(self.state_strip_width_m), 0.0)
        )
        if returned_line_content <= 0.0:
            return 0.0
        nsrc = self._source_zone_bin_count()
        net = self.net_slip_m2()[system, q, :nsrc]
        increment, cancelled = proportional_cancellation_density(
            net,
            returned_line_content,
            self.cell_area_m2,
        )
        self.returned_slip_m2[system, q, :nsrc] += increment
        self.cumulative_cancelled_slip_line_content[system, q] += cancelled
        self.interval_cancelled_slip_line_content[system, q] += cancelled
        return float(cancelled)

    def _coupled_mobile_retained(
        self,
        rates: Mapping[str, np.ndarray],
        dt: float,
    ) -> None:
        """Baseline coupled operator plus exact boundary fate bookkeeping.

        The matrix and solve order are copied from the qualified stiff operator.
        Only the post-solve boundary fluxes are classified.  Mobile material
        leaving x=0 is a returned/surface-annihilated fate; material leaving the
        far boundary is an escaped fate.  Retained content never participates
        directly in either boundary fate.
        """
        if dt <= 0.0:
            return
        self._ensure_reversible_fields()
        n_substeps = max(int(self.coupled_operator_substeps), 1)
        h = float(dt) / float(n_substeps)
        recovery = float(rates["recovery_rate_s"])
        velocity_base = np.asarray(rates["velocity_m_s"], dtype=float)
        encounter = np.asarray(rates["encounter_s"], dtype=float)
        taylor = np.asarray(rates["taylor_completion_s"], dtype=float)

        for system in range(self.c.n_systems):
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
                    returned, escaped = boundary_outflow_per_m(
                        state[0::2], velocity, h
                    )
                    if returned > 0.0:
                        self.cumulative_returned_mobile_per_m[system, q] += returned
                        self.interval_returned_mobile_per_m[system, q] += returned
                        self._cancel_returned_source_slip(system, q, returned)
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
            raise RuntimeError("minimal reversible operator produced nonfinite state")

    def translate_tip(self, da_m: float) -> None:
        """Translate every baseline field plus the returned-slip cancellation."""
        da = max(float(da_m), 0.0)
        if da <= 0.0:
            return
        self._ensure_reversible_fields()
        self.last_tip_radius_before_advance_m = self.tip_radius_m()
        self.interval_tip_radius_max_m = max(
            float(self.interval_tip_radius_max_m),
            float(self.last_tip_radius_before_advance_m),
        )
        for name in (
            "mobile_m2",
            "retained_m2",
            "accumulated_slip_m2",
            "returned_slip_m2",
        ):
            setattr(
                self,
                name,
                self._translate_density_field(
                    getattr(self, name),
                    da,
                    self.dx,
                    self.c.mpz_length_m,
                ),
            )
        self.source_available_m2[...] = self.p.rho_source0_m2
        self.source_capacity_m2[...] = self.p.rho_source0_m2
        self.extension_m += da
        self.last_tip_radius_after_advance_m = self.tip_radius_m()
        self.interval_tip_radius_end_m = self.last_tip_radius_after_advance_m
        self.interval_resharpening_m += max(
            float(self.last_tip_radius_before_advance_m)
            - float(self.last_tip_radius_after_advance_m),
            0.0,
        )
        self.interval_translation_steps += 1
        self.interval_crack_advance_m += da

    def reversibility_diagnostics(self) -> dict[str, float]:
        self._ensure_reversible_fields()
        emitted = float(np.sum(self.cumulative_line_content))
        returned_per_m = float(np.sum(self.cumulative_returned_mobile_per_m))
        escaped_per_m = float(np.sum(self.cumulative_escaped_mobile_per_m))
        returned_line = returned_per_m * max(float(self.state_strip_width_m), 0.0)
        escaped_line = escaped_per_m * max(float(self.state_strip_width_m), 0.0)
        denominator = max(emitted, 1.0e-300)
        return {
            "reversible_returned_mobile_per_m": returned_per_m,
            "reversible_escaped_mobile_per_m": escaped_per_m,
            "reversible_returned_line_content": returned_line,
            "reversible_escaped_line_content": escaped_line,
            "reversible_cancelled_slip_line_content": float(
                np.sum(self.cumulative_cancelled_slip_line_content)
            ),
            "reversible_return_fraction_of_emitted": returned_line / denominator,
            "reversible_escape_fraction_of_emitted": escaped_line / denominator,
            "reversible_cumulative_source_slip_count": self.cumulative_source_slip_count(),
            "reversible_returned_source_slip_count": self.returned_source_slip_count(),
            "reversible_net_source_slip_count": self.local_accumulated_slip_count(),
        }

    def diagnostics(self, residence_time_s: float, K_MPa_sqrt_m: float, T_K: float):
        data = dict(super().diagnostics(residence_time_s, K_MPa_sqrt_m, T_K))
        data.update(self.reversibility_diagnostics())
        data.update(
            {
                "reversible_interval_returned_mobile_per_m": float(
                    np.sum(self.interval_returned_mobile_per_m)
                ),
                "reversible_interval_escaped_mobile_per_m": float(
                    np.sum(self.interval_escaped_mobile_per_m)
                ),
                "reversible_interval_cancelled_slip_line_content": float(
                    np.sum(self.interval_cancelled_slip_line_content)
                ),
            }
        )
        return data

    def reversible_checkpoint_payload(self) -> dict[str, Any]:
        self._ensure_reversible_fields()
        return {
            "schema": STATE_EXTENSION_SCHEMA,
            "returned_slip_m2": np.asarray(self.returned_slip_m2).tolist(),
            "cumulative_returned_mobile_per_m": np.asarray(
                self.cumulative_returned_mobile_per_m
            ).tolist(),
            "cumulative_escaped_mobile_per_m": np.asarray(
                self.cumulative_escaped_mobile_per_m
            ).tolist(),
            "cumulative_cancelled_slip_line_content": np.asarray(
                self.cumulative_cancelled_slip_line_content
            ).tolist(),
            "interval_returned_mobile_per_m": np.asarray(
                self.interval_returned_mobile_per_m
            ).tolist(),
            "interval_escaped_mobile_per_m": np.asarray(
                self.interval_escaped_mobile_per_m
            ).tolist(),
            "interval_cancelled_slip_line_content": np.asarray(
                self.interval_cancelled_slip_line_content
            ).tolist(),
        }

    def restore_reversible_checkpoint_payload(self, payload: Mapping[str, Any] | None) -> None:
        self._ensure_reversible_fields()
        if not payload:
            return
        if payload.get("schema") != STATE_EXTENSION_SCHEMA:
            raise ValueError("minimal reversible checkpoint extension schema mismatch")
        arrays = (
            "returned_slip_m2",
            "cumulative_returned_mobile_per_m",
            "cumulative_escaped_mobile_per_m",
            "cumulative_cancelled_slip_line_content",
            "interval_returned_mobile_per_m",
            "interval_escaped_mobile_per_m",
            "interval_cancelled_slip_line_content",
        )
        for name in arrays:
            setattr(self, name, np.asarray(payload[name], dtype=float))
        if self.returned_slip_m2.shape != self.accumulated_slip_m2.shape:
            raise ValueError("returned-slip checkpoint shape mismatch")

    @classmethod
    def from_existing_state(
        cls,
        state: _BaseEmergentGNDState,
        payload: Mapping[str, Any] | None = None,
    ) -> "MinimalReversibleEmergentGNDState":
        """Promote an authoritative baseline checkpoint state to this subclass."""
        promoted = cls.__new__(cls)
        promoted.__dict__ = copy.deepcopy(state.__dict__)
        promoted._ensure_reversible_fields()
        promoted.restore_reversible_checkpoint_payload(payload)
        return promoted


__all__ = [
    "MODEL_ID",
    "STATE_EXTENSION_SCHEMA",
    "MinimalReversibleEmergentGNDState",
]
