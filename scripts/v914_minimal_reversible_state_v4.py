"""v4 fail-closed surface-return physics for minimal reversible fatigue.

v3 corrected the *diagnostic* definition of cyclic reversal but intentionally
left v2 boundary-cancellation physics unchanged.  That allowed any left-boundary
mobile outflow to cancel the emission-linked blunting ledger, even when the
emitted population was not under a true reverse transport drive.

v4 makes cancellation causal and fail-closed.  Raw left-boundary outflow remains
available as a transport diagnostic, but it changes the blunting state only when
all three conditions hold:

1. the outflow belongs to the Burgers-sign population actually emitted by the
   tensile crack-tip source on that slip system;
2. the effective transport stress at the crack/free-surface boundary is reversed
   relative to that system's positive-tension resolved-stress direction; and
3. positive left-boundary outflow is present.

No cleavage, emission, mobility, storage, event-length, or stochastic law is
changed relative to v3.
"""
from __future__ import annotations

# This module is used in two legitimate import modes:
#   1. direct runner execution, where ROOT/scripts and the external v9.14 package
#      are already on sys.path; and
#   2. pytest/package import as scripts.v914_minimal_reversible_state_v4.
# Reproduce the authoritative runner precedence here so both modes resolve the
# same constitutive package rather than the local driver repo's package shadow.
import os
from pathlib import Path
import sys
from typing import Any, Mapping

_SCRIPTS = Path(__file__).resolve().parent
_DEFAULT_V914 = Path(
    os.environ.get(
        "V914_ROOT",
        "/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v9_14_cyclic_fatigue_knee_search",
    )
)
for _path in (str(_SCRIPTS), str(_DEFAULT_V914)):
    while _path in sys.path:
        sys.path.remove(_path)
# External constitutive package must precede the local repository; scripts must
# precede both for the v2/v3 top-level module names.
sys.path.insert(0, str(_DEFAULT_V914))
sys.path.insert(0, str(_SCRIPTS))

import numpy as np
from scipy.linalg import solve_banded

from v914_minimal_reversible_state_v3 import (
    MinimalReversibleEmerentGNDState as _V3State,
)
from v914_reversible_transport_utils import boundary_outflow_per_m


MODEL_ID = "v9.14_minimal_reversible_mobile_return_v4_physical_return_only"
STATE_EXTENSION_SCHEMA = "v914_minimal_reversible_state_extension_v4"


def physical_surface_return_qualifies(
    *,
    q: int,
    emitted_q: int,
    reverse_drive_at_surface: bool,
    returned_per_m: float,
) -> bool:
    """Return True only for emitted-population outflow under true reversal."""
    return bool(
        int(q) == int(emitted_q)
        and bool(reverse_drive_at_surface)
        and float(returned_per_m) > 0.0
    )


class MinimalReversibleEmergentGNDState(_V3State):
    """v3 signed transport with fail-closed physical return cancellation."""

    def integration_metadata(self) -> dict[str, object]:
        metadata = dict(super().integration_metadata())
        metadata.update(
            {
                "model_id": MODEL_ID,
                "raw_left_boundary_outflow_is_diagnostic_only": True,
                "blunting_cancellation_requires_emitted_population": True,
                "blunting_cancellation_requires_true_reverse_drive": True,
                "physical_return_surface_location": "left_mpz_boundary_x0",
                "checkpoint_promotion_from_v2_v3_allowed": False,
                "transport_physics_changed_from_v3": False,
                "cleavage_physics_changed_from_v3": False,
            }
        )
        return metadata

    def _coupled_mobile_retained(
        self,
        rates: Mapping[str, np.ndarray],
        dt: float,
    ) -> None:
        """v3 stiff transport with causal physical-return cancellation.

        Raw boundary fluxes are always recorded.  Only emitted-population
        left-boundary outflow occurring while the *surface* transport drive is
        truly reversed is allowed to cancel the source-slip/blunting ledger.
        """
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
            reverse_at_surface = bool(reverse_mask[system, 0])

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

                    # Preserve the raw boundary-fate ledger for diagnostics.
                    if returned > 0.0:
                        self.cumulative_returned_mobile_per_m[system, q] += returned
                        self.interval_returned_mobile_per_m[system, q] += returned

                        # Physical state cancellation is much stricter than raw
                        # left-boundary outflow classification.
                        if physical_surface_return_qualifies(
                            q=q,
                            emitted_q=emitted_q,
                            reverse_drive_at_surface=reverse_at_surface,
                            returned_per_m=returned,
                        ):
                            self._cancel_returned_source_slip(system, q, returned)
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
            raise RuntimeError("minimal reversible v4 operator produced nonfinite state")

    def reversibility_diagnostics(self) -> dict[str, float]:
        data = dict(super().reversibility_diagnostics())
        data.update(
            {
                "reversible_raw_return_fraction_of_emitted": data.get(
                    "reversible_return_fraction_of_emitted", 0.0
                ),
                "reversible_physical_return_fraction_of_emitted": data.get(
                    "reversible_reverse_driven_return_fraction_of_emitted", 0.0
                ),
                "reversible_physical_returned_mobile_per_m": data.get(
                    "reversible_reverse_driven_returned_mobile_per_m", 0.0
                ),
            }
        )
        return data

    def reversible_checkpoint_payload(self) -> dict[str, Any]:
        payload = dict(super().reversible_checkpoint_payload())
        payload["schema"] = STATE_EXTENSION_SCHEMA
        return payload

    def restore_reversible_checkpoint_payload(
        self,
        payload: Mapping[str, Any] | None,
    ) -> None:
        self._ensure_reversible_fields()
        if not payload:
            return
        if payload.get("schema") != STATE_EXTENSION_SCHEMA:
            raise ValueError(
                "v4 refuses v2/v3 reversible checkpoints because their "
                "returned-slip cancellation semantics were not fail-closed"
            )
        parent_payload = dict(payload)
        parent_payload["schema"] = "v914_minimal_reversible_state_extension_v3"
        super().restore_reversible_checkpoint_payload(parent_payload)


__all__ = [
    "MODEL_ID",
    "STATE_EXTENSION_SCHEMA",
    "physical_surface_return_qualifies",
    "MinimalReversibleEmergentGNDState",
]
