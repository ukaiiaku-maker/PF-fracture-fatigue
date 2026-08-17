"""Minimal reversible extension of the v9.13/v9.14 1-D MPZ state.

Physics added in this branch is intentionally narrow:

1. Cleavage and new emission retain the qualified opening-only v9.13/v9.14 law.
2. Already-mobile dislocations use the signed cyclic transport intensity
   ``K_transport = K_applied - K_shield`` plus the existing signed GND stress.
3. Mobile content that leaves the left MPZ boundary is counted as return to the
   crack/free surface and annihilated there.
4. Retained/tangled content is *not* recovered by this rule.
5. Returned mobile line content cancels the corresponding near-tip source-slip
   ledger used for permanent blunting, while cumulative activity is preserved.
6. Right-boundary mobile outflow is recorded separately as far-field escape.

No cleavage-barrier change, emission-barrier change, stochastic-law change,
non-Schmid term, or empirical recovery fraction is introduced here.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Mapping

import numpy as np
from scipy.linalg import solve_banded

from arrhenius_fracture.emergent_gnd_state_v913 import (
    EmergentGNDState as _BaseEmergentGNDState,
)

from v914_reversible_transport_utils import (
    boundary_outflow_per_m,
    proportional_cancellation_density,
    signed_transport_stress_fields,
)


MODEL_ID = "v9.14_minimal_reversible_mobile_return_v2_signed_transport"
STATE_EXTENSION_SCHEMA = "v914_minimal_reversible_state_extension_v2"


class MinimalReversibleEmergentGNDState(_BaseEmergentGNDState):
    """Persistent-site MPZ state with signed mobile return/surface annihilation."""

    def __init__(self, candidate, physics):
        super().__init__(candidate, physics)
        self._ensure_reversible_fields()

    def _ensure_reversible_fields(self) -> None:
        if not hasattr(self, "returned_slip_m2"):
            self.returned_slip_m2 = np.zeros_like(self.accumulated_slip_m2)
        shape = (self.c.n_systems, 2)
        for name in (
            "cumulative_returned_mobile_per_m",
            "cumulative_escaped_mobile_per_m",
            "cumulative_cancelled_slip_line_content",
            "interval_returned_mobile_per_m",
            "interval_escaped_mobile_per_m",
            "interval_cancelled_slip_line_content",
        ):
            if not hasattr(self, name):
                setattr(self, name, np.zeros(shape, dtype=float))

        scalar_defaults = {
            "cumulative_transport_channel_time_s": 0.0,
            "cumulative_reverse_channel_time_s": 0.0,
            "cumulative_mobile_exposure_m2_s": 0.0,
            "cumulative_reverse_mobile_exposure_m2_s": 0.0,
            "interval_transport_channel_time_s": 0.0,
            "interval_reverse_channel_time_s": 0.0,
            "interval_mobile_exposure_m2_s": 0.0,
            "interval_reverse_mobile_exposure_m2_s": 0.0,
            "cumulative_transport_K_min_MPa_sqrt_m": math.inf,
            "cumulative_transport_K_max_MPa_sqrt_m": -math.inf,
            "cumulative_transport_tau_min_Pa": math.inf,
            "cumulative_transport_tau_max_Pa": -math.inf,
            "interval_transport_K_min_MPa_sqrt_m": math.inf,
            "interval_transport_K_max_MPa_sqrt_m": -math.inf,
            "interval_transport_tau_min_Pa": math.inf,
            "interval_transport_tau_max_Pa": -math.inf,
        }
        for name, value in scalar_defaults.items():
            if not hasattr(self, name):
                setattr(self, name, float(value))

    def begin_diagnostic_interval(self) -> None:
        super().begin_diagnostic_interval()
        shape = (self.c.n_systems, 2)
        self.interval_returned_mobile_per_m = np.zeros(shape, dtype=float)
        self.interval_escaped_mobile_per_m = np.zeros(shape, dtype=float)
        self.interval_cancelled_slip_line_content = np.zeros(shape, dtype=float)
        self.interval_transport_channel_time_s = 0.0
        self.interval_reverse_channel_time_s = 0.0
        self.interval_mobile_exposure_m2_s = 0.0
        self.interval_reverse_mobile_exposure_m2_s = 0.0
        self.interval_transport_K_min_MPa_sqrt_m = math.inf
        self.interval_transport_K_max_MPa_sqrt_m = -math.inf
        self.interval_transport_tau_min_Pa = math.inf
        self.interval_transport_tau_max_Pa = -math.inf

    def integration_metadata(self) -> dict[str, object]:
        metadata = dict(super().integration_metadata())
        metadata.update(
            {
                "model_id": MODEL_ID,
                "minimal_reversible_mobile_return": True,
                "mobile_transport_driving_K": "signed_K_applied_minus_K_shield",
                "mobile_transport_external_K_clipping": False,
                "mobile_transport_internal_stress": "existing_signed_tau_gnd",
                "cleavage_opening_only_unchanged": True,
                "emission_opening_only_unchanged": True,
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
        if not hasattr(self, "returned_slip_m2"):
            return super().local_accumulated_slip_count()
        weights, _ = self._tip_weights()
        line_content = self.net_slip_m2() * self.cell_area_m2
        return float(
            np.sum(line_content * weights[None, None, :])
            * max(float(self.c.blunting_slip_fraction), 0.0)
        )

    def local_rates(self, K_MPa_sqrt_m: float, T_K: float) -> dict[str, np.ndarray]:
        """Use signed cyclic stress only for transport of existing mobile line.

        ``super().local_rates`` remains authoritative for cleavage, new emission,
        Taylor completion and every barrier parameter.  We replace only the
        Peierls mobile-transport velocity and the encounter rate tied to its
        travelled distance.  This lets unloading/compression reverse a mobile
        dislocation without creating reverse nucleation or opening cleavage.
        """
        rates = dict(super().local_rates(K_MPa_sqrt_m, T_K))
        radius = float(self.tip_radius_m())
        drive_factors = np.asarray(self.emission_drive_factors(), dtype=float)
        tau_gnd = np.asarray(rates["tau_gnd_Pa"], dtype=float)
        fields = signed_transport_stress_fields(
            K_MPa_sqrt_m,
            self.K_shield_MPa_sqrt_m(),
            radius,
            drive_factors,
            tau_gnd,
        )
        tau_transport_eff = np.asarray(fields["tau_effective_Pa"], dtype=float)

        forest = self.forest_density_m2()
        spacing = 1.0 / (2.0 * np.sqrt(forest))
        jump = self.c.jump_fraction_of_forest_spacing * spacing
        p_surface = self.p.peierls.surface(self.p.emission)
        peierls = self._signed_rate(
            p_surface,
            self.p.peierls.stress_fraction * tau_transport_eff,
            T_K,
            self.p.peierls.nu0_s,
        )
        peierls_velocity = jump * peierls
        velocity = (
            max(float(self.c.mobile_transport_velocity_scale), 0.0)
            * peierls_velocity
        )
        mfp = self.c.mean_free_path_coefficient / np.sqrt(forest)
        encounter = (
            max(float(self.c.encounter_efficiency), 0.0)
            * np.abs(peierls_velocity)
            / np.maximum(mfp, 1.0e-30)
        )

        # The power bookkeeping should see the same signed transport stress as
        # the mobile constitutive channel.  Emission fields from the parent are
        # left untouched and therefore remain opening-only.
        rates["velocity_m_s"] = velocity
        rates["peierls_velocity_m_s"] = peierls_velocity
        rates["encounter_s"] = encounter
        rates["tau_external_Pa"] = np.asarray(fields["tau_applied_Pa"], dtype=float)
        rates["tau_nonlocal_shielding_Pa"] = (
            np.asarray(fields["tau_transport_Pa"], dtype=float)
            - np.asarray(fields["tau_applied_Pa"], dtype=float)
        )
        rates["tau_eff_Pa"] = tau_transport_eff
        rates["reversible_transport_K_signed_MPa_sqrt_m"] = np.asarray(
            float(fields["K_transport_MPa_sqrt_m"])
        )
        rates["reversible_tau_transport_external_Pa"] = np.asarray(
            fields["tau_transport_Pa"], dtype=float
        )
        rates["reversible_tau_transport_eff_Pa"] = tau_transport_eff.copy()
        emission_sign = np.asarray(self.c.emission_signs, dtype=float)[:, None]
        rates["reversible_emitted_sign_spatial_velocity_m_s"] = (
            emission_sign * velocity
        )
        return rates

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

    @staticmethod
    def _finite_or_nan(value: float) -> float:
        return float(value) if math.isfinite(float(value)) else math.nan

    def _update_transport_diagnostics(
        self,
        rates: Mapping[str, np.ndarray],
        dt: float,
    ) -> None:
        self._ensure_reversible_fields()
        duration = max(float(dt), 0.0)
        if duration <= 0.0:
            return
        K_transport = float(rates["reversible_transport_K_signed_MPa_sqrt_m"])
        tau = np.asarray(rates["reversible_tau_transport_eff_Pa"], dtype=float)
        emitted_velocity = np.asarray(
            rates["reversible_emitted_sign_spatial_velocity_m_s"], dtype=float
        )
        self.cumulative_transport_K_min_MPa_sqrt_m = min(
            self.cumulative_transport_K_min_MPa_sqrt_m, K_transport
        )
        self.cumulative_transport_K_max_MPa_sqrt_m = max(
            self.cumulative_transport_K_max_MPa_sqrt_m, K_transport
        )
        self.interval_transport_K_min_MPa_sqrt_m = min(
            self.interval_transport_K_min_MPa_sqrt_m, K_transport
        )
        self.interval_transport_K_max_MPa_sqrt_m = max(
            self.interval_transport_K_max_MPa_sqrt_m, K_transport
        )
        if tau.size:
            tau_min = float(np.min(tau))
            tau_max = float(np.max(tau))
            self.cumulative_transport_tau_min_Pa = min(
                self.cumulative_transport_tau_min_Pa, tau_min
            )
            self.cumulative_transport_tau_max_Pa = max(
                self.cumulative_transport_tau_max_Pa, tau_max
            )
            self.interval_transport_tau_min_Pa = min(
                self.interval_transport_tau_min_Pa, tau_min
            )
            self.interval_transport_tau_max_Pa = max(
                self.interval_transport_tau_max_Pa, tau_max
            )
        channel_time = duration * float(emitted_velocity.size)
        reverse_time = duration * float(np.count_nonzero(emitted_velocity < 0.0))
        self.cumulative_transport_channel_time_s += channel_time
        self.cumulative_reverse_channel_time_s += reverse_time
        self.interval_transport_channel_time_s += channel_time
        self.interval_reverse_channel_time_s += reverse_time

    def _coupled_mobile_retained(
        self,
        rates: Mapping[str, np.ndarray],
        dt: float,
    ) -> None:
        """Baseline stiff operator plus signed mobile boundary fate bookkeeping."""
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
                            np.sum(mobile_now[velocity < 0.0])
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
        channel_fraction = (
            self.cumulative_reverse_channel_time_s
            / max(self.cumulative_transport_channel_time_s, 1.0e-300)
        )
        exposure_fraction = (
            self.cumulative_reverse_mobile_exposure_m2_s
            / max(self.cumulative_mobile_exposure_m2_s, 1.0e-300)
        )
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
            "reversible_reverse_transport_channel_fraction": channel_fraction,
            "reversible_mobile_exposure_m2_s": self.cumulative_mobile_exposure_m2_s,
            "reversible_reverse_mobile_exposure_m2_s": self.cumulative_reverse_mobile_exposure_m2_s,
            "reversible_reverse_mobile_exposure_fraction": exposure_fraction,
            "reversible_transport_K_min_MPa_sqrt_m": self._finite_or_nan(
                self.cumulative_transport_K_min_MPa_sqrt_m
            ),
            "reversible_transport_K_max_MPa_sqrt_m": self._finite_or_nan(
                self.cumulative_transport_K_max_MPa_sqrt_m
            ),
            "reversible_transport_tau_min_Pa": self._finite_or_nan(
                self.cumulative_transport_tau_min_Pa
            ),
            "reversible_transport_tau_max_Pa": self._finite_or_nan(
                self.cumulative_transport_tau_max_Pa
            ),
        }

    def diagnostics(self, residence_time_s: float, K_MPa_sqrt_m: float, T_K: float):
        data = dict(super().diagnostics(residence_time_s, K_MPa_sqrt_m, T_K))
        rates = self.local_rates(K_MPa_sqrt_m, T_K)
        emitted_velocity = np.asarray(
            rates["reversible_emitted_sign_spatial_velocity_m_s"], dtype=float
        )
        tau_transport = np.asarray(rates["reversible_tau_transport_eff_Pa"], dtype=float)
        reverse_mobile = 0.0
        for system, sign in enumerate(self.c.emission_signs):
            q = 1 if sign > 0 else 0
            mask = emitted_velocity[system] < 0.0
            reverse_mobile += float(np.sum(self.mobile_m2[system, q, mask]))
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
                "reversible_current_transport_K_signed_MPa_sqrt_m": float(
                    rates["reversible_transport_K_signed_MPa_sqrt_m"]
                ),
                "reversible_current_transport_tau_min_Pa": float(np.min(tau_transport)),
                "reversible_current_transport_tau_max_Pa": float(np.max(tau_transport)),
                "reversible_current_emitted_sign_velocity_min_m_s": float(
                    np.min(emitted_velocity)
                ),
                "reversible_current_emitted_sign_velocity_max_m_s": float(
                    np.max(emitted_velocity)
                ),
                "reversible_current_reverse_spatial_fraction": float(
                    np.mean(emitted_velocity < 0.0)
                ),
                "reversible_current_reverse_mobile_m2": reverse_mobile,
            }
        )
        return data

    def reversible_checkpoint_payload(self) -> dict[str, Any]:
        self._ensure_reversible_fields()
        payload = {
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
        for name in (
            "cumulative_transport_channel_time_s",
            "cumulative_reverse_channel_time_s",
            "cumulative_mobile_exposure_m2_s",
            "cumulative_reverse_mobile_exposure_m2_s",
            "interval_transport_channel_time_s",
            "interval_reverse_channel_time_s",
            "interval_mobile_exposure_m2_s",
            "interval_reverse_mobile_exposure_m2_s",
            "cumulative_transport_K_min_MPa_sqrt_m",
            "cumulative_transport_K_max_MPa_sqrt_m",
            "cumulative_transport_tau_min_Pa",
            "cumulative_transport_tau_max_Pa",
            "interval_transport_K_min_MPa_sqrt_m",
            "interval_transport_K_max_MPa_sqrt_m",
            "interval_transport_tau_min_Pa",
            "interval_transport_tau_max_Pa",
        ):
            payload[name] = float(getattr(self, name))
        return payload

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
        for name in (
            "cumulative_transport_channel_time_s",
            "cumulative_reverse_channel_time_s",
            "cumulative_mobile_exposure_m2_s",
            "cumulative_reverse_mobile_exposure_m2_s",
            "interval_transport_channel_time_s",
            "interval_reverse_channel_time_s",
            "interval_mobile_exposure_m2_s",
            "interval_reverse_mobile_exposure_m2_s",
            "cumulative_transport_K_min_MPa_sqrt_m",
            "cumulative_transport_K_max_MPa_sqrt_m",
            "cumulative_transport_tau_min_Pa",
            "cumulative_transport_tau_max_Pa",
            "interval_transport_K_min_MPa_sqrt_m",
            "interval_transport_K_max_MPa_sqrt_m",
            "interval_transport_tau_min_Pa",
            "interval_transport_tau_max_Pa",
        ):
            setattr(self, name, float(payload[name]))
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
