"""Observed single-front v10.2.30 monotonic/cyclic engine.

The outer FEM remains authoritative for the physical opening field and positive
signed directional J.  The existing cubic angular factor acts in two consistent
places:

* cleavage hazard: ``sigma_haz = sigma_physical/sqrt(gamma_rel)``;
* event dissipation: ``Gamma_haz = gamma_rel*m*DeltaG*/b**2``.

Emission, transport, backstress, shielding, and blunting continue to use the
physical opening stress.  Fixed-DeltaK fatigue supplies its physical event K to
the gate while the observer supplies the nonzero FEM probe K and J.
"""
from __future__ import annotations

import math
from typing import Any

from . import stochastic_avalanche_tip as _avalanche
from .hazard_energy_gate_v10230 import (
    HazardEnergyGatedPersistentSiteCyclicTipEngine,
    probe_to_event_energy_scale,
)
from .hazard_energy_observer_v10230 import current_observation

MODEL_ID = "v10.2.30_observed_hazard_energy_gated_persistent_site_engine"


class ObservedHazardEnergyGatedPersistentSiteCyclicTipEngine(
    HazardEnergyGatedPersistentSiteCyclicTipEngine
):
    """Production engine bound to the latest single-front directional-J probe."""

    observed_hazard_energy_gate_v10230 = True

    @classmethod
    def audit_payload(cls) -> dict[str, Any]:
        payload = super().audit_payload()
        payload["observed_hazard_energy_gate_v10230"] = {
            "model_id": MODEL_ID,
            "single_front_only": True,
            "physical_opening_for_plasticity": True,
            "hazard_opening_scaling": "sigma_physical/sqrt(gamma_rel)",
            "dissipation_scaling": "gamma_rel*m*DeltaG_cleave_eff/b^2",
            "fixed_DeltaK_probe_scaling": "J_probe*(K_event/K_probe)^2",
            "absolute_athermal_Gc": False,
            "adaptive_prediction_and_commit_use_same_gate": True,
            "event_work_accumulated_differentially": True,
            "geometry_veto_policy": "fail_closed_no_partial_state_rollback",
        }
        return payload

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hazard_energy_last_sigma_physical_Pa = 0.0
        self.hazard_energy_last_sigma_scaled_Pa = 0.0
        self.hazard_energy_last_K_event_Pa_sqrt_m = 0.0
        self.hazard_energy_last_K_probe_Pa_sqrt_m = 0.0
        self.hazard_energy_available_event_accum_J_per_m = 0.0
        self.hazard_energy_dissipated_event_accum_J_per_m = 0.0

    def clone_split(self, daughter_fraction=0.5):
        child = super().clone_split(daughter_fraction)
        child.hazard_energy_last_sigma_physical_Pa = float(
            self.hazard_energy_last_sigma_physical_Pa
        )
        child.hazard_energy_last_sigma_scaled_Pa = float(
            self.hazard_energy_last_sigma_scaled_Pa
        )
        child.hazard_energy_last_K_event_Pa_sqrt_m = float(
            self.hazard_energy_last_K_event_Pa_sqrt_m
        )
        child.hazard_energy_last_K_probe_Pa_sqrt_m = float(
            self.hazard_energy_last_K_probe_Pa_sqrt_m
        )
        child.hazard_energy_available_event_accum_J_per_m = float(
            self.hazard_energy_available_event_accum_J_per_m
        )
        child.hazard_energy_dissipated_event_accum_J_per_m = float(
            self.hazard_energy_dissipated_event_accum_J_per_m
        )
        return child

    def _refresh_observed_context(
        self,
        K_event_Pa_sqrt_m: float,
        loading_mode: str,
    ) -> None:
        observation = current_observation()
        event = max(float(K_event_Pa_sqrt_m), 0.0)
        # Validate the fixed-DeltaK conversion at context-install time.  This fails
        # closed before any hazard or process-zone state is advanced.
        probe_to_event_energy_scale(event, observation.K_probe_Pa_sqrt_m)
        self.set_hazard_energy_gate_context(
            J_probe_J_per_m2=observation.J_probe_J_per_m2,
            K_probe_Pa_sqrt_m=observation.K_probe_Pa_sqrt_m,
            K_event_Pa_sqrt_m=event,
            gamma_rel=observation.gamma_rel,
            loading_mode=loading_mode,
            probe_source=(
                f"{observation.source}:root_signed_directional_J"
            ),
        )
        self.hazard_energy_last_K_event_Pa_sqrt_m = event
        self.hazard_energy_last_K_probe_Pa_sqrt_m = float(
            observation.K_probe_Pa_sqrt_m
        )

    def lambda_cleave(self, sig_tip, T):
        """Evaluate the active Arrhenius barrier on the existing angular overdrive."""
        sigma_physical = max(float(sig_tip), 0.0)
        context = self.hazard_energy_gate_context
        gamma = 1.0 if context is None else max(float(context.gamma_rel), 1.0e-300)
        sigma_scaled = sigma_physical / math.sqrt(gamma)
        self.hazard_energy_last_sigma_physical_Pa = sigma_physical
        self.hazard_energy_last_sigma_scaled_Pa = sigma_scaled
        effective, raw, barrier = super().lambda_cleave(sigma_scaled, T)

        # Positive configurational work is a thermodynamic sign condition, not an
        # athermal toughness.  Suppress hazard accumulation only when the event
        # field has no positive energy release at all.
        if context is not None:
            scale = probe_to_event_energy_scale(
                context.K_event_Pa_sqrt_m,
                context.K_probe_Pa_sqrt_m,
            )
            J_event = max(float(context.J_probe_J_per_m2), 0.0) * scale
            if J_event <= 0.0:
                return 0.0, 0.0, barrier
        return effective, raw, barrier

    def _integrate_coupled(self, *args, **kwargs):
        result = super()._integrate_coupled(*args, **kwargs)
        dB = max(float(result.get("dB", 0.0)), 0.0)
        proposed_checkpoint = max(
            float(result.get("hazard_energy_gate_proposed_checkpoint_m", 0.0)),
            0.0,
        )
        proposed_step = proposed_checkpoint * dB
        accepted_step = max(
            float(result.get("hazard_energy_gate_accepted_step_m", 0.0)),
            0.0,
        )
        J_event = max(float(result.get("J_event_scaled_J_per_m2", 0.0)), 0.0)
        Gamma = max(float(result.get("Gamma_haz_J_per_m2", 0.0)), 0.0)
        self.hazard_energy_available_event_accum_J_per_m += J_event * proposed_step
        self.hazard_energy_dissipated_event_accum_J_per_m += Gamma * accepted_step

        if bool(result.get("fired", False)):
            available = float(self.hazard_energy_available_event_accum_J_per_m)
            dissipated = float(self.hazard_energy_dissipated_event_accum_J_per_m)
            tolerance = 1.0e-12 * max(abs(available), abs(dissipated), 1.0)
            if dissipated > available + tolerance:
                raise RuntimeError(
                    "hazard-energy transaction violated integrated work balance: "
                    f"dissipated={dissipated:.9e} J/m, "
                    f"available={available:.9e} J/m"
                )
            update = {
                "energy_available_integrated_J_per_m": available,
                "energy_dissipated_integrated_J_per_m": dissipated,
                "energy_margin_integrated_J_per_m": available - dissipated,
                "integrated_energy_balance_pass": True,
            }
            result.update(update)
            if self.hazard_energy_gate_event_history:
                self.hazard_energy_gate_event_history[-1].update(update)
                self.hazard_energy_gate_last = dict(
                    self.hazard_energy_gate_event_history[-1]
                )
            if _avalanche._PENDING_GEOMETRY_EVENTS:
                descriptor = _avalanche._PENDING_GEOMETRY_EVENTS[-1]
                gate_payload = descriptor.setdefault("hazard_energy_gate", {})
                gate_payload.update(update)
            self.hazard_energy_available_event_accum_J_per_m = 0.0
            self.hazard_energy_dissipated_event_accum_J_per_m = 0.0
        else:
            result.update(
                {
                    "energy_available_integrated_J_per_m": float(
                        self.hazard_energy_available_event_accum_J_per_m
                    ),
                    "energy_dissipated_integrated_J_per_m": float(
                        self.hazard_energy_dissipated_event_accum_J_per_m
                    ),
                }
            )
        return result

    def restore_geometry_veto(self, n_restore=1):
        raise RuntimeError(
            "v10.2.30 geometry realization failed after continuous energy-gated "
            "MPZ translation. Exact rollback requires replay of the coupled step; "
            "the run is stopped rather than restoring only scalar renewal fields."
        )

    def predict_clock_increment(self, K, T, dt):
        self._refresh_observed_context(float(K), "monotonic")
        return super().predict_clock_increment(K, T, dt)

    def preview_cycle_waveform(self, controller, waveform, T_K: float):
        self._refresh_observed_context(float(waveform.Kmax), "cyclic")
        return super().preview_cycle_waveform(controller, waveform, T_K)

    def step(self, K, T, dt):
        self._refresh_observed_context(float(K), "monotonic")
        result = super().step(K, T, dt)
        result.update(self._observed_gate_diagnostics())
        return result

    def cycle_step_waveform(
        self,
        controller,
        waveform,
        T_K: float,
        requested_cycles=None,
        force_cycles=None,
    ):
        self._refresh_observed_context(float(waveform.Kmax), "cyclic")
        result = super().cycle_step_waveform(
            controller,
            waveform,
            T_K,
            requested_cycles=requested_cycles,
            force_cycles=force_cycles,
        )
        result.update(self._observed_gate_diagnostics())
        return result

    def _observed_gate_diagnostics(self) -> dict[str, Any]:
        context = self.hazard_energy_gate_context
        return {
            "observed_hazard_energy_gate_model_id": MODEL_ID,
            "hazard_energy_sigma_physical_Pa": float(
                self.hazard_energy_last_sigma_physical_Pa
            ),
            "hazard_energy_sigma_scaled_Pa": float(
                self.hazard_energy_last_sigma_scaled_Pa
            ),
            "hazard_energy_K_event_Pa_sqrt_m": float(
                self.hazard_energy_last_K_event_Pa_sqrt_m
            ),
            "hazard_energy_K_probe_Pa_sqrt_m": float(
                self.hazard_energy_last_K_probe_Pa_sqrt_m
            ),
            "hazard_energy_gamma_rel": (
                1.0 if context is None else float(context.gamma_rel)
            ),
            "hazard_energy_available_event_accum_J_per_m": float(
                self.hazard_energy_available_event_accum_J_per_m
            ),
            "hazard_energy_dissipated_event_accum_J_per_m": float(
                self.hazard_energy_dissipated_event_accum_J_per_m
            ),
            "hazard_energy_absolute_athermal_Gc_active": False,
        }


__all__ = [
    "MODEL_ID",
    "ObservedHazardEnergyGatedPersistentSiteCyclicTipEngine",
]
