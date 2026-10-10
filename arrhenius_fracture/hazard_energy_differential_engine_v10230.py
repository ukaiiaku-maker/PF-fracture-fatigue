"""Microstep-resolved v10.2.30 hazard-energy-gated production engine."""
from __future__ import annotations

import math
from typing import Any

from . import stochastic_avalanche_tip as _avalanche
from .hazard_energy_observed_engine_v10230 import (
    ObservedHazardEnergyGatedPersistentSiteCyclicTipEngine,
)
from .stochastic_hazard_tip import normalized_progress_rate

MODEL_ID = "v10.2.30_microstep_hazard_energy_gated_engine"


class DifferentialHazardEnergyGatedPersistentSiteCyclicTipEngine(
    ObservedHazardEnergyGatedPersistentSiteCyclicTipEngine
):
    """Apply the thermodynamic crack reward gate inside every Strang microstep."""

    differential_hazard_energy_gate_v10230 = True

    @classmethod
    def audit_payload(cls) -> dict[str, Any]:
        payload = super().audit_payload()
        payload["differential_hazard_energy_gate_v10230"] = {
            "model_id": MODEL_ID,
            "gate_resolution": "every_internal_Strang_microstep",
            "hazard_clock_modified": False,
            "plasticity_opening_modified": False,
            "accepted_da": "proposed_event_length*gate_fraction*dB",
            "work_inequality": "Gamma_haz*da_accepted<=J_event*da_proposed",
            "event_geometry_length": "sum_of_accepted_microstep_advances",
        }
        return payload

    def _integrate_coupled(
        self,
        K: float,
        T: float,
        dt: float,
        stress_override: float | None = None,
        lambda_override: float | None = None,
    ) -> dict[str, Any]:
        self._synchronize_driver_checkpoint_length()
        dt_requested = max(float(dt), 0.0)
        remaining = dt_requested
        consumed = 0.0
        dB_total = 0.0
        dH_total = 0.0
        da_total = 0.0
        da_proposed_total = 0.0
        packet_mean = 0.0
        packet_variance = 0.0
        totals: dict[str, float] = {}
        wake_totals: dict[str, float] = {}
        fired = False
        microsteps = 0
        last_lam = 0.0
        last_raw = 0.0
        last_Gc = 0.0
        last_sig = (
            self.sigma_tip(K)
            if stress_override is None
            else max(float(stress_override), 0.0)
        )
        completed_threshold = 0.0
        completed_action = 0.0
        last_gate: dict[str, Any] = {}
        accepted_event = 0.0
        proposed_event_length = max(float(self.avalanche_event_advance_m), 0.0)
        proposed_event_factor = max(float(self.avalanche_event_length_factor), 0.0)
        velocity_scale = max(float(self.tip_cfg.velocity_scale), 0.0)

        while remaining > 0.0:
            microsteps += 1
            if microsteps > self.tip_cfg.max_internal_steps:
                raise RuntimeError(
                    "v10.2.30 kinetic tip-cell exceeded max_internal_steps; "
                    "reduce the outer time/cycle block"
                )

            threshold = max(float(self.hazard_threshold_action), 1.0e-300)
            sig0 = (
                self.sigma_tip(K)
                if stress_override is None
                else max(float(stress_override), 0.0)
            )
            lam0, raw0, Gc0 = self.lambda_cleave(sig0, T)
            if lambda_override is not None:
                lam0 = max(float(lambda_override), 0.0)
                raw0 = lam0
            lam0 = max(float(lam0), 0.0) if math.isfinite(lam0) else 0.0
            progress0 = normalized_progress_rate(lam0, threshold)
            h = self._substep_limit(remaining, progress0)

            mpz_before = self.mpz.copy()
            W_before = float(self.W_emit)
            first = self._plastic_half_step(0.5 * h, T, sig0)
            sig_mid = (
                self.sigma_tip(K)
                if stress_override is None
                else max(float(stress_override), 0.0)
            )
            lam_mid, raw_mid, Gc_mid = self.lambda_cleave(sig_mid, T)
            if lambda_override is not None:
                lam_mid = max(float(lambda_override), 0.0)
                raw_mid = lam_mid
            lam_mid = (
                max(float(lam_mid), 0.0)
                if math.isfinite(lam_mid)
                else 0.0
            )
            progress_mid = normalized_progress_rate(lam_mid, threshold)
            remaining_progress = max(1.0 - float(self.B), 0.0)

            if (
                progress_mid > 0.0
                and progress_mid * h > remaining_progress + 1.0e-12
            ):
                self.mpz = mpz_before
                self.W_emit = W_before
                h = max(
                    remaining_progress / progress_mid,
                    self.tip_cfg.min_substep_s,
                )
                h = min(h, remaining)
                first = self._plastic_half_step(0.5 * h, T, sig0)
                sig_mid = (
                    self.sigma_tip(K)
                    if stress_override is None
                    else max(float(stress_override), 0.0)
                )
                lam_mid, raw_mid, Gc_mid = self.lambda_cleave(sig_mid, T)
                if lambda_override is not None:
                    lam_mid = max(float(lambda_override), 0.0)
                    raw_mid = lam_mid
                lam_mid = (
                    max(float(lam_mid), 0.0)
                    if math.isfinite(lam_mid)
                    else 0.0
                )
                progress_mid = normalized_progress_rate(lam_mid, threshold)

            dB = min(progress_mid * h, max(1.0 - float(self.B), 0.0))
            dH = dB * threshold
            gate = self._current_gate(T, K, sig_mid)
            gate_fraction = min(max(float(gate["gate_fraction"]), 0.0), 1.0)
            da_proposed = proposed_event_length * velocity_scale * dB
            da = da_proposed * gate_fraction
            packet_rate = (
                proposed_event_length
                * velocity_scale
                * gate_fraction
                / self.tip_cfg.packet_length_m
                * progress_mid
            )
            packet_n = packet_rate * h
            packet_var = self.tip_cfg.packet_length_m ** 2 * packet_n

            advance = self.mpz.advance(da) if da > 0.0 else {}
            sig1 = (
                self.sigma_tip(K)
                if stress_override is None
                else max(float(stress_override), 0.0)
            )
            second = self._plastic_half_step(0.5 * h, T, sig1)

            self._sum_numeric(totals, first)
            self._sum_numeric(totals, second)
            self._sum_numeric(wake_totals, advance)
            self.B += dB
            self.hazard_action_current += dH
            self.micro_advance_total_m += da
            self.hazard_energy_gate_event_advance_accum_m += da
            self.packet_count_mean_total += packet_n
            self.packet_variance_total_m2 += packet_var
            dB_total += dB
            dH_total += dH
            da_total += da
            da_proposed_total += da_proposed
            packet_mean += packet_n
            packet_variance += packet_var
            consumed += h
            remaining = max(remaining - h, 0.0)
            self.t += h
            last_lam, last_raw, last_Gc, last_sig = (
                lam_mid,
                raw_mid,
                Gc_mid,
                sig1,
            )
            self.hazard_last_progress_rate_s = progress_mid
            last_gate = dict(gate)

            available_step = (
                max(float(gate["J_event_scaled_J_per_m2"]), 0.0)
                * da_proposed
            )
            dissipated_step = (
                max(float(gate["Gamma_haz_J_per_m2"]), 0.0) * da
            )
            tolerance_step = 1.0e-12 * max(
                abs(available_step), abs(dissipated_step), 1.0
            )
            if dissipated_step > available_step + tolerance_step:
                raise RuntimeError(
                    "v10.2.30 microstep work inequality failed: "
                    f"dissipated={dissipated_step:.9e} J/m, "
                    f"available={available_step:.9e} J/m"
                )
            self.hazard_energy_available_event_accum_J_per_m += available_step
            self.hazard_energy_dissipated_event_accum_J_per_m += dissipated_step

            if self.B >= 1.0 - 1.0e-10:
                self.B = max(self.B - 1.0, 0.0)
                accepted_event = max(
                    float(self.hazard_energy_gate_event_advance_accum_m), 0.0
                )
                if accepted_event <= 0.0:
                    raise RuntimeError(
                        "cleavage first passage completed without positive "
                        "thermodynamically admissible crack extension"
                    )
                self.a_adv += accepted_event
                self.checkpoint_advance_total_m += accepted_event
                self.n_adv += 1
                fired = True
                completed_threshold = threshold
                completed_action = float(self.hazard_action_current)
                self.hazard_last_completed_threshold = completed_threshold
                self.hazard_last_completed_action = completed_action
                self.hazard_threshold_history.append(completed_threshold)
                self.hazard_event_index += 1
                self.hazard_action_current = 0.0
                self.hazard_threshold_action = self._draw_threshold()

                available = float(
                    self.hazard_energy_available_event_accum_J_per_m
                )
                dissipated = float(
                    self.hazard_energy_dissipated_event_accum_J_per_m
                )
                tolerance = 1.0e-12 * max(
                    abs(available), abs(dissipated), 1.0
                )
                if dissipated > available + tolerance:
                    raise RuntimeError(
                        "v10.2.30 integrated work inequality failed: "
                        f"dissipated={dissipated:.9e} J/m, "
                        f"available={available:.9e} J/m"
                    )
                accepted_factor = accepted_event / max(
                    float(self.avalanche_base_checkpoint_m), 1.0e-300
                )
                effective_gate_fraction = accepted_event / max(
                    proposed_event_length * velocity_scale, 1.0e-300
                )
                event_row = {
                    **last_gate,
                    "event_index": int(self.hazard_event_index - 1),
                    "threshold_action": completed_threshold,
                    "proposed_event_advance_m": (
                        proposed_event_length * velocity_scale
                    ),
                    "proposed_event_length_factor": proposed_event_factor,
                    "accepted_event_advance_m": accepted_event,
                    "accepted_event_length_factor": accepted_factor,
                    "rejected_event_advance_m": max(
                        proposed_event_length * velocity_scale - accepted_event,
                        0.0,
                    ),
                    "effective_event_gate_fraction": effective_gate_fraction,
                    "energy_available_integrated_J_per_m": available,
                    "energy_dissipated_integrated_J_per_m": dissipated,
                    "energy_margin_integrated_J_per_m": available - dissipated,
                    "integrated_energy_balance_pass": True,
                    "gate_evaluated_each_internal_microstep": True,
                }
                self.hazard_energy_gate_last = dict(event_row)
                self.hazard_energy_gate_event_history.append(dict(event_row))
                self.avalanche_last_completed_advance_m = accepted_event
                self.avalanche_last_completed_factor = accepted_factor
                self.avalanche_event_length_history.append(accepted_event)
                _avalanche._PENDING_GEOMETRY_EVENTS.append(
                    {
                        "event_advance_m": accepted_event,
                        "proposed_event_advance_m": (
                            proposed_event_length * velocity_scale
                        ),
                        "event_length_factor": accepted_factor,
                        "proposed_event_length_factor": proposed_event_factor,
                        "threshold_action": completed_threshold,
                        "hazard_seed": int(self.hazard_cfg.seed),
                        "hazard_event_index": int(self.hazard_event_index - 1),
                        "geometry_subsegment_fraction": float(
                            self.avalanche_cfg.geometry_subsegment_fraction
                        ),
                        "hazard_energy_gate": dict(event_row),
                    }
                )
                self.hazard_energy_gate_event_advance_accum_m = 0.0
                self.hazard_energy_available_event_accum_J_per_m = 0.0
                self.hazard_energy_dissipated_event_accum_J_per_m = 0.0
                self._set_current_event_length()
                break

            if h <= 0.0:
                break

        return {
            "fired": fired,
            "n_fire": 1 if fired else 0,
            "v_crack": da_total / consumed if consumed > 0.0 else 0.0,
            "dB": dB_total,
            "physical_hazard_action_step": dH_total,
            "da": da_total,
            "da_proposed": da_proposed_total,
            "dt_consumed": consumed,
            "dt_unused": max(dt_requested - consumed, 0.0),
            "packet_mean": packet_mean,
            "packet_variance_m2": packet_variance,
            "lambda_c": last_lam,
            "lambda_c_raw": last_raw,
            "hazard_progress_rate_s-1": self.hazard_last_progress_rate_s,
            "hazard_threshold_completed_action": completed_threshold,
            "hazard_action_completed": completed_action,
            "hazard_threshold_next_action": float(self.hazard_threshold_action),
            "Gc_J": last_Gc,
            "sigma_tip": last_sig,
            "plastic": totals,
            "advance": wake_totals,
            "microsteps": microsteps,
            **last_gate,
            "hazard_energy_gate_model_id": MODEL_ID,
            "hazard_energy_gate_active": True,
            "hazard_energy_gate_proposed_checkpoint_m": proposed_event_length,
            "hazard_energy_gate_accepted_step_m": da_total,
            "hazard_energy_gate_proposed_step_m": da_proposed_total,
            "hazard_energy_gate_event_accumulated_m": float(
                self.hazard_energy_gate_event_advance_accum_m
            ),
            "hazard_energy_gate_accepted_event_m": accepted_event,
            "energy_available_integrated_J_per_m": (
                float(self.hazard_energy_gate_last.get(
                    "energy_available_integrated_J_per_m", 0.0
                ))
                if fired
                else float(self.hazard_energy_available_event_accum_J_per_m)
            ),
            "energy_dissipated_integrated_J_per_m": (
                float(self.hazard_energy_gate_last.get(
                    "energy_dissipated_integrated_J_per_m", 0.0
                ))
                if fired
                else float(self.hazard_energy_dissipated_event_accum_J_per_m)
            ),
            "avalanche_event_advance_m": accepted_event if fired else 0.0,
            "avalanche_event_length_factor": (
                accepted_event
                / max(float(self.avalanche_base_checkpoint_m), 1.0e-300)
                if fired
                else 0.0
            ),
            "avalanche_current_event_advance_m": float(
                self.avalanche_event_advance_m
            ),
            "avalanche_current_event_length_factor": float(
                self.avalanche_event_length_factor
            ),
        }


__all__ = [
    "MODEL_ID",
    "DifferentialHazardEnergyGatedPersistentSiteCyclicTipEngine",
]
