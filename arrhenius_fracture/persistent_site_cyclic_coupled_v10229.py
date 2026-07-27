"""State-coupled cyclic commit for the v10.2.29 persistent-site engine."""
from __future__ import annotations

from .persistent_site_coupled_hazard_v10229 import integrate_state_coupled_waveform
from .persistent_site_cyclic_v10229 import (
    PersistentSiteCyclicTipEngine,
    cycle_count_from_consumed_time,
)


MODEL_ID = "v10.2.29_persistent_site_state_coupled_cyclic"


class CoupledPersistentSiteCyclicTipEngine(PersistentSiteCyclicTipEngine):
    """Use one state-coupled waveform propagator for block selection and commit."""

    persistent_site_cyclic_v10229 = True
    persistent_site_coupled_hazard_v10229 = True

    def cycle_step_waveform(
        self, controller, waveform, T_K: float, requested_cycles=None, force_cycles=None
    ):
        pred = self.preview_cycle_waveform(controller, waveform, T_K)
        if force_cycles is None:
            block = self._choose_engine_native_cycles(controller, pred, requested_cycles)
            cycles_requested = float(block["cycles"])
        else:
            cycles_requested = max(float(force_cycles), 0.0)
            block = {
                "cycles": cycles_requested,
                "limiter": "global_forced",
                "unlimited_cycles": cycles_requested,
                "candidate_limits": {"global_forced": cycles_requested},
            }

        dt_requested = cycles_requested * float(waveform.period_s)
        B_pre = float(self.B)
        mobile_pre = float(self.mpz.mobile_count)
        retained_pre = float(self.mpz.retained_count)
        escaped_pre = float(self.mpz.escaped_total)
        emitted_pre = float(self.mpz.emitted_total)
        N_pre = float(self.N_em)

        coupled = integrate_state_coupled_waveform(
            self,
            controller,
            waveform,
            T_K,
            cycles_requested,
        )
        cycles_consumed = cycle_count_from_consumed_time(
            coupled["dt_consumed"], waveform.frequency_Hz
        )
        cycles_unused = max(cycles_requested - cycles_consumed, 0.0)
        advance = coupled["advance"]
        plastic = coupled["plastic"]
        diagnostics = self.mpz.diagnostics(self.G, self.nu, self.b, self.f.r0)
        active_signed = self._active_shielding_signed()
        wake_signed = self._wake_shielding_signed()
        self.K_prev = float(waveform.Kmax)

        dN_emit = max(float(self.mpz.emitted_total) - emitted_pre, 0.0)
        dN_mobile = abs(float(self.mpz.mobile_count) - mobile_pre)
        dN_retained = abs(float(self.mpz.retained_count) - retained_pre)
        dN_escape = max(float(self.mpz.escaped_total) - escaped_pre, 0.0)
        coupled_audit = {
            key: value
            for key, value in coupled.items()
            if str(key).startswith("coupled_hazard_")
        }

        return {
            "cycles": cycles_consumed,
            "cycles_requested": cycles_requested,
            "cycles_consumed": cycles_consumed,
            "cycles_unused": cycles_unused,
            "cycle_event_localized": bool(coupled["fired"] and cycles_unused > 0.0),
            "cycle_limiter": str(block.get("limiter", "unknown")),
            "cycle_unlimited": float(block.get("unlimited_cycles", cycles_requested)),
            "cycle_candidate_limits": dict(block.get("candidate_limits", {})),
            "time_s": self.t,
            "Kmax_Pa_sqrt_m": float(waveform.Kmax),
            "Kmin_Pa_sqrt_m": float(waveform.R * waveform.Kmax),
            "DeltaK_Pa_sqrt_m": float(waveform.DeltaK),
            "R": float(waveform.R),
            "frequency_Hz": float(waveform.frequency_Hz),
            "T_K": float(T_K),
            "mu_emit": float(pred.mu_emit),
            "mu_peierls": float(pred.mu_peierls),
            "mu_taylor": float(pred.mu_taylor),
            "mu_escape": float(pred.mu_escape),
            "mu_cleave_pred": float(pred.mu_cleave),
            "store_per_cycle": float(pred.store_per_cycle),
            "mobile_per_cycle": float(pred.mobile_per_cycle),
            "escape_per_cycle": float(pred.escape_per_cycle),
            "peierls_per_cycle": float(pred.peierls_per_cycle),
            "taylor_per_cycle": float(pred.taylor_per_cycle),
            "storage_fraction": float(pred.storage_fraction),
            "lambda_e": float(pred.mu_emit * waveform.frequency_Hz),
            "lambda_c": float(coupled["lambda_c"]),
            "lambda_c_raw": float(coupled["lambda_c_raw"]),
            "B_pre": B_pre,
            "B": float(self.B),
            "N_em": float(self.N_em),
            "N_em_pre_renewal": N_pre,
            "N_em_retained": float(self.N_em),
            "N_em_shed_to_wake": float(
                advance.get("wake_mobile", 0.0) + advance.get("wake_retained", 0.0)
            ),
            "dN_emit_block": dN_emit,
            "dN_store_block": dN_retained,
            "dN_mobile_block": dN_mobile,
            "dN_escape_block": dN_escape,
            "dN_peierls_block": float(pred.peierls_per_cycle * cycles_consumed),
            "dN_taylor_block": float(pred.taylor_per_cycle * cycles_consumed),
            "dB_block": float(coupled["dB"]),
            "physical_hazard_action_block": float(
                coupled.get("physical_hazard_action_step", 0.0)
            ),
            "fired": bool(coupled["fired"]),
            "n_fire": int(coupled["n_fire"]),
            "v_crack": float(coupled["v_crack"]),
            "kinetic_tip_cell_active": True,
            "kinetic_micro_advance_step_m": float(coupled["da"]),
            "kinetic_micro_advance_total_m": float(self.micro_advance_total_m),
            "kinetic_active_K_shield_signed_Pa_sqrt_m": float(active_signed),
            "kinetic_wake_K_shield_signed_Pa_sqrt_m": float(wake_signed),
            "kinetic_internal_substeps": int(coupled["microsteps"]),
            "kinetic_dt_requested_s": dt_requested,
            "kinetic_dt_consumed_s": float(coupled["dt_consumed"]),
            "kinetic_dt_unused_s": float(coupled["dt_unused"]),
            "persistent_site_cyclic_model_id": MODEL_ID,
            "persistent_site_engine_native_predictor": True,
            "legacy_fatigue_barrier_predictor_used": False,
            **coupled_audit,
            **plastic,
            **advance,
            **diagnostics,
        }


__all__ = ["MODEL_ID", "CoupledPersistentSiteCyclicTipEngine"]
