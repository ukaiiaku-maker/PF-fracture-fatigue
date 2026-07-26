"""Engine-native cyclic integration for the v10.2.21 persistent-site tip state.

The legacy fatigue controller remains responsible only for waveform definition and
adaptive-target configuration. All predicted and committed state increments are
obtained from the current signed moving-process-zone engine. No independent
fatigue barrier, source inventory, recovery law, or Paris-type advance law is used.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from .fatigue_v1 import CycleHazardResult
from .persistent_site_source_v10221 import PersistentSiteStateResolvedTipEngine

MODEL_ID = "v10.2.29_persistent_site_engine_native_cyclic"


def cycle_count_from_consumed_time(dt_consumed_s: float, frequency_Hz: float) -> float:
    """Return physical cycles represented by a committed time interval."""
    return max(float(dt_consumed_s), 0.0) * max(float(frequency_Hz), 0.0)


def _numeric(result: dict[str, Any], key: str) -> float:
    value = result.get(key, 0.0)
    return float(value) if isinstance(value, (int, float, np.integer, np.floating)) else 0.0


class PersistentSiteCyclicTipEngine(PersistentSiteStateResolvedTipEngine):
    """Persistent-site engine with an engine-native cyclic preview and commit."""

    persistent_site_cyclic_v10229 = True

    def preview_cycle_waveform(self, controller, waveform, T_K: float) -> CycleHazardResult:
        """Advance a private clone through one waveform cycle.

        The preview uses the installed persistent source, signed transport,
        shielding, blunting, and moving-frame methods. If first passage occurs
        inside the preview cycle, accumulated changes are normalized by the
        elapsed cycle fraction so the adaptive controller receives local per-cycle
        rates rather than legacy surrogate-barrier rates.
        """
        phase = np.asarray(controller._phases(), dtype=float)
        if phase.size < 1:
            raise ValueError("cyclic preview requires at least one waveform phase")
        Kvals = np.asarray(waveform.K_phase(phase), dtype=float)
        dt_phase = float(waveform.period_s) / float(phase.size)
        trial = copy.deepcopy(self)

        mobile0 = float(trial.mpz.mobile_count)
        retained0 = float(trial.mpz.retained_count)
        emitted = trapped = released = escaped = 0.0
        peierls_progress = taylor_progress = 0.0
        cleavage_action = 0.0
        sig_samples: list[float] = []
        emit_weighted_sig = 0.0
        emit_weight = 0.0
        elapsed_s = 0.0

        for kval in Kvals:
            K = max(float(kval), 0.0)
            half = 0.5 * dt_phase

            sig0 = float(trial.sigma_tip(K))
            first = trial._plastic_half_step(half, T_K, sig0)
            sig_mid = float(trial.sigma_tip(K))
            lam_mid, _raw_mid, _Gc_mid = trial.lambda_cleave(sig_mid, T_K)
            lam_mid = max(float(lam_mid), 0.0) if math.isfinite(lam_mid) else 0.0

            remaining_action = max(1.0 - float(trial.B), 0.0)
            dB_unclipped = lam_mid * dt_phase
            dB = min(dB_unclipped, remaining_action)
            da = float(trial.f.da) * float(trial.tip_cfg.velocity_scale) * dB
            if da > 0.0:
                trial.mpz.advance(da)
            trial.B += dB
            trial.micro_advance_total_m += da

            sig1 = float(trial.sigma_tip(K))
            second = trial._plastic_half_step(half, T_K, sig1)
            trial.t += dt_phase
            elapsed_s += dt_phase
            cleavage_action += dB_unclipped
            sig_samples.extend((sig0, sig_mid, sig1))

            for update in (first, second):
                emitted += _numeric(update, "dN_emit")
                trapped += _numeric(update, "dN_trapped")
                released += _numeric(update, "dN_released")
                escaped += _numeric(update, "dN_escaped")
                peierls_progress += half * max(_numeric(update, "peierls_rate_s"), 0.0)
                taylor_progress += half * max(
                    _numeric(update, "taylor_completion_rate_s"), 0.0
                )
                weight = max(_numeric(update, "dN_emit"), 0.0)
                sigma_eff = float(
                    getattr(
                        trial.mpz,
                        "continuum_source_last_sigma_emit_effective_Pa",
                        sig_mid,
                    )
                )
                emit_weighted_sig += weight * max(sigma_eff, 0.0)
                emit_weight += weight

            if trial.B >= 1.0 - 1.0e-12:
                break

        elapsed_cycles = max(elapsed_s * float(waveform.frequency_Hz), 1.0e-300)
        scale = 1.0 / elapsed_cycles
        mobile_delta = abs(float(trial.mpz.mobile_count) - mobile0)
        retained_delta = abs(float(trial.mpz.retained_count) - retained0)

        emitted *= scale
        trapped *= scale
        released *= scale
        escaped *= scale
        peierls_progress *= scale
        taylor_progress *= scale
        cleavage_action *= scale
        mobile_delta *= scale
        retained_delta *= scale

        stored_per_cycle = max(retained_delta, trapped)
        mobile_per_cycle = max(mobile_delta, emitted + released - trapped)
        avg_sigma = float(np.mean(sig_samples)) if sig_samples else 0.0
        max_sigma = float(np.max(sig_samples)) if sig_samples else 0.0
        avg_emit_sigma = (
            emit_weighted_sig / emit_weight if emit_weight > 0.0 else avg_sigma
        )
        storage_fraction = (
            min(max(stored_per_cycle / emitted, 0.0), 1.0) if emitted > 0.0 else 0.0
        )

        return CycleHazardResult(
            mu_emit=max(emitted, 0.0),
            mu_peierls=max(peierls_progress, 0.0),
            mu_taylor=max(taylor_progress, 0.0),
            mu_escape=max(escaped, 0.0),
            mu_cleave=max(cleavage_action, 0.0),
            store_per_cycle=max(stored_per_cycle, 0.0),
            mobile_per_cycle=max(mobile_per_cycle, 0.0),
            escape_per_cycle=max(escaped, 0.0),
            peierls_per_cycle=max(peierls_progress, 0.0),
            taylor_per_cycle=max(taylor_progress, 0.0),
            avg_sigma_tip=avg_sigma,
            max_sigma_tip=max_sigma,
            avg_sigma_emit_eff=max(float(avg_emit_sigma), 0.0),
            storage_fraction=storage_fraction,
        )

    def _choose_engine_native_cycles(self, controller, pred, requested_cycles=None) -> dict:
        old_target = float(controller.cfg.target_dB)
        remaining = max(1.0 - float(self.B), 0.0)
        try:
            controller.cfg.target_dB = min(old_target, remaining)
            return controller.choose_block_cycles_diagnostic(pred, requested_cycles)
        finally:
            controller.cfg.target_dB = old_target

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

        lambda_avg = pred.mu_cleave * float(waveform.frequency_Hz)
        coupled = self._integrate_coupled(
            float(waveform.Kmax),
            T_K,
            dt_requested,
            # Pass the cycle-averaged opening stress. The persistent-source law
            # applies anisotropic drive factors and Taylor backstress exactly once.
            stress_override=float(pred.avg_sigma_tip),
            lambda_override=lambda_avg,
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
            **plastic,
            **advance,
            **diagnostics,
        }


__all__ = [
    "MODEL_ID",
    "PersistentSiteCyclicTipEngine",
    "cycle_count_from_consumed_time",
]
