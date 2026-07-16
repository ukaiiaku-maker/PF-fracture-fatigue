"""Coupled moving-tip kinetic cell for the v10.1 sharp-front solver.

The outer anisotropic FEM still refreshes geometry and directional J at a
coarse checkpoint length (normally 5 micrometres).  Inside that interval this
engine does not jump the process zone.  Instead it maps cleavage action to a
continuous mean crack velocity

    v_c = L_checkpoint * lambda_c,

and advances the moving-frame source/mobile/retained/slip fields continuously
while plastic reactions evolve in parallel.  The relation preserves the mean
velocity of the former renewal model: one checkpoint length per mean
first-passage time.  ``packet_length_m`` is retained for stochastic packet-rate
and variance diagnostics; it is not a numerical geometry increment.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .config import EV_TO_J
from .unified_front import UnifiedMPZFrontEngine


@dataclass
class KineticTipConfig:
    """Numerical and physical controls for the moving-tip cell."""

    enabled: bool = True
    plasticity_enabled: bool = True
    active_shielding: bool = True
    signed_active_shielding: bool = True
    mobile_shield_fraction: float = 1.0
    packet_length_m: float = 2.5e-10
    velocity_scale: float = 1.0
    max_action_substep: float = 0.02
    max_translation_substep_m: float = 1.0e-7
    min_substep_s: float = 1.0e-15
    max_internal_steps: int = 20000
    coupling_scheme: str = "strang"

    def validate(self) -> "KineticTipConfig":
        if self.packet_length_m <= 0.0:
            raise ValueError("packet_length_m must be positive")
        if self.velocity_scale < 0.0:
            raise ValueError("velocity_scale must be non-negative")
        if not (0.0 < self.max_action_substep <= 1.0):
            raise ValueError("max_action_substep must lie in (0, 1]")
        if self.max_translation_substep_m <= 0.0:
            raise ValueError("max_translation_substep_m must be positive")
        if self.min_substep_s <= 0.0:
            raise ValueError("min_substep_s must be positive")
        if self.max_internal_steps < 1:
            raise ValueError("max_internal_steps must be at least one")
        if self.coupling_scheme != "strang":
            raise ValueError("only coupling_scheme='strang' is currently supported")
        return self


class KineticMovingTipFrontEngine(UnifiedMPZFrontEngine):
    """Unified MPZ front engine with continuous moving-frame crack kinetics.

    ``B`` remains the fractional progress toward the next outer FEM geometry
    checkpoint.  During every call, however, ``dB`` also produces a continuous
    local advance ``da = f.da * dB``.  The MPZ is translated by that amount
    immediately, so source exposure, transport, retention, shielding, and
    blunting evolve throughout the interval rather than being shifted by a full
    5 micrometres after the clock fires.
    """

    kinetic_tip_cell_active = True
    _default_tip_config = KineticTipConfig()
    _audit_records: list[dict[str, Any]] = []
    _next_engine_id = 1

    @classmethod
    def configure_default(cls, cfg: KineticTipConfig) -> None:
        cls._default_tip_config = copy.deepcopy(cfg).validate()

    @classmethod
    def reset_audit(cls) -> None:
        cls._audit_records = []
        cls._next_engine_id = 1

    @classmethod
    def audit_payload(cls) -> dict[str, Any]:
        return {
            "schema": "v10.1_kinetic_moving_tip_cell",
            "config": asdict(cls._default_tip_config),
            "records": list(cls._audit_records),
        }

    def __init__(self, fcfg, cleave_barrier, emit_barrier, G_shear: float,
                 nu: float, b: float, manifest, mpz_cfg):
        super().__init__(fcfg, cleave_barrier, emit_barrier, G_shear, nu, b,
                         manifest, mpz_cfg)
        self.tip_cfg = copy.deepcopy(type(self)._default_tip_config).validate()
        self.mpz.cfg.mobile_shield_fraction = float(
            self.tip_cfg.mobile_shield_fraction
        )
        self.micro_advance_total_m = 0.0
        self.checkpoint_advance_total_m = 0.0
        self.packet_count_mean_total = 0.0
        self.packet_variance_total_m2 = 0.0
        self._engine_id = type(self)._next_engine_id
        type(self)._next_engine_id += 1

    def clone_split(self, daughter_fraction=0.5):
        child = super().clone_split(daughter_fraction)
        child.tip_cfg = copy.deepcopy(self.tip_cfg)
        child.micro_advance_total_m = self.micro_advance_total_m
        child.checkpoint_advance_total_m = 0.0
        child.packet_count_mean_total = self.packet_count_mean_total
        child.packet_variance_total_m2 = self.packet_variance_total_m2
        child._engine_id = type(self)._next_engine_id
        type(self)._next_engine_id += 1
        return child

    def _active_shielding_signed(self) -> float:
        if not self.tip_cfg.active_shielding:
            return 0.0
        raw = self.mpz._shielding_raw(
            self.mpz.retained,
            self.mpz.mobile,
            self.mpz.x,
            self.G,
            self.nu,
            self.b,
        )
        if self.tip_cfg.signed_active_shielding:
            return float(raw)
        return max(float(raw), 0.0)

    def _wake_shielding_signed(self) -> float:
        if not bool(self.mpz.cfg.wake_shielding):
            return 0.0
        raw = float(self.mpz.cfg.wake_shield_projection) * self.mpz._shielding_raw(
            self.mpz.wake_retained,
            self.mpz.wake_mobile,
            self.mpz.wake_x,
            self.G,
            self.nu,
            self.b,
        )
        return float(raw) if self.tip_cfg.signed_active_shielding else max(float(raw), 0.0)

    def K_shield(self):
        return self._active_shielding_signed() + self._wake_shielding_signed()

    @staticmethod
    def _sum_numeric(target: dict[str, float], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, (bool, np.bool_)):
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                target[key] = target.get(key, 0.0) + float(value)

    def _plastic_half_step(self, dt: float, T: float, stress: float) -> dict[str, float]:
        if dt <= 0.0 or not self.tip_cfg.plasticity_enabled:
            return {
                "dN_emit": 0.0,
                "dN_trapped": 0.0,
                "dN_released": 0.0,
                "dN_recovered": 0.0,
                "dN_escaped": 0.0,
                "source_sites_refreshed": 0.0,
            }
        result = self.mpz.evolve(dt, T, stress, self.b)
        self.W_emit += (
            max(float(stress), 0.0)
            * self.b
            * self.f.L_pz
            * max(float(result.get("dN_emit", 0.0)), 0.0)
        )
        return result

    def _substep_limit(self, dt_remaining: float, lam: float) -> float:
        h = max(float(dt_remaining), 0.0)
        if lam > 0.0 and math.isfinite(lam):
            h = min(h, self.tip_cfg.max_action_substep / lam)
            denom = max(float(self.f.da) * self.tip_cfg.velocity_scale * lam, 1.0e-300)
            h = min(h, self.tip_cfg.max_translation_substep_m / denom)
            remaining_action = max(1.0 - float(self.B), 0.0)
            if remaining_action > 0.0:
                h = min(h, remaining_action / lam)
        return max(min(h, dt_remaining), min(self.tip_cfg.min_substep_s, dt_remaining))

    def _integrate_coupled(self, K: float, T: float, dt: float,
                           stress_override: float | None = None,
                           lambda_override: float | None = None) -> dict[str, Any]:
        dt_requested = max(float(dt), 0.0)
        remaining = dt_requested
        consumed = 0.0
        dB_total = 0.0
        da_total = 0.0
        packet_mean = 0.0
        packet_variance = 0.0
        totals: dict[str, float] = {}
        wake_totals: dict[str, float] = {}
        fired = False
        microsteps = 0
        last_lam = 0.0
        last_raw = 0.0
        last_Gc = 0.0
        last_sig = self.sigma_tip(K) if stress_override is None else max(float(stress_override), 0.0)

        while remaining > 0.0:
            microsteps += 1
            if microsteps > self.tip_cfg.max_internal_steps:
                raise RuntimeError(
                    "kinetic tip-cell exceeded max_internal_steps; reduce the outer "
                    "time increment or increase --kinetic-max-action-substep"
                )

            sig0 = self.sigma_tip(K) if stress_override is None else max(float(stress_override), 0.0)
            lam0, raw0, Gc0 = self.lambda_cleave(sig0, T)
            if lambda_override is not None:
                lam0 = max(float(lambda_override), 0.0)
                raw0 = lam0
            lam0 = max(float(lam0), 0.0) if math.isfinite(lam0) else 0.0
            h = self._substep_limit(remaining, lam0)

            # Transactional localization: plastic shielding can alter lambda during
            # the first half-step.  Restore and repeat once if that would cross the
            # next FEM checkpoint prematurely.
            mpz_before = self.mpz.copy()
            W_before = float(self.W_emit)
            first = self._plastic_half_step(0.5 * h, T, sig0)
            sig_mid = self.sigma_tip(K) if stress_override is None else max(float(stress_override), 0.0)
            lam_mid, raw_mid, Gc_mid = self.lambda_cleave(sig_mid, T)
            if lambda_override is not None:
                lam_mid = max(float(lambda_override), 0.0)
                raw_mid = lam_mid
            lam_mid = max(float(lam_mid), 0.0) if math.isfinite(lam_mid) else 0.0
            remaining_action = max(1.0 - float(self.B), 0.0)
            if lam_mid > 0.0 and lam_mid * h > remaining_action + 1.0e-12:
                self.mpz = mpz_before
                self.W_emit = W_before
                h = max(remaining_action / lam_mid, self.tip_cfg.min_substep_s)
                h = min(h, remaining)
                first = self._plastic_half_step(0.5 * h, T, sig0)
                sig_mid = self.sigma_tip(K) if stress_override is None else max(float(stress_override), 0.0)
                lam_mid, raw_mid, Gc_mid = self.lambda_cleave(sig_mid, T)
                if lambda_override is not None:
                    lam_mid = max(float(lambda_override), 0.0)
                    raw_mid = lam_mid
                lam_mid = max(float(lam_mid), 0.0) if math.isfinite(lam_mid) else 0.0

            dB = min(lam_mid * h, max(1.0 - float(self.B), 0.0))
            da = float(self.f.da) * self.tip_cfg.velocity_scale * dB
            packet_rate = (
                float(self.f.da)
                * self.tip_cfg.velocity_scale
                / self.tip_cfg.packet_length_m
                * lam_mid
            )
            packet_n = packet_rate * h
            packet_var = self.tip_cfg.packet_length_m ** 2 * packet_n

            advance = self.mpz.advance(da) if da > 0.0 else {}
            sig1 = self.sigma_tip(K) if stress_override is None else max(float(stress_override), 0.0)
            second = self._plastic_half_step(0.5 * h, T, sig1)

            self._sum_numeric(totals, first)
            self._sum_numeric(totals, second)
            self._sum_numeric(wake_totals, advance)
            self.B += dB
            self.micro_advance_total_m += da
            self.packet_count_mean_total += packet_n
            self.packet_variance_total_m2 += packet_var
            dB_total += dB
            da_total += da
            packet_mean += packet_n
            packet_variance += packet_var
            consumed += h
            remaining = max(remaining - h, 0.0)
            self.t += h
            last_lam, last_raw, last_Gc, last_sig = lam_mid, raw_mid, Gc_mid, sig1

            if self.B >= 1.0 - 1.0e-10:
                self.B = max(self.B - 1.0, 0.0)
                self.a_adv += float(self.f.da)
                self.checkpoint_advance_total_m += float(self.f.da)
                self.n_adv += 1
                fired = True
                break

            if h <= 0.0:
                break

        return {
            "fired": fired,
            "n_fire": 1 if fired else 0,
            "v_crack": da_total / consumed if consumed > 0.0 else 0.0,
            "dB": dB_total,
            "da": da_total,
            "dt_consumed": consumed,
            "dt_unused": max(dt_requested - consumed, 0.0),
            "packet_mean": packet_mean,
            "packet_variance_m2": packet_variance,
            "lambda_c": last_lam,
            "lambda_c_raw": last_raw,
            "Gc_J": last_Gc,
            "sigma_tip": last_sig,
            "plastic": totals,
            "advance": wake_totals,
            "microsteps": microsteps,
        }

    def step(self, K, T, dt):
        dt = max(float(dt), 0.0)
        K = max(float(K), 0.0)
        N_pre = self.N_em
        Kshield_pre = self.K_shield()
        r_pre = self.r_eff()
        sig_pre = self.sigma_tip(K)
        lam_e, sig_e, Ge = self.lambda_emit(sig_pre, T)

        coupled = self._integrate_coupled(K, T, dt)
        sig_post = self.sigma_tip(K)
        diagnostics = self.mpz.diagnostics(self.G, self.nu, self.b, self.f.r0)
        active_signed = self._active_shielding_signed()
        wake_signed = self._wake_shielding_signed()
        advance = coupled["advance"]
        plastic = coupled["plastic"]

        record = {
            "engine_id": self._engine_id,
            "material_class": self.manifest.name,
            "time_s": self.t,
            "K_Pa_sqrt_m": K,
            "sigma_tip_Pa": sig_post,
            "lambda_c_s-1": coupled["lambda_c"],
            "crack_velocity_m_s": coupled["v_crack"],
            "micro_advance_step_m": coupled["da"],
            "micro_advance_total_m": self.micro_advance_total_m,
            "checkpoint_progress_action": self.B,
            "checkpoint_committed_total_m": self.checkpoint_advance_total_m,
            "active_mobile": self.mpz.mobile_count,
            "active_retained": self.mpz.retained_count,
            "active_K_shield_signed_Pa_sqrt_m": active_signed,
            "wake_K_shield_signed_Pa_sqrt_m": wake_signed,
            "fired": bool(coupled["fired"]),
            "microsteps": coupled["microsteps"],
        }
        type(self)._audit_records.append(record)

        return {
            "fired": coupled["fired"],
            "n_fire": coupled["n_fire"],
            "v_crack": coupled["v_crack"],
            "sigma_tip": sig_post,
            "sigma_back": 0.0,
            "lambda_e": lam_e,
            "lambda_c": coupled["lambda_c"],
            "lambda_c_raw": coupled["lambda_c_raw"],
            "B": self.B,
            "N_em": self.N_em,
            "r_eff": self.r_eff(),
            "dG_emb_eV": 0.0,
            "G_cleave_eff_eV": coupled["Gc_J"] / EV_TO_J,
            **self.cleavage_diagnostics(sig_post, T),
            "G_emit_eV": Ge / EV_TO_J,
            "W_emit": self.W_emit,
            "sigma_tip_uncapped": (
                max(K - Kshield_pre, 0.0)
                / math.sqrt(2.0 * math.pi * max(r_pre, 1.0e-30))
            ),
            "sigma_cap_active": bool(self.f.sigma_cap > 0.0 and sig_post >= self.f.sigma_cap),
            "dN_emit_raw": float(plastic.get("dN_emit", 0.0)),
            "dN_cap_active": False,
            "N_sat_factor": 1.0,
            "N_sat_active": False,
            "N_em_pre_renewal": N_pre,
            "N_em_retained": self.N_em,
            "N_em_shed_to_wake": float(
                advance.get("wake_mobile", 0.0)
                + advance.get("wake_retained", 0.0)
            ),
            "sigma_back_pre_renewal": 0.0,
            "r_eff_pre_renewal": r_pre,
            "dG_emb_pre_renewal_eV": 0.0,
            "dB_step": coupled["dB"],
            "one_renewal_transaction": True,
            "material_class": self.manifest.name,
            "candidate_id": self.manifest.candidate_id,
            "kinetic_tip_cell_active": True,
            "kinetic_plasticity_enabled": self.tip_cfg.plasticity_enabled,
            "kinetic_active_shielding_enabled": self.tip_cfg.active_shielding,
            "kinetic_signed_active_shielding": self.tip_cfg.signed_active_shielding,
            "kinetic_mobile_shield_fraction": self.tip_cfg.mobile_shield_fraction,
            "kinetic_micro_advance_step_m": coupled["da"],
            "kinetic_micro_advance_total_m": self.micro_advance_total_m,
            "kinetic_checkpoint_progress_m": self.B * float(self.f.da),
            "kinetic_checkpoint_committed_total_m": self.checkpoint_advance_total_m,
            "kinetic_packet_rate_s-1": (
                float(self.f.da)
                * self.tip_cfg.velocity_scale
                / self.tip_cfg.packet_length_m
                * coupled["lambda_c"]
            ),
            "kinetic_packet_mean_step": coupled["packet_mean"],
            "kinetic_packet_variance_step_m2": coupled["packet_variance_m2"],
            "kinetic_dt_consumed_s": coupled["dt_consumed"],
            "kinetic_dt_unused_s": coupled["dt_unused"],
            "kinetic_internal_substeps": coupled["microsteps"],
            "kinetic_active_K_shield_signed_Pa_sqrt_m": active_signed,
            "kinetic_wake_K_shield_signed_Pa_sqrt_m": wake_signed,
            "kinetic_total_K_shield_signed_Pa_sqrt_m": active_signed + wake_signed,
            **plastic,
            **advance,
            **diagnostics,
        }

    def cycle_step_waveform(self, controller, waveform, T_K: float,
                            requested_cycles=None, force_cycles=None):
        """Tau-leap cyclic update using the same moving-tip state.

        Waveform hazards are integrated over phase points.  Plastic reactions use
        the emission-hazard-weighted mean stress, while the exact phase-averaged
        cleavage propensity supplies the crack velocity.  This retains the
        existing cycle-block approximation but removes the post-block 5 um jump.
        """
        phase = controller._phases()
        Kvals = waveform.K_phase(phase)
        dt_phase = waveform.period_s / len(phase)
        sig = np.array([self.sigma_tip(float(k)) for k in Kvals])
        lam_e_site = self.manifest.emission.rate(sig, T_K)
        available = float(np.sum(self.mpz.available_sites))
        mu_emit = float(np.sum(lam_e_site * available) * dt_phase)
        lam_c_phase = np.array([
            self.lambda_cleave(float(s), T_K)[0] for s in sig
        ])
        mu_c = float(np.sum(lam_c_phase) * dt_phase)
        limits = [
            float(requested_cycles if requested_cycles is not None else controller.cfg.block_cycles),
            float(controller.cfg.max_block_cycles),
        ]
        if controller.cfg.adaptive_cycles:
            if mu_c > 0.0 and math.isfinite(controller.cfg.target_dB):
                limits.append(controller.cfg.target_dB / mu_c)
            if mu_emit > 0.0 and math.isfinite(controller.cfg.target_dN_emit):
                limits.append(controller.cfg.target_dN_emit / mu_emit)
        cycles = max(
            float(force_cycles) if force_cycles is not None else min(limits),
            float(controller.cfg.min_block_cycles),
        )
        cycles = min(cycles, float(controller.cfg.max_block_cycles))
        dt_block = cycles * waveform.period_s
        weights = np.maximum(lam_e_site, 0.0)
        avg_sig = (
            float(np.sum(weights * sig) / np.sum(weights))
            if np.sum(weights) > 0.0 else float(np.mean(sig))
        )
        lambda_avg = mu_c * waveform.frequency_Hz
        N_pre = self.N_em
        coupled = self._integrate_coupled(
            waveform.Kmax,
            T_K,
            dt_block,
            stress_override=avg_sig,
            lambda_override=lambda_avg,
        )
        advance = coupled["advance"]
        plastic = coupled["plastic"]
        diag = self.mpz.diagnostics(self.G, self.nu, self.b, self.f.r0)
        active_signed = self._active_shielding_signed()
        wake_signed = self._wake_shielding_signed()
        return {
            "cycles": cycles,
            "cycle_limiter": "kinetic_moving_tip_hazard_state",
            "cycle_unlimited": cycles,
            "time_s": self.t,
            "Kmax_Pa_sqrt_m": waveform.Kmax,
            "DeltaK_Pa_sqrt_m": waveform.DeltaK,
            "R": waveform.R,
            "frequency_Hz": waveform.frequency_Hz,
            "T_K": T_K,
            "mu_emit": mu_emit,
            "mu_cleave_pred": mu_c,
            "lambda_e": mu_emit * waveform.frequency_Hz,
            "lambda_c": lambda_avg,
            "lambda_c_raw": lambda_avg,
            "B": self.B,
            "N_em": self.N_em,
            "N_em_pre_renewal": N_pre,
            "N_em_retained": self.N_em,
            "N_em_shed_to_wake": float(
                advance.get("wake_mobile", 0.0)
                + advance.get("wake_retained", 0.0)
            ),
            "dN_emit": float(plastic.get("dN_emit", 0.0)),
            "dN_emit_raw": float(plastic.get("dN_emit", 0.0)),
            "dB": coupled["dB"],
            "fired": coupled["fired"],
            "n_fire": coupled["n_fire"],
            "v_crack": coupled["v_crack"],
            "kinetic_tip_cell_active": True,
            "kinetic_micro_advance_step_m": coupled["da"],
            "kinetic_micro_advance_total_m": self.micro_advance_total_m,
            "kinetic_active_K_shield_signed_Pa_sqrt_m": active_signed,
            "kinetic_wake_K_shield_signed_Pa_sqrt_m": wake_signed,
            "kinetic_internal_substeps": coupled["microsteps"],
            **plastic,
            **advance,
            **diag,
        }
