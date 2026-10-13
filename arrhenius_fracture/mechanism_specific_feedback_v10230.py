"""Mechanism-specific adapter for the immutable v9.13 state kinetics.

Only the external long-range driving stresses are replaced: cleavage receives
the area-weighted opening stress and each slip system receives its own resolved
FEM shear.  Native short-range GND stress, forest kinetics, backstress,
fractional source integration and prefactors remain unchanged.
"""
from __future__ import annotations

import math
from types import MethodType
import numpy as np


MODEL_ID = "v10.2.30_v3_6_mechanism_specific_feedback_v1"


class MechanismSpecificFeedback:
    def __init__(self, projection, patch_provider, extension_provider):
        self.projection = projection
        self.patch_provider = patch_provider
        self.extension_provider = extension_provider
        self.last_result = None
        self.reference = projection.virgin_correspondence()

    def mechanics(self, K: float):
        self.last_result = self.projection.project(
            self.patch_provider(), crack_extension_m=float(self.extension_provider()),
            K_MPa_sqrt_m=float(K), compute_kinematics=False,
        )
        return self.last_result

    def install(self, state) -> None:
        adapter = self

        def local_rates(this, K_MPa_sqrt_m: float, T_K: float):
            result = adapter.mechanics(K_MPa_sqrt_m)
            singular_per_K = 1.0e6 / math.sqrt(2.0 * math.pi * max(float(this.c.r0_m), float(this.c.b_m), 1e-30))
            tau_ref = np.asarray(adapter.reference["resolved_shear_Pa_at_reference_K"], float)
            tau_external_by_system = np.asarray(result["resolved_shear_Pa"], float) / tau_ref * adapter.projection.reference_K * singular_per_K
            tau_external = tau_external_by_system[:, None]
            tau_gnd = this.tau_gnd_Pa()
            tau_eff = tau_external + tau_gnd

            forest = this.forest_density_m2()
            spacing = 1.0 / (2.0 * np.sqrt(forest))
            jump = this.c.jump_fraction_of_forest_spacing * spacing
            p_surface = this.p.peierls.surface(this.p.emission)
            peierls = this._signed_rate(
                p_surface, this.p.peierls.stress_fraction * tau_eff,
                T_K, this.p.peierls.nu0_s,
            )
            peierls_velocity = jump * peierls
            velocity = max(float(this.c.mobile_transport_velocity_scale), 0.0) * peierls_velocity

            t_surface = this.p.taylor.surface(this.p.emission)
            taylor_phi = spacing / max(this.c.b_m, 1e-30)
            taylor_single = np.maximum(this._signed_rate(
                t_surface, this.p.taylor.stress_fraction * tau_eff * taylor_phi,
                T_K, this.p.taylor.nu0_s,
            ), 0.0)
            corr_length = this.p.taylor_corr_scale / (2.0 * math.sqrt(max(this.p.taylor_corr_rho_c_m2, 1e-300)))
            order = 1.0 + 2.0 * corr_length * np.sqrt(forest)
            taylor = taylor_single / np.maximum(order, 1.0)
            mfp = this.c.mean_free_path_coefficient / np.sqrt(forest)
            encounter = max(float(this.c.encounter_efficiency), 0.0) * np.abs(peierls_velocity) / np.maximum(mfp, 1e-30)

            rho_back, tau_back, sigma_back = this.backstress_state()
            drive_factors = np.asarray(this.emission_drive_factors(), float)
            # Convert each resolved shear to the native emission surface's
            # normal-stress coordinate. This is an identity in the virgin
            # reference state and does not share a cleavage-derived scalar.
            denom = np.maximum(np.abs(drive_factors), 1e-12)
            drive = np.abs(tau_external_by_system) / denom
            sigma_emit = np.maximum(drive - sigma_back, 0.0)
            per_site_rate = this._arrhenius_rate(
                this.p.emission.barrier_eV(sigma_emit, T_K), T_K,
                this.c.emission_nu0_s,
            )
            emission = np.zeros((this.c.n_systems, 2, this.c.n_bins))
            nsrc = this._source_zone_bin_count()
            for system, sign in enumerate(this.c.emission_signs):
                emission[system, 1 if sign > 0 else 0, :nsrc] = per_site_rate[system]
            tau_external_full = np.broadcast_to(tau_external, tau_gnd.shape).copy()
            return {
                "velocity_m_s": velocity, "peierls_velocity_m_s": peierls_velocity,
                "taylor_completion_s": taylor, "encounter_s": encounter,
                "emission_rate_s": emission, "recovery_rate_s": np.asarray(0.0),
                "tau_external_Pa": tau_external_full,
                "tau_nonlocal_shielding_Pa": np.zeros_like(tau_external_full),
                "tau_gnd_Pa": tau_gnd, "tau_eff_Pa": tau_external_full + tau_gnd,
                "emission_drive_Pa": drive, "emission_effective_stress_Pa": sigma_emit,
                "emission_rate_per_site_s": np.asarray(per_site_rate),
                "rho_back_by_system_m2": rho_back, "tau_back_by_system_Pa": tau_back,
                "sigma_back_by_system_Pa": sigma_back,
                "temperature_K": np.asarray(float(T_K)), "tip_radius_m": np.asarray(float(this.c.r0_m)),
            }

        def cleavage_rate(this, K_MPa_sqrt_m: float, T_K: float, *, neutral: bool = False):
            if neutral:
                return this.__class__.cleavage_rate_s(this, K_MPa_sqrt_m, T_K, neutral=True)
            opening = float(adapter.mechanics(K_MPa_sqrt_m)["opening_Pa"]["activation_8_16"])
            opening_ref = float(adapter.reference["opening_Pa_at_reference_K"])
            singular_per_K = 1.0e6 / math.sqrt(2.0 * math.pi * max(float(this.c.r0_m), float(this.c.b_m), 1e-30))
            sigma = max(opening / opening_ref * adapter.projection.reference_K * singular_per_K, 0.0)
            raw = float(this._arrhenius_rate(this.p.cleavage.barrier_eV(sigma, T_K), T_K, this.c.cleavage_nu0_s))
            if this.c.cleavage_hits <= 1.0 + 1e-12:
                return raw
            from scipy.special import gammainc
            exposure = min(raw * this.c.cleavage_correlation_time_s, 1e12)
            return float(gammainc(this.c.cleavage_hits, exposure) / this.c.cleavage_correlation_time_s)

        def adaptive_substep(this, rates, remaining_s):
            remaining = max(float(remaining_s), 0.0)
            if remaining <= 0.0: return 0.0
            # The stiff state advances transport/exchange implicitly. Rates
            # are nevertheless required and checked here; accuracy is bounded
            # by the coupled full/half estimator, not an explicit CFL formula.
            for key in ("velocity_m_s", "taylor_completion_s", "encounter_s", "emission_rate_s"):
                if key not in rates or not np.all(np.isfinite(rates[key])):
                    raise RuntimeError(f"missing or non-finite current rate field: {key}")
            feedback_cap = float(getattr(this, "max_feedback_substep_s", remaining))
            return max(min(remaining, feedback_cap), float(this.c.min_substep_s))

        state.local_rates = MethodType(local_rates, state)
        state.cleavage_rate_s = MethodType(cleavage_rate, state)
        state._substep = MethodType(adaptive_substep, state)
        state.tip_radius_m = MethodType(lambda this: float(this.c.r0_m), state)
        state.K_shield_MPa_sqrt_m = MethodType(lambda this: 0.0, state)


__all__ = ["MODEL_ID", "MechanismSpecificFeedback"]
