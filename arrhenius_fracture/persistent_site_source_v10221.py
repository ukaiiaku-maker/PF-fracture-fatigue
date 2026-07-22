"""Persistent crack-tip nucleation sites with backstress and blunting feedback.

This v10.2.21 overlay removes the finite source-inventory law from the signed
v10.2.18 production stack.  ``rho_site0_m2`` defines a persistent areal density
of nucleation sites.  The aggregate Arrhenius hazard for each reduced slip
system is

    Lambda_s = M_s * lambda_emit(sigma_drive,s - sigma_back,s, T),

where the instantaneous multiplicity is

    M_s = rho_site0_m2 * c_arc * r_tip * w_eff.

The effective along-front width is a state-dependent correlation/mean-free-path
length anchored to the v9.12 calibration width at the forest-density floor and
reduced by the evolving unsigned near-tip dislocation density.  Nucleation sites
are never consumed.  Emitted signed line content changes the Taylor backstress,
shielding, transport, storage, and crack-tip blunting.  Moving-frame crack
advance convects accumulated slip behind the tip and therefore permits natural
resharpening.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math
from types import MethodType
from typing import Any, Callable

import numpy as np

from .anisotropic_emission_v10174 import _drive_factors_for_state
from .campaign_calibrated_tip import _campaign_backstress, _campaign_local_density_m2
from .signed_burgers_shared_v1025 import _signed_advance, _sync_active
from .state_resolved_signed_engine_v10214 import StateResolvedSignedBurgersTipEngine


MODEL_ID = "v10.2.21_persistent_site_backstress_blunting"
SOURCE_MODEL = "persistent_areal_sites_backstress_limited_no_inventory"


@dataclass(frozen=True)
class PersistentSiteConfig:
    """Physical and numerical controls for the persistent-site closure."""

    rho_site0_m2: float
    reference_source_area_m2: float = 25.0e-12
    reference_front_width_m: float = 10.0e-6
    reference_density_m2: float = 5.0e12
    source_zone_length_m: float = 2.0e-6
    minimum_front_width_m: float = 0.0
    maximum_front_width_m: float = 0.0
    implicit_tolerance: float = 1.0e-10
    implicit_max_iterations: int = 96

    def validate(self) -> "PersistentSiteConfig":
        if not math.isfinite(self.rho_site0_m2) or self.rho_site0_m2 <= 0.0:
            raise ValueError("rho_site0_m2 must be positive and finite")
        if self.reference_source_area_m2 <= 0.0:
            raise ValueError("reference_source_area_m2 must be positive")
        if self.reference_front_width_m <= 0.0:
            raise ValueError("reference_front_width_m must be positive")
        if self.reference_density_m2 <= 0.0:
            raise ValueError("reference_density_m2 must be positive")
        if self.source_zone_length_m <= 0.0:
            raise ValueError("source_zone_length_m must be positive")
        if self.minimum_front_width_m < 0.0:
            raise ValueError("minimum_front_width_m must be nonnegative")
        if self.maximum_front_width_m < 0.0:
            raise ValueError("maximum_front_width_m must be nonnegative")
        if self.implicit_tolerance <= 0.0:
            raise ValueError("implicit_tolerance must be positive")
        if self.implicit_max_iterations < 8:
            raise ValueError("implicit_max_iterations must be at least 8")
        return self


def effective_front_width_m(
    rho_unsigned_m2: float,
    *,
    reference_width_m: float,
    reference_density_m2: float,
    minimum_width_m: float,
    maximum_width_m: float,
) -> float:
    """Return a density-limited along-front correlation width.

    The law is anchored so that ``w_eff = reference_width_m`` at
    ``rho_unsigned_m2 = reference_density_m2`` and scales as rho**(-1/2).
    """
    rho = max(float(rho_unsigned_m2), float(reference_density_m2), 1.0)
    width = float(reference_width_m) * math.sqrt(float(reference_density_m2) / rho)
    lower = max(float(minimum_width_m), 1.0e-30)
    upper = max(float(maximum_width_m), lower)
    return min(max(width, lower), upper)


def persistent_site_multiplicity(
    rho_site0_m2: float,
    tip_radius_m: float,
    front_width_m: float,
    active_arc_factor: float,
) -> float:
    """Number of persistent statistically independent sites per reduced system."""
    area = (
        max(float(active_arc_factor), 0.0)
        * max(float(tip_radius_m), 0.0)
        * max(float(front_width_m), 0.0)
    )
    return max(float(rho_site0_m2), 0.0) * area


def solve_backstress_limited_activations(
    *,
    multiplicity: float,
    dt_s: float,
    drive_stress_Pa: float,
    rho_initial_m2: float,
    rho_increment_per_activation_m2: float,
    backstress_prefactor_Pa_sqrt_m2: float,
    rate_function: Callable[[float], float],
    tolerance: float = 1.0e-10,
    max_iterations: int = 96,
) -> float:
    """Implicit mean activation count for persistent sites.

    This is a backward-Euler integration of

        dN/dt = M lambda(max(sigma_drive - k*sqrt(rho0 + a*N), 0)).

    The mechanical gate sets the rate exactly to zero when the evolved
    backstress reaches the resolved drive.  The root is therefore bounded by the
    line content required to reach that blocking state, without a finite source
    reservoir or an arbitrary emission cap.
    """
    M = max(float(multiplicity), 0.0)
    dt = max(float(dt_s), 0.0)
    drive = max(float(drive_stress_Pa), 0.0)
    rho0 = max(float(rho_initial_m2), 0.0)
    rho_per = max(float(rho_increment_per_activation_m2), 0.0)
    kback = max(float(backstress_prefactor_Pa_sqrt_m2), 0.0)
    if M <= 0.0 or dt <= 0.0 or drive <= 0.0:
        return 0.0
    if rho_per <= 0.0 or kback <= 0.0:
        raise RuntimeError("persistent-site emission requires positive backstress coupling")

    back0 = kback * math.sqrt(rho0)
    sigma0 = drive - back0
    if sigma0 <= 0.0:
        return 0.0
    rate0 = max(float(rate_function(sigma0)), 0.0)
    if not math.isfinite(rate0) or rate0 <= 0.0:
        return 0.0

    rho_block = (drive / kback) ** 2
    upper = max((rho_block - rho0) / rho_per, 0.0)
    if upper <= 0.0:
        return 0.0

    def residual(value: float) -> float:
        rho = rho0 + rho_per * max(value, 0.0)
        sigma_eff = drive - kback * math.sqrt(max(rho, 0.0))
        rate = 0.0 if sigma_eff <= 0.0 else max(float(rate_function(sigma_eff)), 0.0)
        if not math.isfinite(rate):
            rate = 0.0
        return value - M * rate * dt

    lo = 0.0
    hi = upper
    if residual(hi) < 0.0:
        raise RuntimeError("failed to bracket persistent-site backstress root")
    scale = max(upper, 1.0)
    for _ in range(int(max_iterations)):
        mid = 0.5 * (lo + hi)
        value = residual(mid)
        if abs(value) <= float(tolerance) * scale or (hi - lo) <= float(tolerance) * scale:
            return max(mid, 0.0)
        if value > 0.0:
            hi = mid
        else:
            lo = mid
    return max(0.5 * (lo + hi), 0.0)


def _source_zone_bin_count(state) -> int:
    length = max(float(state._persistent_site_cfg.source_zone_length_m), float(state.dx))
    return max(min(int(math.ceil(length / float(state.dx))), int(state.n_bins)), 1)


def _source_geometry(state) -> dict[str, float]:
    cfg = state._persistent_site_cfg
    rho_by_system = _campaign_local_density_m2(state)
    rho_width = max(
        float(state.cfg.forest_density_floor_m2) + float(np.sum(rho_by_system)),
        cfg.reference_density_m2,
    )
    minimum = max(float(cfg.minimum_front_width_m), float(state.dx), abs(float(state._persistent_b)))
    maximum = (
        float(cfg.maximum_front_width_m)
        if cfg.maximum_front_width_m > 0.0
        else float(state.length_m)
    )
    width = effective_front_width_m(
        rho_width,
        reference_width_m=cfg.reference_front_width_m,
        reference_density_m2=cfg.reference_density_m2,
        minimum_width_m=minimum,
        maximum_width_m=maximum,
    )
    radius = float(state.blunted_radius(state._persistent_r0_m, state._persistent_b))
    multiplicity = persistent_site_multiplicity(
        cfg.rho_site0_m2,
        radius,
        width,
        state._persistent_active_arc_factor,
    )
    return {
        "tip_radius_m": radius,
        "front_width_m": width,
        "rho_width_m2": rho_width,
        "source_area_m2": state._persistent_active_arc_factor * radius * width,
        "multiplicity_per_system": multiplicity,
    }


def _density_increment_per_activation(state, nsrc: int, line_per_activation: float) -> float:
    length = max(
        float(state.cfg.blunting_length_m),
        float(state.dx),
        abs(float(state._persistent_b)),
        1.0e-12,
    )
    weights = np.exp(-np.asarray(state.x, dtype=float) / length)
    norm = max(float(np.sum(weights)), 1.0e-30)
    source_weight = float(np.sum(weights[:nsrc])) / max(float(nsrc), 1.0)
    strip_width = max(float(state.cfg.blunting_length_m), float(state.dx), 1.0e-12)
    return (
        max(float(line_per_activation), 0.0)
        * source_weight
        / max(norm * float(state.dx) * strip_width, 1.0e-30)
    )


def _persistent_emit(
    self,
    dt: float,
    stress_Pa: float,
    T_K: float,
    system_weights: np.ndarray | None = None,
) -> float:
    dt = max(float(dt), 0.0)
    if dt <= 0.0:
        self.signed_last_source_activations = 0.0
        self.signed_last_line_content = 0.0
        return 0.0
    if system_weights is not None:
        supplied = np.asarray(system_weights, dtype=float)
        if supplied.size and not np.allclose(supplied, 1.0):
            raise RuntimeError("post-hazard system_weights are forbidden")
    if not bool(getattr(self, "_anisotropic_drive_reliable", False)):
        raise RuntimeError("persistent signed emission requires a reliable 2-D tensor drive")

    factors = _drive_factors_for_state(self)
    tau_signed = np.asarray(
        getattr(self, "_anisotropic_tau_signed_Pa", np.zeros(self.n_systems)),
        dtype=float,
    ).reshape(-1)
    if tau_signed.size < self.n_systems:
        tau_signed = np.pad(tau_signed, (0, self.n_systems - tau_signed.size))
    signs = np.sign(tau_signed[: self.n_systems])

    rho0, tau_back0, sigma_back0 = _campaign_backstress(self)
    opening = max(float(stress_Pa), 0.0)
    drive = np.maximum(np.asarray(factors, dtype=float) * opening, 0.0)
    geometry = _source_geometry(self)
    multiplicity = float(geometry["multiplicity_per_system"])
    nsrc = _source_zone_bin_count(self)
    conversion = np.asarray(self._signed_kernel.activation_to_line_content, dtype=float)
    if conversion.shape != (self.n_systems,) or np.any(conversion <= 0.0):
        raise RuntimeError("persistent-site source requires positive line conversion per system")

    G = max(float(getattr(self, "_campaign_G_Pa", 0.0)), 0.0)
    b = abs(float(getattr(self, "_campaign_b", self._persistent_b)))
    scale = max(float(getattr(self, "_campaign_backstress_scale", 1.0)), 0.0)
    resolved = max(abs(float(self.cfg.taylor_stress_fraction)), 1.0e-6)
    backstress_prefactor = scale * G * b / resolved
    if backstress_prefactor <= 0.0:
        raise RuntimeError("persistent-site source requires a positive Taylor backstress")

    activations = np.zeros(self.n_systems, dtype=float)
    line_by_system = np.zeros(self.n_systems, dtype=float)
    rates_initial = np.zeros(self.n_systems, dtype=float)
    rates_final = np.zeros(self.n_systems, dtype=float)
    sigma_final = np.zeros(self.n_systems, dtype=float)
    cfg = self._persistent_site_cfg

    for system in range(self.n_systems):
        if signs[system] == 0.0 or drive[system] <= sigma_back0[system]:
            continue
        density_per_activation = _density_increment_per_activation(
            self, nsrc, float(conversion[system])
        )

        def rate_at(stress: float) -> float:
            if stress <= 0.0:
                return 0.0
            return float(self.emission_rate_per_site(float(stress), T_K))

        sigma_initial = max(float(drive[system] - sigma_back0[system]), 0.0)
        rates_initial[system] = rate_at(sigma_initial)
        activations[system] = solve_backstress_limited_activations(
            multiplicity=multiplicity,
            dt_s=dt,
            drive_stress_Pa=float(drive[system]),
            rho_initial_m2=float(rho0[system]),
            rho_increment_per_activation_m2=density_per_activation,
            backstress_prefactor_Pa_sqrt_m2=backstress_prefactor,
            rate_function=rate_at,
            tolerance=cfg.implicit_tolerance,
            max_iterations=cfg.implicit_max_iterations,
        )
        line_by_system[system] = activations[system] * conversion[system]
        rho_final = float(rho0[system]) + density_per_activation * activations[system]
        sigma_final[system] = max(
            float(drive[system]) - backstress_prefactor * math.sqrt(max(rho_final, 0.0)),
            0.0,
        )
        rates_final[system] = rate_at(sigma_final[system])

    for system in range(self.n_systems):
        amount = float(line_by_system[system]) / float(nsrc)
        if signs[system] > 0.0:
            self.mobile_positive[system, :nsrc] += amount
            self.accumulated_slip_positive[system, :nsrc] += amount
        elif signs[system] < 0.0:
            self.mobile_negative[system, :nsrc] += amount
            self.accumulated_slip_negative[system, :nsrc] += amount
    _sync_active(self)

    emitted_lines = float(np.sum(line_by_system))
    source_activations = float(np.sum(activations))
    self.emitted_total += emitted_lines
    self.signed_source_activations_total += source_activations
    self.signed_line_content_emitted_total += emitted_lines
    self.signed_last_source_activations = source_activations
    self.signed_last_line_content = emitted_lines
    self.signed_last_line_content_by_system = line_by_system.copy()
    self.signed_last_burgers_sign_by_system = signs.copy()

    # Legacy capacity arrays remain full and are diagnostic compatibility fields
    # only.  They do not multiply, cap, or otherwise alter the emission hazard.
    self.available_sites = np.asarray(self.site_capacity, dtype=float).copy()
    self.tip_source_activity = np.ones(self.n_systems, dtype=float)
    self.continuum_source_last_clear_rate_s = 0.0
    self.continuum_source_last_effective_multiplicity = float(
        self.n_systems * multiplicity
    )
    self.continuum_source_last_emission_rate_s = float(
        np.sum(multiplicity * rates_initial)
    )
    self.continuum_source_last_aggregate_hazard_s = self.continuum_source_last_emission_rate_s
    self.continuum_source_last_throughput_bound = math.inf
    self.continuum_source_last_rho_back_m2 = float(np.mean(rho0))
    self.continuum_source_last_tau_back_Pa = float(np.mean(tau_back0))
    self.continuum_source_last_sigma_back_Pa = float(np.mean(sigma_back0))
    self.continuum_source_last_sigma_emit_effective_Pa = float(np.mean(sigma_final))
    self.continuum_source_last_sigma_emit_effective_min_Pa = float(np.min(sigma_final))
    self.persistent_site_last_geometry = dict(geometry)
    self.persistent_site_last_drive_Pa = drive.copy()
    self.persistent_site_last_sigma_back_initial_Pa = sigma_back0.copy()
    self.persistent_site_last_sigma_effective_final_Pa = sigma_final.copy()
    self.persistent_site_last_rate_initial_s = rates_initial.copy()
    self.persistent_site_last_rate_final_s = rates_final.copy()
    self.persistent_site_last_activations = activations.copy()
    self.persistent_site_last_line_content = line_by_system.copy()

    self.anisotropic_last_drive_factors = np.asarray(factors, dtype=float).copy()
    self.anisotropic_last_sigma_opening_Pa = opening
    self.anisotropic_last_rho_back_by_system_m2 = rho0.copy()
    self.anisotropic_last_tau_back_by_system_Pa = tau_back0.copy()
    self.anisotropic_last_sigma_back_by_system_Pa = sigma_back0.copy()
    self.anisotropic_last_sigma_emit_by_system_Pa = sigma_final.copy()
    self.anisotropic_last_lambda_emit_by_system_s = rates_final.copy()
    self.anisotropic_last_probability_by_system = 1.0 - np.exp(
        -np.minimum(np.maximum(rates_initial * dt, 0.0), 700.0)
    )
    self.anisotropic_last_dN_emit_by_system = line_by_system.copy()
    return emitted_lines


def _persistent_advance(self, distance_m: float) -> dict[str, float]:
    # Keep the legacy availability arrays full so the inherited moving-frame
    # operation computes zero source refresh.  Accumulated slip is still convected
    # into the wake, which changes the locally evaluated blunted radius.
    self.available_sites = np.asarray(self.site_capacity, dtype=float).copy()
    self.tip_source_activity = np.ones(self.n_systems, dtype=float)
    radius_before = float(self.blunted_radius(self._persistent_r0_m, self._persistent_b))
    result = _signed_advance(self, distance_m)
    self.available_sites = np.asarray(self.site_capacity, dtype=float).copy()
    self.tip_source_activity = np.ones(self.n_systems, dtype=float)
    radius_after = float(self.blunted_radius(self._persistent_r0_m, self._persistent_b))
    geometry = _source_geometry(self)
    self.persistent_site_last_geometry = dict(geometry)
    result.update(
        {
            "source_sites_refreshed": 0.0,
            "persistent_source_inventory_active": 0.0,
            "persistent_site_multiplicity_per_system": float(
                geometry["multiplicity_per_system"]
            ),
            "persistent_site_front_width_m": float(geometry["front_width_m"]),
            "persistent_site_source_area_m2": float(geometry["source_area_m2"]),
            "tip_radius_before_advance_m": radius_before,
            "tip_radius_after_advance_m": radius_after,
            "tip_resharpening_by_advance_m": max(radius_before - radius_after, 0.0),
        }
    )
    return result


def _persistent_diagnostics(self, G=None, nu=None, b=None, r0=None) -> dict[str, Any]:
    data = dict(self._persistent_base_diagnostics(G, nu, b, r0))
    geometry = _source_geometry(self)
    rho, tau, sigma = _campaign_backstress(self)
    drive = np.asarray(
        getattr(self, "persistent_site_last_drive_Pa", np.zeros(self.n_systems)),
        dtype=float,
    )
    ratio = np.divide(
        sigma,
        np.maximum(drive, 1.0),
        out=np.zeros_like(sigma),
        where=drive > 0.0,
    )
    data.update(
        {
            "persistent_site_source_model": SOURCE_MODEL,
            "persistent_source_inventory_active": False,
            "persistent_source_refresh_active": False,
            "persistent_site_density_m2": float(self._persistent_site_cfg.rho_site0_m2),
            "persistent_site_multiplicity_per_system": float(
                geometry["multiplicity_per_system"]
            ),
            "persistent_site_source_area_m2": float(geometry["source_area_m2"]),
            "persistent_site_front_width_m": float(geometry["front_width_m"]),
            "persistent_site_width_density_m2": float(geometry["rho_width_m2"]),
            "persistent_tip_radius_m": float(geometry["tip_radius_m"]),
            "persistent_source_zone_bins": int(_source_zone_bin_count(self)),
            "persistent_rho_back_mean_m2": float(np.mean(rho)),
            "persistent_sigma_back_mean_Pa": float(np.mean(sigma)),
            "persistent_backstress_drive_ratio_max": float(np.max(ratio)),
            "legacy_source_sites_per_system_active": False,
            "legacy_source_refresh_length_active": False,
        }
    )
    return data


def install_persistent_site_source(
    state,
    *,
    config: PersistentSiteConfig,
    r0_m: float,
    b_m: float,
) -> None:
    cfg = copy.deepcopy(config).validate()
    if not hasattr(state, "mobile_positive") or not hasattr(state, "retained_positive"):
        raise RuntimeError("persistent-site source must be installed after signed Burgers state")
    if abs(float(state.manifest.retained_recovery_rate_s)) > 1.0e-30:
        raise RuntimeError("v10.2.21 top-1 transfer requires retained recovery disabled")
    if abs(float(state.cfg.mobile_recovery_rate_s)) > 1.0e-30:
        raise RuntimeError("v10.2.21 top-1 transfer requires mobile recovery disabled")
    r0 = max(float(r0_m), abs(float(b_m)), 1.0e-12)
    arc_factor = cfg.reference_source_area_m2 / (
        r0 * cfg.reference_front_width_m
    )
    if not math.isfinite(arc_factor) or arc_factor <= 0.0:
        raise RuntimeError("invalid active tip-arc factor")

    state.source_model = SOURCE_MODEL
    state._persistent_site_cfg = cfg
    state._persistent_r0_m = r0
    state._persistent_b = float(b_m)
    state._persistent_active_arc_factor = arc_factor
    state._persistent_base_diagnostics = state.diagnostics
    state.available_sites = np.asarray(state.site_capacity, dtype=float).copy()
    state.tip_source_activity = np.ones(state.n_systems, dtype=float)
    state.persistent_site_last_geometry = _source_geometry(state)
    state.persistent_site_last_drive_Pa = np.zeros(state.n_systems)
    state.persistent_site_last_sigma_back_initial_Pa = np.zeros(state.n_systems)
    state.persistent_site_last_sigma_effective_final_Pa = np.zeros(state.n_systems)
    state.persistent_site_last_rate_initial_s = np.zeros(state.n_systems)
    state.persistent_site_last_rate_final_s = np.zeros(state.n_systems)
    state.persistent_site_last_activations = np.zeros(state.n_systems)
    state.persistent_site_last_line_content = np.zeros(state.n_systems)
    state._emit = MethodType(_persistent_emit, state)
    state.advance = MethodType(_persistent_advance, state)
    state.diagnostics = MethodType(_persistent_diagnostics, state)


class PersistentSiteStateResolvedTipEngine(StateResolvedSignedBurgersTipEngine):
    """Final signed 2-D engine with persistent, backstress-limited source sites."""

    persistent_site_source_active = True
    _persistent_site_config_default: PersistentSiteConfig | None = None

    @classmethod
    def configure_persistent_sites(cls, config: PersistentSiteConfig) -> None:
        cls._persistent_site_config_default = copy.deepcopy(config).validate()

    @classmethod
    def audit_payload(cls) -> dict[str, Any]:
        payload = super().audit_payload()
        cfg = cls._persistent_site_config_default
        payload["persistent_site_source_v10221"] = {
            "model_id": MODEL_ID,
            "source_model": SOURCE_MODEL,
            "finite_source_inventory": False,
            "source_depletion_on_emission": False,
            "source_refresh_on_crack_advance": False,
            "site_multiplicity_in_arrhenius_hazard": True,
            "multiplicity_geometry": "rho_site0*c_arc*r_tip*w_eff",
            "tip_radius_state": "r0+c_blunt*b*local_accumulated_slip",
            "resharpening": "moving_frame_convection_of_accumulated_slip",
            "front_width_state": "reference_width*sqrt(reference_density/rho_unsigned)",
            "backstress_population": "unsigned_mobile_plus_retained_line_content",
            "shielding_population": "signed_retained_line_content",
            "mechanical_zero_drive_gate": True,
            "emission_integrator": "implicit_backward_euler_backstress_root",
            "config": None if cfg is None else asdict(cfg),
        }
        return payload

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = type(self)._persistent_site_config_default
        if cfg is None:
            raise RuntimeError("persistent-site configuration was not installed")
        install_persistent_site_source(
            self.mpz,
            config=cfg,
            r0_m=float(self.f.r0),
            b_m=float(self.b),
        )


__all__ = [
    "MODEL_ID",
    "SOURCE_MODEL",
    "PersistentSiteConfig",
    "PersistentSiteStateResolvedTipEngine",
    "effective_front_width_m",
    "persistent_site_multiplicity",
    "solve_backstress_limited_activations",
    "install_persistent_site_source",
]
