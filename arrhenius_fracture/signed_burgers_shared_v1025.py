"""Shared signed-Burgers population and 2-D-derived shielding closure.

The module is deliberately loading-path agnostic. Both monotonic fracture and
cyclic fatigue install this state model on the same anisotropic tip engine.
Source-site capacity remains a nucleation-opportunity measure. Emitted state is
converted to physical signed line content by a non-fitted conversion supplied in
a mechanically derived kernel artifact.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np

from .anisotropic_emission_v10174 import (
    AnisotropicStochasticAvalancheTipEngine,
    _drive_factors_for_state,
    finite_source_emission_update,
)
from .campaign_calibrated_tip import _campaign_backstress
from .fractional_moving_frame import _translate_toward_tip

MODEL_ID = "v10.2.5_shared_signed_burgers_population"
KERNEL_SCHEMA = "v10.2.5_2d_unit_signed_shielding_kernel"
VALIDATED_SCALAR_TRANSPORT = "validated_scalar"
CHANNEL_RESOLVED_TRANSPORT = "channel_resolved"


@dataclass(frozen=True)
class SignedShieldingKernel:
    """Signed unit-response operator and source normalization.

    ``active_kernel[s, i]`` and ``wake_kernel[s, i]`` are the change in mode-I
    crack-tip stress intensity caused by one positive signed line unit. Negative
    Burgers content reverses the interaction automatically.
    """

    active_kernel: np.ndarray
    wake_kernel: np.ndarray
    active_x_m: np.ndarray
    wake_x_m: np.ndarray
    activation_to_line_content: np.ndarray
    source_capacity_bounds: np.ndarray
    metadata: dict[str, Any]
    source_path: str

    @classmethod
    def from_json(cls, path: str | Path) -> "SignedShieldingKernel":
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text())
        if payload.get("schema") != KERNEL_SCHEMA:
            raise ValueError(
                f"signed shielding kernel schema must be {KERNEL_SCHEMA!r}; "
                f"got {payload.get('schema')!r}"
            )
        required_truth = {
            "candidate_independent": True,
            "counts_are_signed_burgers_lines": True,
            "kernel_from_2d_unit_signed_perturbations": True,
            "normalization_is_mechanically_derived": True,
            "fitted_attenuation_factor": False,
        }
        for key, expected in required_truth.items():
            if payload.get(key) is not expected:
                raise ValueError(f"kernel metadata requires {key}={expected}")
        kernel_source = str(payload.get("kernel_source", ""))
        if kernel_source not in {
            "2d_unit_signed_dislocation_perturbation",
            "2d_unit_signed_slip_perturbation",
        }:
            raise ValueError("kernel_source must identify a 2-D unit signed perturbation")
        normalization_source = str(payload.get("normalization_source", ""))
        if normalization_source not in {
            "2d_unit_slip_to_line_content",
            "process_zone_geometry_and_line_spacing",
            "front_thickness_source_geometry",
        }:
            raise ValueError("normalization_source is not an accepted mechanical derivation")

        active = np.asarray(
            payload["active_kernel_Pa_sqrt_m_per_signed_line"], dtype=float
        )
        wake = np.asarray(
            payload.get("wake_kernel_Pa_sqrt_m_per_signed_line", []), dtype=float
        )
        active_x = np.asarray(payload["active_x_m"], dtype=float)
        wake_x = np.asarray(payload.get("wake_x_m", []), dtype=float)
        conversion = np.asarray(
            payload["activation_to_line_content_by_system"], dtype=float
        ).reshape(-1)
        bounds = np.asarray(
            payload["source_capacity_bounds_per_system"], dtype=float
        )
        if active.ndim != 2 or not np.all(np.isfinite(active)):
            raise ValueError("active shielding kernel must be a finite 2-D array")
        if wake.size == 0:
            wake = np.zeros((active.shape[0], 0), dtype=float)
        if wake.ndim != 2 or not np.all(np.isfinite(wake)):
            raise ValueError("wake shielding kernel must be a finite 2-D array")
        if conversion.shape != (active.shape[0],) or np.any(conversion <= 0.0):
            raise ValueError("one positive activation-to-line conversion is required per system")
        if bounds.shape != (active.shape[0], 2):
            raise ValueError("source-capacity bounds must have shape (n_systems, 2)")
        if np.any(~np.isfinite(bounds)) or np.any(bounds[:, 0] < 0.0) or np.any(
            bounds[:, 1] < bounds[:, 0]
        ):
            raise ValueError("invalid physical source-capacity bounds")
        if active_x.shape != (active.shape[1],) or np.any(~np.isfinite(active_x)):
            raise ValueError("active_x_m must match active-kernel bins")
        if wake_x.shape != (wake.shape[1],) or np.any(~np.isfinite(wake_x)):
            raise ValueError("wake_x_m must match wake-kernel bins")
        metadata = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "active_kernel_Pa_sqrt_m_per_signed_line",
                "wake_kernel_Pa_sqrt_m_per_signed_line",
                "active_x_m",
                "wake_x_m",
                "activation_to_line_content_by_system",
                "source_capacity_bounds_per_system",
            }
        }
        return cls(
            active_kernel=active,
            wake_kernel=wake,
            active_x_m=active_x,
            wake_x_m=wake_x,
            activation_to_line_content=conversion,
            source_capacity_bounds=bounds,
            metadata=metadata,
            source_path=str(source),
        )

    def validate_state(self, state) -> None:
        expected_active = (state.n_systems, state.n_bins)
        expected_wake = (state.n_systems, state.wake_n_bins)
        if self.active_kernel.shape != expected_active:
            raise ValueError(
                f"active kernel shape {self.active_kernel.shape} != {expected_active}"
            )
        if self.wake_kernel.shape != expected_wake:
            raise ValueError(
                f"wake kernel shape {self.wake_kernel.shape} != {expected_wake}"
            )
        if not np.allclose(self.active_x_m, state.x, rtol=1.0e-12, atol=1.0e-18):
            raise ValueError("active kernel coordinates do not match the production MPZ grid")
        if not np.allclose(self.wake_x_m, state.wake_x, rtol=1.0e-12, atol=1.0e-18):
            raise ValueError("wake kernel coordinates do not match the production wake grid")
        capacity = np.asarray(state.site_capacity, dtype=float)
        lo = self.source_capacity_bounds[:, 0]
        hi = self.source_capacity_bounds[:, 1]
        if np.any(capacity < lo) or np.any(capacity > hi):
            raise ValueError(
                "manifest source_sites_per_system lies outside the mechanically "
                "derived source-capacity range; rebuild the campaign search region"
            )

    def audit_payload(self) -> dict[str, Any]:
        return {
            "schema": KERNEL_SCHEMA,
            "source_path": self.source_path,
            "active_shape": list(self.active_kernel.shape),
            "wake_shape": list(self.wake_kernel.shape),
            "activation_to_line_content_by_system": self.activation_to_line_content.tolist(),
            "source_capacity_bounds_per_system": self.source_capacity_bounds.tolist(),
            "active_kernel_min_Pa_sqrt_m_per_signed_line": float(np.min(self.active_kernel)),
            "active_kernel_max_Pa_sqrt_m_per_signed_line": float(np.max(self.active_kernel)),
            "wake_kernel_min_Pa_sqrt_m_per_signed_line": (
                float(np.min(self.wake_kernel)) if self.wake_kernel.size else 0.0
            ),
            "wake_kernel_max_Pa_sqrt_m_per_signed_line": (
                float(np.max(self.wake_kernel)) if self.wake_kernel.size else 0.0
            ),
            **copy.deepcopy(self.metadata),
        }


def _sync_active(state) -> None:
    state.mobile = state.mobile_positive + state.mobile_negative
    state.retained = state.retained_positive + state.retained_negative
    state.accumulated_slip = (
        state.accumulated_slip_positive + state.accumulated_slip_negative
    )


def _sync_wake(state) -> None:
    state.wake_mobile = state.wake_mobile_positive + state.wake_mobile_negative
    state.wake_retained = state.wake_retained_positive + state.wake_retained_negative
    state.wake_slip = state.wake_slip_positive + state.wake_slip_negative


def _signed_active_content(state) -> np.ndarray:
    return (
        state.retained_positive
        - state.retained_negative
        + float(state.cfg.mobile_shield_fraction)
        * (state.mobile_positive - state.mobile_negative)
    )


def _signed_wake_content(state) -> np.ndarray:
    return (
        state.wake_retained_positive
        - state.wake_retained_negative
        + float(state.cfg.mobile_shield_fraction)
        * (state.wake_mobile_positive - state.wake_mobile_negative)
    )


def _active_K(self, G=None, nu=None, b=None) -> float:
    return float(np.sum(self._signed_kernel.active_kernel * _signed_active_content(self)))


def _wake_K(self, G=None, nu=None, b=None) -> float:
    if not bool(self.cfg.wake_shielding):
        return 0.0
    return float(np.sum(self._signed_kernel.wake_kernel * _signed_wake_content(self)))


def _total_K(self, G=None, nu=None, b=None) -> float:
    return _active_K(self) + _wake_K(self)


def _exchange_species(state, k_enc, k_rel, dt: float) -> tuple[float, float]:
    mp, rp, tp, lp = state._exchange(
        state.mobile_positive, state.retained_positive, k_enc, k_rel, dt
    )
    mn, rn, tn, ln = state._exchange(
        state.mobile_negative, state.retained_negative, k_enc, k_rel, dt
    )
    state.mobile_positive, state.retained_positive = mp, rp
    state.mobile_negative, state.retained_negative = mn, rn
    _sync_active(state)
    return float(tp + tn), float(lp + ln)


def _signed_emit(self, dt: float, stress_Pa: float, T_K: float, system_weights=None) -> float:
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
        raise RuntimeError(
            "signed Burgers emission requires a reliable signed 2-D tensor drive"
        )

    factors = _drive_factors_for_state(self)
    tau_signed = np.asarray(
        getattr(self, "_anisotropic_tau_signed_Pa", np.zeros(self.n_systems)),
        dtype=float,
    ).reshape(-1)
    if tau_signed.size < self.n_systems:
        tau_signed = np.pad(tau_signed, (0, self.n_systems - tau_signed.size))
    tau_signed = tau_signed[: self.n_systems]
    signs = np.sign(tau_signed)

    rho, tau_back, sigma_back = _campaign_backstress(self)
    sigma_opening = max(float(stress_Pa), 0.0)
    sigma_emit = np.maximum(factors * sigma_opening - sigma_back, 0.0)
    rates = np.asarray(
        [self.emission_rate_per_site(float(value), T_K) for value in sigma_emit],
        dtype=float,
    )
    rates = np.maximum(rates, 0.0)
    rates[signs == 0.0] = 0.0

    available0 = np.maximum(np.asarray(self.available_sites, dtype=float), 0.0)
    activations, probability = finite_source_emission_update(available0, rates, dt)
    self.available_sites = np.maximum(available0 - activations, 0.0)
    line_by_system = activations * self._signed_kernel.activation_to_line_content
    nsrc = max(min(int(self.cfg.source_bin_count), self.n_bins), 1)
    for system in range(self.n_systems):
        amount = float(line_by_system[system]) / nsrc
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

    reference = np.maximum(np.asarray(self.site_capacity, dtype=float), 0.0)
    activity = np.divide(
        self.available_sites,
        reference,
        out=np.zeros_like(self.available_sites),
        where=reference > 0.0,
    )
    self.tip_source_activity = np.clip(activity, 0.0, 1.0)
    self.continuum_source_last_clear_rate_s = 0.0
    self.continuum_source_last_effective_multiplicity = float(np.sum(self.available_sites))
    self.continuum_source_last_emission_rate_s = float(np.sum(rates * available0))
    self.continuum_source_last_aggregate_hazard_s = self.continuum_source_last_emission_rate_s
    self.continuum_source_last_throughput_bound = float(np.sum(available0))
    self.continuum_source_last_rho_back_m2 = float(np.mean(rho))
    self.continuum_source_last_tau_back_Pa = float(np.mean(tau_back))
    self.continuum_source_last_sigma_back_Pa = float(np.mean(sigma_back))
    self.continuum_source_last_sigma_emit_effective_Pa = float(np.mean(sigma_emit))
    self.continuum_source_last_sigma_emit_effective_min_Pa = float(np.min(sigma_emit))
    self.campaign_source_budget_remaining_total = float(np.sum(self.available_sites))
    self.campaign_source_budget_consumed_total = float(np.sum(reference - self.available_sites))
    self.anisotropic_last_drive_factors = factors.copy()
    self.anisotropic_last_sigma_opening_Pa = sigma_opening
    self.anisotropic_last_rho_back_by_system_m2 = rho.copy()
    self.anisotropic_last_tau_back_by_system_Pa = tau_back.copy()
    self.anisotropic_last_sigma_back_by_system_Pa = sigma_back.copy()
    self.anisotropic_last_sigma_emit_by_system_Pa = sigma_emit.copy()
    self.anisotropic_last_lambda_emit_by_system_s = rates.copy()
    self.anisotropic_last_probability_by_system = probability.copy()
    self.anisotropic_last_dN_emit_by_system = line_by_system.copy()
    return emitted_lines


def _recover_active(state, dt: float) -> float:
    fr = 1.0 - math.exp(
        -min(max(state.manifest.retained_recovery_rate_s, 0.0) * dt, 700.0)
    )
    fm = 1.0 - math.exp(
        -min(max(state.cfg.mobile_recovery_rate_s, 0.0) * dt, 700.0)
    )
    recovered = 0.0
    for name, fraction in (
        ("retained_positive", fr),
        ("retained_negative", fr),
        ("mobile_positive", fm),
        ("mobile_negative", fm),
    ):
        field = getattr(state, name)
        removed = field * fraction
        setattr(state, name, field - removed)
        recovered += float(np.sum(removed))
    _sync_active(state)
    return recovered


def _signed_evolve_wake(state, dt: float, T_K: float, b: float) -> dict[str, float]:
    if dt <= 0.0 or state.wake_mobile_count + state.wake_retained_count <= 0.0:
        return {
            "wake_dN_trapped": 0.0,
            "wake_dN_released": 0.0,
            "wake_dN_recovered": 0.0,
            "wake_dN_mobile_transport_loss": 0.0,
        }
    rho = state.local_forest_density_m2(True)
    rates = state._transport_rates(np.zeros(state.wake_n_bins), rho, T_K, b)
    trapped = released = 0.0
    for suffix in ("positive", "negative"):
        mobile, retained, tr, rel = state._exchange(
            getattr(state, f"wake_mobile_{suffix}"),
            getattr(state, f"wake_retained_{suffix}"),
            rates["encounter"],
            rates["taylor"],
            dt,
        )
        setattr(state, f"wake_mobile_{suffix}", mobile)
        setattr(state, f"wake_retained_{suffix}", retained)
        trapped += float(tr)
        released += float(rel)
    fr = 1.0 - math.exp(
        -min(max(state.manifest.retained_recovery_rate_s, 0.0) * dt, 700.0)
    )
    fm = 1.0 - math.exp(
        -min(max(state.cfg.mobile_recovery_rate_s, 0.0) * dt, 700.0)
    )
    recovered = 0.0
    for name, fraction in (
        ("wake_retained_positive", fr),
        ("wake_retained_negative", fr),
        ("wake_mobile_positive", fm),
        ("wake_mobile_negative", fm),
    ):
        field = getattr(state, name)
        removed = field * fraction
        setattr(state, name, field - removed)
        recovered += float(np.sum(removed))
    _sync_wake(state)
    mobile_by_bin = np.sum(np.maximum(state.wake_mobile, 0.0), axis=0)
    velocity = (
        float(np.sum(rates["velocity"] * mobile_by_bin) / np.sum(mobile_by_bin))
        if np.sum(mobile_by_bin) > 0.0
        else 0.0
    )
    lost = 0.0
    for name in ("wake_mobile_positive", "wake_mobile_negative"):
        moved, amount = state._advect_forward(
            getattr(state, name), max(velocity, 0.0) * dt, state.wake_dx
        )
        setattr(state, name, moved)
        lost += float(amount)
    _sync_wake(state)
    state.wake_discarded_mobile_total += lost
    return {
        "wake_dN_trapped": trapped,
        "wake_dN_released": released,
        "wake_dN_recovered": recovered,
        "wake_dN_mobile_transport_loss": lost,
    }


def _signed_evolve_scalar(
    self, dt_s: float, T_K: float, stress_Pa: float, b: float, system_weights=None
) -> dict[str, float]:
    dt = max(float(dt_s), 0.0)
    emitted = self._emit(dt, stress_Pa, T_K, system_weights)
    stress = self.local_stress_profile_Pa(stress_Pa)
    rho = self.local_forest_density_m2(False)
    rates = self._transport_rates(stress, rho, T_K, b)
    trapped, released = _exchange_species(
        self, rates["encounter"], rates["taylor"], dt
    )
    recovered = _recover_active(self, dt)
    mobile_by_bin = np.sum(np.maximum(self.mobile, 0.0), axis=0)
    if np.sum(mobile_by_bin) > 0.0:
        velocity = float(
            np.sum(rates["velocity"] * mobile_by_bin) / np.sum(mobile_by_bin)
        )
    else:
        nsrc = max(min(int(self.cfg.source_bin_count), self.n_bins), 1)
        velocity = float(np.mean(rates["velocity"][:nsrc]))
    escaped = 0.0
    for name in ("mobile_positive", "mobile_negative"):
        moved, amount = self._advect_forward(
            getattr(self, name), max(velocity, 0.0) * dt, self.dx
        )
        setattr(self, name, moved)
        escaped += float(amount)
    _sync_active(self)
    self.escaped_total += escaped
    self.recovered_total += recovered
    self.time_s += dt
    wake = _signed_evolve_wake(self, dt, T_K, b)
    return {
        "dN_emit": emitted,
        "dN_source_activations": float(self.signed_last_source_activations),
        "dN_trapped": trapped,
        "dN_released": released,
        "dN_recovered": recovered,
        "dN_escaped": escaped,
        "peierls_rate_s": float(np.max(rates["peierls"])),
        "taylor_completion_rate_s": float(np.max(rates["taylor"])),
        "encounter_rate_s": float(np.max(rates["encounter"])),
        "taylor_m_eff": float(np.max(rates["m"])),
        "available_site_fraction": self.available_site_fraction,
        "signed_burgers_transport": 1.0,
        **wake,
    }


def _signed_evolve_channel(
    self, dt_s: float, T_K: float, stress_Pa: float, b: float, system_weights=None
) -> dict[str, float]:
    dt = max(float(dt_s), 0.0)
    emitted = self._emit(dt, stress_Pa, T_K, system_weights)
    sigma_system = np.asarray(
        getattr(
            self,
            "anisotropic_last_sigma_emit_by_system_Pa",
            np.full(self.n_systems, max(float(stress_Pa), 0.0)),
        ),
        dtype=float,
    )
    if bool(getattr(self, "_anisotropic_shared_forest_density", True)):
        rho_shared = self.local_forest_density_m2(False)
        rho_by_system = [rho_shared for _ in range(self.n_systems)]
    else:
        width = max(float(self.cfg.blunting_length_m), float(self.dx), 1.0e-12)
        rho_by_system = [
            np.maximum(
                float(self.cfg.forest_density_floor_m2)
                + np.maximum(self.retained[system], 0.0)
                / max(self.dx * width, 1.0e-30),
                1.0,
            )
            for system in range(self.n_systems)
        ]
    trapped = released = escaped = 0.0
    peierls_max = taylor_max = encounter_max = 0.0
    m_max = 1.0
    rates_by_system = []
    for system in range(self.n_systems):
        profile = self.local_stress_profile_Pa(float(sigma_system[system]))
        rates = self._transport_rates(profile, rho_by_system[system], T_K, b)
        rates_by_system.append(rates)
        for suffix in ("positive", "negative"):
            mobile_name = f"mobile_{suffix}"
            retained_name = f"retained_{suffix}"
            mobile, retained, tr, rel = self._exchange(
                getattr(self, mobile_name)[system : system + 1],
                getattr(self, retained_name)[system : system + 1],
                rates["encounter"],
                rates["taylor"],
                dt,
            )
            getattr(self, mobile_name)[system : system + 1] = mobile
            getattr(self, retained_name)[system : system + 1] = retained
            trapped += float(tr)
            released += float(rel)
        peierls_max = max(peierls_max, float(np.max(rates["peierls"])))
        taylor_max = max(taylor_max, float(np.max(rates["taylor"])))
        encounter_max = max(encounter_max, float(np.max(rates["encounter"])))
        m_max = max(m_max, float(np.max(rates["m"])))
    _sync_active(self)
    recovered = _recover_active(self, dt)
    velocities = np.zeros(self.n_systems, dtype=float)
    for system, rates in enumerate(rates_by_system):
        mobile_by_bin = np.maximum(self.mobile[system], 0.0)
        if np.sum(mobile_by_bin) > 0.0:
            velocity = float(
                np.sum(rates["velocity"] * mobile_by_bin)
                / np.sum(mobile_by_bin)
            )
        else:
            nsrc = max(min(int(self.cfg.source_bin_count), self.n_bins), 1)
            velocity = float(np.mean(rates["velocity"][:nsrc]))
        velocities[system] = max(velocity, 0.0)
        for suffix in ("positive", "negative"):
            name = f"mobile_{suffix}"
            moved, amount = self._advect_forward(
                getattr(self, name)[system : system + 1],
                velocities[system] * dt,
                self.dx,
            )
            getattr(self, name)[system : system + 1] = moved
            escaped += float(amount)
    _sync_active(self)
    self.escaped_total += escaped
    self.recovered_total += recovered
    self.time_s += dt
    wake = _signed_evolve_wake(self, dt, T_K, b)
    self.anisotropic_last_transport_velocity_by_system_m_s = velocities.copy()
    return {
        "dN_emit": emitted,
        "dN_source_activations": float(self.signed_last_source_activations),
        "dN_trapped": trapped,
        "dN_released": released,
        "dN_recovered": recovered,
        "dN_escaped": escaped,
        "peierls_rate_s": peierls_max,
        "taylor_completion_rate_s": taylor_max,
        "encounter_rate_s": encounter_max,
        "taylor_m_eff": m_max,
        "available_site_fraction": self.available_site_fraction,
        "anisotropic_transport_active": 1.0,
        "signed_burgers_transport": 1.0,
        **wake,
    }


def _signed_advance(self, distance_m: float) -> dict[str, float]:
    d = max(float(distance_m), 0.0)
    crossed: dict[str, np.ndarray] = {}
    lost_old_mobile = lost_old_retained = lost_old_slip = 0.0
    lost_active_mobile = lost_active_retained = lost_active_slip = 0.0
    for family in ("mobile", "retained", "slip"):
        for suffix in ("positive", "negative"):
            wake_name = f"wake_{family}_{suffix}"
            old, lost = self._advect_forward(
                getattr(self, wake_name), d, self.wake_dx
            )
            setattr(self, wake_name, old)
            if family == "mobile":
                lost_old_mobile += float(lost)
            elif family == "retained":
                lost_old_retained += float(lost)
            else:
                lost_old_slip += float(lost)
    for active_family, wake_family in (
        ("mobile", "mobile"),
        ("retained", "retained"),
        ("accumulated_slip", "slip"),
    ):
        for suffix in ("positive", "negative"):
            name = f"{active_family}_{suffix}"
            active, wake_add, lost = _translate_toward_tip(
                getattr(self, name), d, self.dx, self.wake_n_bins, self.wake_dx
            )
            setattr(self, name, active)
            wake_name = f"wake_{wake_family}_{suffix}"
            setattr(self, wake_name, getattr(self, wake_name) + wake_add)
            crossed[f"{wake_family}_{suffix}"] = wake_add
            if active_family == "mobile":
                lost_active_mobile += float(lost)
            elif active_family == "retained":
                lost_active_retained += float(lost)
            else:
                lost_active_slip += float(lost)
    _sync_active(self)
    _sync_wake(self)
    self.wake_discarded_mobile_total += lost_old_mobile + lost_active_mobile
    self.wake_discarded_retained_total += lost_old_retained + lost_active_retained
    self.wake_discarded_slip_total += lost_old_slip + lost_active_slip

    before = np.asarray(self.available_sites, dtype=float).copy()
    refresh_scale = max(
        float(getattr(self, "_campaign_refresh_scale", 1.0)), 1.0e-12
    )
    reference_length = max(
        float(self.manifest.source_refresh_length_m), float(self.dx), 1.0e-12
    )
    effective_length = reference_length * refresh_scale
    fraction = 1.0 - math.exp(-min(d / effective_length, 700.0))
    self.available_sites += (self.site_capacity - self.available_sites) * fraction
    self.available_sites = np.clip(self.available_sites, 0.0, self.site_capacity)
    self.tip_source_activity = np.divide(
        self.available_sites,
        self.site_capacity,
        out=np.zeros_like(self.available_sites),
        where=self.site_capacity > 0.0,
    )
    refreshed = max(float(np.sum(self.available_sites - before)), 0.0)
    self.campaign_source_last_refresh_fraction = fraction
    self.campaign_source_last_refresh_length_m = effective_length
    self.campaign_source_budget_remaining_total = float(np.sum(self.available_sites))
    self.campaign_source_budget_consumed_total = float(
        np.sum(self.site_capacity - self.available_sites)
    )
    self.advance_total_m += d
    wake_mobile = sum(
        float(np.sum(crossed.get(f"mobile_{suffix}", 0.0)))
        for suffix in ("positive", "negative")
    )
    wake_retained = sum(
        float(np.sum(crossed.get(f"retained_{suffix}", 0.0)))
        for suffix in ("positive", "negative")
    )
    wake_slip = sum(
        float(np.sum(crossed.get(f"slip_{suffix}", 0.0)))
        for suffix in ("positive", "negative")
    )
    return {
        "wake_mobile": wake_mobile,
        "wake_retained": wake_retained,
        "wake_slip": wake_slip,
        "source_sites_refreshed": refreshed,
        "campaign_source_refresh_fraction": fraction,
        "campaign_source_refresh_length_m": effective_length,
        "active_mobile_postcommit": self.mobile_count,
        "active_retained_postcommit": self.retained_count,
        "wake_mobile_postcommit": self.wake_mobile_count,
        "wake_retained_postcommit": self.wake_retained_count,
        "fractional_moving_frame": 1.0,
        "signed_burgers_conserved_in_moving_frame": 1.0,
    }


def _signed_diagnostics(self, G=None, nu=None, b=None, r0=None) -> dict[str, Any]:
    active = _active_K(self)
    wake = _wake_K(self)
    return {
        "mpz_state_model": MODEL_ID,
        "mpz_mobile_count": self.mobile_count,
        "mpz_retained_count": self.retained_count,
        "mpz_wake_mobile_count": self.wake_mobile_count,
        "mpz_wake_retained_count": self.wake_retained_count,
        "mpz_mobile_signed_count": float(
            np.sum(self.mobile_positive - self.mobile_negative)
        ),
        "mpz_retained_signed_count": float(
            np.sum(self.retained_positive - self.retained_negative)
        ),
        "mpz_wake_mobile_signed_count": float(
            np.sum(self.wake_mobile_positive - self.wake_mobile_negative)
        ),
        "mpz_wake_retained_signed_count": float(
            np.sum(self.wake_retained_positive - self.wake_retained_negative)
        ),
        "mpz_available_site_fraction": self.available_site_fraction,
        "mpz_active_K_shield_Pa_sqrt_m": active,
        "mpz_wake_K_shield_Pa_sqrt_m": wake,
        "mpz_total_K_shield_Pa_sqrt_m": active + wake,
        "mpz_blunted_radius_m": (
            self.blunted_radius(float(r0), float(b))
            if r0 is not None and b is not None
            else None
        ),
        "mpz_emitted_total": self.emitted_total,
        "mpz_source_activations_total": self.signed_source_activations_total,
        "mpz_signed_line_content_emitted_total": (
            self.signed_line_content_emitted_total
        ),
        "mpz_escaped_total": self.escaped_total,
        "mpz_recovered_total": self.recovered_total,
        "mpz_wake_discarded_mobile_total": self.wake_discarded_mobile_total,
        "mpz_wake_discarded_retained_total": self.wake_discarded_retained_total,
        "signed_burgers_population": True,
        "signed_shielding_kernel_model_id": KERNEL_SCHEMA,
    }


def install_signed_burgers_population(
    state, kernel: SignedShieldingKernel, transport_mode: str
) -> None:
    """Install one shared signed state law on a production MPZ instance."""
    kernel.validate_state(state)
    if (
        np.any(state.mobile)
        or np.any(state.retained)
        or np.any(state.wake_mobile)
        or np.any(state.wake_retained)
    ):
        raise RuntimeError("signed Burgers population must be installed before state evolution")
    shape = (state.n_systems, state.n_bins)
    wshape = (state.n_systems, state.wake_n_bins)
    for name in (
        "mobile_positive",
        "mobile_negative",
        "retained_positive",
        "retained_negative",
        "accumulated_slip_positive",
        "accumulated_slip_negative",
    ):
        setattr(state, name, np.zeros(shape, dtype=float))
    for name in (
        "wake_mobile_positive",
        "wake_mobile_negative",
        "wake_retained_positive",
        "wake_retained_negative",
        "wake_slip_positive",
        "wake_slip_negative",
    ):
        setattr(state, name, np.zeros(wshape, dtype=float))
    state._signed_kernel = kernel
    state.signed_source_activations_total = 0.0
    state.signed_line_content_emitted_total = 0.0
    state.signed_last_source_activations = 0.0
    state.signed_last_line_content = 0.0
    state.signed_last_line_content_by_system = np.zeros(state.n_systems)
    state.signed_last_burgers_sign_by_system = np.zeros(state.n_systems)
    state.state_model = MODEL_ID
    state._emit = MethodType(_signed_emit, state)
    selected = str(transport_mode).strip().lower().replace("-", "_")
    if selected == VALIDATED_SCALAR_TRANSPORT:
        state.evolve = MethodType(_signed_evolve_scalar, state)
    elif selected == CHANNEL_RESOLVED_TRANSPORT:
        state.evolve = MethodType(_signed_evolve_channel, state)
    else:
        raise ValueError(f"invalid signed transport mode {transport_mode!r}")
    state.advance = MethodType(_signed_advance, state)
    state.active_K_shielding = MethodType(_active_K, state)
    state.wake_K_shielding = MethodType(_wake_K, state)
    state.shielding_K = MethodType(_total_K, state)
    state.diagnostics = MethodType(_signed_diagnostics, state)
    state._signed_transport_mode = selected
    _sync_active(state)
    _sync_wake(state)


class SignedBurgersAnisotropicTipEngine(AnisotropicStochasticAvalancheTipEngine):
    """Single physical engine shared by monotonic fracture and fatigue."""

    signed_burgers_shared_core_active = True
    _signed_kernel_default: SignedShieldingKernel | None = None
    _signed_transport_mode_default = VALIDATED_SCALAR_TRANSPORT

    @classmethod
    def configure_signed_physics(
        cls,
        kernel: SignedShieldingKernel | str | Path,
        transport_mode: str = VALIDATED_SCALAR_TRANSPORT,
    ) -> None:
        cls._signed_kernel_default = (
            kernel
            if isinstance(kernel, SignedShieldingKernel)
            else SignedShieldingKernel.from_json(kernel)
        )
        cls._signed_transport_mode_default = (
            str(transport_mode).strip().lower().replace("-", "_")
        )

    @classmethod
    def audit_payload(cls) -> dict[str, Any]:
        payload = super().audit_payload()
        kernel = cls._signed_kernel_default
        payload["signed_burgers_shared_physics"] = {
            "model_id": MODEL_ID,
            "same_engine_for_monotonic_and_fatigue": True,
            "population_state": "nonnegative_positive_and_negative_Burgers_species",
            "backstress_population": "absolute_line_content",
            "shielding_population": "signed_line_content",
            "constitutive_K_shield_cap": False,
            "arbitrary_shielding_attenuation": False,
            "transport_mode": cls._signed_transport_mode_default,
            "kernel": kernel.audit_payload() if kernel is not None else None,
        }
        return payload

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kernel = type(self)._signed_kernel_default
        if kernel is None:
            raise RuntimeError(
                "signed shared physics requires a mechanically derived kernel; set "
                "SIGNED_SHIELDING_KERNEL_JSON or configure_signed_physics()"
            )
        install_signed_burgers_population(
            self.mpz, kernel, type(self)._signed_transport_mode_default
        )

    def _active_shielding_raw_uncapped(self) -> float:
        return _active_K(self.mpz)

    def _active_shielding_signed(self) -> float:
        return _active_K(self.mpz)

    def _wake_shielding_signed(self) -> float:
        return _wake_K(self.mpz)

    def K_shield(self):
        return _total_K(self.mpz)

    def _signed_engine_diagnostics(self) -> dict[str, Any]:
        kernel = type(self)._signed_kernel_default
        return {
            "signed_burgers_model_id": MODEL_ID,
            "signed_burgers_shared_monotonic_fatigue_core": True,
            "signed_burgers_transport_mode": self.mpz._signed_transport_mode,
            "signed_burgers_source_activations_total": float(
                self.mpz.signed_source_activations_total
            ),
            "signed_burgers_line_content_emitted_total": float(
                self.mpz.signed_line_content_emitted_total
            ),
            "signed_burgers_last_sign_by_system": (
                self.mpz.signed_last_burgers_sign_by_system.tolist()
            ),
            "signed_burgers_last_line_content_by_system": (
                self.mpz.signed_last_line_content_by_system.tolist()
            ),
            "signed_burgers_active_K_shield_Pa_sqrt_m": _active_K(self.mpz),
            "signed_burgers_wake_K_shield_Pa_sqrt_m": _wake_K(self.mpz),
            "signed_burgers_kernel_source": (
                kernel.source_path if kernel is not None else None
            ),
            "signed_burgers_cap_applied": False,
        }

    def step(self, K, T, dt):
        result = super().step(K, T, dt)
        result.update(self._signed_engine_diagnostics())
        return result

    def cycle_step_waveform(self, *args, **kwargs):
        result = super().cycle_step_waveform(*args, **kwargs)
        result.update(self._signed_engine_diagnostics())
        return result


__all__ = [
    "MODEL_ID",
    "KERNEL_SCHEMA",
    "SignedShieldingKernel",
    "SignedBurgersAnisotropicTipEngine",
    "install_signed_burgers_population",
]
