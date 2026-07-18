"""Exact state-resolved reduced R-curve campaign for v10.2.7.

The reduced path changes only the external mechanical solve. It reconstructs the
complete production configuration, executes the v10.2.6 signed state-resolved
engine, resolves the same signed shielding-kernel family, and supplies signed
emission drives from a candidate-independent 2-D tensor-probe family.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np

from .anisotropic_emission_v10174 import AnisotropicEmissionConfig
from .kinetic_tip_cell import KineticTipConfig
from .material_manifest import MaterialManifest
from .reduced_shared_state_v1025 import (
    _compare_mapping,
    _dataclass_kwargs,
    copy_json,
    state_arrays,
)
from .signed_kernel_family_v1026 import StateResolvedSignedShieldingKernelFamily
from .state_resolved_drive_family_v1027 import StateResolvedSignedDriveFamily
from .state_resolved_signed_engine_v1026 import StateResolvedSignedBurgersTipEngine
from .unified_mpz import MPZConfig

MODEL_ID = "v10.2.7_exact_state_resolved_two_class_campaign"
VALID_MODES = {"full", "plasticity_off", "shielding_off", "backstress_off"}
DEFAULT_TEMPERATURES_K = (300.0, 700.0, 900.0, 1200.0)


def _zero_shielding(self, *args, **kwargs) -> float:
    return 0.0


def _zero_emit(self, dt, stress_Pa, T_K, system_weights=None) -> float:
    self.signed_last_source_activations = 0.0
    self.signed_last_line_content = 0.0
    return 0.0


@dataclass
class ReducedCampaignControl:
    Kdot_MPa_sqrt_m_s: float = 0.005
    Kmax_MPa_sqrt_m: float = 80.0
    dK_MPa_sqrt_m: float = 0.05
    target_extension_um: float = 50.0
    max_outer_steps: int = 2_000_000

    def validate(self) -> "ReducedCampaignControl":
        if self.Kdot_MPa_sqrt_m_s <= 0.0:
            raise ValueError("Kdot must be positive")
        if self.Kmax_MPa_sqrt_m <= 0.0 or self.dK_MPa_sqrt_m <= 0.0:
            raise ValueError("Kmax and dK must be positive")
        if self.target_extension_um <= 0.0 or self.max_outer_steps < 1:
            raise ValueError("target extension and outer-step limit must be positive")
        return self


@dataclass
class StateResolvedProductionConfig:
    front_config: dict[str, Any]
    mpz_config: dict[str, Any]
    tip_config: dict[str, Any]
    anisotropic_config: dict[str, Any]
    G_Pa: float
    poisson: float
    b_m: float
    transport_mode: str
    campaign_backstress_scale: float
    campaign_refresh_scale: float
    kernel_family_path: str
    drive_family_path: str

    @classmethod
    def from_trace(
        cls,
        payload: dict[str, Any],
        *,
        kernel_family_path: str | Path,
        drive_family_path: str | Path,
    ) -> "StateResolvedProductionConfig":
        front = dict(payload["front_config"])
        required_front = {"r0", "L_pz", "da", "sigma_cap", "m_hits", "tau_c"}
        missing = sorted(required_front.difference(front))
        if missing:
            raise ValueError(f"serialized front configuration is missing {missing}")
        _dataclass_kwargs(MPZConfig, dict(payload["mpz_config"]), "MPZConfig")
        _dataclass_kwargs(KineticTipConfig, dict(payload["tip_config"]), "KineticTipConfig")
        _dataclass_kwargs(
            AnisotropicEmissionConfig,
            dict(payload["anisotropic_config"]),
            "AnisotropicEmissionConfig",
        )
        campaign = dict(payload.get("campaign_config", {}))
        if "backstress_scale" not in campaign or "refresh_scale" not in campaign:
            raise ValueError("trace lacks exact campaign backstress/refresh scales")
        transport = str(payload.get("transport_mode", "")).strip().lower().replace("-", "_")
        if transport not in {"validated_scalar", "channel_resolved"}:
            raise ValueError(f"invalid serialized transport mode {transport!r}")
        kernel = StateResolvedSignedShieldingKernelFamily.from_json(kernel_family_path)
        drive = StateResolvedSignedDriveFamily.from_json(drive_family_path)
        drive.validate_against_kernel_family(kernel)
        if not bool(kernel.metadata.get("production_parameterization_allowed", False)):
            raise ValueError("shielding-kernel family is not authorized for parameterization")
        if not bool(drive.metadata.get("production_parameterization_allowed", False)):
            raise ValueError("signed-drive family is not authorized for parameterization")
        return cls(
            front_config=front,
            mpz_config=dict(payload["mpz_config"]),
            tip_config=dict(payload["tip_config"]),
            anisotropic_config=dict(payload["anisotropic_config"]),
            G_Pa=float(payload["G_Pa"]),
            poisson=float(payload["poisson"]),
            b_m=float(payload["b_m"]),
            transport_mode=transport,
            campaign_backstress_scale=float(campaign["backstress_scale"]),
            campaign_refresh_scale=float(campaign["refresh_scale"]),
            kernel_family_path=str(Path(kernel_family_path).expanduser().resolve()),
            drive_family_path=str(Path(drive_family_path).expanduser().resolve()),
        )

    def load_families(self):
        kernel = StateResolvedSignedShieldingKernelFamily.from_json(
            self.kernel_family_path
        )
        drive = StateResolvedSignedDriveFamily.from_json(self.drive_family_path)
        drive.validate_against_kernel_family(kernel)
        return kernel, drive

    def build_engine(self, manifest: MaterialManifest, *, mode: str = "full"):
        mode = str(mode)
        if mode not in VALID_MODES:
            raise ValueError(f"invalid campaign mode {mode!r}")
        front = SimpleNamespace(**copy_json(self.front_config))
        mpz_cfg = MPZConfig(
            **_dataclass_kwargs(MPZConfig, self.mpz_config, "MPZConfig")
        )
        tip_cfg = KineticTipConfig(
            **_dataclass_kwargs(KineticTipConfig, self.tip_config, "KineticTipConfig")
        ).validate()
        anisotropic_cfg = AnisotropicEmissionConfig(
            **_dataclass_kwargs(
                AnisotropicEmissionConfig,
                self.anisotropic_config,
                "AnisotropicEmissionConfig",
            )
        ).validate()
        kernel, drive = self.load_families()
        engine_type = StateResolvedSignedBurgersTipEngine
        engine_type.configure_default(tip_cfg)
        engine_type.configure_campaign(
            0.0 if mode == "backstress_off" else self.campaign_backstress_scale,
            self.campaign_refresh_scale,
        )
        engine_type.configure_anisotropic_emission(anisotropic_cfg)
        engine_type.configure_state_resolved_physics(kernel, self.transport_mode)
        engine = engine_type(
            front,
            manifest.cleavage,
            manifest.emission,
            self.G_Pa,
            self.poisson,
            self.b_m,
            manifest,
            mpz_cfg,
        )
        engine.tip_cfg = tip_cfg
        if mode == "plasticity_off":
            engine.mpz._emit = MethodType(_zero_emit, engine.mpz)
        if mode in {"plasticity_off", "shielding_off"}:
            engine._active_shielding_raw_uncapped = MethodType(_zero_shielding, engine)
            engine._active_shielding_signed = MethodType(_zero_shielding, engine)
            engine._wake_shielding_signed = MethodType(_zero_shielding, engine)
            engine.K_shield = MethodType(_zero_shielding, engine)
        report = self.parity_report(engine, mode=mode)
        if not report["passed"]:
            raise RuntimeError(
                "production/reduced configuration parity failed: "
                + json.dumps(report["differences"], sort_keys=True)
            )
        engine._v1027_config_parity = report
        return engine, drive

    def parity_report(self, engine, *, mode: str) -> dict[str, Any]:
        actual_front = {
            name: getattr(engine.f, name)
            for name in self.front_config
            if hasattr(engine.f, name)
        }
        differences: list[dict[str, Any]] = []
        differences += _compare_mapping(self.front_config, actual_front, "front")
        differences += _compare_mapping(self.mpz_config, asdict(engine.mpz.cfg), "mpz")
        differences += _compare_mapping(self.tip_config, asdict(engine.tip_cfg), "tip")
        differences += _compare_mapping(
            self.anisotropic_config,
            asdict(engine.anisotropic_cfg),
            "anisotropic",
        )
        expected_backstress = (
            0.0 if mode == "backstress_off" else self.campaign_backstress_scale
        )
        scalars = {
            "G_Pa": (self.G_Pa, engine.G),
            "poisson": (self.poisson, engine.nu),
            "b_m": (self.b_m, engine.b),
            "campaign_backstress_scale": (
                expected_backstress,
                engine.mpz._campaign_backstress_scale,
            ),
            "campaign_refresh_scale": (
                self.campaign_refresh_scale,
                engine.mpz._campaign_refresh_scale,
            ),
            "transport_mode": (
                self.transport_mode,
                engine.mpz._signed_transport_mode,
            ),
        }
        for name, (expected, actual) in scalars.items():
            equal = (
                str(expected) == str(actual)
                if isinstance(expected, str)
                else math.isclose(
                    float(expected), float(actual), rel_tol=1.0e-14, abs_tol=1.0e-18
                )
            )
            if not equal:
                differences.append(
                    {"path": name, "expected": expected, "actual": actual}
                )
        return {
            "schema": MODEL_ID,
            "mode": mode,
            "complete_configuration_reconstructed": True,
            "same_v10_2_6_engine_class": True,
            "local_strength_sigma_cap_Pa": float(self.front_config["sigma_cap"]),
            "K_shield_cap_present": False,
            "differences": differences,
            "passed": not differences,
        }


def run_reduced_r_curve(
    manifest: MaterialManifest,
    temperature_K: float,
    production: StateResolvedProductionConfig,
    control: ReducedCampaignControl,
    *,
    mode: str = "full",
) -> dict[str, Any]:
    control = control.validate()
    engine, drive = production.build_engine(manifest, mode=mode)
    da_m = max(float(engine.f.da), 1.0e-30)
    target_advances = max(
        int(math.ceil(control.target_extension_um * 1.0e-6 / da_m)), 1
    )
    K_left = 0.0
    outer_step = 0
    history: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    previous_advances = int(engine.n_adv)

    while K_left < control.Kmax_MPa_sqrt_m and engine.n_adv < target_advances:
        outer_step += 1
        if outer_step > control.max_outer_steps:
            raise RuntimeError("reduced R-curve exceeded outer-step limit")
        dK = min(control.dK_MPa_sqrt_m, control.Kmax_MPa_sqrt_m - K_left)
        K_mid = K_left + 0.5 * dK
        K_mid_Pa = K_mid * 1.0e6
        coordinates = engine._kernel_state_coordinates(K_mid_Pa)
        signed_factors = drive.resolve(**coordinates)
        sigma_local = (
            float(coordinates["opening_strength_fraction"])
            * float(engine.f.sigma_cap)
        )
        engine.mpz._anisotropic_drive_factors = np.abs(signed_factors)
        engine.mpz._anisotropic_tau_signed_Pa = signed_factors * sigma_local
        engine.mpz._anisotropic_drive_reliable = True
        engine.mpz._anisotropic_drive_serial = outer_step

        dt_requested = dK / control.Kdot_MPa_sqrt_m_s
        result = engine.step(K_mid_Pa, float(temperature_K), dt_requested)
        consumed = min(
            max(float(result.get("kinetic_dt_consumed_s", dt_requested)), 0.0),
            dt_requested,
        )
        fired = bool(result.get("fired", False))
        K_right = (
            K_left + control.Kdot_MPa_sqrt_m_s * consumed
            if fired
            else K_left + dK
        )
        current_advances = int(engine.n_adv)
        if current_advances > previous_advances:
            for advance_index in range(previous_advances + 1, current_advances + 1):
                events.append(
                    {
                        "advance_index": advance_index,
                        "crack_extension_um": advance_index * da_m * 1.0e6,
                        "K_MPa_sqrt_m": float(K_right),
                        "temperature_K": float(temperature_K),
                        "mode": mode,
                    }
                )
            previous_advances = current_advances
        history.append(
            {
                "outer_step": outer_step,
                "K_MPa_sqrt_m": float(K_right),
                "K_input_MPa_sqrt_m": float(K_mid),
                "temperature_K": float(temperature_K),
                "mode": mode,
                "fired": fired,
                "checkpoint_advances": current_advances,
                "K_shield_MPa_sqrt_m": float(engine.K_shield()) / 1.0e6,
                "r_eff_over_r0": float(coordinates["r_eff_over_r0"]),
                "opening_strength_fraction": float(
                    coordinates["opening_strength_fraction"]
                ),
                "crack_extension_m": float(coordinates["crack_extension_m"]),
                "drive_signed_factor_0": float(signed_factors[0]),
                "drive_signed_factor_1": (
                    float(signed_factors[1]) if signed_factors.size > 1 else 0.0
                ),
                "mobile_count": float(engine.mpz.mobile_count),
                "retained_count": float(engine.mpz.retained_count),
                "source_activations_total": float(
                    engine.mpz.signed_source_activations_total
                ),
                "signed_line_content_total": float(
                    engine.mpz.signed_line_content_emitted_total
                ),
            }
        )
        K_left = float(K_right)
        if fired and consumed <= 0.0:
            K_left += min(dK, 1.0e-12)

    complete = int(engine.n_adv) >= target_advances
    K_init = float(events[0]["K_MPa_sqrt_m"]) if events else math.nan
    K_final = float(events[-1]["K_MPa_sqrt_m"]) if complete and events else math.nan
    return {
        "schema": MODEL_ID,
        "candidate_id": manifest.candidate_id,
        "material_class": manifest.name,
        "mode": mode,
        "temperature_K": float(temperature_K),
        "status": "complete" if complete else "incomplete",
        "K_init_MPa_sqrt_m": K_init,
        "K_final_MPa_sqrt_m": K_final,
        "R_rise_MPa_sqrt_m": K_final - K_init if complete and events else math.nan,
        "R_rise_fraction": (
            (K_final - K_init) / K_init
            if complete and events and K_init > 0.0
            else math.nan
        ),
        "target_advances": target_advances,
        "checkpoint_advances": int(engine.n_adv),
        "events": events,
        "history": history,
        "final_arrays": state_arrays(engine),
        "production_config_parity": engine._v1027_config_parity,
        "drive_family_audit": drive.audit_payload(),
        "kernel_family_audit": engine._state_kernel_family.audit_payload(),
        "constitutive_K_shield_cap_applied": False,
    }


def _finite_complete(result: dict[str, Any]) -> bool:
    return bool(
        result.get("status") == "complete"
        and math.isfinite(float(result.get("K_init_MPa_sqrt_m", math.nan)))
        and math.isfinite(float(result.get("K_final_MPa_sqrt_m", math.nan)))
    )


def _ratio(high: float, low: float) -> float:
    return high / low if math.isfinite(high) and math.isfinite(low) and low > 0.0 else math.nan


def score_dbtt(results: dict[tuple[str, float], dict[str, Any]]) -> dict[str, Any]:
    low_T, high_T = DEFAULT_TEMPERATURES_K[0], DEFAULT_TEMPERATURES_K[-1]
    full = [results[("full", T)] for T in DEFAULT_TEMPERATURES_K]
    complete = all(_finite_complete(row) for row in full)
    final = np.asarray([row["K_final_MPa_sqrt_m"] for row in full], dtype=float)
    init = np.asarray([row["K_init_MPa_sqrt_m"] for row in full], dtype=float)
    rise_frac = np.asarray([row["R_rise_fraction"] for row in full], dtype=float)
    full_ratio = _ratio(final[-1], final[0])
    off_ratio = _ratio(
        results[("plasticity_off", high_T)]["K_final_MPa_sqrt_m"],
        results[("plasticity_off", low_T)]["K_final_MPa_sqrt_m"],
    )
    full_temp_rise = final[-1] - final[0]
    shield_temp_rise = (
        results[("shielding_off", high_T)]["K_final_MPa_sqrt_m"]
        - results[("shielding_off", low_T)]["K_final_MPa_sqrt_m"]
    )
    shielding_temp_fraction = (
        (full_temp_rise - shield_temp_rise) / full_temp_rise
        if full_temp_rise > 1.0e-12
        else math.nan
    )
    high_full_R = float(full[-1]["R_rise_MPa_sqrt_m"])
    high_shield_R = float(
        results[("shielding_off", high_T)]["R_rise_MPa_sqrt_m"]
    )
    shielding_R_fraction = (
        (high_full_R - high_shield_R) / high_full_R
        if high_full_R > 1.0e-12
        else math.nan
    )
    backstress_high_R = float(
        results[("backstress_off", high_T)]["R_rise_MPa_sqrt_m"]
    )
    monotonic_fraction = (
        float(np.mean(np.diff(final) >= -1.0e-9)) if complete else 0.0
    )
    strict = bool(
        complete
        and 5.0 <= init[0] <= 25.0
        and final[-1] <= 80.0
        and full_ratio >= 1.5
        and rise_frac[0] <= 0.15
        and rise_frac[-1] >= 0.20
        and math.isfinite(off_ratio)
        and off_ratio <= 1.25
        and math.isfinite(shielding_temp_fraction)
        and shielding_temp_fraction >= 0.50
        and math.isfinite(shielding_R_fraction)
        and shielding_R_fraction >= 0.30
        and backstress_high_R > 0.0
        and monotonic_fraction >= 0.90
    )
    objective = 0.0
    objective += 20.0 * max(0.0, 1.5 - (full_ratio if math.isfinite(full_ratio) else 0.0)) ** 2
    objective += 15.0 * max(0.0, rise_frac[0] - 0.15) ** 2
    objective += 15.0 * max(0.0, 0.20 - rise_frac[-1]) ** 2
    objective += 15.0 * max(0.0, (off_ratio if math.isfinite(off_ratio) else 10.0) - 1.25) ** 2
    objective += 20.0 * max(0.0, 0.50 - (shielding_temp_fraction if math.isfinite(shielding_temp_fraction) else -1.0)) ** 2
    objective += 10.0 * max(0.0, 0.30 - (shielding_R_fraction if math.isfinite(shielding_R_fraction) else -1.0)) ** 2
    objective += 5.0 * max(0.0, 0.90 - monotonic_fraction) ** 2
    if not complete:
        objective += 1.0e6
    return {
        "target_class": "DBTT",
        "strict_reduced_pass": strict,
        "objective": float(objective),
        "full_endpoint_ratio": full_ratio,
        "low_R_rise_fraction": float(rise_frac[0]),
        "high_R_rise_fraction": float(rise_frac[-1]),
        "plasticity_off_endpoint_ratio": off_ratio,
        "shielding_fraction_of_temperature_rise": shielding_temp_fraction,
        "shielding_fraction_of_high_T_R_rise": shielding_R_fraction,
        "backstress_off_high_T_R_rise_MPa_sqrt_m": backstress_high_R,
        "monotonic_temperature_fraction": monotonic_fraction,
        **{
            f"full_K_init_{int(T)}K": float(value)
            for T, value in zip(DEFAULT_TEMPERATURES_K, init)
        },
        **{
            f"full_K_final_{int(T)}K": float(value)
            for T, value in zip(DEFAULT_TEMPERATURES_K, final)
        },
    }


def score_weakt(results: dict[tuple[str, float], dict[str, Any]]) -> dict[str, Any]:
    low_T, high_T = DEFAULT_TEMPERATURES_K[0], DEFAULT_TEMPERATURES_K[-1]
    full = [results[("full", T)] for T in DEFAULT_TEMPERATURES_K]
    complete = all(_finite_complete(row) for row in full)
    final = np.asarray([row["K_final_MPa_sqrt_m"] for row in full], dtype=float)
    init = np.asarray([row["K_init_MPa_sqrt_m"] for row in full], dtype=float)
    rise = np.asarray([row["R_rise_MPa_sqrt_m"] for row in full], dtype=float)
    rise_frac = np.asarray([row["R_rise_fraction"] for row in full], dtype=float)
    final_span = float(np.max(final) / np.min(final)) if complete and np.min(final) > 0 else math.nan
    init_span = float(np.max(init) / np.min(init)) if complete and np.min(init) > 0 else math.nan
    mean_R = float(np.mean(rise)) if complete else math.nan
    off_mean_R = float(
        np.mean(
            [
                results[("plasticity_off", T)]["R_rise_MPa_sqrt_m"]
                for T in (low_T, high_T)
            ]
        )
    )
    shield_mean_R = float(
        np.mean(
            [
                results[("shielding_off", T)]["R_rise_MPa_sqrt_m"]
                for T in (low_T, high_T)
            ]
        )
    )
    plasticity_fraction = (
        (mean_R - off_mean_R) / mean_R if mean_R > 1.0e-12 else math.nan
    )
    shielding_fraction = (
        (mean_R - shield_mean_R) / mean_R if mean_R > 1.0e-12 else math.nan
    )
    strict = bool(
        complete
        and 5.0 <= float(np.min(init))
        and float(np.max(final)) <= 60.0
        and final_span <= 1.20
        and init_span <= 1.20
        and float(np.min(rise_frac)) >= 0.05
        and float(np.max(rise_frac)) <= 0.25
        and float(np.min(rise)) >= 0.5
        and math.isfinite(plasticity_fraction)
        and plasticity_fraction >= 0.30
        and math.isfinite(shielding_fraction)
        and shielding_fraction >= 0.15
    )
    objective = 0.0
    objective += 20.0 * max(0.0, (final_span if math.isfinite(final_span) else 10.0) - 1.20) ** 2
    objective += 15.0 * max(0.0, (init_span if math.isfinite(init_span) else 10.0) - 1.20) ** 2
    objective += 15.0 * max(0.0, 0.05 - float(np.nanmin(rise_frac))) ** 2
    objective += 10.0 * max(0.0, float(np.nanmax(rise_frac)) - 0.25) ** 2
    objective += 10.0 * max(0.0, 0.5 - float(np.nanmin(rise))) ** 2
    objective += 10.0 * max(0.0, 0.30 - (plasticity_fraction if math.isfinite(plasticity_fraction) else -1.0)) ** 2
    objective += 5.0 * max(0.0, 0.15 - (shielding_fraction if math.isfinite(shielding_fraction) else -1.0)) ** 2
    if not complete:
        objective += 1.0e6
    return {
        "target_class": "weakT",
        "strict_reduced_pass": strict,
        "objective": float(objective),
        "full_final_temperature_span_ratio": final_span,
        "full_init_temperature_span_ratio": init_span,
        "minimum_R_rise_MPa_sqrt_m": float(np.nanmin(rise)),
        "maximum_R_rise_MPa_sqrt_m": float(np.nanmax(rise)),
        "minimum_R_rise_fraction": float(np.nanmin(rise_frac)),
        "maximum_R_rise_fraction": float(np.nanmax(rise_frac)),
        "plasticity_fraction_of_mean_R_rise": plasticity_fraction,
        "shielding_fraction_of_mean_R_rise": shielding_fraction,
        **{
            f"full_K_init_{int(T)}K": float(value)
            for T, value in zip(DEFAULT_TEMPERATURES_K, init)
        },
        **{
            f"full_K_final_{int(T)}K": float(value)
            for T, value in zip(DEFAULT_TEMPERATURES_K, final)
        },
    }


def score_ceramic_reference(results: dict[tuple[str, float], dict[str, Any]]) -> dict[str, Any]:
    full = [results[("full", T)] for T in DEFAULT_TEMPERATURES_K]
    complete = all(_finite_complete(row) for row in full)
    final = np.asarray([row["K_final_MPa_sqrt_m"] for row in full], dtype=float)
    rise_frac = np.asarray([row["R_rise_fraction"] for row in full], dtype=float)
    span = float(np.max(final) / np.min(final)) if complete and np.min(final) > 0 else math.nan
    passed = bool(
        complete
        and span <= 1.20
        and float(np.nanmax(np.abs(rise_frac))) <= 0.05
    )
    return {
        "target_class": "ceramic",
        "frozen_reference_pass": passed,
        "full_final_temperature_span_ratio": span,
        "maximum_abs_R_rise_fraction": float(np.nanmax(np.abs(rise_frac))),
    }


__all__ = [
    "MODEL_ID",
    "DEFAULT_TEMPERATURES_K",
    "ReducedCampaignControl",
    "StateResolvedProductionConfig",
    "run_reduced_r_curve",
    "score_dbtt",
    "score_weakt",
    "score_ceramic_reference",
]
