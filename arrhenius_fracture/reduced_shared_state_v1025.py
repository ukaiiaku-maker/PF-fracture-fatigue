"""Exact reduced/replay construction for the v10.2.5 signed shared engine.

The replay object is reconstructed from the complete serialized production
configuration. No front, MPZ, kinetic, anisotropic, campaign, or transport
setting is silently replaced by a reduced-model default.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .anisotropic_emission_v10174 import AnisotropicEmissionConfig
from .kinetic_tip_cell import KineticTipConfig
from .material_manifest import MaterialManifest
from .reduced_shared_state_v1023 import load_manifest
from .signed_burgers_shared_v1025 import (
    MODEL_ID as SIGNED_MODEL_ID,
    SignedBurgersAnisotropicTipEngine,
    SignedShieldingKernel,
)
from .unified_mpz import MPZConfig

MODEL_ID = "v10.2.5_exact_config_signed_replay"


def _dataclass_kwargs(cls, payload: dict[str, Any], label: str) -> dict[str, Any]:
    names = {item.name for item in fields(cls)}
    missing = sorted(name for name in names if name not in payload)
    if missing:
        raise ValueError(f"serialized {label} is missing fields: {missing}")
    return {name: payload[name] for name in names}


def _normalize(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _compare_mapping(
    expected: dict[str, Any], actual: dict[str, Any], prefix: str
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    for key in sorted(set(expected) | set(actual)):
        path = f"{prefix}.{key}"
        if key not in expected or key not in actual:
            differences.append(
                {
                    "path": path,
                    "expected": expected.get(key),
                    "actual": actual.get(key),
                }
            )
            continue
        lhs = _normalize(expected[key])
        rhs = _normalize(actual[key])
        if isinstance(lhs, float) or isinstance(rhs, float):
            try:
                equal = math.isclose(
                    float(lhs), float(rhs), rel_tol=1.0e-14, abs_tol=1.0e-18
                )
            except (TypeError, ValueError):
                equal = lhs == rhs
        else:
            equal = lhs == rhs
        if not equal:
            differences.append({"path": path, "expected": lhs, "actual": rhs})
    return differences


@dataclass
class ExactProductionConfig:
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
    signed_kernel_path: str

    @classmethod
    def from_trace(
        cls,
        payload: dict[str, Any],
        signed_kernel_path: str | Path,
    ) -> "ExactProductionConfig":
        front = dict(payload["front_config"])
        required_front = {"r0", "L_pz", "da", "sigma_cap", "m_hits", "tau_c"}
        missing_front = sorted(required_front.difference(front))
        if missing_front:
            raise ValueError(
                f"serialized front configuration is missing {missing_front}"
            )
        _dataclass_kwargs(MPZConfig, dict(payload["mpz_config"]), "MPZConfig")
        _dataclass_kwargs(
            KineticTipConfig, dict(payload["tip_config"]), "KineticTipConfig"
        )
        _dataclass_kwargs(
            AnisotropicEmissionConfig,
            dict(payload["anisotropic_config"]),
            "AnisotropicEmissionConfig",
        )
        campaign = dict(payload.get("campaign_config", {}))
        if "backstress_scale" not in campaign or "refresh_scale" not in campaign:
            raise ValueError(
                "trace lacks exact campaign backstress/refresh scales; regenerate it "
                "with state_equivalence_trace_v1025"
            )
        transport = (
            str(payload.get("transport_mode", ""))
            .strip()
            .lower()
            .replace("-", "_")
        )
        if transport not in {"validated_scalar", "channel_resolved"}:
            raise ValueError(f"invalid serialized transport mode {transport!r}")
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
            signed_kernel_path=str(Path(signed_kernel_path).expanduser().resolve()),
        )

    def build_engine(
        self, manifest: MaterialManifest
    ) -> SignedBurgersAnisotropicTipEngine:
        front = SimpleNamespace(**copy_json(self.front_config))
        mpz_cfg = MPZConfig(
            **_dataclass_kwargs(MPZConfig, self.mpz_config, "MPZConfig")
        )
        tip_cfg = KineticTipConfig(
            **_dataclass_kwargs(
                KineticTipConfig, self.tip_config, "KineticTipConfig"
            )
        ).validate()
        anisotropic_cfg = AnisotropicEmissionConfig(
            **_dataclass_kwargs(
                AnisotropicEmissionConfig,
                self.anisotropic_config,
                "AnisotropicEmissionConfig",
            )
        ).validate()
        kernel = SignedShieldingKernel.from_json(self.signed_kernel_path)
        engine_type = SignedBurgersAnisotropicTipEngine
        engine_type.configure_default(tip_cfg)
        engine_type.configure_campaign(
            self.campaign_backstress_scale,
            self.campaign_refresh_scale,
        )
        engine_type.configure_anisotropic_emission(anisotropic_cfg)
        engine_type.configure_signed_physics(kernel, self.transport_mode)
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
        report = self.parity_report(engine)
        if not report["passed"]:
            raise RuntimeError(
                "production/replay configuration parity failed: "
                + json.dumps(report["differences"], sort_keys=True)
            )
        engine._v1025_config_parity = report
        return engine

    def parity_report(self, engine) -> dict[str, Any]:
        actual_front = {
            name: getattr(engine.f, name)
            for name in self.front_config
            if hasattr(engine.f, name)
        }
        actual_mpz = asdict(engine.mpz.cfg)
        actual_tip = asdict(engine.tip_cfg)
        actual_anisotropic = asdict(engine.anisotropic_cfg)
        differences = []
        differences += _compare_mapping(self.front_config, actual_front, "front")
        differences += _compare_mapping(self.mpz_config, actual_mpz, "mpz")
        differences += _compare_mapping(self.tip_config, actual_tip, "tip")
        differences += _compare_mapping(
            self.anisotropic_config, actual_anisotropic, "anisotropic"
        )
        scalars = {
            "G_Pa": (self.G_Pa, engine.G),
            "poisson": (self.poisson, engine.nu),
            "b_m": (self.b_m, engine.b),
            "campaign_backstress_scale": (
                self.campaign_backstress_scale,
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
            if isinstance(expected, str):
                equal = str(expected) == str(actual)
            else:
                equal = math.isclose(
                    float(expected),
                    float(actual),
                    rel_tol=1.0e-14,
                    abs_tol=1.0e-18,
                )
            if not equal:
                differences.append(
                    {"path": name, "expected": expected, "actual": actual}
                )
        return {
            "schema": MODEL_ID,
            "complete_front_config": True,
            "complete_mpz_config": True,
            "complete_tip_config": True,
            "complete_anisotropic_config": True,
            "local_strength_sigma_cap_preserved_Pa": float(
                self.front_config["sigma_cap"]
            ),
            "K_shield_cap_present": False,
            "differences": differences,
            "passed": not differences,
        }


def copy_json(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def state_arrays(engine) -> dict[str, np.ndarray]:
    state = engine.mpz
    names = (
        "mobile",
        "retained",
        "accumulated_slip",
        "available_sites",
        "site_capacity",
        "wake_mobile",
        "wake_retained",
        "wake_slip",
        "mobile_positive",
        "mobile_negative",
        "retained_positive",
        "retained_negative",
        "accumulated_slip_positive",
        "accumulated_slip_negative",
        "wake_mobile_positive",
        "wake_mobile_negative",
        "wake_retained_positive",
        "wake_retained_negative",
        "wake_slip_positive",
        "wake_slip_negative",
    )
    return {
        name: np.asarray(getattr(state, name), dtype=float).copy()
        for name in names
    }


def read_schedule(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "dt_s",
        "temperature_K",
        "K_Pa_sqrt_m",
        "drive_factor_0",
        "drive_factor_1",
        "tau_signed_0_Pa",
        "tau_signed_1_Pa",
    }
    if not rows:
        raise ValueError(f"empty replay schedule: {path}")
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(
            f"signed replay schedule is missing {missing}; regenerate the 2-D trace"
        )
    parsed = []
    for index, row in enumerate(rows):
        parsed.append(
            {
                "row_index": index,
                "dt_s": float(row["dt_s"]),
                "temperature_K": float(row["temperature_K"]),
                "K_Pa_sqrt_m": float(row["K_Pa_sqrt_m"]),
                "drive_factors": np.array(
                    [
                        float(row["drive_factor_0"]),
                        float(row["drive_factor_1"]),
                    ],
                    dtype=float,
                ),
                "tau_signed_Pa": np.array(
                    [
                        float(row["tau_signed_0_Pa"]),
                        float(row["tau_signed_1_Pa"]),
                    ],
                    dtype=float,
                ),
                "expected_fired": (
                    str(row.get("expected_fired", "")).strip().lower()
                    in {"1", "true", "yes"}
                    if row.get("expected_fired", "") != ""
                    else None
                ),
            }
        )
    return parsed


def replay_exact_signed_state(
    manifest: MaterialManifest,
    schedule: str | Path | list[dict[str, Any]],
    config: ExactProductionConfig,
) -> dict[str, Any]:
    rows = (
        read_schedule(schedule)
        if isinstance(schedule, (str, Path))
        else list(schedule)
    )
    engine = config.build_engine(manifest)
    history = []
    all_fired = True
    for row in rows:
        factors = np.asarray(row["drive_factors"], dtype=float)
        tau = np.asarray(row["tau_signed_Pa"], dtype=float)
        if factors.shape != (engine.mpz.n_systems,) or tau.shape != (
            engine.mpz.n_systems,
        ):
            raise ValueError(
                "signed drive dimension does not match production systems"
            )
        engine.mpz._anisotropic_drive_factors = factors.copy()
        engine.mpz._anisotropic_tau_signed_Pa = tau.copy()
        engine.mpz._anisotropic_drive_reliable = True
        result = engine.step(
            float(row["K_Pa_sqrt_m"]),
            float(row["temperature_K"]),
            max(float(row["dt_s"]), 0.0),
        )
        fired = bool(result.get("fired", False))
        expected = row.get("expected_fired")
        matched = expected is None or fired == bool(expected)
        all_fired = all_fired and matched
        history.append(
            {
                "row_index": int(row.get("row_index", len(history))),
                "fired": fired,
                "fired_matches_expected": matched,
                "K_Pa_sqrt_m": float(row["K_Pa_sqrt_m"]),
                "temperature_K": float(row["temperature_K"]),
                "dt_s": float(row["dt_s"]),
                "sigma_tip_Pa": float(result.get("sigma_tip", 0.0)),
                "sigma_cap_active": bool(result.get("sigma_cap_active", False)),
                "K_shield_Pa_sqrt_m": float(engine.K_shield()),
                "source_activations_total": float(
                    engine.mpz.signed_source_activations_total
                ),
                "signed_line_content_total": float(
                    engine.mpz.signed_line_content_emitted_total
                ),
                "mobile_signed_count": float(
                    np.sum(
                        engine.mpz.mobile_positive
                        - engine.mpz.mobile_negative
                    )
                ),
                "retained_signed_count": float(
                    np.sum(
                        engine.mpz.retained_positive
                        - engine.mpz.retained_negative
                    )
                ),
            }
        )
    return {
        "schema": MODEL_ID,
        "signed_model_id": SIGNED_MODEL_ID,
        "candidate_id": manifest.candidate_id,
        "n_schedule_rows": len(rows),
        "history": history,
        "all_fired_flags_match": all_fired,
        "config_parity": engine._v1025_config_parity,
        "same_engine_for_monotonic_and_fatigue": True,
        "constitutive_K_shield_cap_applied": False,
        "_final_arrays": state_arrays(engine),
    }


__all__ = [
    "MODEL_ID",
    "ExactProductionConfig",
    "load_manifest",
    "read_schedule",
    "replay_exact_signed_state",
    "state_arrays",
]
