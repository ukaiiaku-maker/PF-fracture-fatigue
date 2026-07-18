"""v10.2.2 uncapped dislocation shielding.

The v10.1 campaign wrapper clipped the linearly superposed dislocation-induced
stress-intensity contribution to a fitted manifest value.  This module removes
that constitutive clip.  Shielding is instead limited only by the modeled
population physics: finite source capacity, Taylor back stress on emission,
Peierls--Taylor transport and escape, retained-population recovery, and
moving-frame transfer into the wake.

The old manifest value is retained only as a diagnostic reference so cap
exceedance can be quantified without affecting the kinetics.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
from typing import Any


MODEL_ID = "v10.2.2_uncapped_population_limited_shielding"
_SAMPLES: list[dict[str, float]] = []


def reset_physical_shielding_audit() -> None:
    _SAMPLES.clear()


def _legacy_cap_reference_Pa_sqrt_m(engine: Any) -> float:
    return max(
        float(getattr(engine.manifest, "max_K_shield_MPa_sqrt_m", 0.0)),
        0.0,
    ) * 1.0e6


def physical_shielding_audit_payload() -> dict[str, Any]:
    raw = [abs(float(row["raw_Pa_sqrt_m"])) for row in _SAMPLES]
    effective = [abs(float(row["effective_Pa_sqrt_m"])) for row in _SAMPLES]
    difference = [
        abs(float(row["raw_Pa_sqrt_m"]) - float(row["effective_Pa_sqrt_m"]))
        for row in _SAMPLES
    ]
    legacy = [max(float(row["legacy_cap_reference_Pa_sqrt_m"]), 0.0) for row in _SAMPLES]
    exceed = [
        bool(cap > 0.0 and value > cap)
        for value, cap in zip(raw, legacy)
    ]
    ratios = [
        value / cap
        for value, cap in zip(raw, legacy)
        if cap > 0.0
    ]
    return {
        "schema": MODEL_ID,
        "shielding_law": "signed_linear_elastic_superposition_from_evolving_dislocation_population",
        "constitutive_K_shield_clip_applied": False,
        "legacy_manifest_cap_used_in_kinetics": False,
        "legacy_manifest_cap_retained_as_diagnostic_reference": True,
        "population_saturation_controls": [
            "finite_crack_tip_source_capacity",
            "Taylor_backstress_reduces_emission_rate",
            "Peierls_Taylor_transport_and_active_zone_escape",
            "retained_population_recovery",
            "moving_frame_transfer_from_active_zone_to_wake",
        ],
        "new_fitted_saturation_parameter_introduced": False,
        "n_shielding_samples": len(_SAMPLES),
        "maximum_abs_raw_K_shield_Pa_sqrt_m": max(raw, default=0.0),
        "maximum_abs_effective_K_shield_Pa_sqrt_m": max(effective, default=0.0),
        "maximum_abs_raw_minus_effective_Pa_sqrt_m": max(difference, default=0.0),
        "n_samples_above_legacy_cap_reference": int(sum(exceed)),
        "maximum_raw_to_legacy_cap_ratio": max(ratios, default=0.0),
        "samples": list(_SAMPLES),
    }


@contextmanager
def install_uncapped_physical_shielding():
    """Temporarily replace the campaign hard cap by the raw shielding field."""
    from .campaign_calibrated_tip import CampaignCalibratedTipEngine

    original_active = CampaignCalibratedTipEngine._active_shielding_signed
    original_diagnostics = CampaignCalibratedTipEngine._campaign_diagnostics

    def uncapped_active(self) -> float:
        return float(self._active_shielding_raw_uncapped())

    def uncapped_diagnostics(self) -> dict[str, Any]:
        payload = dict(original_diagnostics(self))
        raw = float(self._active_shielding_raw_uncapped())
        effective = float(self._active_shielding_signed())
        legacy_cap = _legacy_cap_reference_Pa_sqrt_m(self)
        payload.update(
            {
                "campaign_active_K_shield_raw_Pa_sqrt_m": raw,
                "campaign_active_K_shield_effective_Pa_sqrt_m": effective,
                "campaign_active_K_shield_cap_Pa_sqrt_m": 0.0,
                "campaign_legacy_K_shield_cap_reference_Pa_sqrt_m": legacy_cap,
                "campaign_shielding_cap_applied": False,
                "campaign_shielding_population_limited": True,
                "campaign_shielding_model_id": MODEL_ID,
            }
        )
        _SAMPLES.append(
            {
                "raw_Pa_sqrt_m": raw,
                "effective_Pa_sqrt_m": effective,
                "legacy_cap_reference_Pa_sqrt_m": legacy_cap,
            }
        )
        return payload

    CampaignCalibratedTipEngine._active_shielding_signed = uncapped_active
    CampaignCalibratedTipEngine._campaign_diagnostics = uncapped_diagnostics
    try:
        yield
    finally:
        CampaignCalibratedTipEngine._active_shielding_signed = original_active
        CampaignCalibratedTipEngine._campaign_diagnostics = original_diagnostics


def _rewrite_json_if_present(path: Path, update: dict[str, Any]) -> None:
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return
    if isinstance(payload, dict):
        payload.update(update)
        path.write_text(json.dumps(payload, indent=2))


def write_physical_shielding_audit(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    payload = physical_shielding_audit_payload()

    fixed_path = root / "v10_2_1_fixed_deltaK_control.json"
    if fixed_path.is_file():
        try:
            fixed = json.loads(fixed_path.read_text())
        except Exception:
            fixed = {}
        target_kmax = float(fixed.get("target_Kmax_MPa_sqrt_m", 0.0)) * 1.0e6
        max_raw = float(payload["maximum_abs_raw_K_shield_Pa_sqrt_m"])
        payload["target_Kmax_Pa_sqrt_m"] = target_kmax
        payload["maximum_abs_raw_K_shield_over_target_Kmax"] = (
            max_raw / target_kmax if target_kmax > 0.0 else None
        )

    exact = float(payload["maximum_abs_raw_minus_effective_Pa_sqrt_m"])
    scale = max(float(payload["maximum_abs_raw_K_shield_Pa_sqrt_m"]), 1.0)
    payload["raw_equals_effective_within_relative_1e_12"] = bool(
        exact <= max(1.0e-6, 1.0e-12 * scale)
    )

    _rewrite_json_if_present(
        root / "v10_1_driver_modes.json",
        {
            "schema": MODEL_ID,
            "manifest_K_shield_cap_enabled": False,
            "legacy_manifest_K_shield_cap_reference_only": True,
            "active_shielding_saturation": "population_dynamics_only",
        },
    )
    _rewrite_json_if_present(
        root / "v10_1_1_source_model.json",
        {
            "schema": MODEL_ID,
            "cleavage_shielding_bound": "none; signed raw elastic dislocation field",
            "legacy_manifest_K_shield_cap_reference_only": True,
            "new_dimensional_saturation_parameters": 0,
        },
    )

    path = root / "v10_2_2_physical_shielding.json"
    path.write_text(json.dumps(payload, indent=2))
    return payload


__all__ = [
    "MODEL_ID",
    "install_uncapped_physical_shielding",
    "physical_shielding_audit_payload",
    "reset_physical_shielding_audit",
    "write_physical_shielding_audit",
]
