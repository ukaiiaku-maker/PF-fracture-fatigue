"""Trace the exact v10.2.5 signed production engine for reduced replay."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import csv
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .reduced_shared_state_v1025 import state_arrays
from .signed_burgers_shared_v1025 import SignedBurgersAnisotropicTipEngine

MODEL_ID = "v10.2.5_2d_exact_signed_state_trace"


def _asdict_or_public(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    result = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            item = getattr(value, name)
        except Exception:
            continue
        if isinstance(item, (bool, int, float, str)) or item is None:
            result[name] = item
    return result


def engine_config(engine) -> dict[str, Any]:
    kernel = type(engine)._signed_kernel_default
    return {
        "schema": MODEL_ID,
        "front_config": _asdict_or_public(engine.f),
        "mpz_config": _asdict_or_public(engine.mpz.cfg),
        "tip_config": _asdict_or_public(engine.tip_cfg),
        "anisotropic_config": _asdict_or_public(engine.anisotropic_cfg),
        "campaign_config": {
            "backstress_scale": float(engine.mpz._campaign_backstress_scale),
            "refresh_scale": float(engine.mpz._campaign_refresh_scale),
        },
        "G_Pa": float(engine.G),
        "poisson": float(engine.nu),
        "b_m": float(engine.b),
        "material_manifest": engine.manifest.as_dict(),
        "transport_mode": str(engine.mpz._signed_transport_mode),
        "signed_kernel_path": kernel.source_path if kernel is not None else None,
        "local_strength_sigma_cap_is_not_Kshield_cap": True,
        "state_class": type(engine.mpz).__name__,
        "engine_class": type(engine).__name__,
    }


@contextmanager
def capture_exact_signed_trace() -> Iterator[dict[str, Any]]:
    engine_type = SignedBurgersAnisotropicTipEngine
    original = engine_type.step
    trace: dict[str, Any] = {"schema": MODEL_ID, "records": [], "last_engine": None}

    def traced_step(self, K, T, dt):
        result = original(self, K, T, dt)
        factors = np.asarray(self.mpz._anisotropic_drive_factors, dtype=float)
        tau = np.asarray(self.mpz._anisotropic_tau_signed_Pa, dtype=float)
        signs = np.asarray(self.mpz.signed_last_burgers_sign_by_system, dtype=float)
        record = {
            "trace_index": len(trace["records"]),
            "dt_s": float(dt),
            "temperature_K": float(T),
            "K_Pa_sqrt_m": float(K),
            "drive_factor_0": float(factors[0]),
            "drive_factor_1": float(factors[1]),
            "tau_signed_0_Pa": float(tau[0]),
            "tau_signed_1_Pa": float(tau[1]),
            "emitted_burgers_sign_0": float(signs[0]),
            "emitted_burgers_sign_1": float(signs[1]),
            "expected_fired": bool(result.get("fired", False)),
            "expected_sigma_tip_Pa": float(result.get("sigma_tip", 0.0)),
            "expected_sigma_cap_active": bool(result.get("sigma_cap_active", False)),
            "expected_K_shield_Pa_sqrt_m": float(self.K_shield()),
            "expected_mobile_count": float(self.mpz.mobile_count),
            "expected_retained_count": float(self.mpz.retained_count),
            "expected_mobile_signed_count": float(np.sum(self.mpz.mobile_positive - self.mpz.mobile_negative)),
            "expected_retained_signed_count": float(np.sum(self.mpz.retained_positive - self.mpz.retained_negative)),
            "expected_source_activations_total": float(self.mpz.signed_source_activations_total),
            "expected_line_content_total": float(self.mpz.signed_line_content_emitted_total),
            "expected_micro_advance_total_m": float(getattr(self, "micro_advance_total_m", 0.0)),
        }
        trace["records"].append(record)
        trace["last_engine"] = self
        return result

    engine_type.step = traced_step
    try:
        yield trace
    finally:
        engine_type.step = original


def write_exact_signed_trace(trace: dict[str, Any], root: str | Path) -> dict[str, Any]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    records = list(trace.get("records", []))
    engine = trace.get("last_engine")
    if not records or engine is None:
        raise RuntimeError("no signed production-engine steps were captured")
    schedule = root / "v10_2_5_2d_signed_replay_schedule.csv"
    with schedule.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    final_state = root / "v10_2_5_2d_signed_final_state.npz"
    np.savez_compressed(final_state, **state_arrays(engine))
    config_path = root / "v10_2_5_2d_exact_engine_config.json"
    config_path.write_text(json.dumps(engine_config(engine), indent=2))
    audit = {
        "schema": MODEL_ID,
        "n_records": len(records),
        "schedule": str(schedule),
        "final_spatial_state": str(final_state),
        "engine_config": str(config_path),
        "complete_configuration_serialized": True,
        "signed_resolved_shear_serialized": True,
        "positive_negative_Burgers_species_serialized": True,
        "local_strength_sigma_cap_serialized": True,
        "constitutive_K_shield_cap_applied": False,
        "same_engine_for_monotonic_and_fatigue": True,
    }
    (root / "v10_2_5_2d_signed_state_trace.json").write_text(
        json.dumps(audit, indent=2)
    )
    return audit


__all__ = [
    "MODEL_ID", "capture_exact_signed_trace", "write_exact_signed_trace", "engine_config"
]
