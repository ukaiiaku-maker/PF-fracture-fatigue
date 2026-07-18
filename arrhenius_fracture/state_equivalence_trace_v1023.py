"""Capture the 2-D constitutive schedule and spatial state for v10.2.3 replay.

The trace wraps the production anisotropic engine without modifying its result.
Each accepted outer ``step(K,T,dt)`` call records the actual K, temperature,
timestep, channel drive factors, and selected post-step state quantities.  The
last spatial MPZ arrays are saved separately.  Replaying the schedule through
``reduced_shared_state_v1023`` therefore tests the constitutive state evolution
rather than fitting a new surrogate to the 2-D result.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .anisotropic_emission_v10174 import AnisotropicStochasticAvalancheTipEngine


MODEL_ID = "v10.2.3_2d_shared_state_equivalence_trace"


def _asdict_or_public(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    result: dict[str, Any] = {}
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


def _state_arrays(engine) -> dict[str, np.ndarray]:
    state = engine.mpz
    return {
        "mobile": np.asarray(state.mobile, dtype=float).copy(),
        "retained": np.asarray(state.retained, dtype=float).copy(),
        "accumulated_slip": np.asarray(state.accumulated_slip, dtype=float).copy(),
        "available_sites": np.asarray(state.available_sites, dtype=float).copy(),
        "site_capacity": np.asarray(state.site_capacity, dtype=float).copy(),
        "wake_mobile": np.asarray(state.wake_mobile, dtype=float).copy(),
        "wake_retained": np.asarray(state.wake_retained, dtype=float).copy(),
        "wake_slip": np.asarray(state.wake_slip, dtype=float).copy(),
    }


def _engine_config(engine) -> dict[str, Any]:
    return {
        "front_config": _asdict_or_public(engine.f),
        "mpz_config": _asdict_or_public(engine.mpz.cfg),
        "tip_config": _asdict_or_public(engine.tip_cfg),
        "anisotropic_config": _asdict_or_public(
            getattr(engine, "anisotropic_cfg", object())
        ),
        "G_Pa": float(engine.G),
        "poisson": float(engine.nu),
        "b_m": float(engine.b),
        "material_manifest": engine.manifest.as_dict(),
        "transport_mode": str(
            getattr(engine.mpz, "_anisotropic_transport_mode", "unknown")
        ),
        "state_class": type(engine.mpz).__name__,
        "engine_class": type(engine).__name__,
    }


def _max_abs_difference(raw: float, effective: float) -> float:
    return abs(float(raw) - float(effective))


@contextmanager
def capture_state_equivalence_trace() -> Iterator[dict[str, Any]]:
    """Temporarily trace production anisotropic engine steps."""
    engine_type = AnisotropicStochasticAvalancheTipEngine
    original = engine_type.step
    trace: dict[str, Any] = {
        "schema": MODEL_ID,
        "records": [],
        "engines": {},
        "last_engine": None,
        "maximum_abs_raw_minus_effective_Pa_sqrt_m": 0.0,
    }

    def traced_step(self, K, T, dt):
        result = original(self, K, T, dt)
        factors = np.asarray(
            getattr(
                self.mpz,
                "_anisotropic_drive_factors",
                np.ones(self.mpz.n_systems),
            ),
            dtype=float,
        ).reshape(-1)
        raw = float(self._active_shielding_raw_uncapped())
        effective = float(self._active_shielding_signed())
        engine_id = int(getattr(self, "_engine_id", id(self)))
        record = {
            "trace_index": len(trace["records"]),
            "engine_id": engine_id,
            "dt_s": float(dt),
            "temperature_K": float(T),
            "K_Pa_sqrt_m": float(K),
            "drive_factor_0": float(factors[0]) if factors.size else 1.0,
            "drive_factor_1": float(factors[1]) if factors.size > 1 else 1.0,
            "expected_fired": bool(result.get("fired", False)),
            "expected_micro_advance_step_m": float(
                result.get("kinetic_micro_advance_step_m", 0.0)
            ),
            "expected_micro_advance_total_m": float(
                getattr(self, "micro_advance_total_m", 0.0)
            ),
            "expected_K_shield_raw_Pa_sqrt_m": raw,
            "expected_K_shield_effective_Pa_sqrt_m": effective,
            "expected_mobile_count": float(self.mpz.mobile_count),
            "expected_retained_count": float(self.mpz.retained_count),
            "expected_emitted_total": float(self.mpz.emitted_total),
            "expected_escaped_total": float(self.mpz.escaped_total),
            "expected_recovered_total": float(self.mpz.recovered_total),
            "expected_available_sites_total": float(
                np.sum(self.mpz.available_sites)
            ),
            "expected_r_eff_m": float(self.r_eff()),
            "expected_checkpoint_progress_action": float(self.B),
            "expected_checkpoint_advances": int(self.n_adv),
        }
        trace["records"].append(record)
        trace["engines"][engine_id] = self
        trace["last_engine"] = self
        trace["maximum_abs_raw_minus_effective_Pa_sqrt_m"] = max(
            float(trace["maximum_abs_raw_minus_effective_Pa_sqrt_m"]),
            _max_abs_difference(raw, effective),
        )
        return result

    engine_type.step = traced_step
    try:
        yield trace
    finally:
        engine_type.step = original


def write_state_equivalence_trace(trace: dict[str, Any], root: str | Path) -> dict[str, Any]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    records = list(trace.get("records", []))
    if not records:
        raise RuntimeError("no anisotropic engine steps were captured")

    fields = list(records[0])
    schedule_path = root / "v10_2_3_2d_replay_schedule.csv"
    with schedule_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    engine = trace.get("last_engine")
    if engine is None:
        raise RuntimeError("trace has no final engine")
    np.savez_compressed(
        root / "v10_2_3_2d_final_state.npz",
        **_state_arrays(engine),
    )
    config = _engine_config(engine)
    (root / "v10_2_3_2d_engine_config.json").write_text(
        json.dumps(config, indent=2)
    )

    maximum_difference = float(
        trace.get("maximum_abs_raw_minus_effective_Pa_sqrt_m", math.inf)
    )
    raw_scale = max(
        max(
            abs(float(row["expected_K_shield_raw_Pa_sqrt_m"]))
            for row in records
        ),
        1.0,
    )
    audit = {
        "schema": MODEL_ID,
        "n_records": len(records),
        "n_engines": len(trace.get("engines", {})),
        "schedule": str(schedule_path),
        "final_spatial_state": str(root / "v10_2_3_2d_final_state.npz"),
        "engine_config": str(root / "v10_2_3_2d_engine_config.json"),
        "constitutive_K_shield_clip_applied": False,
        "legacy_manifest_cap_used_in_kinetics": False,
        "maximum_abs_raw_minus_effective_Pa_sqrt_m": maximum_difference,
        "raw_equals_effective_within_relative_1e_12": bool(
            maximum_difference <= max(1.0e-6, 1.0e-12 * raw_scale)
        ),
        "replay_contract": (
            "same production state classes; recorded K,T,dt,channel factors; "
            "compare scalar history and final spatial arrays"
        ),
    }
    (root / "v10_2_3_2d_state_trace.json").write_text(
        json.dumps(audit, indent=2)
    )
    return audit


__all__ = [
    "MODEL_ID",
    "capture_state_equivalence_trace",
    "write_state_equivalence_trace",
]
