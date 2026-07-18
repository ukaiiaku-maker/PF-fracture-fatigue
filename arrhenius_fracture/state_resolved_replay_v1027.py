"""Exact active-state replay for the v10.2.6/v10.2.7 shared engine."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .material_manifest import MaterialManifest
from .reduced_shared_state_v1025 import state_arrays
from .state_resolved_reduced_campaign_v1027 import StateResolvedProductionConfig

MODEL_ID = "v10.2.7_exact_state_resolved_replay"


def read_signed_schedule(path: str | Path) -> list[dict[str, Any]]:
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
        raise ValueError(f"empty signed schedule: {path}")
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"signed schedule is missing {missing}")
    parsed = []
    for index, row in enumerate(rows):
        parsed.append(
            {
                "row_index": index,
                "dt_s": float(row["dt_s"]),
                "temperature_K": float(row["temperature_K"]),
                "K_Pa_sqrt_m": float(row["K_Pa_sqrt_m"]),
                "drive_factors": np.asarray(
                    [float(row["drive_factor_0"]), float(row["drive_factor_1"])],
                    dtype=float,
                ),
                "tau_signed_Pa": np.asarray(
                    [float(row["tau_signed_0_Pa"]), float(row["tau_signed_1_Pa"])],
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


def replay_state_resolved_trace(
    manifest: MaterialManifest,
    schedule: str | Path | list[dict[str, Any]],
    production: StateResolvedProductionConfig,
) -> dict[str, Any]:
    rows = read_signed_schedule(schedule) if isinstance(schedule, (str, Path)) else list(schedule)
    engine, _drive = production.build_engine(manifest, mode="full")
    history: list[dict[str, Any]] = []
    all_fired = True
    for row in rows:
        factors = np.asarray(row["drive_factors"], dtype=float)
        tau = np.asarray(row["tau_signed_Pa"], dtype=float)
        if factors.shape != (engine.mpz.n_systems,) or tau.shape != (
            engine.mpz.n_systems,
        ):
            raise ValueError("recorded signed drive dimension does not match production systems")
        engine.mpz._anisotropic_drive_factors = factors.copy()
        engine.mpz._anisotropic_tau_signed_Pa = tau.copy()
        engine.mpz._anisotropic_drive_reliable = True
        engine.mpz._anisotropic_drive_serial = int(row.get("row_index", len(history)))
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
                "K_shield_Pa_sqrt_m": float(engine.K_shield()),
                "mobile_count": float(engine.mpz.mobile_count),
                "retained_count": float(engine.mpz.retained_count),
                "source_activations_total": float(
                    engine.mpz.signed_source_activations_total
                ),
                "signed_line_content_total": float(
                    engine.mpz.signed_line_content_emitted_total
                ),
                "state_coordinates": dict(engine._signed_last_state_coordinates),
            }
        )
    return {
        "schema": MODEL_ID,
        "candidate_id": manifest.candidate_id,
        "n_schedule_rows": len(rows),
        "all_fired_flags_match": all_fired,
        "history": history,
        "final_arrays": state_arrays(engine),
        "production_config_parity": engine._v1027_config_parity,
        "constitutive_K_shield_cap_applied": False,
    }


__all__ = ["MODEL_ID", "read_signed_schedule", "replay_state_resolved_trace"]
