"""Validate measured signed stations and project them onto the reduced MPZ grid.

The v10.2.12 measured active+wake schema remains supported.  v10.2.14 adds a
fail-closed active-only schema: active curves are measured and cross-validated,
while every wake coefficient is generated as an explicit exact zero because the
current scalar wake state cannot represent two-dimensional signed line positions.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .physical_fem_snapshot_v10212 import RESPONSE_COLUMNS

MODEL_ID = "v10.2.14_measured_active_station_to_mpz_grid_projection"
STATION_SCHEMA = "v10.2.12_fem_resolved_signed_spatial_station_responses"
ACTIVE_ONLY_STATION_SCHEMA = (
    "v10.2.14_production_mesh_active_signed_spatial_station_responses"
)
ACCEPTED_STATION_SCHEMAS = {STATION_SCHEMA, ACTIVE_ONLY_STATION_SCHEMA}


def _finite(row: dict[str, str], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name}")
    return value


def _coefficient(row: dict[str, str], mode: str) -> float:
    content = _finite(row, "delta_signed_line_content")
    if content == 0.0:
        raise ValueError("signed line content must be nonzero")
    return (
        _finite(row, f"K_{mode}_base_Pa_sqrt_m")
        - _finite(row, f"K_{mode}_perturbed_Pa_sqrt_m")
    ) / content


def _validate_measured_group(
    key: tuple,
    rows: list[dict[str, str]],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[float, float, dict[str, Any]]:
    by_sign = {-1: [], 1: []}
    for row in rows:
        sign = int(float(row["burgers_sign"]))
        content = _finite(row, "delta_signed_line_content")
        if sign not in by_sign or math.copysign(1.0, content) != float(sign):
            raise ValueError(f"{key} has inconsistent Burgers sign/content")
        by_sign[sign].append(row)
    if any(not by_sign[sign] for sign in (-1, 1)):
        raise ValueError(f"{key} lacks a complete positive/negative pair")
    for sign in (-1, 1):
        magnitudes = {
            round(abs(_finite(row, "delta_signed_line_content")), 15)
            for row in by_sign[sign]
        }
        if len(magnitudes) < 2:
            raise ValueError(f"{key} sign {sign:+d} requires at least two amplitudes")

    outputs = []
    diagnostics = {"key": list(key), "modes": {}}
    for mode in ("I", "II"):
        values = np.asarray([_coefficient(row, mode) for row in rows], dtype=float)
        reference = float(np.median(values))
        scale = max(float(np.max(np.abs(values))), abs(reference), 1.0)
        allowed = max(float(absolute_tolerance), float(relative_tolerance) * scale)
        deviation = float(np.max(np.abs(values - reference)))
        if deviation > allowed:
            raise ValueError(
                f"{key} mode {mode} fails signed/multi-amplitude linearity: "
                f"deviation={deviation:.9e}, allowed={allowed:.9e}"
            )
        sign_means = {
            str(sign): float(
                np.mean([_coefficient(row, mode) for row in by_sign[sign]])
            )
            for sign in (-1, 1)
        }
        sign_difference = abs(sign_means["-1"] - sign_means["1"])
        if sign_difference > allowed:
            raise ValueError(f"{key} mode {mode} fails normalized Burgers antisymmetry")
        diagnostics["modes"][mode] = {
            "coefficient_mean": float(np.mean(values)),
            "coefficient_median": reference,
            "maximum_deviation": deviation,
            "allowed_deviation": allowed,
            "sign_means": sign_means,
            "sign_mean_difference": sign_difference,
        }
        outputs.append(float(np.mean(values)))
    return outputs[0], outputs[1], diagnostics


def _leave_one_out_error(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if x.size < 3:
        return {
            "available": False,
            "maximum_relative_error": math.inf,
            "reason": "at least three measured stations are required",
        }
    errors = []
    scale_floor = max(float(np.max(np.abs(y))) * 1.0e-3, 1.0e-12)
    for index in range(1, x.size - 1):
        keep = np.ones(x.size, dtype=bool)
        keep[index] = False
        predicted = float(np.interp(x[index], x[keep], y[keep]))
        error = abs(predicted - float(y[index])) / max(
            abs(float(y[index])), scale_floor
        )
        errors.append(
            {
                "station_index": int(index),
                "x_m": float(x[index]),
                "measured": float(y[index]),
                "predicted": predicted,
                "relative_error": float(error),
            }
        )
    return {
        "available": True,
        "maximum_relative_error": max(
            (row["relative_error"] for row in errors), default=0.0
        ),
        "checks": errors,
    }


def _append_projected_rows(
    expanded_rows: list[dict[str, Any]],
    *,
    state_id: str,
    coordinates: tuple[float, float, float],
    region: str,
    system: int,
    full_x: np.ndarray,
    projected_I: np.ndarray,
    projected_II: np.ndarray,
    magnitudes: list[float],
    base_I: float,
    base_II: float,
    interaction_schema: str,
    ribbon_width_m: float,
) -> None:
    for bin_index, x_m in enumerate(full_x):
        for sign in (-1, 1):
            for magnitude in magnitudes:
                content = float(sign) * float(magnitude)
                expanded_rows.append(
                    {
                        "state_id": state_id,
                        "r_eff_over_r0": coordinates[0],
                        "opening_strength_fraction": coordinates[1],
                        "crack_extension_m": coordinates[2],
                        "region": region,
                        "system": system,
                        "bin": bin_index,
                        "x_m": float(x_m),
                        "burgers_sign": sign,
                        "delta_signed_line_content": content,
                        "K_I_base_Pa_sqrt_m": base_I,
                        "K_I_perturbed_Pa_sqrt_m": (
                            base_I - float(projected_I[bin_index]) * content
                        ),
                        "K_II_base_Pa_sqrt_m": base_II,
                        "K_II_perturbed_Pa_sqrt_m": (
                            base_II - float(projected_II[bin_index]) * content
                        ),
                        "interaction_integral_schema": interaction_schema,
                        "ribbon_width_m": ribbon_width_m,
                        "mesh_area_ratio": 1.0,
                    }
                )


def expand_station_response_files(
    paths: Iterable[str | Path],
    *,
    relative_linearity_tolerance: float = 0.03,
    absolute_linearity_tolerance: float = 1.0e-9,
    spatial_cross_validation_tolerance: float = 0.10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    physical_inputs = []
    all_rows: list[dict[str, str]] = []
    state_audits: dict[str, dict[str, Any]] = {}
    state_active_only: dict[str, bool] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        audit_path = path.with_suffix(".audit.json")
        if not audit_path.is_file():
            raise FileNotFoundError(audit_path)
        audit = json.loads(audit_path.read_text())
        schema = str(audit.get("schema", ""))
        if schema not in ACCEPTED_STATION_SCHEMAS:
            raise ValueError(
                f"{audit_path} must use one of {sorted(ACCEPTED_STATION_SCHEMAS)}"
            )
        if audit.get("physical_fem_responses_generated") is not True:
            raise ValueError(f"{audit_path} is not a physical FEM response audit")
        if audit.get("responses_are_measured_stations_not_full_grid") is not True:
            raise ValueError(f"{audit_path} does not distinguish measured stations")
        active_only = bool(
            schema == ACTIVE_ONLY_STATION_SCHEMA
            or (
                audit.get("active_kernel_mechanically_measured") is True
                and audit.get("wake_kernel_mechanically_measured") is False
                and audit.get("wake_shielding_supported") is False
            )
        )
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            missing = sorted(set(RESPONSE_COLUMNS).difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing response columns {missing}")
        if not rows:
            raise ValueError(f"{path} is empty")
        if active_only and any(
            str(row.get("region", "")).strip().lower() != "active"
            for row in rows
        ):
            raise ValueError(f"{path} active-only audit contains non-active rows")
        state_id = str(audit["state_id"])
        if state_id in state_audits:
            raise ValueError(f"duplicate physical response file for state {state_id}")
        state_audits[state_id] = audit
        state_active_only[state_id] = active_only
        all_rows.extend(rows)
        physical_inputs.append(
            {"path": str(path.resolve()), "audit": audit, "active_only": active_only}
        )

    grouped: dict[tuple, list[dict[str, str]]] = defaultdict(list)
    state_coordinates: dict[str, tuple[float, float, float]] = {}
    for row in all_rows:
        state_id = str(row["state_id"])
        coordinates = (
            _finite(row, "r_eff_over_r0"),
            _finite(row, "opening_strength_fraction"),
            _finite(row, "crack_extension_m"),
        )
        old = state_coordinates.setdefault(state_id, coordinates)
        if not np.allclose(old, coordinates, rtol=1.0e-12, atol=1.0e-18):
            raise ValueError(f"state {state_id} has inconsistent coordinates")
        key = (
            state_id,
            str(row["region"]).strip().lower(),
            int(row["system"]),
            int(row["bin"]),
            _finite(row, "x_m"),
        )
        grouped[key].append(row)

    measured_coefficients: dict[
        tuple[str, str, int], list[tuple[int, float, float, float]]
    ] = defaultdict(list)
    linearity = []
    base_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    magnitudes_by_state: dict[str, set[float]] = defaultdict(set)
    interaction_schema_by_state: dict[str, str] = {}
    for key, rows in grouped.items():
        state_id, region, system, bin_index, x_m = key
        H_I, H_II, diagnostic = _validate_measured_group(
            key,
            rows,
            relative_tolerance=relative_linearity_tolerance,
            absolute_tolerance=absolute_linearity_tolerance,
        )
        measured_coefficients[(state_id, region, system)].append(
            (bin_index, x_m, H_I, H_II)
        )
        linearity.append(diagnostic)
        for row in rows:
            base_values[(state_id, "I")].append(
                _finite(row, "K_I_base_Pa_sqrt_m")
            )
            base_values[(state_id, "II")].append(
                _finite(row, "K_II_base_Pa_sqrt_m")
            )
            magnitudes_by_state[state_id].add(
                abs(_finite(row, "delta_signed_line_content"))
            )
            schema = str(row["interaction_integral_schema"])
            old_schema = interaction_schema_by_state.setdefault(state_id, schema)
            if old_schema != schema:
                raise ValueError(
                    f"state {state_id} contains multiple interaction schemas"
                )

    expanded_rows: list[dict[str, Any]] = []
    projection_checks = []
    all_cv_available = True
    worst_cv_error = 0.0
    forced_zero_wake_states = []
    for state_id, audit in state_audits.items():
        grids = {
            "active": np.asarray(audit["full_active_grid_x_m"], dtype=float),
            "wake": np.asarray(audit["full_wake_grid_x_m"], dtype=float),
        }
        systems = sorted(
            {key[2] for key in measured_coefficients if key[0] == state_id}
        )
        if not systems:
            raise ValueError(f"state {state_id} contains no measured systems")
        coordinates = state_coordinates[state_id]
        magnitudes = sorted(magnitudes_by_state[state_id])
        base_I = float(np.median(base_values[(state_id, "I")]))
        base_II = float(np.median(base_values[(state_id, "II")]))
        interaction_schema = interaction_schema_by_state[state_id]
        active_only = state_active_only[state_id]
        for region, full_x in grids.items():
            if full_x.size == 0:
                continue
            for system in systems:
                if region == "wake" and active_only:
                    projected_I = np.zeros(full_x.size, dtype=float)
                    projected_II = np.zeros(full_x.size, dtype=float)
                    projection_checks.append(
                        {
                            "state_id": state_id,
                            "region": "wake",
                            "system": system,
                            "measured_bins": [],
                            "measured_x_m": [],
                            "full_grid_count": int(full_x.size),
                            "cross_validation_available": False,
                            "cross_validation_required": False,
                            "maximum_relative_cross_validation_error": 0.0,
                            "wake_kernel_forced_zero": True,
                            "reason": (
                                "scalar wake state lacks two-dimensional signed "
                                "line positions"
                            ),
                        }
                    )
                    _append_projected_rows(
                        expanded_rows,
                        state_id=state_id,
                        coordinates=coordinates,
                        region=region,
                        system=system,
                        full_x=full_x,
                        projected_I=projected_I,
                        projected_II=projected_II,
                        magnitudes=magnitudes,
                        base_I=base_I,
                        base_II=base_II,
                        interaction_schema=interaction_schema,
                        ribbon_width_m=float(audit["ribbon_width_m"]),
                    )
                    if state_id not in forced_zero_wake_states:
                        forced_zero_wake_states.append(state_id)
                    continue

                stations = sorted(
                    measured_coefficients[(state_id, region, system)]
                )
                if len(stations) < 2:
                    raise ValueError(
                        f"state {state_id} {region} system {system} requires at "
                        "least two measured stations"
                    )
                bins = np.asarray([row[0] for row in stations], dtype=int)
                x = np.asarray([row[1] for row in stations], dtype=float)
                H_I = np.asarray([row[2] for row in stations], dtype=float)
                H_II = np.asarray([row[3] for row in stations], dtype=float)
                if bins[0] != 0 or bins[-1] != full_x.size - 1:
                    raise ValueError(
                        f"state {state_id} {region} system {system} lacks "
                        "endpoint station coverage"
                    )
                if np.any(np.diff(x) <= 0.0):
                    raise ValueError(
                        "measured station coordinates must be strictly increasing"
                    )
                projected_I = np.interp(full_x, x, H_I)
                projected_II = np.interp(full_x, x, H_II)
                cv_I = _leave_one_out_error(x, H_I)
                cv_II = _leave_one_out_error(x, H_II)
                available = bool(cv_I["available"] and cv_II["available"])
                maximum = max(
                    float(cv_I["maximum_relative_error"]),
                    float(cv_II["maximum_relative_error"]),
                )
                all_cv_available = all_cv_available and available
                worst_cv_error = max(worst_cv_error, maximum)
                projection_checks.append(
                    {
                        "state_id": state_id,
                        "region": region,
                        "system": system,
                        "measured_bins": bins.tolist(),
                        "measured_x_m": x.tolist(),
                        "full_grid_count": int(full_x.size),
                        "mode_I_leave_one_out": cv_I,
                        "mode_II_leave_one_out": cv_II,
                        "cross_validation_available": available,
                        "cross_validation_required": True,
                        "maximum_relative_cross_validation_error": maximum,
                    }
                )
                _append_projected_rows(
                    expanded_rows,
                    state_id=state_id,
                    coordinates=coordinates,
                    region=region,
                    system=system,
                    full_x=full_x,
                    projected_I=projected_I,
                    projected_II=projected_II,
                    magnitudes=magnitudes,
                    base_I=base_I,
                    base_II=base_II,
                    interaction_schema=interaction_schema,
                    ribbon_width_m=float(audit["ribbon_width_m"]),
                )

    report = {
        "schema": MODEL_ID,
        "accepted_station_schemas": sorted(ACCEPTED_STATION_SCHEMAS),
        "physical_input_count": len(physical_inputs),
        "physical_measured_row_count": len(all_rows),
        "projected_full_grid_row_count": len(expanded_rows),
        "physical_linearity_checks": linearity,
        "projection_checks": projection_checks,
        "piecewise_linear_spatial_projection": True,
        "subelement_rows_claimed_as_direct_fem": False,
        "all_measured_curves_have_leave_one_out_validation": all_cv_available,
        "all_curves_have_leave_one_out_validation": all_cv_available,
        "maximum_relative_spatial_cross_validation_error": worst_cv_error,
        "spatial_cross_validation_tolerance": float(
            spatial_cross_validation_tolerance
        ),
        "spatial_cross_validation_passed": bool(
            all_cv_available
            and worst_cv_error <= float(spatial_cross_validation_tolerance)
        ),
        "active_kernel_mechanically_measured": all(
            state_active_only.values()
        ) if state_active_only else False,
        "wake_kernel_mechanically_measured": not any(
            state_active_only.values()
        ) if state_active_only else False,
        "wake_shielding_supported": not any(
            state_active_only.values()
        ) if state_active_only else False,
        "wake_kernel_forced_zero": bool(forced_zero_wake_states),
        "forced_zero_wake_state_ids": sorted(forced_zero_wake_states),
    }
    return expanded_rows, physical_inputs, report


__all__ = [
    "MODEL_ID",
    "STATION_SCHEMA",
    "ACTIVE_ONLY_STATION_SCHEMA",
    "ACCEPTED_STATION_SCHEMAS",
    "expand_station_response_files",
]
