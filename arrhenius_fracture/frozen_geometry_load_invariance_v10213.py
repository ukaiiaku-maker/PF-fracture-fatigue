"""Frozen-geometry load-invariance validation for active signed kernels.

At fixed crack geometry and internal fields, the linear-elastic influence of a
unit active signed-slip perturbation must be independent of the external load.
v10.2.14 deliberately excludes the scalar wake from this validation because the
current wake state does not preserve two-dimensional line positions.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable

import numpy as np

from .physical_fem_snapshot_v10212 import RESPONSE_COLUMNS, load_snapshot
from .physical_fem_station_responses_v10212 import generate_station_responses

MODEL_ID = "v10.2.14_active_frozen_geometry_load_invariance"
LOAD_RESPONSE_COLUMNS = tuple(RESPONSE_COLUMNS) + (
    "parent_state_id",
    "load_scale",
    "captured_opening_strength_fraction",
    "scaled_opening_proxy",
    "cumulative_crack_path_extension_m",
)


def _scale_token(value: float) -> str:
    return f"{float(value):.8g}".replace("-", "m").replace(".", "p")


def _scaled_snapshot(source: Path, destination: Path, scale: float) -> dict[str, Any]:
    shutil.copytree(source, destination)
    metadata_path = destination / "snapshot.json"
    payload = json.loads(metadata_path.read_text())
    parent_state_id = str(payload["state_id"])
    original_opening = float(payload["opening_strength_fraction"])
    payload["state_id"] = f"{parent_state_id}__load_{_scale_token(scale)}"
    payload["Uy_top_m"] = float(payload["Uy_top_m"]) * float(scale)
    payload["Uy_bot_m"] = float(payload["Uy_bot_m"]) * float(scale)
    payload["opening_strength_fraction"] = min(
        max(original_opening * float(scale), 0.0), 1.0
    )
    payload["frozen_geometry_parent_state_id"] = parent_state_id
    payload["frozen_geometry_load_scale"] = float(scale)
    payload["opening_value_is_linear_load_proxy"] = True
    payload["active_kernel_supported"] = True
    payload["wake_kernel_supported"] = False
    metadata_path.write_text(json.dumps(payload, indent=2))
    return {
        "parent_state_id": parent_state_id,
        "captured_opening_strength_fraction": original_opening,
        "scaled_opening_proxy": payload["opening_strength_fraction"],
    }


def _coefficient(row: dict[str, Any], mode: str) -> float:
    delta = float(row["delta_signed_line_content"])
    if not math.isfinite(delta) or abs(delta) <= 0.0:
        raise ValueError("signed line-content perturbation must be finite and nonzero")
    base = float(row[f"K_{mode}_base_Pa_sqrt_m"])
    perturbed = float(row[f"K_{mode}_perturbed_Pa_sqrt_m"])
    value = (base - perturbed) / delta
    if not math.isfinite(value):
        raise ValueError("non-finite signed shielding coefficient")
    return value


def _validate_coefficients(
    rows: list[dict[str, Any]],
    *,
    linearity_tolerance: float,
    load_invariance_tolerance: float,
    significance_floor_fraction: float,
) -> dict[str, Any]:
    if any(str(row.get("region", "")) != "active" for row in rows):
        raise ValueError("v10.2.14 load invariance accepts active response rows only")
    grouped: dict[tuple[Any, ...], dict[float, list[float]]] = {}
    for row in rows:
        spatial = (
            str(row["region"]), int(row["system"]), int(row["bin"]),
            float(row["x_m"]),
        )
        scale = float(row["load_scale"])
        for mode in ("I", "II"):
            grouped.setdefault((*spatial, mode), {}).setdefault(scale, []).append(
                _coefficient(row, mode)
            )
    if not grouped:
        raise ValueError("no active signed response coefficients were generated")

    scale_means: dict[tuple[Any, ...], dict[float, float]] = {}
    within_load_checks = []
    global_maximum = 0.0
    for key, by_scale in grouped.items():
        means = {}
        for scale, values in sorted(by_scale.items()):
            array = np.asarray(values, dtype=float)
            mean = float(np.mean(array))
            means[scale] = mean
            global_maximum = max(global_maximum, abs(mean), float(np.max(np.abs(array))))
            denominator = max(abs(mean), 1.0e-30)
            relative_spread = float(np.max(np.abs(array - mean)) / denominator)
            within_load_checks.append({
                "region": key[0], "system": key[1], "bin": key[2],
                "x_m": key[3], "mode": key[4], "load_scale": scale,
                "coefficient_mean": mean,
                "maximum_relative_sign_amplitude_spread": relative_spread,
                "passed": relative_spread <= float(linearity_tolerance),
            })
        scale_means[key] = means

    floor = max(global_maximum * float(significance_floor_fraction), 1.0e-12)
    load_checks = []
    for key, means in scale_means.items():
        if len(means) < 3:
            raise ValueError("each spatial coefficient requires at least three load scales")
        values = np.asarray(list(means.values()), dtype=float)
        reference = float(np.median(values))
        relative = np.abs(values - reference) / max(abs(reference), floor)
        worst = float(np.max(relative))
        load_checks.append({
            "region": key[0], "system": key[1], "bin": key[2],
            "x_m": key[3], "mode": key[4],
            "reference_coefficient": reference,
            "coefficients_by_load_scale": {
                f"{scale:.12g}": value for scale, value in sorted(means.items())
            },
            "maximum_relative_load_variation": worst,
            "passed": worst <= float(load_invariance_tolerance),
        })

    within_passed = all(item["passed"] for item in within_load_checks)
    load_passed = all(item["passed"] for item in load_checks)
    return {
        "within_load_sign_amplitude_linearity_passed": within_passed,
        "frozen_geometry_load_invariance_passed": load_passed,
        "linearity_tolerance": float(linearity_tolerance),
        "load_invariance_tolerance": float(load_invariance_tolerance),
        "significance_floor_fraction": float(significance_floor_fraction),
        "significance_floor_Pa_sqrt_m_per_signed_line": floor,
        "maximum_within_load_relative_spread": max(
            (item["maximum_relative_sign_amplitude_spread"] for item in within_load_checks),
            default=math.inf,
        ),
        "maximum_relative_load_variation": max(
            (item["maximum_relative_load_variation"] for item in load_checks),
            default=math.inf,
        ),
        "within_load_checks": within_load_checks,
        "load_invariance_checks": load_checks,
    }


def evaluate_frozen_geometry_load_invariance(
    snapshot_root: str | Path,
    *,
    outroot: str | Path,
    load_scales: Iterable[float] = (0.5, 1.0, 1.5),
    perturbation_magnitudes: Iterable[float] = (0.25, 0.5),
    ribbon_width_m: float | None = None,
    minimum_station_spacing_m: float | None = None,
    linearity_tolerance: float = 0.03,
    load_invariance_tolerance: float = 0.05,
    significance_floor_fraction: float = 1.0e-3,
    minimum_residual_stiffness_fraction: float = 1.0e-3,
) -> dict[str, Any]:
    source = Path(snapshot_root).expanduser().resolve()
    if not (source / "snapshot.json").is_file():
        raise FileNotFoundError(source / "snapshot.json")
    out = Path(outroot)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)

    scales = sorted({float(value) for value in load_scales})
    if len(scales) < 3 or 1.0 not in scales:
        raise ValueError("load-invariance validation requires at least three scales including 1.0")
    if any(not math.isfinite(value) or value <= 0.0 for value in scales):
        raise ValueError("load scales must be positive and finite")
    if not (0.0 < float(linearity_tolerance) < 1.0):
        raise ValueError("linearity tolerance must lie in (0,1)")
    if not (0.0 < float(load_invariance_tolerance) < 1.0):
        raise ValueError("load-invariance tolerance must lie in (0,1)")
    if not (0.0 < float(significance_floor_fraction) < 1.0):
        raise ValueError("significance-floor fraction must lie in (0,1)")

    original = load_snapshot(source)
    original_meta = original["metadata"]
    combined_rows: list[dict[str, Any]] = []
    generated = []
    with tempfile.TemporaryDirectory(prefix="v10214_active_frozen_geometry_") as temporary:
        temporary_root = Path(temporary)
        for scale in scales:
            scaled_root = temporary_root / f"snapshot_{_scale_token(scale)}"
            metadata = _scaled_snapshot(source, scaled_root, scale)
            response_path = out / f"active_station_responses_load_{_scale_token(scale)}.csv"
            report = generate_station_responses(
                scaled_root,
                out_csv=response_path,
                magnitudes=perturbation_magnitudes,
                ribbon_width_m=ribbon_width_m,
                minimum_station_spacing_m=minimum_station_spacing_m,
                minimum_residual_stiffness_fraction=(
                    minimum_residual_stiffness_fraction
                ),
            )
            if report.get("wake_shielding_supported") is not False:
                raise RuntimeError("active-only evaluator unexpectedly enabled wake shielding")
            rows = list(csv.DictReader(response_path.open(newline="")))
            for row in rows:
                row.update({
                    "parent_state_id": metadata["parent_state_id"],
                    "load_scale": float(scale),
                    "captured_opening_strength_fraction": metadata[
                        "captured_opening_strength_fraction"
                    ],
                    "scaled_opening_proxy": metadata["scaled_opening_proxy"],
                    "cumulative_crack_path_extension_m": float(
                        original_meta.crack_extension_m
                    ),
                })
            combined_rows.extend(rows)
            generated.append({
                "load_scale": scale,
                "responses": str(response_path.resolve()),
                "audit": str(response_path.with_suffix(".audit.json").resolve()),
                "response_rows": len(rows),
                "station_report": report,
            })

    combined_path = out / "active_frozen_geometry_load_sweep_responses.csv"
    with combined_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOAD_RESPONSE_COLUMNS)
        writer.writeheader()
        writer.writerows(combined_rows)

    checks = _validate_coefficients(
        combined_rows,
        linearity_tolerance=float(linearity_tolerance),
        load_invariance_tolerance=float(load_invariance_tolerance),
        significance_floor_fraction=float(significance_floor_fraction),
    )
    passed = bool(
        checks["within_load_sign_amplitude_linearity_passed"]
        and checks["frozen_geometry_load_invariance_passed"]
    )
    payload = {
        "schema": MODEL_ID,
        "snapshot": str(source),
        "parent_state_id": original_meta.state_id,
        "cumulative_crack_path_extension_m": float(original_meta.crack_extension_m),
        "captured_opening_strength_fraction": float(
            original_meta.opening_strength_fraction
        ),
        "load_scales": scales,
        "responses": str(combined_path.resolve()),
        "generated_load_cases": generated,
        "fixed_crack_geometry": True,
        "fixed_internal_fields": True,
        "opening_is_validation_coordinate": True,
        "opening_is_production_interpolation_axis": False,
        "active_kernel_mechanically_measured": True,
        "wake_kernel_mechanically_measured": False,
        "wake_shielding_supported": False,
        "minimum_residual_stiffness_fraction": float(
            minimum_residual_stiffness_fraction
        ),
        "checks": checks,
        "load_invariance_passed": passed,
        "production_parameterization_allowed": False,
    }
    (out / "frozen_geometry_load_invariance.json").write_text(
        json.dumps(payload, indent=2)
    )
    if not passed:
        raise RuntimeError(
            "active frozen-geometry signed shielding response is not load invariant; "
            "inspect the generated audit before building a production atlas"
        )
    return payload


__all__ = [
    "MODEL_ID",
    "LOAD_RESPONSE_COLUMNS",
    "evaluate_frozen_geometry_load_invariance",
]
