#!/usr/bin/env python3
"""Build a v10.2.6 state-resolved signed shielding-kernel family.

Required response CSV columns
-----------------------------
state_id,r_eff_over_r0,opening_strength_fraction,crack_extension_m,
region,system,bin,x_m,burgers_sign,delta_signed_line_content,
K_I_base_Pa_sqrt_m,K_I_perturbed_Pa_sqrt_m,
K_II_base_Pa_sqrt_m,K_II_perturbed_Pa_sqrt_m

Every state/region/system/bin requires both Burgers signs and at least two
nonzero perturbation magnitudes for each sign.  The normalized influence
coefficients must agree across sign and amplitude.  The script then quantifies
state dependence and either records a validated fixed kernel or emits an
interpolated family over the sampled state envelope.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from arrhenius_fracture.signed_kernel_family_v1026 import SCHEMA, STATE_AXES


def _load_normalization(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    required = {
        "normalization_source",
        "activation_to_line_content_by_system",
        "source_capacity_bounds_per_system",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise SystemExit(f"normalization artifact is missing {missing}")
    if payload["normalization_source"] not in {
        "2d_unit_slip_to_line_content",
        "plastic_distortion_burgers_integral",
        "process_zone_geometry_and_line_spacing",
        "front_thickness_source_geometry",
    }:
        raise SystemExit("normalization artifact is not mechanically derived")
    if bool(payload.get("fitted_to_toughness_or_fatigue", False)):
        raise SystemExit("normalization must not be fitted to fracture/fatigue output")
    return payload


def _float(row: dict[str, str], name: str) -> float:
    try:
        value = float(row[name])
    except Exception as exc:
        raise SystemExit(f"invalid {name!r} in response row: {row}") from exc
    if not math.isfinite(value):
        raise SystemExit(f"non-finite {name!r} in response row")
    return value


def _coefficient(row: dict[str, str], mode: str) -> float:
    base = _float(row, f"K_{mode}_base_Pa_sqrt_m")
    perturbed = _float(row, f"K_{mode}_perturbed_Pa_sqrt_m")
    content = _float(row, "delta_signed_line_content")
    return (base - perturbed) / content


def _validate_group(
    key,
    rows: list[dict[str, str]],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[float, float, dict[str, Any]]:
    signs = {int(float(row["burgers_sign"])) for row in rows}
    if signs != {-1, 1}:
        raise SystemExit(f"{key} lacks a complete +/- Burgers pair")
    by_sign: dict[int, list[dict[str, str]]] = {-1: [], 1: []}
    for row in rows:
        sign = int(float(row["burgers_sign"]))
        content = _float(row, "delta_signed_line_content")
        if sign not in {-1, 1} or content == 0.0:
            raise SystemExit(f"{key} contains an invalid signed perturbation")
        if math.copysign(1.0, content) != float(sign):
            raise SystemExit(
                f"{key} delta_signed_line_content sign disagrees with burgers_sign"
            )
        by_sign[sign].append(row)
    for sign in (-1, 1):
        magnitudes = sorted(
            {
                round(abs(_float(row, "delta_signed_line_content")), 15)
                for row in by_sign[sign]
            }
        )
        if len(magnitudes) < 2:
            raise SystemExit(
                f"{key} sign {sign:+d} requires at least two perturbation magnitudes"
            )

    diagnostics: dict[str, Any] = {"key": list(key), "modes": {}}
    outputs = []
    for mode in ("I", "II"):
        values = np.asarray([_coefficient(row, mode) for row in rows], dtype=float)
        reference = float(np.median(values))
        scale = max(float(np.max(np.abs(values))), abs(reference), 1.0)
        allowed = max(float(absolute_tolerance), float(relative_tolerance) * scale)
        maximum_deviation = float(np.max(np.abs(values - reference)))
        if maximum_deviation > allowed:
            raise SystemExit(
                f"{key} mode {mode} fails signed/multi-amplitude linearity: "
                f"max deviation={maximum_deviation:.9e}, allowed={allowed:.9e}"
            )
        sign_means = {
            str(sign): float(
                np.mean([_coefficient(row, mode) for row in by_sign[sign]])
            )
            for sign in (-1, 1)
        }
        sign_difference = abs(sign_means["-1"] - sign_means["1"])
        if sign_difference > allowed:
            raise SystemExit(
                f"{key} mode {mode} fails Burgers antisymmetry after normalization"
            )
        amplitude_checks = []
        for sign in (-1, 1):
            ordered = sorted(
                by_sign[sign],
                key=lambda row: abs(_float(row, "delta_signed_line_content")),
            )
            smallest = ordered[0]
            c0 = _coefficient(smallest, mode)
            n0 = abs(_float(smallest, "delta_signed_line_content"))
            for row in ordered[1:]:
                n = abs(_float(row, "delta_signed_line_content"))
                coefficient = _coefficient(row, mode)
                ratio = coefficient / c0 if abs(c0) > 0.0 else math.nan
                amplitude_checks.append(
                    {
                        "sign": sign,
                        "small_magnitude": n0,
                        "large_magnitude": n,
                        "normalized_coefficient_ratio": ratio,
                    }
                )
        diagnostics["modes"][mode] = {
            "coefficient_mean": float(np.mean(values)),
            "coefficient_median": reference,
            "coefficient_min": float(np.min(values)),
            "coefficient_max": float(np.max(values)),
            "maximum_deviation": maximum_deviation,
            "allowed_deviation": allowed,
            "sign_means": sign_means,
            "sign_mean_difference": sign_difference,
            "amplitude_checks": amplitude_checks,
        }
        outputs.append(float(np.mean(values)))
    return outputs[0], outputs[1], diagnostics


def _state_variation(
    state_rows: list[dict[str, Any]],
    *,
    reference_state_id: str,
    significance_floor_fraction: float,
    absolute_floor: float,
) -> dict[str, Any]:
    reference = next(
        (row for row in state_rows if row["state_id"] == reference_state_id),
        None,
    )
    if reference is None:
        raise SystemExit(f"reference state {reference_state_id!r} is absent")
    report: dict[str, Any] = {"reference_state_id": reference_state_id}
    maxima = []
    for label in ("active_I", "wake_I", "active_II", "wake_II"):
        ref = np.asarray(reference[label], dtype=float)
        scale_max = max(float(np.max(np.abs(ref))) if ref.size else 0.0, 1.0)
        floor = max(float(absolute_floor), float(significance_floor_fraction) * scale_max)
        denominator = np.maximum(np.abs(ref), floor)
        worst = 0.0
        worst_state = reference_state_id
        for row in state_rows:
            value = np.asarray(row[label], dtype=float)
            relative = (
                np.abs(value - ref) / denominator
                if value.size
                else np.zeros_like(value)
            )
            candidate = float(np.max(relative)) if relative.size else 0.0
            if candidate > worst:
                worst = candidate
                worst_state = row["state_id"]
        report[label] = {
            "significance_floor": floor,
            "maximum_relative_variation": worst,
            "worst_state_id": worst_state,
        }
        maxima.append(worst)
    report["maximum_relative_variation_all_modes"] = max(maxima, default=0.0)
    report["maximum_relative_variation_mode_I"] = max(
        report["active_I"]["maximum_relative_variation"],
        report["wake_I"]["maximum_relative_variation"],
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--relative-linearity-tolerance", type=float, default=0.03)
    parser.add_argument(
        "--absolute-linearity-tolerance-Pa-sqrt-m-per-line",
        type=float,
        default=1.0e-9,
    )
    parser.add_argument("--fixed-kernel-tolerance", type=float, default=0.05)
    parser.add_argument("--significance-floor-fraction", type=float, default=1.0e-3)
    parser.add_argument(
        "--absolute-significance-floor-Pa-sqrt-m-per-line",
        type=float,
        default=1.0e-9,
    )
    parser.add_argument("--reference-state-id")
    parser.add_argument("--interpolation-neighbors", type=int, default=8)
    parser.add_argument("--minimum-distinct-r", type=int, default=2)
    parser.add_argument("--minimum-distinct-opening", type=int, default=3)
    parser.add_argument("--minimum-distinct-extension", type=int, default=2)
    parser.add_argument(
        "--authorize-production-parameterization",
        action="store_true",
        help="Set only after independent review of coverage and normalization.",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    if not args.responses.is_file():
        raise SystemExit(f"response table is missing: {args.responses}")
    normalization = _load_normalization(args.normalization)
    with args.responses.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "state_id",
        *STATE_AXES,
        "region",
        "system",
        "bin",
        "x_m",
        "burgers_sign",
        "delta_signed_line_content",
        "K_I_base_Pa_sqrt_m",
        "K_I_perturbed_Pa_sqrt_m",
        "K_II_base_Pa_sqrt_m",
        "K_II_perturbed_Pa_sqrt_m",
    }
    if not rows:
        raise SystemExit("empty signed interaction-integral response table")
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise SystemExit(f"response table is missing {missing}")

    grouped = defaultdict(list)
    state_coordinates: dict[str, tuple[float, float, float]] = {}
    for row in rows:
        state_id = str(row["state_id"]).strip()
        coordinates = tuple(_float(row, name) for name in STATE_AXES)
        existing = state_coordinates.setdefault(state_id, coordinates)
        if not np.allclose(existing, coordinates, rtol=1.0e-12, atol=1.0e-18):
            raise SystemExit(f"state {state_id!r} has inconsistent coordinates")
        region = str(row["region"]).strip().lower()
        if region not in {"active", "wake"}:
            raise SystemExit(f"invalid region {region!r}")
        system = int(row["system"])
        bin_index = int(row["bin"])
        x_m = _float(row, "x_m")
        grouped[(state_id, region, system, bin_index, x_m)].append(row)

    distinct = {
        axis: len({round(coords[index], 14) for coords in state_coordinates.values()})
        for index, axis in enumerate(STATE_AXES)
    }
    required_counts = {
        "r_eff_over_r0": int(args.minimum_distinct_r),
        "opening_strength_fraction": int(args.minimum_distinct_opening),
        "crack_extension_m": int(args.minimum_distinct_extension),
    }
    coverage_passed = all(
        distinct[axis] >= required_counts[axis] for axis in STATE_AXES
    )

    n_systems = max(key[2] for key in grouped) + 1
    state_payloads: list[dict[str, Any]] = []
    linearity_checks = []
    active_x_reference = None
    wake_x_reference = None
    for state_id in sorted(state_coordinates):
        arrays = {}
        coordinates_by_region = {}
        for region in ("active", "wake"):
            keys = [key for key in grouped if key[0] == state_id and key[1] == region]
            if not keys:
                arrays[f"{region}_I"] = np.zeros((n_systems, 0))
                arrays[f"{region}_II"] = np.zeros((n_systems, 0))
                coordinates_by_region[region] = np.zeros(0)
                continue
            n_bins = max(key[3] for key in keys) + 1
            matrix_I = np.full((n_systems, n_bins), np.nan)
            matrix_II = np.full((n_systems, n_bins), np.nan)
            x = np.full(n_bins, np.nan)
            for key in keys:
                _, _, system, bin_index, x_m = key
                H_I, H_II, diagnostic = _validate_group(
                    key,
                    grouped[key],
                    relative_tolerance=float(args.relative_linearity_tolerance),
                    absolute_tolerance=float(
                        args.absolute_linearity_tolerance_Pa_sqrt_m_per_line
                    ),
                )
                matrix_I[system, bin_index] = H_I
                matrix_II[system, bin_index] = H_II
                if math.isfinite(x[bin_index]) and not math.isclose(
                    x[bin_index], x_m, rel_tol=1.0e-12, abs_tol=1.0e-18
                ):
                    raise SystemExit(
                        f"inconsistent x coordinate for {state_id}/{region}/bin {bin_index}"
                    )
                x[bin_index] = x_m
                linearity_checks.append(diagnostic)
            if np.any(~np.isfinite(matrix_I)) or np.any(~np.isfinite(matrix_II)):
                raise SystemExit(f"{state_id}/{region} has an incomplete system/bin matrix")
            if np.any(~np.isfinite(x)):
                raise SystemExit(f"{state_id}/{region} has incomplete coordinates")
            arrays[f"{region}_I"] = matrix_I
            arrays[f"{region}_II"] = matrix_II
            coordinates_by_region[region] = x
        if active_x_reference is None:
            active_x_reference = coordinates_by_region["active"]
            wake_x_reference = coordinates_by_region["wake"]
        elif not np.allclose(
            active_x_reference,
            coordinates_by_region["active"],
            rtol=1.0e-12,
            atol=1.0e-18,
        ) or not np.allclose(
            wake_x_reference,
            coordinates_by_region["wake"],
            rtol=1.0e-12,
            atol=1.0e-18,
        ):
            raise SystemExit("active/wake grids differ across mechanical states")
        coords = state_coordinates[state_id]
        state_payloads.append(
            {
                "state_id": state_id,
                **{name: coords[i] for i, name in enumerate(STATE_AXES)},
                **arrays,
            }
        )

    conversion = np.asarray(
        normalization["activation_to_line_content_by_system"], dtype=float
    )
    bounds = np.asarray(
        normalization["source_capacity_bounds_per_system"], dtype=float
    )
    if conversion.shape != (n_systems,) or bounds.shape != (n_systems, 2):
        raise SystemExit("normalization dimensions do not match response systems")
    if np.any(conversion <= 0.0) or np.any(~np.isfinite(conversion)):
        raise SystemExit("activation-to-line conversion must be positive and finite")

    reference_state_id = args.reference_state_id or state_payloads[0]["state_id"]
    variation = _state_variation(
        state_payloads,
        reference_state_id=reference_state_id,
        significance_floor_fraction=float(args.significance_floor_fraction),
        absolute_floor=float(
            args.absolute_significance_floor_Pa_sqrt_m_per_line
        ),
    )
    fixed_accepted = bool(
        variation["maximum_relative_variation_mode_I"]
        <= float(args.fixed_kernel_tolerance)
    )
    production_allowed = bool(
        args.authorize_production_parameterization and coverage_passed
    )
    if args.authorize_production_parameterization and not coverage_passed:
        raise SystemExit(
            "cannot authorize production parameterization: state-envelope coverage "
            f"is insufficient; distinct={distinct}, required={required_counts}"
        )

    states_json = []
    for row in state_payloads:
        states_json.append(
            {
                "state_id": row["state_id"],
                **{name: row[name] for name in STATE_AXES},
                "active_kernel_I_Pa_sqrt_m_per_signed_line": row[
                    "active_I"
                ].tolist(),
                "wake_kernel_I_Pa_sqrt_m_per_signed_line": row[
                    "wake_I"
                ].tolist(),
                "active_kernel_II_Pa_sqrt_m_per_signed_line": row[
                    "active_II"
                ].tolist(),
                "wake_kernel_II_Pa_sqrt_m_per_signed_line": row[
                    "wake_II"
                ].tolist(),
            }
        )
    payload = {
        "schema": SCHEMA,
        "candidate_independent": True,
        "counts_are_signed_burgers_lines": True,
        "kernel_from_signed_interaction_integral": True,
        "positive_and_negative_perturbations": True,
        "multiple_perturbation_magnitudes": True,
        "multi_amplitude_validation_passed": True,
        "normalization_is_mechanically_derived": True,
        "fitted_attenuation_factor": False,
        "constitutive_K_shield_cap": False,
        "normalization_source": normalization["normalization_source"],
        "normalization_artifact": str(args.normalization.resolve()),
        "unit_response_table": str(args.responses.resolve()),
        "state_axes": list(STATE_AXES),
        "state_coverage": {
            "distinct_values": distinct,
            "required_distinct_values": required_counts,
            "coverage_passed": coverage_passed,
        },
        "active_x_m": np.asarray(active_x_reference).tolist(),
        "wake_x_m": np.asarray(wake_x_reference).tolist(),
        "activation_to_line_content_by_system": conversion.tolist(),
        "source_capacity_bounds_per_system": bounds.tolist(),
        "fixed_kernel_assessment": {
            **variation,
            "tolerance": float(args.fixed_kernel_tolerance),
            "fixed_kernel_accepted": fixed_accepted,
        },
        "interpolation": {
            "method": "fixed_reference" if fixed_accepted else "inverse_distance",
            "neighbors": max(int(args.interpolation_neighbors), 2),
            "power": 2.0,
            "envelope_relative_tolerance": 1.0e-10,
            "extrapolation_allowed": False,
        },
        "production_parameterization_allowed": production_allowed,
        "linearity_relative_tolerance": float(
            args.relative_linearity_tolerance
        ),
        "linearity_absolute_tolerance_Pa_sqrt_m_per_line": float(
            args.absolute_linearity_tolerance_Pa_sqrt_m_per_line
        ),
        "linearity_checks": linearity_checks,
        "states": states_json,
        "derivation_notes": normalization.get("derivation_notes", ""),
        "geometry": normalization.get("geometry", {}),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                "out": str(args.out),
                "n_states": len(states_json),
                "n_systems": n_systems,
                "coverage_passed": coverage_passed,
                "fixed_kernel_accepted": fixed_accepted,
                "maximum_mode_I_state_variation": variation[
                    "maximum_relative_variation_mode_I"
                ],
                "production_parameterization_allowed": production_allowed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
