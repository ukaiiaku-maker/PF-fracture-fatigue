#!/usr/bin/env python3
"""Build a signed shielding kernel from 2-D unit perturbation responses.

Input CSV columns:
  region, system, bin, x_m, burgers_sign,
  delta_K_tip_Pa_sqrt_m, delta_signed_line_content

Each active/wake channel-bin must contain both Burgers signs.  Dividing the 2-D
response by signed line content should produce the same coefficient for the two
signs; violation of that antisymmetry/linearity check aborts generation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

SCHEMA = "v10.2.5_2d_unit_signed_shielding_kernel"


def _load_normalization(path: Path) -> dict:
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
        "process_zone_geometry_and_line_spacing",
        "front_thickness_source_geometry",
    }:
        raise SystemExit("normalization artifact is not mechanically derived")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--relative-linearity-tolerance", type=float, default=0.02)
    parser.add_argument("--absolute-linearity-tolerance-Pa-sqrt-m", type=float, default=1.0e-9)
    parser.add_argument(
        "--kernel-source",
        choices=(
            "2d_unit_signed_dislocation_perturbation",
            "2d_unit_signed_slip_perturbation",
        ),
        required=True,
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    normalization = _load_normalization(args.normalization)
    with args.responses.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "region", "system", "bin", "x_m", "burgers_sign",
        "delta_K_tip_Pa_sqrt_m", "delta_signed_line_content",
    }
    if not rows:
        raise SystemExit("empty 2-D unit-response table")
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise SystemExit(f"response table is missing {missing}")

    groups = defaultdict(list)
    for row in rows:
        region = row["region"].strip().lower()
        if region not in {"active", "wake"}:
            raise SystemExit(f"invalid region {region!r}")
        system = int(row["system"])
        bin_index = int(row["bin"])
        x_m = float(row["x_m"])
        sign = int(float(row["burgers_sign"]))
        content = float(row["delta_signed_line_content"])
        delta_K = float(row["delta_K_tip_Pa_sqrt_m"])
        if sign not in {-1, 1} or not math.isfinite(content) or content == 0.0:
            raise SystemExit("each unit response requires a nonzero signed line content and sign +/-1")
        if math.copysign(1.0, content) != float(sign):
            raise SystemExit("delta_signed_line_content sign disagrees with burgers_sign")
        coefficient = delta_K / content
        groups[(region, system, bin_index, x_m)].append((sign, coefficient))

    n_systems = max(key[1] for key in groups) + 1
    region_bins = {}
    diagnostics = []
    for region in ("active", "wake"):
        keys = [key for key in groups if key[0] == region]
        if not keys:
            region_bins[region] = (np.zeros((n_systems, 0)), np.zeros(0))
            continue
        n_bins = max(key[2] for key in keys) + 1
        kernel = np.full((n_systems, n_bins), np.nan)
        x = np.full(n_bins, np.nan)
        for key in keys:
            _, system, bin_index, x_m = key
            values = groups[key]
            signs = {item[0] for item in values}
            if signs != {-1, 1}:
                raise SystemExit(f"{key} lacks a +/- Burgers response pair")
            by_sign = {
                sign: float(np.mean([value for s, value in values if s == sign]))
                for sign in (-1, 1)
            }
            scale = max(abs(by_sign[-1]), abs(by_sign[1]), 1.0)
            difference = abs(by_sign[-1] - by_sign[1])
            allowed = max(
                args.absolute_linearity_tolerance_Pa_sqrt_m,
                args.relative_linearity_tolerance * scale,
            )
            if difference > allowed:
                raise SystemExit(
                    f"signed response is not antisymmetric/linear for {key}: "
                    f"H-={by_sign[-1]:.9e}, H+={by_sign[1]:.9e}"
                )
            kernel[system, bin_index] = 0.5 * (by_sign[-1] + by_sign[1])
            if math.isfinite(x[bin_index]) and not math.isclose(
                x[bin_index], x_m, rel_tol=1.0e-12, abs_tol=1.0e-18
            ):
                raise SystemExit(f"inconsistent x coordinate for {region} bin {bin_index}")
            x[bin_index] = x_m
            diagnostics.append({
                "region": region,
                "system": system,
                "bin": bin_index,
                "H_negative": by_sign[-1],
                "H_positive": by_sign[1],
                "difference": difference,
                "allowed": allowed,
            })
        if np.any(~np.isfinite(kernel)) or np.any(~np.isfinite(x)):
            raise SystemExit(f"{region} response matrix has missing system/bin entries")
        region_bins[region] = (kernel, x)

    active, active_x = region_bins["active"]
    wake, wake_x = region_bins["wake"]
    conversion = np.asarray(
        normalization["activation_to_line_content_by_system"], dtype=float
    )
    bounds = np.asarray(normalization["source_capacity_bounds_per_system"], dtype=float)
    if conversion.shape != (n_systems,) or bounds.shape != (n_systems, 2):
        raise SystemExit("normalization dimensions do not match the response systems")

    payload = {
        "schema": SCHEMA,
        "candidate_independent": True,
        "counts_are_signed_burgers_lines": True,
        "kernel_from_2d_unit_signed_perturbations": True,
        "normalization_is_mechanically_derived": True,
        "fitted_attenuation_factor": False,
        "kernel_source": args.kernel_source,
        "normalization_source": normalization["normalization_source"],
        "normalization_artifact": str(args.normalization.resolve()),
        "unit_response_table": str(args.responses.resolve()),
        "active_x_m": active_x.tolist(),
        "wake_x_m": wake_x.tolist(),
        "active_kernel_Pa_sqrt_m_per_signed_line": active.tolist(),
        "wake_kernel_Pa_sqrt_m_per_signed_line": wake.tolist(),
        "activation_to_line_content_by_system": conversion.tolist(),
        "source_capacity_bounds_per_system": bounds.tolist(),
        "linearity_relative_tolerance": float(args.relative_linearity_tolerance),
        "linearity_absolute_tolerance_Pa_sqrt_m": float(args.absolute_linearity_tolerance_Pa_sqrt_m),
        "linearity_checks": diagnostics,
        "geometry": normalization.get("geometry", {}),
        "derivation_notes": normalization.get("derivation_notes", ""),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "out": str(args.out),
        "n_systems": n_systems,
        "active_bins": int(active.shape[1]),
        "wake_bins": int(wake.shape[1]),
        "active_kernel_min": float(np.min(active)),
        "active_kernel_max": float(np.max(active)),
        "contains_shielding_and_antishielding": bool(np.min(active) < 0.0 < np.max(active)),
    }, indent=2))


if __name__ == "__main__":
    main()
