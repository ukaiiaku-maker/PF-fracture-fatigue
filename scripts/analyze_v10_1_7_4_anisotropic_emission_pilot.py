#!/usr/bin/env python3
"""Analyze the v10.1.7.4 anisotropic-emission pilot."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _records(case_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads(
        (case_dir / "kinetic_tip_cell_audit_v101.json").read_text()
    )
    return list(payload.get("records", []))


def _event_curve(case_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    fired = [row for row in _records(case_dir) if bool(row.get("fired", False))]
    if not fired:
        raise RuntimeError(f"no fired records in {case_dir}")
    extension = np.asarray(
        [
            float(
                row.get(
                    "kinetic_micro_advance_total_m",
                    row.get("micro_advance_total_m", 0.0),
                )
            )
            * 1.0e6
            for row in fired
        ],
        dtype=float,
    )
    extension = np.maximum(extension - extension[0], 0.0)
    toughness = np.asarray(
        [float(row["K_Pa_sqrt_m"]) / 1.0e6 for row in fired],
        dtype=float,
    )
    return extension, toughness


def _interp(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x = np.asarray(x)[order]
    y = np.asarray(y)[order]
    keep = np.concatenate(([True], np.diff(x) > 1.0e-12))
    return np.interp(grid, x[keep], y[keep])


def _anisotropic_record_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        row for row in records if "anisotropic_drive_reliable" in row
    ]
    if not selected:
        return {
            "record_count": 0,
            "reliable_fraction": 0.0,
            "nonzero_drive_fraction": 0.0,
            "mean_channel_factor_difference": math.nan,
            "maximum_channel_factor": math.nan,
            "post_hazard_weighting_count": 0,
        }

    reliable = [
        bool(row.get("anisotropic_drive_reliable", False))
        for row in selected
    ]
    factors = []
    post_count = 0
    for row in selected:
        values = np.asarray(
            row.get("anisotropic_drive_factors", []), dtype=float
        )
        if values.size >= 2 and np.all(np.isfinite(values[:2])):
            factors.append(values[:2])
        if bool(row.get("anisotropic_post_hazard_weighting_applied", False)):
            post_count += 1

    matrix = np.vstack(factors) if factors else np.zeros((0, 2))
    nonzero = (
        np.any(matrix > 1.0e-12, axis=1)
        if matrix.size
        else np.zeros(0, dtype=bool)
    )
    return {
        "record_count": len(selected),
        "reliable_fraction": float(np.mean(reliable)),
        "nonzero_drive_fraction": (
            float(np.mean(nonzero)) if nonzero.size else 0.0
        ),
        "mean_channel_factor_difference": (
            float(np.mean(np.abs(matrix[:, 0] - matrix[:, 1])))
            if matrix.size
            else math.nan
        ),
        "maximum_channel_factor": (
            float(np.max(matrix)) if matrix.size else math.nan
        ),
        "post_hazard_weighting_count": int(post_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--theta", type=float, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    tag = f"T{args.temperature:g}_th{args.theta:g}"
    scalar_dir = root / "scalar_reference" / tag
    fixed_dir = root / "fixed_original" / tag
    segmented_dir = root / "segmented_deterministic" / tag
    stochastic_dirs = [
        root / "stochastic_avalanche" / f"seed_{seed}" / tag
        for seed in args.seeds
    ]

    scalar_x, scalar_k = _event_curve(scalar_dir)
    fixed_x, fixed_k = _event_curve(fixed_dir)
    segmented_x, segmented_k = _event_curve(segmented_dir)

    common_max = min(
        float(np.max(scalar_x)),
        float(np.max(fixed_x)),
        float(np.max(segmented_x)),
    )
    grid = np.linspace(0.0, common_max, max(25, int(common_max / 5.0) + 1))
    scalar_grid = _interp(scalar_x, scalar_k, grid)
    fixed_grid = _interp(fixed_x, fixed_k, grid)
    segmented_grid = _interp(segmented_x, segmented_k, grid)

    fixed_range = max(float(np.ptp(fixed_grid)), 1.0e-12)
    wrapper_rms = float(np.sqrt(np.mean((segmented_grid - fixed_grid) ** 2)))
    wrapper_norm = 100.0 * wrapper_rms / fixed_range
    scalar_delta = float(np.mean(fixed_grid - scalar_grid))
    scalar_delta_pct = (
        100.0 * scalar_delta / max(float(np.mean(scalar_grid)), 1.0e-12)
    )

    all_anisotropic_records = _records(fixed_dir) + _records(segmented_dir)
    for directory in stochastic_dirs:
        all_anisotropic_records.extend(_records(directory))
    metrics = _anisotropic_record_metrics(all_anisotropic_records)

    old_assessment_path = root / "stochastic_avalanche_assessment.json"
    stochastic_assessment = (
        json.loads(old_assessment_path.read_text())
        if old_assessment_path.exists()
        else {}
    )

    assessment = {
        "schema": "v10.1.7.4_anisotropic_emission_pilot",
        "temperature_K": args.temperature,
        "crystal_theta_deg": args.theta,
        "n_stochastic_seeds": len(args.seeds),
        "anisotropic_wrapper_rms_MPa_sqrt_m": wrapper_rms,
        "anisotropic_wrapper_normalized_rms_percent": wrapper_norm,
        "anisotropic_mean_shift_from_scalar_MPa_sqrt_m": scalar_delta,
        "anisotropic_mean_shift_from_scalar_percent": scalar_delta_pct,
        **metrics,
        "deterministic_geometry_equivalence_pass": wrapper_norm <= 0.1,
        "all_tensor_drives_reliable": metrics["reliable_fraction"] == 1.0,
        "nonzero_channel_drive_observed": metrics["nonzero_drive_fraction"] > 0.0,
        "no_post_hazard_directional_weighting": (
            metrics["post_hazard_weighting_count"] == 0
        ),
        "stochastic_avalanche_pilot_pass": bool(
            stochastic_assessment.get("pilot_pass", False)
        ),
        "stochastic_assessment": stochastic_assessment,
    }
    assessment["pilot_pass"] = all(
        [
            assessment["deterministic_geometry_equivalence_pass"],
            assessment["all_tensor_drives_reliable"],
            assessment["nonzero_channel_drive_observed"],
            assessment["no_post_hazard_directional_weighting"],
        ]
    )

    (root / "anisotropic_emission_assessment.json").write_text(
        json.dumps(assessment, indent=2)
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.plot(scalar_x, scalar_k, linestyle="--", linewidth=2.0,
            label="scalar emission reference")
    ax.plot(fixed_x, fixed_k, linewidth=2.2,
            label="anisotropic deterministic")
    ax.plot(segmented_x, segmented_k, linewidth=1.3, linestyle=":",
            label="anisotropic wrapped deterministic")
    ax.set_xlabel("Crack-path extension after initiation (µm)")
    ax.set_ylabel(r"$K$ at advance (MPa$\sqrt{\mathrm{m}}$)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "anisotropic_vs_scalar_R_curve.png", dpi=220)
    plt.close(fig)

    print("\nAnisotropic emission pilot")
    print(f"deterministic wrapper normalized RMS: {wrapper_norm:.6f}%")
    print(f"mean anisotropic shift from scalar: {scalar_delta_pct:+.3f}%")
    print(f"reliable tensor-drive fraction: {metrics['reliable_fraction']:.5f}")
    print(f"nonzero drive fraction: {metrics['nonzero_drive_fraction']:.5f}")
    print(f"mean channel-factor difference: {metrics['mean_channel_factor_difference']:.6g}")
    print(f"post-hazard weighting count: {metrics['post_hazard_weighting_count']}")
    print(f"pilot_pass={assessment['pilot_pass']}")


if __name__ == "__main__":
    main()
