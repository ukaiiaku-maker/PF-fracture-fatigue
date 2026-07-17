#!/usr/bin/env python3
"""Analyze the v10.1.7.3 stochastic avalanche-length pilot."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _load_case(case_dir: Path, label: str, mode: str, seed: int,
               base_checkpoint_um: float) -> dict[str, Any]:
    summary = json.loads((case_dir / "summary.json").read_text())[0]
    audit = json.loads((case_dir / "kinetic_tip_cell_audit_v101.json").read_text())
    records = audit.get("records", [])
    fired = [record for record in records if bool(record.get("fired", False))]
    if not fired:
        raise ValueError(f"no fired records in {case_dir}")

    event_k = np.asarray([
        _float(record.get("K_Pa_sqrt_m")) / 1.0e6 for record in fired
    ], dtype=float)
    event_total_um = np.asarray([
        _float(record.get("kinetic_micro_advance_total_m",
                          record.get("micro_advance_total_m", 0.0)), 0.0) * 1.0e6
        for record in fired
    ], dtype=float)
    event_ext_um = np.maximum(event_total_um - event_total_um[0], 0.0)
    event_lengths_um = np.asarray([
        _float(record.get("avalanche_event_advance_m"), base_checkpoint_um * 1.0e-6)
        * 1.0e6 for record in fired
    ], dtype=float)
    thresholds = np.asarray([
        _float(record.get("hazard_last_completed_threshold"), 1.0)
        for record in fired
    ], dtype=float)

    geometry_path = case_dir / "stochastic_avalanche_geometry_events.json"
    geometry = json.loads(geometry_path.read_text()) if geometry_path.exists() else []
    n_subsegments = np.asarray([
        int(row.get("n_subsegments", 1)) for row in geometry
    ], dtype=int)

    return {
        "label": label,
        "mode": mode,
        "seed": int(seed),
        "case_dir": str(case_dir),
        "K_init_MPa_sqrt_m": float(event_k[0]),
        "K_final_MPa_sqrt_m": float(event_k[-1]),
        "K_mean_MPa_sqrt_m": float(np.mean(event_k)),
        "n_events": int(event_k.size),
        "n_advances_summary": int(summary.get("n_advances", event_k.size)),
        "final_extension_um": float(event_total_um[-1]),
        "event_extension_um": event_ext_um,
        "event_K_MPa_sqrt_m": event_k,
        "event_lengths_um": event_lengths_um,
        "thresholds": thresholds,
        "event_length_mean_um": float(np.mean(event_lengths_um)),
        "event_length_std_um": float(np.std(event_lengths_um)),
        "event_length_min_um": float(np.min(event_lengths_um)),
        "event_length_max_um": float(np.max(event_lengths_um)),
        "event_length_cv": float(np.std(event_lengths_um) / max(np.mean(event_lengths_um), 1e-300)),
        "threshold_mean": float(np.mean(thresholds)),
        "threshold_std": float(np.std(thresholds)),
        "threshold_length_correlation": (
            float(np.corrcoef(thresholds, event_lengths_um)[0, 1])
            if thresholds.size > 1 and np.std(thresholds) > 0.0 and np.std(event_lengths_um) > 0.0
            else math.nan
        ),
        "geometry_event_count": len(geometry),
        "geometry_subsegments_mean": float(np.mean(n_subsegments)) if n_subsegments.size else 1.0,
    }


def _interp(row: dict[str, Any], grid: np.ndarray) -> np.ndarray:
    x = np.asarray(row["event_extension_um"], dtype=float)
    y = np.asarray(row["event_K_MPa_sqrt_m"], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    keep = np.concatenate(([True], np.diff(x) > 1.0e-12))
    return np.interp(grid, x[keep], y[keep])


def _smooth(values: np.ndarray, window: int = 7) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return values.copy()
    width = min(max(int(window), 3), values.size)
    if width % 2 == 0:
        width -= 1
    pad = width // 2
    padded = np.pad(values, pad, mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def _detrended_corr(a: np.ndarray, b: np.ndarray) -> float:
    ar = np.asarray(a) - _smooth(np.asarray(a))
    br = np.asarray(b) - _smooth(np.asarray(b))
    if ar.size < 3 or np.std(ar) <= 0.0 or np.std(br) <= 0.0:
        return math.nan
    return float(np.corrcoef(ar, br)[0, 1])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    excluded = {
        "event_extension_um", "event_K_MPa_sqrt_m", "event_lengths_um", "thresholds"
    }
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key in excluded or key in seen:
                continue
            seen.add(key)
            keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _plot_ensemble(root: Path, fixed: dict[str, Any], segmented: dict[str, Any],
                   stochastic: list[dict[str, Any]], grid: np.ndarray,
                   stack: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    ax.plot(fixed["event_extension_um"], fixed["event_K_MPa_sqrt_m"],
            linestyle="--", linewidth=2.0, label="fixed 5 µm deterministic")
    ax.plot(segmented["event_extension_um"], segmented["event_K_MPa_sqrt_m"],
            linewidth=2.0, label="segmented deterministic")
    for row in stochastic:
        ax.plot(row["event_extension_um"], row["event_K_MPa_sqrt_m"],
                linewidth=1.0, alpha=0.55, label=f"seed {row['seed']}")
    mean = np.mean(stack, axis=0)
    q10 = np.quantile(stack, 0.10, axis=0)
    q90 = np.quantile(stack, 0.90, axis=0)
    ax.fill_between(grid, q10, q90, alpha=0.22, label="stochastic 10–90%")
    ax.plot(grid, mean, linewidth=2.3, label="stochastic mean")
    ax.set_xlabel("Crack extension after initiation (µm)")
    ax.set_ylabel("K at advance (MPa√m)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(root / "stochastic_avalanche_R_curve_ensemble.png", dpi=220)
    plt.close(fig)


def _plot_lengths(root: Path, stochastic: list[dict[str, Any]], base_um: float) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for row in stochastic:
        lengths = np.asarray(row["event_lengths_um"])
        ax.plot(np.arange(1, len(lengths) + 1), lengths, marker="o", markersize=3,
                label=f"seed {row['seed']}")
    ax.axhline(base_um, linestyle="--", linewidth=1.5, label="deterministic 5 µm")
    ax.set_xlabel("Event index")
    ax.set_ylabel("Crack-growth event length (µm)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "stochastic_avalanche_event_lengths.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    all_lengths = np.concatenate([np.asarray(row["event_lengths_um"]) for row in stochastic])
    ax.hist(all_lengths, bins=min(20, max(6, int(np.sqrt(all_lengths.size)))))
    ax.axvline(base_um, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Crack-growth event length (µm)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(root / "stochastic_avalanche_event_length_distribution.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for row in stochastic:
        ax.scatter(row["thresholds"], row["event_lengths_um"], s=22,
                   label=f"seed {row['seed']}")
    ax.set_xlabel("Integrated hazard threshold, Ξ")
    ax.set_ylabel("Crack-growth event length (µm)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "stochastic_avalanche_threshold_vs_length.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--class", dest="material_class", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--theta", type=float, default=45.0)
    parser.add_argument("--base-checkpoint-um", type=float, default=5.0)
    args = parser.parse_args()

    root = args.root.resolve()
    tag = f"T{args.temperature:g}_th{args.theta:g}"
    fixed = _load_case(root / "fixed_original" / tag,
                       "fixed_original", "deterministic", 0,
                       args.base_checkpoint_um)
    segmented = _load_case(root / "segmented_deterministic" / tag,
                           "segmented_deterministic", "deterministic", 0,
                           args.base_checkpoint_um)
    stochastic = [
        _load_case(root / "stochastic_avalanche" / f"seed_{seed}" / tag,
                   f"seed_{seed}", "exponential_threshold_scaled", seed,
                   args.base_checkpoint_um)
        for seed in args.seeds
    ]
    rows = [fixed, segmented, *stochastic]

    common_max = min(float(np.max(row["event_extension_um"])) for row in rows)
    grid_step = max(args.base_checkpoint_um, 1.0e-6)
    grid = np.arange(0.0, math.floor(common_max / grid_step) * grid_step + 1.0e-9,
                     grid_step)
    if grid.size < 3:
        raise RuntimeError("insufficient common extension for avalanche analysis")

    fixed_grid = _interp(fixed, grid)
    segmented_grid = _interp(segmented, grid)
    stack = np.vstack([_interp(row, grid) for row in stochastic])
    ensemble_mean = np.mean(stack, axis=0)
    q10 = np.quantile(stack, 0.10, axis=0)
    q90 = np.quantile(stack, 0.90, axis=0)

    fixed_range = max(float(np.ptp(fixed_grid)), 1.0e-12)
    segmented_rms = float(np.sqrt(np.mean((segmented_grid - fixed_grid) ** 2)))
    segmented_norm = 100.0 * segmented_rms / fixed_range
    ensemble_bias = float(np.mean(ensemble_mean - segmented_grid))
    ensemble_bias_pct = 100.0 * ensemble_bias / max(float(np.mean(segmented_grid)), 1.0e-12)
    ensemble_rms = float(np.sqrt(np.mean((ensemble_mean - segmented_grid) ** 2)))
    band_width = float(np.mean(q90 - q10))
    corrs = [_detrended_corr(curve, segmented_grid) for curve in stack]
    finite_corrs = [value for value in corrs if math.isfinite(value)]

    all_lengths = np.concatenate([np.asarray(row["event_lengths_um"]) for row in stochastic])
    all_thresholds = np.concatenate([np.asarray(row["thresholds"]) for row in stochastic])
    length_mean = float(np.mean(all_lengths))
    length_std = float(np.std(all_lengths))
    length_cv = length_std / max(length_mean, 1.0e-300)
    threshold_length_corr = (
        float(np.corrcoef(all_thresholds, all_lengths)[0, 1])
        if np.std(all_thresholds) > 0.0 and np.std(all_lengths) > 0.0 else math.nan
    )

    assessment = {
        "schema": "v10.1.7.3_stochastic_avalanche_length_pilot",
        "material_class": args.material_class,
        "temperature_K": args.temperature,
        "n_stochastic_seeds": len(stochastic),
        "common_extension_max_um": float(grid[-1]),
        "segmented_control_rms_difference_MPa_sqrt_m": segmented_rms,
        "segmented_control_normalized_rms_percent_of_fixed_range": segmented_norm,
        "stochastic_ensemble_mean_bias_MPa_sqrt_m": ensemble_bias,
        "stochastic_ensemble_mean_bias_percent": ensemble_bias_pct,
        "stochastic_ensemble_rms_difference_MPa_sqrt_m": ensemble_rms,
        "mean_10_90_band_width_MPa_sqrt_m": band_width,
        "mean_detrended_seed_correlation_to_segmented_deterministic": (
            float(np.mean(finite_corrs)) if finite_corrs else math.nan
        ),
        "maximum_detrended_seed_correlation_to_segmented_deterministic": (
            float(np.max(finite_corrs)) if finite_corrs else math.nan
        ),
        "event_length_sample_count": int(all_lengths.size),
        "event_length_mean_um": length_mean,
        "event_length_std_um": length_std,
        "event_length_cv": length_cv,
        "event_length_min_um": float(np.min(all_lengths)),
        "event_length_max_um": float(np.max(all_lengths)),
        "threshold_length_correlation": threshold_length_corr,
        "segmentation_only_preserves_fixed_response": segmented_norm <= 2.0,
        "ensemble_mean_within_5_percent": abs(ensemble_bias_pct) <= 5.0,
        "event_length_mean_within_20_percent": (
            abs(length_mean - args.base_checkpoint_um) <= 0.2 * args.base_checkpoint_um
        ),
        "event_lengths_are_broad": length_cv >= 0.30,
        "visible_nonzero_band": band_width >= 0.25,
        "geometry_waveform_decorrelated": (
            bool(finite_corrs) and float(np.mean(finite_corrs)) <= 0.98
        ),
        "geometry_subsegments_re_equilibrated": False,
    }
    assessment["pilot_pass"] = all([
        assessment["segmentation_only_preserves_fixed_response"],
        assessment["ensemble_mean_within_5_percent"],
        assessment["event_length_mean_within_20_percent"],
        assessment["event_lengths_are_broad"],
        assessment["visible_nonzero_band"],
        assessment["geometry_waveform_decorrelated"],
    ])

    _write_csv(root / "stochastic_avalanche_case_summary.csv", rows)
    events = {
        row["label"]: {
            "extension_um": np.asarray(row["event_extension_um"]).tolist(),
            "K_MPa_sqrt_m": np.asarray(row["event_K_MPa_sqrt_m"]).tolist(),
            "event_length_um": np.asarray(row["event_lengths_um"]).tolist(),
            "threshold": np.asarray(row["thresholds"]).tolist(),
        }
        for row in rows
    }
    (root / "stochastic_avalanche_event_curves.json").write_text(
        json.dumps(events, indent=2)
    )
    (root / "stochastic_avalanche_assessment.json").write_text(
        json.dumps(assessment, indent=2)
    )

    _plot_ensemble(root, fixed, segmented, stochastic, grid, stack)
    _plot_lengths(root, stochastic, args.base_checkpoint_um)

    print("\nStochastic avalanche-length pilot")
    print(f"segmentation-only normalized RMS: {segmented_norm:.3f}%")
    print(f"ensemble mean bias: {ensemble_bias_pct:+.3f}%")
    print(f"mean 10-90% band: {band_width:.3f} MPa√m")
    print(f"mean event length: {length_mean:.3f} ± {length_std:.3f} µm")
    print(f"mean detrended correlation: {assessment['mean_detrended_seed_correlation_to_segmented_deterministic']:.5f}")
    print(f"pilot_pass={assessment['pilot_pass']}")


if __name__ == "__main__":
    main()
