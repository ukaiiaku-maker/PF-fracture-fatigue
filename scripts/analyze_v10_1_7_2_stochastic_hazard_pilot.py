#!/usr/bin/env python3
"""Analyze the v10.1.7.2 deterministic-versus-stochastic DBTT pilot."""
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


def _load_case(
    case_dir: Path,
    label: str,
    mode: str,
    seed: int,
    checkpoint_advance_um: float,
) -> dict[str, Any]:
    audit = json.loads((case_dir / "kinetic_tip_cell_audit_v101.json").read_text())
    records = audit.get("records", [])
    fired = [record for record in records if bool(record.get("fired", False))]
    if not fired:
        raise ValueError(f"no fired records in {case_dir}")

    event_k = np.asarray([
        _float(record.get("K_Pa_sqrt_m")) / 1.0e6 for record in fired
    ], dtype=float)
    if not np.all(np.isfinite(event_k)):
        raise ValueError(f"non-finite event K in {case_dir}")

    event_ext = []
    for index, record in enumerate(fired):
        advance = _float(
            record.get("kinetic_micro_advance_total_m", record.get("micro_advance_total_m"))
        )
        event_ext.append(
            advance * 1.0e6 if math.isfinite(advance)
            else (index + 1) * checkpoint_advance_um
        )
    event_ext = np.asarray(event_ext, dtype=float)
    event_ext -= event_ext[0]

    thresholds = np.asarray([
        _float(record.get("hazard_last_completed_threshold")) for record in fired
    ], dtype=float)
    if not np.all(np.isfinite(thresholds)) or np.any(thresholds <= 0.0):
        raise ValueError(f"invalid hazard threshold history in {case_dir}")

    active = np.asarray([
        _float(record.get("developed_state_active_count"), 0.0) for record in fired
    ], dtype=float)
    retained = np.asarray([
        _float(record.get("developed_state_retained_count"), 0.0) for record in fired
    ], dtype=float)
    backstress = np.asarray([
        _float(record.get("sigma_emission_backstress_Pa"), 0.0) / 1.0e9
        for record in fired
    ], dtype=float)
    shielding = np.asarray([
        _float(record.get("campaign_active_K_shield_effective_Pa_sqrt_m"), 0.0) / 1.0e6
        for record in fired
    ], dtype=float)

    return {
        "label": label,
        "mode": mode,
        "seed": seed,
        "case_dir": str(case_dir),
        "extension_um": event_ext,
        "K_MPa_sqrt_m": event_k,
        "thresholds": thresholds,
        "active": active,
        "retained": retained,
        "backstress_GPa": backstress,
        "shielding_MPa_sqrt_m": shielding,
        "n_events": int(len(event_k)),
        "K_init_MPa_sqrt_m": float(event_k[0]),
        "K_final_MPa_sqrt_m": float(event_k[-1]),
        "K_mean_MPa_sqrt_m": float(np.mean(event_k)),
        "threshold_mean": float(np.mean(thresholds)),
        "threshold_std": float(np.std(thresholds, ddof=1)) if len(thresholds) > 1 else 0.0,
        "threshold_min": float(np.min(thresholds)),
        "threshold_max": float(np.max(thresholds)),
        "late_active_mean": float(np.mean(active[-min(10, len(active)):])),
        "late_retained_mean": float(np.mean(retained[-min(10, len(retained)):])),
        "late_backstress_GPa": float(np.mean(backstress[-min(10, len(backstress)):])),
        "late_shielding_MPa_sqrt_m": float(np.mean(shielding[-min(10, len(shielding)):])),
    }


def _common_stack(cases: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(case["K_MPa_sqrt_m"]) for case in cases)
    x = np.asarray(cases[0]["extension_um"][:n], dtype=float)
    stack = np.vstack([np.asarray(case["K_MPa_sqrt_m"][:n], dtype=float) for case in cases])
    return x, stack


def _detrend(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    x = np.arange(len(values), dtype=float)
    degree = min(3, max(len(values) - 1, 0))
    if degree < 1:
        return values - np.mean(values)
    return values - np.polyval(np.polyfit(x, values, degree), x)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = _detrend(a)
    bb = _detrend(b)
    if np.std(aa) <= 0.0 or np.std(bb) <= 0.0:
        return math.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    excluded = {
        "extension_um", "K_MPa_sqrt_m", "thresholds", "active", "retained",
        "backstress_GPa", "shielding_MPa_sqrt_m",
    }
    keys = []
    seen = set()
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


def _plot_ensemble(root: Path, deterministic: dict[str, Any], stochastic: list[dict[str, Any]]) -> None:
    x, stack = _common_stack(stochastic)
    n = len(x)
    det = np.asarray(deterministic["K_MPa_sqrt_m"][:n], dtype=float)
    mean = np.mean(stack, axis=0)
    p10 = np.percentile(stack, 10.0, axis=0)
    p90 = np.percentile(stack, 90.0, axis=0)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for case in stochastic:
        ax.plot(
            case["extension_um"][:n], case["K_MPa_sqrt_m"][:n],
            linewidth=0.8, alpha=0.35,
        )
    ax.fill_between(x, p10, p90, alpha=0.25, label="Stochastic 10–90%")
    ax.plot(x, mean, linewidth=2.2, label="Stochastic mean")
    ax.plot(x, det, linewidth=2.2, linestyle="--", label="Deterministic")
    ax.set_xlabel("Crack extension after initiation (µm)")
    ax.set_ylabel("K at advance (MPa√m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "stochastic_hazard_R_curve_ensemble.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    all_thresholds = np.concatenate([case["thresholds"] for case in stochastic])
    ax.hist(all_thresholds, bins=25, density=True, alpha=0.65, label="Sampled thresholds")
    xx = np.linspace(0.0, max(float(np.percentile(all_thresholds, 99.0)), 1.0), 250)
    ax.plot(xx, np.exp(-xx), linewidth=2.0, label="Exp(1) density")
    ax.set_xlabel("Integrated hazard threshold")
    ax.set_ylabel("Probability density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "stochastic_hazard_threshold_distribution.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    init = [case["K_init_MPa_sqrt_m"] for case in stochastic]
    ax.scatter(np.arange(1, len(init) + 1), init, label="Stochastic seeds")
    ax.axhline(deterministic["K_init_MPa_sqrt_m"], linestyle="--", label="Deterministic")
    ax.set_xlabel("Seed index")
    ax.set_ylabel("Initiation K (MPa√m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "stochastic_hazard_initiation_scatter.png", dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--class", dest="material_class", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--theta", type=float, default=45.0)
    parser.add_argument("--checkpoint-advance-um", type=float, default=5.0)
    args = parser.parse_args()

    root = args.root.resolve()
    temp_tag = f"T{args.temperature:g}_th{args.theta:g}"
    deterministic = _load_case(
        root / "deterministic" / temp_tag,
        "deterministic", "deterministic", 0,
        args.checkpoint_advance_um,
    )
    stochastic = [
        _load_case(
            root / "stochastic" / f"seed_{seed}" / temp_tag,
            f"seed_{seed}", "exponential", seed,
            args.checkpoint_advance_um,
        )
        for seed in args.seeds
    ]

    x, stack = _common_stack(stochastic)
    n = len(x)
    det = np.asarray(deterministic["K_MPa_sqrt_m"][:n], dtype=float)
    ensemble_mean = np.mean(stack, axis=0)
    residual = ensemble_mean - det
    mean_bias = float(np.mean(residual))
    mean_level = max(float(np.mean(np.abs(det))), 1.0e-12)
    mean_bias_percent = 100.0 * mean_bias / mean_level
    rms_difference = float(np.sqrt(np.mean(residual ** 2)))
    normalized_rms_percent = 100.0 * rms_difference / max(float(np.ptp(det)), 1.0e-12)
    band_width = np.percentile(stack, 90.0, axis=0) - np.percentile(stack, 10.0, axis=0)
    mean_band_width = float(np.mean(band_width))
    seed_correlations = [
        _safe_corr(case["K_MPa_sqrt_m"][:n], det) for case in stochastic
    ]
    finite_correlations = [value for value in seed_correlations if math.isfinite(value)]
    all_thresholds = np.concatenate([case["thresholds"] for case in stochastic])

    rows = [deterministic, *stochastic]
    for index, case in enumerate(stochastic):
        case["detrended_correlation_to_deterministic"] = seed_correlations[index]
    _write_csv(root / "stochastic_hazard_case_summary.csv", rows)

    event_payload = {
        "deterministic": {
            "extension_um": deterministic["extension_um"].tolist(),
            "K_MPa_sqrt_m": deterministic["K_MPa_sqrt_m"].tolist(),
            "thresholds": deterministic["thresholds"].tolist(),
        },
        "stochastic": {
            case["label"]: {
                "extension_um": case["extension_um"].tolist(),
                "K_MPa_sqrt_m": case["K_MPa_sqrt_m"].tolist(),
                "thresholds": case["thresholds"].tolist(),
            }
            for case in stochastic
        },
    }
    (root / "stochastic_hazard_event_curves.json").write_text(
        json.dumps(event_payload, indent=2)
    )

    assessment = {
        "schema": "v10.1.7.2_stochastic_hazard_pilot",
        "material_class": args.material_class,
        "temperature_K": args.temperature,
        "n_stochastic_seeds": len(stochastic),
        "n_common_events": n,
        "ensemble_mean_bias_MPa_sqrt_m": mean_bias,
        "ensemble_mean_bias_percent": mean_bias_percent,
        "ensemble_rms_difference_MPa_sqrt_m": rms_difference,
        "ensemble_normalized_rms_percent_of_deterministic_range": normalized_rms_percent,
        "mean_10_90_band_width_MPa_sqrt_m": mean_band_width,
        "mean_detrended_seed_correlation_to_deterministic": (
            float(np.mean(finite_correlations)) if finite_correlations else math.nan
        ),
        "hazard_threshold_sample_count": int(len(all_thresholds)),
        "hazard_threshold_sample_mean": float(np.mean(all_thresholds)),
        "hazard_threshold_sample_std": float(np.std(all_thresholds, ddof=1)),
        "ensemble_mean_within_5_percent": abs(mean_bias_percent) <= 5.0,
        "visible_nonzero_band": mean_band_width > 0.05,
        "seed_paths_not_identical": bool(np.max(np.std(stack, axis=0)) > 1.0e-6),
    }
    assessment["pilot_pass"] = bool(
        assessment["ensemble_mean_within_5_percent"]
        and assessment["visible_nonzero_band"]
        and assessment["seed_paths_not_identical"]
    )
    (root / "stochastic_hazard_assessment.json").write_text(
        json.dumps(assessment, indent=2)
    )
    _plot_ensemble(root, deterministic, stochastic)

    print("\nStochastic hazard pilot summary")
    print(f"common events: {n}")
    print(f"ensemble mean bias: {mean_bias:+.4f} MPa√m ({mean_bias_percent:+.3f}%)")
    print(f"mean 10–90% band width: {mean_band_width:.4f} MPa√m")
    print(f"sample threshold mean/std: {assessment['hazard_threshold_sample_mean']:.4f} / {assessment['hazard_threshold_sample_std']:.4f}")
    print(f"pilot pass: {assessment['pilot_pass']}")


if __name__ == "__main__":
    main()
