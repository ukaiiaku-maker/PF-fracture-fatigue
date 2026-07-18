#!/usr/bin/env python3
"""Analyze paired scalar/anisotropic v10.1.7.4 orientation-temperature runs."""
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


def _finite(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _load_projected_extension_um(root: Path) -> float:
    paths = sorted(root.glob("steps_*K.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one step CSV in {root}; found {paths}")
    lines = [line.strip() for line in paths[0].read_text().splitlines() if line.strip()]
    header = [token.strip() for token in lines[0].lstrip("# ").split(",")]
    index = header.index("crack_extension_m")
    return max(float(line.split(",")[index]) for line in lines[1:]) * 1.0e6


def _load_case(root: Path, model: str, temperature: float, theta: float) -> dict[str, Any]:
    summary = json.loads((root / "summary.json").read_text())[0]
    audit = json.loads((root / "kinetic_tip_cell_audit_v101.json").read_text())
    records = audit.get("records", [])
    fired = [row for row in records if bool(row.get("fired", False))]
    if not fired:
        raise RuntimeError(f"no fired records in {root}")

    event_k = np.asarray([
        _finite(row.get("K_Pa_sqrt_m")) / 1.0e6 for row in fired
    ], dtype=float)
    event_path = np.asarray([
        _finite(row.get("micro_advance_total_m"), 0.0) * 1.0e6 for row in fired
    ], dtype=float)
    event_path = np.maximum(event_path - event_path[0], 0.0)

    row: dict[str, Any] = {
        "model": model,
        "temperature_K": float(temperature),
        "theta_deg": float(theta),
        "case_dir": str(root),
        "Kc_first_MPa_sqrt_m": _finite(summary.get("Kc_first_MPa_sqrt_m")),
        "K_event_mean_MPa_sqrt_m": float(np.mean(event_k)),
        "K_event_final_MPa_sqrt_m": float(event_k[-1]),
        "n_events": int(event_k.size),
        "n_advances": int(summary.get("n_advances", event_k.size)),
        "projected_extension_um": _load_projected_extension_um(root),
        "deflection_deg": _finite(summary.get("deflection_deg")),
        "path_span_dy_um": 1000.0 * _finite(summary.get("path_span_dy_mm"), 0.0),
        "N_em_final": _finite(summary.get("N_em_final")),
        "W_emit_J_per_m": _finite(summary.get("W_emit_J_per_m")),
        "event_path_um": event_path,
        "event_K_MPa_sqrt_m": event_k,
    }

    if model != "anisotropic":
        return row

    aniso_records = [
        value for value in records if "anisotropic_drive_reliable" in value
    ]
    if not aniso_records:
        raise RuntimeError(f"no anisotropic records in {root}")

    factors = np.asarray([
        value.get("anisotropic_drive_factors", [math.nan, math.nan])
        for value in aniso_records
    ], dtype=float)
    tau = np.asarray([
        value.get("anisotropic_tau_signed_Pa", [math.nan, math.nan])
        for value in aniso_records
    ], dtype=float)
    sigma_emit = np.asarray([
        value.get("anisotropic_sigma_emit_by_system_Pa", [math.nan, math.nan])
        for value in aniso_records
    ], dtype=float)
    lambdas = np.asarray([
        value.get("anisotropic_lambda_emit_by_system_s", [math.nan, math.nan])
        for value in aniso_records
    ], dtype=float)
    velocities = np.asarray([
        value.get("anisotropic_transport_velocity_by_system_m_s", [math.nan, math.nan])
        for value in aniso_records
    ], dtype=float)
    reliable = np.asarray([
        bool(value.get("anisotropic_drive_reliable", False))
        for value in aniso_records
    ], dtype=bool)
    post_weight = np.asarray([
        bool(value.get("anisotropic_post_hazard_weighting_applied", False))
        for value in aniso_records
    ], dtype=bool)
    dominant = np.argmax(np.nan_to_num(factors, nan=-math.inf), axis=1)
    switches = int(np.count_nonzero(np.diff(dominant) != 0)) if dominant.size > 1 else 0
    fractions = np.bincount(dominant, minlength=2) / max(dominant.size, 1)
    final = aniso_records[-1]

    row.update({
        "reliable_fraction": float(np.mean(reliable)),
        "post_hazard_weighting_count": int(np.count_nonzero(post_weight)),
        "factor_0_mean": float(np.nanmean(factors[:, 0])),
        "factor_1_mean": float(np.nanmean(factors[:, 1])),
        "factor_0_max": float(np.nanmax(factors[:, 0])),
        "factor_1_max": float(np.nanmax(factors[:, 1])),
        "factor_overall_max": float(np.nanmax(factors)),
        "fraction_any_factor_gt_1": float(np.mean(np.nanmax(factors, axis=1) > 1.0)),
        "mean_channel_factor_difference": float(np.nanmean(np.abs(factors[:, 0] - factors[:, 1]))),
        "dominant_channel": int(np.argmax(np.nanmean(factors, axis=0))),
        "dominant_channel_0_fraction": float(fractions[0]),
        "dominant_channel_1_fraction": float(fractions[1]),
        "dominant_channel_switch_count": switches,
        "tau_0_abs_mean_GPa": float(np.nanmean(np.abs(tau[:, 0])) / 1.0e9),
        "tau_1_abs_mean_GPa": float(np.nanmean(np.abs(tau[:, 1])) / 1.0e9),
        "sigma_emit_0_mean_GPa": float(np.nanmean(sigma_emit[:, 0]) / 1.0e9),
        "sigma_emit_1_mean_GPa": float(np.nanmean(sigma_emit[:, 1]) / 1.0e9),
        "log10_lambda_emit_0_median": float(np.nanmedian(np.log10(np.maximum(lambdas[:, 0], 1.0e-300)))),
        "log10_lambda_emit_1_median": float(np.nanmedian(np.log10(np.maximum(lambdas[:, 1], 1.0e-300)))),
        "log10_velocity_0_median": float(np.nanmedian(np.log10(np.maximum(velocities[:, 0], 1.0e-300)))),
        "log10_velocity_1_median": float(np.nanmedian(np.log10(np.maximum(velocities[:, 1], 1.0e-300)))),
        "cumulative_emitted_total": _finite(final.get("developed_state_cumulative_emitted"), 0.0),
        "active_mobile_final": _finite(final.get("developed_state_mobile_count"), 0.0),
        "active_retained_final": _finite(final.get("developed_state_retained_count"), 0.0),
        "active_K_shield_final_MPa_sqrt_m": _finite(
            final.get("campaign_active_K_shield_effective_Pa_sqrt_m"), 0.0
        ) / 1.0e6,
        "source_budget_remaining_final": _finite(
            final.get("campaign_source_budget_remaining"), 0.0
        ),
    })
    return row


def _paired_curve_metrics(scalar: dict[str, Any], aniso: dict[str, Any]) -> dict[str, float]:
    xmax = min(
        float(np.max(scalar["event_path_um"])),
        float(np.max(aniso["event_path_um"])),
    )
    grid = np.arange(0.0, math.floor(xmax / 5.0) * 5.0 + 1.0e-9, 5.0)
    if grid.size < 3:
        return {
            "paired_curve_mean_shift_percent": math.nan,
            "paired_curve_normalized_rms_percent": math.nan,
            "paired_curve_common_path_um": xmax,
        }
    ks = np.interp(grid, scalar["event_path_um"], scalar["event_K_MPa_sqrt_m"])
    ka = np.interp(grid, aniso["event_path_um"], aniso["event_K_MPa_sqrt_m"])
    mean_shift = 100.0 * float(np.mean(ka - ks)) / max(float(np.mean(ks)), 1.0e-12)
    scale = max(float(np.ptp(ks)), 1.0e-12)
    rms = 100.0 * float(np.sqrt(np.mean((ka - ks) ** 2))) / scale
    return {
        "paired_curve_mean_shift_percent": mean_shift,
        "paired_curve_normalized_rms_percent": rms,
        "paired_curve_common_path_um": float(grid[-1]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    excluded = {"event_path_um", "event_K_MPa_sqrt_m"}
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


def _plot_kc(root: Path, rows: list[dict[str, Any]], thetas: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for theta in thetas:
        for model, linestyle, marker in (
            ("scalar", "--", "o"),
            ("anisotropic", "-", "s"),
        ):
            selected = sorted(
                [r for r in rows if r["model"] == model and r["theta_deg"] == theta],
                key=lambda value: value["temperature_K"],
            )
            ax.plot(
                [r["temperature_K"] for r in selected],
                [r["Kc_first_MPa_sqrt_m"] for r in selected],
                marker=marker,
                linestyle=linestyle,
                label=f"{model}, {theta:g}°",
            )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("First-passage toughness (MPa√m)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(root / "orientation_temperature_Kc.png", dpi=220)
    plt.close(fig)


def _plot_shift(root: Path, paired: list[dict[str, Any]], thetas: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for theta in thetas:
        selected = sorted(
            [r for r in paired if r["theta_deg"] == theta],
            key=lambda value: value["temperature_K"],
        )
        ax.plot(
            [r["temperature_K"] for r in selected],
            [r["paired_curve_mean_shift_percent"] for r in selected],
            marker="o",
            label=f"{theta:g}°",
        )
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Anisotropic mean R-curve shift (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "anisotropic_R_curve_shift_vs_temperature.png", dpi=220)
    plt.close(fig)


def _plot_factors(root: Path, rows: list[dict[str, Any]], thetas: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    aniso = [row for row in rows if row["model"] == "anisotropic"]
    for theta in thetas:
        selected = sorted(
            [r for r in aniso if r["theta_deg"] == theta],
            key=lambda value: value["temperature_K"],
        )
        temperature = [r["temperature_K"] for r in selected]
        ax.plot(temperature, [r["factor_0_mean"] for r in selected], marker="o", label=f"(110), {theta:g}°")
        ax.plot(temperature, [r["factor_1_mean"] for r in selected], marker="s", linestyle="--", label=f"(1-10), {theta:g}°")
    ax.axhline(1.0, linewidth=1.0, linestyle=":")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Mean anisotropic emission drive factor")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(root / "channel_drive_factors_vs_temperature.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--temperatures", nargs="+", type=float, required=True)
    parser.add_argument("--thetas", nargs="+", type=float, required=True)
    parser.add_argument("--target-extension-um", type=float, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    rows: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for theta in args.thetas:
        for temperature in args.temperatures:
            tag = f"T{temperature:g}_th{theta:g}"
            scalar = _load_case(root / "scalar" / tag, "scalar", temperature, theta)
            aniso = _load_case(root / "anisotropic" / tag, "anisotropic", temperature, theta)
            rows.extend([scalar, aniso])
            pair = {
                "temperature_K": float(temperature),
                "theta_deg": float(theta),
                "Kc_scalar_MPa_sqrt_m": scalar["Kc_first_MPa_sqrt_m"],
                "Kc_anisotropic_MPa_sqrt_m": aniso["Kc_first_MPa_sqrt_m"],
                "Kc_shift_percent": 100.0 * (
                    aniso["Kc_first_MPa_sqrt_m"] - scalar["Kc_first_MPa_sqrt_m"]
                ) / max(scalar["Kc_first_MPa_sqrt_m"], 1.0e-12),
                "deflection_scalar_deg": scalar["deflection_deg"],
                "deflection_anisotropic_deg": aniso["deflection_deg"],
                "N_em_scalar_final": scalar["N_em_final"],
                "N_em_anisotropic_final": aniso["N_em_final"],
                "factor_0_mean": aniso["factor_0_mean"],
                "factor_1_mean": aniso["factor_1_mean"],
                "factor_overall_max": aniso["factor_overall_max"],
                "fraction_any_factor_gt_1": aniso["fraction_any_factor_gt_1"],
                "dominant_channel": aniso["dominant_channel"],
                "dominant_channel_switch_count": aniso["dominant_channel_switch_count"],
                "reliable_fraction": aniso["reliable_fraction"],
                "post_hazard_weighting_count": aniso["post_hazard_weighting_count"],
            }
            pair.update(_paired_curve_metrics(scalar, aniso))
            paired.append(pair)

    _write_csv(root / "orientation_temperature_case_summary.csv", rows)
    _write_csv(root / "orientation_temperature_paired_summary.csv", paired)

    aniso_rows = [row for row in rows if row["model"] == "anisotropic"]
    factor_ranges_by_temperature = {}
    for temperature in args.temperatures:
        selected = [row for row in aniso_rows if row["temperature_K"] == temperature]
        dominant_means = [max(row["factor_0_mean"], row["factor_1_mean"]) for row in selected]
        factor_ranges_by_temperature[f"{temperature:g}"] = float(np.ptp(dominant_means))

    assessment = {
        "schema": "v10.1.7.4_orientation_temperature_sweep",
        "temperatures_K": [float(value) for value in args.temperatures],
        "orientations_deg": [float(value) for value in args.thetas],
        "target_extension_um": float(args.target_extension_um),
        "n_paired_conditions": len(paired),
        "all_cases_reached_target": all(
            row["projected_extension_um"] + 1.0e-6 >= args.target_extension_um
            for row in rows
        ),
        "all_tensor_drives_reliable": all(
            math.isclose(row["reliable_fraction"], 1.0, rel_tol=0.0, abs_tol=1.0e-12)
            for row in aniso_rows
        ),
        "post_hazard_weighting_total": int(sum(
            row["post_hazard_weighting_count"] for row in aniso_rows
        )),
        "nonzero_channel_drive_all_conditions": all(
            max(row["factor_0_max"], row["factor_1_max"]) > 0.0
            for row in aniso_rows
        ),
        "maximum_factor_observed": float(max(row["factor_overall_max"] for row in aniso_rows)),
        "any_factor_above_scalar_reference": any(
            row["factor_overall_max"] > 1.0 for row in aniso_rows
        ),
        "dominant_channels_observed": sorted(set(
            int(row["dominant_channel"]) for row in aniso_rows
        )),
        "channel_order_changes_across_conditions": len(set(
            int(row["dominant_channel"]) for row in aniso_rows
        )) > 1,
        "dominant_factor_range_by_temperature": factor_ranges_by_temperature,
        "orientation_response_detected": max(factor_ranges_by_temperature.values()) >= 0.02,
        "maximum_abs_Kc_shift_percent": float(max(abs(row["Kc_shift_percent"]) for row in paired)),
        "maximum_abs_mean_R_curve_shift_percent": float(max(abs(row["paired_curve_mean_shift_percent"]) for row in paired)),
    }
    assessment["pilot_pass"] = all([
        assessment["all_cases_reached_target"],
        assessment["all_tensor_drives_reliable"],
        assessment["post_hazard_weighting_total"] == 0,
        assessment["nonzero_channel_drive_all_conditions"],
    ])
    (root / "orientation_temperature_assessment.json").write_text(
        json.dumps(assessment, indent=2)
    )

    _plot_kc(root, rows, args.thetas)
    _plot_shift(root, paired, args.thetas)
    _plot_factors(root, rows, args.thetas)

    print("\nAnisotropic emission orientation-temperature sweep")
    print(f"paired conditions: {len(paired)}")
    print(f"all tensor drives reliable: {assessment['all_tensor_drives_reliable']}")
    print(f"post-hazard weighting total: {assessment['post_hazard_weighting_total']}")
    print(f"maximum factor observed: {assessment['maximum_factor_observed']:.6f}")
    print(f"orientation response detected: {assessment['orientation_response_detected']}")
    print(f"maximum |Kc shift|: {assessment['maximum_abs_Kc_shift_percent']:.3f}%")
    print(f"maximum |mean R-curve shift|: {assessment['maximum_abs_mean_R_curve_shift_percent']:.3f}%")
    print(f"pilot_pass={assessment['pilot_pass']}")


if __name__ == "__main__":
    main()
