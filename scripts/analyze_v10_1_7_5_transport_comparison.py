#!/usr/bin/env python3
"""Analyze the 500 um v10.1.7.5 transport-model comparison."""
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


CASE_NAMES = (
    "scalar_reference",
    "anisotropic_validated_transport",
    "anisotropic_channel_transport",
)


def _finite(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _first_present(row: dict[str, Any], names: tuple[str, ...], default=math.nan) -> float:
    for name in names:
        if name in row:
            value = _finite(row.get(name), math.nan)
            if math.isfinite(value):
                return value
    return float(default)


def _load_projected_extension_um(root: Path) -> float:
    paths = sorted(root.glob("steps_*K.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one steps CSV in {root}; found {paths}")
    lines = [line.strip() for line in paths[0].read_text().splitlines() if line.strip()]
    header = [token.strip() for token in lines[0].lstrip("# ").split(",")]
    index = header.index("crack_extension_m")
    return max(float(line.split(",")[index]) for line in lines[1:]) * 1.0e6


def _load_case(root: Path, name: str, temperature: float, theta: float) -> dict[str, Any]:
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

    mobile = np.asarray([
        _first_present(row, (
            "developed_state_mobile_count",
            "mpz_mobile_count",
        ), 0.0)
        for row in fired
    ], dtype=float)
    retained = np.asarray([
        _first_present(row, (
            "developed_state_retained_count",
            "mpz_retained_count",
        ), 0.0)
        for row in fired
    ], dtype=float)
    emitted = np.asarray([
        _first_present(row, (
            "developed_state_cumulative_emitted",
            "mpz_emitted_total",
        ), 0.0)
        for row in fired
    ], dtype=float)
    escaped = np.asarray([
        _first_present(row, (
            "developed_state_escaped_total",
            "mpz_escaped_total",
        ), 0.0)
        for row in fired
    ], dtype=float)
    recovered = np.asarray([
        _first_present(row, (
            "developed_state_recovered_total",
            "mpz_recovered_total",
        ), 0.0)
        for row in fired
    ], dtype=float)
    shielding = np.asarray([
        _first_present(row, (
            "campaign_active_K_shield_effective_Pa_sqrt_m",
            "active_K_shield_Pa_sqrt_m",
            "mpz_active_K_shield_Pa_sqrt_m",
        ), 0.0) / 1.0e6
        for row in fired
    ], dtype=float)
    source_remaining = np.asarray([
        _first_present(row, (
            "campaign_source_budget_remaining",
            "campaign_source_budget_remaining_total",
            "tip_source_effective_multiplicity_total",
        ), math.nan)
        for row in fired
    ], dtype=float)
    blunted_radius = np.asarray([
        _first_present(row, (
            "developed_state_blunted_radius_m",
            "mpz_blunted_radius_m",
            "blunted_radius_m",
        ), math.nan) * 1.0e6
        for row in fired
    ], dtype=float)

    result: dict[str, Any] = {
        "case": name,
        "temperature_K": float(temperature),
        "theta_deg": float(theta),
        "case_dir": str(root),
        "Kc_first_MPa_sqrt_m": _finite(summary.get("Kc_first_MPa_sqrt_m")),
        "K_event_mean_MPa_sqrt_m": float(np.mean(event_k)),
        "K_event_final_MPa_sqrt_m": float(event_k[-1]),
        "n_events": int(event_k.size),
        "projected_extension_um": _load_projected_extension_um(root),
        "N_em_final": _finite(summary.get("N_em_final")),
        "event_path_um": event_path,
        "event_K_MPa_sqrt_m": event_k,
        "mobile_history": mobile,
        "retained_history": retained,
        "emitted_history": emitted,
        "escaped_history": escaped,
        "recovered_history": recovered,
        "shielding_history_MPa_sqrt_m": shielding,
        "source_remaining_history": source_remaining,
        "blunted_radius_history_um": blunted_radius,
        "mobile_final": float(mobile[-1]),
        "retained_final": float(retained[-1]),
        "emitted_final": float(emitted[-1]),
        "escaped_final": float(escaped[-1]),
        "recovered_final": float(recovered[-1]),
        "active_shielding_final_MPa_sqrt_m": float(shielding[-1]),
        "source_remaining_final": float(source_remaining[-1]),
        "blunted_radius_final_um": float(blunted_radius[-1]),
    }

    anisotropic_records = [
        row for row in records if "anisotropic_drive_reliable" in row
    ]
    if anisotropic_records:
        factors = np.asarray([
            row.get("anisotropic_drive_factors", [math.nan, math.nan])
            for row in anisotropic_records
        ], dtype=float)
        result.update(
            {
                "transport_mode": str(
                    anisotropic_records[-1].get("anisotropic_transport_mode", "unknown")
                ),
                "reliable_fraction": float(np.mean([
                    bool(row.get("anisotropic_drive_reliable", False))
                    for row in anisotropic_records
                ])),
                "post_hazard_weighting_count": int(sum(
                    bool(row.get("anisotropic_post_hazard_weighting_applied", False))
                    for row in anisotropic_records
                )),
                "factor_0_mean": float(np.nanmean(factors[:, 0])),
                "factor_1_mean": float(np.nanmean(factors[:, 1])),
                "factor_overall_max": float(np.nanmax(factors)),
                "first_drive_factors": factors[0].copy(),
            }
        )
    return result


def _interp(case: dict[str, Any], key: str, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(case[key], dtype=float)
    path = np.asarray(case["event_path_um"], dtype=float)
    mask = np.isfinite(path) & np.isfinite(values)
    if np.count_nonzero(mask) < 2:
        return np.full(grid.shape, math.nan)
    return np.interp(grid, path[mask], values[mask])


def _pair_metrics(a: dict[str, Any], b: dict[str, Any], target_um: float) -> dict[str, Any]:
    xmax = min(
        float(np.max(a["event_path_um"])),
        float(np.max(b["event_path_um"])),
        float(target_um),
    )
    grid = np.arange(0.0, math.floor(xmax / 5.0) * 5.0 + 1.0e-9, 5.0)
    if grid.size < 3:
        raise RuntimeError("insufficient common crack-extension range")

    ka = _interp(a, "event_K_MPa_sqrt_m", grid)
    kb = _interp(b, "event_K_MPa_sqrt_m", grid)
    delta_k = kb - ka
    scale = max(float(np.ptp(ka)), float(np.mean(np.abs(ka))), 1.0e-12)

    metrics: dict[str, Any] = {
        "case_a": a["case"],
        "case_b": b["case"],
        "common_path_um": float(grid[-1]),
        "Kc_shift_percent": 100.0 * (
            b["Kc_first_MPa_sqrt_m"] - a["Kc_first_MPa_sqrt_m"]
        ) / max(a["Kc_first_MPa_sqrt_m"], 1.0e-12),
        "mean_R_curve_shift_percent": 100.0 * float(np.mean(delta_k)) / max(
            float(np.mean(ka)), 1.0e-12
        ),
        "normalized_R_curve_rms_percent": 100.0 * float(
            np.sqrt(np.mean(delta_k**2))
        ) / scale,
        "maximum_abs_K_difference_MPa_sqrt_m": float(np.max(np.abs(delta_k))),
    }

    for key, label in (
        ("mobile_history", "mobile"),
        ("retained_history", "retained"),
        ("emitted_history", "emitted"),
        ("escaped_history", "escaped"),
        ("recovered_history", "recovered"),
        ("shielding_history_MPa_sqrt_m", "active_shielding"),
        ("source_remaining_history", "source_remaining"),
        ("blunted_radius_history_um", "blunted_radius"),
    ):
        va = _interp(a, key, grid)
        vb = _interp(b, key, grid)
        difference = vb - va
        metrics[f"final_{label}_difference"] = float(vb[-1] - va[-1])
        metrics[f"maximum_abs_{label}_difference"] = float(np.nanmax(np.abs(difference)))

    return metrics


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    excluded = {
        "event_path_um",
        "event_K_MPa_sqrt_m",
        "mobile_history",
        "retained_history",
        "emitted_history",
        "escaped_history",
        "recovered_history",
        "shielding_history_MPa_sqrt_m",
        "source_remaining_history",
        "blunted_radius_history_um",
        "first_drive_factors",
    }
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in excluded and key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _plot_series(root: Path, cases: list[dict[str, Any]], key: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for case in cases:
        ax.plot(case["event_path_um"], case[key], marker="o", markersize=3, label=case["case"])
    ax.set_xlabel("Crack extension (µm)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / filename, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--theta", type=float, required=True)
    parser.add_argument("--target-extension-um", type=float, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    tag = f"T{args.temperature:g}_th{args.theta:g}"
    cases = [
        _load_case(root / name / tag, name, args.temperature, args.theta)
        for name in CASE_NAMES
    ]
    by_name = {case["case"]: case for case in cases}
    validated = by_name["anisotropic_validated_transport"]
    channel = by_name["anisotropic_channel_transport"]
    scalar = by_name["scalar_reference"]

    transport_pair = _pair_metrics(validated, channel, args.target_extension_um)
    validated_vs_scalar = _pair_metrics(scalar, validated, args.target_extension_um)
    channel_vs_scalar = _pair_metrics(scalar, channel, args.target_extension_um)

    first_factor_difference = float(np.max(np.abs(
        np.asarray(validated.get("first_drive_factors", [math.nan, math.nan]), dtype=float)
        - np.asarray(channel.get("first_drive_factors", [math.nan, math.nan]), dtype=float)
    )))

    assessment = {
        "schema": "v10.1.7.5_transport_comparison_500um",
        "temperature_K": float(args.temperature),
        "theta_deg": float(args.theta),
        "target_extension_um": float(args.target_extension_um),
        "all_cases_reached_target": all(
            case["projected_extension_um"] + 1.0e-6 >= args.target_extension_um
            for case in cases
        ),
        "anisotropic_tensor_drives_reliable": all(
            math.isclose(case.get("reliable_fraction", 0.0), 1.0, abs_tol=1.0e-12)
            for case in (validated, channel)
        ),
        "post_hazard_weighting_total": int(sum(
            case.get("post_hazard_weighting_count", 0)
            for case in (validated, channel)
        )),
        "first_drive_factor_max_abs_difference": first_factor_difference,
        "same_initial_anisotropic_drive": bool(first_factor_difference <= 1.0e-10),
        "transport_pair": transport_pair,
        "validated_transport_vs_scalar": validated_vs_scalar,
        "channel_transport_vs_scalar": channel_vs_scalar,
    }
    assessment["pilot_pass"] = all(
        [
            assessment["all_cases_reached_target"],
            assessment["anisotropic_tensor_drives_reliable"],
            assessment["post_hazard_weighting_total"] == 0,
            assessment["same_initial_anisotropic_drive"],
        ]
    )

    _write_csv(root / "transport_comparison_case_summary.csv", cases)
    (root / "transport_comparison_assessment.json").write_text(
        json.dumps(assessment, indent=2)
    )

    _plot_series(
        root,
        cases,
        "event_K_MPa_sqrt_m",
        "K at crack advance (MPa√m)",
        "transport_model_R_curve_comparison.png",
    )
    _plot_series(
        root,
        cases,
        "mobile_history",
        "Active mobile dislocation content",
        "transport_model_mobile_history.png",
    )
    _plot_series(
        root,
        cases,
        "retained_history",
        "Active retained dislocation content",
        "transport_model_retained_history.png",
    )
    _plot_series(
        root,
        cases,
        "shielding_history_MPa_sqrt_m",
        "Active shielding (MPa√m)",
        "transport_model_active_shielding.png",
    )
    _plot_series(
        root,
        cases,
        "emitted_history",
        "Cumulative emitted content",
        "transport_model_cumulative_emission.png",
    )
    _plot_series(
        root,
        cases,
        "source_remaining_history",
        "Remaining campaign source budget",
        "transport_model_source_budget.png",
    )

    print("\nv10.1.7.5 anisotropic transport comparison")
    print(f"common transport-pair path: {transport_pair['common_path_um']:.1f} µm")
    print(f"initial drive match: {assessment['same_initial_anisotropic_drive']}")
    print(
        "validated -> channel-resolved mean R-curve shift: "
        f"{transport_pair['mean_R_curve_shift_percent']:.3f}%"
    )
    print(
        "validated -> channel-resolved normalized R-curve RMS: "
        f"{transport_pair['normalized_R_curve_rms_percent']:.3f}%"
    )
    print(
        "validated -> channel-resolved max |ΔK|: "
        f"{transport_pair['maximum_abs_K_difference_MPa_sqrt_m']:.3f} MPa√m"
    )
    print(f"pilot_pass={assessment['pilot_pass']}")


if __name__ == "__main__":
    main()
