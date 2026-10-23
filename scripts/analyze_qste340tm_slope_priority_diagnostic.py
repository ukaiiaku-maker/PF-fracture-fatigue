#!/usr/bin/env python3
"""Create clearly labelled slope-priority guidance from legacy state arrays."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.qste340tm_restored_binding import opening_surface, TEMPERATURE_K
from scripts.complete_qste340tm_active_calibration import prepare_observation_schedule
from scripts.refit_qste340tm_restored_native import (
    ALT_REGISTRY, DATA, PARENT_CANDIDATE, PARENT_REGISTRY, R_VALUES,
    _stress_trace, build_records, fit_opening, load_parent_binding, metrics,
)

BASE = DATA / "model_conditional/stateful_exp_floor_capacity_policy/restored_active_completion"
SOURCE = BASE / "completed_candidates/trapping_dominated_c224f83bc45d.json"
OUT = BASE / "slope_priority_continuation"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    records = prepare_observation_schedule(build_records("index_rank", premeasurement=True))
    legacy = json.loads(SOURCE.read_text())
    parent = load_parent_binding(ALT_REGISTRY, PARENT_REGISTRY, candidate_id=PARENT_CANDIDATE)
    stress = _stress_trace(records, legacy["states"], 8)
    fits = {
        mode: fit_opening(mode, parent.manifest.cleavage.surface, records, stress)
        for mode in ("slope_priority", "slope_priority_weak_offset")
    }
    observed = np.asarray([row["observed_dadn_m_per_cycle"] for row in records])
    r_values = np.asarray([row["R"] for row in records])
    delta_k = np.asarray([row["DeltaK_MPa_sqrt_m"] for row in records])
    rows = []
    for index, record in enumerate(records):
        row = {
            "series_id": record["series_id"], "source_order": record["source_order"],
            "R": record["R"], "DeltaK_MPa_sqrt_m": record["DeltaK_MPa_sqrt_m"],
            "observed_dadn_m_per_cycle": record["observed_dadn_m_per_cycle"],
        }
        for mode, fit in fits.items():
            prediction = float(fit["prediction"][index])
            block = fit["slope_priority_evaluation"]["blocks"][str(record["R"])]
            row[f"{mode}_prediction_m_per_cycle"] = prediction
            row[f"{mode}_raw_residual_log10"] = float(np.log10(prediction / observed[index]))
            row[f"{mode}_centered_residual_log10"] = (
                row[f"{mode}_raw_residual_log10"] - block["bias_log10_rate"]
            )
        rows.append(row)
    write_csv(OUT / "legacy_guidance_predictions.csv", rows)

    colors = {0.1: "#0072B2", 0.3: "#E69F00", 0.5: "#009E73", 0.7: "#CC79A7"}
    weak = np.asarray(fits["slope_priority_weak_offset"]["prediction"])
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.7), sharex=True, sharey=True)
    for axis, r_value in zip(axes.flat, R_VALUES):
        mask = r_values == r_value
        order = np.argsort(delta_k[mask])
        axis.scatter(delta_k[mask], observed[mask], s=10, color="#444444", alpha=0.45, label="measured")
        axis.plot(delta_k[mask][order], weak[mask][order], color=colors[r_value], lw=2,
                  label="legacy-state slope-priority guidance")
        axis.set(xscale="log", yscale="log", xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)")
        axis.text(0.04, 0.93, f"R = {r_value:.1f}", transform=axis.transAxes, va="top")
    axes[0, 0].set_ylabel("da/dN (m/cycle)")
    axes[1, 0].set_ylabel("da/dN (m/cycle)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "raw_curves_common_axes.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.7), sharex=True, sharey=True)
    for axis, r_value in zip(axes.flat, R_VALUES):
        mask = r_values == r_value
        order = np.argsort(delta_k[mask])
        obs_center = np.log10(observed[mask]) - np.mean(np.log10(observed[mask]))
        pred_center = np.log10(weak[mask]) - np.mean(np.log10(weak[mask]))
        axis.plot(delta_k[mask][order], obs_center[order], color="#444444", lw=1.4, label="measured centered")
        axis.plot(delta_k[mask][order], pred_center[order], color=colors[r_value], lw=2,
                  label="model centered")
        axis.set(xscale="log", xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)")
        axis.text(0.04, 0.93, f"R = {r_value:.1f}", transform=axis.transAxes, va="top")
    axes[0, 0].set_ylabel("centered log10(da/dN)")
    axes[1, 0].set_ylabel("centered log10(da/dN)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "centered_curves_common_axes.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    evaluation = fits["slope_priority_weak_offset"]["slope_priority_evaluation"]
    slope_rows = []
    for r_value in R_VALUES:
        block = evaluation["blocks"][str(r_value)]
        slope_rows.append({
            "R": r_value,
            "observed_global": block["observed_global_slope"],
            "predicted_global": block["predicted_global_slope"],
            **{f"observed_region_{i+1}": value for i, value in enumerate(block["observed_regional_slopes"])},
            **{f"predicted_region_{i+1}": value for i, value in enumerate(block["predicted_regional_slopes"])},
            "bias_log10_rate": block["bias_log10_rate"],
            "centered_rmse_log10": block["centered_rmse_log10"],
            "shape_pass": block["shape_pass"],
            "worst_normalized_violation": block["worst_normalized_violation"],
        })
    write_csv(OUT / "slope_and_offset_summary.csv", slope_rows)

    labels = ("global", "low", "middle", "high")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), sharey=True)
    for axis, row in zip(axes.flat, slope_rows):
        target = [row["observed_global"], row["observed_region_1"],
                  row["observed_region_2"], row["observed_region_3"]]
        model = [row["predicted_global"], row["predicted_region_1"],
                 row["predicted_region_2"], row["predicted_region_3"]]
        x = np.arange(4)
        axis.bar(x - 0.18, target, width=0.36, color="#555555", label="measured")
        axis.bar(x + 0.18, model, width=0.36, color=colors[row["R"]], label="model")
        axis.set(xticks=x, xticklabels=labels, xlabel="regression window")
        axis.text(0.04, 0.93, f"R = {row['R']:.1f}", transform=axis.transAxes, va="top")
    axes[0, 0].set_ylabel("log-log slope")
    axes[1, 0].set_ylabel("log-log slope")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "global_and_regional_slopes.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    coordinates = fits["slope_priority_weak_offset"]["coordinates"]
    from arrhenius_fracture.qste340tm_restored_binding import OpeningCoordinates
    surface = opening_surface(parent.manifest.cleavage.surface, OpeningCoordinates(**coordinates))
    sigma = np.linspace(0.0, 30.0e9, 1001)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), sharex=True)
    axes[0, 0].plot(sigma / 1e9, surface.G_eV(sigma, TEMPERATURE_K), color="#0072B2")
    axes[0, 0].set(ylabel=r"$G^*$ (eV)")
    axes[0, 1].plot(sigma / 1e9, surface.activation_volume_m3(sigma, TEMPERATURE_K), color="#D55E00")
    axes[0, 1].set(ylabel=r"$V^*$ (m$^3$)")
    axes[1, 0].plot(sigma / 1e9, surface.enthalpy_eV(sigma, TEMPERATURE_K), color="#009E73")
    axes[1, 0].set(ylabel=r"$H^*$ (eV)", xlabel="opening stress (GPa)")
    axes[1, 1].plot(sigma / 1e9, surface.entropy_kB(sigma, TEMPERATURE_K), color="#CC79A7")
    axes[1, 1].set(ylabel=r"$S^*/k_B$", xlabel="opening stress (GPa)")
    axes[0, 0].set_xlabel("opening stress (GPa)")
    axes[0, 1].set_xlabel("opening stress (GPa)")
    fig.tight_layout()
    fig.savefig(OUT / "barrier_thermodynamic_audit.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    result = {
        "schema": "qste340tm-slope-priority-legacy-guidance/1",
        "role": "legacy stale-state diagnostic only; not a repaired native calibration",
        "source_candidate": str(SOURCE.resolve()),
        "observation_count": len(records),
        "fits": {
            mode: {
                "coordinates": fit["coordinates"],
                "metrics": metrics(records, np.asarray(fit["prediction"])),
                "slope_priority_evaluation": fit["slope_priority_evaluation"],
                "barrier_qualification": fit["barrier_qualification"],
                "solver_nfev": fit["solver_nfev"],
            } for mode, fit in fits.items()
        },
        "repaired_native_runs": [],
    }
    for progress_path in sorted((BASE / "restarts").glob("trapping_dominated_*/outer_progress.json")):
        progress = json.loads(progress_path.read_text())
        payload = progress.get("identity", {}).get("payload", {})
        if payload.get("schema") != "qste340tm-restored-active-chronological/2":
            continue
        result["repaired_native_runs"].append({
            "progress_path": str(progress_path.resolve()),
            "identity_sha256": progress["identity"]["sha256"],
            "status": progress["status"],
            "completed_observations": len(progress["completed_observation_rows"]),
            "required_observations": 760,
            "current_r_index": progress["current_r_index"],
            "current_observation_position": progress["current_observation_position"],
            "current_load_substep": progress["current_load_substep"],
            "phase_count": payload["phase_count"],
            "load_substeps": payload["load_substeps"],
            "calibration_eligible_when_complete": payload["calibration_eligible"],
            "resume_command": (
                "python scripts/complete_qste340tm_active_calibration.py "
                f"--candidate trapping_dominated --phase-count {payload['phase_count']} "
                f"--load-substeps {payload['load_substeps']} --support-points 0 "
                "--frequency-hz 10 --alignment index_rank --slice-seconds 600 --max-slices 1 "
                "--state-opening-seed-result literature_data/fatigue_recovered_review/"
                "model_conditional/stateful_exp_floor_capacity_policy/restored_active_completion/"
                "completed_candidates/trapping_dominated_c224f83bc45d.json "
                "--state-opening-seed-mode shape"
            ),
        })
    (OUT / "legacy_guidance_assessment.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
