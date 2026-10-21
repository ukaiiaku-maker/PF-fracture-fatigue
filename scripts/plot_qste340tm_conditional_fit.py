#!/usr/bin/env python3
"""Plot the QSTE340TM conditional opening fit and its room-temperature G*."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.material_manifest import KB_EV_PER_K
from scripts.calibrate_qste340tm_opening_reference import (
    PHASE_STEPS,
    SMOOTHING_RATIO,
    TIP_RADIUS_M,
    predict_log10_rate,
)
from scripts.fatigue_informed_joint_barrier_v3_v10230 import SmoothBroadBasis


TEMPERATURE_C = 25.0
TEMPERATURE_K = 298.15


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def vector(parameters: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [
            parameters["g_floor_Gstar_over_kBT"],
            parameters["g_amplitude_Gstar_over_kBT"],
            math.log(parameters["sigma_b_Pa"] / 1.0e9),
            math.log(parameters["exponent"]),
        ]
    )


def stress_history(delta_k: np.ndarray, load_ratio: float) -> np.ndarray:
    phase = 2.0 * math.pi * (np.arange(PHASE_STEPS) + 0.5) / PHASE_STEPS
    fraction = (
        0.5 * (1.0 + load_ratio)
        + 0.5 * (1.0 - load_ratio) * np.cos(phase)
    )
    kmax = delta_k / (1.0 - load_ratio)
    return (
        kmax[:, None]
        * 1.0e6
        * fraction[None, :]
        / math.sqrt(2.0 * math.pi * TIP_RADIUS_M)
    )


def barrier_eV(parameters: dict[str, float], stress_Pa: np.ndarray) -> np.ndarray:
    basis = SmoothBroadBasis(
        sigma_b_Pa=parameters["sigma_b_Pa"],
        exponent=parameters["exponent"],
        smoothing_ratio=SMOOTHING_RATIO,
    )
    dimensionless = (
        parameters["g_floor_Gstar_over_kBT"]
        + parameters["g_amplitude_Gstar_over_kBT"] * basis.value(stress_Pa)
    )
    return dimensionless * KB_EV_PER_K * TEMPERATURE_K


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("literature_data/fatigue_recovered_review")
    )
    args = parser.parse_args()
    root = args.data_root.resolve()
    conditional = root / "model_conditional"
    fit_path = conditional / "conditional_opening_reference.json"
    fit = json.loads(fit_path.read_text())
    rows = load_csv(root / "normalized/qste340tm_constant_amplitude.csv")
    fits = {float(item["frequency_Hz"]): item for item in fit["frequency_sensitivity"]}

    colors = {0.1: "#0072B2", 0.3: "#D55E00", 0.5: "#009E73", 0.7: "#CC79A7"}
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.0), sharex=False, sharey=True)
    for axis, r_value in zip(axes.flat, sorted(colors)):
        block = [row for row in rows if float(row["R"]) == r_value]
        delta_k = np.asarray([float(row["reported_DeltaK_MPa_sqrt_m"]) for row in block])
        observed = np.asarray([float(row["dadn_m_per_cycle"]) for row in block])
        grid = np.geomspace(delta_k.min(), delta_k.max(), 240)
        stress = stress_history(grid, r_value)
        curves = {}
        for frequency, item in fits.items():
            curves[frequency] = 10.0 ** predict_log10_rate(
                vector(item["parameters"]), stress, frequency
            )
        low = np.minimum(curves[1.0], curves[20.0])
        high = np.maximum(curves[1.0], curves[20.0])
        axis.fill_between(grid, low, high, color="#999999", alpha=0.24, label="1–20 Hz refit span")
        axis.plot(grid, curves[10.0], color="#111111", linewidth=2.0, label="10 Hz fit")
        axis.scatter(
            delta_k, observed, s=12, color=colors[r_value], alpha=0.72,
            edgecolors="none", label="measured/author-reduced",
        )
        metric = fit["nominal_fit"]["block_metrics"][str(r_value)]
        axis.text(
            0.04, 0.95,
            f"R = {r_value:.1f}\n$m_{{obs}}$ = {metric['observed_loglog_slope']:.2f}\n"
            f"$m_{{fit}}$ = {metric['predicted_loglog_slope']:.2f}",
            transform=axis.transAxes, va="top", fontsize=9,
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.22)
        axis.set_xlabel(r"$\Delta K$ (MPa$\sqrt{\mathrm{m}}$)")
    axes[0, 0].set_ylabel(r"$da/dN$ (m cycle$^{-1}$)")
    axes[1, 0].set_ylabel(r"$da/dN$ (m cycle$^{-1}$)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("QSTE340TM constant-amplitude data and conditional opening-reference fit", y=1.045)
    fig.text(
        0.5, 0.005,
        "10 Hz nominal; 1–20 Hz curves refitted independently. Temperature = 298.15 K.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.98))
    for extension in ("png", "svg"):
        fig.savefig(conditional / f"qste340tm_deltaK_dadN_fit.{extension}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    stress_bounds = fit["data_informed_model_stress_interval_Pa"]
    stress = np.linspace(0.0, stress_bounds[1] * 1.04, 600)
    fig, axis = plt.subplots(figsize=(8.4, 5.6))
    styles = {1.0: ("#56B4E9", "--"), 10.0: ("#111111", "-"), 20.0: ("#E69F00", "-.")}
    barrier_ranges = {}
    for frequency in (1.0, 10.0, 20.0):
        parameters = fits[frequency]["parameters"]
        barrier = barrier_eV(parameters, stress)
        barrier_ranges[str(frequency)] = [float(barrier.min()), float(barrier.max())]
        color, linestyle = styles[frequency]
        axis.plot(stress / 1.0e9, barrier, color=color, linestyle=linestyle,
                  linewidth=2.1, label=f"{frequency:g} Hz refit")
    axis.axvspan(stress_bounds[0] / 1.0e9, stress_bounds[1] / 1.0e9,
                 color="#999999", alpha=0.12, label="data-informed stress interval")
    axis.set_xlabel(r"local opening stress $\sigma$ (GPa)")
    axis.set_ylabel(r"effective $G^*(\sigma, 298.15\,\mathrm{K})$ (eV)")
    axis.set_title("Conditional QSTE340TM opening barrier at 25 °C")
    axis.grid(True, alpha=0.24)
    axis.legend(frameon=False)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(conditional / f"qste340tm_Gstar_vs_stress_298p15K.{extension}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    binding = {
        "schema": "qste340tm-room-temperature-binding/1",
        "temperature_C": TEMPERATURE_C,
        "temperature_K": TEMPERATURE_K,
        "temperature_source": "user supplied",
        "conditional_calibration": str(fit_path.relative_to(ROOT)),
        "conditional_calibration_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
        "frequency_nominal_Hz": 10.0,
        "frequency_sensitivity_Hz": [1.0, 10.0, 20.0],
        "Gstar_eV_ranges_over_plotted_stress": barrier_ranges,
        "data_informed_model_stress_interval_Pa": stress_bounds,
        "figures": [
            "qste340tm_deltaK_dadN_fit.png",
            "qste340tm_deltaK_dadN_fit.svg",
            "qste340tm_Gstar_vs_stress_298p15K.png",
            "qste340tm_Gstar_vs_stress_298p15K.svg",
        ],
        "claim_limit": "Absolute eV scaling uses the user-supplied 298.15 K, but the opening-reference fit remains non-transferable across R and is not a complete steel material.",
    }
    (conditional / "room_temperature_binding.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(binding, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
