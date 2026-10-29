#!/usr/bin/env python3
"""Analyze the paired 1-D and native 2-D signed-bulk QSTE340TM result."""
from __future__ import annotations

import csv
import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.refit_qste340tm_emission_owned_blunting import COLORS, R_VALUES

BASE = ROOT / "literature_data/fatigue_recovered_review/model_conditional/emission_owned_blunting"
FIT = BASE / "resolved_increment_1d/signed_bulk_joint_refit_20261028"
MATRIX = BASE / "native_signed_bulk_matrix_20261028"
OLD_MATRIX = BASE / "full_2d_released_sloping_matrix/native_2d_rates.csv"
EVENT_LENGTH_M = 5.0e-6


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def interpolated(block: list[dict[str, str]], field: str, target: float) -> float:
    x = np.asarray([float(row["DeltaK_MPa_sqrt_m"]) for row in block])
    y = np.asarray([float(row[field]) for row in block])
    order = np.argsort(x)
    return float(10 ** np.interp(np.log10(target), np.log10(x[order]), np.log10(y[order])))


def analyze_native(entry: dict) -> dict:
    trace = read_csv(Path(entry["csv"]))
    cycles = np.cumsum([float(row["fatigue_cycles"]) for row in trace])
    event_cycles = [
        float(cycles[index]) for index, row in enumerate(trace)
        if float(row["n_fire"]) >= 0.5
    ]
    waits = np.diff(event_cycles)
    if len(waits) >= 1:
        window = float(np.mean(waits))
        rate = EVENT_LENGTH_M / window
        classification = "developed_interevent"
    elif event_cycles:
        window = math.nan
        rate = EVENT_LENGTH_M / event_cycles[0]
        classification = "startup_secant_only"
    else:
        window = float(cycles[-1])
        rate = EVENT_LENGTH_M / window
        classification = "runout_upper_bound"
    def values(name: str) -> list[float]:
        return [float(row.get(name, 0.0) or 0.0) for row in trace]
    return {
        "R": float(entry["R"]), "DeltaK_MPa_sqrt_m": float(entry["DeltaK"]),
        "accepted_cycles": float(cycles[-1]),
        "event_cycles": json.dumps(event_cycles), "event_count": len(event_cycles),
        "developed_window_cycles": window,
        "growth_rate_m_per_cycle": rate, "classification": classification,
        "minimum_KJ_MPa_sqrt_m": min(values("KJ_Pa_sqrtm")) / 1e6,
        "maximum_KJ_MPa_sqrt_m": max(values("KJ_Pa_sqrtm")) / 1e6,
        "minimum_abs_reaction_N_per_m": min(abs(value) for value in values("Ftop_N")),
        "maximum_abs_reaction_N_per_m": max(abs(value) for value in values("Ftop_N")),
        "final_radius_m": values("mpz_blunted_radius_m")[-1],
        "maximum_radius_m": max(values("mpz_blunted_radius_m")),
        "final_bulk_tip_opening_count": values("mpz_bulk_tip_opening_slip_count")[-1],
        "cumulative_bulk_opening_absorbed": values("mpz_bulk_tip_absorbed_opening_total")[-1],
        "cumulative_bulk_absolute_absorbed": values("mpz_bulk_tip_absorbed_absolute_total")[-1],
        "cyclic_plastic_work_J": values("cyclic_plastic_work_J")[-1],
        "source_csv": entry["csv"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--fit", type=Path, default=FIT)
    parser.add_argument("--out", type=Path, default=None,
                        help="Write derived files here; defaults to the input matrix.")
    args = parser.parse_args(argv)
    matrix = args.matrix
    fit = args.fit
    out = matrix if args.out is None else args.out
    out.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((matrix / "MATRIX.json").read_text())
    native = [analyze_native(entry) for entry in receipt["cases"]]
    native.sort(key=lambda row: (row["R"], row["DeltaK_MPa_sqrt_m"]))
    with (out / "native_2d_rates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(native[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(native)
    (out / "native_2d_rates.json").write_text(
        json.dumps(native, indent=2, sort_keys=True) + "\n"
    )

    one_d_bulk = read_csv(fit / "predictions_signed_bulk.csv")
    one_d_no_bulk = read_csv(fit / "predictions_no_bulk.csv")
    old_native = read_csv(OLD_MATRIX)
    comparison = []
    slopes = {}
    biases = {}
    figure, axis = plt.subplots(figsize=(10.2, 7.0))
    for r_value in R_VALUES:
        block = [row for row in one_d_bulk if float(row["R"]) == r_value]
        no_block = [row for row in one_d_no_bulk if float(row["R"]) == r_value]
        x = np.asarray([float(row["DeltaK_MPa_sqrt_m"]) for row in block])
        observed = np.asarray([float(row["observed_dadn_m_per_cycle"]) for row in block])
        predicted = np.asarray([float(row["predicted_dadn_m_per_cycle"]) for row in block])
        no_predicted = np.asarray([float(row["predicted_dadn_m_per_cycle"]) for row in no_block])
        order = np.argsort(x)
        color = COLORS[r_value]
        axis.plot(x[order], observed[order], color=color, lw=2.1,
                  label=f"R={r_value:.1f} experiment")
        axis.plot(x[order], predicted[order], color=color, lw=1.0, ls=":", alpha=0.8)
        axis.scatter(x[::max(len(x)//18, 1)], predicted[::max(len(x)//18, 1)],
                     facecolors="white", edgecolors=color, s=23, linewidths=0.9)

        rows = [row for row in native if row["R"] == r_value]
        nx = np.asarray([row["DeltaK_MPa_sqrt_m"] for row in rows])
        ny = np.asarray([row["growth_rate_m_per_cycle"] for row in rows])
        norder = np.argsort(nx)
        axis.plot(nx[norder], ny[norder], color=color, lw=1.25, ls="--")
        axis.scatter(nx, ny, marker="*", s=145, color=color,
                     edgecolors="black", linewidths=0.5, zorder=6,
                     label=f"R={r_value:.1f} signed-bulk 2-D")
        old = [row for row in old_native if float(row["R"]) == r_value]
        ox = np.asarray([float(row["DeltaK_MPa_sqrt_m"]) for row in old])
        oy = np.asarray([float(row["growth_rate_m_per_cycle"]) for row in old])
        axis.scatter(ox, oy, marker="x", s=42, color=color, linewidths=1.1,
                     alpha=0.55)

        slopes[str(r_value)] = float(np.polyfit(np.log10(nx), np.log10(ny), 1)[0])
        errors = []
        for row in rows:
            dk = row["DeltaK_MPa_sqrt_m"]
            exp_rate = interpolated(block, "observed_dadn_m_per_cycle", dk)
            fit_rate = interpolated(block, "predicted_dadn_m_per_cycle", dk)
            no_rate = interpolated(no_block, "predicted_dadn_m_per_cycle", dk)
            error = math.log10(row["growth_rate_m_per_cycle"] / exp_rate)
            errors.append(error)
            comparison.append(dict(row) | {
                "experimental_rate_m_per_cycle": exp_rate,
                "signed_bulk_1d_rate_m_per_cycle": fit_rate,
                "no_bulk_1d_rate_m_per_cycle": no_rate,
                "log10_2d_over_experiment": error,
                "log10_2d_over_signed_bulk_1d": math.log10(
                    row["growth_rate_m_per_cycle"] / fit_rate
                ),
            })
        biases[str(r_value)] = float(np.mean(errors))

    axis.set(xscale="log", yscale="log", xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)",
             ylabel=r"$da/dN$ (m/cycle)",
             title="QSTE340TM signed-bulk refit and native 2-D transfer")
    axis.grid(False)
    axis.legend(ncol=2, fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(out / "experiment_1d_2d_signed_bulk_overlay.png", dpi=240,
                   bbox_inches="tight")
    figure.savefig(out / "experiment_1d_2d_signed_bulk_overlay.svg",
                   bbox_inches="tight")
    plt.close(figure)

    with (out / "rate_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(comparison)
    errors = np.asarray([row["log10_2d_over_experiment"] for row in comparison])
    transfer = np.asarray([row["log10_2d_over_signed_bulk_1d"] for row in comparison])
    analysis = {
        "schema": "qste340tm-native-signed-bulk-analysis/1",
        "case_count": len(native),
        "developed_interevent_count": sum(
            row["classification"] == "developed_interevent" for row in native
        ),
        "native_2d_rmse_log10_vs_experiment": float(np.sqrt(np.mean(errors ** 2))),
        "native_2d_bias_log10_vs_experiment": float(np.mean(errors)),
        "native_2d_rmse_log10_vs_signed_bulk_1d": float(np.sqrt(np.mean(transfer ** 2))),
        "native_2d_bias_log10_vs_signed_bulk_1d": float(np.mean(transfer)),
        "native_2d_slopes": slopes,
        "native_2d_block_biases_log10_vs_experiment": biases,
        "block_bias_span_log10": max(biases.values()) - min(biases.values()),
        "loading_control": "fixed_local_kmax",
        "continuum_bulk_plasticity": True,
        "signed_GND_tip_absorption": True,
        "material_fit": str(receipt.get("fit", fit / "fit.json")),
    }
    (out / "ANALYSIS.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

