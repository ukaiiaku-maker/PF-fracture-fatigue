#!/usr/bin/env python3
"""Event-window convergence test for the native signed-bulk refit matrix.

The accepted matrix measures rate from the two waits between three 5 um
events (15 um total), while the bound blunting memory length is 17.35 um.
This analysis reads an extended same-material matrix and reports every wait,
checks that its first three events reproduce the accepted matrix, and scores
the accepted early window against a late window beyond one blunting length.
All outputs go to the extended matrix root; accepted evidence is only read.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_qste340tm_signed_bulk_2d_matrix import EVENT_LENGTH_M, interpolated
from scripts.refit_qste340tm_emission_owned_blunting import COLORS, R_VALUES

BASE = ROOT / "literature_data/fatigue_recovered_review/model_conditional/emission_owned_blunting"
ACCEPTED = BASE / "native_signed_bulk_refit_20261028/matrix"
EXTENDED = BASE / "native_signed_bulk_refit_20261028_extended_7events"
FIT_1D = BASE / "resolved_increment_1d/signed_bulk_joint_refit_20261028"
# Waits are indexed from 1: wait k separates events k and k+1.
WINDOWS = {"accepted_waits_1_2": (1, 2), "late_waits_4_6": (4, 6)}


def event_cycles(csv_path: Path) -> list[float]:
    total, events = 0.0, []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            total += float(row["fatigue_cycles"])
            if float(row["n_fire"]) >= 0.5:
                events.append(total)
    return events


def block_metrics(rows: list[dict], key: str, experiment: list[dict]) -> dict:
    out = {"slopes": {}, "block_biases_log10": {}, "errors": []}
    for r_value in R_VALUES:
        block = [row for row in rows if row["R"] == r_value and row[key] is not None]
        exp_block = [row for row in experiment if float(row["R"]) == r_value]
        if len(block) < 2:
            continue
        dk = np.asarray([row["DeltaK"] for row in block])
        rate = np.asarray([row[key] for row in block])
        out["slopes"][str(r_value)] = float(np.polyfit(np.log10(dk), np.log10(rate), 1)[0])
        errors = [math.log10(r / interpolated(exp_block, "observed_dadn_m_per_cycle", k))
                  for k, r in zip(dk, rate)]
        out["block_biases_log10"][str(r_value)] = float(np.mean(errors))
        out["errors"] += errors
    errors = np.asarray(out.pop("errors"))
    biases = out["block_biases_log10"].values()
    if not errors.size:
        return out | {"case_count": 0}
    out |= {"case_count": int(errors.size),
            "rmse_log10_vs_experiment": float(np.sqrt(np.mean(errors**2))),
            "bias_log10_vs_experiment": float(np.mean(errors)),
            "block_bias_span_log10": float(max(biases) - min(biases))}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended", type=Path, default=EXTENDED)
    parser.add_argument("--accepted", type=Path, default=ACCEPTED)
    args = parser.parse_args(argv)
    receipt = json.loads((args.extended / "MATRIX.json").read_text())
    accepted = {(float(r["R"]), float(r["DeltaK_MPa_sqrt_m"])): json.loads(r["event_cycles"])
                for r in csv.DictReader((args.accepted / "native_2d_rates.csv").open())}
    experiment = list(csv.DictReader((FIT_1D / "predictions_signed_bulk.csv").open()))
    rows = []
    for case in receipt["cases"]:
        key = (float(case["R"]), float(case["DeltaK"]))
        events = event_cycles(Path(case["csv"])) if case["status"] in {"completed", "reused"} else []
        waits = np.diff(events).tolist()
        row = {"R": key[0], "DeltaK": key[1], "status": case["status"],
               "event_cycles": events, "waits": waits,
               "first_three_events_match_accepted":
                   bool(len(events) >= 3 and np.allclose(events[:3], accepted[key], rtol=0, atol=1e-9))}
        for name, (first, last) in WINDOWS.items():
            window = waits[first - 1:last]
            row[name] = (EVENT_LENGTH_M / float(np.mean(window))
                         if len(window) == last - first + 1 else None)
        if row["accepted_waits_1_2"] and row["late_waits_4_6"]:
            row["late_over_accepted_log10"] = math.log10(row["late_waits_4_6"] / row["accepted_waits_1_2"])
        rows.append(row)
    rows.sort(key=lambda r: (r["R"], r["DeltaK"]))
    summary = {
        "schema": "qste340tm-signed-bulk-extended-event-window/1",
        "extended_matrix": str(args.extended), "accepted_matrix": str(args.accepted),
        "material_fit": receipt.get("fit"), "target_events": receipt.get("target_events"),
        "event_length_m": EVENT_LENGTH_M,
        "cases_reproducing_accepted_first_three_events":
            sum(r["first_three_events_match_accepted"] for r in rows),
        "windows": {name: block_metrics(rows, name, experiment) for name in WINDOWS},
        "cases": rows,
    }
    (args.extended / "EXTENDED_ANALYSIS.json").write_text(json.dumps(summary, indent=2) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    for r_value in R_VALUES:
        color = COLORS[r_value]
        block = [r for r in rows if r["R"] == r_value]
        exp_block = sorted((r for r in experiment if float(r["R"]) == r_value),
                           key=lambda r: float(r["DeltaK_MPa_sqrt_m"]))
        axes[0].plot([float(r["DeltaK_MPa_sqrt_m"]) for r in exp_block],
                     [float(r["observed_dadn_m_per_cycle"]) for r in exp_block],
                     color=color, lw=2.0, label=f"R={r_value:.1f} experiment")
        for name, marker in (("accepted_waits_1_2", "o"), ("late_waits_4_6", "*")):
            pts = [(r["DeltaK"], r[name]) for r in block if r[name]]
            if pts:
                axes[0].plot(*zip(*pts), marker=marker, ls="--" if marker == "*" else ":",
                             color=color, ms=9 if marker == "*" else 5)
        for r in block:
            if r["waits"]:
                axes[1].plot(range(1, len(r["waits"]) + 1),
                             np.asarray(r["waits"]) / r["waits"][0], color=color, alpha=0.75)
    axes[0].set(xscale="log", yscale="log", xlabel=r"$\Delta K$ (MPa$\sqrt{m}$)",
                ylabel=r"$da/dN$ (m/cycle)",
                title="o: waits 1-2 (accepted)   *: waits 4-6")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].axhline(1.0, color="0.6", lw=0.8)
    axes[1].set(xlabel="wait index (5 um each)", ylabel="wait / first wait",
                title="Inter-event wait evolution")
    figure.tight_layout()
    figure.savefig(args.extended / "extended_event_windows.png", dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
