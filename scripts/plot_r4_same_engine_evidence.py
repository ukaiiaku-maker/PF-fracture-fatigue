#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    coupled = json.loads(
        (root / "current_coupled_run2" / "current_coupled_r4.json").read_text()
    )
    fatigue = json.loads(
        (root / "resolved_stochastic_fatigue_run2" / "result.json").read_text()
    )
    spatial = list(csv.DictReader(
        (root / "current_law_spatial" / "steps_0900K.csv").open()
    ))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = axes[0, 0]
    for name, label in (("no_void", "void disabled"),
                        ("passive_feedback_disabled", "passive cavity"),
                        ("active_void", "active cavity")):
        rows = coupled["arms"][name]["history"]
        ax.plot(range(1, len(rows) + 1),
                [row["after"]["geometry_signature"][3] * 1e6 for row in rows],
                marker="o", label=label)
    ax.set(xlabel="accepted external stage", ylabel="checkpoint extension (µm)",
           title="Current-backend coupled fixture")
    ax.legend()

    ax = axes[0, 1]
    active = coupled["arms"]["active_void"]["history"]
    no_void = coupled["arms"]["no_void"]["history"]
    ax.plot(range(1, len(active) + 1),
            [row["consumed_K_Pa_sqrt_m"] * 1e-6 for row in active],
            marker="o", label="active cavity input")
    ax.plot(range(1, len(no_void) + 1),
            [row["consumed_K_Pa_sqrt_m"] * 1e-6 for row in no_void],
            marker="s", label="void-disabled input")
    ax.set(xlabel="accepted external stage", ylabel="consumed K (MPa√m)",
           title="Measured fixed-void mechanics feedback")
    ax.legend()

    ax = axes[1, 0]
    ax.plot([float(row["Uapp_m"]) * 1e6 for row in spatial],
            [(float(row["a_tip_m"]) - 5.0e-4) * 1e6 for row in spatial],
            marker="o")
    for row in spatial:
        ax.annotate(str(int(float(row["step"]))),
                    (float(row["Uapp_m"]) * 1e6,
                     (float(row["a_tip_m"]) - 5.0e-4) * 1e6))
    ax.set(xlabel="prescribed opening (µm)", ylabel="projected extension (µm)",
           title="Fresh current-law spatial load/unload/reload")

    ax = axes[1, 1]
    a_n = fatigue["a_N"]
    ax.step([row["cycle"] for row in a_n],
            [row["a_m"] * 1e6 for row in a_n], where="post")
    ax.scatter([row["cycle"] for row in fatigue["events"]],
               [row["extension_m"] * 1e6 for row in fatigue["events"]],
               color="tab:red", s=18, label="native reward")
    ax.set(xlabel="cycle", ylabel="checkpoint extension (µm)",
           title="Resolved stochastic recurrent growth")
    ax.legend()

    path = root / "r4_combined_results.png"
    fig.savefig(path, dpi=180)
    print(path)


if __name__ == "__main__":
    main()
