#!/usr/bin/env python3
"""Render an inspectable final topology and energy history for a v11 case."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint


def _energy_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def plot_case(case: str | Path) -> Path:
    root = Path(case).resolve()
    checkpoint = restore_branch_checkpoint(root / "checkpoint" / "latest.json")
    rows = _energy_rows(root / "energy_ledger.csv")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    network = checkpoint.state.crack_network
    all_x = [point[0] * 1e6 for branch in network.branches for point in branch.path]
    all_y = [point[1] * 1e6 for branch in network.branches for point in branch.path]
    for branch in network.branches:
        x = [point[0] * 1e6 for point in branch.path]
        y = [point[1] * 1e6 for point in branch.path]
        style = "-" if branch.status == "active" else "--"
        axes[0].plot(x, y, style, marker="o", ms=3, lw=2, label=f"{branch.branch_id} ({branch.status})")
    axes[0].set_title("Final exact crack topology")
    axes[0].set_xlabel("x (µm)")
    axes[0].set_ylabel("y (µm)")
    axes[0].set_aspect("equal", adjustable="box")
    if all_x:
        xmax = max(all_x)
        axes[0].set_xlim(max(min(all_x), xmax - 75.0), xmax + 5.0)
        ymax = max((abs(value) for value in all_y), default=0.0)
        axes[0].set_ylim(-max(12.0, ymax + 5.0), max(12.0, ymax + 5.0))
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, loc="best")

    if rows:
        step = [int(row["step"]) for row in rows]
        stored = [float(row["stored_energy_J_per_m"]) for row in rows]
        released = [float(row["released_energy_J_per_m"]) for row in rows]
        cost = [float(row["dissipative_cost_J_per_m"]) for row in rows]
        axes[1].plot(step, stored, label="stored")
        axes[1].plot(step, released, label="released/action")
        axes[1].plot(step, cost, label="dissipative cost/action")
    axes[1].set_title("Accepted energy history")
    axes[1].set_xlabel("accepted step")
    axes[1].set_ylabel("energy (J/m)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="best")

    status = checkpoint.termination_reason or "incomplete checkpoint"
    figure.suptitle(f"{root.name}\n{status}", fontsize=11)
    output = root / "v11_final_topology_and_energy.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    metadata = {
        "schema": "v11.visual-snapshot/1", "image": output.name,
        "termination_reason": checkpoint.termination_reason,
        "active_tip_ids": list(network.active_tip_ids),
        "branch_count": len(network.branches),
        "topology_fingerprint": checkpoint.topology_fingerprint,
    }
    (root / "v11_visual_snapshot.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="+")
    args = parser.parse_args(argv)
    for case in args.cases:
        print(plot_case(case))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
