#!/usr/bin/env python3
"""Run and preserve active no-void and reflected prescribed-hole trajectories."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from scripts.run_off_axis_prescribed_void_benchmark_v2 import RADIUS, finite, solve_case
from scripts.screen_informative_off_axis_operating_point import (
    LOAD_SCALE, MATERIAL, TEMPERATURE_K, centers,
)

OUT = ROOT/"artifacts/informative_off_axis_active_trajectories"
CENTERS = centers(2.2, 1.4)
CASES = ("no_void", "above", "below")


def dump(path: Path, value):
    path.write_text(json.dumps(finite(value), indent=2, sort_keys=True)+"\n")


def run_saved(case: str):
    result_path = OUT/f"{case}.json"
    checkpoint_path = OUT/f"{case}_latest_valid_restart.json"
    if result_path.exists() and checkpoint_path.exists():
        return json.loads(result_path.read_text())
    result, state = solve_case(
        case, run_scheduler=True, include_finite_diagnostic=False,
        material_family=MATERIAL, temperature_K=TEMPERATURE_K,
        load_scale=LOAD_SCALE, cavity_centers=CENTERS,
        maximum_observation_s=1.0e-3,
        observation_crossing_multiples=4.0, maximum_events=8,
        return_state=True,
    )
    dump(result_path, result)
    manifest = write_checkpoint(state, checkpoint_path, compression="gzip")
    restored = restore_checkpoint(checkpoint_path)
    receipt = {
        "checkpoint_manifest": checkpoint_path.name,
        "state_file": manifest["state_file"],
        "direct_fingerprint": complete_accepted_state_fingerprint(state),
        "restored_fingerprint": complete_accepted_state_fingerprint(restored),
        "direct_restored_equal": (
            complete_accepted_state_fingerprint(state)
            == complete_accepted_state_fingerprint(restored)),
    }
    dump(OUT/f"{case}_restart_receipt.json", receipt)
    print("TRAJECTORY_COMPLETE", case, flush=True)
    return result


def make_plot(results):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)
    for ax, case in zip(axes, CASES):
        result = results[case]
        for path in result["scheduler"]["terminal_crack_paths_m"]:
            xy = np.asarray(path, dtype=float)
            ax.plot(xy[:, 0]*1e6, xy[:, 1]*1e6, "o-", lw=1.7, ms=3)
        if case != "no_void":
            center = CENTERS[case]
            ax.add_patch(Circle((center[0]*1e6, center[1]*1e6),
                                RADIUS*1e6, fill=False, lw=1.5, color="tab:red"))
        ax.axhline(0.0, color="0.8", lw=0.7)
        ax.set_title(case.replace("_", " "))
        ax.set_xlabel("x (µm)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("y (µm)")
    fig.suptitle("Active ceramic-like reflected off-axis trajectories (domain-sensitive)")
    fig.tight_layout()
    fig.savefig(OUT/"crack_paths.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {case: run_saved(case) for case in CASES}
    make_plot(results)
    summary = {
        "classification": "EXPLORATORY_DOMAIN_SENSITIVE_INTERACTION_TRAJECTORIES",
        "material_family": MATERIAL, "temperature_K": TEMPERATURE_K,
        "load_scale": LOAD_SCALE,
        "selected_geometry": {"d_over_R": 2.2, "h_over_R": 1.4,
                              "straight_clearance_m": 2.2e-5},
        "domain_sensitivity_status": "COMPLETED_SENSITIVITY_NOT_CONVERGED",
        "cases": {
            case: {
                "path_or_stopping_condition": result["scheduler"]["path_or_stopping_condition"],
                "observation_window_s": result["scheduler"]["finite_observation_window_s"],
                "state_bound_directional_observation_count": result["scheduler"]["state_bound_directional_observation_count"],
                "selected_events": result["scheduler"]["selected_events"],
                "terminal_crack_paths_m": result["scheduler"]["terminal_crack_paths_m"],
                "restart": f"{case}_latest_valid_restart.json",
            } for case, result in results.items()
        },
        "plot": "crack_paths.png",
    }
    dump(OUT/"summary.json", summary)


if __name__ == "__main__":
    main()
