#!/usr/bin/env python3
"""Plot live high-cycle checkpoint history and the latest signed MPZ state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _field(payload: dict, vector: np.ndarray, name: str):
    for field in payload.get("fields", []):
        if field.get("name") == name:
            start = int(field["start"]); stop = int(field["stop"])
            shape = tuple(int(value) for value in field.get("shape", []))
            data = vector[start:stop]
            return data.reshape(shape) if shape else data
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args(argv)
    root = args.run_root.resolve()
    checkpoint_path = root / "high_cycle_live_checkpoint.json"
    state_path = root / "high_cycle_live_state.npz"
    if not checkpoint_path.is_file() or not state_path.is_file():
        raise SystemExit("no live high-cycle checkpoint is available")
    checkpoint = json.loads(checkpoint_path.read_text())
    vector = np.asarray(np.load(state_path)["active_vector"], dtype=float)
    history = _history(root / "high_cycle_live_history.jsonl")
    outputs = []

    if history:
        cycles = [row.get("cycles_from_engine_time") for row in history]
        valid = [index for index, value in enumerate(cycles) if value is not None]
        if valid:
            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            ax.plot(
                [index + 1 for index in valid],
                [float(cycles[index]) for index in valid],
                marker="o",
                markersize=3,
            )
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_xlabel("Committed checkpoint")
            ax.set_ylabel("Cycles from engine time")
            ax.set_title("Live high-cycle committed progress")
            ax.grid(True, which="both", alpha=0.25)
            path = root / "live_high_cycle_progress.png"
            fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
            outputs.append(path.name)

        names = ("mobile_count", "retained_count", "sigma_back_Pa", "lambda_cleave_s")
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        plotted = False
        for name in names:
            x = []
            y = []
            for row in history:
                value = row.get("diagnostics", {}).get(name)
                cycle = row.get("cycles_from_engine_time")
                if value is not None and cycle is not None:
                    x.append(float(cycle)); y.append(float(value))
            if x:
                ax.plot(x, y, marker="o", markersize=2, label=name)
                plotted = True
        if plotted:
            ax.set_xscale("symlog", linthresh=1.0)
            ax.set_yscale("symlog", linthresh=1.0e-30)
            ax.set_xlabel("Cycles")
            ax.set_ylabel("Diagnostic value")
            ax.set_title("Live mechanical and MPZ diagnostics")
            ax.legend()
            ax.grid(True, which="both", alpha=0.25)
            path = root / "live_mechanical_response.png"
            fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
            outputs.append(path.name)
        else:
            plt.close(fig)

    arrays = {}
    for name in (
        "mobile_positive", "mobile_negative",
        "retained_positive", "retained_negative",
        "accumulated_slip_positive", "accumulated_slip_negative",
    ):
        value = _field(checkpoint, vector, name)
        if value is not None:
            arrays[name] = np.asarray(value, dtype=float).reshape(-1)
    if arrays:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for name, values in arrays.items():
            ax.plot(np.arange(values.size), values, label=name)
        ax.set_xlabel("MPZ bin")
        ax.set_ylabel("Signed-state channel magnitude")
        ax.set_title("Latest signed MPZ state checkpoint")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        path = root / "live_mpz_state_profiles.png"
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
        outputs.append(path.name)

        activity = np.zeros(max(values.size for values in arrays.values()))
        for values in arrays.values():
            activity[: values.size] += np.abs(values)
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.plot(np.arange(activity.size), activity)
        ax.set_xlabel("MPZ bin")
        ax.set_ylabel("Absolute signed-state activity")
        ax.set_title("Latest MPZ activity proxy")
        ax.grid(True, alpha=0.25)
        path = root / "live_mpz_activity_proxy.png"
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
        outputs.append(path.name)

    manifest = {
        "schema": "v10.2.30_live_high_cycle_visuals_v1",
        "checkpoint_reason": checkpoint.get("reason"),
        "cycles_from_engine_time": checkpoint.get("cycles_from_engine_time"),
        "outputs": outputs,
    }
    (root / "live_high_cycle_visual_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
