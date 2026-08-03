#!/usr/bin/env python3
"""Generate human-readable diagnostics for a v10.2.30 high-cycle run.

The sharp-front model has no diffuse phase-field damage variable.  When the
opt-in active-state snapshot is present, this tool plots the actual signed MPZ
mobile, retained, and accumulated-slip profiles and an explicitly labelled MPZ
activity proxy.  It never labels that proxy as a physical phase-field damage
field.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text())


def _records(root: Path) -> list[dict[str, Any]]:
    payload = _load_json(root / "kinetic_tip_cell_audit_v101.json", {})
    if isinstance(payload, dict):
        rows = payload.get("records", [])
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    return [row for row in rows if isinstance(row, dict)]


def _mode_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cumulative = 0.0
    operation = 0
    for record_index, record in enumerate(records):
        for mode in record.get("coupled_hazard_modes", []):
            if not isinstance(mode, dict):
                continue
            operation += 1
            cycles = max(float(mode.get("cycles", 0.0)), 0.0)
            start = cumulative
            cumulative += cycles
            rows.append(
                {
                    **mode,
                    "record_index": record_index,
                    "operation": operation,
                    "cycles": cycles,
                    "cycle_start": start,
                    "cycle_end": cumulative,
                }
            )
    return rows


def _final_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    return records[-1] if records else {}


def _step_row(root: Path) -> dict[str, float]:
    paths = sorted(root.glob("steps_*K.csv"))
    if not paths:
        return {}
    data = np.genfromtxt(paths[0], delimiter=",", names=True, dtype=float)
    if getattr(data, "shape", ()) == ():
        row = data
    elif len(data) > 0:
        row = data[-1]
    else:
        return {}
    return {name: float(row[name]) for name in row.dtype.names or ()}


def plot_mode_timeline(root: Path, modes: list[dict[str, Any]]) -> Path | None:
    if not modes:
        return None
    names = sorted({str(row.get("mode", "unknown")) for row in modes})
    ymap = {name: index for index, name in enumerate(names)}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for row in modes:
        name = str(row.get("mode", "unknown"))
        start = max(float(row["cycle_start"]), 1.0e-6)
        end = max(float(row["cycle_end"]), start)
        y = ymap[name]
        if end > start:
            ax.plot([start, end], [y, y], linewidth=6, solid_capstyle="butt")
        else:
            ax.scatter([start], [y], marker="x", s=45)
    ax.set_xscale("log")
    ax.set_yticks(list(ymap.values()), list(ymap.keys()))
    ax.set_xlabel("Cumulative consumed cycles")
    ax.set_ylabel("High-cycle operation")
    ax.set_title("High-cycle mode timeline")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    path = root / "high_cycle_mode_timeline.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_validation_history(root: Path, modes: list[dict[str, Any]]) -> Path | None:
    rows = [
        row
        for row in modes
        if "hazard_error" in row or "drift_error" in row
    ]
    if not rows:
        return None
    operations = np.asarray([row["operation"] for row in rows], dtype=float)
    hazard = np.asarray(
        [max(float(row.get("hazard_error", np.nan)), 1.0e-18) for row in rows]
    )
    drift = np.asarray(
        [max(float(row.get("drift_error", np.nan)), 1.0e-18) for row in rows]
    )
    accepted = np.asarray(
        [str(row.get("mode", "")) == "slow_projective" for row in rows]
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(operations, hazard, marker="o", label="Hazard validation error")
    ax.plot(operations, drift, marker="s", label="State validation error")
    for operation, is_accepted in zip(operations, accepted):
        if not is_accepted:
            ax.axvline(operation, linewidth=0.6, alpha=0.2)
    ax.axhline(1.0e-3, linestyle="--", linewidth=1.0, label="1e-3 reference")
    ax.set_yscale("log")
    ax.set_xlabel("Operation index")
    ax.set_ylabel("Relative validation error")
    ax.set_title("Projective-map validation history")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = root / "high_cycle_validation_history.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_mechanical_state(
    root: Path,
    final: dict[str, Any],
    step: dict[str, float],
) -> Path | None:
    if not final and not step:
        return None
    stress_labels = ["Tip stress", "Backstress"]
    stress_values = [
        float(step.get("sigma_tip_Pa", 0.0)) / 1.0e9,
        float(final.get("persistent_sigma_back_Pa", step.get("sigma_back_Pa", 0.0)))
        / 1.0e9,
    ]
    state_labels = ["Mobile", "Retained", "Emitted", "Escaped"]
    state_values = [
        float(final.get("state_mobile_count", step.get("mpz_mobile_count", 0.0))),
        float(final.get("state_retained_count", step.get("mpz_retained_count", 0.0))),
        float(final.get("state_emitted_total", step.get("pz_emit_total", 0.0))),
        float(final.get("state_escaped_total", step.get("mpz_escaped_total", 0.0))),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    axes[0].bar(stress_labels, stress_values)
    axes[0].set_ylabel("Stress (GPa)")
    axes[0].set_title("Final crack-tip mechanical response")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(state_labels, state_values)
    axes[1].set_yscale("symlog", linthresh=1.0e-12)
    axes[1].set_ylabel("Population / ledger count")
    axes[1].set_title("Final process-zone state")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = root / "final_mechanical_response.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _snapshot(final: dict[str, Any]) -> dict[str, Any] | None:
    value = final.get("coupled_hazard_active_state_snapshot")
    return value if isinstance(value, dict) else None


def _snapshot_arrays(snapshot: dict[str, Any]) -> dict[str, np.ndarray]:
    vector = np.asarray(snapshot.get("vector", []), dtype=float)
    arrays: dict[str, np.ndarray] = {}
    for field in snapshot.get("fields", []):
        if not isinstance(field, dict):
            continue
        start = int(field.get("start", 0))
        stop = int(field.get("stop", start))
        shape = tuple(int(value) for value in field.get("shape", []))
        name = f"{field.get('owner', 'unknown')}.{field.get('name', 'unknown')}"
        raw = vector[start:stop]
        arrays[name] = raw.reshape(shape) if shape else raw.copy()
    return arrays


def _profile(array: np.ndarray | None) -> np.ndarray | None:
    if array is None:
        return None
    value = np.asarray(array, dtype=float)
    if value.ndim == 0:
        return value.reshape(1)
    if value.ndim == 1:
        return value
    axes = tuple(range(value.ndim - 1))
    return np.sum(value, axis=axes)


def plot_mpz_profiles(root: Path, final: dict[str, Any]) -> list[Path]:
    snapshot = _snapshot(final)
    if snapshot is None:
        return []
    arrays = _snapshot_arrays(snapshot)
    names = {
        "mobile_pos": "mpz.mobile_positive",
        "mobile_neg": "mpz.mobile_negative",
        "retained_pos": "mpz.retained_positive",
        "retained_neg": "mpz.retained_negative",
        "slip_pos": "mpz.accumulated_slip_positive",
        "slip_neg": "mpz.accumulated_slip_negative",
    }
    profiles = {key: _profile(arrays.get(name)) for key, name in names.items()}
    available = [value for value in profiles.values() if value is not None]
    if not available:
        return []
    n_bins = max(value.size for value in available)
    x = np.linspace(0.0, 1.0, n_bins)

    def padded(value: np.ndarray | None) -> np.ndarray:
        if value is None:
            return np.zeros(n_bins)
        if value.size == n_bins:
            return value
        return np.pad(value, (0, n_bins - value.size))

    mobile_activity = padded(profiles["mobile_pos"]) + padded(profiles["mobile_neg"])
    retained_activity = padded(profiles["retained_pos"]) + padded(profiles["retained_neg"])
    slip_activity = padded(profiles["slip_pos"]) + padded(profiles["slip_neg"])
    mobile_signed = padded(profiles["mobile_pos"]) - padded(profiles["mobile_neg"])
    retained_signed = padded(profiles["retained_pos"]) - padded(profiles["retained_neg"])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, mobile_activity, label="Mobile activity")
    ax.plot(x, retained_activity, label="Retained activity")
    ax.plot(x, slip_activity, label="Accumulated-slip activity")
    ax.plot(x, mobile_signed, linestyle="--", label="Signed mobile")
    ax.plot(x, retained_signed, linestyle="--", label="Signed retained")
    ax.set_xlabel("Normalized active-MPZ coordinate")
    ax.set_ylabel("Bin-integrated state")
    ax.set_title("Final signed moving-process-zone profiles")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    profile_path = root / "final_mpz_state_profiles.png"
    fig.savefig(profile_path, dpi=180)
    plt.close(fig)

    proxy = np.vstack([mobile_activity, retained_activity, slip_activity])
    fig, ax = plt.subplots(figsize=(10, 3.8))
    image = ax.imshow(proxy, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_yticks([0, 1, 2], ["Mobile", "Retained", "Accumulated slip"])
    ax.set_xlabel("Active-MPZ bin")
    ax.set_title("MPZ activity proxy — not a phase-field damage variable")
    fig.colorbar(image, ax=ax, label="State activity")
    fig.tight_layout()
    proxy_path = root / "mpz_activity_proxy.png"
    fig.savefig(proxy_path, dpi=180)
    plt.close(fig)
    return [profile_path, proxy_path]


def plot_summary_panel(
    root: Path,
    records: list[dict[str, Any]],
    modes: list[dict[str, Any]],
    step: dict[str, float],
) -> Path:
    final = _final_record(records)
    summary = _load_json(root / "high_cycle_summary.json", {}) or {}
    control = _load_json(root / "v10_2_30_fixed_deltaK_control.json", {}) or {}
    exit_code = (root / "exit_code.txt").read_text().strip() if (root / "exit_code.txt").is_file() else "unknown"
    wall = (root / "wall_seconds.txt").read_text().strip() if (root / "wall_seconds.txt").is_file() else "unknown"
    delta_k = float(step.get("fatigue_DeltaK_target_Pa_sqrtm", 0.0)) / 1.0e6
    kmax = float(step.get("fatigue_Kmax_target_Pa_sqrtm", 0.0)) / 1.0e6
    lines = [
        "v10.2.30 high-cycle run summary",
        "",
        f"Exit code: {exit_code}",
        f"Wall time: {wall} s",
        f"Consumed cycles: {float(summary.get('cycles_consumed', 0.0)):.6g}",
        f"Delta K: {delta_k:.6g} MPa sqrt(m)",
        f"Kmax: {kmax:.6g} MPa sqrt(m)",
        f"Fired records: {int(summary.get('fired_records', 0))}",
        f"Final B: {float(final.get('B', 0.0)):.6e}",
        f"Physical hazard action: {float(final.get('physical_hazard_action_block', 0.0)):.6e}",
        f"Backstress: {float(final.get('persistent_sigma_back_Pa', 0.0))/1e9:.6g} GPa",
        f"Mobile population: {float(final.get('state_mobile_count', 0.0)):.6g}",
        f"Retained population: {float(final.get('state_retained_count', 0.0)):.6g}",
        f"Mode operations: {len(modes)}",
        f"Fatigue status: {control.get('fatigue_censor_status', control.get('status', 'unknown'))}",
    ]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.axis("off")
    ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11)
    path = root / "high_cycle_summary_panel.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def analyze(root: Path) -> list[Path]:
    records = _records(root)
    modes = _mode_rows(records)
    final = _final_record(records)
    step = _step_row(root)
    paths: list[Path] = []
    for path in (
        plot_mode_timeline(root, modes),
        plot_validation_history(root, modes),
        plot_mechanical_state(root, final, step),
    ):
        if path is not None:
            paths.append(path)
    paths.extend(plot_mpz_profiles(root, final))
    paths.append(plot_summary_panel(root, records, modes, step))
    manifest = {
        "schema": "v10.2.30_high_cycle_visual_diagnostics_v1",
        "run_root": str(root.resolve()),
        "active_state_snapshot_available": _snapshot(final) is not None,
        "files": [path.name for path in paths],
        "damage_field_claimed": False,
        "mpz_activity_proxy_labelled": _snapshot(final) is not None,
    }
    (root / "high_cycle_visual_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"run directory does not exist: {root}")
    paths = analyze(root)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
