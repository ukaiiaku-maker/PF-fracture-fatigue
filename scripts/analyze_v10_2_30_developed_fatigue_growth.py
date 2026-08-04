#!/usr/bin/env python3
"""Analyze event-to-event fatigue growth and developed da/dN for v10.2.30."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCHEMA = "v10.2.30_developed_event_to_event_fatigue_growth_v1"


def _json(path: Path, default):
    if not path.is_file():
        return default
    return json.loads(path.read_text())


def _closest_energy_trial(event: dict, committed_m: float) -> dict:
    rows = [row for row in event.get("trial_rows", []) if isinstance(row, dict)]
    if not rows:
        return {}
    return min(
        rows,
        key=lambda row: abs(float(row.get("trial_length_m", 0.0)) - committed_m),
    )


def enrich_events(events: list[dict], root: Path) -> tuple[list[dict], dict]:
    geometry = _json(root / "stochastic_avalanche_geometry_events.json", [])
    energy = _json(root / "hazard_energy_gated_events_v10_2_30.json", [])
    kinetic_payload = _json(root / "kinetic_tip_cell_audit_v101.json", {})
    kinetic = kinetic_payload.get("records", []) if isinstance(kinetic_payload, dict) else []
    control = _json(root / "v10_2_30_fixed_deltaK_control.json", {})
    manifest = _json(root / "high_cycle_run_manifest.json", {})
    parameter = _json(root / "v10_2_22_parameter_selection.json", {})

    provenance = {
        "run_path": str(root),
        "git_head": manifest.get("git_head"),
        "command": manifest.get("generic_launcher"),
        "environment": manifest.get("environment", {}),
        "parameter_option": control.get("parameter_option") or parameter.get("option_key"),
        "temperature_K": control.get("temperature_K", 300.0),
        "deltaK_MPa_sqrt_m": control.get("target_deltaK_MPa_sqrt_m"),
        "Kmax_MPa_sqrt_m": control.get("target_Kmax_MPa_sqrt_m"),
        "R": control.get("R"),
        "frequency_Hz": control.get("frequency_Hz"),
        "hazard_seed": control.get("cleavage_hazard_seed", manifest.get("hazard_seed")),
        "cycle_censor": control.get("cycles_max", manifest.get("cycles_max_censor")),
    }
    for index, row in enumerate(events):
        geom = geometry[index] if index < len(geometry) else {}
        gate = energy[index] if index < len(energy) else {}
        audit = kinetic[index] if index < len(kinetic) else {}
        committed = float(row.get("path_advance_m", 0.0))
        trial = _closest_energy_trial(gate, committed)
        modes = audit.get("coupled_hazard_modes", [])
        mode_names = [str(item.get("mode")) for item in modes if isinstance(item, dict)]
        row.update(
            {
                "parameter_option": provenance["parameter_option"],
                "temperature_K": provenance["temperature_K"],
                "deltaK_MPa_sqrt_m": provenance["deltaK_MPa_sqrt_m"],
                "Kmax_MPa_sqrt_m": provenance["Kmax_MPa_sqrt_m"],
                "R": provenance["R"],
                "frequency_Hz": provenance["frequency_Hz"],
                "hazard_seed": provenance["hazard_seed"],
                "threshold_action": geom.get("threshold_action"),
                "physical_hazard_action": audit.get("physical_hazard_action_block"),
                "event_length_factor": geom.get("event_length_factor"),
                "stochastic_proposed_advance_m": geom.get(
                    "stochastic_proposed_event_length_m"
                ),
                "energy_admissible_advance_m": gate.get(
                    "energy_admissible_event_length_m"
                ),
                "energy_available_J_per_m": trial.get("elastic_release_event_J_per_m"),
                "energy_required_J_per_m": trial.get("hazard_dissipation_J_per_m"),
                "energy_residual_J_per_m": trial.get("energy_residual_J_per_m"),
                "energy_gate_outcome": gate.get("arrest_reason"),
                "geometry_commit_inserted": gate.get("inserted"),
                "geometry_transaction_mode": geom.get("geometry_transaction_mode"),
                "acceleration_modes": ";".join(mode_names),
                "dmd_event_guard": any(
                    item.get("failure_reason") == "dmd_event_guard"
                    for item in modes if isinstance(item, dict)
                ),
                "exact_fallback_entered": "exact_cycle_burst" in mode_names,
                "transient_localization_entered": (
                    "event_localization_transient" in mode_names
                ),
                "private_trials_counted_as_cycles": False,
            }
        )
    return events, provenance


def _rows(path: Path):
    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    if not data.dtype.names:
        raise ValueError(f"no named columns in {path}")
    return data


def _column(rows, name: str, default: float = 0.0):
    if name not in (rows.dtype.names or ()):
        return np.full(rows.shape, default, dtype=float)
    return np.asarray(rows[name], dtype=float)


def _path_lengths(root: Path, temperature_K: float) -> list[float]:
    path = root / f"crack_path_{int(round(temperature_K))}K.csv"
    if not path.is_file():
        return []
    rows = _rows(path)
    x = _column(rows, "x_m")
    y = _column(rows, "y_m")
    if x.size < 2:
        return []
    return np.hypot(np.diff(x), np.diff(y)).tolist()


def extract_events(rows, path_increments: list[float]) -> list[dict]:
    cycles = np.maximum(_column(rows, "fatigue_cycles"), 0.0)
    cumulative_cycles = np.cumsum(cycles)
    extension = np.maximum(_column(rows, "crack_extension_m"), 0.0)
    advances = np.maximum(_column(rows, "da_block_m"), 0.0)
    fired = _column(rows, "n_fire") > 0.0
    event_indices = np.flatnonzero(fired & (advances > 0.0))
    events: list[dict] = []
    previous_cycles = 0.0
    previous_extension = 0.0
    cumulative_path = 0.0
    for event_number, index in enumerate(event_indices, start=1):
        cycles_post = float(cumulative_cycles[index])
        interval = max(cycles_post - previous_cycles, 0.0)
        da = float(advances[index])
        extension_post = float(extension[index])
        extension_pre = max(extension_post - da, previous_extension)
        ds = (
            float(path_increments[event_number - 1])
            if event_number - 1 < len(path_increments)
            else da
        )
        cumulative_path += ds
        event = {
            "event_index": event_number,
            "step": int(round(float(_column(rows, "step")[index]))),
            "row_index": int(index),
            "cycles_pre": previous_cycles,
            "cycles_post": cycles_post,
            "cycles_between_events": interval,
            "projected_advance_m": da,
            "path_advance_m": ds,
            "projected_extension_pre_m": extension_pre,
            "projected_extension_post_m": extension_post,
            "projected_extension_mid_m": 0.5 * (extension_pre + extension_post),
            "path_extension_post_m": cumulative_path,
            "da_dN_m_per_cycle": da / interval if interval > 0.0 else math.nan,
            "ds_dN_m_per_cycle": ds / interval if interval > 0.0 else math.nan,
            "tortuosity": ds / da if da > 0.0 else math.nan,
        }
        for source, target in (
            ("B", "B_post"),
            ("sigma_back_Pa", "sigma_back_Pa"),
            ("mpz_mobile_count", "mobile_count"),
            ("mpz_retained_count", "retained_count"),
            ("mpz_K_shield_Pa_sqrt_m", "K_shield_Pa_sqrt_m"),
            ("lambda_c", "lambda_c_per_s"),
        ):
            event[target] = float(_column(rows, source)[index])
        events.append(event)
        previous_cycles = cycles_post
        previous_extension = extension_post
    return events


def _interval_rate(events: list[dict], start_m: float, stop_m: float) -> dict:
    selected = [
        row for row in events
        if row["projected_extension_post_m"] > start_m
        and row["projected_extension_pre_m"] < stop_m
    ]
    if not selected:
        return {"event_count": 0, "da_m": 0.0, "dN": 0.0, "da_dN": None}
    da = sum(float(row["projected_advance_m"]) for row in selected)
    dN = sum(float(row["cycles_between_events"]) for row in selected)
    return {
        "event_count": len(selected),
        "da_m": da,
        "dN": dN,
        "da_dN": da / dN if dN > 0.0 else None,
    }


def _moving_windows(events: list[dict], width_m: float) -> list[dict]:
    if not events or width_m <= 0.0:
        return []
    final = float(events[-1]["projected_extension_post_m"])
    windows = []
    stop = width_m
    while stop <= final + 1.0e-15:
        start = max(stop - width_m, 0.0)
        result = _interval_rate(events, start, stop)
        result.update(
            {
                "window_start_m": start,
                "window_stop_m": stop,
                "window_mid_m": 0.5 * (start + stop),
            }
        )
        windows.append(result)
        stop += width_m
    return windows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(events: list[dict], windows: list[dict], root: Path) -> list[str]:
    outputs = []
    if not events:
        return outputs
    cycles = np.asarray([row["cycles_post"] for row in events], dtype=float)
    extension_um = 1.0e6 * np.asarray(
        [row["projected_extension_post_m"] for row in events], dtype=float
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(cycles, extension_um, marker="o")
    if np.all(cycles > 0.0):
        ax.set_xscale("log")
    ax.set_xlabel("Cumulative cycles")
    ax.set_ylabel("Projected crack extension (µm)")
    ax.set_title("Event-resolved crack extension")
    ax.grid(True, which="both", alpha=0.25)
    path = root / "crack_extension_vs_cycles.png"
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
    outputs.append(path.name)

    rates = np.asarray([row["da_dN_m_per_cycle"] for row in events], dtype=float)
    midpoint_um = 1.0e6 * np.asarray(
        [row["projected_extension_mid_m"] for row in events], dtype=float
    )
    valid = np.isfinite(rates) & (rates > 0.0)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.scatter(midpoint_um[valid], rates[valid])
    ax.set_yscale("log")
    ax.set_xlabel("Projected crack extension (µm)")
    ax.set_ylabel("Event-level da/dN (m/cycle)")
    ax.set_title("Event-to-event fatigue crack-growth rate")
    ax.grid(True, which="both", alpha=0.25)
    path = root / "event_da_dN_vs_extension.png"
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
    outputs.append(path.name)

    valid_windows = [row for row in windows if row.get("da_dN") not in (None, 0.0)]
    if valid_windows:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.plot(
            [1.0e6 * row["window_mid_m"] for row in valid_windows],
            [row["da_dN"] for row in valid_windows],
            marker="o",
        )
        ax.set_yscale("log")
        ax.set_xlabel("Window midpoint extension (µm)")
        ax.set_ylabel("Window da/dN (m/cycle)")
        ax.set_title("Moving-window fatigue crack-growth rate")
        ax.grid(True, which="both", alpha=0.25)
        path = root / "window_da_dN_vs_extension.png"
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
        outputs.append(path.name)
    return outputs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--target-extension-um", type=float, default=100.0)
    parser.add_argument("--development-extension-um", type=float, default=20.0)
    parser.add_argument("--stability-window-um", type=float, default=50.0)
    parser.add_argument("--moving-window-um", type=float, default=25.0)
    args = parser.parse_args(argv)

    root = args.run_root.resolve()
    tag = f"{int(round(args.temperature_K)):04d}K"
    steps_path = root / f"steps_{tag}.csv"
    if not steps_path.is_file():
        raise SystemExit(f"missing completed step history: {steps_path}")
    rows = _rows(steps_path)
    events = extract_events(rows, _path_lengths(root, args.temperature_K))
    events, provenance = enrich_events(events, root)
    target_m = args.target_extension_um * 1.0e-6
    development_m = args.development_extension_um * 1.0e-6
    stability_m = args.stability_window_um * 1.0e-6
    final_extension = (
        float(events[-1]["projected_extension_post_m"]) if events else 0.0
    )
    final_cycles = float(np.sum(np.maximum(_column(rows, "fatigue_cycles"), 0.0)))
    developed = _interval_rate(events, development_m, max(final_extension, development_m))
    stability_start = max(final_extension - stability_m, development_m)
    stability_mid = 0.5 * (stability_start + final_extension)
    early = _interval_rate(events, stability_start, stability_mid)
    late = _interval_rate(events, stability_mid, final_extension)
    ratio = None
    if early.get("da_dN") and late.get("da_dN"):
        ratio = float(late["da_dN"]) / float(early["da_dN"])
    stable = bool(
        developed["event_count"] >= 10
        and final_extension - development_m >= stability_m
        and ratio is not None
        and 0.5 <= ratio <= 2.0
    )
    windows = _moving_windows(events, args.moving_window_um * 1.0e-6)
    outputs = _plot(events, windows, root)
    _write_csv(root / f"fatigue_event_growth_{tag}.csv", events)
    _write_csv(root / f"fatigue_growth_windows_{tag}.csv", windows)

    status = (
        "growth_target_reached"
        if final_extension >= target_m
        else "partial_growth"
        if events
        else "no_committed_crack_event"
    )
    payload = {
        "schema": SCHEMA,
        "status": status,
        "temperature_K": float(args.temperature_K),
        "target_extension_um": float(args.target_extension_um),
        "development_extension_um": float(args.development_extension_um),
        "stability_window_um": float(args.stability_window_um),
        "event_count": len(events),
        "cycles_consumed": final_cycles,
        "final_projected_extension_um": final_extension * 1.0e6,
        "target_reached": final_extension >= target_m,
        "developed_interval": developed,
        "stability_early_interval": early,
        "stability_late_interval": late,
        "late_to_early_rate_ratio": ratio,
        "stable_growth_provisional": stable,
        "stability_definition": (
            "at_least_10_events_and_50um_developed_growth_and_"
            "late_to_early_cumulative_rate_ratio_between_0p5_and_2"
        ),
        "event_measurements": events,
        "moving_windows": windows,
        "plots": outputs,
        "provenance": provenance,
        "censor_or_failure_reason": (
            None if events else _json(root / "v10_2_30_fixed_deltaK_control.json", {}).get(
                "censor_status", "no_committed_crack_event"
            )
        ),
        "empirical_Paris_law_fit": False,
    }
    (root / "developed_fatigue_growth_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
