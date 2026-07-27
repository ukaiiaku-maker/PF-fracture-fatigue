#!/usr/bin/env python3
"""Extract initiation, development, and developed-state fatigue-growth intervals."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


SCHEMA = "v10.2.29_developed_fatigue_growth_v1"
BASE_REQUIRED = {
    "step",
    "crack_extension_m",
    "da_block_m",
    "n_fire",
    "fatigue_cycles",
}


def read_rows(path: Path) -> np.ndarray:
    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    missing = BASE_REQUIRED - set(data.dtype.names or ())
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    return data


def _first_existing(names: Iterable[str], available: set[str]) -> str | None:
    for name in names:
        if name in available:
            return name
    return None


def _value(rows: np.ndarray, index: int, names: Iterable[str]) -> float:
    key = _first_existing(names, set(rows.dtype.names or ()))
    if key is None:
        return float("nan")
    value = float(rows[key][index])
    return value if math.isfinite(value) else float("nan")


def _load_control(run_root: Path) -> dict:
    path = run_root / "v10_2_29_fixed_deltaK_control.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing fixed-DeltaK audit: {path}")
    payload = json.loads(path.read_text())
    if payload.get("schema") != "v10.2.29_persistent_site_fixed_deltaK_v1":
        raise ValueError(f"unexpected fixed-DeltaK schema in {path}")
    return payload


def extract(
    rows: np.ndarray,
    control: dict,
    *,
    developed_start_um: float,
    developed_end_um: float | None,
) -> tuple[list[dict], list[dict], dict]:
    cycles = np.maximum(np.asarray(rows["fatigue_cycles"], float), 0.0)
    cumulative = np.cumsum(cycles)
    extension = np.maximum(np.asarray(rows["crack_extension_m"], float), 0.0)
    advance = np.maximum(np.asarray(rows["da_block_m"], float), 0.0)
    fired = np.asarray(rows["n_fire"], float) > 0.0
    event_indices = np.flatnonzero(fired & (advance > 0.0))

    Kmax = float(control["target_Kmax_MPa_sqrt_m"]) * 1.0e6
    DeltaK = float(control["target_deltaK_MPa_sqrt_m"]) * 1.0e6
    developed_start_m = max(float(developed_start_um), 0.0) * 1.0e-6
    developed_end_m = (
        None
        if developed_end_um is None
        else max(float(developed_end_um), float(developed_start_um)) * 1.0e-6
    )

    events: list[dict] = []
    measurements: list[dict] = []
    previous_event_index: int | None = None
    previous_event_cycles = 0.0
    initiation_post_m: float | None = None

    for event_number, row_index in enumerate(event_indices, start=1):
        da = float(advance[row_index])
        a_post = float(extension[row_index])
        a_pre = max(a_post - da, 0.0)
        cycles_post = float(cumulative[row_index])

        if event_number == 1:
            initiation_post_m = a_post
            interval_cycles = cycles_post
            stage = "initiation"
            rate = float("nan")
            extension_since_initiation_pre_m = 0.0
            eligible = False
        else:
            if previous_event_index is None or initiation_post_m is None:
                raise RuntimeError("internal event-ordering error")
            interval_cycles = max(cycles_post - previous_event_cycles, 0.0)
            rate = da / interval_cycles if interval_cycles > 0.0 else float("nan")
            extension_since_initiation_pre_m = max(a_pre - initiation_post_m, 0.0)
            window_tol = max(1.0e-15, 1.0e-12 * max(developed_start_m, 1.0e-30))
            in_lower = extension_since_initiation_pre_m + window_tol >= developed_start_m
            in_upper = (
                True
                if developed_end_m is None
                else extension_since_initiation_pre_m < developed_end_m - window_tol
            )
            eligible = bool(in_lower and in_upper and interval_cycles > 0.0)
            stage = "developed" if eligible else "microstructure_development"

        start_index = 0 if previous_event_index is None else previous_event_index + 1
        block_count = int(row_index - start_index + 1)
        record = {
            "event_index": int(event_number),
            "row_index": int(row_index),
            "step": int(round(float(rows["step"][row_index]))),
            "stage": stage,
            "measurement_eligible": eligible,
            "cycles_pre": float(previous_event_cycles),
            "cycles_post": cycles_post,
            "cycles_between_events": interval_cycles,
            "accepted_blocks_between_events": block_count,
            "event_advance_m": da,
            "da_dN_m_per_cycle": rate,
            "a_pre_m": a_pre,
            "a_post_m": a_post,
            "a_mid_m": 0.5 * (a_pre + a_post),
            "extension_since_initiation_pre_m": extension_since_initiation_pre_m,
            "extension_since_initiation_mid_m": (
                extension_since_initiation_pre_m + 0.5 * da
            ),
            "Kmax_target_Pa_sqrt_m": Kmax,
            "DeltaK_target_Pa_sqrt_m": DeltaK,
            "KJ_probe_event_pre_Pa_sqrt_m": _value(
                rows,
                row_index,
                ("KJ_probe_Pa_sqrtm", "KJ_Pa_sqrtm"),
            ),
            "sigma_back_event_pre_Pa": _value(
                rows,
                row_index,
                (
                    "sigma_back_Pa",
                    "sigma_back",
                    "persistent_sigma_back_Pa",
                ),
            ),
            "active_K_shield_event_pre_Pa_sqrt_m": _value(
                rows,
                row_index,
                (
                    "active_K_shield_signed_Pa_sqrt_m",
                    "kinetic_active_K_shield_signed_Pa_sqrt_m",
                    "Kshield_active_Pa_sqrtm",
                ),
            ),
            "mobile_event_pre": _value(
                rows,
                row_index,
                ("mobile_count", "active_mobile", "N_mobile"),
            ),
            "retained_event_pre": _value(
                rows,
                row_index,
                ("retained_count", "active_retained", "N_retained"),
            ),
            "emitted_total_event_pre": _value(
                rows,
                row_index,
                ("emitted_total", "N_em"),
            ),
        }
        events.append(record)
        if eligible:
            measurements.append(record)

        previous_event_index = int(row_index)
        previous_event_cycles = cycles_post

    total_cycles = float(cumulative[-1]) if cumulative.size else 0.0
    final_extension = float(extension[-1]) if extension.size else 0.0
    status = (
        "right_censored_no_event"
        if not events
        else "initiated_only"
        if len(events) == 1
        else "developed_measurements"
        if measurements
        else "propagated_before_developed_window"
    )
    summary = {
        "status": status,
        "event_count": len(events),
        "propagation_event_count": max(len(events) - 1, 0),
        "developed_measurement_count": len(measurements),
        "cycles_total": total_cycles,
        "final_crack_extension_m": final_extension,
        "initiation_post_extension_m": initiation_post_m,
        "developed_start_extension_from_initiation_m": developed_start_m,
        "developed_end_extension_from_initiation_m": developed_end_m,
        "developed_window_reached": bool(
            initiation_post_m is not None
            and final_extension - initiation_post_m >= developed_start_m
        ),
    }
    return events, measurements, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--developed-start-um", type=float, default=25.0)
    parser.add_argument("--developed-end-um", type=float)
    parser.add_argument("--require-developed-measurement", action="store_true")
    args = parser.parse_args()

    tag = f"{int(round(args.temperature_K)):04d}K"
    source = args.run_root / f"steps_{tag}.csv"
    control = _load_control(args.run_root)
    events, measurements, summary = extract(
        read_rows(source),
        control,
        developed_start_um=args.developed_start_um,
        developed_end_um=args.developed_end_um,
    )
    if args.require_developed_measurement and not measurements:
        raise SystemExit(
            "ERROR: no fatigue-growth interval lies in the requested developed-state window"
        )

    events_csv = args.run_root / f"fatigue_all_events_{tag}.csv"
    developed_csv = args.run_root / f"fatigue_developed_growth_{tag}.csv"
    json_path = args.run_root / f"fatigue_developed_growth_{tag}.json"
    write_csv(events_csv, events)
    write_csv(developed_csv, measurements)
    payload = {
        "schema": SCHEMA,
        "source_steps_csv": str(source.resolve()),
        "temperature_K": float(args.temperature_K),
        "parameter_option": control.get("parameter_option"),
        "target_deltaK_MPa_sqrt_m": float(control["target_deltaK_MPa_sqrt_m"]),
        "target_Kmax_MPa_sqrt_m": float(control["target_Kmax_MPa_sqrt_m"]),
        "R": float(control["R"]),
        "frequency_Hz": float(control["frequency_Hz"]),
        "rate_definition": (
            "current event advance divided by consumed cycles since the preceding "
            "committed event"
        ),
        "initiation_event_used_as_rate_measurement": False,
        "developed_window_reference": "crack extension after the first committed event",
        "smoothing_or_Paris_fit_applied": False,
        "summary": summary,
        "events": events,
        "developed_measurements": measurements,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
