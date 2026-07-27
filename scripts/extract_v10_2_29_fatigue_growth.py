#!/usr/bin/env python3
"""Extract event-resolved fatigue growth measurements from v10.2.29 output."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "v10.2.29_event_resolved_fatigue_growth_v1"
REQUIRED = {
    "step", "KJ_Pa_sqrtm", "crack_extension_m", "da_block_m",
    "n_fire", "fatigue_cycles",
}


def read_rows(path: Path) -> np.ndarray:
    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    missing = REQUIRED - set(data.dtype.names or ())
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    return data


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weights = np.maximum(np.asarray(weights, float), 0.0)
    values = np.asarray(values, float)
    total = float(np.sum(weights))
    return float(np.sum(values * weights) / total) if total > 0.0 else float(values[-1])


def extract_events(rows: np.ndarray, R: float) -> list[dict]:
    cycles = np.maximum(np.asarray(rows["fatigue_cycles"], float), 0.0)
    cumulative = np.cumsum(cycles)
    K = np.maximum(np.asarray(rows["KJ_Pa_sqrtm"], float), 0.0)
    extension = np.maximum(np.asarray(rows["crack_extension_m"], float), 0.0)
    advance = np.maximum(np.asarray(rows["da_block_m"], float), 0.0)
    fired = np.asarray(rows["n_fire"], float) > 0.0
    event_indices = np.flatnonzero(fired & (advance > 0.0))
    events = []
    previous_index = -1
    previous_cycles = 0.0
    previous_extension = 0.0
    for event_number, index in enumerate(event_indices, start=1):
        start = previous_index + 1
        stop = index + 1
        interval_cycles = max(float(cumulative[index]) - previous_cycles, 0.0)
        da = float(advance[index])
        a_post = float(extension[index])
        a_pre_row = max(a_post - da, 0.0)
        a_pre = previous_extension if previous_index >= 0 else a_pre_row
        if not math.isclose(a_pre, a_pre_row, rel_tol=1e-10, abs_tol=1e-15):
            a_pre = a_pre_row
        K_window = K[start:stop]
        weights = cycles[start:stop]
        K_event = float(K[index])
        K_weighted = weighted_mean(K_window, weights)
        events.append({
            "event_index": event_number,
            "step": int(round(float(rows["step"][index]))),
            "row_index": int(index),
            "cycles_pre": previous_cycles,
            "cycles_post": float(cumulative[index]),
            "cycles_between_events": interval_cycles,
            "event_advance_m": da,
            "da_dN_raw_m_per_cycle": da / interval_cycles if interval_cycles > 0.0 else float("nan"),
            "a_pre_m": a_pre,
            "a_post_m": a_post,
            "a_mid_m": 0.5 * (a_pre + a_post),
            "extension_mid_m": 0.5 * (a_pre + a_post),
            "Kmax_event_pre_Pa_sqrt_m": K_event,
            "DeltaK_event_pre_Pa_sqrt_m": max((1.0 - R) * K_event, 0.0),
            "Kmax_cycle_weighted_Pa_sqrt_m": K_weighted,
            "DeltaK_cycle_weighted_Pa_sqrt_m": max((1.0 - R) * K_weighted, 0.0),
            "Kmax_min_between_events_Pa_sqrt_m": float(np.min(K_window)),
            "Kmax_max_between_events_Pa_sqrt_m": float(np.max(K_window)),
            "accepted_blocks_between_events": int(stop - start),
            "zero_cycle_interval": bool(interval_cycles <= 0.0),
        })
        previous_index = int(index)
        previous_cycles = float(cumulative[index])
        previous_extension = a_post
    return events


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
    parser.add_argument("--R", type=float, default=0.1)
    parser.add_argument("--require-event", action="store_true")
    args = parser.parse_args()
    tag = f"{int(round(args.temperature_K)):04d}K"
    source = args.run_root / f"steps_{tag}.csv"
    events = extract_events(read_rows(source), float(args.R))
    if args.require_event and not events:
        raise SystemExit(f"ERROR: no committed fatigue geometry event in {source}")
    csv_path = args.run_root / f"fatigue_event_growth_{tag}.csv"
    json_path = args.run_root / f"fatigue_event_growth_{tag}.json"
    write_csv(csv_path, events)
    payload = {
        "schema": SCHEMA,
        "source_steps_csv": str(source.resolve()),
        "temperature_K": float(args.temperature_K),
        "R": float(args.R),
        "event_count": len(events),
        "measurements": events,
        "rate_definition": "event advance divided by consumed cycles between committed geometry events",
        "smoothing_or_Paris_fit_applied": False,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
