#!/usr/bin/env python3
"""Choose eight derivative-focused fatigue loads from exact cycle screens."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = (
    (1.0e6, "LOWER_FINITE_HCF", "accelerated"),
    (1.0e4, "DEVELOPED_HCF_LOW", "accelerated"),
    (1.0e3, "DEVELOPED_HCF_MID", "accelerated"),
    (1.0e2, "DEVELOPED_HCF_HIGH", "accelerated"),
    (25.0, "UPPER_HCF", "accelerated"),
    (8.0, "HCF_LCF_OVERLAP", "both"),
    (2.0, "EXPLICIT_LCF", "explicit"),
    (0.5, "EXPLICIT_NEAR_MONOTONIC", "explicit"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def candidate_payloads(root: Path) -> list[dict]:
    payloads = []
    for path in sorted(root.rglob("state_screen.json")):
        payload = json.loads(path.read_text())
        if "candidate_id" in payload and "points" in payload:
            payloads.append(payload)
    if not payloads:
        raise RuntimeError("no candidate state screens")
    ids = [str(payload["candidate_id"]) for payload in payloads]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate candidate state screens")
    return payloads


def interpolate(fraction: np.ndarray, cycles: np.ndarray, target: float) -> float:
    valid = np.isfinite(cycles) & (cycles > 0.0)
    x, y = fraction[valid], np.log10(cycles[valid])
    if len(x) < 3:
        raise RuntimeError("insufficient finite screen points")
    order = np.argsort(y)
    return float(np.interp(math.log10(target), y[order], x[order]))


def main() -> int:
    args = parse_args()
    rows, screen_rows = [], []
    for payload in candidate_payloads(args.screen_root):
        candidate = str(payload["candidate_id"])
        reference = float(payload["reference_deltaK_MPa_sqrt_m"])
        screen = pd.DataFrame(payload["points"]).sort_values("fraction")
        fractions = screen.fraction.to_numpy(float)
        cycles = pd.to_numeric(screen.projected_mean_first_passage_cycles, errors="coerce").to_numpy(float)
        if np.any(np.diff(fractions) <= 0.0):
            raise RuntimeError(f"non-increasing fractions: {candidate}")
        selected = []
        for target, regime, mode in TARGETS:
            value = interpolate(fractions, cycles, target)
            value = float(np.clip(value, fractions.min(), fractions.max()))
            predicted = 10.0 ** float(np.interp(value, fractions, np.log10(np.maximum(cycles, 1e-300))))
            selected.append((value, target, regime, mode, predicted))
        rounded = [round(item[0], 6) for item in selected]
        if len(set(rounded)) != len(TARGETS):
            raise RuntimeError(f"screen cannot resolve eight distinct targets: {candidate} {rounded}")
        for value, target, regime, mode, predicted in selected:
            rows.append({
                "candidate_id": candidate,
                "normalized_f": value,
                "deltaK_MPa_sqrt_m": reference * value,
                "selection_regime": regime,
                "target_projected_cycles_per_event": target,
                "screen_interpolated_projected_cycles_per_event": predicted,
                "primary_integration_mode": mode.upper(),
                "accelerated_required": mode in {"accelerated", "both"},
                "explicit_required": mode in {"explicit", "both"},
                "R": 0.1,
                "frequency_Hz": 1000.0,
                "temperature_K": 300.0,
                "target_extension_um": 100.0,
            })
        q = screen.copy()
        q["candidate_id"] = candidate
        screen_rows.extend(q.to_dict("records"))
    output = pd.DataFrame(rows).sort_values(["candidate_id", "normalized_f"])
    counts = output.groupby("candidate_id").size()
    if not counts.eq(8).all():
        raise RuntimeError("every candidate must receive exactly eight loads")
    accelerated = output.groupby("candidate_id").accelerated_required.sum()
    explicit = output.groupby("candidate_id").explicit_required.sum()
    if not accelerated.eq(6).all() or not explicit.eq(3).all():
        raise RuntimeError("mode coverage contract failed")
    args.out.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out / "prospective_slope_fatigue_loads.csv", index=False, float_format="%.17g")
    pd.DataFrame(screen_rows).to_csv(args.out / "prospective_slope_state_screen_points.csv", index=False)
    (args.out / "prospective_slope_load_manifest.json").write_text(json.dumps({
        "schema": "v914_prospective_slope_adaptive_loads_v1",
        "candidate_count": len(counts),
        "loads_per_candidate": 8,
        "accelerated_loads_per_candidate": 6,
        "explicit_loads_per_candidate": 3,
        "overlap_loads_per_candidate": 1,
        "HCF_points_per_candidate": 6,
        "fixed_dense_grid_used": False,
        "selection_uses_fatigue_growth_result": False,
        "physics_changed": False,
    }, indent=2, sort_keys=True) + "\n")
    print(f"V914_SLOPE_LOAD_SELECTION_COMPLETE candidates={len(counts)} loads={len(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
