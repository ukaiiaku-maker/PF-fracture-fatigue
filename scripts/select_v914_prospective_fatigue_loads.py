#!/usr/bin/env python3
"""Select sparse adaptive fatigue loads from exact virgin-cycle screens."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = (
    (1.0e6, "VHCF_1E6"),
    (1.0e4, "HCF_1E4"),
    (100.0, "RARE_HCF_LOWER"),
    (25.0, "RARE_HCF_20PLUS"),
    (8.0, "HCF_LCF_OVERLAP"),
    (4.0, "TRANSITION_3_TO_10"),
    (2.0, "LCF_1_TO_3"),
    (0.5, "SUBCYCLE_0P1_TO_1"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def interpolate_fraction(fraction: np.ndarray, cycles: np.ndarray, target: float) -> float:
    valid = np.isfinite(cycles) & (cycles > 0.0)
    x = fraction[valid]
    y = np.log10(cycles[valid])
    if len(x) < 2:
        return float("nan")
    order = np.argsort(y)
    y = y[order]
    x = x[order]
    value = math.log10(target)
    return float(np.interp(value, y, x, left=x[0], right=x[-1]))


def main() -> int:
    args = parse_args()
    payloads = []
    for path in sorted((args.screen_root / "cases").glob("*.json")):
        payloads.append(json.loads(path.read_text()))
    if not payloads:
        # The production screen writes one candidate payload per subdirectory.
        # Do not admit the root run contract merely because its schema shares
        # the same prefix.
        for path in sorted(args.screen_root.glob("*/state_screen.json")):
            payload = json.loads(path.read_text())
            if (
                payload.get("schema", "").startswith("v914_endurance_knee_state_screen")
                and "candidate_id" in payload
                and "points" in payload
            ):
                payloads.append(payload)
    if not payloads:
        raise RuntimeError("no state-screen candidate payloads found")
    rows = []
    screens = []
    for payload in payloads:
        candidate_id = str(payload["candidate_id"])
        reference = float(payload["reference_deltaK_MPa_sqrt_m"])
        points = pd.DataFrame(payload["points"]).sort_values("fraction")
        points["candidate_id"] = candidate_id
        screens.append(points)
        fraction = points.fraction.to_numpy(float)
        cycles = pd.to_numeric(points.projected_mean_first_passage_cycles, errors="coerce").to_numpy(float)
        selected: list[tuple[float, str, float]] = []
        for target, regime in TARGETS:
            value = interpolate_fraction(fraction, cycles, target)
            value = float(np.clip(value, fraction.min(), fraction.max()))
            selected.append((value, regime, target))
        selected.extend(
            [
                (float(fraction.min()), "SCREEN_LOWER_ENDPOINT", float("nan")),
                (float(fraction.max()), "SCREEN_UPPER_ENDPOINT", float("nan")),
            ]
        )
        # Preserve the more discriminating target when rounded loads coincide.
        unique: dict[float, tuple[str, float]] = {}
        for value, regime, target in selected:
            key = round(value, 5)
            if key not in unique or regime == "HCF_LCF_OVERLAP":
                unique[key] = (regime, target)
        for value, (regime, target) in sorted(unique.items()):
            predicted = 10.0 ** float(
                np.interp(value, fraction, np.log10(np.maximum(cycles, 1e-300)))
            )
            if regime == "HCF_LCF_OVERLAP":
                mode = "BOTH_ACCELERATED_AND_EXPLICIT"
            elif predicted <= 10.0:
                mode = "EXPLICIT"
            else:
                mode = "ACCELERATED"
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "normalized_f": value,
                    "deltaK_MPa_sqrt_m": reference * value,
                    "selection_regime": regime,
                    "target_projected_cycles_per_event": target,
                    "screen_interpolated_projected_cycles_per_event": predicted,
                    "primary_integration_mode": mode,
                    "accelerated_required": mode in {"ACCELERATED", "BOTH_ACCELERATED_AND_EXPLICIT"},
                    "explicit_required": mode in {"EXPLICIT", "BOTH_ACCELERATED_AND_EXPLICIT"},
                    "R": 0.1,
                    "frequency_Hz": 1000.0,
                    "temperature_K": 300.0,
                    "target_extension_um": 100.0,
                }
            )
    output = pd.DataFrame(rows).sort_values(["candidate_id", "normalized_f"])
    counts = output.groupby("candidate_id").size()
    if not counts.between(6, 10).all():
        raise RuntimeError(f"adaptive load counts outside 6--10: {counts.to_dict()}")
    args.out.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out / "prospective_fatigue_adaptive_loads.csv", index=False)
    pd.concat(screens, ignore_index=True).to_csv(args.out / "prospective_fatigue_state_screen_points.csv", index=False)
    (args.out / "prospective_fatigue_load_selection_manifest.json").write_text(
        json.dumps(
            {
                "schema": "v914_prospective_adaptive_fatigue_loads_v1",
                "candidate_count": len(counts),
                "load_count": len(output),
                "minimum_loads_per_candidate": int(counts.min()),
                "maximum_loads_per_candidate": int(counts.max()),
                "fixed_dense_grid_used": False,
                "accelerated_explicit_overlap_per_candidate": bool(
                    output.groupby("candidate_id").primary_integration_mode.apply(
                        lambda values: (values == "BOTH_ACCELERATED_AND_EXPLICIT").any()
                    ).all()
                ),
                "physics_changed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"V914_ADAPTIVE_LOAD_SELECTION_COMPLETE candidates={len(counts)} loads={len(output)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
