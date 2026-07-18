#!/usr/bin/env python3
"""Compare a traced v10.2.6 2-D state with exact v10.2.7 replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from arrhenius_fracture.reduced_shared_state_v1023 import load_manifest
from arrhenius_fracture.state_resolved_reduced_campaign_v1027 import (
    StateResolvedProductionConfig,
)
from arrhenius_fracture.state_resolved_replay_v1027 import replay_state_resolved_trace


def _metrics(expected, actual):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.shape != actual.shape:
        raise ValueError(f"shape mismatch {expected.shape} != {actual.shape}")
    delta = actual - expected
    scale = max(float(np.max(np.abs(expected))) if expected.size else 0.0, 1.0)
    return {
        "maximum_abs_error": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "maximum_relative_to_expected_scale": (
            float(np.max(np.abs(delta))) / scale if delta.size else 0.0
        ),
        "expected_sum": float(np.sum(expected)),
        "actual_sum": float(np.sum(actual)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--kernel-family", type=Path, required=True)
    parser.add_argument("--drive-family", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-10)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    schedule = args.trace_root / "v10_2_5_2d_signed_replay_schedule.csv"
    config_path = args.trace_root / "v10_2_5_2d_exact_engine_config.json"
    expected_path = args.trace_root / "v10_2_5_2d_signed_final_state.npz"
    for path in (schedule, config_path, expected_path):
        if not path.is_file():
            raise SystemExit(f"required trace file is missing: {path}")

    manifest = load_manifest(manifest_path=args.manifest)
    config = StateResolvedProductionConfig.from_trace(
        json.loads(config_path.read_text()),
        kernel_family_path=args.kernel_family,
        drive_family_path=args.drive_family,
    )
    replay = replay_state_resolved_trace(manifest, schedule, config)
    expected = np.load(expected_path)
    actual = replay["final_arrays"]
    names = sorted(set(expected.files).intersection(actual))
    missing_expected = sorted(set(actual).difference(expected.files))
    missing_actual = sorted(set(expected.files).difference(actual))
    metrics = {name: _metrics(expected[name], actual[name]) for name in names}
    maximum_relative = max(
        (row["maximum_relative_to_expected_scale"] for row in metrics.values()),
        default=0.0,
    )
    maximum_abs = max(
        (row["maximum_abs_error"] for row in metrics.values()), default=0.0
    )
    passed = bool(
        not missing_expected
        and not missing_actual
        and replay["all_fired_flags_match"]
        and replay["production_config_parity"]["passed"]
        and maximum_relative <= float(args.relative_tolerance)
        and maximum_abs <= float(args.absolute_tolerance)
    )
    assessment = {
        "schema": "v10.2.7_exact_state_resolved_replay_assessment",
        "candidate_id": manifest.candidate_id,
        "n_schedule_rows": replay["n_schedule_rows"],
        "state_arrays_compared": names,
        "missing_expected_arrays": missing_expected,
        "missing_actual_arrays": missing_actual,
        "array_metrics": metrics,
        "maximum_array_relative_error": maximum_relative,
        "maximum_array_absolute_error": maximum_abs,
        "all_fired_flags_match": replay["all_fired_flags_match"],
        "complete_configuration_parity": replay["production_config_parity"]["passed"],
        "constitutive_K_shield_cap_applied": False,
        "relative_tolerance": float(args.relative_tolerance),
        "absolute_tolerance": float(args.absolute_tolerance),
        "passed": passed,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "state_resolved_replay_assessment.json").write_text(
        json.dumps(assessment, indent=2)
    )
    print(json.dumps(assessment, indent=2))
    if not passed:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
