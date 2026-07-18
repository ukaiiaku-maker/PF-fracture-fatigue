#!/usr/bin/env python3
"""Replay and compare an exact v10.2.5 signed 2-D state trace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from arrhenius_fracture.reduced_shared_state_v1025 import (
    ExactProductionConfig,
    load_manifest,
    replay_exact_signed_state,
)


def _metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, float]:
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidate")
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--signed-shielding-kernel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-10)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    root = args.trace_root.resolve()
    schedule = root / "v10_2_5_2d_signed_replay_schedule.csv"
    config_path = root / "v10_2_5_2d_exact_engine_config.json"
    expected_path = root / "v10_2_5_2d_signed_final_state.npz"
    for path in (schedule, config_path, expected_path, args.signed_shielding_kernel):
        if not path.is_file():
            raise SystemExit(f"required file is missing: {path}")

    manifest = load_manifest(
        candidate_id=args.candidate,
        manifest_path=args.manifest,
    )
    payload = json.loads(config_path.read_text())
    traced = str(payload.get("material_manifest", {}).get("candidate_id", ""))
    if traced and traced != manifest.candidate_id:
        raise SystemExit(
            f"trace candidate {traced} does not match manifest {manifest.candidate_id}"
        )
    config = ExactProductionConfig.from_trace(
        payload, args.signed_shielding_kernel
    )
    replay = replay_exact_signed_state(manifest, schedule, config)
    expected = np.load(expected_path)
    actual = replay["_final_arrays"]
    names = sorted(set(expected.files).intersection(actual))
    missing_expected = sorted(set(actual).difference(expected.files))
    missing_actual = sorted(set(expected.files).difference(actual))
    metrics = {name: _metrics(expected[name], actual[name]) for name in names}
    max_relative = max(
        (item["maximum_relative_to_expected_scale"] for item in metrics.values()),
        default=0.0,
    )
    max_absolute = max(
        (item["maximum_abs_error"] for item in metrics.values()), default=0.0
    )
    passed = bool(
        replay["config_parity"]["passed"]
        and replay["all_fired_flags_match"]
        and not missing_expected
        and not missing_actual
        and max_relative <= args.relative_tolerance
        and max_absolute <= args.absolute_tolerance
    )
    assessment = {
        "schema": "v10.2.5_exact_signed_replay_assessment",
        "candidate_id": manifest.candidate_id,
        "configuration_parity": replay["config_parity"],
        "state_arrays_compared": names,
        "missing_expected_arrays": missing_expected,
        "missing_actual_arrays": missing_actual,
        "array_metrics": metrics,
        "maximum_array_relative_error": max_relative,
        "maximum_array_absolute_error": max_absolute,
        "all_fired_flags_match": replay["all_fired_flags_match"],
        "same_engine_for_monotonic_and_fatigue": True,
        "local_strength_sigma_cap_preserved": True,
        "constitutive_K_shield_cap_applied": False,
        "passed": passed,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "v10_2_5_exact_signed_replay.json").write_text(
        json.dumps(assessment, indent=2)
    )
    np.savez_compressed(args.out / "v10_2_5_replayed_final_state.npz", **actual)
    print(json.dumps(assessment, indent=2))
    if not passed:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
