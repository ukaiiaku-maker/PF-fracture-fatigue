#!/usr/bin/env python3
"""Replay a v10.2.3 2-D trace and compare the complete final spatial state."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from arrhenius_fracture.reduced_shared_state_v1023 import (
    SharedReducedConfig,
    load_manifest,
    replay_shared_state,
    write_shared_result,
)


def _config_from_trace(payload: dict) -> SharedReducedConfig:
    front = payload["front_config"]
    mpz = payload["mpz_config"]
    tip = payload["tip_config"]
    transport = str(payload.get("transport_mode", "validated_scalar"))
    if transport == "unknown":
        transport = "validated_scalar"
    return SharedReducedConfig(
        r0_m=float(front["r0"]),
        checkpoint_da_um=float(front["da"]) * 1.0e6,
        mpz_length_um=float(mpz["length_m"]) * 1.0e6,
        mpz_n_bins=int(mpz["n_bins"]),
        wake_length_um=float(mpz["wake_length_m"]) * 1.0e6,
        wake_n_bins=int(mpz["wake_n_bins"]),
        source_bin_count=int(mpz["source_bin_count"]),
        blunting_length_um=float(mpz["blunting_length_m"]) * 1.0e6,
        shielding_core_m=float(mpz["shielding_core_m"]),
        forest_density_floor_m2=float(mpz["forest_density_floor_m2"]),
        mobile_shield_fraction=float(tip["mobile_shield_fraction"]),
        G_Pa=float(payload["G_Pa"]),
        poisson=float(payload["poisson"]),
        b_m=float(payload["b_m"]),
        cleavage_hits=float(front["m_hits"]),
        cleavage_tau_s=float(front["tau_c"]),
        transport_mode=transport,
        max_action_substep=float(tip["max_action_substep"]),
        max_translation_substep_m=float(tip["max_translation_substep_m"]),
        max_internal_steps=int(tip["max_internal_steps"]),
    ).validate()


def _array_metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.shape != actual.shape:
        raise ValueError(f"state-array shape mismatch: {expected.shape} != {actual.shape}")
    difference = actual - expected
    abs_max = float(np.max(np.abs(difference))) if difference.size else 0.0
    scale = max(float(np.max(np.abs(expected))) if expected.size else 0.0, 1.0)
    l2_expected = float(np.linalg.norm(expected.ravel()))
    l2_difference = float(np.linalg.norm(difference.ravel()))
    return {
        "maximum_abs_error": abs_max,
        "maximum_relative_to_expected_scale": abs_max / scale,
        "relative_l2_error": l2_difference / max(l2_expected, 1.0),
        "expected_sum": float(np.sum(expected)),
        "actual_sum": float(np.sum(actual)),
        "sum_error": float(np.sum(difference)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidate")
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--mode", default="full")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-10)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(
            f"output already exists: {args.out}\nUse a new versioned comparison path."
        )
    trace_root = args.trace_root.resolve()
    schedule = trace_root / "v10_2_3_2d_replay_schedule.csv"
    engine_config = trace_root / "v10_2_3_2d_engine_config.json"
    expected_state_path = trace_root / "v10_2_3_2d_final_state.npz"
    for path in (schedule, engine_config, expected_state_path):
        if not path.is_file():
            raise SystemExit(f"required trace file is missing: {path}")

    manifest = load_manifest(
        candidate_id=args.candidate,
        manifest_path=args.manifest,
    )
    config_payload = json.loads(engine_config.read_text())
    traced_candidate = str(
        config_payload.get("material_manifest", {}).get("candidate_id", "")
    )
    if traced_candidate and traced_candidate != manifest.candidate_id:
        raise SystemExit(
            f"trace candidate {traced_candidate} does not match manifest "
            f"{manifest.candidate_id}"
        )
    config = _config_from_trace(config_payload)
    replay = replay_shared_state(
        manifest,
        schedule,
        config,
        mode=args.mode,
    )
    args.out.mkdir(parents=True, exist_ok=False)
    write_shared_result(replay, args.out)

    expected = np.load(expected_state_path)
    actual = replay["_final_arrays"]
    names = sorted(set(expected.files).intersection(actual))
    missing_expected = sorted(set(actual).difference(expected.files))
    missing_actual = sorted(set(expected.files).difference(actual))
    metrics = {
        name: _array_metrics(expected[name], actual[name])
        for name in names
    }
    maximum_relative = max(
        (row["maximum_relative_to_expected_scale"] for row in metrics.values()),
        default=0.0,
    )
    maximum_abs = max(
        (row["maximum_abs_error"] for row in metrics.values()),
        default=0.0,
    )
    scalar_errors = [
        abs(float(value))
        for row in replay["history"]
        for key, value in row.items()
        if key.startswith("error_") and math.isfinite(float(value))
    ]
    maximum_scalar_abs = max(scalar_errors, default=0.0)
    passed = bool(
        not missing_expected
        and not missing_actual
        and replay["all_fired_flags_match"]
        and replay["raw_equals_effective_when_shielding_active"]
        and maximum_relative <= float(args.relative_tolerance)
        and maximum_abs <= float(args.absolute_tolerance)
    )
    assessment = {
        "schema": "v10.2.3_shared_state_replay_equivalence_assessment",
        "candidate_id": manifest.candidate_id,
        "mode": args.mode,
        "n_schedule_rows": replay["n_schedule_rows"],
        "state_arrays_compared": names,
        "missing_expected_arrays": missing_expected,
        "missing_actual_arrays": missing_actual,
        "array_metrics": metrics,
        "maximum_array_relative_error": maximum_relative,
        "maximum_array_absolute_error": maximum_abs,
        "maximum_scalar_absolute_error": maximum_scalar_abs,
        "all_fired_flags_match": replay["all_fired_flags_match"],
        "raw_equals_effective_when_shielding_active": replay[
            "raw_equals_effective_when_shielding_active"
        ],
        "relative_tolerance": float(args.relative_tolerance),
        "absolute_tolerance": float(args.absolute_tolerance),
        "passed": passed,
    }
    (args.out / "shared_state_replay_equivalence.json").write_text(
        json.dumps(assessment, indent=2)
    )
    print(json.dumps(assessment, indent=2), flush=True)
    if not passed:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
