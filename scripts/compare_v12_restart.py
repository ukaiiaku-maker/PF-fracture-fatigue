#!/usr/bin/env python3
"""Compare two V12 accepted-boundary checkpoints field by field."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from arrhenius_fracture.multifront_checkpoint_v12 import (
    checkpoint_field_hashes,
    load_accepted_boundary_checkpoint_v12,
)


def _array_comparison(left: Any, right: Any) -> dict[str, Any]:
    a = np.asarray(left)
    b = np.asarray(right)
    same_shape = a.shape == b.shape
    exact = bool(same_shape and np.array_equal(a, b))
    return {
        "shape": list(a.shape),
        "same_shape": same_shape,
        "byte_exact": exact,
        "maximum_absolute_difference": (
            0.0 if exact else float(np.max(np.abs(a - b)))
        ) if same_shape and a.size else None,
    }


def _scalar_differences(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        rows = []
        for key in sorted(set(left) | set(right)):
            rows.extend(_scalar_differences(
                left.get(key, "<missing>"), right.get(key, "<missing>"),
                f"{path}.{key}" if path else str(key),
            ))
        return rows
    if isinstance(left, list) and isinstance(right, list):
        rows = []
        for index in range(max(len(left), len(right))):
            rows.extend(_scalar_differences(
                left[index] if index < len(left) else "<missing>",
                right[index] if index < len(right) else "<missing>",
                f"{path}[{index}]",
            ))
        return rows
    return [] if left == right else [{"path": path, "uninterrupted": left, "restarted": right}]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("uninterrupted", type=Path)
    parser.add_argument("restarted", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    left = load_accepted_boundary_checkpoint_v12(args.uninterrupted)
    right = load_accepted_boundary_checkpoint_v12(args.restarted)
    fem_arrays = {}
    for label, name in (
        ("displacement", "displacement"),
        ("ep", "ep_gp"),
        ("rho", "rho_gp"),
        ("damage", "damage"),
        ("vacancy", "vacancy_fraction_gp"),
    ):
        fem_arrays[label] = _array_comparison(
            getattr(left.accepted_fem_state, name),
            getattr(right.accepted_fem_state, name),
        )
    fem_arrays["sigma_gp"] = _array_comparison(
        left.accepted_stress_field, right.accepted_stress_field,
    )
    left_hashes = checkpoint_field_hashes(
        left.runtime,
        accepted_fem_state_reference={"accepted_state_id": left.runtime.accepted_state_id},
    )
    right_hashes = checkpoint_field_hashes(
        right.runtime,
        accepted_fem_state_reference={"accepted_state_id": right.runtime.accepted_state_id},
    )
    field_equal = {key: left_hashes[key] == right_hashes[key] for key in left_hashes}
    result = {
        "schema": "v12.restart-equivalence-comparison/1",
        "uninterrupted": str(args.uninterrupted.resolve()),
        "restarted": str(args.restarted.resolve()),
        "accepted_boundary": {
            "physical_time_s_equal": left.physical_time_s == right.physical_time_s,
            "accepted_opening_m_equal": left.accepted_opening_m == right.accepted_opening_m,
            "step_count_equal": left.step_count == right.step_count,
            "mechanics_source_identity_equal": (
                left.mechanics_source_identity == right.mechanics_source_identity
            ),
            "accepted_state_id_equal": (
                left.runtime.accepted_state_id == right.runtime.accepted_state_id
            ),
            "stress_field_state_id_equal": (
                left.runtime.stress_field_state_id == right.runtime.stress_field_state_id
            ),
        },
        "fem_arrays": fem_arrays,
        "runtime_field_equal": field_equal,
        "mutable_engine_metadata_differences": _scalar_differences(
            {
                key: {field: item for field, item in value.to_dict().items()
                      if field != "checkpoint_payload_b64"}
                for key, value in left.runtime.process_engines.items()
            },
            {
                key: {field: item for field, item in value.to_dict().items()
                      if field != "checkpoint_payload_b64"}
                for key, value in right.runtime.process_engines.items()
            },
        ),
        "all_fem_arrays_byte_exact": all(row["byte_exact"] for row in fem_arrays.values()),
        "all_runtime_fields_equal": all(field_equal.values()),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
