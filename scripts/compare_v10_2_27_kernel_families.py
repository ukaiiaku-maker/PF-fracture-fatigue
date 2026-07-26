#!/usr/bin/env python3
"""Compare consecutive v10.2.27 target-grid kernel families for convergence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.kernel_registry_v10227 import family_physics_fingerprint

EXPECTED_SCHEMA = "v10.2.14_active_only_real_signed_2d_shielding_atlas"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _states(payload: dict) -> dict[str, dict]:
    result = {}
    for index, state in enumerate(payload.get("states", [])):
        state_id = str(state.get("state_id", f"state_{index}"))
        if state_id in result:
            raise ValueError(f"duplicate state_id {state_id!r}")
        result[state_id] = state
    if len(result) < 2:
        raise ValueError("kernel convergence comparison requires at least two states")
    return result


def _array(payload: dict, key: str) -> np.ndarray:
    value = np.asarray(payload.get(key), dtype=float)
    if value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError(f"{key} is empty or non-finite")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--maximum-relative-kernel-change", type=float, default=0.02)
    parser.add_argument(
        "--maximum-absolute-kernel-change-Pa-sqrt-m-per-line",
        type=float,
        default=100.0,
    )
    parser.add_argument("--maximum-extension-change-um", type=float, default=20.0)
    parser.add_argument(
        "--maximum-normalization-relative-change", type=float, default=1e-6
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    previous_path = args.previous.expanduser().resolve()
    current_path = args.current.expanduser().resolve()
    previous = json.loads(previous_path.read_text())
    current = json.loads(current_path.read_text())
    failures = []
    for name, payload in (("previous", previous), ("current", current)):
        if payload.get("schema") != EXPECTED_SCHEMA:
            failures.append(f"{name}_schema")
        if payload.get("production_parameterization_allowed") is not True:
            failures.append(f"{name}_production_parameterization_allowed")
        if payload.get("active_kernel_mechanically_measured") is not True:
            failures.append(f"{name}_active_kernel_mechanically_measured")
        if payload.get("wake_kernel_forced_zero") is not True:
            failures.append(f"{name}_wake_kernel_forced_zero")

    previous_x = _array(previous, "active_x_m")
    current_x = _array(current, "active_x_m")
    if previous_x.shape != current_x.shape:
        failures.append("active_grid_shape")
        active_grid_max_change_m = math.inf
    else:
        active_grid_max_change_m = float(np.max(np.abs(previous_x - current_x)))
        if active_grid_max_change_m > 1.0e-12:
            failures.append("active_grid_values")

    previous_states = _states(previous)
    current_states = _states(current)
    if set(previous_states) != set(current_states):
        failures.append("state_ids")
    common = sorted(set(previous_states) & set(current_states))

    maximum_extension_change_um = 0.0
    maximum_absolute_kernel_change = 0.0
    maximum_relative_kernel_change = 0.0
    squared_change = 0.0
    squared_scale = 0.0
    state_rows = []
    for state_id in common:
        left = previous_states[state_id]
        right = current_states[state_id]
        extension_change_um = 1.0e6 * abs(
            float(left["crack_extension_m"]) - float(right["crack_extension_m"])
        )
        maximum_extension_change_um = max(
            maximum_extension_change_um, extension_change_um
        )
        row_absolute = 0.0
        row_relative = 0.0
        for key in (
            "active_kernel_I_Pa_sqrt_m_per_signed_line",
            "active_kernel_II_Pa_sqrt_m_per_signed_line",
        ):
            first = _array(left, key)
            second = _array(right, key)
            if first.shape != second.shape:
                failures.append(f"{state_id}_{key}_shape")
                continue
            difference = second - first
            absolute = float(np.max(np.abs(difference)))
            scale = max(
                float(np.max(np.abs(first))),
                float(np.max(np.abs(second))),
                1.0,
            )
            relative = absolute / scale
            row_absolute = max(row_absolute, absolute)
            row_relative = max(row_relative, relative)
            maximum_absolute_kernel_change = max(
                maximum_absolute_kernel_change, absolute
            )
            maximum_relative_kernel_change = max(
                maximum_relative_kernel_change, relative
            )
            squared_change += float(np.sum(difference * difference))
            squared_scale += (
                float(np.sum(first * first) + np.sum(second * second)) / 2.0
            )
        state_rows.append(
            {
                "state_id": state_id,
                "previous_extension_m": float(left["crack_extension_m"]),
                "current_extension_m": float(right["crack_extension_m"]),
                "extension_change_um": extension_change_um,
                "maximum_absolute_kernel_change_Pa_sqrt_m_per_line": row_absolute,
                "maximum_relative_kernel_change": row_relative,
            }
        )

    relative_l2_change = math.sqrt(squared_change / max(squared_scale, 1.0e-300))
    kernel_change_passed = (
        maximum_relative_kernel_change
        <= float(args.maximum_relative_kernel_change)
        and relative_l2_change <= float(args.maximum_relative_kernel_change)
    ) or (
        maximum_absolute_kernel_change
        <= float(args.maximum_absolute_kernel_change_Pa_sqrt_m_per_line)
        and relative_l2_change <= float(args.maximum_relative_kernel_change)
    )
    if not kernel_change_passed:
        failures.append("active_kernel_change")
    if maximum_extension_change_um > float(args.maximum_extension_change_um):
        failures.append("anchor_extension_change")

    previous_conversion = _array(
        previous, "activation_to_line_content_by_system"
    )
    current_conversion = _array(
        current, "activation_to_line_content_by_system"
    )
    previous_bounds = _array(previous, "source_capacity_bounds_per_system")
    current_bounds = _array(current, "source_capacity_bounds_per_system")
    normalization_relative_change = 0.0
    for key, first, second in (
        (
            "activation_to_line_content_by_system",
            previous_conversion,
            current_conversion,
        ),
        ("source_capacity_bounds_per_system", previous_bounds, current_bounds),
    ):
        if first.shape != second.shape:
            failures.append(f"{key}_shape")
            normalization_relative_change = math.inf
            continue
        relative = float(
            np.max(np.abs(second - first) / np.maximum(np.abs(first), 1.0e-300))
        )
        normalization_relative_change = max(normalization_relative_change, relative)
    if normalization_relative_change > float(
        args.maximum_normalization_relative_change
    ):
        failures.append("normalization_change")

    previous_sha = _sha256(previous_path)
    current_sha = _sha256(current_path)
    previous_physics = family_physics_fingerprint(previous_path)
    current_physics = family_physics_fingerprint(current_path)
    result = {
        "schema": "v10.2.27_kernel_self_consistency_comparison_v1",
        "previous_family_label": previous_path.parent.name + "/" + previous_path.name,
        "previous_family_sha256": previous_sha,
        "previous_family_physics_fingerprint": previous_physics,
        "current_family_label": current_path.parent.name + "/" + current_path.name,
        "current_family_sha256": current_sha,
        "current_family_physics_fingerprint": current_physics,
        "same_file_sha256": previous_sha == current_sha,
        "same_physics_fingerprint": previous_physics == current_physics,
        "state_count_previous": len(previous_states),
        "state_count_current": len(current_states),
        "active_grid_maximum_change_m": active_grid_max_change_m,
        "maximum_extension_change_um": maximum_extension_change_um,
        "maximum_absolute_kernel_change_Pa_sqrt_m_per_line": (
            maximum_absolute_kernel_change
        ),
        "maximum_relative_kernel_change": maximum_relative_kernel_change,
        "relative_l2_kernel_change": relative_l2_change,
        "maximum_normalization_relative_change": normalization_relative_change,
        "tolerances": {
            "maximum_relative_kernel_change": float(
                args.maximum_relative_kernel_change
            ),
            "maximum_absolute_kernel_change_Pa_sqrt_m_per_line": float(
                args.maximum_absolute_kernel_change_Pa_sqrt_m_per_line
            ),
            "maximum_extension_change_um": float(
                args.maximum_extension_change_um
            ),
            "maximum_normalization_relative_change": float(
                args.maximum_normalization_relative_change
            ),
        },
        "states": state_rows,
        "converged": not failures,
        "failures": sorted(set(failures)),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["converged"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
