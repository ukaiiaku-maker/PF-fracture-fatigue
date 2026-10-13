#!/usr/bin/env python3
"""Compare complete terminal native checkpoints for the F1 qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


IGNORED_KEYS = {
    "created_unix_s", "output_directory", "output_dir", "restart_from",
    "wall_time_s", "elapsed_seconds", "timestamp_unix_s",
}


def _generation(checkpoint: Path) -> tuple[dict[str, Any], Path]:
    manifest = json.loads((checkpoint / "run_state_checkpoint.json").read_text())
    directory = checkpoint / "run_state_generations" / manifest["generation"]
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"checkpoint hash mismatch for {directory / name}")
    return manifest, directory


def _normal(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normal(item) for key, item in value.items() if key not in IGNORED_KEYS}
    if isinstance(value, list):
        return [_normal(item) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        return Path(value).name
    return value


def _json_differences(left: Any, right: Any, path: str = "") -> list[str]:
    if type(left) is not type(right):
        return [f"{path}: type {type(left).__name__} != {type(right).__name__}"]
    if isinstance(left, dict):
        rows = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                rows.append(f"{path}/{key}: missing")
            else:
                rows.extend(_json_differences(left[key], right[key], f"{path}/{key}"))
        return rows
    if isinstance(left, list):
        if len(left) != len(right):
            return [f"{path}: length {len(left)} != {len(right)}"]
        rows = []
        for index, (a, b) in enumerate(zip(left, right)):
            rows.extend(_json_differences(a, b, f"{path}/{index}"))
        return rows
    if isinstance(left, float) and isinstance(right, float):
        if np.isnan(left) and np.isnan(right):
            return []
    return [] if left == right else [f"{path}: {left!r} != {right!r}"]


def compare(left: Path, right: Path) -> dict[str, Any]:
    left_manifest, left_dir = _generation(left)
    right_manifest, right_dir = _generation(right)
    arrays = {}
    with np.load(left_dir / "state.npz", allow_pickle=False) as a, np.load(
        right_dir / "state.npz", allow_pickle=False
    ) as b:
        for key in sorted(set(a.files) | set(b.files)):
            if key not in a.files or key not in b.files:
                arrays[key] = {"equal": False, "reason": "missing"}
                continue
            exact = np.array_equal(a[key], b[key], equal_nan=True)
            arrays[key] = {
                "equal": bool(exact),
                "shape": list(a[key].shape),
                "max_abs_difference": 0.0 if exact else float(
                    np.max(np.abs(np.asarray(a[key], dtype=float) - np.asarray(b[key], dtype=float)))
                ),
            }
    json_rows = {}
    for name in ("outer.json", "kinetic.json"):
        a = _normal(json.loads((left_dir / name).read_text()))
        b = _normal(json.loads((right_dir / name).read_text()))
        json_rows[name] = _json_differences(a, b)
    passed = all(row["equal"] for row in arrays.values()) and not any(json_rows.values())
    return {
        "schema": "f1-current-source-full-state-restart-comparison/1",
        "status": "PASS" if passed else "FAIL",
        "comparison": "same current source, loading, seed, and terminal physical step",
        "uninterrupted_generation": left_manifest["generation"],
        "restarted_generation": right_manifest["generation"],
        "state_arrays": arrays,
        "json_differences_after_nonphysical_provenance_normalization": json_rows,
        "complete_terminal_state_equal": passed,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninterrupted", type=Path, required=True)
    parser.add_argument("--restarted", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare(args.uninterrupted, args.restarted)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "PASS":
        raise SystemExit("terminal restart comparison failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
