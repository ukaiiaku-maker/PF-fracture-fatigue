#!/usr/bin/env python3
"""Classify one Stage 3 case without confusing censoring with failure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_single_summary(case_root: Path) -> dict[str, Any]:
    path = case_root / "summary.json"
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or len(payload) != 1:
        count = len(payload) if isinstance(payload, list) else "non-list"
        raise ValueError(f"expected one summary row in {path}; found {count}")
    return dict(payload[0])


def _load_crack_path(path: Path) -> np.ndarray:
    """Read both current plain-header CSVs and older commented-header files."""
    try:
        structured = np.genfromtxt(
            path,
            delimiter=",",
            names=True,
            dtype=float,
            encoding=None,
            autostrip=True,
        )
        names = structured.dtype.names
        if names:
            columns = [np.atleast_1d(np.asarray(structured[name], dtype=float)) for name in names]
            data = np.column_stack(columns)
            if data.ndim == 2 and data.shape[1] >= 2:
                return data
    except (TypeError, ValueError, OSError):
        pass

    # Compatibility fallback for older files whose header starts with '#'.
    data = np.genfromtxt(path, delimiter=",", comments="#", dtype=float, ndmin=2)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"expected at least two crack-path columns in {path}; got {data.shape}")
    data = data[np.all(np.isfinite(data[:, :2]), axis=1)]
    return data


def _projected_extension_m(case_root: Path, temperature_K: float) -> tuple[float | None, Path | None]:
    expected = case_root / f"crack_path_{int(round(temperature_K))}K.csv"
    candidates = [expected]
    candidates.extend(
        path for path in sorted(case_root.glob("crack_path_*K.csv"))
        if path != expected and "front" not in path.name and "branch" not in path.name
    )
    for path in candidates:
        if not path.is_file():
            continue
        data = _load_crack_path(path)
        if data.shape[0] < 2:
            continue
        return float(data[-1, 0] - data[0, 0]), path
    return None, None


def classify(case_root: Path, target_extension_um: float) -> dict[str, Any]:
    root = case_root.expanduser().resolve()
    summary = _load_single_summary(root)
    temperature = float(summary["T"])
    projected, crack_path = _projected_extension_m(root, temperature)
    target_m = float(target_extension_um) * 1.0e-6
    tolerance_m = max(0.5e-6, 0.005 * target_m)
    reached = bool(projected is not None and projected >= target_m - tolerance_m)
    first_passage = summary.get("Kc_first_MPa_sqrt_m") is not None
    if reached:
        status = "complete_target_extension"
    elif first_passage:
        status = "incomplete_after_first_passage"
    else:
        status = "right_censored_no_first_passage"
    return {
        "schema": "v10.2.17_stage3_case_status",
        "case_root": str(root),
        "temperature_K": temperature,
        "target_extension_um": float(target_extension_um),
        "projected_extension_um": None if projected is None else projected * 1.0e6,
        "target_tolerance_um": tolerance_m * 1.0e6,
        "crack_path_file": None if crack_path is None else str(crack_path),
        "first_passage_recorded": first_passage,
        "Kc_first_MPa_sqrt_m": summary.get("Kc_first_MPa_sqrt_m"),
        "status": status,
        "complete": reached,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", required=True, type=Path)
    parser.add_argument("--target-extension-um", required=True, type=float)
    args = parser.parse_args()
    payload = classify(args.case_root, args.target_extension_um)
    output = args.case_root / "stage3_case_status.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for marker in ("COMPLETE", "INCOMPLETE", "CENSORED"):
        path = args.case_root / marker
        if path.exists():
            path.unlink()
    marker = {
        "complete_target_extension": "COMPLETE",
        "incomplete_after_first_passage": "INCOMPLETE",
        "right_censored_no_first_passage": "CENSORED",
    }[payload["status"]]
    (args.case_root / marker).write_text(payload["status"] + "\n")
    print(json.dumps({key: payload[key] for key in (
        "status", "temperature_K", "projected_extension_um", "Kc_first_MPa_sqrt_m",
        "crack_path_file",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
