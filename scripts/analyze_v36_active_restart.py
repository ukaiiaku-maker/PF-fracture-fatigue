#!/usr/bin/env python3
"""Summarize an interrupted/uninterrupted V3.6 active-state comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_generation(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = path / "v3_6_restart.json" if path.is_dir() else path
    manifest = json.loads(manifest_path.read_text())
    generation = manifest_path.parent / manifest["generation"]
    raw = generation.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["generation_sha256"]:
        raise RuntimeError(f"generation hash mismatch: {generation}")
    return json.loads(raw), {
        "manifest": str(manifest_path.resolve()),
        "generation": str(generation.resolve()),
        "generation_sha256": hashlib.sha256(raw).hexdigest(),
    }


def array(payload: dict[str, Any]) -> np.ndarray:
    if payload.get("kind") != "ndarray":
        raise TypeError("expected serialized ndarray")
    result = np.asarray(payload["data"], dtype=np.dtype(payload["dtype"]))
    return result.reshape(payload["shape"])


def numeric_differences(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if type(left) is not type(right):
        return [{"path": path, "kind": "type", "left": type(left).__name__,
                 "right": type(right).__name__}]
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}/{key}"
            if key not in left or key not in right:
                rows.append({"path": child, "kind": "missing"})
            else:
                rows.extend(numeric_differences(left[key], right[key], child))
    elif isinstance(left, list):
        if len(left) != len(right):
            rows.append({"path": path, "kind": "length", "left": len(left),
                         "right": len(right)})
        else:
            for index, (first, second) in enumerate(zip(left, right)):
                rows.extend(numeric_differences(first, second, f"{path}/{index}"))
    elif left != right:
        row: dict[str, Any] = {"path": path, "kind": "value", "left": left, "right": right}
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            row["absolute_difference"] = abs(float(left) - float(right))
            row["scaled_difference"] = row["absolute_difference"] / max(
                abs(float(left)), abs(float(right)), 1.0,
            )
        rows.append(row)
    return rows


def state_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    history = payload["history"]["active_ledger"]
    gamma = np.asarray(history["plastic_shear"], dtype=float)
    swept = np.asarray(history["cumulative_abs_swept_area_per_volume"], dtype=float)
    native = payload["state"]
    mobile = array(native["mobile_m2"])
    accumulated = array(native["accumulated_slip_m2"])
    activation = array(native["cumulative_source_activations"])
    return {
        "cycles": float(payload["cycles"]),
        "phase_intervals": int(payload["phase_intervals"]),
        "hazard": float(payload["hazard"]),
        "pending_threshold": float(payload["pending_threshold"]),
        "event_count": len(payload["events"]),
        "mobile_density_max_m2": float(np.max(np.abs(mobile))),
        "accumulated_slip_density_max_m2": float(np.max(np.abs(accumulated))),
        "cumulative_source_activations_max": float(np.max(np.abs(activation))),
        "plastic_shear_max_abs": float(np.max(np.abs(gamma))),
        "cumulative_abs_swept_area_per_volume_max": float(np.max(swept)),
        "projected_beta_p_max_abs": float(payload["last_metrics"]["projected_beta_p_max_abs"]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninterrupted", type=Path, required=True)
    parser.add_argument("--restarted", type=Path, required=True)
    parser.add_argument("--interruption", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    uninterrupted, uninterrupted_source = load_generation(args.uninterrupted)
    restarted, restarted_source = load_generation(args.restarted)
    interruption, interruption_source = load_generation(args.interruption)
    differences = numeric_differences(uninterrupted, restarted)
    metrics = state_metrics(uninterrupted)
    interruption_metrics = state_metrics(interruption)
    passed = (
        not differences
        and metrics["mobile_density_max_m2"] > 0.0
        and metrics["cumulative_source_activations_max"] > 0.0
        and metrics["plastic_shear_max_abs"] > 0.0
        and metrics["cumulative_abs_swept_area_per_volume_max"] > 0.0
    )
    report = {
        "schema": "v3.6-active-native-restart-qualification/1",
        "status": "PASS" if passed else "FAIL",
        "scope": "bounded same-row native positive-R cyclic restart qualification",
        "load_MPa_sqrt_m": uninterrupted["case"]["Kmax_MPa_sqrt_m"],
        "phase_steps": uninterrupted["case"]["phase_steps"],
        "seed": uninterrupted["case"]["seed"],
        "interruption": {**interruption_source, **interruption_metrics},
        "terminal_uninterrupted": {**uninterrupted_source, **metrics},
        "terminal_restarted": {**restarted_source, **state_metrics(restarted)},
        "terminal_payload_exact": not differences,
        "difference_count": len(differences),
        "differences": differences,
        "return_ledger_interpretation": (
            "The established R=0.1 opening waveform did not reverse slip sign; "
            "the cumulative-absolute swept ledger was nevertheless nonzero and "
            "restart-owned. No reversal was prescribed for this coverage test."
        ),
        "scientific_limit": (
            "This is restart/integration evidence for a populated native state, "
            "not a new fatigue-production or calibrated wake claim."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown = args.out.with_suffix(".md")
    markdown.write_text(
        "# V3.6 populated native restart qualification\n\n"
        f"Status: **{report['status']}**. At {report['load_MPa_sqrt_m']:.6f} MPa sqrt(m), "
        f"the fresh-process interrupted and uninterrupted terminal payloads were "
        f"{'exactly equal' if report['terminal_payload_exact'] else 'different'}.\n\n"
        f"The terminal native state contained a maximum mobile density of "
        f"{metrics['mobile_density_max_m2']:.6e} m^-2, maximum cumulative source "
        f"activation {metrics['cumulative_source_activations_max']:.6e}, maximum "
        f"material-frame plastic shear {metrics['plastic_shear_max_abs']:.6e}, and "
        f"cumulative absolute swept area per volume "
        f"{metrics['cumulative_abs_swept_area_per_volume_max']:.6e}.\n\n"
        + report["return_ledger_interpretation"] + "\n\n"
        + report["scientific_limit"] + "\n"
    )
    if not passed:
        raise SystemExit("V3.6 active restart qualification failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
