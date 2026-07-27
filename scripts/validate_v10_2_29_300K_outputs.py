#!/usr/bin/env python3
"""Validate the focused v10.2.29 300 K monotonic/cyclic gate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

CANONICAL = {
    "v913_paper_peak01_0242980_persistent_sites": "v913_zeroD_sobol_0242980",
    "v913_paper_dbtt01_0202500_persistent_sites": "v913_zeroD_sobol_0202500",
    "v913_paper_weakT01_0129902_persistent_sites": "v913_zeroD_sobol_0129902",
    "v913_paper_ceramic01_0077080_persistent_sites": "v913_zeroD_sobol_0077080",
}


def _load(path: Path) -> Any:
    if not path.is_file():
        raise AssertionError(f"missing required file: {path}")
    return json.loads(path.read_text())


def _finite_nonnegative(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise AssertionError(f"{name} must be finite and nonnegative; got {value!r}")
    return number


def _kinetic_audit(root: Path) -> dict:
    return _load(root / "kinetic_tip_cell_audit_v101.json")


def _persistent_records(root: Path, cyclic: bool) -> list[dict]:
    payload = _kinetic_audit(root)
    records = list(payload.get("records", []))
    if not records:
        raise AssertionError(f"no kinetic audit records in {root}")
    selected = [
        record
        for record in records
        if (str(record.get("loading_mode", "monotonic")) == "cyclic") is cyclic
    ]
    if not selected:
        mode = "cyclic" if cyclic else "monotonic"
        raise AssertionError(f"no {mode} records in {root}")
    for index, record in enumerate(selected):
        prefix = f"{root}:record[{index}]"
        if record.get("persistent_source_inventory_active") is not False:
            raise AssertionError(f"{prefix}: finite source inventory became active")
        _finite_nonnegative(
            record.get("persistent_site_multiplicity_per_system", 0.0),
            prefix + ":multiplicity",
        )
        if float(record.get("persistent_site_multiplicity_per_system", 0.0)) <= 0.0:
            raise AssertionError(f"{prefix}: persistent multiplicity must be positive")
        for key in (
            "persistent_site_front_width_m",
            "persistent_site_source_area_m2",
            "persistent_tip_radius_m",
        ):
            if _finite_nonnegative(record.get(key, 0.0), prefix + ":" + key) <= 0.0:
                raise AssertionError(f"{prefix}: {key} must be positive")
        _finite_nonnegative(
            record.get("persistent_aggregate_emission_hazard_s", 0.0),
            prefix + ":aggregate_hazard",
        )
        _finite_nonnegative(
            record.get("persistent_sigma_back_Pa", 0.0),
            prefix + ":sigma_back",
        )
        if cyclic:
            requested = _finite_nonnegative(record.get("cycles_requested", 0.0), prefix + ":requested")
            consumed = _finite_nonnegative(record.get("cycles_consumed", 0.0), prefix + ":consumed")
            unused = _finite_nonnegative(record.get("cycles_unused", 0.0), prefix + ":unused")
            if consumed > requested + 1.0e-10 * max(requested, 1.0):
                raise AssertionError(f"{prefix}: consumed cycles exceed requested cycles")
            if not math.isclose(
                requested - consumed,
                unused,
                rel_tol=1.0e-10,
                abs_tol=1.0e-10,
            ):
                raise AssertionError(f"{prefix}: requested-consumed != unused")
            if record.get("event_localized") and unused <= 0.0:
                raise AssertionError(f"{prefix}: localized event has no unused proposed cycles")
            if record.get("engine_native_cycle_predictor") is not True:
                raise AssertionError(f"{prefix}: engine-native predictor not recorded")
    return selected


def _selection(root: Path) -> tuple[str, str]:
    payload = _load(root / "v10_2_22_parameter_selection.json")
    option = str(payload.get("option_key", ""))
    candidate = str(payload.get("candidate_id", ""))
    if CANONICAL.get(option) != candidate:
        raise AssertionError(f"noncanonical option/candidate in {root}: {option!r}/{candidate!r}")
    row = payload.get("exact_registry_row") or {}
    temperature = row.get("Tref_K")
    if temperature is None:
        raise AssertionError(f"selection audit lacks exact full-precision registry row: {root}")
    return option, candidate


def _persistent_model(root: Path) -> None:
    payload = _load(root / "v10_2_22_persistent_site_model.json")
    required = {
        "persistent_sites": True,
        "finite_source_inventory": False,
        "source_refresh": False,
        "backstress_limited_emission": True,
        "dynamic_tip_blunting": True,
        "moving_frame_resharpening": True,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise AssertionError(f"{root}: persistent model mismatch {key}={payload.get(key)!r}")


def _compare_numeric_csvs(reference: Path, overlay: Path) -> int:
    count = 0
    ref_files = {path.relative_to(reference): path for path in reference.rglob("*.csv")}
    new_files = {path.relative_to(overlay): path for path in overlay.rglob("*.csv")}
    for relative in sorted(set(ref_files) & set(new_files)):
        try:
            ref = np.genfromtxt(ref_files[relative], delimiter=",", names=True)
            new = np.genfromtxt(new_files[relative], delimiter=",", names=True)
        except (OSError, ValueError, TypeError):
            continue
        if ref.dtype.names is None or new.dtype.names is None:
            continue
        if ref.dtype.names != new.dtype.names or ref.shape != new.shape:
            raise AssertionError(f"numeric CSV structure differs: {relative}")
        for name in ref.dtype.names:
            a = np.asarray(ref[name])
            b = np.asarray(new[name])
            if a.dtype.kind not in "iufcb" or b.dtype.kind not in "iufcb":
                continue
            if not np.array_equal(a, b, equal_nan=True):
                raise AssertionError(f"monotonic numeric output differs: {relative}:{name}")
        count += 1
    if count < 1:
        raise AssertionError(
            f"no common numeric CSV outputs were compared between {reference} and {overlay}"
        )
    return count


def validate_monotonic_pair(reference: Path, overlay: Path) -> dict:
    if _selection(reference) != _selection(overlay):
        raise AssertionError("monotonic pair selected different exact parameter rows")
    _persistent_model(reference)
    _persistent_model(overlay)
    ref_records = _persistent_records(reference, cyclic=False)
    new_records = _persistent_records(overlay, cyclic=False)
    if ref_records != new_records:
        raise AssertionError("v10.2.29 monotonic kinetic audit is not identical to v10.2.28")
    compared = _compare_numeric_csvs(reference, overlay)
    return {
        "reference": str(reference),
        "overlay": str(overlay),
        "records": len(ref_records),
        "numeric_csvs_compared": compared,
    }


def validate_cyclic(root: Path) -> dict:
    option, candidate = _selection(root)
    _persistent_model(root)
    records = _persistent_records(root, cyclic=True)
    fatigue = _load(root / "v10_2_29_fatigue_model_audit.json")
    required = {
        "parameter_refit": False,
        "persistent_site_source": True,
        "finite_source_inventory": False,
        "source_depletion": False,
        "source_refresh": False,
        "explicit_recovery": False,
        "engine_native_cycle_predictor": True,
        "legacy_fatigue_barrier_predictor_used": False,
        "duplicate_spatial_fatigue_state": False,
        "full_field_cyclic_mechanics": False,
        "consumed_cycle_accounting": True,
    }
    for key, expected in required.items():
        if fatigue.get(key) != expected:
            raise AssertionError(f"{root}: fatigue model mismatch {key}={fatigue.get(key)!r}")
    consumed = sum(float(record.get("cycles_consumed", 0.0)) for record in records)
    localized = sum(bool(record.get("event_localized", False)) for record in records)
    return {
        "root": str(root),
        "option": option,
        "candidate": candidate,
        "records": len(records),
        "cycles_consumed_from_audit": consumed,
        "localized_event_records": localized,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbtt-v10228", type=Path, required=True)
    parser.add_argument("--dbtt-v10229", type=Path, required=True)
    parser.add_argument("--weakt-v10228", type=Path, required=True)
    parser.add_argument("--weakt-v10229", type=Path, required=True)
    parser.add_argument("--cyclic", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = {
        "schema": "v10.2.29_300K_validation_v1",
        "temperature_K": 300.0,
        "monotonic_pairs": [
            validate_monotonic_pair(args.dbtt_v10228, args.dbtt_v10229),
            validate_monotonic_pair(args.weakt_v10228, args.weakt_v10229),
        ],
        "cyclic": validate_cyclic(args.cyclic),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
