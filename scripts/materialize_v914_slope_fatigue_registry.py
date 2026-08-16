#!/usr/bin/env python3
"""Transfer fracture-qualified slope candidates into v9.14 bit-for-bit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


ACTIVE_FIELDS = (
    "Tref_K", "cleave_G00_eV", "cleave_gT_eV_per_K", "cleave_sigc0_GPa",
    "cleave_sT_GPa_per_K", "cleave_exp_a", "cleave_exp_n", "cleave_floor_frac",
    "emit_G00_eV", "emit_gT_eV_per_K", "emit_sigc0_GPa", "emit_sT_GPa_per_K",
    "emit_exp_a", "emit_exp_n", "emit_floor_frac", "peierls_H0_eV",
    "peierls_activation_entropy_kB", "peierls_exp_a", "peierls_exp_n", "peierls_nu0_s",
    "taylor_H0_eV", "taylor_activation_entropy_kB", "taylor_exp_a", "taylor_exp_n",
    "taylor_nu0_s", "rho_source0_m2", "taylor_corr_rho_c_m2", "taylor_corr_scale", "c_blunt",
)


def fingerprint(row: pd.Series) -> str:
    payload = {field: float(row[field]) for field in ACTIVE_FIELDS}
    if not all(math.isfinite(value) for value in payload.values()):
        raise RuntimeError("nonfinite active parameter")
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--fracture-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    design = pd.read_csv(args.registry)
    fracture = pd.read_csv(args.fracture_results)
    qualified = fracture[fracture.fracture_qualified_for_fatigue.astype(bool)].copy()
    source = design.set_index("prospective_candidate_id")
    rows, audits = [], []
    for result in qualified.itertuples(index=False):
        candidate = str(result.candidate_id)
        original = source.loc[candidate]
        original_fp = fingerprint(original)
        if original_fp != str(original.parameter_fingerprint):
            raise RuntimeError(f"source fingerprint mismatch: {candidate}")
        k300 = float(result.K300_MPa_sqrt_m)
        parent_k300 = float(original.parent_K300_MPa_sqrt_m)
        reference = float(original.parent_fatigue_reference_deltaK_MPa_sqrt_m) * k300 / parent_k300
        record = {field: original[field] for field in ACTIVE_FIELDS}
        record.update({
            "candidate_id": candidate,
            "material_class": "prospective_slope_design",
            "campaign_parent_id": original.parent_candidate_id,
            "parent_family": original.parent_family,
            "design_axis": original.design_axis,
            "design_sign": original.design_sign,
            "source_parameter_fingerprint": original_fp,
            "parameter_fingerprint": original_fp,
            "stageA_K50_300K_MPa_sqrt_m": k300,
            "stageA_parent_K50_300K_MPa_sqrt_m": parent_k300,
            "stageA_relative_error": float(result.K300_relative_error),
            "stageA_tier": "within_5pct_parent",
            "stageA_selected_for_endurance_knee": "true",
            "stageA_endurance_relative_tolerance_limit": float(result.K300_gate_tolerance),
            "fatigue_reference_deltaK_MPa_sqrt_m": reference,
            "fatigue_reference_provenance": "exact_parent_reference_scaled_by_measured_candidate_parent_K50_ratio",
            "R": 0.1, "frequency_Hz": 1000.0, "temperature_K": 300.0,
            "target_extension_um": 100.0,
        })
        if fingerprint(pd.Series(record)) != original_fp:
            raise RuntimeError(f"transfer changed physics: {candidate}")
        rows.append(record)
        audits.append({
            "candidate_id": candidate,
            "parent_candidate_id": original.parent_candidate_id,
            "source_parameter_fingerprint": original_fp,
            "fatigue_parameter_fingerprint": fingerprint(pd.Series(record)),
            "active_parameter_count": len(ACTIVE_FIELDS),
            "round_trip_identity": True,
            "K300_MPa_sqrt_m": k300,
            "parent_K300_MPa_sqrt_m": parent_k300,
            "K300_relative_error": float(result.K300_relative_error),
            "fatigue_reference_deltaK_MPa_sqrt_m": reference,
            "parameter_refit_for_fatigue": False,
            "physics_changed": False,
        })
    output = pd.DataFrame(rows).sort_values("candidate_id")
    if output.empty:
        raise RuntimeError("no fracture-qualified candidates")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "prospective_slope_fatigue_registry.csv"
    output.to_csv(path, index=False, float_format="%.17g")
    with path.open(newline="") as stream:
        runtime = {row["candidate_id"]: row for row in csv.DictReader(stream)}
    for row in output.itertuples(index=False):
        parsed = pd.Series(runtime[str(row.candidate_id)])
        if fingerprint(parsed) != str(row.parameter_fingerprint):
            raise RuntimeError(f"runtime CSV round-trip mismatch: {row.candidate_id}")
        for field in ACTIVE_FIELDS:
            if float(parsed[field]) != float(getattr(row, field)):
                raise RuntimeError(f"runtime field changed: {row.candidate_id}:{field}")
    pd.DataFrame(audits).to_csv(args.out / "prospective_slope_fatigue_registry_roundtrip_audit.csv", index=False)
    (args.out / "prospective_slope_fatigue_registry_manifest.json").write_text(json.dumps({
        "schema": "v914_prospective_slope_fatigue_registry_v1",
        "candidate_count": len(output),
        "active_parameter_count": len(ACTIVE_FIELDS),
        "all_round_trip_identity": True,
        "fatigue_specific_refit": False,
        "physics_changed": False,
    }, indent=2, sort_keys=True) + "\n")
    print(f"V914_SLOPE_REGISTRY_COMPLETE candidates={len(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
