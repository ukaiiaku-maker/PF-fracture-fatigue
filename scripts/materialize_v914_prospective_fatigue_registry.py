#!/usr/bin/env python3
"""Materialize an exact v9.13-to-v9.14 prospective fatigue registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


PARENT_K300 = {
    "v913_zeroD_sobol_0202500": 26.28653661187115,
    "v913_zeroD_sobol_0242980": 26.530904648171045,
}
PARENT_REFERENCE_DELTAK = {
    "v913_zeroD_sobol_0202500": 21.02530765128298,
    "v913_zeroD_sobol_0242980": 21.289546465050222,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--k300-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def physical_fingerprint(row: pd.Series, fields: list[str]) -> str:
    payload = {field: float(row[field]) for field in fields}
    if any(not math.isfinite(value) for value in payload.values()):
        raise ValueError("nonfinite active parameter")
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def main() -> int:
    args = parse_args()
    registry = pd.read_csv(args.registry)
    selection = pd.read_csv(args.selection)
    cases = pd.read_csv(args.k300_results)
    cases = cases[cases.temperature_K.eq(300.0)]
    k300 = cases.set_index("candidate_id")["K_50um_MPa_sqrt_m"].astype(float)
    selected = registry[registry.prospective_candidate_id.isin(selection.candidate_id)].copy()
    if set(selected.prospective_candidate_id) != set(selection.candidate_id):
        raise RuntimeError("selection and prospective registry do not agree")
    active = [
        field
        for field in (
            "Tref_K", "cleave_G00_eV", "cleave_gT_eV_per_K", "cleave_sigc0_GPa",
            "cleave_sT_GPa_per_K", "cleave_exp_a", "cleave_exp_n", "cleave_floor_frac",
            "emit_G00_eV", "emit_gT_eV_per_K", "emit_sigc0_GPa", "emit_sT_GPa_per_K",
            "emit_exp_a", "emit_exp_n", "emit_floor_frac", "peierls_H0_eV",
            "peierls_activation_entropy_kB", "peierls_exp_a", "peierls_exp_n", "peierls_nu0_s",
            "taylor_H0_eV", "taylor_activation_entropy_kB", "taylor_exp_a", "taylor_exp_n",
            "taylor_nu0_s", "rho_source0_m2", "taylor_corr_rho_c_m2", "taylor_corr_scale", "c_blunt",
        )
        if field in selected.columns
    ]
    if len(active) != 29:
        raise RuntimeError(f"expected 29 active parameters, found {len(active)}")
    rows = []
    audit = []
    for _, source in selected.iterrows():
        row = source.copy()
        candidate_id = str(row.prospective_candidate_id)
        parent_id = str(row.parent_candidate_id)
        value = float(k300.loc[candidate_id])
        parent = PARENT_K300[parent_id]
        relative = abs(value - parent) / parent
        if relative > 0.05 + 1e-12:
            raise RuntimeError(f"candidate exceeds exact K300 gate: {candidate_id} {relative}")
        source_fingerprint = physical_fingerprint(row, active)
        if source_fingerprint != str(row.parameter_fingerprint):
            raise RuntimeError(f"source fingerprint mismatch: {candidate_id}")
        record = {field: row[field] for field in active}
        record.update(
            {
                "candidate_id": candidate_id,
                "material_class": "endurance_knee",
                "campaign_parent_id": parent_id,
                "prospective_design_family": row.design_family,
                "prospective_design_role": row.design_role,
                "source_parameter_fingerprint": source_fingerprint,
                "stageA_K50_300K_MPa_sqrt_m": value,
                "stageA_parent_K50_300K_MPa_sqrt_m": parent,
                "stageA_relative_error": relative,
                "stageA_tier": "near",
                "stageA_selected_for_endurance_knee": "true",
                "stageA_endurance_relative_tolerance_limit": 0.05,
                "fatigue_reference_deltaK_MPa_sqrt_m": (
                    PARENT_REFERENCE_DELTAK[parent_id] * value / parent
                ),
                "fatigue_reference_provenance": "exact_parent_reference_scaled_by_measured_candidate_parent_K50_ratio",
                "R": 0.1,
                "frequency_Hz": 1000.0,
                "temperature_K": 300.0,
                "target_extension_um": 100.0,
            }
        )
        transferred_fingerprint = physical_fingerprint(pd.Series(record), active)
        if transferred_fingerprint != source_fingerprint:
            raise RuntimeError(f"round-trip physical mismatch: {candidate_id}")
        record["parameter_fingerprint"] = transferred_fingerprint
        rows.append(record)
        audit.append(
            {
                "candidate_id": candidate_id,
                "parent_candidate_id": parent_id,
                "source_parameter_fingerprint": source_fingerprint,
                "fatigue_parameter_fingerprint": transferred_fingerprint,
                "round_trip_identity": True,
                "active_parameter_count": len(active),
                "K300_MPa_sqrt_m": value,
                "parent_K300_MPa_sqrt_m": parent,
                "K300_relative_error": relative,
                "fatigue_reference_deltaK_MPa_sqrt_m": record["fatigue_reference_deltaK_MPa_sqrt_m"],
                "parameter_refit_for_fatigue": False,
                "physics_changed": False,
            }
        )
    output = pd.DataFrame(rows).sort_values("candidate_id")
    if output.parameter_fingerprint.duplicated().any():
        raise RuntimeError("duplicate transferred parameter fingerprint")
    args.out.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out / "prospective_fatigue_registry.csv", index=False)
    pd.DataFrame(audit).to_csv(args.out / "prospective_fatigue_registry_roundtrip_audit.csv", index=False)
    (args.out / "prospective_fatigue_registry_manifest.json").write_text(
        json.dumps(
            {
                "schema": "v914_prospective_fatigue_registry_v1",
                "candidate_count": len(output),
                "active_parameter_count": len(active),
                "all_round_trip_identity": True,
                "parameter_refit_for_fatigue": False,
                "physics_changed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"V914_PROSPECTIVE_REGISTRY_COMPLETE candidates={len(output)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
