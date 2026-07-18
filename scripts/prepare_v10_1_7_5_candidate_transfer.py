#!/usr/bin/env python3
"""Build portable PF material manifests from reduced-model candidate rows.

The v9.10.4 narrow-DBTT campaign searched 29 constitutive parameters.  The
active-shielding cap was not one of them: ``ReducedFrontSettings`` fixed
``max_K_shield_MPa_sqrt_m=1.0`` for every candidate.  Consequently the ranking
CSV may omit that column.  This transfer utility injects the documented fixed
campaign value and records its provenance rather than inferring a cap from a
measured shielding response.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_SOURCE_FIELDS = (
    "candidate_id",
    "cleave_G00_eV",
    "cleave_gT_eV_per_K",
    "cleave_sigc0_GPa",
    "cleave_sT_GPa_per_K",
    "cleave_exp_a",
    "cleave_exp_n",
    "cleave_floor_frac",
    "emit_G00_eV",
    "emit_gT_eV_per_K",
    "emit_sigc0_GPa",
    "emit_sT_GPa_per_K",
    "emit_exp_a",
    "emit_exp_n",
    "emit_floor_frac",
    "peierls_H0_eV",
    "peierls_activation_entropy_kB",
    "peierls_exp_a",
    "peierls_exp_n",
    "peierls_nu0_s",
    "taylor_H0_eV",
    "taylor_activation_entropy_kB",
    "taylor_exp_a",
    "taylor_exp_n",
    "taylor_nu0_s",
    "taylor_corr_rho_c_m2",
    "taylor_corr_scale",
    "source_sites_per_system",
    "encounter_efficiency",
    "retained_recovery_rate_s",
    "source_refresh_length_um",
    "c_blunt",
)

FIXED_REDUCED_CAMPAIGN_FIELDS = {
    "max_K_shield_MPa_sqrt_m": 1.0,
}

MANIFEST_FIELDS = REQUIRED_SOURCE_FIELDS + tuple(FIXED_REDUCED_CAMPAIGN_FIELDS)
# Backward-compatible export used by the focused tests and external utilities.
REQUIRED_FIELDS = MANIFEST_FIELDS

MODES = (
    "full",
    "plasticity_off",
    "blunting_off",
    "backstress_off",
    "shielding_off",
    "background_field_off",
)


def _temperature_schedule(row: pd.Series) -> list[float]:
    for key in (
        "refinement_transition_temperatures_K",
        "moving_1d_temperatures_K",
        "temperatures_K_json",
    ):
        if key in row.index and pd.notna(row[key]):
            values = [float(x) for x in json.loads(str(row[key]))]
            if len(values) >= 2:
                return values
    low = row.get("coarse_transition_low_T_K")
    high = row.get("coarse_transition_high_T_K")
    if pd.notna(low) and pd.notna(high):
        return [float(low), float(high)]
    raise ValueError(f"candidate {row.get('candidate_id')} has no transition schedule")


def _normalize_source(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    missing_fields = sorted(set(REQUIRED_SOURCE_FIELDS).difference(source.columns))
    if missing_fields:
        raise ValueError(f"source is missing required manifest fields: {missing_fields}")

    normalized = source.copy()
    provenance: dict[str, str] = {}
    for field, fixed_value in FIXED_REDUCED_CAMPAIGN_FIELDS.items():
        if field not in normalized.columns:
            normalized[field] = float(fixed_value)
            provenance[field] = "v9.10.4 ReducedFrontSettings fixed campaign value"
        else:
            numeric = pd.to_numeric(normalized[field], errors="coerce")
            if numeric.isna().any() or not np.all(np.isfinite(numeric.to_numpy(dtype=float))):
                raise ValueError(f"source column {field!r} contains non-finite values")
            if (numeric < 0.0).any():
                raise ValueError(f"source column {field!r} must be nonnegative")
            normalized[field] = numeric.astype(float)
            provenance[field] = "source column"
    return normalized, provenance


def prepare(
    source: pd.DataFrame,
    candidate_ids: list[str],
    out: Path,
) -> pd.DataFrame:
    normalized, fixed_field_provenance = _normalize_source(source)

    selected = normalized[normalized.candidate_id.astype(str).isin(candidate_ids)].copy()
    missing_candidates = sorted(set(candidate_ids).difference(selected.candidate_id.astype(str)))
    if missing_candidates:
        raise ValueError(f"source is missing candidates: {missing_candidates}")
    if selected.candidate_id.astype(str).duplicated().any():
        duplicates = selected.loc[
            selected.candidate_id.astype(str).duplicated(False), "candidate_id"
        ].astype(str).tolist()
        raise ValueError(f"candidate rows are not unique: {duplicates}")

    order = {candidate: index for index, candidate in enumerate(candidate_ids)}
    selected["_order"] = selected.candidate_id.astype(str).map(order)
    selected = selected.sort_values("_order")

    manifest_root = out / "material_manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    case_rows: list[dict[str, object]] = []

    for _, row in selected.iterrows():
        candidate = str(row.candidate_id)
        candidate_root = manifest_root / candidate
        candidate_root.mkdir(parents=True, exist_ok=True)

        base = {field: row[field] for field in MANIFEST_FIELDS}
        base["target_class"] = "DBTT"
        base["candidate_id"] = candidate
        base_manifest = candidate_root / "candidate.csv"
        pd.DataFrame([base]).to_csv(base_manifest, index=False)

        blunting = dict(base)
        blunting["c_blunt"] = 0.0
        blunting_manifest = candidate_root / "candidate_blunting_off.csv"
        pd.DataFrame([blunting]).to_csv(blunting_manifest, index=False)

        schedule = _temperature_schedule(row)
        endpoints = [("low", float(schedule[0])), ("high", float(schedule[-1]))]
        bracket = str(row.get("transition_bracket", f"T{schedule[0]:g}_{schedule[-1]:g}K"))

        for endpoint, temperature in endpoints:
            for mode in MODES:
                case_rows.append(
                    {
                        "candidate_id": candidate,
                        "transition_bracket": bracket,
                        "endpoint": endpoint,
                        "T_K": temperature,
                        "mode": mode,
                        "material_manifest": str(
                            blunting_manifest if mode == "blunting_off" else base_manifest
                        ),
                        "tip_plasticity": 0 if mode == "plasticity_off" else 1,
                        "active_shielding": 0
                        if mode in {"plasticity_off", "shielding_off"}
                        else 1,
                        "backstress_scale": 0.0 if mode == "backstress_off" else 1.0,
                        "forest_density_floor_override_m2": 0.0
                        if mode == "background_field_off"
                        else "default",
                    }
                )

    cases = pd.DataFrame(case_rows)
    cases.to_csv(out / "transfer_cases.csv", index=False)
    cases.to_csv(out / "transfer_cases.tsv", index=False, sep="\t")
    selected.drop(columns=["_order"]).to_csv(out / "selected_candidate_rows.csv", index=False)
    (out / "fixed_field_provenance.json").write_text(
        json.dumps(
            {
                "schema": "v10.1.7.5_fixed_reduced_campaign_fields",
                "fields": {
                    field: {
                        "value": float(selected[field].iloc[0]),
                        "provenance": fixed_field_provenance[field],
                    }
                    for field in FIXED_REDUCED_CAMPAIGN_FIELDS
                },
            },
            indent=2,
        )
    )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["DBTT_A0003408", "DBTT_A0000353"],
    )
    args = parser.parse_args()

    source_path = args.source.resolve()
    out_path = args.out.resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    candidates = list(args.candidates)
    cases = prepare(pd.read_csv(source_path), candidates, out_path)

    provenance = {
        "schema": "v10.1.7.5_transfer_preparation",
        "source": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "candidate_ids": candidates,
        "n_candidates": len(candidates),
        "n_modes": len(MODES),
        "n_endpoints_per_candidate": 2,
        "n_cases": int(len(cases)),
        "fixed_campaign_fields": str(out_path / "fixed_field_provenance.json"),
    }
    (out_path / "transfer_preparation.json").write_text(json.dumps(provenance, indent=2))

    print(
        cases.groupby(["candidate_id", "transition_bracket"], as_index=False).agg(
            endpoint_temperatures=("T_K", lambda x: json.dumps(sorted(set(float(v) for v in x)))),
            n_cases=("mode", "count"),
        ).to_string(index=False),
        flush=True,
    )
    print(
        "fixed transfer field: max_K_shield_MPa_sqrt_m=1.0 "
        "(v9.10.4 ReducedFrontSettings)",
        flush=True,
    )
    print(f"wrote {len(cases)} cases to {out_path / 'transfer_cases.tsv'}", flush=True)


if __name__ == "__main__":
    main()
