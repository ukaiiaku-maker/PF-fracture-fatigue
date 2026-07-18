#!/usr/bin/env python3
"""Build portable PF material manifests from reduced-model candidate rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_FIELDS = (
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
    "max_K_shield_MPa_sqrt_m",
)

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


def prepare(source: pd.DataFrame, candidate_ids: list[str], out: Path) -> pd.DataFrame:
    missing_fields = sorted(set(REQUIRED_FIELDS).difference(source.columns))
    if missing_fields:
        raise ValueError(f"source is missing required manifest fields: {missing_fields}")

    selected = source[source.candidate_id.astype(str).isin(candidate_ids)].copy()
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

        base = {field: row[field] for field in REQUIRED_FIELDS}
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

    args.out.mkdir(parents=True, exist_ok=True)
    cases = prepare(pd.read_csv(args.source), list(args.candidates), args.out)
    print(
        cases.groupby(["candidate_id", "transition_bracket"], as_index=False).agg(
            endpoint_temperatures=("T_K", lambda x: json.dumps(sorted(set(float(v) for v in x)))),
            n_cases=("mode", "count"),
        ).to_string(index=False),
        flush=True,
    )
    print(f"wrote {len(cases)} cases to {args.out / 'transfer_cases.tsv'}", flush=True)


if __name__ == "__main__":
    main()
