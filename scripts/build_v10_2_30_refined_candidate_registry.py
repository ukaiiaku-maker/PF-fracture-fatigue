#!/usr/bin/env python3
"""Translate immutable legacy candidate coordinates into a refined-v10 registry."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ATLAS_COORDINATES = (
    "cleave_G00_eV", "cleave_sigc0_GPa", "cleave_exp_a",
    "cleave_exp_n", "cleave_floor_frac", "emit_G00_eV",
    "emit_sigc0_GPa",
    "emit_exp_a", "emit_exp_n", "emit_floor_frac", "peierls_H0_eV",
    "peierls_activation_entropy_kB", "peierls_exp_a", "peierls_exp_n",
    "taylor_H0_eV", "taylor_activation_entropy_kB",
    "taylor_exp_a", "taylor_exp_n", "rho_source0_m2",
    "taylor_corr_rho_c_m2", "taylor_corr_scale", "c_blunt",
)

# These coordinates occur in older expanded CSV exports but are common-physics
# constants, not dimensions of the immutable 22-D atlas.  Preserve them when
# present; otherwise retain the qualified-v10 template value.
OPTIONAL_COMMON_COORDINATES = (
    "cleave_gT_eV_per_K", "cleave_sT_GPa_per_K",
    "emit_gT_eV_per_K", "emit_sT_GPa_per_K",
    "peierls_nu0_s", "taylor_nu0_s",
)
DIRECT_COORDINATES = ATLAS_COORDINATES + OPTIONAL_COMMON_COORDINATES


def _sha(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--candidate-id", action="append", required=True)
    parser.add_argument("--template", type=Path, default=Path(
        "arrhenius_fracture/data/materials/v10_2_27_paper_four_class_registry.csv"))
    parser.add_argument("--out-registry", required=True, type=Path)
    parser.add_argument("--out-selection", required=True, type=Path)
    parser.add_argument("--n-bins", type=int, default=80)
    args = parser.parse_args(argv)
    if args.n_bins < 4:
        raise SystemExit("--n-bins must be at least 4")
    wanted = set(args.candidate_id)
    if args.source.suffix == ".parquet":
        import pandas as pd
        table = pd.read_parquet(args.source, filters=[("candidate_id", "in", sorted(wanted))])
        source = {
            str(row["candidate_id"]): {key: str(value) for key, value in row.items()}
            for row in table.to_dict(orient="records")
        }
    else:
        with args.source.open(newline="") as stream:
            source = {
                row["candidate_id"]: row for row in csv.DictReader(stream)
                if row["candidate_id"] in wanted
            }
    with args.template.open(newline="") as stream:
        reader = csv.DictReader(stream); fields = list(reader.fieldnames or []); template = next(reader)
    rows, candidates = [], []
    for candidate_id in args.candidate_id:
        original = source.get(candidate_id)
        if original is None:
            raise SystemExit(f"candidate absent from source: {candidate_id}")
        missing = [key for key in ATLAS_COORDINATES if original.get(key, "") == ""]
        if missing:
            raise SystemExit(f"candidate {candidate_id} lacks coordinates: {missing}")
        row = dict(template)
        row.update({key: original[key] for key in ATLAS_COORDINATES})
        row.update({
            key: original[key] for key in OPTIONAL_COMMON_COORDINATES
            if original.get(key, "") != ""
        })
        implicit_temperature_neutral = not any(
            original.get(key, "") != "" for key in (
                "cleave_gT_eV_per_K", "cleave_sT_GPa_per_K",
                "emit_gT_eV_per_K", "emit_sT_GPa_per_K",
            )
        )
        if implicit_temperature_neutral:
            for key in (
                "cleave_gT_eV_per_K", "cleave_sT_GPa_per_K",
                "emit_gT_eV_per_K", "emit_sT_GPa_per_K",
            ):
                row[key] = "0.0"
        source_tref = float(original.get("Tref_K", row["Tref_K"]))
        if source_tref != float(row["Tref_K"]):
            if float(original["cleave_gT_eV_per_K"]) != 0.0 or float(original["emit_gT_eV_per_K"]) != 0.0 or float(original["cleave_sT_GPa_per_K"]) != 0.0 or float(original["emit_sT_GPa_per_K"]) != 0.0:
                raise SystemExit(
                    f"candidate {candidate_id} has nonzero temperature slopes; "
                    "Tref normalization would change physics"
                )
        option = "v10230_refined_" + candidate_id.lower()
        row.update({
            "option_key": option, "candidate_id": candidate_id,
            "material_class": "DBTT", "role": "refined v10 candidate search",
            "mechanism_summary": "immutable legacy coordinates on qualified v10 common physics",
            "validation_status": "unmeasured refined-v10 search candidate",
            "n_bins_recommended": str(args.n_bins),
        })
        coordinate_payload = {key: original[key] for key in ATLAS_COORDINATES}
        candidates.append({
            "candidate_id": candidate_id, "option_key": option,
            "source_coordinate_sha256": _sha(coordinate_payload),
            "source_row_sha256": _sha(original),
            "source_Tref_K": source_tref,
            "qualified_Tref_K": float(row["Tref_K"]),
            "implicit_temperature_neutral_coordinates": implicit_temperature_neutral,
            "Tref_normalization_rate_invariant": source_tref == float(row["Tref_K"]) or all(
                float(original[key]) == 0.0 for key in (
                    "cleave_gT_eV_per_K", "emit_gT_eV_per_K",
                    "cleave_sT_GPa_per_K", "emit_sT_GPa_per_K"
                )
            ),
        })
        rows.append(row)
    args.out_registry.parent.mkdir(parents=True, exist_ok=True)
    with args.out_registry.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    registry_sha = hashlib.sha256(args.out_registry.read_bytes()).hexdigest()
    selection = {
        "schema": "v10.2.30_refined_candidate_registry_v1",
        "canonical_option_order": [row["option_key"] for row in rows],
        "installed_registry_sha256": registry_sha,
        "source": str(args.source.resolve()), "candidates": candidates,
        "common_physics_source": str(args.template.resolve()),
        "common_physics_changed": False, "numerical_bins": args.n_bins,
        "legacy_m64_label_applicable": False,
    }
    args.out_selection.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
