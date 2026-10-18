#!/usr/bin/env python3
"""Create fixed-cohesion/vary-plasticity complete-material descendants.

This is an acquisition operation, not a fitted correction: the selected
cohesive vector and every donor EXP_floor vector are copied byte-for-byte from
the existing registry, then rebound as explicitly new complete materials.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_r10_global_mechanism_search import canonical_hash


ARMS = (
    ("smooth_exponential", "cohesive_athermal", "AS"),
    ("smooth_exponential", "cohesive_mixed", "MS"),
    ("bilinear", "cohesive_athermal", "AB"),
    ("bilinear", "cohesive_mixed", "MB"),
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--cohesive-candidate-id", required=True)
    parser.add_argument("--plasticity-source-ids", required=True,
                        help="Comma-separated source-material IDs.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    with args.registry.open(newline="") as stream:
        registry = list(csv.DictReader(stream))
    cohesive_matches = [r for r in registry
                        if r["candidate_id"] == args.cohesive_candidate_id]
    if len(cohesive_matches) != 1:
        raise SystemExit("cohesive candidate must resolve to exactly one row")
    cohesive = cohesive_matches[0]
    donor_ids = [x.strip() for x in args.plasticity_source_ids.split(",") if x.strip()]
    by_source = {}
    for row in registry:
        by_source.setdefault(row["source_material_candidate_id"], row)

    output = []
    cohesive_token = cohesive["source_material_candidate_id"].removeprefix("R10F_V2_")
    for donor_id in donor_ids:
        if donor_id not in by_source:
            raise SystemExit(f"missing plasticity source {donor_id}")
        donor = by_source[donor_id]
        donor_token = donor_id.removeprefix("R10F_V2_")
        physical_id = f"R10F_12H_FC_{cohesive_token}_VP_{donor_token}"
        physical_payload = {
            "family": cohesive["family"],
            "cohesive_parameter_vector": json.loads(cohesive["cohesive_parameter_vector"]),
            "plasticity_parameter_vector": json.loads(donor["parent_complete_parameter_vector"]),
            "cohesive_parameter_sha256": cohesive["cohesive_parameter_sha256"],
            "plasticity_physics_sha256": donor["plasticity_physics_sha256"],
            "acquisition": "R10_12H_FIXED_COHESION_VARY_PLASTICITY",
        }
        physical_sha = canonical_hash(physical_payload)
        for shape, mode, code in ARMS:
            row = dict(cohesive)
            row.update({
                "candidate_id": f"{physical_id}__{code}",
                "source_material_candidate_id": physical_id,
                "parent_candidate_id": donor["parent_candidate_id"],
                "parent_complete_row_sha256": donor["parent_complete_row_sha256"],
                "parent_complete_parameter_vector": donor["parent_complete_parameter_vector"],
                "plasticity_physics_sha256": donor["plasticity_physics_sha256"],
                "selected_shape": shape,
                "selected_mode": mode,
                "search_design": "FIXED_COHESION_VARY_PLASTICITY",
                "mechanism_prediction": "TO_BE_MEASURED_FROM_CURRENT_STATE",
                "plastic_exposure_log10": donor.get("plastic_exposure_log10", ""),
                "source_cohesive_candidate_id": cohesive["candidate_id"],
                "source_plasticity_material_id": donor_id,
                "physical_vector_sha256": physical_sha,
            })
            row["complete_row_sha256"] = canonical_hash({
                "physical_vector_sha256": physical_sha,
                "shape": shape,
                "mode": mode,
            })
            output.append(row)

    fields = []
    for row in output:
        for key in row:
            if key not in fields:
                fields.append(key)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps({
        "physical_vectors": len(donor_ids),
        "execution_identities": len(output),
        "cohesive_parameter_sha256": cohesive["cohesive_parameter_sha256"],
        "distinct_plasticity_fingerprints": len({r["plasticity_physics_sha256"] for r in output}),
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
