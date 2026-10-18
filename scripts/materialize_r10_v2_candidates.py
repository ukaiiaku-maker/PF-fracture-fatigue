#!/usr/bin/env python3
"""Reconstruct exact, previously unpromoted V2 physical vectors by Sobol ID."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.cohesive_arrhenius import (  # noqa: E402
    BILINEAR, SMOOTH_EXPONENTIAL, FAMILY, OPENING_ALTERNATIVES_FAMILY,
)
from scripts.run_r10_global_mechanism_search import canonical_hash, read_sources  # noqa: E402
from scripts.run_r10_v2_opening_search import (  # noqa: E402
    MODES, SEED, TEMPERATURES, access_load, gamma_T, generate, material_row,
    mutate_dislocations, plastic_exposure, thermal_class, valid_transport,
    write_csv,
)

ARMS = (
    (SMOOTH_EXPONENTIAL, "cohesive_athermal", "AS"),
    (SMOOTH_EXPONENTIAL, "cohesive_mixed", "MS"),
    (BILINEAR, "cohesive_athermal", "AB"),
    (BILINEAR, "cohesive_mixed", "MB"),
)
FIXED_PARENT_IDS = {
    "P25_TRANSFER_V1_RANK1", "P40_TRANSFER_CALIBRATED_GEN2",
    "v913_zeroD_sobol_0202500", "v913_zeroD_sobol_0129902",
}


def read_anchors(path: Path) -> list[dict]:
    anchors = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            cohesive = json.loads(row["cohesive_parameter_vector"])
            if cohesive not in anchors:
                anchors.append(cohesive)
    return anchors


def reconstruct(index: int, points, mutations, sources: list[dict],
                anchors: list[dict]) -> tuple[str, dict, dict, str]:
    item = generate(index, points[index], sources)
    design_index = index % 3
    if design_index == 0:
        design = "FIXED_COHESION_VARY_PLASTICITY"
        cohesive = dict(anchors[(index // 3) % len(anchors)])
        base = sources[(index // (3 * len(anchors))) % len(sources)]
        parent = mutate_dislocations(base, mutations[index])
    elif design_index == 1:
        design = "FIXED_PLASTICITY_VARY_COHESION"
        cohesive = dict(item["cohesive"])
        fixed = [row for row in sources if row["candidate_id"] in FIXED_PARENT_IDS]
        base = fixed[(index // 3) % len(fixed)]
        parent = dict(base)
    else:
        design = "JOINT_COHESION_PLASTICITY"
        cohesive = dict(item["cohesive"])
        base = item["parent"]
        parent = mutate_dislocations(base, mutations[index])
    if min(gamma_T(cohesive, temperature) for temperature in TEMPERATURES) <= .2:
        raise ValueError(f"R10F_V2_{index:05d} violates the cohesive-energy domain")
    if not valid_transport(parent):
        raise ValueError(f"R10F_V2_{index:05d} violates the transport domain")
    return design, cohesive, parent, str(base["candidate_id"])


def execution_rows(source_id: str, design: str, cohesive: dict, parent: dict,
                   parent_id: str) -> list[dict]:
    source = material_row(source_id, design, cohesive, parent, parent_id)
    rows = []
    for shape, mode, code in ARMS:
        roots = [access_load(cohesive, shape, mode, temperature)[0]
                 for temperature in TEMPERATURES]
        exposure, mechanism = plastic_exposure(parent, roots[0])
        row = dict(source)
        row.update({
            "family": FAMILY,
            "additive_family": OPENING_ALTERNATIVES_FAMILY,
            "source_material_candidate_id": source_id,
            "candidate_id": f"{source_id}__{code}",
            "selected_shape": shape,
            "selected_mode": mode,
            "predicted_Kaccess_300K": roots[0],
            "mechanism_prediction": mechanism,
            "thermal_prediction": thermal_class(roots),
            "plastic_exposure_log10": exposure,
            "materialization": "EXACT_V2_SOBOL_RECONSTRUCTION",
        })
        row["complete_row_sha256"] = canonical_hash({
            "family": FAMILY,
            "additive_family": OPENING_ALTERNATIVES_FAMILY,
            "source_complete_row_sha256": source["complete_row_sha256"],
            "opening_mode": mode,
            "cohesive_shape": shape,
        })
        rows.append(row)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-registry", type=Path, required=True)
    parser.add_argument("--parent-registry", type=Path, required=True)
    parser.add_argument("--anchor-registry", type=Path, required=True)
    parser.add_argument("--candidate-ids", required=True)
    parser.add_argument("--log2-bank", type=int, default=15)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    sources = read_sources(args.original_registry, args.parent_registry)
    anchors = read_anchors(args.anchor_registry)
    points = qmc.Sobol(d=5, scramble=True, seed=SEED).random_base2(args.log2_bank)
    mutations = qmc.Sobol(d=11, scramble=True, seed=20261017).random_base2(
        args.log2_bank)
    rows = []
    for token in (part.strip() for part in args.candidate_ids.split(",")):
        if not token:
            continue
        index = int(token.removeprefix("R10F_V2_"))
        if not 0 <= index < len(points):
            raise ValueError(f"candidate outside the declared Sobol bank: {token}")
        source_id = f"R10F_V2_{index:05d}"
        design, cohesive, parent, parent_id = reconstruct(
            index, points, mutations, sources, anchors)
        rows.extend(execution_rows(source_id, design, cohesive, parent, parent_id))
    write_csv(args.out, rows)
    print(json.dumps({
        "physical_vectors": len(rows) // len(ARMS),
        "execution_identities": len(rows),
        "distinct_cohesive_vectors": len({row["cohesive_parameter_sha256"] for row in rows}),
        "distinct_plasticity_fingerprints": len({row["plasticity_physics_sha256"] for row in rows}),
        "out": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
