#!/usr/bin/env python3
"""Recover the historical v9.14 search and design the Paris-first batch.

The historical 1,024-row Sobol population is treated as immutable source
material.  The new search varies only its 22 candidate-surface coordinates;
the eight historical row-level CommonPhysics overrides are recorded but fixed
to the current calibrated v7 transport/three-hit configuration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V914 = Path(os.environ.get(
    "V914_ROOT",
    "/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v9_14_cyclic_fatigue_knee_search",
))
HISTORICAL_GENERATOR = V914 / "scripts/generate_v914_endurance_knee_manifold.py"
HISTORICAL_REGISTRY = V914 / "runtime_inputs/v914/endurance_knee_global_300K_1024.csv"
HISTORICAL_RANKING = V914 / "runs/v914_endurance_knee_global_300K_stageA_1024/ranked_candidates.csv"
HISTORICAL_SEARCH_MANIFEST = V914 / "runs/v914_endurance_knee_global_300K_stageA_1024/search_manifest.json"
HISTORICAL_RUN_CONTRACT = V914 / "runs/v914_endurance_knee_global_300K_stageA_1024/run_contract.json"
CURRENT_PHYSICS = ROOT / "artifacts/v914_calibrated_v7_transport_extension/calibrated_v7_transport_extension.json"
CURRENT_SELECTION = ROOT / "arrhenius_fracture/data/materials/v10_2_31_endurance_knee_ABCD_selection.json"
CURRENT_ABCD_REGISTRY = ROOT / "arrhenius_fracture/data/materials/v10_2_31_endurance_knee_ABCD_registry.csv"

ABCD = {
    "v914_endurance_knee_0462": ("A", "direct barrier"),
    "v914_endurance_knee_0658": ("B", "plastic-state controlled"),
    "v914_endurance_knee_0554": ("C", "timescale crossover"),
    "v914_endurance_knee_0133": ("D", "mixed"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    names = fields or list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def historical_bounds() -> tuple[dict[str, tuple[float, float, str]], list[str], list[str]]:
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(V914))
        namespace = runpy.run_path(str(HISTORICAL_GENERATOR))
    finally:
        sys.path[:] = old_path
    candidate = dict(namespace["GLOBAL_300K_BOUNDS"])
    physics = dict(namespace["PHYSICS_OVERRIDE_BOUNDS"])
    combined = {**candidate, **physics}
    return combined, list(candidate), list(physics)


def inverse_coordinate(value: float, low: float, high: float, transform: str) -> float:
    if transform == "linear":
        coordinate = (value - low) / (high - low)
    elif transform == "log10":
        coordinate = (math.log10(value) - math.log10(low)) / (math.log10(high) - math.log10(low))
    else:
        raise ValueError(f"unsupported historical transform {transform!r}")
    if not -2.0e-12 <= coordinate <= 1.0 + 2.0e-12:
        raise ValueError(f"historical value {value} maps outside [{low}, {high}]")
    return min(max(float(coordinate), 0.0), 1.0)


def maximin_indices(matrix: np.ndarray, ids: list[str], seeds: list[int], count: int) -> list[int]:
    selected = list(dict.fromkeys(int(index) for index in seeds))
    if len(selected) > count:
        raise ValueError("seed population exceeds requested design")
    distance = np.full(len(matrix), np.inf)
    for index in selected:
        distance = np.minimum(distance, np.sum((matrix - matrix[index]) ** 2, axis=1))
    distance[selected] = -np.inf
    while len(selected) < count:
        maximum = float(np.max(distance))
        ties = np.flatnonzero(np.isclose(distance, maximum, rtol=0.0, atol=1.0e-15))
        chosen = min((int(index) for index in ties), key=lambda index: ids[index])
        selected.append(chosen)
        distance = np.minimum(distance, np.sum((matrix - matrix[chosen]) ** 2, axis=1))
        distance[selected] = -np.inf
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/v914_paris_reparameterization")
    parser.add_argument("--initial-count", type=int, default=32)
    parser.add_argument("--fidelity-pair-count", type=int, default=16)
    args = parser.parse_args()
    if not 24 <= args.initial_count <= 40:
        raise SystemExit("initial-count must lie in the requested 24-40 range")
    if not 12 <= args.fidelity_pair_count <= args.initial_count:
        raise SystemExit("fidelity-pair-count must include A-D plus 8-12 diverse candidates")
    args.out.mkdir(parents=True, exist_ok=True)

    bounds, candidate_fields, physics_fields = historical_bounds()
    source = read_csv(HISTORICAL_REGISTRY)
    if len(source) != 1024 or len({row["candidate_id"] for row in source}) != 1024:
        raise RuntimeError("historical candidate population is not the audited 1,024 unique rows")
    ranking = {row["candidate_id"]: row for row in read_csv(HISTORICAL_RANKING)}
    ids = [row["candidate_id"] for row in source]
    index_by_id = {candidate_id: index for index, candidate_id in enumerate(ids)}
    seed_indices = [index_by_id[candidate_id] for candidate_id in ABCD]

    coordinates = np.asarray([
        [inverse_coordinate(float(row[field]), *bounds[field]) for field in candidate_fields]
        for row in source
    ])
    selected = maximin_indices(coordinates, ids, seed_indices, args.initial_count)
    fidelity_ids = {ids[index] for index in selected[:args.fidelity_pair_count]}

    abc_rows: list[dict[str, Any]] = []
    for candidate_id, (letter, role) in ABCD.items():
        index = index_by_id[candidate_id]
        row: dict[str, Any] = {
            "ABCD_option": letter,
            "mechanism_probe_role": role,
            "historical_search_rank": ranking[candidate_id]["search_rank"],
            **source[index],
        }
        row.update({f"u__{field}": coordinates[index, j] for j, field in enumerate(candidate_fields)})
        abc_rows.append(row)
    write_csv(args.out / "historical_ABCD_candidates.csv", abc_rows)

    bounds_payload = {
        "schema": "v914_historical_search_bounds_v1",
        "historical_full_dimension": len(bounds),
        "historical_bounds_in_generator_order": [
            {"field": field, "lower": bounds[field][0], "upper": bounds[field][1], "transform": bounds[field][2]}
            for field in bounds
        ],
        "new_paris_search_candidate_dimension": len(candidate_fields),
        "new_paris_search_varied_fields": candidate_fields,
        "fixed_current_common_physics_fields": physics_fields,
        "fixed_physics_reason": (
            "The corrected mission fixes calibrated transferred-2D common physics, calibrated transport, "
            "and three-hit cleavage; historical row-level CommonPhysics overrides remain provenance only."
        ),
    }
    (args.out / "historical_search_bounds.json").write_text(
        json.dumps(bounds_payload, indent=2, sort_keys=True) + "\n"
    )

    pool_rows: list[dict[str, Any]] = []
    for index, source_row in enumerate(source):
        candidate_id = ids[index]
        row = {key: value for key, value in source_row.items() if key not in physics_fields}
        row.update({
            "source_candidate_id": candidate_id,
            "source_historical_sobol_index": int(candidate_id.rsplit("_", 1)[1]),
            "paris_initial_design_order": "",
            "paris_design_role": "UNEVALUATED_HISTORICAL_POOL",
            "historical_monotonic_K50_300K_MPa_sqrt_m": ranking[candidate_id]["y__K50_1d_T300K"],
            "current_common_physics_mode": "FIXED_CALIBRATED_V7_TRANSPORT_EXTENSION",
        })
        row.update({f"u__{field}": coordinates[index, j] for j, field in enumerate(candidate_fields)})
        pool_rows.append(row)
    write_csv(args.out / "candidate_pool.csv", pool_rows)

    registry_rows: list[dict[str, Any]] = []
    for design_order, index in enumerate(selected):
        source_row = source[index]
        candidate_id = ids[index]
        row = dict(pool_rows[index])
        row.update({
            "paris_initial_design_order": design_order,
            "paris_design_role": (
                f"ABCD_{ABCD[candidate_id][0]}" if candidate_id in ABCD
                else "DIVERSE_FIDELITY_PAIR" if candidate_id in fidelity_ids
                else "SPACE_FILLING_INITIAL"
            ),
        })
        registry_rows.append(row)
    write_csv(args.out / "candidate_registry.csv", registry_rows)

    provenance = {
        "schema": "v914_historical_fatigue_search_provenance_v1",
        "historical_population": 1024,
        "historical_generator": {
            "method": "scipy.stats.qmc.Sobol scrambled power-of-two population",
            "seed": 1914,
            "candidate_id_format": "v914_endurance_knee_{sobol_index:04d}",
            "source": str(HISTORICAL_GENERATOR.resolve()),
            "sha256": digest(HISTORICAL_GENERATOR),
        },
        "historical_candidate_registry": {"path": str(HISTORICAL_REGISTRY.resolve()), "sha256": digest(HISTORICAL_REGISTRY)},
        "historical_ranking": {"path": str(HISTORICAL_RANKING.resolve()), "sha256": digest(HISTORICAL_RANKING)},
        "historical_search_manifest": {"path": str(HISTORICAL_SEARCH_MANIFEST.resolve()), "sha256": digest(HISTORICAL_SEARCH_MANIFEST)},
        "historical_run_contract": {"path": str(HISTORICAL_RUN_CONTRACT.resolve()), "sha256": digest(HISTORICAL_RUN_CONTRACT)},
        "historical_ABCD_selection": {
            "selection_was_not_top_four_search_rank": True,
            "purpose": "four hand-designated mechanistic probes for sparse independent 2-D validation",
            "roles": {candidate_id: {"option": option, "role": role} for candidate_id, (option, role) in ABCD.items()},
            "selection_record": str(CURRENT_SELECTION.resolve()),
            "selection_record_sha256": digest(CURRENT_SELECTION),
            "registry_sha256": digest(CURRENT_ABCD_REGISTRY),
        },
        "experimental_reference_recovery": {
            "status": "NO_DEFENSIBLE_QUANTITATIVE_LOCAL_ENVELOPE_IN_PRIOR_JOINT_PARIS_STUDY",
            "evidence": "scripts/finalize_v913_joint_paris_slope_study.py and its verified final manifest",
            "action": "use external replaceable target configuration; never use historical weak-T m as target",
        },
        "new_initial_design": {
            "count": args.initial_count,
            "selection": "greedy maximin in exact inverse-transformed 22-D historical Sobol coordinates, seeded by A-D",
            "fidelity_pair_count": args.fidelity_pair_count,
            "common_random_seed": 1720,
        },
        "fixed_forward_physics": {"path": str(CURRENT_PHYSICS.resolve()), "sha256": digest(CURRENT_PHYSICS)},
        "physics_changed": False,
    }
    (args.out / "historical_fatigue_search_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"V914_PARIS_DESIGN_COMPLETE population={len(source)} initial={len(registry_rows)} "
        f"fidelity_pairs={len(fidelity_ids)} dimensions={len(candidate_fields)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
