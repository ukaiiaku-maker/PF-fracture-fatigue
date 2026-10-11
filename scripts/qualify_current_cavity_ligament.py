#!/usr/bin/env python3
"""Qualify the retained cavity state's candidate-specific ligament operator."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID  # noqa: E402
from arrhenius_fracture.shared_v5_cavity_adapter import (  # noqa: E402
    _directional_provider, _finite, _restore_pair,
)
from arrhenius_fracture.topology_transaction_v11 import (  # noqa: E402
    complete_accepted_state_fingerprint,
)
from arrhenius_fracture.voiding_production_v5 import (  # noqa: E402
    _first_ray_cavity_intersection,
)

PAIR = ROOT / "review_packet/current_science_pilots_6abbf10/v11_migration_pair.json"
RAW_OUT = ROOT / "integration_evidence/current_science_pilots_6abbf10/cavity_ligament"
REVIEW_OUT = ROOT / "review_packet/current_science_pilots_6abbf10"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    states, migration = _restore_pair(PAIR)
    state = states["enabled"]
    before = complete_accepted_state_fingerprint(state)
    tensor, effective_k, audit, rows = _directional_provider(1200.0)(state)
    after = complete_accepted_state_fingerprint(state)
    selected = [
        row for row in rows
        if row.get("drive_source") == "MATCHED_COMPLETE_LIGAMENT_TOPOLOGY_ENERGY"
    ]
    if len(selected) != 1:
        raise RuntimeError("expected one candidate-specific ligament dispatch")
    row = selected[0]
    observation = row["ligament_energy_observation"]
    if observation["status"] != "AVAILABLE":
        raise RuntimeError("complete-ligament observation did not qualify")
    rate = next(
        item for item in audit["fracture_engine_rows"]
        if item["candidate_id"] == row["candidate_id"]
    )
    start = state.crack_network.branch(ROOT_BRANCH_ID).tip
    candidate = next(
        item for item in state.competition.candidates
        if item.candidate_id == row["candidate_id"]
    )
    entry = _first_ray_cavity_intersection(state, start, candidate.direction_xy)
    cavity = state.void_state.cavities[0]
    RAW_OUT.mkdir(parents=True, exist_ok=True)
    full = _finite({
        "schema": "retained-current-cavity-ligament-qualification/1",
        "migration_manifest": str(PAIR.relative_to(ROOT)),
        "migration": migration.get("migration"),
        "accepted_state_fingerprint_before": before,
        "accepted_state_fingerprint_after": after,
        "accepted_state_preserved": before == after,
        "physical_time_advanced": False,
        "transport_solve_performed": False,
        "crack_tip_m": start,
        "cavity_center_m": cavity.center_m,
        "cavity_equivalent_radius_m": cavity.radius_m,
        "actual_polygon_intersection_m": entry,
        "center_separation_m": math.dist(start, cavity.center_m),
        "solid_ligament_m": math.dist(start, entry),
        "adjacent_element_tensor_Pa": tensor,
        "state_bound_effective_K_Pa_sqrt_m": effective_k,
        "candidate_directional_row": row,
        "candidate_rate_row": rate,
    })
    raw_path = RAW_OUT / "candidate_specific_complete_ligament_observation.json"
    raw_path.write_text(json.dumps(full, indent=2, sort_keys=True) + "\n")
    level_rows = [{
        "refinement_level": level["refinement_level"],
        "mesh_nodes": level["mesh_nodes"],
        "mesh_elements": level["mesh_elements"],
        "Pi_unextended_J_per_m": level["Pi_unextended_J_per_m"],
        "Pi_connected_J_per_m": level["Pi_connected_J_per_m"],
        "G_ligament_J_per_m2": level["G_ligament_J_per_m2"],
        "connection_certificate_passed": level["connection_certificate"]["passed"],
        "crack_graph_outside_actual_cavity_polygon": level[
            "connection_certificate"
        ]["crack_graph_outside_cavity"],
        "exact_no_intact_node_or_element_path": level[
            "connection_certificate"
        ]["exact_no_intact_node_or_element_path"],
    } for level in observation["levels"]]
    compact = {
        "schema": "retained-current-cavity-ligament-summary/1",
        "status": "QUALIFIED_COMPLETE_LIGAMENT_OPERATION",
        "candidate_id": row["candidate_id"],
        "drive_source": row["drive_source"],
        "local_J_substituted": row["local_J_substituted"],
        "center_separation_um": full["center_separation_m"] * 1.0e6,
        "solid_ligament_um": full["solid_ligament_m"] * 1.0e6,
        "G_ligament_J_per_m2": observation["G_ligament_J_per_m2"],
        "mesh_relative_error": observation["mesh_relative_error"],
        "mesh_relative_limit": observation["mesh_relative_limit"],
        "levels": level_rows,
        "directional_drive_status": row["directional_drive_status"],
        "K_energy_equivalent_MPa_sqrt_m": row[
            "K_energy_equivalent_Pa_sqrt_m"
        ] / 1.0e6,
        "candidate_crossing_time_s": rate["crossing_time_s"],
        "candidate_effective_rate_s": rate["effective_rate_s"],
        "accepted_state_preserved": before == after,
        "physical_time_advanced": False,
        "transport_solve_performed": False,
        "raw_evidence": str(raw_path.relative_to(ROOT)),
        "raw_evidence_sha256": sha256(raw_path),
        "interpretation": (
            "finite complete-ligament topology energy under the original candidate "
            "hazard route; distinct from and not substituted for local J"
        ),
    }
    (REVIEW_OUT / "CAVITY_LIGAMENT_QUALIFICATION_v4.json").write_text(
        json.dumps(compact, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
