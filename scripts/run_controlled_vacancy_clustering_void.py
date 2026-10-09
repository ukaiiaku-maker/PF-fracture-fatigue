#!/usr/bin/env python3
"""Controlled functional sentinel for one complete vacancy-created void path."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.legacy_front_process_adapter import LegacyFrontProcessAdapter
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle, require_bound_identity
from arrhenius_fracture.void_formation_mechanism import active_void_formation_mechanism
from arrhenius_fracture.voiding_production_v5 import (
    _bundle_from_state, _canonical_front_process_owner,
    deterministic_trajectory, downstream_front_transaction,
    equilibrate_fixed_load_with_production_fem, ligament_transaction,
    refine_downstream_source, sharp_front_load_provider,
)
from arrhenius_fracture.voiding_v5 import VoidPhase


RETAINED_CRACK_PATH_M = (
    (0.0, 8.572527594031472e-20),
    (0.0005725993004046688, 1.5602085896855586e-20),
)


def _require(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    bundle = material_bundle("DBTT")
    trace = []
    state, rows = deterministic_trajectory(
        bundle=bundle, vacancy_coupled=True, stop_before_ligament=True,
        crack_path_m=RETAINED_CRACK_PATH_M,
        boundary_segments=128, radial_layers=48,
        shape_regular_local_level="B", state_trace=trace,
    )
    labels = {label for label, _ in trace}
    required = {
        "available_site", "multi_hit_2", "stabilization", "subgrid_void",
        "geometric_promotion",
    }
    _require(required <= labels, "controlled path omitted stages: "+repr(required-labels))
    # Controlled mechanism sentinel: raise only the prescribed displacement,
    # leaving every immutable material/kinetic row untouched, so the existing
    # whole-topology energy gate can accept the ligament.
    state = equilibrate_fixed_load_with_production_fem(
        replace(state, displacement=2.0*state.displacement)
    )
    state, ligament = ligament_transaction(state)
    _require(ligament.accepted, "controlled ligament failed the existing energy gate")
    state, source = refine_downstream_source(
        state, max_refinement_levels=1,
        refinement_region="complete_cavity_ring",
        quality_improvement="constrained_v1",
    )
    _require(source["status"] == "SOURCE_TENSOR_QUALIFIED",
             "controlled downstream source did not qualify: "+repr(source))
    state, birth, _, _ = downstream_front_transaction(state)
    _require(birth is not None and birth.accepted, "controlled child birth was rejected")
    state, continuation, _, _ = downstream_front_transaction(state, continuation=True)
    _require(continuation is not None and continuation.accepted,
             "controlled child continuation was rejected")
    _require(state.void_state.cavities[0].phase == VoidPhase.DOWNSTREAM_FRONT_ACTIVE,
             "controlled path did not activate the downstream child")
    child_id = "void-front-1"
    _require(state.crack_network.active_tip_ids == (child_id,),
             "downstream child is not the sole active front")
    owner = _canonical_front_process_owner(state, child_id)
    adapter = LegacyFrontProcessAdapter(state.material, _bundle_from_state(state))
    _require(adapter.factory_identity["factory"] == "canonical_front_engine",
             "child does not use the canonical legacy factory")

    # Continue the same accepted child owner through five additional physical
    # process intervals.  These are deliberately below cleavage first passage:
    # they exercise emission/MPZ evolution without manufacturing topology.
    intervals = []
    for index in range(5):
        loads = sharp_front_load_provider(state, branch_id=child_id)["directional"]
        K = max(float(row["K_directional_Pa_sqrt_m"]) for row in loads)
        payload = state.tip_process_state["by_branch"][owner]
        progress = max((float(item.residual_action)
                        for item in state.competition.hazard_states), default=0.0)
        advanced = adapter.advance_front_process(
            payload, K, 900.0, 1.0e-10,
            topology_owned_cleavage_action_before=progress,
            topology_owned_cleavage_action_after=progress,
        )
        tip = dict(state.tip_process_state)
        by_branch = dict(tip["by_branch"])
        by_branch[owner] = advanced.advanced_front_state
        tip["by_branch"] = by_branch
        state = replace(state, tip_process_state=tip)
        intervals.append({
            "index": index+1,
            "emitted_population": advanced.updated_emission_state["N_em"],
            "r_tip_m": advanced.updated_r_tip_m,
            "cleavage_event": advanced.cleavage_event_occurred,
        })
    _require(len(intervals) == 5 and not any(row["cleavage_event"] for row in intervals),
             "bounded continuation did not preserve the child process owner")
    mechanism = active_void_formation_mechanism()
    _require(mechanism.mechanism_id == "VACANCY_CLUSTERING_V1",
             "controlled path selected the wrong void mechanism")
    _require(dict(require_bound_identity(state))["material_bundle_sha256"] == bundle.bundle_hash,
             "controlled episode changed the immutable fracture bundle")

    print("LEGACY_FRONT_ACTIVE")
    print("VACANCY_FIELD_ACTIVE")
    print("VACANCY_CLUSTERING_EMBRYO")
    print("STABLE_SUBGRID_VOID")
    print("EXPLICIT_CAVITY_PROMOTED")
    print("CRACK_VOID_LIGAMENT_CONNECTED")
    print("DOWNSTREAM_CHILD_CREATED")
    print("DOWNSTREAM_CHILD_USES_LEGACY_FRONT_ENGINE")
    print("DOWNSTREAM_CHILD_CONTINUES")
    print("accepted_post_void_front_process_intervals =", len(intervals))
    print("controlled_reservoir = TEST_ONLY_EXTREME_RESERVOIR")
    print("CONTROLLED_VACANCY_CLUSTERING_VOID = PASS")


if __name__ == "__main__":
    main()
