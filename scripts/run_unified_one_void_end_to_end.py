#!/usr/bin/env python3
"""Execute the central DBTT one-void path through child continuation."""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.cavity_free_boundary_verification import verify_cavity_free_boundary
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle, require_bound_identity
from arrhenius_fracture.voiding_production_v5 import (
    cavity_source_resolution_metrics,
    deterministic_trajectory,
    directional_clock_rates,
    downstream_front_transaction,
    equilibrate_fixed_load_with_production_fem,
    fresh_sharp_front_engine,
    ligament_transaction,
    refine_downstream_source,
    restore_sharp_front_engine,
    sharp_front_engine_fingerprint,
)
from arrhenius_fracture.voiding_v5 import VoidPhase


RETAINED_CRACK_PATH_M = (
    (0.0, 8.572527594031472e-20),
    (0.0005725993004046688, 1.5602085896855586e-20),
)
OPERATIONS = (
    "CONNECTED_VOID",
    "DOWNSTREAM_SOURCE_FIRST_PASSAGE",
    "DOWNSTREAM_CHILD_CREATED",
    "DOWNSTREAM_CHILD_CONTINUED",
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _consumed_event_ids(state):
    return {
        event_id for event_id, row in state.junction_process_state.get(
            "directional_event_provenance", {}
        ).items() if row.get("status") == "CONSUMED_AT_OWNED_SOURCE"
    }


def _checkpoint_round_trip(state, path):
    before = complete_accepted_state_fingerprint(state)
    write_checkpoint(state, path)
    restored = restore_checkpoint(path)
    _require(complete_accepted_state_fingerprint(restored) == before,
             "checkpoint round trip changed the accepted state")
    return restored


def _rollback_matrix(connected, directory):
    stages = (
        "source_observation", "source_clock_update", "downstream_child_activation",
        "downstream_child_support_rebuild", "equilibrium", "energy_gate",
        "process_state_update", "late_event_veto",
    )
    identity = complete_accepted_state_fingerprint(connected)
    for stage in stages:
        operations = []
        try:
            downstream_front_transaction(
                connected, failure_stage=stage, operation_log=operations,
            )
        except RuntimeError as exc:
            _require(str(exc) == "injected:" + stage,
                     f"rollback stage {stage} failed for the wrong reason: {exc}")
        else:
            raise RuntimeError(f"rollback injection {stage} was not reached")
        _require(stage in operations, f"rollback injection {stage} was not observed")
        _require(complete_accepted_state_fingerprint(connected) == identity,
                 f"rollback injection {stage} changed the accepted input")

    checkpoint = directory / "checkpoint-publication.json"
    write_checkpoint(connected, checkpoint)
    try:
        write_checkpoint(
            replace(connected, checkpoint_generation=connected.checkpoint_generation + 1),
            checkpoint,
            failure_injector=lambda stage, _trial: (_ for _ in ()).throw(
                RuntimeError("injected:checkpoint_publication")
            ) if stage == "checkpoint_write" else None,
        )
    except RuntimeError as exc:
        _require(str(exc) == "injected:checkpoint_publication",
                 "checkpoint publication failed for the wrong reason")
    else:
        raise RuntimeError("checkpoint publication injection was not reached")
    restored = restore_checkpoint(checkpoint)
    _require(complete_accepted_state_fingerprint(restored) == identity,
             "failed checkpoint publication replaced the accepted checkpoint")
    return len(stages) + 1


def main():
    bundle = material_bundle("DBTT")
    print("[run] building fixed 128x48 pre-ligament trajectory", flush=True)
    state, _ = deterministic_trajectory(
        bundle=bundle,
        stop_before_ligament=True,
        crack_path_m=RETAINED_CRACK_PATH_M,
        boundary_segments=128,
        radial_layers=48,
        shape_regular_local_level="B",
    )
    print("[run] fixed pre-ligament trajectory accepted", flush=True)
    state = equilibrate_fixed_load_with_production_fem(
        replace(state, displacement=2.0 * state.displacement)
    )
    print("[run] doubled-load equilibrium accepted", flush=True)
    connected, ligament = ligament_transaction(state)
    _require(ligament.accepted, "central DBTT ligament event was rejected")
    _require(connected.void_state.cavities[0].phase == VoidPhase.CONNECTED_VOID,
             "ligament event did not produce CONNECTED_VOID")
    _require(not connected.crack_network.active_tip_ids,
             "connected cavity retained an active upstream front")
    print("[run] connected DBTT cavity accepted", flush=True)

    connected, source_audit = refine_downstream_source(
        connected,
        max_refinement_levels=1,
        refinement_region="complete_cavity_ring",
        quality_improvement="constrained_v1",
    )
    _require(source_audit["status"] == "SOURCE_TENSOR_QUALIFIED",
             "downstream source did not qualify: " + repr(source_audit.get("attempts")))
    print("[run] downstream source qualified", flush=True)
    weak = verify_cavity_free_boundary(connected)
    _require(weak.passed, "connected cavity failed the weak free-boundary contract")
    metrics = cavity_source_resolution_metrics(connected)
    tensor = np.asarray(metrics["tensor_Pa"], dtype=float)
    normal = np.asarray(metrics["normal_xy"], dtype=float)
    tangent = np.asarray(metrics["tangent_xy"], dtype=float)
    sigma_tt = float(metrics["sigma_tt_Pa"])
    np.testing.assert_allclose(tensor, sigma_tt * np.outer(tangent, tangent), rtol=0.0, atol=1e-8)
    _require(abs(float(normal @ tensor @ normal)) <= 1e-8
             and abs(float(normal @ tensor @ tangent)) <= 1e-8,
             "source tensor has nonzero local normal or shear traction")
    rates = directional_clock_rates(connected, tensor)
    _require(len(rates) == 1 and sum(row["effective_rate_s"] > 0 for row in rates) == 1,
             "connected source does not have exactly one active clock owner")
    _require(math.isfinite(rates[0]["crossing_time_s"]),
             "connected source has no finite first passage")

    identity = dict(require_bound_identity(connected))
    radius = connected.void_state.cavities[0].radius_m
    root_engine = fresh_sharp_front_engine(connected.material, bundle)
    _require(root_engine.r_eff() != radius, "r_tip was aliased to R_void")
    before_actions = int(connected.event_counters.get("topology_actions", 0))
    before_branches = len(connected.crack_network.branches)
    before_history = len(connected.void_state.event_history)

    with tempfile.TemporaryDirectory(prefix="unified-one-void-") as temporary:
        directory = Path(temporary)
        _checkpoint_round_trip(connected, directory / "connected-before-source.json")
        _checkpoint_round_trip(connected, directory / "immediately-before-child-birth.json")

        child, birth, birth_operations, birth_causal = downstream_front_transaction(connected)
        _require(birth is not None and birth.accepted, "downstream child birth was not accepted")
        _require(len(child.crack_network.branches) == before_branches + 1,
                 "child birth did not create exactly one front")
        _require(child.crack_network.active_tip_ids == ("void-front-1",),
                 "child birth did not leave one ordinary active downstream front")
        _require(int(child.event_counters.get("topology_actions", 0)) == before_actions + 1,
                 "child birth did not consume exactly one topology transaction")
        _require(sum(len(row["emitted_event_ids"]) for row in birth_causal["cleavage"]) == 1,
                 "child birth did not consume exactly one threshold")
        _require(len(child.void_state.event_history) == before_history + 1,
                 "child birth did not append exactly one physical event")
        _require(dict(require_bound_identity(child)) == identity,
                 "child birth changed material/core identity")
        owner = child.tip_process_state["owner_by_front"]["void-front-1"]
        restored_engine = restore_sharp_front_engine(
            child.material, bundle, child.tip_process_state["by_branch"][owner],
        )
        _require(sharp_front_engine_fingerprint(restored_engine)
                 == sharp_front_engine_fingerprint(fresh_sharp_front_engine(child.material, bundle)),
                 "child FrontEngine was not initialized by the canonical factory")
        _require(restored_engine.r_eff() != radius, "child r_tip was aliased to R_void")
        print("[run] downstream child created", flush=True)
        _checkpoint_round_trip(child, directory / "immediately-after-child-birth.json")
        _checkpoint_round_trip(child, directory / "immediately-before-child-continuation.json")

        child_path_size = len(child.crack_network.branch("void-front-1").path)
        child_actions = int(child.event_counters.get("topology_actions", 0))
        terminal, continuation, continuation_operations, continuation_causal = (
            downstream_front_transaction(child, continuation=True)
        )
        _require(continuation is not None and continuation.accepted,
                 "ordinary child continuation was not accepted")
        _require(len(terminal.crack_network.branch("void-front-1").path) == child_path_size + 1,
                 "continuation did not extend the downstream child once")
        _require(int(terminal.event_counters.get("topology_actions", 0)) == child_actions + 1,
                 "continuation did not consume exactly one topology transaction")
        _require(sum(len(row["emitted_event_ids"])
                     for row in continuation_causal["cleavage"]) == 1,
                 "continuation did not consume exactly one threshold")
        _require(terminal.crack_network.active_tip_ids == ("void-front-1",),
                 "continuation did not retain one ordinary active downstream front")
        _require(dict(require_bound_identity(terminal)) == identity,
                 "continuation changed material/core identity")
        print("[run] downstream child continued", flush=True)
        for ledger in (
            "ordinary_crack_fractured_length_m", "traversed_void_free_span_m",
            "active_front_coordinate_advance_m",
        ):
            _require(ledger in terminal.void_state.length_ledgers,
                     "missing separate length ledger: " + ledger)
        _checkpoint_round_trip(terminal, directory / "immediately-after-child-continuation.json")
        rollback_count = _rollback_matrix(connected, directory)
        print("[run] restart and rollback checks complete", flush=True)

    print("weak_boundary_verification = PASS")
    print("source_tensor_qualification = PASS")
    print("raw_cst_boundary_traction_diagnostic =",
          metrics["RAW_CST_BOUNDARY_TRACTION_DIAGNOSTIC"]["normalized_traction"])
    print("piecewise_equilibrated_traction_diagnostic = RETAINED_NON_AUTHORITATIVE_DIAGNOSTIC")
    print("checkpoint_restart_stages = 5 PASS")
    print(f"rollback_injections = {rollback_count} PASS")
    for operation in OPERATIONS:
        print(operation)
    print("UNIFIED_2D_ONE_VOID_END_TO_END = PASS")


if __name__ == "__main__":
    main()
