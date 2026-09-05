from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from arrhenius_fracture.branch_checkpoint_v11 import ProductionBranchCheckpoint
from arrhenius_fracture.branch_cluster_v11 import create_unresolved_branch_cluster
from arrhenius_fracture.crack_network_v11 import CrackNetworkState
from arrhenius_fracture.directional_competition_v11 import (
    CleavageCandidate, DirectionalCompetitionState, competition_state_to_dict,
)
from arrhenius_fracture.general_multifront_v12 import (
    FrontCandidateObservation, ResourcePolicy,
)
from arrhenius_fracture.multifront_checkpoint_v12 import (
    checkpoint_bytes, load_v11_checkpoint_as_v12,
    restore_multifront_checkpoint, write_multifront_checkpoint,
)
from arrhenius_fracture.multifront_output_v12 import (
    FRONT_FIELDS, OWNER_FIELDS, canonical_csv_bytes, front_observation_row, owner_row,
)
from arrhenius_fracture.topology_transaction_v11 import LiveFEMTopologyState


def candidates():
    return (
        CleavageCandidate.create(
            plane_family="100", plane_variant="x", direction_xy=(1.0, 0.0),
            normal_xy=(0.0, 1.0), gamma_rel=1.0,
            orientation_convention="test",
        ),
        CleavageCandidate.create(
            plane_family="100", plane_variant="y", direction_xy=(0.0, 1.0),
            normal_xy=(-1.0, 0.0), gamma_rel=1.0,
            orientation_convention="test",
        ),
    )


def v11_checkpoint(two_front=False):
    inventory = candidates()
    competition = DirectionalCompetitionState.initialize(inventory, global_hazard_seed=91)
    network = CrackNetworkState.one_tip(((0.0, 0.0),), initial_orientation_rad=0.0)
    cluster = None
    if two_front:
        network, cluster = create_unresolved_branch_cluster(
            network, parent_branch_id=network.active_tip_ids[0],
            candidate_ids=(inventory[0].candidate_id, inventory[1].candidate_id),
            event_index=3,
            shared_process_state={"ownership": "shared"},
            conserved_ledgers={"mobile": 2.0, "retained": 1.0},
        )
        competitions = {
            front_id: DirectionalCompetitionState.initialize(
                inventory, global_hazard_seed=91 + index,
            )
            for index, front_id in enumerate(network.active_tip_ids)
        }
    else:
        competitions = {network.active_tip_ids[0]: competition}
    state = LiveFEMTopologyState(
        mesh=SimpleNamespace(), boundary=SimpleNamespace(),
        damage=np.zeros(1), displacement=np.zeros(2),
        ep_gp=np.zeros(1), rho_gp=np.zeros(1), elasticity_D=np.eye(2),
        material=SimpleNamespace(name="test"), cohesive_network=SimpleNamespace(),
        crack_network=network, competition=next(iter(competitions.values())),
        tip_process_state={}, junction_process_state={"accepted_stress_state_id": "stress:v11"},
        energy_ledgers={}, rng_state={"seed": 91},
        event_counters={
            "accepted_steps": 4, "shared_state_updates": 4,
            "topology_actions": int(two_front),
        },
        stored_energy_J_per_m=1.0,
    )
    return ProductionBranchCheckpoint(
        state=state,
        shared_process_state={"mpz_fields": {"advance_total_m": 5.0e-6}},
        physical_time_s=10.0, accepted_load=2.0e-6,
        mesh_identity="mesh", boundary_condition_state={"opening": 2.0e-6},
        provider_runtime=None, provider_cache_identity="cache",
        topology_fingerprint="accepted:v11", front_competitions=competitions,
        branch_clusters=(() if cluster is None else (cluster,)),
        projected_extension_m=0.0, physical_extension_m=0.0,
        handoff_guard_diagnostics={},
    )


def observation(runtime, front_id, candidate_id):
    owner = runtime.owner_by_front[front_id]
    return FrontCandidateObservation(
        accepted_state_id=runtime.accepted_state_id,
        stress_field_state_id=runtime.stress_field_state_id,
        front_id=front_id, owner_id=owner, candidate_id=candidate_id,
        tip_coordinates_m=runtime.crack_network.branch(front_id).tip,
        signed_local_J_J_per_m2=1.0, marginal_G_J_per_m2=1.0,
        kinetic_J_used_J_per_m2=1.0, directional_K_MPa_sqrt_m=1.0,
        directional_rate_per_s=1.0, tensor=(1.0, 0.0, 0.0),
        tensor_reliability="qualified", controlling_scalar_K_tip_id=front_id,
        tensor_probe_tip_id=front_id,
    )


def test_v11_one_front_compatibility_is_physical_and_stochastic_identity():
    source = v11_checkpoint(False)
    runtime = load_v11_checkpoint_as_v12(
        source, resource_policy=ResourcePolicy("disabled", 1, None),
    )
    assert runtime.crack_network.to_json() == source.state.crack_network.to_json()
    front = runtime.active_front_ids[0]
    assert runtime.front_runtimes[front].competition_state == competition_state_to_dict(
        source.front_competitions[front]
    )
    assert runtime.owner_by_front[front] in runtime.process_regions
    assert runtime.resource_policy.branching_mode == "disabled"


def test_v11_two_front_cluster_maps_to_one_generic_region_without_state_split():
    source = v11_checkpoint(True)
    runtime = load_v11_checkpoint_as_v12(
        source, resource_policy=ResourcePolicy("mechanistic", 2, None),
    )
    assert runtime.crack_network.to_json() == source.state.crack_network.to_json()
    assert len(runtime.active_front_ids) == 2
    assert len(runtime.process_regions) == 1
    assert len(runtime.process_engines) == 1
    assert len(runtime.junctions) == 1
    assert set(runtime.owner_by_front.values()) == set(runtime.process_regions)
    assert runtime.cumulative_branch_births == 1
    assert next(iter(runtime.process_regions.values())).cumulative_process_advance_m == 5.0e-6


def test_v12_checkpoint_bytes_ignore_mapping_insertion_order(tmp_path):
    runtime = load_v11_checkpoint_as_v12(
        v11_checkpoint(True), resource_policy=ResourcePolicy("mechanistic", 2, None),
    )
    reordered = replace(
        runtime,
        front_runtimes=dict(reversed(tuple(runtime.front_runtimes.items()))),
        owner_by_front=dict(reversed(tuple(runtime.owner_by_front.items()))),
        process_regions=dict(reversed(tuple(runtime.process_regions.items()))),
    )
    reference = {"kind": "immutable_v11", "sha256": "a" * 64}
    assert checkpoint_bytes(runtime, accepted_fem_state_reference=reference) == checkpoint_bytes(
        reordered, accepted_fem_state_reference=dict(reversed(tuple(reference.items())))
    )
    path = tmp_path / "checkpoint.json"
    write_multifront_checkpoint(runtime, path, accepted_fem_state_reference=reference)
    restored = restore_multifront_checkpoint(path)
    assert restored.to_dict() == runtime.to_dict()


def test_front_and_owner_outputs_carry_required_fields_and_are_order_invariant():
    runtime = load_v11_checkpoint_as_v12(
        v11_checkpoint(True), resource_policy=ResourcePolicy("mechanistic", 2, None),
    )
    front_rows = []
    owner_controls = {}
    for front_id, front_runtime in runtime.front_runtimes.items():
        for candidate_id in front_runtime.candidate_ids:
            item = observation(runtime, front_id, candidate_id)
            front_rows.append(front_observation_row(
                runtime, item, selected_event_tip_id=runtime.active_front_ids[0],
            ))
            owner_controls[runtime.owner_by_front[front_id]] = front_id
    owner_rows = [
        owner_row(runtime, owner_id, controlling_front_id=front_id)
        for owner_id, front_id in owner_controls.items()
    ]
    assert set(front_rows[0]) == set(FRONT_FIELDS)
    assert set(owner_rows[0]) == set(OWNER_FIELDS)
    assert canonical_csv_bytes(FRONT_FIELDS, front_rows) == canonical_csv_bytes(
        FRONT_FIELDS, reversed(front_rows)
    )
    assert canonical_csv_bytes(OWNER_FIELDS, owner_rows) == canonical_csv_bytes(
        OWNER_FIELDS, reversed(owner_rows)
    )
