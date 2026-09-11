from dataclasses import replace
import math
import os
from pathlib import Path
import pickle

import pytest

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.crack_network_v11 import CrackBranchState
from arrhenius_fracture.directional_competition_v11 import CleavageCandidate
from arrhenius_fracture.hybrid_directional_drive_v5 import (
    K_INTERPRETATION,
    accepted_state_identity,
    hybrid_directional_drive_provider,
)
from arrhenius_fracture.live_topology_kernel_v11 import evaluate_exact_topology
from arrhenius_fracture.topology_transaction_v11 import LiveFEMTopologyState
from arrhenius_fracture.voiding_production_v5 import (
    _conform_bulk_point,
    _refresh_downstream_boundary_context,
    capture_sharp_front_engine,
    directional_sharp_front_rates,
    downstream_front_transaction,
    fresh_sharp_front_engine,
)
from tests.test_live_topology_kernel_v11 import (
    legacy_from_live,
    live_branched_request,
    live_straight_request,
)


SOURCE_SHA = "ea4649c3e676edbf0259feb305aedcdf59400aa8"
CHILD_ID = "void-front-1"
CHECKPOINT_ROOT = Path(os.environ.get(
    "V5_HYBRID_CHECKPOINT_ROOT", Path(__file__).resolve().parents[1]
))


def _checkpoint(name):
    return restore_checkpoint(
        CHECKPOINT_ROOT / "artifacts/voiding_v5_finalization_v2/checkpoints" / name
    )


@pytest.fixture(scope="module")
def far_child_state():
    state = _checkpoint("downstream_front_before_continuation.json")
    payload = capture_sharp_front_engine(fresh_sharp_front_engine(state.material))
    tips = dict(state.tip_process_state)
    by_branch = dict(tips.get("by_branch", {}))
    prior = by_branch.get(CHILD_ID, {})
    by_branch[CHILD_ID] = {
        **payload,
        **({"last_event_identity": prior["last_event_identity"]}
           if "last_event_identity" in prior else {}),
    }
    tips.update({"active_branch_id": CHILD_ID, "by_branch": by_branch})
    return replace(state, tip_process_state=tips)


@pytest.fixture(scope="module")
def near_void_child_state():
    state = _checkpoint("connected_before_downstream.json")
    start = tuple(state.void_state.cavities[0].connection_exit_m)
    candidate = state.competition.candidates[0]
    child = CrackBranchState(
        CHILD_ID,
        state.crack_network.primary_branch_id,
        1,
        int(state.event_counters.get("topology_actions", 0)) + 1,
        (start,),
        (float(candidate.angle_rad),),
        local_state={
            "candidate_id": candidate.candidate_id,
            "front_engine_state_owner": CHILD_ID,
        },
    )
    network = replace(
        state.crack_network,
        branches=state.crack_network.branches + (child,),
        geometry_generation=state.crack_network.geometry_generation + 1,
        branching_enabled=True,
    )
    payload = capture_sharp_front_engine(fresh_sharp_front_engine(state.material))
    return replace(
        state,
        crack_network=network,
        tip_process_state={"active_branch_id": CHILD_ID, "by_branch": {CHILD_ID: payload}},
    )


def _hybrid(
    state, candidates_by_tip, *, contour_radius_m=6.0e-5,
    overlap=False, deltas=(3.0e-5, 2.0e-5), levels=(2, 3),
):
    return hybrid_directional_drive_provider(
        state,
        candidates_by_tip,
        contour_radius_m=contour_radius_m,
        provider_contract_contour_radius_m=contour_radius_m,
        marginal_delta_a_m=deltas,
        marginal_mesh_levels=levels,
        evaluate_overlap_marginals=overlap,
        source_commit=os.environ.get("VOIDING_V5_SOURCE_COMMIT", SOURCE_SHA),
        prepare_support_state=_refresh_downstream_boundary_context,
        conform_trial_endpoint=lambda trial, endpoint: _conform_bulk_point(
            trial, endpoint, identity="HYBRID_TEST_ENDPOINT_CONFORMING",
        ),
    )


def _state_from_request(request):
    reference = evaluate_exact_topology(request)
    return LiveFEMTopologyState(
        mesh=request.mesh,
        boundary=request.boundary,
        damage=request.damage,
        displacement=request.displacement,
        ep_gp=request.ep_gp,
        rho_gp=request.rho_gp,
        elasticity_D=request.elasticity_D,
        material=request.material,
        cohesive_network=request.cohesive_network,
        crack_network=request.crack_network,
        competition=None,
        tip_process_state={},
        junction_process_state={},
        energy_ledgers={},
        rng_state={},
        event_counters={},
        stored_energy_J_per_m=reference["base_equilibrium"][
            "recoverable_potential_energy_J_per_m"
        ],
    )


def _relative(first, second):
    return abs(float(first) - float(second)) / max(abs(float(first)), abs(float(second)), 1.0e-12)


def test_1_no_void_root_provider_parity_is_exact():
    request = live_straight_request(10.0e-6)
    baseline = evaluate_exact_topology(request)
    explicit_empty = evaluate_exact_topology(
        replace(request, cavity_free_surface_inventory=())
    )
    assert explicit_empty["topology_fingerprint"] == baseline["topology_fingerprint"]
    assert pickle.dumps(legacy_from_live(explicit_empty), protocol=5) == pickle.dumps(
        legacy_from_live(baseline), protocol=5
    )
    assert all(
        not contour["cavity_boundary_intersects"]
        and not contour["cavity_polygon_intersects"]
        for tip in explicit_empty["tips"]
        for row in tip["directional"]
        for contour in row["nested_contour_diagnostics"]
    )


def test_2_two_tip_branching_ownership_never_mixes_scalar_or_state_identity():
    request = live_branched_request()
    state = _state_from_request(request)
    candidate_by_id = {
        item.candidate_id: item
        for candidates in request.candidates_by_tip.values() for item in candidates
    }
    owned = {
        tip: (candidate_by_id[state.crack_network.branch(tip).local_state["candidate_id"]],)
        for tip in state.crack_network.active_tip_ids
    }
    result = _hybrid(state, owned, contour_radius_m=1.0e-5, levels=(1,))
    expected = {
        (tip, candidate.candidate_id)
        for tip, candidates in owned.items() for candidate in candidates
    }
    actual = {(row["tip_id"], row["candidate_id"]) for row in result["directional"]}
    assert actual == expected
    assert len({row["accepted_state_id"] for row in result["directional"]}) == 1
    assert len({row["stress_field_state_id"] for row in result["directional"]}) == 1
    assert all(row["topology_fingerprint"] == result["topology_fingerprint"]
               for row in result["directional"])


def test_3_far_void_child_uses_valid_nested_local_J(far_child_state):
    result = _hybrid(
        far_child_state, {CHILD_ID: far_child_state.competition.candidates},
        levels=(1,),
    )
    row = result["directional"][0]
    assert row["local_contour_valid"]
    assert row["drive_source"] == "VALID_NESTED_LOCAL_J"
    assert row["G_kinetic_used_J_per_m2"] == max(row["G_local_J_per_m2"], 0.0)
    assert row["marginal_trial_rows"] == []
    valid = [item for item in row["nested_contour_diagnostics"] if item["geometrically_valid"]]
    assert len(valid) >= 2
    assert any(_relative(a["signed_J_J_per_m2"], b["signed_J_J_per_m2"]) <= 0.15
               for a, b in zip(valid, valid[1:]))


def test_4_overlap_local_and_marginal_agree_without_blending(far_child_state):
    result = _hybrid(
        far_child_state, {CHILD_ID: far_child_state.competition.candidates},
        overlap=True,
    )
    row = result["directional"][0]
    assert row["local_contour_valid"]
    assert row["drive_source"] == "VALID_NESTED_LOCAL_J"
    assert len(row["marginal_trial_rows"]) == 4
    assert _relative(row["G_local_J_per_m2"], row["G_marginal_J_per_m2"]) <= 0.15
    assert row["G_kinetic_used_J_per_m2"] == max(row["G_local_J_per_m2"], 0.0)


def test_5_near_void_rejects_contour_and_uses_converged_exact_marginal(
    near_void_child_state, tmp_path,
):
    before = accepted_state_identity(near_void_child_state)
    result = _hybrid(
        near_void_child_state,
        {CHILD_ID: near_void_child_state.competition.candidates},
    )
    row = result["directional"][0]
    assert not row["local_contour_valid"]
    assert row["cavity_boundary_intersects_J_domain"]
    assert "cavity_free_surface_intersects_contour" in row["local_J_invalid_reason"]
    assert row["drive_source"] == "EXACT_FIXED_VOID_VIRTUAL_EXTENSION_MARGINAL_G"
    assert row["K_interpretation"] == K_INTERPRETATION
    assert result["trial_cache_creation_count"] == result["trial_cache_destruction_count"] == 4
    trials = {(item["delta_a_m"], item["local_mesh_level"]): item
              for item in row["marginal_trial_rows"]}
    assert all(item["status"] == "CERTIFIED_EXACT_FIXED_VOID_MARGINAL"
               and item["accepted_state_preserved"]
               and item["fixed_void_geometry_preserved"]
               and item["kinetic_process_state_preserved"]
               for item in trials.values())
    for level in (2, 3):
        assert _relative(
            trials[(3.0e-5, level)]["G_marginal_J_per_m2"],
            trials[(2.0e-5, level)]["G_marginal_J_per_m2"],
        ) <= 0.15
    for delta in (3.0e-5, 2.0e-5):
        assert _relative(
            trials[(delta, 2)]["G_marginal_J_per_m2"],
            trials[(delta, 3)]["G_marginal_J_per_m2"],
        ) <= 0.15
    assert accepted_state_identity(near_void_child_state) == before

    checkpoint = tmp_path / "near-void-child.json"
    write_checkpoint(near_void_child_state, checkpoint)
    restored = restore_checkpoint(checkpoint)
    assert accepted_state_identity(restored) == before
    replay = _hybrid(
        restored, {CHILD_ID: restored.competition.candidates},
        deltas=(2.0e-5,), levels=(3,),
    )["directional"][0]
    assert replay["G_marginal_J_per_m2"] == pytest.approx(
        trials[(2.0e-5, 3)]["G_marginal_J_per_m2"], rel=1.0e-12
    )
    assert replay["accepted_state_id"] == row["accepted_state_id"]
    assert replay["stress_field_state_id"] == row["stress_field_state_id"]


def test_6_reflected_candidate_pair_has_symmetric_directional_drive(far_child_state):
    angle = math.radians(20.0)
    candidates = tuple(
        CleavageCandidate.create(
            plane_family="cleavage",
            plane_variant=f"reflected-{sign:+d}",
            direction_xy=(math.cos(angle), sign * math.sin(angle)),
            normal_xy=(-sign * math.sin(angle), math.cos(angle)),
            gamma_rel=1.0,
            orientation_convention="reflected bounded hybrid pair",
        )
        for sign in (-1, 1)
    )
    result = _hybrid(far_child_state, {CHILD_ID: candidates}, levels=(1,))
    rows = result["directional"]
    assert all(row["tip_id"] == CHILD_ID and row["local_contour_valid"] for row in rows)
    assert _relative(rows[0]["G_kinetic_used_J_per_m2"], rows[1]["G_kinetic_used_J_per_m2"]) <= 1.0e-3
    assert rows[0]["accepted_state_id"] == rows[1]["accepted_state_id"]
    assert rows[0]["stress_field_state_id"] == rows[1]["stress_field_state_id"]


def test_7_real_child_continuation_uses_one_hybrid_first_passage_and_topology_action(
    far_child_state,
):
    before_actions = int(far_child_state.event_counters["topology_actions"])
    before_pending = sum(len(item.pending_events)
                         for item in far_child_state.competition.hazard_states)
    continued, event, operations, audit = downstream_front_transaction(
        far_child_state, continuation=True,
    )
    assert event is not None and event.accepted
    assert int(continued.event_counters["topology_actions"]) == before_actions + 1
    emitted = [event_id for row in audit["cleavage"] for event_id in row["emitted_event_ids"]]
    assert len(emitted) == 1
    assert len(audit["selected_proposal_candidate_ids"]) == 1
    assert audit["selected_proposal_candidate_ids"] == audit["emitted_winner_candidate_ids"]
    assert sum(len(item.pending_events)
               for item in continued.competition.hazard_states) == before_pending
    assert audit["front_load_rows"][0]["drive_source"] in {
        "VALID_NESTED_LOCAL_J", "EXACT_FIXED_VOID_VIRTUAL_EXTENSION_MARGINAL_G",
    }
    assert audit["cleavage"][0]["K_interpretation"] == K_INTERPRETATION
    assert audit["cleavage"][0]["tip_id"] == CHILD_ID
    assert operations.count("accepted_snapshot") == 1
    assert operations.count("topology_verification") == 1
