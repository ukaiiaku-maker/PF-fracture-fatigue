from dataclasses import replace
import copy
import inspect
import json
import math
import os
from pathlib import Path
import pickle

import pytest

import arrhenius_fracture.hybrid_directional_drive_v5 as hybrid_module
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.crack_network_v11 import CrackBranchState
from arrhenius_fracture.directional_competition_v11 import CleavageCandidate
from arrhenius_fracture.hybrid_directional_drive_v5 import (
    K_INTERPRETATION,
    MARGINAL_G_DELTA_A_RELATIVE_LIMIT,
    MARGINAL_G_MESH_RELATIVE_LIMIT,
    MarginalDriveNotCertified,
    accepted_state_identity,
    hybrid_directional_drive_provider,
    marginal_convergence_diagnostics,
)
from arrhenius_fracture.live_topology_kernel_v11 import evaluate_exact_topology
from arrhenius_fracture.topology_transaction_v11 import LiveFEMTopologyState
from arrhenius_fracture.unified_fracture_material_v5 import benchmark_material_bundle, bind_identity

BUNDLE = benchmark_material_bundle("DBTT")

from arrhenius_fracture.voiding_production_v5 import (
    _conform_bulk_point,
    _refresh_downstream_boundary_context,
    capture_sharp_front_engine,
    directional_sharp_front_rates,
    downstream_front_transaction,
    fresh_sharp_front_engine,
    restore_sharp_front_engine,
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
    return bind_identity(restore_checkpoint(
        CHECKPOINT_ROOT / "artifacts/voiding_v5_finalization_v2/checkpoints" / name
    ), BUNDLE)


@pytest.fixture(scope="module")
def far_child_state():
    state = _checkpoint("downstream_front_before_continuation.json")
    branches = tuple(
        replace(
            branch,
            local_state={
                **branch.local_state,
                **({"front_engine_state_owner": CHILD_ID}
                   if branch.branch_id == CHILD_ID else {}),
            },
        )
        for branch in state.crack_network.branches
    )
    state = replace(state, crack_network=replace(state.crack_network, branches=branches))
    payload = capture_sharp_front_engine(fresh_sharp_front_engine(state.material, BUNDLE))
    tips = dict(state.tip_process_state)
    by_branch = dict(tips.get("by_branch", {}))
    prior = by_branch.get(CHILD_ID, {})
    by_branch[CHILD_ID] = {
        **payload,
        **({"last_event_identity": prior["last_event_identity"]}
           if "last_event_identity" in prior else {}),
    }
    tips.update({
        "active_branch_id": CHILD_ID,
        "by_branch": by_branch,
        "owner_by_front": {CHILD_ID: CHILD_ID},
    })
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
    payload = capture_sharp_front_engine(fresh_sharp_front_engine(state.material, BUNDLE))
    return replace(
        state,
        crack_network=network,
        tip_process_state={
            "active_branch_id": CHILD_ID,
            "by_branch": {CHILD_ID: payload},
            "owner_by_front": {CHILD_ID: CHILD_ID},
        },
    )


def _hybrid(
    state, candidates_by_tip, *, contour_radius_m=6.0e-5,
    overlap=False, deltas=(3.0e-5, 2.0e-5), levels=(2, 3),
    process_owner_by_tip=None,
):
    if process_owner_by_tip is None:
        process_owner_by_tip = {
            tip_id: str(state.crack_network.branch(tip_id).local_state[
                "front_engine_state_owner"
            ])
            for tip_id in state.crack_network.active_tip_ids
        }
    return hybrid_directional_drive_provider(
        state,
        candidates_by_tip,
        process_owner_by_tip=process_owner_by_tip,
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
    owners = {
        tip_id: f"process-owner:{tip_id}"
        for tip_id in request.crack_network.active_tip_ids
    }
    branches = tuple(
        replace(
            branch,
            local_state={
                **branch.local_state,
                **({"front_engine_state_owner": owners[branch.branch_id]}
                   if branch.branch_id in owners else {}),
            },
        )
        for branch in request.crack_network.branches
    )
    network = replace(request.crack_network, branches=branches)
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
        crack_network=network,
        competition=None,
        tip_process_state={
            "by_branch": {owner: {"synthetic_owner": True} for owner in owners.values()},
            "owner_by_front": owners,
        },
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


def _synthetic_marginal_identity(front_id="front-A", owner_id="owner-A"):
    return {
        "candidate_id": "candidate-A",
        "owning_front_id": front_id,
        "process_owner_id": owner_id,
        "accepted_state_id": "accepted-A",
        "stress_field_state_id": "stress-A",
        "topology_fingerprint": "topology-A",
        "void_fingerprint": "void-A",
        "source_commit": SOURCE_SHA,
    }


def _synthetic_marginal_rows(values, *, identity=None):
    identity = _synthetic_marginal_identity() if identity is None else identity
    rows = []
    for (delta, level), G_value in values.items():
        rows.append({
            **identity,
            "status": "CERTIFIED_EXACT_FIXED_VOID_MARGINAL",
            "delta_a_m": delta,
            "local_mesh_level": level,
            "G_marginal_J_per_m2": G_value,
            "Pi_base_J_per_m": 1.0,
            "Pi_trial_J_per_m": 1.0 - float(G_value) * float(delta),
        })
    return rows


def _synthetic_family(values):
    return marginal_convergence_diagnostics(
        _synthetic_marginal_rows(values),
        delta_a_values_m=(3.0e-5, 2.0e-5),
        local_mesh_levels=(2, 3),
        expected_identity=_synthetic_marginal_identity(),
    )


def _print_marginal_diagnostics(label, diagnostics):
    print("MARGINAL_DIAGNOSTICS " + json.dumps(
        {"case": label, **diagnostics}, sort_keys=True,
    ))


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
    assert all(
        row["process_owner_id"] == f"process-owner:{row['tip_id']}"
        for row in result["directional"]
    )


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
    finest_smallest = next(
        trial for trial in row["marginal_trial_rows"]
        if trial["delta_a_m"] == 2.0e-5 and trial["local_mesh_level"] == 3
    )
    assert _relative(
        row["G_local_J_per_m2"], finest_smallest["G_marginal_J_per_m2"]
    ) <= 0.15
    assert not row["marginal_convergence_passed"]
    assert row["G_marginal_J_per_m2"] is None
    assert row["G_kinetic_used_J_per_m2"] == max(row["G_local_J_per_m2"], 0.0)


def test_5_near_void_rejects_contour_and_requires_converged_exact_marginal(
    near_void_child_state, tmp_path,
):
    before = accepted_state_identity(near_void_child_state)
    process_before = pickle.dumps((
        near_void_child_state.competition,
        near_void_child_state.tip_process_state,
        near_void_child_state.rng_state,
        near_void_child_state.event_counters,
        near_void_child_state.void_state,
    ), protocol=5)
    result = _hybrid(
        near_void_child_state,
        {CHILD_ID: near_void_child_state.competition.candidates},
    )
    row = result["directional"][0]
    assert not row["local_contour_valid"]
    assert row["cavity_boundary_intersects_J_domain"]
    assert "cavity_free_surface_intersects_contour" in row["local_J_invalid_reason"]
    assert row["drive_source"] == "MARGINAL_G_NOT_CONVERGED"
    assert row["K_interpretation"] == K_INTERPRETATION
    assert row["directional_drive_status"] == "DIRECTIONAL_DRIVE_UNAVAILABLE"
    assert row["marginal_unavailable_reason"] == "MARGINAL_G_MESH_NOT_CONVERGED"
    assert row["G_marginal_J_per_m2"] is None
    assert row["G_kinetic_used_J_per_m2"] == 0.0
    assert row["K_energy_equivalent_Pa_sqrt_m"] == 0.0
    assert row["effective_rate"] == 0.0
    rates = directional_sharp_front_rates(
        near_void_child_state, result["directional"], branch_id=CHILD_ID,
    )
    assert all(rate["effective_rate_s"] == 0.0 for rate in rates)
    assert all(math.isinf(rate["crossing_time_s"]) for rate in rates)
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
        ) <= MARGINAL_G_DELTA_A_RELATIVE_LIMIT
    mesh_errors = []
    for delta in (3.0e-5, 2.0e-5):
        mesh_errors.append(_relative(
            trials[(delta, 2)]["G_marginal_J_per_m2"],
            trials[(delta, 3)]["G_marginal_J_per_m2"],
        ))
    assert max(mesh_errors) > MARGINAL_G_MESH_RELATIVE_LIMIT
    assert accepted_state_identity(near_void_child_state) == before
    assert pickle.dumps((
        near_void_child_state.competition,
        near_void_child_state.tip_process_state,
        near_void_child_state.rng_state,
        near_void_child_state.event_counters,
        near_void_child_state.void_state,
    ), protocol=5) == process_before

    checkpoint = tmp_path / "near-void-child.json"
    write_checkpoint(near_void_child_state, checkpoint)
    restored = restore_checkpoint(checkpoint)
    assert accepted_state_identity(restored) == before
    replay = _hybrid(
        restored, {CHILD_ID: restored.competition.candidates},
        deltas=(3.0e-5, 2.0e-5), levels=(2, 3),
    )["directional"][0]
    assert replay["G_marginal_J_per_m2"] is None
    assert replay["G_kinetic_used_J_per_m2"] == 0.0
    replay_trials = {
        (item["delta_a_m"], item["local_mesh_level"]): item
        for item in replay["marginal_trial_rows"]
    }
    assert replay_trials[(2.0e-5, 3)]["G_marginal_J_per_m2"] == pytest.approx(
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
    if audit["front_load_rows"][0]["drive_source"].startswith("EXACT_FIXED_VOID"):
        assert audit["front_load_rows"][0]["marginal_convergence_passed"]
    assert audit["cleavage"][0]["K_interpretation"] == K_INTERPRETATION
    assert audit["cleavage"][0]["tip_id"] == CHILD_ID
    assert audit["cleavage"][0]["process_owner_id"] == CHILD_ID
    assert operations.count("accepted_snapshot") == 1
    assert operations.count("topology_verification") == 1
    payload = continued.tip_process_state["by_branch"][CHILD_ID]
    assert payload["last_step_audit"]["fired"]
    assert payload["last_step_audit"]["n_fire"] == 1
    engine = restore_sharp_front_engine(continued.material, BUNDLE, payload)
    assert engine.r_eff() == pytest.approx(
        engine.f.r0 + engine.f.c_blunt * engine.b * engine.N_em
    )
    assert engine.r_eff() != pytest.approx(
        continued.void_state.cavities[0].radius_m
    )
    assert "R_void" not in inspect.signature(directional_sharp_front_rates).parameters


def test_8_converged_marginal_family_uses_smallest_delta_on_finest_mesh():
    values = {
        (3.0e-5, 2): 101.0,
        (3.0e-5, 3): 102.0,
        (2.0e-5, 2): 103.0,
        (2.0e-5, 3): 104.0,
    }
    diagnostics = _synthetic_family(values)
    _print_marginal_diagnostics("CONVERGED", diagnostics)
    assert diagnostics["marginal_convergence_passed"]
    assert diagnostics["authoritative_delta_a_m"] == 2.0e-5
    assert diagnostics["authoritative_mesh_level"] == 3
    assert diagnostics["authoritative_G_marginal_J_per_m2"] == 104.0
    assert diagnostics["marginal_trial_count_expected"] == 4
    assert diagnostics["marginal_trial_count_observed"] == 4
    assert diagnostics["marginal_all_trials_certified"]


def test_9_mesh_nonconvergence_rejects_finite_certified_family():
    diagnostics = _synthetic_family({
        (3.0e-5, 2): 100.0,
        (3.0e-5, 3): 130.0,
        (2.0e-5, 2): 100.0,
        (2.0e-5, 3): 101.0,
    })
    _print_marginal_diagnostics("MESH_NONCONVERGED", diagnostics)
    assert not diagnostics["marginal_convergence_passed"]
    assert diagnostics["marginal_unavailable_reason"] == "MARGINAL_G_MESH_NOT_CONVERGED"
    assert diagnostics["authoritative_G_marginal_J_per_m2"] is None


def test_10_delta_a_nonconvergence_rejects_mesh_converged_family():
    diagnostics = _synthetic_family({
        (3.0e-5, 2): 130.0,
        (3.0e-5, 3): 131.0,
        (2.0e-5, 2): 100.0,
        (2.0e-5, 3): 101.0,
    })
    _print_marginal_diagnostics("DELTA_A_NONCONVERGED", diagnostics)
    assert all(
        value <= MARGINAL_G_MESH_RELATIVE_LIMIT
        for value in diagnostics["marginal_mesh_relative_errors_by_delta"].values()
    )
    assert not diagnostics["marginal_convergence_passed"]
    assert diagnostics["marginal_unavailable_reason"] == "MARGINAL_G_DELTA_A_NOT_CONVERGED"
    assert diagnostics["authoritative_G_marginal_J_per_m2"] is None


def test_11_incomplete_or_uncertified_marginal_family_is_unavailable():
    values = {
        (3.0e-5, 2): 101.0,
        (3.0e-5, 3): 102.0,
        (2.0e-5, 2): 103.0,
        (2.0e-5, 3): 104.0,
    }
    rows = _synthetic_marginal_rows(values)
    incomplete = marginal_convergence_diagnostics(
        rows[:-1],
        delta_a_values_m=(3.0e-5, 2.0e-5),
        local_mesh_levels=(2, 3),
        expected_identity=_synthetic_marginal_identity(),
    )
    assert incomplete["marginal_unavailable_reason"] == "INCOMPLETE_MARGINAL_TRIAL_FAMILY"
    uncertified_rows = copy.deepcopy(rows)
    uncertified_rows[-1]["status"] = "DIRECTIONAL_DRIVE_UNAVAILABLE"
    uncertified = marginal_convergence_diagnostics(
        uncertified_rows,
        delta_a_values_m=(3.0e-5, 2.0e-5),
        local_mesh_levels=(2, 3),
        expected_identity=_synthetic_marginal_identity(),
    )
    _print_marginal_diagnostics("INCOMPLETE", incomplete)
    _print_marginal_diagnostics("UNCERTIFIED", uncertified)
    assert uncertified["marginal_unavailable_reason"] == "MARGINAL_TRIAL_NOT_CERTIFIED"
    assert not incomplete["marginal_convergence_passed"]
    assert not uncertified["marginal_convergence_passed"]


def test_12_sign_inconsistent_family_is_unavailable_except_at_numerical_zero():
    inconsistent = _synthetic_family({
        (3.0e-5, 2): -100.0,
        (3.0e-5, 3): -101.0,
        (2.0e-5, 2): 100.0,
        (2.0e-5, 3): 101.0,
    })
    assert not inconsistent["marginal_signed_G_consistent"]
    assert inconsistent["marginal_unavailable_reason"] == "MARGINAL_G_SIGN_INCONSISTENT"
    zero_scale = 1.0e-6
    numerical_zero = _synthetic_family({
        (3.0e-5, 2): -zero_scale,
        (3.0e-5, 3): zero_scale,
        (2.0e-5, 2): -zero_scale,
        (2.0e-5, 3): zero_scale,
    })
    _print_marginal_diagnostics("SIGN_INCONSISTENT", inconsistent)
    _print_marginal_diagnostics("SIGNED_NUMERICAL_ZERO", numerical_zero)
    assert numerical_zero["marginal_signed_G_consistent"]
    assert numerical_zero["marginal_zero_drive_classification"] == (
        "ALL_SIGNED_G_WITHIN_NUMERICAL_ZERO"
    )
    assert numerical_zero["authoritative_G_marginal_J_per_m2"] == 0.0


def test_13_actual_process_owner_and_absent_owner_fail_closed_before_trial(
    far_child_state, tmp_path, monkeypatch,
):
    process_owner = "canonical-process-owner-A"
    branches = tuple(
        replace(
            branch,
            local_state={**branch.local_state, "front_engine_state_owner": process_owner},
        ) if branch.branch_id == CHILD_ID else branch
        for branch in far_child_state.crack_network.branches
    )
    original_payload = far_child_state.tip_process_state["by_branch"][CHILD_ID]
    state = replace(
        far_child_state,
        crack_network=replace(far_child_state.crack_network, branches=branches),
        tip_process_state={
            **far_child_state.tip_process_state,
            "by_branch": {process_owner: original_payload},
            "owner_by_front": {CHILD_ID: process_owner},
        },
    )
    owner_by_tip = {CHILD_ID: process_owner}

    def converged_trial(_state, *, delta_a_m, local_mesh_level, **kwargs):
        G_value = 100.0 + 1000.0 * float(delta_a_m) + 0.01 * int(local_mesh_level)
        return {
            "status": "CERTIFIED_EXACT_FIXED_VOID_MARGINAL",
            "G_marginal_J_per_m2": G_value,
            "Pi_base_J_per_m": 1.0,
            "Pi_trial_J_per_m": 1.0 - G_value * float(delta_a_m),
            "delta_a_m": float(delta_a_m),
            "local_mesh_level": int(local_mesh_level),
        }

    monkeypatch.setattr(hybrid_module, "_exact_marginal_trial", converged_trial)
    result = _hybrid(
        state, {CHILD_ID: state.competition.candidates}, overlap=True,
        process_owner_by_tip=owner_by_tip,
    )
    row = result["directional"][0]
    assert row["tip_id"] == CHILD_ID
    assert row["process_owner_id"] == process_owner
    assert all(
        trial["trial_cache_identity"]["process_owner_id"] == process_owner
        and trial["process_owner_id"] == process_owner
        for trial in row["marginal_trial_rows"]
    )
    checkpoint = tmp_path / "distinct-process-owner.json"
    write_checkpoint(state, checkpoint)
    restored = restore_checkpoint(checkpoint)
    replay = _hybrid(
        restored, {CHILD_ID: restored.competition.candidates}, overlap=True,
        process_owner_by_tip=restored.tip_process_state["owner_by_front"],
    )["directional"][0]
    assert replay["process_owner_id"] == process_owner
    assert replay["accepted_state_id"] == row["accepted_state_id"]
    assert replay["G_kinetic_used_J_per_m2"] == row["G_kinetic_used_J_per_m2"]

    trial_calls = 0

    def forbidden_trial(*args, **kwargs):
        nonlocal trial_calls
        trial_calls += 1
        raise AssertionError("owner rejection must precede marginal trial")

    monkeypatch.setattr(hybrid_module, "_exact_marginal_trial", forbidden_trial)
    invalid = dict(owner_by_tip)
    invalid[CHILD_ID] = "absent-owner"
    with pytest.raises(RuntimeError, match="absent process owner"):
        _hybrid(
            state, {CHILD_ID: state.competition.candidates}, overlap=True,
            process_owner_by_tip=invalid,
        )
    assert trial_calls == 0


def test_14_two_fronts_share_one_process_owner_without_cache_or_observation_collision(
    monkeypatch,
):
    request = live_branched_request()
    state = _state_from_request(request)
    shared_owner = "shared-unresolved-process"
    branches = tuple(
        replace(
            branch,
            local_state={**branch.local_state, "front_engine_state_owner": shared_owner},
        )
        for branch in state.crack_network.branches
    )
    owner_by_tip = {tip: shared_owner for tip in state.crack_network.active_tip_ids}
    state = replace(
        state,
        crack_network=replace(state.crack_network, branches=branches),
        tip_process_state={
            "by_branch": {shared_owner: {"synthetic_owner": True}},
            "owner_by_front": owner_by_tip,
        },
    )
    candidate_by_id = {
        item.candidate_id: item
        for candidates in request.candidates_by_tip.values() for item in candidates
    }
    owned = {
        tip: (candidate_by_id[state.crack_network.branch(tip).local_state["candidate_id"]],)
        for tip in state.crack_network.active_tip_ids
    }

    def converged_trial(_state, *, delta_a_m, local_mesh_level, **kwargs):
        G_value = 100.0 + 1000.0 * float(delta_a_m) + 0.01 * int(local_mesh_level)
        return {
            "status": "CERTIFIED_EXACT_FIXED_VOID_MARGINAL",
            "G_marginal_J_per_m2": G_value,
            "Pi_base_J_per_m": 1.0,
            "Pi_trial_J_per_m": 1.0 - G_value * float(delta_a_m),
            "delta_a_m": float(delta_a_m),
            "local_mesh_level": int(local_mesh_level),
        }

    monkeypatch.setattr(hybrid_module, "_exact_marginal_trial", converged_trial)
    result = _hybrid(
        state, owned, contour_radius_m=1.0e-5, overlap=True,
        process_owner_by_tip=owner_by_tip,
    )
    rows = result["directional"]
    assert len({row["tip_id"] for row in rows}) == 2
    assert {row["process_owner_id"] for row in rows} == {shared_owner}
    cache_front_candidate_pairs = {
        (
            trial["trial_cache_identity"]["selected_front_id"],
            trial["trial_cache_identity"]["candidate_id"],
        )
        for row in rows for trial in row["marginal_trial_rows"]
    }
    assert cache_front_candidate_pairs == {
        (row["tip_id"], row["candidate_id"]) for row in rows
    }
    assert all(
        trial["owning_front_id"] == row["tip_id"]
        and trial["process_owner_id"] == shared_owner
        for row in rows for trial in row["marginal_trial_rows"]
    )


def test_15_expected_noncertification_fails_closed_and_programming_errors_propagate(
    near_void_child_state, monkeypatch,
):
    before = pickle.dumps(near_void_child_state, protocol=5)

    def expected_noncertification(*args, **kwargs):
        raise MarginalDriveNotCertified("EXACT_V12_SUPPORT_NOT_CERTIFIED")

    monkeypatch.setattr(
        hybrid_module, "_exact_marginal_trial", expected_noncertification
    )
    result = _hybrid(
        near_void_child_state,
        {CHILD_ID: near_void_child_state.competition.candidates},
    )
    row = result["directional"][0]
    assert row["directional_drive_status"] == "DIRECTIONAL_DRIVE_UNAVAILABLE"
    assert row["drive_source"] == "MARGINAL_G_NOT_CONVERGED"
    assert row["G_kinetic_used_J_per_m2"] == 0.0
    assert row["K_energy_equivalent_Pa_sqrt_m"] == 0.0
    assert row["effective_rate"] == 0.0
    assert result["trial_cache_creation_count"] == 4
    assert result["trial_cache_destruction_count"] == 4
    assert pickle.dumps(near_void_child_state, protocol=5) == before

    def unexpected_runtime(*args, **kwargs):
        raise RuntimeError("unexpected programming runtime failure")

    monkeypatch.setattr(hybrid_module, "_exact_marginal_trial", unexpected_runtime)
    with pytest.raises(RuntimeError, match="unexpected programming runtime failure"):
        _hybrid(
            near_void_child_state,
            {CHILD_ID: near_void_child_state.competition.candidates},
        )

    def unexpected_value(*args, **kwargs):
        raise ValueError("unexpected malformed array")

    monkeypatch.setattr(hybrid_module, "_exact_marginal_trial", unexpected_value)
    with pytest.raises(ValueError, match="unexpected malformed array"):
        _hybrid(
            near_void_child_state,
            {CHILD_ID: near_void_child_state.competition.candidates},
        )
    assert pickle.dumps(near_void_child_state, protocol=5) == before
