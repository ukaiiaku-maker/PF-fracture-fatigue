from dataclasses import dataclass, replace
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture.crack_network_v11 import CrackNetworkState
from arrhenius_fracture.directional_competition_v11 import (
    CleavageCandidate, CompletedDirectionalEvent, DirectionalCompetitionState,
    DirectionalHazardState,
    competition_state_to_dict,
)
from arrhenius_fracture.general_multifront_v12 import (
    CouplingEvidence, FrontRuntimeState, MultiFrontRuntimeState, ProcessEngineState,
    ProcessRegionState, ResourcePolicy, TopologyProposal, commit_selected_proposal,
    recompute_process_region_connectivity,
)
from arrhenius_fracture.production_multifront_v12 import (
    AdaptedAcceptedState, DirectionalObservationBatch, ExactTrialDeltaV12,
    ProposalTrialOutcome, SolvedAcceptedState, accepted_fem_state_fingerprint,
    apply_exact_trial_delta,
)
from arrhenius_fracture.stateful_multifront_production_v12 import (
    AtomicIntervalTransactionWriterV12,
    CurrentSourceMultiFrontProductionContextV12, StatefulProductionInterlock,
    _apply_realized_cleavage_intersections_v12,
    build_arbitrary_region_request_v12, build_stateful_production_hooks,
    event_endpoint_iteration_limit,
    run_stateful_accepted_interval_v12,
    stress_field_identity,
)
from arrhenius_fracture.topology_transaction_v11 import (
    TopologyArm, extend_network_arm, mark_coalesced,
)


def candidate(name="forward"):
    return CleavageCandidate.create(
        plane_family="100", plane_variant=name, direction_xy=(1.0, 0.0),
        normal_xy=(0.0, 1.0), gamma_rel=1.0,
        orientation_convention="v6.4-fixture",
    )


def fixture(
    tmp_path, *, rate=0.0, stale=False, exact_trials=False,
    sigma_value=1.0, pending=False, adapted_damage=None,
):
    item = candidate()
    hazard = DirectionalHazardState(item.candidate_id, previous_rate_per_s=rate)
    if pending:
        event = CompletedDirectionalEvent(item.candidate_id, 1, 0.0, 0.0, 1.0, 1.0)
        hazard = DirectionalHazardState(
            item.candidate_id, action=1.0, previous_rate_per_s=rate,
            completed_event_count=1, residual_action=0.0,
            last_completion_time_s=0.0, pending_events=(event,),
            current_threshold_action=2.0,
        )
    competition = DirectionalCompetitionState(
        candidates=(item,), hazard_states=(hazard,), global_hazard_seed=3621,
    )
    network = CrackNetworkState.one_tip(((0.0, 0.0),), initial_orientation_rad=0.0)
    front = network.active_tip_ids[0]
    engine = ProcessEngineState(
        "engine:root", "source:root", {"mobile": 1.0}, {"retained": 0.0},
        {"net": 1.0}, local_process_coordinate_m=0.0,
    )
    runtime = MultiFrontRuntimeState.one_front(
        network, FrontRuntimeState(
            front, competition_state_to_dict(competition), (item.candidate_id,),
            {"seed": 3621},
        ), engine, resource_policy=ResourcePolicy(
            "mechanistic", None, None, "experimental",
            "v11_correlated_proposal_compatibility",
        ), accepted_state_id="legacy", stress_field_state_id="legacy",
    )
    state = SimpleNamespace(
        crack_network=network,
        mesh=SimpleNamespace(
            nodes=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            elems=np.array([[0, 1, 2]]), area_e=np.array([0.5]),
        ),
        displacement=np.array([0.0, 0.0]), ep_gp=np.array([[0.0]]),
        rho_gp=np.array([[1.0]]), damage=np.array([0.0]), elasticity_D=np.eye(3),
    )
    sigma = np.full((3, 1), float(sigma_value))

    def make_engine():
        return SimpleNamespace(mpz=SimpleNamespace(
            advance_total_m=0.0,
            mobile=np.array([[1.0]]), retained=np.array([[0.0]]),
            wake_mobile=np.array([[0.0]]), wake_retained=np.array([[0.0]]),
        ))

    def adapted(current, inventory, context):
        if adapted_damage is not None:
            current = SimpleNamespace(**current.__dict__)
            current.damage = np.full_like(current.damage, float(adapted_damage))
        return AdaptedAcceptedState(current, {"kind": "already_adapted_cached"})

    def solved(adapted_state, context):
        digest = accepted_fem_state_fingerprint(adapted_state.fem_state)
        return SolvedAcceptedState(
            adapted_state.fem_state, sigma,
            {"accepted_fem_state_sha256": "stale" if stale else digest},
            context.accepted_opening_m + 1e-9,
            context.physical_time_s + 8.4,
            "fixture:assembly-only",
        )

    def request(solved_state, state_runtime, context):
        return {"topology_fingerprint": state_runtime.topology_fingerprint}

    def provider(_request, state_runtime, context):
        branch = state_runtime.crack_network.branch(front)
        return {"tips": [{
            "tip_xy_m": list(branch.tip),
            "directional": [{
                "candidate_id": item.candidate_id,
                "J_local_signed_J_per_m2": 1.0,
                "K_directional_Pa_sqrt_m": 1.0e6,
                "local_J_valid": True,
                "local_J_invalid_reason": None,
            }],
        }]}

    def evolve(owner, observation, duration, context):
        actual = make_engine()
        actual.mpz.advance_total_m = owner.local_process_coordinate_m
        context.actual_engines[owner.engine_id] = actual
        context.interval_info_by_owner[owner.engine_id] = {
            "interval_evolved": True, "kinetic_dt_consumed_s": duration,
            "kinetic_dt_unused_s": 0.0, "da": 0.0,
            "event_moving_frame_renewal_count": 0,
        }
        return owner.evolve_once()

    def renew(engine_id, selected, distance, context):
        info = context.interval_info_by_owner[engine_id]
        info["event_moving_frame_renewal_count"] = int(selected)
        info["event_moving_frame_advance_m"] = distance if selected else 0.0
        if selected:
            context.actual_engines[engine_id].mpz.advance_total_m += distance
        return {"selected": selected, "distance_m": distance}

    def recapture(owner, actual, selected, context):
        return replace(
            owner, update_count=owner.update_count,
            event_renewal_count=owner.event_renewal_count + int(selected),
            local_process_coordinate_m=actual.mpz.advance_total_m,
        )

    def trial_executor(state, proposal):
        from arrhenius_fracture.multifront_competition_v12 import finalize_competitions
        runtime = context.provisional_runtime
        preview = next(iter(context.previewed_clock_states.values()))
        finalized = finalize_competitions(
            runtime, preview, selected_proposal=proposal, accepted=True,
        ).runtime
        post = commit_selected_proposal(finalized, proposal)
        # The isolated exact trial carries structural registries before the
        # production owner's single physical renewal.
        regions = dict(post.process_regions)
        engines = dict(post.process_engines)
        for owner_id, prior_region in runtime.process_regions.items():
            if owner_id in regions and prior_region.process_engine_id in engines:
                engines[prior_region.process_engine_id] = runtime.process_engines[
                    prior_region.process_engine_id
                ]
                regions[owner_id] = replace(
                    regions[owner_id],
                    cumulative_process_advance_m=runtime.process_engines[
                        prior_region.process_engine_id
                    ].local_process_coordinate_m,
                )
        post = replace(post, process_regions=regions, process_engines=engines)
        delta = _delta_from_post(runtime, post, proposal)
        trial_state = SimpleNamespace(**state.__dict__)
        trial_state.crack_network = post.crack_network
        return ProposalTrialOutcome(
            proposal, True, trial_state, "accepted_exact_fixture",
            exact_realized_crack_network=post.crack_network,
            topology_fingerprint=post.topology_fingerprint,
            realized_endpoints_m=delta.realized_endpoints_m,
            realized_arm_lengths_m=delta.realized_lengths_m,
            exact_delta=delta, exact_post_sigma_gp=sigma,
        )

    context = CurrentSourceMultiFrontProductionContextV12(
        args={"T_K": 700.0}, mechanical_configuration={"theta": 40.0},
        accepted_fem_state=state, accepted_stress_field=sigma,
        runtime=runtime, provider_runtime={"cache": True},
        destination_cache_root=tmp_path / "cache",
        output_root=tmp_path / "production-output",
        checkpoint_path=tmp_path / "unused.json", candidates=(item,),
        stress_available=True, mechanics_source_identity="fixture:assembly-only",
        adapter_configuration={
            "da_phys_m": 5e-6, "temperature_K": 700.0,
            "accepted_mesh_adapter": adapted,
            "accepted_mechanics_loader": solved,
            "fractional_mechanics_invariant_fixture": True,
            "exact_request_builder": request, "provider_lookup": provider,
            "process_engine_evolver": evolve, "event_renewer": renew,
            "process_engine_recapture": recapture,
            "accepted_boundary_stress_rebuilder": lambda _state, _context: sigma,
            "correlation_interval_s": 0.0,
            **({"topology_trial_executor": trial_executor} if exact_trials else {}),
        },
    )
    return context


def test_explicit_hook_result_types_reject_raw_tuple(tmp_path):
    context = fixture(tmp_path)
    context.adapter_configuration["accepted_mesh_adapter"] = lambda *args: (args[0],)
    with pytest.raises(StatefulProductionInterlock, match="invalid result"):
        run_stateful_accepted_interval_v12(context, 8.4, dry_run_discard=True)
    assert context.transaction_phase == "accepted"
    assert not (tmp_path / "production-output").exists()


def test_transition_checkpoint_marked_pre_adapted_is_not_refined_twice(tmp_path):
    context = fixture(tmp_path)
    context.adapter_configuration["accepted_pre_adapted_source_sha256"] = (
        accepted_fem_state_fingerprint(context.accepted_fem_state)
    )
    context.adapter_configuration["accepted_mesh_adapter"] = lambda *args: (_ for _ in ()).throw(
        AssertionError("pre-adapted source was adapted twice")
    )
    result = build_stateful_production_hooks(context).adapt_accepted_mesh(
        context.accepted_fem_state,
        {
            front_id: front.mechanically_active_candidate_ids
            for front_id, front in context.runtime.front_runtimes.items()
        },
    )
    assert result.fem_state is context.accepted_fem_state
    assert result.adaptation_record["additional_refinement_performed"] is False


def test_accepted_fem_fingerprint_is_checkpoint_roundtrip_stable(tmp_path):
    context = fixture(tmp_path)
    state = context.accepted_fem_state
    restored = pickle.loads(pickle.dumps(state, protocol=5))
    assert accepted_fem_state_fingerprint(state) == accepted_fem_state_fingerprint(
        restored
    )


def test_stress_identity_uses_state_mesh_topology_and_source(tmp_path):
    context = fixture(tmp_path)
    first = stress_field_identity(
        context.accepted_fem_state, context.accepted_stress_field,
        mechanics_source_identity="source:A", accepted_state_id="accepted:A",
    )
    second = stress_field_identity(
        context.accepted_fem_state, context.accepted_stress_field * 2.0,
        mechanics_source_identity="source:A", accepted_state_id="accepted:A",
    )
    third = stress_field_identity(
        context.accepted_fem_state, context.accepted_stress_field,
        mechanics_source_identity="source:B", accepted_state_id="accepted:A",
    )
    assert len({first, second, third}) == 3
    source = fixture(tmp_path / "unavailable")
    unavailable = CurrentSourceMultiFrontProductionContextV12(
        args=source.args, mechanical_configuration=source.mechanical_configuration,
        accepted_fem_state=source.accepted_fem_state,
        accepted_stress_field=None, runtime=source.runtime,
        provider_runtime=source.provider_runtime,
        destination_cache_root=tmp_path / "unavailable-cache",
        output_root=tmp_path / "unavailable-output",
        checkpoint_path=tmp_path / "unavailable-checkpoint",
        candidates=source.candidates, stress_available=False,
        mechanics_source_identity="explicitly-unavailable",
    )
    assert unavailable.accepted_stress_field is None
    assert "UNAVAILABLE" in unavailable.stress_field_state_id
    with pytest.raises(StatefulProductionInterlock, match="unavailable"):
        unavailable.require_usable_stress()


def test_finite_physical_zero_stress_is_available_and_identity_bound(tmp_path):
    context = fixture(tmp_path, sigma_value=0.0)
    assert context.stress_available
    assert context.accepted_stress_field is not None
    assert not context.accepted_stress_field.flags.writeable
    assert "UNAVAILABLE" not in context.stress_field_state_id
    context.require_usable_stress()
    result = run_stateful_accepted_interval_v12(context, 8.4, dry_run_discard=True)
    assert result.disposition == "no_event"


def test_unavailable_stress_rejects_numerical_sentinel(tmp_path):
    source = fixture(tmp_path / "source")
    with pytest.raises(StatefulProductionInterlock, match="must not carry"):
        CurrentSourceMultiFrontProductionContextV12(
            args={}, mechanical_configuration={},
            accepted_fem_state=source.accepted_fem_state,
            accepted_stress_field=np.zeros((3, 1)), runtime=source.runtime,
            provider_runtime={}, destination_cache_root=tmp_path / "cache",
            output_root=tmp_path / "output", checkpoint_path=tmp_path / "latest",
            candidates=source.candidates, stress_available=False,
        )


def test_stale_sigma_fem_pair_fails_before_output(tmp_path):
    context = fixture(tmp_path, stale=True)
    before = context.fingerprint
    with pytest.raises(StatefulProductionInterlock, match="stale sigma"):
        run_stateful_accepted_interval_v12(context, 8.4, dry_run_discard=True)
    assert context.fingerprint == before
    assert context.accepted_stress_field is not None
    assert not context.accepted_stress_field.flags.writeable
    assert not (tmp_path / "production-output").exists()


def test_integrated_no_event_interval_runs_all_stages_and_discards(tmp_path):
    context = fixture(tmp_path, rate=0.0)
    before = context.runtime
    result = run_stateful_accepted_interval_v12(context, 8.4, dry_run_discard=True)
    assert result.disposition == "no_event"
    assert result.selected_event_ids == ()
    assert not result.output_published
    owner_before = next(iter(before.process_engines.values()))
    owner_after = next(iter(context.runtime.process_engines.values()))
    assert owner_after.update_count == owner_before.update_count + 1
    assert owner_after.event_renewal_count == owner_before.event_renewal_count
    assert context.physical_time_s == pytest.approx(8.4)
    assert context.provider_solve_count == context.mechanics_solve_count == 0
    assert not (tmp_path / "production-output").exists()
    names = [row["hook"] for row in result.lifecycle]
    assert names == [
        "accepted_mesh_adaptation", "accepted_fem_solve", "exact_request",
        "cache_only_provider_lookup", "directional_observations",
        "competition_preview", "process_interval_evolution",
        "process_interval_validation", "competition_finalization", "event_renewal",
        "process_region_connectivity",
    ]


def test_pending_boundary_event_uses_adapted_same_opening_fem_state(tmp_path):
    context = fixture(
        tmp_path, pending=True, exact_trials=True, adapted_damage=0.375,
    )
    before_time = context.physical_time_s
    before_opening = context.accepted_opening_m
    result = run_stateful_accepted_interval_v12(
        context, 8.4, dry_run_discard=True,
    )
    assert result.disposition == "accepted_event"
    assert result.accepted_duration_s == 0.0
    assert context.physical_time_s == before_time
    assert context.accepted_opening_m == before_opening
    np.testing.assert_array_equal(context.accepted_fem_state.damage, [0.375])


@pytest.mark.parametrize("stage", (
    "engine_finalization_to_renewal", "renewal_to_output_staging",
    "output_staging", "output_to_checkpoint_staging", "checkpoint_to_commit_marker",
))
def test_integrated_failure_restores_prior_context(stage, tmp_path):
    context = fixture(tmp_path)
    before = context.fingerprint
    with pytest.raises(StatefulProductionInterlock, match="injected"):
        run_stateful_accepted_interval_v12(
            context, 8.4, dry_run_discard=False, failure_stage=stage,
        )
    assert context.fingerprint == before
    pointer = tmp_path / "production-output" / "LATEST"
    assert not pointer.exists()


def test_atomic_writer_publishes_one_authoritative_pointer(tmp_path):
    context = fixture(tmp_path)
    result = run_stateful_accepted_interval_v12(context, 8.4)
    assert result.output_published
    root = tmp_path / "production-output"
    tx = (root / "LATEST").read_text().strip()
    committed = root / "transactions" / tx
    assert (committed / "COMMITTED").is_file()
    assert (committed / "transaction_manifest.json").is_file()
    assert not (root / ".staging" / tx).exists()
    directory, manifest = AtomicIntervalTransactionWriterV12(root).read_latest_committed()
    assert directory == committed and manifest["transaction_id"] == tx


def _proposal(runtime, action, index=1):
    front = runtime.active_front_ids[0]
    owner = runtime.owner_by_front[front]
    tip = runtime.crack_network.branch(front).tip
    if action == "one_arm":
        candidates = ("cleave:a",); endpoints = ((tip[0] + 0.4e-6, tip[1]),)
        target = None
    elif action == "two_arm":
        candidates = ("cleave:a", "cleave:b")
        endpoints = ((tip[0] + 0.5e-6, tip[1] - 0.1e-6),
                     (tip[0] + 0.6e-6, tip[1] + 0.1e-6))
        target = None
    elif action == "coalescence":
        candidates = (); endpoints = (); target = runtime.active_front_ids[1]
    else:
        candidates = (); endpoints = (); target = None
    return TopologyProposal(
        f"trial:{action}:{index}", action, front, owner, candidates, 1.0,
        endpoints, target_front_id=target,
    )


def _branched_runtime(count=1):
    network = CrackNetworkState.one_tip(((0.0, 0.0),), initial_orientation_rad=0.0)
    front = network.active_tip_ids[0]
    runtime = MultiFrontRuntimeState.one_front(
        network, FrontRuntimeState(front, {"action": 0.0},
                                   ("cleave:a", "cleave:b"), {"seed": 7}),
        ProcessEngineState("engine:root", "source:root", {"mobile": 1.0},
                           {"retained": 2.0}, {"net": 1.0}),
        resource_policy=ResourcePolicy("mechanistic", None, None, "experimental"),
        accepted_state_id="accepted:0", stress_field_state_id="stress:0",
    )
    while len(runtime.active_front_ids) < count:
        front = runtime.active_front_ids[0]
        fronts = dict(runtime.front_runtimes)
        fronts[front] = fronts[front].activate_complete_competition()
        runtime = replace(runtime, front_runtimes=fronts)
        runtime = commit_selected_proposal(runtime, _proposal(runtime, "two_arm", len(runtime.active_front_ids)))
    return runtime


def _exact_outcome(before, proposal):
    post = commit_selected_proposal(before, proposal)
    delta = _delta_from_post(before, post, proposal)
    fem = SimpleNamespace(crack_network=post.crack_network)
    return ProposalTrialOutcome(
        proposal, True, fem, "accepted_exact_fixture",
        exact_realized_crack_network=post.crack_network,
        topology_fingerprint=post.topology_fingerprint,
        realized_endpoints_m=delta.realized_endpoints_m,
        realized_arm_lengths_m=delta.realized_lengths_m,
        exact_delta=delta, exact_post_sigma_gp=np.ones((3, 1)),
    ), post


def _delta_from_post(before, post, proposal):
    return ExactTrialDeltaV12(
        pre_crack_network=before.crack_network, post_crack_network=post.crack_network,
        continued_front_ids=tuple(sorted(set(before.active_front_ids) & set(post.active_front_ids))),
        created_front_ids=tuple(sorted(set(post.active_front_ids) - set(before.active_front_ids))),
        retired_front_ids=tuple(sorted(set(before.active_front_ids) - set(post.active_front_ids))),
        coalescence_target_front_id=proposal.target_front_id,
        junctions_after=post.junctions, owner_by_front_after=post.owner_by_front,
        process_regions_after=post.process_regions, process_engines_after=post.process_engines,
        front_runtimes_after=post.front_runtimes, reservoirs_after=post.reservoirs,
        scheduler_after=post.scheduler,
        cumulative_branch_births_after=post.cumulative_branch_births,
        cumulative_coalescences_after=post.cumulative_coalescences,
        cumulative_retirements_after=post.cumulative_retirements,
        transaction_records_after=post.transaction_records,
        output_counters_after=post.output_counters,
        termination_reason_after=post.termination_reason,
        policy_bound_after=post.policy_bound,
        realized_endpoints_m=proposal.end_points_m,
        realized_lengths_m=tuple(
            np.linalg.norm(np.asarray(end) - np.asarray(before.crack_network.branch(proposal.front_id).tip))
            for end in proposal.end_points_m
        ),
        wake_mutation={"kind": "exact_fixture"}, released_energy_J_per_m=2.0,
        dissipative_cost_J_per_m=1.0,
    )


def test_integrated_one_arm_event_consumes_selected_and_physically_renews_once(tmp_path):
    context = fixture(tmp_path, rate=0.2, exact_trials=True)
    before = next(iter(context.runtime.process_engines.values()))
    result = run_stateful_accepted_interval_v12(context, 8.4, dry_run_discard=True)
    after = next(iter(context.runtime.process_engines.values()))
    assert result.disposition == "accepted_event"
    assert len(result.selected_event_ids) == 1
    assert result.accepted_duration_s < 8.4
    assert context.physical_time_s == pytest.approx(
        result.selected_completion_time_s, abs=1.0e-10
    )
    assert context.accepted_opening_m == pytest.approx(
        1e-9 * result.accepted_duration_s / 8.4
    )
    assert after.update_count == before.update_count + 1
    assert after.event_renewal_count == before.event_renewal_count + 1
    assert after.local_process_coordinate_m == pytest.approx(5e-6)
    assert context.trial_cache.audit()["live_entry_count"] == 0
    assert context.trial_cache.audit()["creation_count"] == context.trial_cache.audit()["destruction_count"]


def test_submicrosecond_event_uses_local_bracket_and_absolute_endpoint_tolerance(tmp_path):
    context = fixture(tmp_path, rate=1.0e6, exact_trials=True)
    result = run_stateful_accepted_interval_v12(
        context, 8.4, dry_run_discard=True,
    )
    assert result.disposition == "accepted_event"
    assert result.accepted_duration_s < 1.0e-4
    assert abs(
        context.physical_time_s - result.selected_completion_time_s
    ) <= 1.0e-10
    assert context.provider_lookup_count < 20


def test_endpoint_search_budget_resolves_production_absolute_time_tolerance():
    tolerance = 1.0e-10 / 8.4
    limit = event_endpoint_iteration_limit(tolerance)
    assert limit >= int(np.ceil(np.log2(1.0 / tolerance))) + 32
    assert (2.0 ** -(limit - 32)) <= tolerance


def test_event_endpoint_is_first_event_side_of_discontinuous_mechanics_rate(tmp_path):
    context = fixture(tmp_path, rate=0.1, exact_trials=True)

    def discontinuous_rate(_observation, _candidate, owner):
        return (
            0.3
            if owner.provisional_solved_state.accepted_load_m >= 0.5e-9
            else 0.1
        )

    context.adapter_configuration["directional_rate_adapter"] = discontinuous_rate
    result = run_stateful_accepted_interval_v12(
        context, 8.4, dry_run_discard=True,
    )
    assert result.disposition == "accepted_event"
    assert result.accepted_duration_s == pytest.approx(4.2, abs=1.0e-9)
    assert result.selected_completion_time_s == pytest.approx(
        context.physical_time_s, abs=1.0e-12
    )


@pytest.mark.parametrize("action,count", (
    ("one_arm", 1), ("two_arm", 1), ("coalescence", 3), ("retirement", 2),
))
def test_exact_trial_delta_directly_applies_real_active_front_changes(action, count):
    before = _branched_runtime(count)
    proposal = _proposal(before, action)
    outcome, expected = _exact_outcome(before, proposal)
    actual = apply_exact_trial_delta(before, outcome)
    assert actual.crack_network == expected.crack_network
    assert actual.active_front_ids == expected.active_front_ids
    assert actual.owner_by_front == expected.owner_by_front
    assert actual.registry_fingerprint == expected.registry_fingerprint


@dataclass(frozen=True)
class _RequestFixture:
    cluster_frame: object = None


def test_arbitrary_region_request_maps_four_tips_three_owners(monkeypatch, tmp_path):
    runtime = _branched_runtime(4)
    junctions = sorted(runtime.junctions)
    evidence = {
        junction: CouplingEvidence(junction, 5e-6, 2e-6, 1e-6, 10e-6, 8e-6, False)
        for junction in junctions
    }
    runtime = recompute_process_region_connectivity(runtime, evidence)
    assert len(runtime.process_regions) == 4
    first, second = runtime.active_front_ids[:2]
    owner_a, owner_b = runtime.owner_by_front[first], runtime.owner_by_front[second]
    region_a, region_b = runtime.process_regions[owner_a], runtime.process_regions[owner_b]
    regions = dict(runtime.process_regions)
    regions[owner_a] = ProcessRegionState(
        owner_a, region_a.member_front_ids | region_b.member_front_ids,
        frozenset((junctions[-1],)), region_a.cumulative_process_advance_m,
        region_a.process_engine_id, region_a.source_state_id,
        generation=max(region_a.generation, region_b.generation),
        source_reservoir_id=region_a.source_reservoir_id,
    )
    del regions[owner_b]
    owners = dict(runtime.owner_by_front); owners[second] = owner_a
    engines = dict(runtime.process_engines); del engines[region_b.process_engine_id]
    runtime = replace(
        runtime, process_regions=regions, owner_by_front=owners,
        process_engines=engines,
    )
    assert len(runtime.active_front_ids) == 4
    assert sorted(len(x.member_front_ids) for x in runtime.process_regions.values()) == [1, 1, 2]
    state = SimpleNamespace(
        crack_network=runtime.crack_network,
        mesh=SimpleNamespace(nodes=np.zeros((1, 2)), elems=np.zeros((0, 3), dtype=int)),
        displacement=np.zeros(1), ep_gp=np.zeros((1, 1)), rho_gp=np.ones((1, 1)),
        damage=np.zeros(1), elasticity_D=np.eye(3),
    )
    context = CurrentSourceMultiFrontProductionContextV12(
        args={}, mechanical_configuration={}, accepted_fem_state=state,
        accepted_stress_field=np.ones((3, 1)), runtime=runtime,
        provider_runtime={}, destination_cache_root=tmp_path / "cache",
        output_root=tmp_path / "out", checkpoint_path=tmp_path / "latest",
        candidates=(), mechanics_source_identity="fixture",
        adapter_configuration={"cfg": object()},
    )
    monkeypatch.setattr(
        "arrhenius_fracture.sharp_front_v11_branching._request",
        lambda *args, **kwargs: _RequestFixture(),
    )
    solved = SolvedAcceptedState(
        state, np.ones((3, 1)), {}, 0.0, 0.0, "fixture",
    )
    request = build_arbitrary_region_request_v12(solved, runtime, context)
    frame = request.cluster_frame
    assert frame["mode"] == "arbitrary_process_regions"
    assert set(frame["frame_by_tip"]) == set(runtime.active_front_ids)
    assert len(frame["process_region_frame_by_owner"]) == 3
    assert sum(
        value["frame_kind"] == "unresolved_shared_region"
        for value in frame["process_region_frame_by_owner"].values()
    ) == 1

    omitted = runtime.active_front_ids[0]
    subset = tuple(
        front_id for front_id in runtime.active_front_ids if front_id != omitted
    )
    request = build_arbitrary_region_request_v12(
        solved, runtime, context, active_front_ids=subset,
    )
    subset_frame = request.cluster_frame
    assert set(subset_frame["frame_by_tip"]) == set(subset)
    assert omitted not in {
        front_id
        for owner in subset_frame["process_region_frame_by_owner"].values()
        for front_id in owner["member_front_ids"]
    }


def test_intersecting_cleavage_arm_is_one_atomic_coalescence_transaction():
    before = _branched_runtime(2)
    proposal = _proposal(before, "one_arm")
    nominal = commit_selected_proposal(before, proposal)
    incoming = proposal.front_id
    target = next(
        front_id for front_id in before.active_front_ids if front_id != incoming
    )
    start = before.crack_network.branch(incoming).tip
    end = before.crack_network.branch(target).tip
    arm = TopologyArm(
        proposal.candidate_ids[0], incoming, start, end,
        float(np.linalg.norm(np.asarray(end) - np.asarray(start))), 0.0,
    )
    realized = extend_network_arm(before.crack_network, arm)
    realized = mark_coalesced(realized, incoming, target)
    actual = _apply_realized_cleavage_intersections_v12(
        before, nominal, realized, (arm,), {incoming: target},
    )

    assert actual.active_front_ids == (target,)
    assert actual.cumulative_coalescences == before.cumulative_coalescences + 1
    assert len(actual.transaction_records) == len(before.transaction_records) + 1
    assert actual.scheduler.accepted_transaction_index == (
        before.scheduler.accepted_transaction_index + 1
    )
    record = actual.transaction_records[-1]
    assert record.action_type == "one_arm"
    assert record.coalescence_target_front_id == target
    assert incoming in record.retired_front_ids
    assert record.realized_endpoints_m == (end,)
    assert record.post_active_front_count == 1
    assert actual.crack_network.branch(incoming).status == "merged"
    actual.validate()


def test_unclipped_binary_arm_realization_matches_nominal_atomic_topology():
    before = _branched_runtime(1)
    proposal = _proposal(before, "two_arm")
    nominal = commit_selected_proposal(before, proposal)
    parent = before.crack_network.branch(proposal.front_id)
    child_by_candidate = {
        nominal.crack_network.branch(front_id).local_state["candidate_id"]: front_id
        for front_id in nominal.active_front_ids
    }
    child_ids = set(child_by_candidate.values())
    branches = []
    for branch in nominal.crack_network.branches:
        if branch.branch_id not in child_ids:
            branches.append(branch)
            continue
        local_state = dict(branch.local_state)
        local_state.pop("committed_edges", None)
        branches.append(replace(
            branch,
            path=(parent.tip,),
            orientation_history_rad=(parent.current_orientation_rad,),
            local_state=local_state,
        ))
    realized = replace(
        nominal.crack_network,
        branches=tuple(branches),
        geometry_generation=before.crack_network.geometry_generation,
    )
    arms = []
    for candidate_id, endpoint in zip(
        proposal.candidate_ids, proposal.end_points_m
    ):
        arm = TopologyArm(
            candidate_id, child_by_candidate[candidate_id], parent.tip, endpoint,
            float(np.linalg.norm(np.asarray(endpoint) - np.asarray(parent.tip))),
            0.0,
        )
        realized = extend_network_arm(realized, arm)
        arms.append(arm)
    actual = _apply_realized_cleavage_intersections_v12(
        before, nominal, realized, tuple(arms), {},
    )
    assert actual is nominal
    assert actual.topology_fingerprint == nominal.topology_fingerprint


def test_pure_hook_detects_physical_engine_mutation(tmp_path):
    context = fixture(tmp_path)
    context.actual_engines["engine:root"] = SimpleNamespace(value=1)
    def mutate():
        context.actual_engines["engine:root"].value = 2
    with pytest.raises(StatefulProductionInterlock, match="pure hook mutated"):
        context.invoke("bad_pure_hook", mutate, pure=True)


@pytest.mark.parametrize("member", range(1, 7))
def test_atomic_failure_after_each_output_category_is_unpublished(member, tmp_path):
    context = fixture(tmp_path)
    with pytest.raises(StatefulProductionInterlock, match="injected"):
        run_stateful_accepted_interval_v12(
            context, 8.4, failure_stage=f"output_staging_after_{member}",
        )
    writer = AtomicIntervalTransactionWriterV12(tmp_path / "production-output")
    assert writer.read_latest_committed() is None


def test_archive_cli_reports_closed_evidence_without_retrying_executor(tmp_path):
    from arrhenius_fracture.sharp_front_current_source_multifront_v12 import (
        v6_4_integrated_interval_dry_run,
    )
    base = Path("/Volumes/Data/Data/Nanopillar_calculation")
    n1 = base / "PF-fracture-fatigue_current_source_branching_integrated_envelope_v5_4_step1_migrations_20260901T130000Z_final/control_max1/step0000001_migrated_v5_4.json"
    n2 = base / "PF-fracture-fatigue_current_source_branching_completion_v5_4_2_20260901T204630Z/theta40_v5_4_2_enabled_max2_seed3621/checkpoint/latest.json"
    result = v6_4_integrated_interval_dry_run((
        "--v6-4-integrated-interval-dry-run", "--branching-mode", "mechanistic",
        "--shared-region-branching", "experimental",
        "--v5-4-2-n1-checkpoint", str(n1), "--v5-4-2-n2-checkpoint", str(n2),
        "--fresh-output-destination", str(tmp_path / "output"),
        "--fresh-cache-destination", str(tmp_path / "cache"),
    ))
    assert result["integrated_interval_transaction"] == "FAIL_CLOSED"
    assert "CLOSED_EVIDENCE_LIMITATION" in result["cases"]["native_N1"]["reason"]
    assert "no archive replay attempted" in result["cases"]["native_N2"]["reason"]
    assert not result["cases"]["native_N2"]["provider_lookup_attempted"]
    assert result["provider_solve_count"] == result["mechanics_solve_count"] == 0
    assert result["workers_started"] == 0
    assert not (tmp_path / "output").exists()
