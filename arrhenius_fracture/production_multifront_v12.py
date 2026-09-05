"""Source-complete generic V12 production orchestration.

No worker is started by importing this module.  The driver is dependency-
injected at the physical boundaries, while registry evolution, correlated
proposal formation, global scheduling, topology transactions, renewal,
output, and checkpoint sequencing are owned here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import pickle
from typing import Any, Callable, Mapping, Sequence

from .general_multifront_v12 import (
    AcceptedIntervalResult, FrontCandidateObservation, FrontRuntimeState,
    MultiFrontRuntimeState, ProcessEngineState, TopologyProposal,
    advance_accepted_interval, canonical_hash, commit_selected_proposal,
    form_correlated_topology_proposals,
)
from .live_topology_kernel_v12 import DynamicExactTopologyProviderV12


SCHEMA = "v12.general-multifront-production-driver/1"
BOUNDARY = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"


@dataclass(frozen=True)
class AdaptedAcceptedState:
    fem_state: Any
    adaptation_record: Any


@dataclass(frozen=True)
class SolvedAcceptedState:
    fem_state: Any
    sigma_gp: Any
    measurement: Mapping[str, Any]
    accepted_load_m: float
    physical_time_s: float
    mechanics_source_identity: str


@dataclass(frozen=True)
class DirectionalObservationBatch:
    observations: Sequence[FrontCandidateObservation]
    rates_by_front_and_candidate: Mapping[str, Mapping[str, float]]
    endpoints_by_front_and_candidate: Mapping[
        str, Mapping[str, tuple[float, float]]
    ]
    local_mechanics_records: Sequence[Mapping[str, Any]]

    def validate(self, runtime: MultiFrontRuntimeState) -> None:
        expected = {
            (front_id, candidate_id)
            for front_id, front in runtime.front_runtimes.items()
            for candidate_id in front.mechanically_active_candidate_ids
        }
        observed = {(row.front_id, row.candidate_id) for row in self.observations}
        rates = {
            (front_id, candidate_id)
            for front_id, values in self.rates_by_front_and_candidate.items()
            for candidate_id in values
        }
        endpoints = {
            (front_id, candidate_id)
            for front_id, values in self.endpoints_by_front_and_candidate.items()
            for candidate_id in values
        }
        if not expected or observed != expected or rates != expected or endpoints != expected:
            raise ValueError(
                "directional batch must cover every mechanically active front/candidate exactly once"
            )
        if not self.local_mechanics_records:
            raise ValueError("directional batch requires local mechanics qualification")


@dataclass(frozen=True)
class PreSolveInventory:
    accepted_state_id: str
    accepted_fem_state_reference_sha256: str
    topology_fingerprint: str
    active_front_ids: tuple[str, ...]
    owner_ids: tuple[str, ...]
    candidate_ids_by_front: Mapping[str, tuple[str, ...]]
    mechanically_active_candidate_ids_by_front: Mapping[str, tuple[str, ...]]
    dormant_candidate_ids_by_front: Mapping[str, tuple[str, ...]]
    process_engine_ids: tuple[str, ...]
    junction_ids: tuple[str, ...]
    reservoir_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_state_id": self.accepted_state_id,
            "accepted_fem_state_reference_sha256": self.accepted_fem_state_reference_sha256,
            "topology_fingerprint": self.topology_fingerprint,
            "active_front_ids": list(self.active_front_ids),
            "owner_ids": list(self.owner_ids),
            "candidate_ids_by_front": {
                key: list(value) for key, value in self.candidate_ids_by_front.items()
            },
            "mechanically_active_candidate_ids_by_front": {
                key: list(value)
                for key, value in self.mechanically_active_candidate_ids_by_front.items()
            },
            "dormant_candidate_ids_by_front": {
                key: list(value) for key, value in self.dormant_candidate_ids_by_front.items()
            },
            "process_engine_ids": list(self.process_engine_ids),
            "junction_ids": list(self.junction_ids),
            "reservoir_ids": list(self.reservoir_ids),
        }


def prepare_pre_solve_inventory(
    runtime: MultiFrontRuntimeState, *, accepted_fem_state_reference: Mapping[str, Any],
) -> PreSolveInventory:
    """Reach the production pre-solve boundary without provider lookup."""
    runtime.validate()
    return PreSolveInventory(
        accepted_state_id=runtime.accepted_state_id,
        accepted_fem_state_reference_sha256=canonical_hash(
            dict(accepted_fem_state_reference)
        ),
        topology_fingerprint=runtime.topology_fingerprint,
        active_front_ids=runtime.active_front_ids,
        owner_ids=tuple(sorted(runtime.process_regions)),
        candidate_ids_by_front={
            key: value.candidate_ids for key, value in runtime.front_runtimes.items()
        },
        mechanically_active_candidate_ids_by_front={
            key: value.mechanically_active_candidate_ids
            for key, value in runtime.front_runtimes.items()
        },
        dormant_candidate_ids_by_front={
            key: value.dormant_candidate_ids
            for key, value in runtime.front_runtimes.items()
        },
        process_engine_ids=tuple(sorted(runtime.process_engines)),
        junction_ids=tuple(sorted(runtime.junctions)),
        reservoir_ids=tuple(sorted(runtime.reservoirs)),
    )


@dataclass(frozen=True)
class ProductionHooks:
    adapt_accepted_mesh: Callable[
        [Any, Mapping[str, tuple[str, ...]]], AdaptedAcceptedState
    ]
    solve_accepted_fem: Callable[..., SolvedAcceptedState]
    build_exact_topology_request: Callable[
        [SolvedAcceptedState, MultiFrontRuntimeState], Any
    ]
    extract_directional_observations: Callable[
        [Mapping[str, Any], MultiFrontRuntimeState], DirectionalObservationBatch,
    ]
    evolve_process_engine: Callable[
        [ProcessEngineState, FrontCandidateObservation, float], ProcessEngineState
    ]
    advance_front_clock: Callable[
        [FrontRuntimeState, Sequence[FrontCandidateObservation], float], FrontRuntimeState
    ]
    trial_topology_proposal: Callable[[Any, TopologyProposal], "ProposalTrialOutcome"]
    write_output: Callable[[MultiFrontRuntimeState, AcceptedIntervalResult], None]
    write_checkpoint: Callable[[MultiFrontRuntimeState, Any], None]
    # V6.3 owns these transaction boundaries explicitly.  Defaults preserve
    # the source compatibility of V6.2 callers while the reviewed factory
    # supplies every one of them.
    context: Any | None = None
    preview_competition: Callable[..., Any] | None = None
    finalize_competition: Callable[..., Any] | None = None
    renew_selected_event: Callable[..., Any] | None = None
    validate_process_interval: Callable[..., Any] | None = None
    destroy_trial_cache: Callable[..., Any] | None = None


@dataclass(frozen=True)
class ProductionIntervalResult:
    runtime: MultiFrontRuntimeState
    accepted_fem_state: Any
    provider_result: Mapping[str, Any]
    accepted_interval: AcceptedIntervalResult


@dataclass(frozen=True)
class ProposalTrialOutcome:
    proposal: TopologyProposal
    accepted: bool
    trial_fem_state: Any
    reason: str
    exact_realized_crack_network: Any | None = None
    exact_topology_transaction: Mapping[str, Any] | None = None
    created_front_ids: tuple[str, ...] = ()
    retired_front_ids: tuple[str, ...] = ()
    junction_id: str | None = None
    clipped_or_coalesced_disposition: str | None = None
    realized_endpoints_m: tuple[tuple[float, float], ...] = ()
    realized_arm_lengths_m: tuple[float, ...] = ()
    wake_mutation: Mapping[str, Any] | None = None
    stored_energy_release_J_per_m: float = 0.0
    stored_energy_cost_J_per_m: float = 0.0
    topology_fingerprint: str | None = None
    geometry_fingerprint: str | None = None
    accepted_state_fingerprint: str | None = None
    exact_delta: "ExactTrialDeltaV12 | None" = None
    exact_post_sigma_gp: Any | None = None

    def require_exact_accepted_trial(self) -> None:
        if not self.accepted:
            return
        if self.exact_realized_crack_network is None or self.trial_fem_state is None:
            raise RuntimeError("accepted topology outcome lacks the exact trial state")
        topology = hashlib.sha256(
            self.exact_realized_crack_network.to_json().encode()
        ).hexdigest()
        fem_topology = hashlib.sha256(
            self.trial_fem_state.crack_network.to_json().encode()
        ).hexdigest()
        if topology != fem_topology or self.topology_fingerprint != topology:
            raise RuntimeError("accepted trial topology fingerprints do not agree")
        if len(self.realized_endpoints_m) != len(self.realized_arm_lengths_m):
            raise RuntimeError("accepted trial endpoint and arm-length inventories differ")


@dataclass(frozen=True)
class ExactTrialDeltaV12:
    """Complete accepted registry delta produced by one isolated exact trial."""
    pre_crack_network: Any
    post_crack_network: Any
    continued_front_ids: tuple[str, ...]
    created_front_ids: tuple[str, ...]
    retired_front_ids: tuple[str, ...]
    coalescence_target_front_id: str | None
    junctions_after: Mapping[str, Any]
    owner_by_front_after: Mapping[str, str]
    process_regions_after: Mapping[str, Any]
    process_engines_after: Mapping[str, ProcessEngineState]
    front_runtimes_after: Mapping[str, FrontRuntimeState]
    reservoirs_after: Mapping[str, Any]
    scheduler_after: Any
    cumulative_branch_births_after: int
    cumulative_coalescences_after: int
    cumulative_retirements_after: int
    transaction_records_after: tuple[Any, ...]
    output_counters_after: Mapping[str, int]
    termination_reason_after: str | None
    policy_bound_after: bool
    realized_endpoints_m: tuple[tuple[float, float], ...]
    realized_lengths_m: tuple[float, ...]
    wake_mutation: Mapping[str, Any]
    released_energy_J_per_m: float
    dissipative_cost_J_per_m: float

    def validate(self, runtime: MultiFrontRuntimeState) -> None:
        if self.pre_crack_network != runtime.crack_network:
            raise RuntimeError("exact trial delta pre-network is stale")
        active = set(self.post_crack_network.active_tip_ids)
        if active != set(self.front_runtimes_after) or active != set(self.owner_by_front_after):
            raise RuntimeError("exact trial delta active-front registries disagree")
        if set(self.created_front_ids).intersection(self.retired_front_ids):
            raise RuntimeError("exact trial delta creates and retires the same front")
        if len(self.realized_endpoints_m) != len(self.realized_lengths_m):
            raise RuntimeError("exact trial delta endpoint/length cardinality differs")
        region_engines = {
            value.process_engine_id for value in self.process_regions_after.values()
        }
        if region_engines != set(self.process_engines_after):
            raise RuntimeError("exact trial delta owner/engine registries disagree")
        post_branch_ids = {item.branch_id for item in self.post_crack_network.branches}
        if any(
            item.parent_branch_id not in post_branch_ids
            or not set(item.child_branch_ids).issubset(post_branch_ids)
            for item in self.junctions_after.values()
        ):
            raise RuntimeError("exact trial delta junction references a nonexistent branch")
        before = set(self.pre_crack_network.active_tip_ids)
        expected_created = active - before
        expected_retired = before - active
        if set(self.created_front_ids) != expected_created:
            raise RuntimeError("exact trial delta created-front inventory is incomplete")
        if set(self.retired_front_ids) != expected_retired:
            raise RuntimeError("exact trial delta retired-front inventory is incomplete")
        if set(self.continued_front_ids) != before.intersection(active):
            raise RuntimeError("exact trial delta continued-front inventory is incomplete")


def apply_exact_trial_delta(
    runtime: MultiFrontRuntimeState, outcome: ProposalTrialOutcome,
) -> MultiFrontRuntimeState:
    """Apply a complete exact delta directly, with no nominal reconstruction."""
    if not outcome.accepted or outcome.exact_delta is None:
        raise RuntimeError("integrated event commit requires a complete exact trial delta")
    delta = outcome.exact_delta
    delta.validate(runtime)
    direct = replace(
        runtime, crack_network=delta.post_crack_network,
        front_runtimes=dict(delta.front_runtimes_after),
        owner_by_front=dict(delta.owner_by_front_after),
        process_regions=dict(delta.process_regions_after),
        process_engines=dict(delta.process_engines_after),
        junctions=dict(delta.junctions_after), reservoirs=dict(delta.reservoirs_after),
        scheduler=delta.scheduler_after,
        cumulative_branch_births=delta.cumulative_branch_births_after,
        cumulative_coalescences=delta.cumulative_coalescences_after,
        cumulative_retirements=delta.cumulative_retirements_after,
        transaction_records=tuple(delta.transaction_records_after),
        output_counters=dict(delta.output_counters_after),
        termination_reason=delta.termination_reason_after,
        policy_bound=delta.policy_bound_after,
    )
    direct.validate()
    return direct


def accepted_fem_state_fingerprint(state: Any) -> str:
    """Hash the canonical serialized form of one accepted FEM state.

    ``LiveFEMTopologyState`` freezes arrays and mapping containers on
    construction.  Some producer-side objects therefore have a different
    first pickle representation from the same object after checkpoint load
    (notably writable array provenance and pickle-safe frozen mappings).  A
    checkpoint identity must describe the restart object, not those transient
    in-memory representation details.  Normalize through exactly one pickle
    reconstruction before hashing so producer and loader bind the same bytes.
    """
    normalized = pickle.loads(pickle.dumps(state, protocol=5))
    return hashlib.sha256(pickle.dumps(normalized, protocol=5)).hexdigest()


def rollover_accepted_state(
    runtime: MultiFrontRuntimeState, accepted_fem_state: Any,
) -> MultiFrontRuntimeState:
    """Derive fresh accepted/stress identities from the committed FEM object."""
    topology = hashlib.sha256(
        accepted_fem_state.crack_network.to_json().encode()
    ).hexdigest()
    if topology != runtime.topology_fingerprint:
        raise RuntimeError("accepted FEM and V12 topology differ during state rollover")
    fem_hash = accepted_fem_state_fingerprint(accepted_fem_state)
    stress_hash = hashlib.sha256(pickle.dumps({
        "displacement": accepted_fem_state.displacement,
        "ep_gp": accepted_fem_state.ep_gp,
        "rho_gp": accepted_fem_state.rho_gp,
        "damage": accepted_fem_state.damage,
        "elasticity_D": accepted_fem_state.elasticity_D,
        "topology": topology,
    }, protocol=5)).hexdigest()
    return replace(
        runtime,
        accepted_state_id=f"accepted:{fem_hash}",
        stress_field_state_id=f"stress:{stress_hash}",
        compatibility_provenance={
            **dict(runtime.compatibility_provenance or {}),
            "accepted_fem_state_sha256": fem_hash,
            "accepted_topology_sha256": topology,
        },
    )


def commit_exact_trial_outcome(
    runtime: MultiFrontRuntimeState, outcome: ProposalTrialOutcome,
) -> MultiFrontRuntimeState:
    """Commit registries around the exact accepted trial, never rebuilt geometry."""
    outcome.require_exact_accepted_trial()
    if not outcome.accepted:
        return runtime
    proposal = replace(
        outcome.proposal,
        end_points_m=tuple(outcome.realized_endpoints_m),
    )
    generic = commit_selected_proposal(runtime, proposal)
    if generic.policy_bound:
        return generic
    exact = outcome.exact_realized_crack_network
    if proposal.action_type == "two_arm" and set(exact.active_tip_ids) != set(generic.active_front_ids):
        before_active = set(runtime.active_front_ids)
        generic_created = set(generic.active_front_ids) - before_active
        exact_created = set(exact.active_tip_ids) - before_active
        if len(generic_created) != 2 or len(exact_created) != 2:
            raise RuntimeError("accepted binary trial has inconsistent created-front cardinality")
        def by_candidate(network, ids):
            result = {}
            for branch_id in ids:
                candidate = str(network.branch(branch_id).local_state.get("candidate_id"))
                if candidate in result:
                    raise RuntimeError("binary accepted trial has duplicate daughter candidate identity")
                result[candidate] = branch_id
            return result
        generic_by_candidate = by_candidate(generic.crack_network, generic_created)
        exact_by_candidate = by_candidate(exact, exact_created)
        if set(generic_by_candidate) != set(exact_by_candidate):
            raise RuntimeError("accepted binary trial daughter candidates differ from V12 proposal")
        remap = {
            generic_by_candidate[key]: exact_by_candidate[key]
            for key in sorted(generic_by_candidate)
        }
        front_runtimes = {}
        for key, value in generic.front_runtimes.items():
            new = remap.get(key, key)
            front_runtimes[new] = replace(value, front_id=new)
        owner_by_front = {
            remap.get(key, key): value for key, value in generic.owner_by_front.items()
        }
        process_regions = {
            key: replace(
                value,
                member_front_ids=frozenset(remap.get(x, x) for x in value.member_front_ids),
            ) for key, value in generic.process_regions.items()
        }
        junctions = dict(generic.junctions)
        created_key = generic.transaction_records[-1].created_junction_id
        if created_key is not None:
            item = junctions.pop(created_key)
            new_key = outcome.junction_id or created_key
            junctions[new_key] = replace(
                item, junction_id=new_key,
                child_branch_ids=tuple(remap.get(x, x) for x in item.child_branch_ids),
            )
            process_regions = {
                key: replace(
                    value,
                    unresolved_junction_ids=frozenset(
                        new_key if x == created_key else x
                        for x in value.unresolved_junction_ids
                    ),
                ) for key, value in process_regions.items()
            }
        generic = replace(
            generic, crack_network=exact, front_runtimes=front_runtimes,
            owner_by_front=owner_by_front, process_regions=process_regions,
            junctions=junctions,
        )
    if set(exact.active_tip_ids) != set(generic.active_front_ids):
        raise RuntimeError("accepted trial and V12 registries have different active fronts")
    committed = replace(generic, crack_network=exact)
    record = committed.transaction_records[-1]
    exact_topology = committed.topology_fingerprint
    fem_topology = hashlib.sha256(
        outcome.trial_fem_state.crack_network.to_json().encode()
    ).hexdigest()
    record = replace(
        record,
        created_front_ids=tuple(outcome.created_front_ids or record.created_front_ids),
        retired_front_ids=tuple(outcome.retired_front_ids or record.retired_front_ids),
        created_junction_id=outcome.junction_id or record.created_junction_id,
        realized_lengths_m=tuple(outcome.realized_arm_lengths_m),
        realized_endpoints_m=tuple(outcome.realized_endpoints_m),
        clipped_or_coalesced_disposition=outcome.clipped_or_coalesced_disposition,
        wake_mutation=dict(outcome.wake_mutation or {}),
        stored_energy_release_J_per_m=outcome.stored_energy_release_J_per_m,
        stored_energy_cost_J_per_m=outcome.stored_energy_cost_J_per_m,
        post_topology_fingerprint=exact_topology,
        accepted_fem_topology_fingerprint=fem_topology,
        geometry_fingerprint=outcome.geometry_fingerprint or exact_topology,
        exact_accepted_trial_fingerprint=(
            outcome.accepted_state_fingerprint
            or accepted_fem_state_fingerprint(outcome.trial_fem_state)
        ),
    )
    committed = replace(
        committed,
        transaction_records=committed.transaction_records[:-1] + (record,),
    )
    committed = rollover_accepted_state(committed, outcome.trial_fem_state)
    record = replace(
        committed.transaction_records[-1],
        accepted_state_id_after=committed.accepted_state_id,
    )
    return replace(
        committed,
        transaction_records=committed.transaction_records[:-1] + (record,),
    )


class GenericMultiFrontProductionDriver:
    def __init__(self, hooks: ProductionHooks, provider: DynamicExactTopologyProviderV12):
        self.hooks = hooks
        self.provider = provider

    def run_accepted_interval(
        self, runtime: MultiFrontRuntimeState, accepted_fem_state: Any, *, duration_s: float,
    ) -> ProductionIntervalResult:
        """Execute the production order through one accepted topology transaction."""
        candidate_inventory = {
            key: value.mechanically_active_candidate_ids
            for key, value in runtime.front_runtimes.items()
        }
        adapted = self.hooks.adapt_accepted_mesh(accepted_fem_state, candidate_inventory)
        solved = self.hooks.solve_accepted_fem(adapted)
        request = self.hooks.build_exact_topology_request(solved, runtime)
        provider_result = self.provider.evaluate(request)
        observations, actions_by_front, endpoints_by_front = (
            self.hooks.extract_directional_observations(provider_result, runtime)
        )
        proposals: list[TopologyProposal] = []
        for front_id in runtime.active_front_ids:
            proposals.extend(form_correlated_topology_proposals(
                front_id=front_id,
                owner_id=runtime.owner_by_front[front_id],
                action_proposals=tuple(actions_by_front.get(front_id, ())),
                end_points_by_candidate=endpoints_by_front.get(front_id, {}),
            ))
        # Every trial starts from the identical accepted solved state.  Only a
        # complete one- or two-arm proposal is trialled; members of an atomic
        # pair are never exposed as independent global clocks.
        trials = tuple(
            self.hooks.trial_topology_proposal(solved, proposal)
            for proposal in proposals
        )
        if any(outcome.proposal != proposal for outcome, proposal in zip(trials, proposals)):
            raise RuntimeError("topology trial changed proposal identity")
        admissible = tuple(outcome.proposal for outcome in trials if outcome.accepted)
        interval = advance_accepted_interval(
            runtime, observations, admissible, duration_s=duration_s,
            evolve_engine=self.hooks.evolve_process_engine,
            advance_front_clock=self.hooks.advance_front_clock,
        )
        committed = interval.state
        accepted_after = solved
        if interval.selected_proposal is not None:
            selected_outcome = next(
                outcome for outcome in trials
                if outcome.accepted and outcome.proposal == interval.selected_proposal
            )
            committed = commit_exact_trial_outcome(committed, selected_outcome)
            accepted_after = selected_outcome.trial_fem_state
        else:
            committed = rollover_accepted_state(committed, accepted_after)
        self.hooks.write_output(committed, interval)
        self.hooks.write_checkpoint(committed, accepted_after)
        return ProductionIntervalResult(committed, accepted_after, provider_result, interval)


def production_call_graph() -> tuple[dict[str, Any], ...]:
    """Machine-readable source trace for the real generic interval order."""
    return (
        {"order": 1, "stage": "generic_entrypoint", "symbol": "sharp_front_current_source_multifront_v12.run_production_interval"},
        {"order": 2, "stage": "adaptive_mesh", "symbol": "ProductionHooks.adapt_accepted_mesh"},
        {"order": 3, "stage": "accepted_fem_solve", "symbol": "ProductionHooks.solve_accepted_fem"},
        {"order": 4, "stage": "exact_topology_provider", "symbol": "DynamicExactTopologyProviderV12.evaluate"},
        {"order": 5, "stage": "directional_local_J_and_tensor_probes", "symbol": "ProductionHooks.extract_directional_observations"},
        {"order": 6, "stage": "correlated_same_tip_proposals", "symbol": "form_correlated_topology_proposals"},
        {"order": 7, "stage": "isolated_atomic_energy_trials", "symbol": "ProductionHooks.trial_topology_proposal"},
        {"order": 8, "stage": "one_process_update_per_owner_and_global_scheduler", "symbol": "advance_accepted_interval"},
        {"order": 9, "stage": "atomic_topology_transaction_and_moving_frame_renewal", "symbol": "commit_selected_proposal"},
        {"order": 10, "stage": "canonical_output", "symbol": "ProductionHooks.write_output"},
        {"order": 11, "stage": "production_checkpoint", "symbol": "ProductionHooks.write_checkpoint"},
    )


def current_source_symbol_bindings() -> dict[str, tuple[str, ...]]:
    """Bind each injected boundary to the reviewed current production source."""
    return {
        "generic_entrypoint": (
            "arrhenius_fracture.sharp_front_current_source_multifront_v12.run_production_interval",
        ),
        "adaptive_mesh": (
            "arrhenius_fracture.adaptive_multitip_mesh_v11.adapt_accepted_state_for_trials",
        ),
        "accepted_fem_solve": (
            "arrhenius_fracture.fem.assemble_mechanics",
            "arrhenius_fracture.fem.solve_dirichlet",
            "arrhenius_fracture.plasticity.update_plasticity",
        ),
        "exact_topology_provider": (
            "arrhenius_fracture.live_topology_kernel_v12.DynamicExactTopologyProviderV12.evaluate",
            "arrhenius_fracture.live_topology_kernel_v11.evaluate_exact_topology",
        ),
        "directional_local_J": (
            "arrhenius_fracture.live_topology_kernel_v11.compute_J_integral",
            "arrhenius_fracture.tip_directional_observation_v11.observations_from_provider",
        ),
        "tensor_probe": (
            "arrhenius_fracture.anisotropic_emission_v10174.bind_explicit_accepted_tensor_drive",
        ),
        "atomic_energy_trial": (
            "arrhenius_fracture.topology_transaction_v11.execute_topology_trial",
        ),
        "process_engine_update": (
            "arrhenius_fracture.process_update_semantics_v11.require_full_accepted_interval_consumption",
        ),
        "scheduler": (
            "arrhenius_fracture.general_multifront_v12.form_correlated_topology_proposals",
            "arrhenius_fracture.general_multifront_v12.advance_accepted_interval",
        ),
        "moving_frame_renewal": (
            "arrhenius_fracture.tip_directional_observation_v11.apply_post_interval_event_renewal",
        ),
        "output": (
            "arrhenius_fracture.multifront_output_v12.front_observation_row",
            "arrhenius_fracture.multifront_output_v12.owner_row",
        ),
        "checkpoint": (
            "arrhenius_fracture.multifront_checkpoint_v12.write_multifront_checkpoint",
        ),
    }


__all__ = [
    "AdaptedAcceptedState", "BOUNDARY", "DirectionalObservationBatch",
    "ExactTrialDeltaV12", "GenericMultiFrontProductionDriver", "PreSolveInventory",
    "ProductionHooks", "ProductionIntervalResult", "ProposalTrialOutcome", "SCHEMA",
    "SolvedAcceptedState", "apply_exact_trial_delta",
    "accepted_fem_state_fingerprint", "commit_exact_trial_outcome",
    "current_source_symbol_bindings", "prepare_pre_solve_inventory", "production_call_graph",
    "rollover_accepted_state",
]
