"""State-owned preview/advance/commit scheduler for physical events.

This module contains adapters around the existing event owners.  It does not
define kinetic parameters, thresholds, topology criteria, or RNG draws.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import json
import math
import pickle
import time
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


SCHEMA = "unified-2d.global-physical-event-scheduler/1"
TIME_TOLERANCE_S = 1.0e-10
GROWTH_PACKET_SAFETY = 0.5


def _sha(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _accepted_fingerprint(state: Any) -> str:
    from .topology_transaction_v11 import complete_accepted_state_fingerprint
    return complete_accepted_state_fingerprint(state)


def _topology_fingerprint(state: Any) -> str:
    return _sha((state.crack_network, state.void_state, state.event_counters))


def mechanical_provider_dependency_identity(state: Any) -> str:
    """Identity of accepted geometry/operators owned by mechanics providers."""
    return _sha((
        np.asarray(state.mesh.nodes), np.asarray(state.mesh.elems),
        getattr(state, "boundary", None), state.crack_network,
        getattr(state, "cohesive_network", None),
        None if not hasattr(state, "damage") else np.asarray(state.damage),
        None if not hasattr(state, "elasticity_D") else np.asarray(state.elasticity_D),
        getattr(state, "material", None), getattr(state, "sharp_wake_model_id", None),
    ))


def transport_provider_dependency_identity(state: Any) -> str:
    """Identity of accepted connectivity/operators owned by transport providers."""
    transport = getattr(state, "vacancy_transport_state_v2", None)
    transport_mesh = None if transport is None else transport.transport_mesh
    return _sha((
        np.asarray(state.mesh.nodes), np.asarray(state.mesh.elems),
        None if transport_mesh is None else np.asarray(transport_mesh.nodes),
        None if transport_mesh is None else np.asarray(transport_mesh.elems),
        None if transport is None else getattr(
            transport, "surface_registry_fingerprint", None),
        None if transport is None else getattr(
            transport, "parameter_row_fingerprint", None),
    ))


def numerical_provider_dependency_identity(state: Any) -> str:
    """Combined identity of accepted owners used for provider invalidation.

    This is intentionally narrower than the complete/topology fingerprints used
    to bind event previews. History variables and stochastic clocks change the
    current forcing, but do not reconstruct a basis unless an operator owner
    below changes.
    """
    return _sha((
        mechanical_provider_dependency_identity(state),
        transport_provider_dependency_identity(state),
    ))


def _selected_event_identities(scheduler: Any) -> tuple[Mapping[str, Any], ...]:
    if scheduler is None:
        return ()
    identities = []
    for row in scheduler.get("events", ()):
        event_type = row.get("selected_event")
        packet = bool(row.get("growth_packet_global_scheduler_boundary", False))
        if event_type is None and not packet:
            continue
        if event_type is None:
            event_type = "GROWTH_PACKET_BOUNDARY"
        owner = tuple(row.get("selected_owner") or ())
        preview = next((item for item in row.get("eligible_previews", ())
                        if item.get("event_type") == event_type
                        and (not owner or tuple((item.get("owner_kind"),
                                                item.get("owner_id"))) == owner)), {})
        identities.append(MappingProxyType({
            "event_type": event_type,
            "owner_kind": owner[0] if len(owner) > 0 else preview.get("owner_kind"),
            "owner_id": owner[1] if len(owner) > 1 else preview.get("owner_id"),
            "event_ids": tuple(preview.get("event_ids", ())),
            "candidate_ids": tuple(preview.get("candidate_ids", ())),
            "decision": "NUMERICAL_BOUNDARY" if packet and not owner else "ACCEPTED",
            "elapsed_time_s": float(row.get("elapsed_time_s", 0.0)),
            "growth_packet_boundary": packet,
        }))
    return tuple(identities)


def _terminal_energy_decision(audit: Mapping[str, Any] | None) -> Any:
    """Return the terminal physical energy record from a composed audit."""
    current = audit
    while current is not None:
        decision = current.get("energy_decision")
        if decision is not None:
            return decision
        current = current.get("remainder")
    return None


def _provider_manager_snapshot(manager: Any) -> Any:
    if manager is None:
        return None
    if hasattr(manager, "snapshot_binding"):
        return manager.snapshot_binding()
    names = ("mechanics", "transport", "dependency_identity", "generation",
             "transition_count", "retired", "audit")
    return {name: copy.copy(getattr(manager, name)) for name in names
            if hasattr(manager, name)}


def _restore_provider_manager(manager: Any, snapshot: Any) -> None:
    if manager is None or snapshot is None:
        return
    if hasattr(manager, "restore_binding"):
        manager.restore_binding(snapshot)
        return
    for name, value in snapshot.items():
        setattr(manager, name, value)


def _rng_fingerprint(state: Any) -> str:
    void_rng = None if state.void_state is None else state.void_state.rng_state
    direct_rng = ()
    if state.void_state is not None:
        direct = getattr(state.void_state, "dislocation_interaction_state", None)
        if direct is not None:
            direct_rng = tuple(
                (site.site_id, site.interaction_rng_identity) for site in direct.sites
            )
    site_rng = ()
    transport = state.vacancy_transport_state_v2
    if transport is not None:
        site_rng = tuple((site.site_id, site.rng_fingerprint) for site in transport.site_states)
    return _sha((state.rng_state, void_rng, site_rng, direct_rng))


@dataclass(frozen=True)
class PhysicalEventPreview:
    event_type: str
    owner_kind: str
    owner_id: str
    crossing_time_s: float
    accepted_state_fingerprint: str
    source_fingerprint: str
    topology_fingerprint: str
    event_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    threshold_identity: tuple[tuple[str, Any], ...]
    rng_fingerprint: str
    commit_payload: Any
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if not self.event_type or not self.owner_kind or not self.owner_id:
            raise ValueError("physical event preview requires complete ownership")
        time = float(self.crossing_time_s)
        if math.isnan(time) or time < 0.0:
            raise ValueError("crossing time must be nonnegative or infinity")
        if not all((self.accepted_state_fingerprint, self.source_fingerprint,
                    self.topology_fingerprint, self.rng_fingerprint)):
            raise ValueError("physical event preview fingerprints are incomplete")
        if len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("physical event preview contains duplicate event IDs")

    @property
    def eligible(self) -> bool:
        return math.isfinite(self.crossing_time_s)


@dataclass(frozen=True)
class AcceptedIntervalContinuationV1:
    """Restart owner for a partially consumed outer hold interval.

    The accepted state stored beside this object is always at a complete
    global-scheduler transaction boundary.  ``mechanical_endpoint_state`` is
    the original accepted endpoint solve for the outer interval; retaining it
    prevents a restart from applying its plastic increment a second time.
    """
    mechanical_endpoint_state: Any
    remaining_duration_s: float
    requested_duration_s: float
    macro_start_time_s: float
    load_history: Mapping[str, Any]
    driver_context: Mapping[str, Any]
    selected_events: tuple[str, ...] = ()
    completed_transaction_count: int = 0
    all_owners_previewed: bool = True
    schema: str = "unified-2d.accepted-interval-continuation/1"

    def __post_init__(self) -> None:
        if (not math.isfinite(float(self.remaining_duration_s))
                or self.remaining_duration_s < 0.0
                or not math.isfinite(float(self.requested_duration_s))
                or self.requested_duration_s <= 0.0):
            raise ValueError("interval continuation durations are invalid")
        if self.remaining_duration_s > self.requested_duration_s:
            raise ValueError("interval continuation exceeds its requested duration")

    def audit_payload(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "remaining_duration_s": self.remaining_duration_s,
            "requested_duration_s": self.requested_duration_s,
            "macro_start_time_s": self.macro_start_time_s,
            "selected_events": list(self.selected_events),
            "completed_transaction_count": self.completed_transaction_count,
            "all_owners_previewed": self.all_owners_previewed,
            "mechanical_endpoint_fingerprint": _accepted_fingerprint(
                self.mechanical_endpoint_state),
        }

def _validate_original(state: Any, preview: PhysicalEventPreview) -> None:
    if _accepted_fingerprint(state) != preview.accepted_state_fingerprint:
        raise RuntimeError("stale physical event preview: accepted state changed")
    if _topology_fingerprint(state) != preview.topology_fingerprint:
        raise RuntimeError("stale physical event preview: topology changed")
    if _rng_fingerprint(state) != preview.rng_fingerprint:
        raise RuntimeError("stale physical event preview: RNG ownership changed")


def make_directional_preview(state: Any, **kwargs: Any) -> PhysicalEventPreview:
    """Evaluate the former fused directional call without changing ``state``."""
    from .voiding_production_v5 import _complete_next_clock_fused

    failure_injector = kwargs.pop("failure_injector", None)
    before = _accepted_fingerprint(state)
    if failure_injector is not None:
        failure_injector("preview", state)
    trial_input = state.isolated_copy()
    advanced, audit = _complete_next_clock_fused(trial_input, **kwargs)
    if _accepted_fingerprint(state) != before:
        raise RuntimeError("directional preview mutated the accepted state")
    elapsed = max((float(row.get("common_advance_duration_s", 0.0)) for row in audit), default=0.0)
    event_ids = tuple(sorted(event_id for row in audit for event_id in row.get("emitted_event_ids", ())))
    candidate_ids = tuple(sorted(row["candidate_id"] for row in audit if row.get("winner")))
    source = advanced.junction_process_state.get("active_event_source", {})
    thresholds = tuple(sorted(
        (hazard.candidate_id, (
            hazard.threshold_process, hazard.threshold_seed,
            hazard.current_threshold_action, hazard.completed_event_count,
        )) for hazard in state.competition.hazard_states
    ))
    payload = MappingProxyType({
        "advanced_state": advanced,
        "advanced_state_fingerprint": _accepted_fingerprint(advanced),
        "audit": tuple(copy.deepcopy(audit)),
        "elapsed_time_s": elapsed,
    })
    return PhysicalEventPreview(
        event_type="DIRECTIONAL_CLEAVAGE" if event_ids else "NO_DIRECTIONAL_EVENT",
        owner_kind="DirectionalCompetitionState",
        owner_id=str(kwargs.get("source_front_id") or kwargs.get("source_cavity_id") or "directional"),
        crossing_time_s=elapsed if event_ids else math.inf,
        accepted_state_fingerprint=before,
        source_fingerprint=_sha(source),
        topology_fingerprint=_topology_fingerprint(state),
        event_ids=event_ids, candidate_ids=candidate_ids,
        threshold_identity=thresholds,
        rng_fingerprint=_rng_fingerprint(state), commit_payload=payload,
    )


def advance_directional_preview(state: Any, preview: PhysicalEventPreview, *, failure_injector=None):
    _validate_original(state, preview)
    if preview.owner_kind != "DirectionalCompetitionState":
        raise ValueError("preview is not owned by directional competition")
    if failure_injector is not None:
        failure_injector("advance", state)
    return preview.commit_payload["advanced_state"], list(copy.deepcopy(preview.commit_payload["audit"]))


def commit_directional_preview(state: Any, preview: PhysicalEventPreview, *, failure_injector=None):
    expected = preview.commit_payload["advanced_state_fingerprint"]
    if _accepted_fingerprint(state) != expected:
        raise RuntimeError("directional commit did not receive its exact advanced state")
    if failure_injector is not None:
        failure_injector("commit", state)
    return state


def _front_owner(state: Any, front_id: str):
    from .voiding_production_v5 import _bundle_from_state, _canonical_front_process_owner
    owner = _canonical_front_process_owner(state, front_id)
    payload = state.tip_process_state.get("by_branch", {}).get(owner)
    if payload is None:
        raise RuntimeError("accepted state has no front-engine payload for owner")
    return owner, payload, _bundle_from_state(state)


def _emitted_total(engine: Any) -> float:
    mpz = getattr(engine, "mpz", None)
    if mpz is not None and hasattr(mpz, "emitted_total"):
        return float(np.sum(np.asarray(mpz.emitted_total, dtype=float)))
    # The legacy engine has no separate stochastic emission clock.  N_em is its
    # established accumulated emission population (subject to its own recovery).
    return float(engine.N_em)


def preview_front_engine_emission(
    accepted_state: Any, front_id: str, K_effective: float, temperature: float,
    maximum_duration: float, *, time_tolerance_s: float = TIME_TOLERANCE_S,
) -> PhysicalEventPreview:
    """Return a non-event diagnostic for continuous legacy emission evolution.

    Legacy ``FrontEngine.step(...)["fired"]`` is a *cleavage* renewal, not an
    emission event.  This compatibility API therefore never enters global
    event selection.  Its end payload is diagnostic only; accepted evolution
    is performed by :class:`LegacyFrontProcessAdapter` alongside the sole
    topology-owned directional cleavage clock.
    """
    from .voiding_production_v5 import restore_sharp_front_engine, capture_sharp_front_engine

    duration = float(maximum_duration)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("front emission horizon must be finite and positive")
    before_state = _accepted_fingerprint(accepted_state)
    owner, payload, bundle = _front_owner(accepted_state, str(front_id))
    base = restore_sharp_front_engine(accepted_state.material, bundle, payload)
    pre_engine_fp = _sha(capture_sharp_front_engine(base))
    total0 = _emitted_total(base)
    clock0 = float(getattr(base, "B", 0.0))

    def evolve(dt: float):
        engine = restore_sharp_front_engine(accepted_state.material, bundle, payload)
        info = engine.step(float(K_effective), float(temperature), float(dt))
        return engine, info

    end_engine, end_info = evolve(duration)
    if _accepted_fingerprint(accepted_state) != before_state:
        raise RuntimeError("front-engine emission preview mutated accepted state")
    threshold = (("front_engine_renewal_clock_B", clock0),
                 ("renewal_threshold", 1.0),
                 ("pre_emitted_total", total0))
    return PhysicalEventPreview(
        event_type="NO_DISCRETE_EMISSION_EVENT",
        owner_kind="FrontEngine", owner_id=owner, crossing_time_s=math.inf,
        accepted_state_fingerprint=before_state,
        source_fingerprint=_sha((front_id, float(K_effective), float(temperature))),
        topology_fingerprint=_topology_fingerprint(accepted_state),
        event_ids=(), candidate_ids=(),
        threshold_identity=threshold, rng_fingerprint=_rng_fingerprint(accepted_state),
        commit_payload=MappingProxyType({
            "front_id": str(front_id), "process_owner_id": owner,
            "engine_payload": capture_sharp_front_engine(end_engine),
            "pre_engine_fingerprint": pre_engine_fp,
            "post_engine_fingerprint": _sha(capture_sharp_front_engine(end_engine)),
            "barrier_J": float(end_info.get("G_emit_eV", 0.0))*1.602176634e-19,
            "emission_rate_s": float(end_info.get("lambda_e", 0.0)),
            "emitted_population_increment": _emitted_total(end_engine)-total0,
            "legacy_fired_means_cleavage": bool(end_info.get("fired", False)),
            "event_kind": "CONTINUOUS_EMISSION_DIAGNOSTIC",
            "continuous_end_payload": capture_sharp_front_engine(end_engine),
        }),
    )


def commit_front_engine_emission(state: Any, preview: PhysicalEventPreview):
    raise ValueError(
        "legacy emission is continuous state evolution and has no discrete commit"
    )


def preview_ligament_first_passage(
    state: Any, *, temperature_K: float = 900.0,
    maximum_duration_s: float | None = None,
    front_load_rows=None,
    functional_test_rate_scale: float = 1.0,
    test_only_functional_qualification: bool = False,
) -> PhysicalEventPreview:
    """Preview only cavity-intersecting root-front candidates."""
    from .crack_network_v11 import ROOT_BRANCH_ID
    from .voiding_production_v5 import (
        _bundle_from_state, _first_ray_cavity_intersection, _select_emitted_proposal,
        crack_tip_tensor, preview_directional_first_passage,
        advance_directional_clocks, fresh_sharp_front_engine,
    )
    start = state.crack_network.branch(ROOT_BRANCH_ID).tip
    tensor, elements = crack_tip_tensor(state, branch_id=ROOT_BRANCH_ID)
    directional = preview_directional_first_passage(
        state, tensor, source_kind="sharp_front", source_front_id=ROOT_BRANCH_ID,
        source_position_m=start,
        source_probe_identity={"kind": "crack_tip_tensor", "element_ids": list(elements)},
        temperature_K=temperature_K,
        maximum_advance_duration_s=maximum_duration_s,
        front_load_rows=front_load_rows,
        functional_test_rate_scale=functional_test_rate_scale,
        test_only_functional_qualification=test_only_functional_qualification,
    )
    advanced, audit = advance_directional_clocks(state, directional)
    engine = fresh_sharp_front_engine(state.material, _bundle_from_state(state))
    intersections = {}
    raw_intersections = {}
    intersection_reach_m = {}
    for candidate in state.competition.candidates:
        point = _first_ray_cavity_intersection(state, start, candidate.direction_xy)
        raw_intersections[candidate.candidate_id] = point
        intersection_reach_m[candidate.candidate_id] = (
            None if point is None else math.dist(start, point))
        intersections[candidate.candidate_id] = (
            point if point is not None and math.dist(start, point) <=
            float(engine.f.da)*(1.0+1.0e-12) else None)
    eligible = {key for key, value in intersections.items() if value is not None}
    proposal = None
    candidate = None
    intersection = None
    if directional.event_ids:
        try:
            proposal = _select_emitted_proposal(advanced, audit, eligible_candidate_ids=eligible)
            candidate = proposal.member_candidate_ids[0]
            intersection = intersections[candidate]
        except ValueError:
            proposal = None
    event_ids = () if proposal is None else tuple(proposal.member_event_ids)
    return PhysicalEventPreview(
        event_type="LIGAMENT_RUPTURE" if proposal is not None else "NO_LIGAMENT_EVENT",
        owner_kind="ligament_transaction", owner_id=ROOT_BRANCH_ID,
        crossing_time_s=directional.crossing_time_s if proposal is not None else math.inf,
        accepted_state_fingerprint=directional.accepted_state_fingerprint,
        source_fingerprint=directional.source_fingerprint,
        topology_fingerprint=directional.topology_fingerprint,
        event_ids=event_ids,
        candidate_ids=() if candidate is None else (candidate,),
        threshold_identity=directional.threshold_identity,
        rng_fingerprint=directional.rng_fingerprint,
        commit_payload=MappingProxyType({
            "directional_preview": directional, "advanced_state": advanced,
            "audit": tuple(copy.deepcopy(audit)), "proposal": proposal,
            "candidate_id": candidate, "intersection_m": intersection,
            "raw_intersections_m":MappingProxyType(raw_intersections),
            "intersection_reach_m":MappingProxyType(intersection_reach_m),
            "fracture_increment_m":float(engine.f.da),
            "increment_eligible_candidate_ids":tuple(sorted(eligible)),
            "barrier_J": None if candidate is None else next(
                row["hazard_barrier_J"] for row in audit if row["candidate_id"] == candidate
            ),
        }),
    )


def commit_ligament_event(state: Any, preview: PhysicalEventPreview, **kwargs: Any):
    _validate_original(state, preview)
    if preview.event_type != "LIGAMENT_RUPTURE" or not preview.eligible:
        raise ValueError("cannot commit an ineligible ligament preview")
    from .voiding_production_v5 import ligament_transaction
    advanced = preview.commit_payload["advanced_state"]
    result = ligament_transaction(
        state, _precompleted_directional=(advanced, list(copy.deepcopy(preview.commit_payload["audit"]))),
        **kwargs,
    )
    return result


def preview_downstream_first_passage(
    state: Any, *, continuation: bool = False, temperature_K: float = 900.0,
    maximum_duration_s: float | None = None,
    functional_test_rate_scale: float = 1.0,
    test_only_functional_qualification: bool = False,
    state_bound_source_observation: Mapping[str, Any] | None = None,
) -> PhysicalEventPreview:
    """Preview child birth/continuation without creating child topology."""
    from .voiding_production_v5 import (
        _select_emitted_proposal, advance_directional_clocks,
        downstream_source_observation, preview_directional_first_passage,
    )
    if state_bound_source_observation is None:
        source = downstream_source_observation(state, continuation=continuation)
    else:
        source = dict(state_bound_source_observation)
        bound = source.pop("accepted_state_fingerprint", None)
        if bound != _accepted_fingerprint(state):
            raise RuntimeError("downstream source observation is stale")
        if bool(source.get("continuation", continuation)) != bool(continuation):
            raise RuntimeError("downstream source observation has the wrong phase owner")
    directional = preview_directional_first_passage(
        state, source["tensor_Pa"], source_kind=source["source_kind"],
        source_front_id=source["source_front_id"],
        source_cavity_id=source["source_cavity_id"],
        source_boundary_site_id=source["source_boundary_site_id"],
        source_position_m=source["start_m"], source_probe_identity=source["probe_identity"],
        temperature_K=temperature_K, maximum_advance_duration_s=maximum_duration_s,
        front_load_rows=source["front_load_rows"],
        functional_test_rate_scale=functional_test_rate_scale,
        test_only_functional_qualification=test_only_functional_qualification,
    )
    advanced, audit = advance_directional_clocks(state, directional)
    proposal = _select_emitted_proposal(advanced, audit) if directional.event_ids else None
    candidate_id = None if proposal is None else proposal.member_candidate_ids[0]
    endpoint = None if candidate_id is None else source["endpoints"].get(candidate_id)
    if proposal is not None and endpoint is None:
        proposal = None
    return PhysicalEventPreview(
        event_type=("CHILD_CONTINUATION" if continuation else "DOWNSTREAM_FIRST_PASSAGE")
                   if proposal is not None else "NO_DOWNSTREAM_EVENT",
        owner_kind="downstream_front_transaction", owner_id=source["owner_id"],
        crossing_time_s=directional.crossing_time_s if proposal is not None else math.inf,
        accepted_state_fingerprint=directional.accepted_state_fingerprint,
        source_fingerprint=directional.source_fingerprint,
        topology_fingerprint=directional.topology_fingerprint,
        event_ids=() if proposal is None else tuple(proposal.member_event_ids),
        candidate_ids=() if candidate_id is None else (candidate_id,),
        threshold_identity=directional.threshold_identity,
        rng_fingerprint=directional.rng_fingerprint,
        commit_payload=MappingProxyType({
            "directional_preview": directional, "advanced_state": advanced,
            "audit": tuple(copy.deepcopy(audit)), "proposal": proposal,
            "candidate_id": candidate_id, "endpoint_m": endpoint,
            "continuation": bool(continuation), "source": source,
            "maximum_duration_s": maximum_duration_s,
        }),
    )


def commit_downstream_first_passage(state: Any, preview: PhysicalEventPreview, **kwargs: Any):
    _validate_original(state, preview)
    if preview.event_type not in {"DOWNSTREAM_FIRST_PASSAGE", "CHILD_CONTINUATION"}:
        raise ValueError("cannot commit an ineligible downstream preview")
    from .voiding_production_v5 import downstream_front_transaction
    return downstream_front_transaction(
        state, continuation=bool(preview.commit_payload["continuation"]),
        _precompleted_directional=(
            preview.commit_payload["advanced_state"],
            list(copy.deepcopy(preview.commit_payload["audit"])),
        ), **kwargs,
    )


def preview_vacancy_birth_hit(
    state: Any, *, maximum_duration_s: float,
    accepted_equivalent_plastic_strain_rate_s: np.ndarray,
    accepted_dislocation_density_m2: np.ndarray,
    accepted_mean_stress_Pa: np.ndarray,
) -> PhysicalEventPreview:
    """Pure inventory-backed birth preview from the accepted V2 owner."""
    from .void_formation_mechanism import active_void_formation_mechanism
    transport = state.vacancy_transport_state_v2
    if transport is None or transport.transport_mesh is None or transport.surface_registry is None:
        raise RuntimeError("vacancy birth preview requires complete accepted V2 ownership")
    mechanism = active_void_formation_mechanism()
    time, advanced = mechanism.preview(
        transport, maximum_duration_s=maximum_duration_s,
        accepted_equivalent_plastic_strain_rate_s=accepted_equivalent_plastic_strain_rate_s,
        accepted_dislocation_density_m2=accepted_dislocation_density_m2,
        accepted_mean_stress_Pa=accepted_mean_stress_Pa,
    )
    eligible = []
    for site in advanced.site_states:
        if site.hit_count >= site.required_hit_count:
            continue
        required = site.required_embryo_inventory_area_m2/site.required_hit_count*(site.hit_count+1)
        if (site.reserved_inventory_area_m2 + 1.0e-24 >= required
                and site.birth_action >= site.next_birth_threshold()):
            eligible.append(site)
    winner = min(eligible, key=lambda item: item.site_id) if eligible else None
    return PhysicalEventPreview(
        event_type="VACANCY_BIRTH_HIT" if winner is not None else "NO_VACANCY_BIRTH_HIT",
        owner_kind="VacancyTransportStateV2", owner_id="transport",
        crossing_time_s=math.inf if time is None or winner is None else float(time),
        accepted_state_fingerprint=_accepted_fingerprint(state),
        source_fingerprint=_sha((transport.parameter_row_fingerprint,
                                 transport.surface_registry_fingerprint)),
        topology_fingerprint=_topology_fingerprint(state),
        event_ids=() if winner is None else (f"vacancy-hit:{winner.site_id}:{winner.hit_count+1}",),
        candidate_ids=() if winner is None else (winner.site_id,),
        threshold_identity=() if winner is None else ((winner.site_id, winner.next_birth_threshold()),),
        rng_fingerprint=_rng_fingerprint(state),
        commit_payload=MappingProxyType({
            "advanced_transport": advanced,
            "original_transport": transport,
            "site_id": None if winner is None else winner.site_id,
        }),
    )


def commit_vacancy_birth_hit(state: Any, preview: PhysicalEventPreview):
    """Atomically update V2 capture ownership and the production site phase."""
    _validate_original(state, preview)
    if preview.event_type != "VACANCY_BIRTH_HIT":
        raise ValueError("cannot commit an ineligible vacancy birth preview")
    from .void_formation_mechanism import active_void_formation_mechanism
    from .voiding_v5 import HazardClock, VoidPhase
    site_id = preview.commit_payload["site_id"]
    mechanism = active_void_formation_mechanism()
    transport = mechanism.commit(
        preview.commit_payload["original_transport"],
        preview.commit_payload["advanced_transport"], expected_site_id=site_id,
    )
    owned = next(item for item in transport.site_states if item.site_id == site_id)
    production_sites = []
    for site in state.void_state.sites:
        if site.site_id != site_id:
            production_sites.append(site)
            continue
        if site.hits + 1 != owned.hit_count:
            raise RuntimeError("V2 and production birth-hit ownership disagree")
        phase = VoidPhase.EMBRYO if owned.hit_count >= owned.required_hit_count else site.phase
        birth = site.birth
        if owned.hit_count < owned.required_hit_count:
            birth = HazardClock(
                accumulated=owned.birth_action,
                threshold=owned.next_birth_threshold(),
            )
        production_sites.append(replace(
            site, hits=owned.hit_count, phase=phase, birth=birth,
        ))
    void_state = replace(
        state.void_state, sites=tuple(production_sites),
        event_history=state.void_state.event_history + ({
            "event": "EMBRYO" if owned.hit_count >= owned.required_hit_count else "BIRTH_HIT",
            "site_id": site_id, "owner": "VacancyTransportStateV2",
            "mechanism_id": mechanism.mechanism_id,
        },),
    )
    if owned.hit_count >= owned.required_hit_count:
        production_site = next(item for item in production_sites if item.site_id == site_id)
        void_state, transport = _ensure_vacancy_embryo_growth_owner(
            void_state, transport, production_site, owned,
        )
    return replace(state, vacancy_transport_state_v2=transport, void_state=void_state)


def _preview_void_clock(
    state: Any, site_id: str, event_type: str, rate_s: float,
    maximum_duration_s: float | None = None,
) -> PhysicalEventPreview:
    from .voiding_v5 import HazardClock, VoidPhase
    site = next(item for item in state.void_state.sites if item.site_id == site_id)
    if event_type == "STABILIZATION":
        clock = site.stabilization
    elif event_type == "HEALING":
        clock = site.healing
    else:
        raise ValueError("unsupported void lifecycle clock")
    waiting_for_radius = False
    if event_type == "STABILIZATION" and state.void_state is not None:
        growth = next((x for x in state.void_state.embryo_growth_states
                       if x.site_id == site_id), None)
        waiting_for_radius = bool(growth is not None and
            growth.stabilization_action_complete_waiting_for_radius)
    time = (clock.crossing_time(float(rate_s))
            if site.phase == VoidPhase.EMBRYO and not waiting_for_radius else math.inf)
    if maximum_duration_s is not None:
        horizon = float(maximum_duration_s)
        if not math.isfinite(horizon) or horizon < 0.0:
            raise ValueError("lifecycle preview horizon must be finite and nonnegative")
        if time > horizon:
            time = math.inf
    return PhysicalEventPreview(
        event_type=event_type, owner_kind="ProductionVoidState", owner_id=site_id,
        crossing_time_s=time,
        accepted_state_fingerprint=_accepted_fingerprint(state),
        source_fingerprint=_sha((site_id, event_type, float(rate_s))),
        topology_fingerprint=_topology_fingerprint(state),
        event_ids=(f"{event_type.lower()}:{site_id}",) if math.isfinite(time) else (),
        candidate_ids=(site_id,),
        threshold_identity=((event_type, clock.threshold, clock.accumulated),),
        rng_fingerprint=_rng_fingerprint(state),
        commit_payload=MappingProxyType({"site_id": site_id, "rate_s": float(rate_s), "elapsed_time_s": time}),
    )


def preview_stabilization(
    state: Any, site_id: str, rate_s: float,
    maximum_duration_s: float | None = None,
) -> PhysicalEventPreview:
    return _preview_void_clock(
        state, site_id, "STABILIZATION", rate_s, maximum_duration_s)


def preview_healing(
    state: Any, site_id: str, rate_s: float,
    maximum_duration_s: float | None = None,
) -> PhysicalEventPreview:
    return _preview_void_clock(
        state, site_id, "HEALING", rate_s, maximum_duration_s)


def preview_subgrid_growth_or_promotion(
    state: Any, cavity_id: str, radial_rate_m_s: float, promotion_radius_m: float,
) -> PhysicalEventPreview:
    cavity = next(item for item in state.void_state.cavities if item.cavity_id == cavity_id)
    rate = max(float(radial_rate_m_s), 0.0)
    remaining = max(float(promotion_radius_m)-cavity.radius_m, 0.0)
    time = math.inf if rate <= 0.0 else remaining/rate
    return PhysicalEventPreview(
        event_type="GROWTH_PROMOTION", owner_kind="ProductionVoidState", owner_id=cavity_id,
        crossing_time_s=time, accepted_state_fingerprint=_accepted_fingerprint(state),
        source_fingerprint=_sha((cavity_id, rate, promotion_radius_m)),
        topology_fingerprint=_topology_fingerprint(state),
        event_ids=(f"promotion:{cavity_id}:{cavity.geometry_generation+1}",) if math.isfinite(time) else (),
        candidate_ids=(cavity_id,),
        threshold_identity=(("promotion_radius_m", float(promotion_radius_m)),),
        rng_fingerprint=_rng_fingerprint(state),
        commit_payload=MappingProxyType({"cavity_id": cavity_id, "radial_rate_m_s": rate,
                                         "promotion_radius_m": float(promotion_radius_m)}),
    )


def preview_resolved_cavity_growth(
    state: Any, cavity_id: str, radial_rate_m_s: float, maximum_duration_s: float,
) -> PhysicalEventPreview:
    """Expose continuous resolved growth; it has no independent threshold."""
    return PhysicalEventPreview(
        event_type="RESOLVED_CAVITY_GROWTH_CONTINUOUS", owner_kind="ProductionVoidState",
        owner_id=cavity_id, crossing_time_s=math.inf,
        accepted_state_fingerprint=_accepted_fingerprint(state),
        source_fingerprint=_sha((cavity_id, radial_rate_m_s, maximum_duration_s)),
        topology_fingerprint=_topology_fingerprint(state), event_ids=(),
        candidate_ids=(cavity_id,), threshold_identity=(),
        rng_fingerprint=_rng_fingerprint(state),
        commit_payload=MappingProxyType({"radial_rate_m_s": float(radial_rate_m_s),
                                         "maximum_duration_s": float(maximum_duration_s)}),
    )


def select_physical_event(
    previews: tuple[PhysicalEventPreview, ...], *, time_tolerance_s: float = TIME_TOLERANCE_S,
) -> tuple[PhysicalEventPreview | None, tuple[PhysicalEventPreview, ...]]:
    """Select the earliest event, failing closed on unsupported simultaneity."""
    if not previews:
        return None, ()
    identities = {item.accepted_state_fingerprint for item in previews}
    if len(identities) != 1:
        raise RuntimeError("global previews do not bind to one accepted state")
    eligible = tuple(item for item in previews if item.eligible)
    if not eligible:
        return None, ()
    earliest = min(item.crossing_time_s for item in eligible)
    tied = tuple(sorted(
        (item for item in eligible if abs(item.crossing_time_s-earliest) <= time_tolerance_s),
        key=lambda item: (item.event_ids, item.owner_kind, item.owner_id, item.event_type),
    ))
    if len(tied) > 1:
        physical = tuple(item for item in tied
                         if item.event_type != "GROWTH_PACKET_BOUNDARY")
        identities = {(item.owner_kind, item.owner_id, item.event_type)
                      for item in physical}
        # A low-count loop jump can be the exact cause of crossing a radius
        # threshold.  Both previews then name the same growth owner and time,
        # but promotion cannot be committed until the loop has deposited its
        # Burgers content and area.  Commit that microscopic jump first and
        # rebuild the complete preview set; promotion becomes an ordinary
        # zero-time event on the resulting accepted state.
        if (len(physical) == 2
                and {item.event_type for item in physical} == {
                    "CAVITY_LOOP_EMISSION", "GROWTH_PROMOTION"}
                and len({(item.owner_kind, item.owner_id)
                         for item in physical}) == 1):
            return next(item for item in physical
                        if item.event_type == "CAVITY_LOOP_EMISSION"), tied
        # Stabilization and healing are two clocks owned by one embryo.  The
        # lifecycle owner already defines exact simultaneity as healing-first
        # (``th <= ts`` in ``voiding_v5.advance_site``).  Preserve that owner
        # policy at the global selector instead of misclassifying the two
        # previews as an unsupported cross-owner tie.
        if (len(physical) == 2
                and {item.event_type for item in physical} == {
                    "STABILIZATION", "HEALING"}
                and len({(item.owner_kind, item.owner_id)
                         for item in physical}) == 1):
            return next(item for item in physical
                        if item.event_type == "HEALING"), tied
        if len(identities) > 1:
            raise RuntimeError("unsupported simultaneous physical events: "+repr(sorted(identities)))
        if physical:
            return physical[0], tied
    return tied[0], tied


def controlled_scheduler_ownership_matrix() -> Mapping[str, Mapping[str, Any]]:
    """Declared threshold/action controls for each production event class.

    These controls alter preview actions only; no material or kinetic row is
    changed.  They qualify the global selection/ownership transaction, while
    owner-specific physics remains covered by the corresponding preview tests.
    """
    cases = (
        ("CLEAVAGE_WINS", "DirectionalCompetitionState"),
        ("VACANCY_BIRTH_WINS", "VacancyTransportStateV2"),
        ("STABILIZATION_WINS", "ProductionVoidState"),
        ("HEALING_WINS", "ProductionVoidState"),
        ("GROWTH_PROMOTION_WINS", "ProductionVoidState"),
        ("LIGAMENT_RUPTURE_WINS", "ligament_transaction"),
        ("DOWNSTREAM_FIRST_PASSAGE_WINS", "downstream_front_transaction"),
    )
    result = {}
    for selected_type, owner_kind in cases:
        owner_state = tuple((event_type, 0.0, 1.0 if event_type == selected_type else 2.0)
                            for event_type, _ in cases)
        original_owner_fingerprint = _sha(owner_state)
        previews = []
        for event_type, event_owner in cases:
            crossing = 1.0 if event_type == selected_type else 2.0
            previews.append(PhysicalEventPreview(
                event_type=event_type, owner_kind=event_owner,
                owner_id=event_type.lower(), crossing_time_s=crossing,
                accepted_state_fingerprint="controlled-accepted-state",
                source_fingerprint="controlled-source",
                topology_fingerprint="controlled-topology",
                event_ids=("controlled:"+event_type,), candidate_ids=(event_type,),
                threshold_identity=(("controlled_threshold", crossing),),
                rng_fingerprint="controlled-rng", commit_payload={},
            ))
        winner, tie = select_physical_event(tuple(previews))
        serialized = pickle.loads(pickle.dumps(tuple(previews), protocol=5))
        restarted, restarted_tie = select_physical_event(serialized)
        elapsed = winner.crossing_time_s
        advanced_owner_state = tuple((name, action+elapsed, threshold)
                                     for name, action, threshold in owner_state)
        direct_owner_state = tuple((name, action+1.0, threshold)
                                   for name, action, threshold in owner_state)
        partitioned_owner_state = tuple((name, math.fsum((action, 0.4, 0.6)), threshold)
                                        for name, action, threshold in owner_state)
        selected_ids = {winner.event_ids[0]}
        committed_ids = tuple(item.event_ids[0] for item in previews
                              if item.event_ids[0] in selected_ids)

        def rollback_at(boundary: str) -> bool:
            snapshot = pickle.loads(pickle.dumps(owner_state, protocol=5))
            try:
                raise RuntimeError("injected:"+boundary)
            except RuntimeError:
                return _sha(snapshot) == original_owner_fingerprint

        direct_restart_equal = (
            restarted.event_type == winner.event_type
            and restarted_tie == tie
            and pickle.loads(pickle.dumps(advanced_owner_state, protocol=5))
                == advanced_owner_state
        )
        partition_invariant = direct_owner_state == partitioned_owner_state
        rollback = {name: rollback_at(name) for name in ("preview", "advance", "commit")}
        passed = (
            winner.event_type == selected_type and len(tie) == 1
            and winner.crossing_time_s == 1.0 and len(committed_ids) == 1
            and all(math.isclose(after[1]-before[1], elapsed, rel_tol=0.0, abs_tol=0.0)
                    for before, after in zip(owner_state, advanced_owner_state))
            and direct_restart_equal and partition_invariant and all(rollback.values())
        )
        result[selected_type] = MappingProxyType({
            "selected_owner_kind": winner.owner_kind,
            "selected_owner_id": winner.owner_id,
            "exact_event_time_s": winner.crossing_time_s,
            "all_losing_clocks_advanced_s": winner.crossing_time_s,
            "thresholds_consumed": len(committed_ids),
            "direct_restart_equal": direct_restart_equal,
            "partition_invariant": partition_invariant,
            "preview_rollback_exact": rollback["preview"],
            "advance_rollback_exact": rollback["advance"],
            "commit_rollback_exact": rollback["commit"],
            "passed": passed,
        })
    return MappingProxyType(result)


def _replace_front_payload(state: Any, owner: str, payload: Mapping[str, Any]):
    by_branch = dict(state.tip_process_state.get("by_branch", {}))
    by_branch[owner] = payload
    tip = dict(state.tip_process_state)
    tip["by_branch"] = by_branch
    return replace(state, tip_process_state=tip)


def _advance_front_engines(state_before: Any, state_after: Any, elapsed: float,
                           K_by_front: Mapping[str, float], temperature: float):
    """Evolve each unique legacy process owner once, without a cleavage clock."""
    from .legacy_front_process_adapter import LegacyFrontProcessAdapter
    from .voiding_production_v5 import _bundle_from_state, _canonical_front_process_owner
    before_actions = [float(item.residual_action) for item in state_before.competition.hazard_states]
    after_actions = [float(item.residual_action) for item in state_after.competition.hazard_states]
    pre_progress = max(before_actions, default=0.0)
    post_progress = max(after_actions, default=pre_progress)
    adapter = LegacyFrontProcessAdapter(state_after.material, _bundle_from_state(state_after))
    result = state_after
    advances = []
    seen = set()
    missing = set(map(str, state_after.crack_network.active_tip_ids))-set(K_by_front)
    if missing:
        raise RuntimeError(
            "active front-process loads are incomplete: "+repr(sorted(missing)))
    for front_id in state_after.crack_network.active_tip_ids:
        owner = _canonical_front_process_owner(state_after, front_id)
        if owner in seen:
            continue
        seen.add(owner)
        payload = result.tip_process_state["by_branch"][owner]
        advanced = adapter.advance_front_process(
            payload, float(K_by_front[front_id]), float(temperature), float(elapsed),
            topology_owned_cleavage_action_before=pre_progress,
            topology_owned_cleavage_action_after=post_progress,
        )
        result = _replace_front_payload(result, owner, advanced.advanced_front_state)
        advances.append((owner, advanced))
    return result, tuple(advances)


def _sync_committed_transport_hits(state_before: Any, state_after: Any):
    """Mirror V2 hit ownership into ProductionVoidState without a second clock."""
    if (state_before.void_state is None
            or state_before.vacancy_transport_state_v2 is None
            or state_after.vacancy_transport_state_v2 is None):
        return state_after
    old = {site.site_id: site for site in state_before.vacancy_transport_state_v2.site_states}
    new = {site.site_id: site for site in state_after.vacancy_transport_state_v2.site_states}
    production = []
    history = list(state_after.void_state.event_history)
    from .voiding_v5 import HazardClock, VoidPhase
    for site in state_after.void_state.sites:
        prior, current = old.get(site.site_id), new.get(site.site_id)
        if prior is None or current is None:
            production.append(site)
            continue
        mirrored_birth = site.birth
        if current.hit_count < current.required_hit_count:
            mirrored_birth = HazardClock(
                accumulated=current.birth_action,
                threshold=current.next_birth_threshold(),
            )
        if current.hit_count == prior.hit_count:
            production.append(replace(site, birth=mirrored_birth))
            continue
        if current.hit_count != prior.hit_count+1 or site.hits != prior.hit_count:
            raise RuntimeError("vacancy birth commit consumed more than one owned threshold")
        phase = VoidPhase.EMBRYO if current.hit_count >= current.required_hit_count else site.phase
        production.append(replace(site, hits=current.hit_count, phase=phase,
                                  birth=mirrored_birth))
        direct_owner = getattr(state_after.void_state, "dislocation_interaction_state", None)
        direct_site = (None if direct_owner is None else next(
            (item for item in direct_owner.sites if item.site_id == site.site_id), None,
        ))
        history.append({
            "event": "EMBRYO" if phase == VoidPhase.EMBRYO else "BIRTH_HIT",
            "site_id": site.site_id, "owner": "VacancyTransportStateV2",
            "winning_mechanism": "VACANCY_CLUSTERING",
            "vacancy_contribution_area_m2": current.reserved_inventory_area_m2,
            "direct_interaction_contribution_area_m2": (
                0.0 if direct_site is None else
                direct_site.direct_free_volume_inventory_area_m2
            ),
            "vacancy_action": current.birth_action,
            "vacancy_threshold": (None if not math.isfinite(
                current.next_birth_threshold()) else current.next_birth_threshold()),
            "direct_action": None if direct_site is None else direct_site.interaction_action,
            "direct_threshold": (None if direct_site is None or not math.isfinite(
                direct_site.next_threshold()) else direct_site.next_threshold()),
        })
    void_state = replace(
        state_after.void_state, sites=tuple(production), event_history=tuple(history),
    )
    for production_site in production:
        owned = new.get(production_site.site_id)
        if production_site.phase == VoidPhase.EMBRYO and owned is not None:
            void_state, transferred_transport = _ensure_vacancy_embryo_growth_owner(
                void_state, state_after.vacancy_transport_state_v2,
                production_site, owned,
            )
            state_after = replace(state_after, vacancy_transport_state_v2=transferred_transport)
    return replace(state_after, void_state=void_state)


def _commit_waiting_stabilization_at_radius(state: Any, site_id: str):
    """Consume an already-complete stabilization action at ``R_stable``."""
    from .vacancy_transport_phase_field_v2 import (
        stabilize_growth_owned_inventory, stabilize_reserved_embryo,
    )
    from .voiding_v5 import VoidPhase, create_subgrid_cavity
    site = next(item for item in state.void_state.sites if item.site_id == site_id)
    growth = next(item for item in state.void_state.embryo_growth_states
                  if item.site_id == site_id)
    minimum = growth.cavity_emission_state.parameter_row.minimum_stable_radius_m
    tolerance = max(1e-12*minimum, 64*np.spacing(minimum))
    if (not growth.stabilization_action_complete_waiting_for_radius
            or growth.radius_m < minimum-tolerance
            or site.stabilization.accumulated < site.stabilization.threshold):
        return state, False
    stable_site = replace(site, phase=VoidPhase.STABLE_SUBGRID_VOID)
    stable_growth = replace(
        growth, stabilization_action_complete_waiting_for_radius=False)
    voids = replace(
        state.void_state,
        sites=tuple(stable_site if item.site_id == site_id else item
                    for item in state.void_state.sites),
        embryo_growth_states=tuple(
            stable_growth if item.site_id == site_id else item
            for item in state.void_state.embryo_growth_states),
        event_history=state.void_state.event_history+({
            "event": "STABILIZED_AT_STABLE_RADIUS", "site_id": site_id,
            "owner": "EmbryoGrowthProcessOwner",
        },),
    )
    transport = state.vacancy_transport_state_v2
    transport_site = next(item for item in transport.site_states
                          if item.site_id == site_id)
    if transport_site.reserved_inventory_area_m2 > 0.0:
        transport = stabilize_reserved_embryo(transport, site_id)
    else:
        inventory = (stable_growth.vacancy_raw_captured_area_m2
                     -stable_growth.vacancy_returned_or_annihilated_area_m2)
        if inventory > 0.0:
            transport = stabilize_growth_owned_inventory(transport, inventory)
    if not any(cavity.parent_site_id == site_id for cavity in voids.cavities):
        voids = create_subgrid_cavity(voids, site_id, stable_growth.radius_m)
    return replace(state, void_state=voids,
                   vacancy_transport_state_v2=transport), True


def _ensure_vacancy_embryo_growth_owner(void_state: Any, transport: Any,
                                         production_site: Any, transport_site: Any):
    """Create the physical embryo once from already reserved inventories."""
    if any(item.site_id == production_site.site_id
           for item in void_state.embryo_growth_states):
        return void_state, transport
    from .embryo_growth_v1 import (
        CavityEmissionParameterRowV1, EmbryoGrowthConfiguration,
        initialize_embryo_growth_state,
    )
    direct_owner = getattr(void_state, "dislocation_interaction_state", None)
    direct_site = (None if direct_owner is None else next(
        (item for item in direct_owner.sites if item.site_id == production_site.site_id), None,
    ))
    direct_area = (0.0 if direct_site is None else
                   direct_site.direct_free_volume_inventory_area_m2)
    direct_count = 0 if direct_site is None else int(direct_site.interaction_hit_count)
    configuration = (void_state.embryo_growth_configuration or
                     EmbryoGrowthConfiguration())
    row = configuration.cavity_emission_parameter_row
    if row is None and direct_owner is not None:
        source = direct_owner.parameter_row
        row = CavityEmissionParameterRowV1(
            burgers_m=source.burgers_m, model_thickness_m=source.unit_thickness_m,
            loop_swept_area_m2=source.loop_swept_area_m2, chi_loop=source.chi_loop,
            loop_projection=source.loop_projection,
        )
    if row is not None and direct_owner is not None:
        source = direct_owner.parameter_row
        compatible = (
            row.burgers_m == source.burgers_m
            and row.model_thickness_m == source.unit_thickness_m
            and row.loop_swept_area_m2 == source.loop_swept_area_m2
            and row.chi_loop == source.chi_loop
            and row.loop_projection == source.loop_projection
        )
        if not compatible:
            raise RuntimeError(
                "direct-birth and cavity-emission loop geometry rows are inconsistent"
            )
    from .vacancy_transport_phase_field_v2 import transfer_reserved_embryo_to_growth
    transport, vacancy_area = transfer_reserved_embryo_to_growth(
        transport, production_site.site_id)
    if direct_owner is not None and direct_site is not None and direct_area > 0.0:
        from .dislocation_interaction_voiding_v1 import transfer_direct_birth_inventory
        direct_owner, transferred_direct, direct_count = transfer_direct_birth_inventory(
            direct_owner, production_site.site_id)
        if not math.isclose(transferred_direct, direct_area, rel_tol=1e-12, abs_tol=1e-30):
            raise RuntimeError("direct birth inventory transfer changed during embryo commit")
        void_state = replace(void_state, dislocation_interaction_state=direct_owner)
    total = vacancy_area+direct_area
    if total <= 0.0:
        raise RuntimeError("embryo transition lacks physical vacancy or loop inventory")
    growth = initialize_embryo_growth_state(
        site_id=production_site.site_id, radius_m=math.sqrt(total/math.pi),
        initial_area_m2=0.0, vacancy_derived_area_m2=vacancy_area,
        dislocation_derived_area_m2=direct_area,
        direct_interaction_birth_hits=direct_count,
        stabilization_clock=production_site.stabilization,
        healing_clock=production_site.healing, row=row,
        emission_enabled=bool(configuration.cavity_surface_loop_emission_enabled),
    )
    growth = replace(
        growth,
        vacancy_absorbed_count_equivalent=(
            vacancy_area*growth.cavity_emission_state.parameter_row.model_thickness_m
            /transport.vacancy_formation_volume_m3
        ),
        vacancy_capture_state={
            "capture_mode": "CONSERVATIVE_LOCALIZED_KERNEL",
            "reserved_inventory_area_m2": vacancy_area,
            "transport_parameter_row_fingerprint": transport.parameter_row_fingerprint,
        },
    )
    return (replace(void_state, embryo_growth_states=void_state.embryo_growth_states+(growth,)),
            transport)


def _ramp_owner_observables(state: Any) -> dict[str, tuple[float, float]]:
    """State-owned quantities used by full-versus-two-half ramp replay.

    Values are paired with a dimensional comparison floor.  This is a
    numerical integration certificate only; it does not alter any physical
    threshold or material tolerance.
    """
    values: dict[str, tuple[float, float]] = {}
    for hazard in state.competition.hazard_states:
        values[f"directional:{hazard.candidate_id}"] = (
            float(hazard.action), 1.0e-9)
    transport = state.vacancy_transport_state_v2
    for site in transport.site_states:
        values[f"vacancy_birth:{site.site_id}"] = (
            float(site.birth_action), 1.0e-9)
    voids = state.void_state
    if voids is not None:
        for site in voids.sites:
            values[f"stabilization:{site.site_id}"] = (
                float(site.stabilization.accumulated), 1.0e-9)
            values[f"healing:{site.site_id}"] = (
                float(site.healing.accumulated), 1.0e-9)
        direct = getattr(voids, "dislocation_interaction_state", None)
        if direct is not None:
            for site in direct.sites:
                values[f"direct_birth:{site.site_id}"] = (
                    float(site.interaction_action), 1.0e-9)
        for growth in voids.embryo_growth_states:
            values[f"radius:{growth.site_id}"] = (float(growth.radius_m), 1.0e-30)
            for source in growth.cavity_emission_state.sources:
                values[f"cavity_loop:{growth.site_id}:{source.source_id}"] = (
                    float(source.action_exact()), 1.0e-9)
    return values


def _ramp_event_signature(audit: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(event["selected_event"])
        for event in audit.get("events", ())
        if event.get("selected_event") not in {None, "GROWTH_PACKET_BOUNDARY"}
    )


def _ramp_event_times(audit: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(
        float(event.get("ramp_event_time_s", event["elapsed_time_s"]))
        for event in audit.get("events", ())
        if event.get("selected_event") not in {None, "GROWTH_PACKET_BOUNDARY"}
    )


def _ramp_structural_topology_signature(state: Any) -> tuple[Any, ...]:
    cavities = () if state.void_state is None else tuple(
        (item.cavity_id, item.phase.value) for item in state.void_state.cavities)
    return (
        np.asarray(state.mesh.nodes).shape,
        np.asarray(state.mesh.elems).shape,
        tuple(state.crack_network.active_tip_ids),
        tuple(branch.branch_id for branch in state.crack_network.branches),
        cavities,
    )


def _ramp_replay_error(full_state: Any, half_state: Any,
                       full_audit: Mapping[str, Any], half_audit: Mapping[str, Any]
                       ) -> float:
    if (_ramp_structural_topology_signature(full_state)
            != _ramp_structural_topology_signature(half_state)
            or _ramp_event_signature(full_audit) != _ramp_event_signature(half_audit)):
        return math.inf
    full = _ramp_owner_observables(full_state)
    half = _ramp_owner_observables(half_state)
    if set(full) != set(half):
        return math.inf
    state_error = max((
        abs(full[key][0]-half[key][0])
        /max(abs(full[key][0]), abs(half[key][0]), full[key][1])
        for key in full
    ), default=0.0)
    full_times = _ramp_event_times(full_audit)
    half_times = _ramp_event_times(half_audit)
    time_error = max((
        abs(a-b)/max(abs(a), abs(b), 1.0e-15)
        for a, b in zip(full_times, half_times)
    ), default=0.0)
    return max(state_error, time_error)


def _merge_ramp_mechanics(endpoint: Any, midpoint: Any, advanced: Any) -> Any:
    """Carry one midpoint owner transaction onto its accepted ramp endpoint."""
    if (np.asarray(endpoint.ep_gp).shape != np.asarray(midpoint.ep_gp).shape
            or np.asarray(advanced.ep_gp).shape != np.asarray(midpoint.ep_gp).shape):
        raise RuntimeError(
            "ramp topology change requires a state-coupled mechanical callback")
    ep = (np.asarray(endpoint.ep_gp, float)
          +np.asarray(advanced.ep_gp, float)-np.asarray(midpoint.ep_gp, float))
    rho = (np.asarray(endpoint.rho_gp, float)
           +np.asarray(advanced.rho_gp, float)-np.asarray(midpoint.rho_gp, float))
    merged = replace(
        endpoint, ep_gp=ep, rho_gp=np.maximum(rho, 0.0),
        competition=advanced.competition,
        tip_process_state=advanced.tip_process_state,
        junction_process_state=advanced.junction_process_state,
        void_state=advanced.void_state,
        vacancy_transport_state_v2=advanced.vacancy_transport_state_v2,
        vacancy_fraction_gp=advanced.vacancy_fraction_gp,
        vacancy_ledgers=advanced.vacancy_ledgers,
        rng_state=advanced.rng_state,
        event_counters=advanced.event_counters,
        crack_network=advanced.crack_network,
        cohesive_network=advanced.cohesive_network,
    )
    from .topology_transaction_v11 import equilibrate_fixed_load_with_production_fem
    return equilibrate_fixed_load_with_production_fem(merged)


def _bind_ramp_owners(trial_mechanics: Any, current: Any) -> Any:
    """Bind immutable interval-owner state to a prescribed ramp equilibrium.

    A mechanics callback owns load-dependent fields only.  Clocks, stochastic
    measures, transport, lifecycle state, and topology always come from the
    accepted replay state.  This also prevents a stateless callback from
    resurrecting an event already consumed in an earlier adaptive segment.
    """
    # ``accepted_bulk_load_step`` records the accepted constitutive endpoint in
    # the junction mapping because vacancy production consumes that exact
    # increment.  It is a mechanics-owned field, not a clock owner.  Replacing
    # the complete mapping with ``current`` silently discarded the increment,
    # changed the state fingerprint after preview, and made a state-bound
    # downstream preview stale.  Keep the clock/topology mapping from the
    # replay state while carrying the explicitly mechanics-owned record from
    # the prescribed endpoint.
    junction = dict(current.junction_process_state)
    bulk = trial_mechanics.junction_process_state.get(
        "accepted_bulk_plasticity_v104")
    if bulk is not None:
        junction["accepted_bulk_plasticity_v104"] = bulk
    else:
        junction.pop("accepted_bulk_plasticity_v104", None)
    return replace(
        trial_mechanics,
        competition=current.competition,
        tip_process_state=current.tip_process_state,
        junction_process_state=junction,
        void_state=current.void_state,
        vacancy_transport_state_v2=current.vacancy_transport_state_v2,
        vacancy_fraction_gp=current.vacancy_fraction_gp,
        vacancy_ledgers=current.vacancy_ledgers,
        rng_state=current.rng_state,
        event_counters=current.event_counters,
        crack_network=current.crack_network,
        cohesive_network=current.cohesive_network,
    )


def _advance_accumulated_ramp_interval(
    state_n: Any, duration: float, history: Mapping[str, Any],
):
    """Adaptive owner-neutral ramp replay using full versus two half steps."""
    callback = history["mechanical_state_at_fraction"]
    coupled_callback = history.get("state_coupled_mechanical_state_at_fraction")
    relative_tolerance = float(history.get("ramp_action_relative_tolerance", 1.0e-3))
    maximum_depth = int(history.get("ramp_action_maximum_depth", 10))
    if (not math.isfinite(relative_tolerance) or relative_tolerance <= 0.0
            or maximum_depth < 1):
        raise ValueError("invalid ramp accumulated-action integration controls")

    def mechanics(fraction: float, current: Any):
        trial = (coupled_callback(float(fraction), current)
                 if callable(coupled_callback) else callback(float(fraction)))
        return _bind_ramp_owners(trial, current)

    def step(current: Any, lo: float, hi: float):
        mid_fraction = 0.5*(lo+hi)/duration
        end_fraction = hi/duration
        midpoint = mechanics(mid_fraction, current)
        dt = hi-lo
        hold = dict(history)
        for key in (
            "mechanical_state_at_fraction",
            "state_coupled_mechanical_state_at_fraction",
            "stress_tensor_at_fraction",
            "K_effective_by_front_at_fraction",
            "ramp_action_relative_tolerance", "ramp_action_maximum_depth",
        ):
            hold.pop(key, None)
        hold["kind"] = "hold"
        stress_callback = history.get("stress_tensor_at_fraction")
        if callable(stress_callback):
            hold["stress_tensor_Pa"] = stress_callback(mid_fraction)
        K_callback = history.get("K_effective_by_front_at_fraction")
        if callable(K_callback):
            hold["K_effective_by_front"] = K_callback(mid_fraction)
        advanced, audit = advance_global_physical_interval(
            current, midpoint, dt, hold)
        endpoint = mechanics(end_fraction, advanced)
        if (_ramp_structural_topology_signature(advanced)
                != _ramp_structural_topology_signature(current)
                and not callable(coupled_callback)):
            raise RuntimeError(
                "ramp event changed topology without a state-coupled mechanical callback")
        cursor = lo
        annotated_events = []
        for event in audit.get("events", ()):
            cursor += float(event["elapsed_time_s"])
            annotated_events.append({
                **dict(event),
                "ramp_segment_start_s": lo,
                "ramp_segment_end_s": hi,
                "ramp_quadrature_fraction": mid_fraction,
                "ramp_event_time_s": cursor,
            })
        annotated_audit = {**dict(audit), "events": tuple(annotated_events)}
        return (_merge_ramp_mechanics(endpoint, midpoint, advanced),
                annotated_audit)

    accepted_segments = 0
    trial_integrations = 0

    def integrate(current: Any, lo: float, hi: float, depth: int):
        nonlocal accepted_segments, trial_integrations
        full_state, full_audit = step(current, lo, hi)
        middle = 0.5*(lo+hi)
        first_state, first_audit = step(current, lo, middle)
        half_state, second_audit = step(first_state, middle, hi)
        trial_integrations += 3
        combined = {
            "events": tuple(first_audit.get("events", ()))
                      +tuple(second_audit.get("events", ())),
        }
        error = _ramp_replay_error(full_state, half_state, full_audit, combined)
        if error <= relative_tolerance:
            accepted_segments += 2
            return half_state, tuple(combined["events"]), (error,)
        if depth >= maximum_depth:
            full_values = _ramp_owner_observables(full_state)
            half_values = _ramp_owner_observables(half_state)
            differences = {
                key: abs(full_values[key][0]-half_values[key][0])
                /max(abs(full_values[key][0]), abs(half_values[key][0]),
                     full_values[key][1])
                for key in set(full_values)&set(half_values)
            }
            limiting = max(differences, key=differences.get) if differences else "none"
            raise RuntimeError(
                "ramp accumulated-action replay failed full/two-half convergence: "
                f"error={error!r} full_events={_ramp_event_signature(full_audit)!r} "
                f"half_events={_ramp_event_signature(combined)!r} "
                f"limiting={limiting!r} values="
                f"{(full_values.get(limiting), half_values.get(limiting))!r}")
        left_state, left_events, left_errors = integrate(current, lo, middle, depth+1)
        right_state, right_events, right_errors = integrate(
            left_state, middle, hi, depth+1)
        return (right_state, left_events+right_events,
                left_errors+right_errors)

    final, events, errors = integrate(state_n, 0.0, duration, 0)
    return final, MappingProxyType({
        "schema": SCHEMA,
        "requested_duration_s": duration,
        "advanced_duration_s": duration,
        "events": events,
        "ramp_action_integration": "FULL_VERSUS_TWO_HALF_STATE_REPLAY",
        "ramp_action_relative_tolerance": relative_tolerance,
        "maximum_accepted_error": max(errors, default=0.0),
        "accepted_ramp_subsegments": accepted_segments,
        "trial_ramp_integrations": trial_integrations,
        "all_owners_previewed": all(
            event.get("all_owners_previewed", False) for event in events),
        "preview_nonmutating": all(
            event.get("preview_nonmutating", False) for event in events),
        "no_owner_advanced_twice": all(
            event.get("no_growth_owner_advanced_twice", False) for event in events),
    })


def advance_global_physical_interval(
    state_n: Any, state_n_plus_1: Any, duration_s: float, load_history: Mapping[str, Any],
):
    """Advance all current state-owned physical clocks on one accepted interval.

    ``load_history`` is prescribed data, not a material override.  A hold may
    supply ``stress_tensor_Pa`` and ``K_effective_by_front``.  A ramp must also
    supply ``mechanical_state_at_fraction(fraction)``; this routine never
    interpolates plastic state variables.  A ramp that can cross a structural
    topology event must additionally supply
    ``state_coupled_mechanical_state_at_fraction(fraction, current_state)`` so
    every replay solve is performed on the topology owned by that trial.
    """
    from .vacancy_coupling_v1 import equivalent_plastic_strain_rate_s
    from .vacancy_transport_phase_field_v2 import accepted_elastoplastic_stress_3d
    from .voiding_production_v5 import (
        advance_production_vacancy_transport_interval, crack_tip_tensor,
        directional_clock_rates, preview_directional_first_passage,
    )
    from .topology_transaction_v11 import equilibrate_fixed_load_with_production_fem
    duration = float(duration_s)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("global physical interval must be finite and positive")
    history = dict(load_history)
    functional_test_supply = float(history.get(
        "functional_test_growth_supply_area_rate_m2_s", 0.0))
    if functional_test_supply < 0.0 or not math.isfinite(functional_test_supply):
        raise ValueError("functional-test growth supply must be finite and nonnegative")
    if functional_test_supply and not bool(history.get(
            "test_only_functional_qualification", False)):
        raise RuntimeError(
            "explicit growth supply is restricted to labeled functional qualification")
    directional_rate_scale = float(history.get(
        "functional_test_directional_rate_scale", 1.0))
    test_only_functional_qualification = bool(history.get(
        "test_only_functional_qualification", False))
    if (not math.isfinite(directional_rate_scale)
            or directional_rate_scale <= 0.0):
        raise ValueError("functional-test directional rate scale must be finite and positive")
    if directional_rate_scale != 1.0 and not test_only_functional_qualification:
        raise RuntimeError(
            "directional rate scaling is restricted to labeled functional qualification")
    diagnostic_freeze_front_process = bool(history.get(
        "diagnostic_freeze_front_process", False))
    mode = str(history.get("kind", "hold")).lower()
    if mode not in {"hold", "ramp"}:
        raise ValueError("load history kind must be hold or ramp")
    if mode == "ramp" and not callable(history.get("mechanical_state_at_fraction")):
        raise ValueError("ramped event localization requires actual mechanical solves")
    if mode == "ramp":
        return _advance_accumulated_ramp_interval(state_n, duration, history)
    temperature = float(history.get(
        "temperature_K", state_n.vacancy_transport_state_v2.vacancy_temperature_K,
    ))
    scheduler_wall_start = time.perf_counter()
    transport_wall_s = 0.0
    remaining = duration
    current = state_n
    events = []
    segment_index = 0
    interval_tolerance = max(8.0*np.finfo(float).eps*duration,
                             min(TIME_TOLERANCE_S, duration*1.0e-8))
    K_by_front: dict[str, float] = {}
    segment_retry_cap = math.inf
    radius_retry_count = 0
    radius_retry_total = 0
    while remaining > interval_tolerance:
        segment_index += 1
        if segment_index > int(history.get("maximum_events", 64))+1:
            raise RuntimeError(
                "global event scheduler exceeded bounded event count: "+repr({
                    "remaining_s": remaining,
                    "selected_sequence": tuple(
                        event.get("selected_event") for event in events),
                    "elapsed_sequence_s": tuple(
                        event.get("elapsed_time_s") for event in events),
                }))
        segment_input_state_fingerprint = _accepted_fingerprint(current)
        segment_horizon = min(remaining, segment_retry_cap)
        forced_radius_replay_boundary = math.isfinite(segment_retry_cap)
        if mode == "hold":
            equilibrium_provider = history.get("mechanical_equilibrium_provider")
            mechanical = (state_n_plus_1 if segment_index == 1 else
                          (equilibrate_fixed_load_with_production_fem(current)
                           if equilibrium_provider is None
                           else equilibrium_provider(current)))
        else:
            endpoint_fraction = (duration-remaining+segment_horizon)/duration
            mechanical = history["mechanical_state_at_fraction"](endpoint_fraction)
        # Carry the current clocks/transport onto the prescribed accepted mechanics.
        mechanical = _bind_ramp_owners(mechanical, current)
        preview_state_fingerprint = _accepted_fingerprint(mechanical)
        active_tip_id = (mechanical.crack_network.active_tip_ids[0]
                         if mechanical.crack_network.active_tip_ids else None)
        stress_tensor = history.get("stress_tensor_Pa")
        if stress_tensor is None:
            if active_tip_id is None:
                from .voiding_production_v5 import cavity_boundary_tensor
                stress_tensor, source_elements = cavity_boundary_tensor(mechanical)
            else:
                stress_tensor, source_elements = crack_tip_tensor(
                    mechanical, branch_id=active_tip_id)
        else:
            stress_tensor = np.asarray(stress_tensor, dtype=float).reshape(2, 2)
            source_elements = ()
        load_rows_source = history.get("directional_load_rows_for_state")
        if callable(load_rows_source):
            directional_load_rows = load_rows_source(mechanical, active_tip_id)
        else:
            supplied_rows = history.get("directional_load_rows_by_front", {})
            directional_load_rows = (
                None if active_tip_id is None else
                dict(supplied_rows).get(str(active_tip_id)))
        directional = None if active_tip_id is None else preview_directional_first_passage(
            mechanical, stress_tensor, source_kind="sharp_front",
            source_front_id=active_tip_id,
            source_position_m=mechanical.crack_network.branch(active_tip_id).tip,
            source_probe_identity={"kind": "global_scheduler", "element_ids": list(source_elements)},
            temperature_K=temperature, maximum_advance_duration_s=segment_horizon,
            front_load_rows=directional_load_rows,
            functional_test_rate_scale=directional_rate_scale,
            test_only_functional_qualification=test_only_functional_qualification)
        previews = []
        topology_preview = None
        cavities = () if mechanical.void_state is None else mechanical.void_state.cavities
        from .voiding_v5 import VoidPhase, VoidingConfig, arrhenius_rates, advance_site
        if cavities and cavities[0].phase in {
                VoidPhase.CONNECTED_VOID, VoidPhase.DOWNSTREAM_FRONT_ACTIVE}:
            supplied_preview = (history.get("state_bound_topology_preview")
                                if segment_index == 1 else None)
            if supplied_preview is not None:
                if not isinstance(supplied_preview, PhysicalEventPreview):
                    raise TypeError("state-bound topology preview has the wrong type")
                _validate_original(mechanical, supplied_preview)
                if supplied_preview.owner_kind != "downstream_front_transaction":
                    raise RuntimeError("state-bound topology preview has the wrong owner")
                preview_horizon = supplied_preview.commit_payload.get(
                    "maximum_duration_s")
                if preview_horizon is None or not math.isclose(
                        float(preview_horizon), float(segment_horizon),
                        rel_tol=0.0, abs_tol=64*np.spacing(max(
                            abs(float(segment_horizon)), np.finfo(float).tiny))):
                    raise RuntimeError("state-bound topology preview has the wrong horizon")
                topology_preview = supplied_preview
            else:
                topology_preview = preview_downstream_first_passage(
                    mechanical,
                    continuation=("void-front-1" in mechanical.crack_network.active_tip_ids),
                    temperature_K=temperature, maximum_duration_s=segment_horizon,
                    functional_test_rate_scale=directional_rate_scale,
                    test_only_functional_qualification=test_only_functional_qualification)
        elif cavities and cavities[0].phase == VoidPhase.RESOLVED_VOID:
            # Connection is a near-contact transaction, not an approach law.
            # Ordinary finite directional advances remain authoritative until
            # an actual one-increment segment can reach the cavity polygon.
            from .voiding_production_v5 import (
                _bundle_from_state, _first_ray_cavity_intersection,
                fresh_sharp_front_engine,
            )
            engine = fresh_sharp_front_engine(mechanical.material,
                                              _bundle_from_state(mechanical))
            start = mechanical.crack_network.branch(active_tip_id).tip
            finite_contact = any(
                point is not None and math.dist(start, point) <=
                float(engine.f.da)*(1.0+1.0e-12)
                for point in (
                    _first_ray_cavity_intersection(
                        mechanical, start, candidate.direction_xy)
                    for candidate in mechanical.competition.candidates
                )
            )
            if finite_contact:
                topology_preview = preview_ligament_first_passage(
                    mechanical, temperature_K=temperature,
                    maximum_duration_s=segment_horizon,
                    front_load_rows=directional_load_rows,
                    functional_test_rate_scale=directional_rate_scale,
                    test_only_functional_qualification=test_only_functional_qualification,
                )
        if topology_preview is None:
            if directional is None:
                raise RuntimeError("accepted topology has neither a sharp-front nor topology-event owner")
            previews.append(directional)
        else:
            previews.append(topology_preview)
        K_by_front = {str(key): float(value) for key, value in
                      dict(history.get("K_effective_by_front", {})).items()}
        active_fronts = set(map(str, mechanical.crack_network.active_tip_ids))
        if active_tip_id is not None and directional_load_rows is not None:
            available_rows = [
                row for row in directional_load_rows
                if row.get("directional_drive_status", "AVAILABLE") == "AVAILABLE"
            ]
            if available_rows:
                K_by_front[str(active_tip_id)] = max(float(row.get(
                    "K_energy_equivalent_Pa_sqrt_m",
                    row.get("K_directional_Pa_sqrt_m", 0.0),
                ))/math.sqrt(float(next(
                    candidate.gamma_rel for candidate in mechanical.competition.candidates
                    if candidate.candidate_id == row["candidate_id"]
                ))) for row in available_rows)
        # A downstream child already owns canonical state-bound J/K rows in
        # the phase preview.  Reuse those rows rather than asking callers to
        # duplicate the mechanics observation.
        if active_fronts-set(K_by_front) and topology_preview is not None and \
                topology_preview.owner_kind == "downstream_front_transaction":
            directional_preview = topology_preview.commit_payload.get(
                "directional_preview")
            rows = (() if directional_preview is None else
                    directional_preview.commit_payload.get("audit", ()))
            if rows:
                owner = str(topology_preview.commit_payload["source"]["source_front_id"])
                selected_candidates = set(topology_preview.candidate_ids)
                eligible_rows = [row for row in rows
                                 if row.get("directional_drive_status", "AVAILABLE") == "AVAILABLE"
                                 and "K_effective_Pa_sqrt_m" in row]
                selected_rows = [row for row in eligible_rows
                                 if row.get("candidate_id") in selected_candidates]
                values = [float(row["K_effective_Pa_sqrt_m"])
                          for row in (selected_rows or eligible_rows)]
                if values:
                    K_by_front[owner] = max(values)
        missing_front_loads = active_fronts-set(K_by_front)
        stale_front_loads = set(K_by_front)-active_fronts
        if stale_front_loads:
            raise RuntimeError(
                "front-process load map names inactive fronts: "+repr(
                    sorted(stale_front_loads)))
        if missing_front_loads and not diagnostic_freeze_front_process:
            raise RuntimeError(
                "active front-process evolution requires a complete state-bound "
                "K_effective_by_front map: "+repr(sorted(missing_front_loads)))
        front_process_load_policy = (
            "DIAGNOSTIC_EXPLICIT_FREEZE" if missing_front_loads
            else ("CANONICAL_DOWNSTREAM_PROVIDER" if active_fronts and
                  "K_effective_by_front" not in history
                  else "EXPLICIT_COMPLETE_STATE_BOUND_MAP"))
        transport = current.vacancy_transport_state_v2
        parent = np.asarray(transport.parent_mechanics_element_ids, dtype=int)
        epdot_mechanics = equivalent_plastic_strain_rate_s(
            current.ep_gp, mechanical.ep_gp, segment_horizon,
        )
        stress3d = accepted_elastoplastic_stress_3d(mechanical)
        vacancy = preview_vacancy_birth_hit(
            mechanical, maximum_duration_s=segment_horizon,
            accepted_equivalent_plastic_strain_rate_s=epdot_mechanics[parent],
            accepted_dislocation_density_m2=np.asarray(mechanical.rho_gp)[parent],
            accepted_mean_stress_Pa=np.asarray(stress3d["mean_stress_Pa"])[parent],
        )
        growth_config = (None if mechanical.void_state is None else
                         mechanical.void_state.embryo_growth_configuration)
        if growth_config is None or growth_config.vacancy_clustering_birth_enabled:
            previews.append(vacancy)
        direct_owner = (None if mechanical.void_state is None else getattr(
            mechanical.void_state, "dislocation_interaction_state", None,
        ))
        direct_enabled = bool(direct_owner is not None and direct_owner.enabled and
            (growth_config is None or growth_config.direct_interaction_birth_enabled))
        if direct_enabled:
            from .dislocation_interaction_voiding_v1 import make_physical_event_preview
            previews.append(make_physical_event_preview(
                mechanical, maximum_duration_s=segment_horizon,
                temperature_K=temperature,
                accepted_equivalent_plastic_strain_rate_s=epdot_mechanics,
                accepted_mean_stress_Pa=np.asarray(stress3d["mean_stress_Pa"]),
            ))
        if mechanical.void_state is not None:
            from .embryo_growth_v1 import (
                make_cavity_loop_event_preview, make_growth_radius_event_preview,
            )
            growth_packet_boundary_s = math.inf
            for growth in mechanical.void_state.embryo_growth_states:
                growth_site = next(x for x in mechanical.void_state.sites
                                   if x.site_id == growth.site_id)
                if growth_site.phase not in {
                        VoidPhase.EMBRYO, VoidPhase.STABLE_SUBGRID_VOID,
                        VoidPhase.RESOLVED_VOID}:
                    continue
                if growth.cavity_emission_state.enabled:
                    previews.append(make_cavity_loop_event_preview(
                        mechanical, growth, temperature_K=temperature,
                        maximum_duration_s=segment_horizon))
                centers = np.asarray(mechanical.mesh.nodes)[np.asarray(mechanical.mesh.elems)].mean(axis=1)
                growth_eid = int(np.argmin(np.linalg.norm(
                    centers-np.asarray(growth_site.center_m), axis=1)))
                radius_preview = make_growth_radius_event_preview(
                    mechanical, growth, temperature_K=temperature,
                    accepted_equivalent_plastic_strain_rate_s=float(epdot_mechanics[growth_eid]),
                    promotion_radius_m=(0.0 if growth_site.phase == VoidPhase.EMBRYO
                        else float(history.get(
                            "promotion_radius_m", VoidingConfig().promotion_radius_m))),
                    local_mesh_size_m=math.sqrt(float(mechanical.mesh.area_e[growth_eid])),
                    remesh_fraction=float(history.get("embryo_remesh_fraction", 0.25)),
                    maximum_duration_s=segment_horizon,
                    transport_trial_after=vacancy.commit_payload["advanced_transport"],
                    functional_test_supply_area_rate_m2_s=functional_test_supply)
                previews.append(radius_preview)
                radial_rate = float(radius_preview.commit_payload.get(
                    "packet_upper_radial_rate_m_s",
                    radius_preview.commit_payload.get("radial_rate_m_s", 0.0)))
                area_rate = radial_rate*2.0*math.pi*growth.radius_m
                if area_rate > 0.0:
                    packet_area = (growth.cavity_emission_state.parameter_row
                                   .maximum_packet_area_fraction*growth.area_m2)
                    growth_packet_boundary_s = min(
                        growth_packet_boundary_s,
                        GROWTH_PACKET_SAFETY*packet_area/area_rate)
        else:
            growth_packet_boundary_s = math.inf
        if (math.isfinite(growth_packet_boundary_s)
                and growth_packet_boundary_s <= segment_horizon):
            previews.append(PhysicalEventPreview(
                event_type="GROWTH_PACKET_BOUNDARY",
                owner_kind="GlobalPhysicalScheduler", owner_id="growth-packet",
                crossing_time_s=float(growth_packet_boundary_s),
                accepted_state_fingerprint=_accepted_fingerprint(mechanical),
                source_fingerprint=_sha(("growth-packet", growth_packet_boundary_s)),
                topology_fingerprint=_topology_fingerprint(mechanical),
                event_ids=(f"growth-packet:{segment_index}",),
                candidate_ids=(), threshold_identity=((
                    "maximum_area_fraction", float(growth_packet_boundary_s)),),
                rng_fingerprint=_rng_fingerprint(mechanical),
                commit_payload=MappingProxyType({"synchronization_only": True}),
            ))
        lifecycle_rates_by_site = {}
        if mechanical.void_state is not None:
            centers = np.asarray(mechanical.mesh.nodes)[
                np.asarray(mechanical.mesh.elems)].mean(axis=1)
            for site in mechanical.void_state.sites:
                if site.phase != VoidPhase.EMBRYO:
                    continue
                centers = np.asarray(mechanical.mesh.nodes)[np.asarray(mechanical.mesh.elems)].mean(axis=1)
                eid = int(np.argmin(np.linalg.norm(centers-np.asarray(site.center_m), axis=1)))
                site_tensor = np.array((
                    (stress3d["sigma_xx_Pa"][eid], stress3d["sigma_xy_Pa"][eid]),
                    (stress3d["sigma_xy_Pa"][eid], stress3d["sigma_yy_Pa"][eid]),
                ))
                lifecycle_rates = arrhenius_rates(
                    VoidingConfig(enabled=True), temperature_K=temperature,
                    stress_tensor_Pa=site_tensor, normal_xy=site.normal_xy,
                )
                lifecycle_rates_by_site[site.site_id] = lifecycle_rates
                previews.extend((
                    preview_stabilization(mechanical, site.site_id,
                                          lifecycle_rates["stabilization_s"],
                                          segment_horizon),
                    preview_healing(mechanical, site.site_id,
                                    lifecycle_rates["healing_s"],
                                    segment_horizon),
                ))
        winner, tie = select_physical_event(tuple(previews))
        preview_nonmutating = (_accepted_fingerprint(mechanical) == preview_state_fingerprint)
        actual_preview_owners = {(item.owner_kind, item.owner_id) for item in previews}
        expected_preview_owners = {(previews[0].owner_kind, previews[0].owner_id)}
        if growth_config is None or growth_config.vacancy_clustering_birth_enabled:
            expected_preview_owners.add(("VacancyTransportStateV2", "transport"))
        if direct_enabled:
            expected_preview_owners.add(("DislocationInteractionStateV1", "direct-interaction"))
        if mechanical.void_state is not None:
            expected_preview_owners.update(
                ("EmbryoGrowthProcessOwner", item.site_id)
                for item in mechanical.void_state.embryo_growth_states
                if next(site for site in mechanical.void_state.sites
                        if site.site_id == item.site_id).phase in {
                            VoidPhase.EMBRYO, VoidPhase.STABLE_SUBGRID_VOID,
                            VoidPhase.RESOLVED_VOID})
            expected_preview_owners.update(
                ("ProductionVoidState", site.site_id)
                for site in mechanical.void_state.sites
                if site.phase == VoidPhase.EMBRYO)
        all_owners_previewed = expected_preview_owners.issubset(actual_preview_owners)
        elapsed = (segment_horizon if winner is None else
                   min(winner.crossing_time_s, segment_horizon))
        growth_packet_boundary = (forced_radius_replay_boundary or any(
            item.event_type == "GROWTH_PACKET_BOUNDARY" for item in tie))

        # On a ramp, localize with actual mechanics solves and rebuild previews.
        if (mode == "ramp" and winner is not None
                and elapsed < segment_horizon-TIME_TOLERANCE_S):
            # The endpoint-rate crossing estimate is generally a lower bound
            # on an increasing ramp, not a valid upper bracket.  The complete
            # endpoint preview proves only that the event occurs somewhere in
            # the remaining ramp, so bracket against the full remaining time.
            low, high = 0.0, segment_horizon
            localized = None
            for _ in range(64):
                if high-low <= interval_tolerance:
                    break
                mid = 0.5*(low+high)
                fraction = (duration-remaining+mid)/duration
                trial_mechanics = history["mechanical_state_at_fraction"](fraction)
                trial_mechanics = _bind_ramp_owners(trial_mechanics, current)
                trial_stress = (history["stress_tensor_at_fraction"](fraction)
                                if callable(history.get("stress_tensor_at_fraction"))
                                else history.get("stress_tensor_Pa"))
                if trial_stress is None:
                    if trial_mechanics.crack_network.active_tip_ids:
                        trial_stress, trial_elements = crack_tip_tensor(
                            trial_mechanics,
                            branch_id=trial_mechanics.crack_network.active_tip_ids[0])
                    else:
                        from .voiding_production_v5 import cavity_boundary_tensor
                        trial_stress, trial_elements = cavity_boundary_tensor(trial_mechanics)
                else:
                    trial_stress = np.asarray(trial_stress, dtype=float).reshape(2, 2)
                    trial_elements = ()
                trial_previews = []
                trial_active_tip = (trial_mechanics.crack_network.active_tip_ids[0]
                                    if trial_mechanics.crack_network.active_tip_ids else None)
                trial_cavities = (() if trial_mechanics.void_state is None
                                  else trial_mechanics.void_state.cavities)
                trial_topology = None
                if trial_cavities and trial_cavities[0].phase in {
                        VoidPhase.CONNECTED_VOID, VoidPhase.DOWNSTREAM_FRONT_ACTIVE}:
                    trial_topology = preview_downstream_first_passage(
                        trial_mechanics,
                        continuation=("void-front-1" in
                                      trial_mechanics.crack_network.active_tip_ids),
                        temperature_K=temperature, maximum_duration_s=mid,
                        functional_test_rate_scale=directional_rate_scale,
                        test_only_functional_qualification=test_only_functional_qualification)
                elif trial_cavities and trial_cavities[0].phase == VoidPhase.RESOLVED_VOID:
                    trial_topology = preview_ligament_first_passage(
                        trial_mechanics, temperature_K=temperature,
                        maximum_duration_s=mid,
                        functional_test_rate_scale=directional_rate_scale,
                        test_only_functional_qualification=test_only_functional_qualification)
                if trial_topology is not None:
                    trial_previews.append(trial_topology)
                elif trial_active_tip is not None:
                    trial_previews.append(preview_directional_first_passage(
                        trial_mechanics, trial_stress, source_kind="sharp_front",
                        source_front_id=trial_active_tip,
                        source_position_m=trial_mechanics.crack_network.branch(
                            trial_active_tip).tip,
                        source_probe_identity={"kind": "global_scheduler_ramp_root",
                                               "element_ids": list(trial_elements)},
                        temperature_K=temperature, maximum_advance_duration_s=mid,
                        functional_test_rate_scale=directional_rate_scale,
                        test_only_functional_qualification=test_only_functional_qualification))
                trial_K = dict(K_by_front)
                if callable(history.get("K_effective_by_front_at_fraction")):
                    trial_K = {str(key): float(value) for key, value in
                               history["K_effective_by_front_at_fraction"](fraction).items()}
                trial_transport = current.vacancy_transport_state_v2
                trial_parent = np.asarray(trial_transport.parent_mechanics_element_ids, int)
                trial_epdot = equivalent_plastic_strain_rate_s(
                    current.ep_gp, trial_mechanics.ep_gp, max(mid, np.finfo(float).tiny),
                )
                trial_stress3d = accepted_elastoplastic_stress_3d(trial_mechanics)
                trial_vacancy = preview_vacancy_birth_hit(
                    trial_mechanics, maximum_duration_s=max(mid, np.finfo(float).tiny),
                    accepted_equivalent_plastic_strain_rate_s=trial_epdot[trial_parent],
                    accepted_dislocation_density_m2=np.asarray(trial_mechanics.rho_gp)[trial_parent],
                    accepted_mean_stress_Pa=np.asarray(trial_stress3d["mean_stress_Pa"])[trial_parent],
                )
                if growth_config is None or growth_config.vacancy_clustering_birth_enabled:
                    trial_previews.append(trial_vacancy)
                if direct_enabled:
                    from .dislocation_interaction_voiding_v1 import make_physical_event_preview
                    trial_previews.append(make_physical_event_preview(
                        trial_mechanics,
                        maximum_duration_s=max(mid, np.finfo(float).tiny),
                        temperature_K=temperature,
                        accepted_equivalent_plastic_strain_rate_s=trial_epdot,
                        accepted_mean_stress_Pa=np.asarray(trial_stress3d["mean_stress_Pa"]),
                    ))
                if trial_mechanics.void_state is not None:
                    from .embryo_growth_v1 import (
                        make_cavity_loop_event_preview, make_growth_radius_event_preview,
                    )
                    trial_centers = np.asarray(trial_mechanics.mesh.nodes)[
                        np.asarray(trial_mechanics.mesh.elems)].mean(axis=1)
                    trial_packet_boundary_s = math.inf
                    for trial_growth in trial_mechanics.void_state.embryo_growth_states:
                        trial_site = next(x for x in trial_mechanics.void_state.sites
                                          if x.site_id == trial_growth.site_id)
                        if trial_site.phase not in {
                                VoidPhase.EMBRYO, VoidPhase.STABLE_SUBGRID_VOID,
                                VoidPhase.RESOLVED_VOID}:
                            continue
                        if trial_growth.cavity_emission_state.enabled:
                            trial_previews.append(make_cavity_loop_event_preview(
                                trial_mechanics, trial_growth,
                                temperature_K=temperature, maximum_duration_s=mid))
                        trial_eid = int(np.argmin(np.linalg.norm(
                            trial_centers-np.asarray(trial_site.center_m), axis=1)))
                        trial_radius_preview = make_growth_radius_event_preview(
                            trial_mechanics, trial_growth, temperature_K=temperature,
                            accepted_equivalent_plastic_strain_rate_s=float(
                                trial_epdot[trial_eid]),
                            promotion_radius_m=(0.0
                                if trial_site.phase == VoidPhase.EMBRYO
                                else float(history.get(
                                    "promotion_radius_m", VoidingConfig().promotion_radius_m))),
                            local_mesh_size_m=math.sqrt(float(
                                trial_mechanics.mesh.area_e[trial_eid])),
                            remesh_fraction=float(history.get(
                                "embryo_remesh_fraction", 0.25)),
                            maximum_duration_s=mid,
                            transport_trial_after=trial_vacancy.commit_payload[
                                "advanced_transport"],
                            functional_test_supply_area_rate_m2_s=functional_test_supply)
                        trial_previews.append(trial_radius_preview)
                        trial_radial_rate = float(trial_radius_preview.commit_payload.get(
                            "packet_upper_radial_rate_m_s",
                            trial_radius_preview.commit_payload.get(
                                "radial_rate_m_s", 0.0)))
                        trial_area_rate = (trial_radial_rate*2.0*math.pi
                                           *trial_growth.radius_m)
                        if trial_area_rate > 0.0:
                            trial_packet_area = (
                                trial_growth.cavity_emission_state.parameter_row
                                .maximum_packet_area_fraction*trial_growth.area_m2)
                            trial_packet_boundary_s = min(
                                trial_packet_boundary_s,
                                GROWTH_PACKET_SAFETY*trial_packet_area
                                /trial_area_rate)
                    for trial_site in trial_mechanics.void_state.sites:
                        if trial_site.phase != VoidPhase.EMBRYO:
                            continue
                        trial_eid = int(np.argmin(np.linalg.norm(
                            trial_centers-np.asarray(trial_site.center_m), axis=1)))
                        trial_tensor = np.array((
                            (trial_stress3d["sigma_xx_Pa"][trial_eid],
                             trial_stress3d["sigma_xy_Pa"][trial_eid]),
                            (trial_stress3d["sigma_xy_Pa"][trial_eid],
                             trial_stress3d["sigma_yy_Pa"][trial_eid])))
                        trial_lifecycle = arrhenius_rates(
                            VoidingConfig(enabled=True), temperature_K=temperature,
                            stress_tensor_Pa=trial_tensor,
                            normal_xy=trial_site.normal_xy)
                        trial_previews.extend((
                            preview_stabilization(
                                trial_mechanics, trial_site.site_id,
                                trial_lifecycle["stabilization_s"], mid),
                            preview_healing(
                                trial_mechanics, trial_site.site_id,
                                trial_lifecycle["healing_s"], mid)))
                    if (math.isfinite(trial_packet_boundary_s)
                            and trial_packet_boundary_s <= mid):
                        trial_previews.append(PhysicalEventPreview(
                            event_type="GROWTH_PACKET_BOUNDARY",
                            owner_kind="GlobalPhysicalScheduler",
                            owner_id="growth-packet",
                            crossing_time_s=float(trial_packet_boundary_s),
                            accepted_state_fingerprint=_accepted_fingerprint(
                                trial_mechanics),
                            source_fingerprint=_sha((
                                "growth-packet", trial_packet_boundary_s)),
                            topology_fingerprint=_topology_fingerprint(
                                trial_mechanics),
                            event_ids=(f"growth-packet-ramp:{segment_index}",),
                            candidate_ids=(), threshold_identity=((
                                "maximum_area_fraction",
                                float(trial_packet_boundary_s)),),
                            rng_fingerprint=_rng_fingerprint(trial_mechanics),
                            commit_payload=MappingProxyType({
                                "synchronization_only": True}),
                        ))
                trial_winner, trial_tie = select_physical_event(tuple(trial_previews))
                if trial_winner is not None and trial_winner.crossing_time_s <= mid+interval_tolerance:
                    high = mid
                    localized = (trial_mechanics, trial_stress, trial_elements,
                                 trial_K, trial_previews, trial_winner, trial_tie)
                else:
                    low = mid
            elapsed = high
            fraction = (duration-remaining+elapsed)/duration
            if localized is None:
                # No rebuilt interior preview established a crossing.  Never
                # retain the stale endpoint-derived winner in that case.
                elapsed = segment_horizon
                fraction = (duration-remaining+elapsed)/duration
                winner = None
                tie = ()
                previews = tuple(item for item in previews if not item.eligible)
                mechanical = history["mechanical_state_at_fraction"](fraction)
                mechanical = _bind_ramp_owners(mechanical, current)
            else:
                (mechanical, stress_tensor, source_elements, K_by_front,
                 previews, winner, tie) = localized
            growth_packet_boundary = any(
                item.event_type == "GROWTH_PACKET_BOUNDARY" for item in tie)

            # Every quantity consumed by the accepted advance must belong to
            # the localized mechanical state.  In particular, never retain
            # endpoint-ramp plastic increments or a stale active-tip owner
            # after the root bracket has moved to an interior load state.
            active_tip_id = (mechanical.crack_network.active_tip_ids[0]
                             if mechanical.crack_network.active_tip_ids else None)
            transport = current.vacancy_transport_state_v2
            parent = np.asarray(transport.parent_mechanics_element_ids, dtype=int)
            epdot_mechanics = equivalent_plastic_strain_rate_s(
                current.ep_gp, mechanical.ep_gp,
                max(elapsed, np.finfo(float).tiny),
            )
            stress3d = accepted_elastoplastic_stress_3d(mechanical)
            lifecycle_rates_by_site = {}
            if mechanical.void_state is not None:
                localized_centers = np.asarray(mechanical.mesh.nodes)[
                    np.asarray(mechanical.mesh.elems)].mean(axis=1)
                for localized_site in mechanical.void_state.sites:
                    if localized_site.phase != VoidPhase.EMBRYO:
                        continue
                    localized_eid = int(np.argmin(np.linalg.norm(
                        localized_centers-np.asarray(
                            localized_site.center_m), axis=1)))
                    localized_tensor = np.array((
                        (stress3d["sigma_xx_Pa"][localized_eid],
                         stress3d["sigma_xy_Pa"][localized_eid]),
                        (stress3d["sigma_xy_Pa"][localized_eid],
                         stress3d["sigma_yy_Pa"][localized_eid]),
                    ))
                    lifecycle_rates_by_site[localized_site.site_id] = arrhenius_rates(
                        VoidingConfig(enabled=True), temperature_K=temperature,
                        stress_tensor_Pa=localized_tensor,
                        normal_xy=localized_site.normal_xy,
                    )

        external = {
            item.event_type+":"+item.owner_id: item.crossing_time_s
            for item in previews if item.eligible
        }
        transport_wall_start = time.perf_counter()
        transported, transport_audit = advance_production_vacancy_transport_interval(
            current, mechanical, duration_s=max(elapsed, np.finfo(float).tiny),
            external_event_times_s=external,
            transport_linear_solver_provider=history.get(
                "transport_linear_solver_provider"),
        )
        transport_wall_s += time.perf_counter()-transport_wall_start
        failure_injector = history.get("failure_injector")
        if (winner is not None and winner.event_type in {
                "GROWTH_STABLE_RADIUS", "GROWTH_PROMOTION",
                "GROWTH_REMESH_CADENCE"} and callable(failure_injector)):
            failure_injector("radius_root_after_transport", current)
        transport_elapsed = float(transport_audit.get("elapsed_time_s", elapsed))
        if transport_elapsed < elapsed:
            if elapsed-transport_elapsed > max(interval_tolerance, 64*np.spacing(elapsed)):
                raise RuntimeError("transport owner shortened a localized scheduler interval")
            elapsed = transport_elapsed
        if winner is not None and winner.event_type == "VACANCY_BIRTH_HIT":
            # The event locator owns the threshold-qualified transport state.
            # Re-solving independently at the localized root can land one
            # roundoff below the action/inventory threshold and must not turn
            # a selected event into a no-op.  Consume exactly the immutable
            # winning preview before any post-birth growth owner is advanced.
            from .void_formation_mechanism import active_void_formation_mechanism
            mechanism = active_void_formation_mechanism()
            committed_transport = mechanism.commit(
                winner.commit_payload["original_transport"],
                winner.commit_payload["advanced_transport"],
                expected_site_id=winner.commit_payload["site_id"],
            )
            transported = replace(
                transported, vacancy_transport_state_v2=committed_transport)
        # Commit the phase-owned directional process over every accepted
        # elapsed interval, including a nonwinning connected-cavity or child
        # first passage.  Re-preview at the actual shortened interval; copy
        # only clock/process ownership, never stale mechanics, transport, or
        # growth fields from the preview state.
        directional_owner_advance = None
        phase_cavities = (() if mechanical.void_state is None
                          else mechanical.void_state.cavities)
        if (phase_cavities and phase_cavities[0].phase in {
                VoidPhase.CONNECTED_VOID, VoidPhase.DOWNSTREAM_FRONT_ACTIVE}):
            cached_phase_horizon = (None if topology_preview is None else
                                    topology_preview.commit_payload.get(
                                        "maximum_duration_s"))
            if (topology_preview is not None and cached_phase_horizon is not None
                    and math.isclose(float(elapsed),
                                     float(cached_phase_horizon),
                                     rel_tol=0.0,
                                     abs_tol=64*np.spacing(max(
                                         abs(float(elapsed)),
                                         np.finfo(float).tiny)))):
                phase_preview = topology_preview
            else:
                phase_preview = preview_downstream_first_passage(
                    mechanical,
                    continuation=(phase_cavities[0].phase ==
                                  VoidPhase.DOWNSTREAM_FRONT_ACTIVE),
                    temperature_K=temperature, maximum_duration_s=elapsed,
                    functional_test_rate_scale=directional_rate_scale,
                    test_only_functional_qualification=test_only_functional_qualification)
            directional_at_elapsed = phase_preview.commit_payload["directional_preview"]
        elif phase_cavities and phase_cavities[0].phase == VoidPhase.RESOLVED_VOID:
            phase_preview = preview_ligament_first_passage(
                mechanical, temperature_K=temperature,
                maximum_duration_s=elapsed,
                front_load_rows=directional_load_rows,
                functional_test_rate_scale=directional_rate_scale,
                test_only_functional_qualification=test_only_functional_qualification)
            directional_at_elapsed = phase_preview.commit_payload["directional_preview"]
        elif active_tip_id is not None:
            directional_at_elapsed = preview_directional_first_passage(
                mechanical, stress_tensor, source_kind="sharp_front",
                source_front_id=active_tip_id,
                source_position_m=mechanical.crack_network.branch(active_tip_id).tip,
                source_probe_identity={"kind": "global_scheduler", "element_ids": list(source_elements)},
                temperature_K=temperature, maximum_advance_duration_s=elapsed,
                front_load_rows=directional_load_rows,
                functional_test_rate_scale=directional_rate_scale,
                test_only_functional_qualification=test_only_functional_qualification)
        else:
            directional_at_elapsed = None
        if directional_at_elapsed is not None:
            directional_state = directional_at_elapsed.commit_payload["advanced_state"]
            before_action = {item.candidate_id: float(item.action)
                             for item in current.competition.hazard_states}
            after_action = {item.candidate_id: float(item.action)
                            for item in directional_state.competition.hazard_states}
            directional_owner_advance = {
                "owner_kind": directional_at_elapsed.owner_kind,
                "owner_id": directional_at_elapsed.owner_id,
                "elapsed_time_s": float(elapsed),
                "action_before": before_action,
                "action_after": after_action,
                "action_increment": {
                    key: after_action[key]-before_action[key]
                    for key in before_action},
                "event_ids": tuple(directional_at_elapsed.event_ids),
            }
            transported = replace(
                transported, competition=directional_state.competition,
                junction_process_state=directional_state.junction_process_state)
        if missing_front_loads:
            front_process_advances = ()
        else:
            transported, front_process_advances = _advance_front_engines(
                current, transported, elapsed, K_by_front, temperature,
            )
        winning_direct_site = (
            winner.commit_payload["site_id"]
            if winner is not None and winner.event_type == "DISLOCATION_INTERACTION_HIT"
            else None
        )
        if direct_enabled:
            from .dislocation_interaction_voiding_v1 import advance_live_direct_owner
            transported = advance_live_direct_owner(
                mechanical, transported, elapsed_time_s=elapsed,
                temperature_K=temperature,
                accepted_equivalent_plastic_strain_rate_s=epdot_mechanics,
                accepted_mean_stress_Pa=np.asarray(stress3d["mean_stress_Pa"]),
                winning_site_id=winning_direct_site,
            )
        # A born embryo/cavity owns one continuous mechanism-resolved growth
        # law on the same elapsed interval as transport and every losing
        # fracture clock. The uncalibrated cavity-emission row is opt-in; a
        # nonzero vacancy rate must be accompanied by the transport owner's
        # exact inventory debit for this segment.
        embryo_growth_audits = []
        preexisting_growth_site_ids = set(
            () if current.void_state is None else
            (item.site_id for item in current.void_state.embryo_growth_states))
        growth_owners = (() if transported.void_state is None else
                         transported.void_state.embryo_growth_states)
        for growth_owner in growth_owners:
            if elapsed <= 0.0:
                # Immediate topology/lifecycle events carry no physical time;
                # continuous transport/growth owners must not be advanced with
                # a fabricated positive certificate interval.
                continue
            if growth_owner.site_id not in preexisting_growth_site_ids:
                # Birth occurs at the end of this accepted scheduler segment;
                # the new owner must not be evolved retroactively over the
                # pre-birth portion of the interval.
                continue
            production_site = next(item for item in transported.void_state.sites
                                   if item.site_id == growth_owner.site_id)
            if production_site.phase not in {
                VoidPhase.EMBRYO, VoidPhase.STABLE_SUBGRID_VOID,
                VoidPhase.RESOLVED_VOID,
            }:
                continue
            centers = np.asarray(transported.mesh.nodes)[np.asarray(transported.mesh.elems)].mean(axis=1)
            eid = int(np.argmin(np.linalg.norm(
                centers-np.asarray(production_site.center_m), axis=1,
            )))
            local_epdot = float(epdot_mechanics[eid])
            coefficient = float(history.get("embryo_bulk_accommodation_coefficient", 1.0))
            bulk_area_rate = 2.0*math.pi*growth_owner.radius_m**2*coefficient*local_epdot
            from .embryo_growth_v1 import (
                RESOLVED, advance_live_embryo_growth, source_local_fem_tensors,
                transport_flux_certificate,
            )
            owner_kind = "ACTIVE_CAVITY" if growth_owner.representation == RESOLVED else "SUBGRID_SITE"
            owner_id = growth_owner.cavity_id if owner_kind == "ACTIVE_CAVITY" else growth_owner.site_id
            certificate = transport_flux_certificate(
                current.vacancy_transport_state_v2,
                transported.vacancy_transport_state_v2,
                owner_id=owner_id, owner_kind=owner_kind)
            certified_transport_after = transported.vacancy_transport_state_v2
            if owner_kind == "SUBGRID_SITE" and certificate.raw_absorbed_area_m2 > 0.0:
                from .vacancy_transport_phase_field_v2 import transfer_reserved_embryo_to_growth
                transferred_transport, transferred_area = transfer_reserved_embryo_to_growth(
                    transported.vacancy_transport_state_v2, growth_owner.site_id)
                if not math.isclose(transferred_area, certificate.raw_absorbed_area_m2,
                                    rel_tol=2e-12, abs_tol=1e-30):
                    raise RuntimeError("post-birth vacancy capture was not atomically transferred")
                transported = replace(transported,
                                      vacancy_transport_state_v2=transferred_transport)
            source_tensor, source_sigma_zz = source_local_fem_tensors(transported, growth_owner)
            configuration = transported.void_state.embryo_growth_configuration
            loop_enabled = (growth_owner.cavity_emission_state.enabled if configuration is None else
                            configuration.cavity_surface_loop_emission_enabled)
            radius_owner_preview = next((
                item for item in previews
                if item.owner_kind == "EmbryoGrowthProcessOwner"
                and item.owner_id == growth_owner.site_id
                and "source_regimes" in item.commit_payload
            ), None)
            source_regimes = ({} if radius_owner_preview is None else
                              dict(radius_owner_preview.commit_payload[
                                  "source_regimes"]))
            def solve_growth_transport_packet(packet_live, packet_dt):
                """Re-solve transport after each accepted mechanics/radius packet."""
                from .vacancy_transport_phase_field_v2 import (
                    accepted_elastoplastic_stress_3d as packet_stress_3d,
                    solve_transport_interval,
                    transfer_reserved_embryo_to_growth,
                )
                packet_before_transport = packet_live.vacancy_transport_state_v2
                packet_parent = np.asarray(
                    packet_before_transport.parent_mechanics_element_ids, int)
                packet_stress = packet_stress_3d(packet_live)
                # The accepted plastic source rate belongs to this global
                # interval; stress and dislocation density are refreshed from
                # the just-equilibrated packet state.
                packet_epdot = np.asarray(epdot_mechanics, float)[packet_parent]
                packet_rho = np.asarray(packet_live.rho_gp, float)[packet_parent]
                packet_mean = np.asarray(packet_stress["mean_stress_Pa"], float)[packet_parent]
                packet_after_transport = solve_transport_interval(
                    packet_before_transport,
                    packet_before_transport.transport_mesh,
                    packet_before_transport.surface_registry,
                    row=packet_before_transport.parameter_row,
                    duration_s=packet_dt,
                    accepted_equivalent_plastic_strain_rate_s=packet_epdot,
                    accepted_dislocation_density_m2=packet_rho,
                    accepted_mean_stress_Pa=packet_mean,
                    linear_solver_provider=history.get(
                        "transport_linear_solver_provider"),
                )
                committed_transport = packet_after_transport
                if owner_kind == "SUBGRID_SITE":
                    committed_transport, _ = transfer_reserved_embryo_to_growth(
                        packet_after_transport, growth_owner.site_id)
                return (replace(packet_live,
                                vacancy_transport_state_v2=committed_transport),
                        packet_before_transport, packet_after_transport)
            growth_kwargs = dict(
                site_id=growth_owner.site_id, duration_s=elapsed,
                transport_certificate=certificate,
                transport_before_state=current.vacancy_transport_state_v2,
                transport_after_state=certified_transport_after,
                bulk_accommodation_area_rate_m2_s=bulk_area_rate,
                direct_interaction_area_rate_m2_s=functional_test_supply,
                stress_tensor_Pa=source_tensor,
                sigma_zz_Pa=source_sigma_zz,
                temperature_K=temperature,
                support_radius_m=float(history.get("embryo_plastic_kernel_radius_m", 5.0e-5)),
                local_mesh_size_m=math.sqrt(float(transported.mesh.area_e[eid])),
                remesh_fraction=float(history.get("embryo_remesh_fraction", 0.25)),
                # A radius-root trial may move past the remesh cadence.  Keep
                # topology publication deferred until the coupled radius has
                # been verified against the frozen target below.
                remesh_callback=(
                    (lambda mechanical_state, _before, _after: mechanical_state)
                    if winner is not None
                    and winner.event_type == "GROWTH_REMESH_CADENCE"
                    else history.get("commit_embryo_growth_remesh")),
                loop_emission_enabled=loop_enabled,
                source_tensor_observer=source_local_fem_tensors,
                equilibrium_callback=history.get("mechanical_equilibrium_provider"),
                transport_packet_solver=solve_growth_transport_packet,
                failure_injector=history.get("failure_injector"),
                forced_regimes_by_source=source_regimes,
                forced_regime=("LOW_COUNT_EXACT" if winner is not None
                    and winner.event_type == "CAVITY_LOOP_EMISSION"
                    and winner.owner_id == growth_owner.site_id else None))
            if (winner is not None and winner.event_type == "CAVITY_LOOP_EMISSION"
                    and winner.owner_id == growth_owner.site_id):
                from .embryo_growth_v1 import EmbryoGrowthProcessOwner
                growth_process_owner = (EmbryoGrowthProcessOwner()
                    if configuration is None else EmbryoGrowthProcessOwner(configuration))
                transported, growth_audit = growth_process_owner.commit_loop_event(
                    mechanical, winner,
                    transport_committed_live_state=transported,
                    **growth_kwargs)
            else:
                transported, growth_audit = advance_live_embryo_growth(
                    transported, **growth_kwargs)
            if int(growth_audit["packet_count"]) != 1:
                packet_rows = tuple(growth_audit.get("packets", ()))
                raise RuntimeError(
                    "an embryo-growth packet bypassed the global scheduler boundary: "
                    f"selected_event={selected_type!r}, elapsed_s={elapsed!r}, "
                    f"packet_count={growth_audit['packet_count']!r}, "
                    f"first_packet_duration_s="
                    f"{(packet_rows[0].get('packet_duration_s') if packet_rows else None)!r}, "
                    f"last_packet_duration_s="
                    f"{(packet_rows[-1].get('packet_duration_s') if packet_rows else None)!r}, "
                    f"maximum_packet_area_fraction="
                    f"{max((row.get('applied_packet_area_fraction', 0.0) for row in packet_rows), default=0.0)!r}")
            embryo_growth_audits.append({
                "site_id": growth_owner.site_id, **dict(growth_audit),
            })
        transported = _sync_committed_transport_hits(current, transported)
        selected_type = None if winner is None else winner.event_type
        radius_transition_verified = False
        verified_radius_m = None
        verified_target_radius_m = None
        for lifecycle_site_id, lifecycle_rates in lifecycle_rates_by_site.items():
            site = next(item for item in transported.void_state.sites
                        if item.site_id == lifecycle_site_id)
            evolved_void, lifecycle_events = advance_site(
                transported.void_state, site.site_id, elapsed,
                rates=lifecycle_rates,
            )
            transported = replace(transported, void_state=evolved_void)
            updated_site = next(item for item in evolved_void.sites
                                if item.site_id == site.site_id)
            if evolved_void.embryo_growth_states:
                synchronized = tuple(
                    replace(item, stabilization_clock=updated_site.stabilization,
                            healing_clock=updated_site.healing)
                    if item.site_id == site.site_id else item
                    for item in evolved_void.embryo_growth_states
                )
                transported = replace(transported, void_state=replace(
                    transported.void_state, embryo_growth_states=synchronized,
                ))
            stabilized = any(event in {
                "STABILIZED", "HEALING_PROHIBITED_RESIDUAL_DEFECT",
            } for event in lifecycle_events)
            if stabilized:
                from .vacancy_transport_phase_field_v2 import (
                    stabilize_growth_owned_inventory, stabilize_reserved_embryo,
                )
                transport_site = next(item for item in transported.vacancy_transport_state_v2.site_states
                                      if item.site_id == site.site_id)
                if transport_site.reserved_inventory_area_m2 > 0.0:
                    transported = replace(
                        transported,
                        vacancy_transport_state_v2=stabilize_reserved_embryo(
                            transported.vacancy_transport_state_v2, site.site_id,
                        ),
                    )
                else:
                    growth_inventory = next(
                        item.vacancy_raw_captured_area_m2
                        -item.vacancy_returned_or_annihilated_area_m2
                        for item in transported.void_state.embryo_growth_states
                        if item.site_id == site.site_id)
                    if growth_inventory > 0.0:
                        transported = replace(
                            transported,
                            vacancy_transport_state_v2=stabilize_growth_owned_inventory(
                                transported.vacancy_transport_state_v2, growth_inventory))
                if not any(c.parent_site_id == site.site_id
                           for c in transported.void_state.cavities):
                    from .voiding_v5 import create_subgrid_cavity
                    growth = next(item for item in transported.void_state.embryo_growth_states
                                  if item.site_id == site.site_id)
                    transported = replace(
                        transported,
                        void_state=create_subgrid_cavity(
                            transported.void_state, site.site_id, growth.radius_m,
                        ),
                    )
            elif "HEALED" in lifecycle_events:
                from .embryo_growth_v1 import heal_embryo
                from .vacancy_transport_phase_field_v2 import (
                    heal_embryo_reservation, heal_growth_owned_inventory,
                )
                healed_owners = []
                returned_area = 0.0
                for item in transported.void_state.embryo_growth_states:
                    if item.site_id == site.site_id:
                        healed, healing_audit = heal_embryo(item)
                        healed_owners.append(healed)
                        returned_area = float(healing_audit["returned_vacancy_area_m2"])
                    else:
                        healed_owners.append(item)
                transport_site = next(x for x in transported.vacancy_transport_state_v2.site_states
                                      if x.site_id == site.site_id)
                if transport_site.reserved_inventory_area_m2 > 0.0:
                    healed_transport = heal_embryo_reservation(
                        transported.vacancy_transport_state_v2,
                        transported.vacancy_transport_state_v2.transport_mesh,
                        site.site_id, disposition="RETURN_TO_FIELD")
                else:
                    healed_transport = heal_growth_owned_inventory(
                        transported.vacancy_transport_state_v2,
                        transported.vacancy_transport_state_v2.transport_mesh,
                        site.site_id, returned_area, disposition="RETURN_TO_FIELD")
                transported = replace(
                    transported,
                    vacancy_transport_state_v2=healed_transport,
                    void_state=replace(
                        transported.void_state,
                        embryo_growth_states=tuple(healed_owners),
                    ),
                )
        if winner is not None and winner.event_type in {
                "GROWTH_STABLE_RADIUS", "GROWTH_PROMOTION", "GROWTH_REMESH_CADENCE"}:
            target_radius = float(winner.commit_payload["target_radius_m"])
            committed_growth = next(
                x for x in transported.void_state.embryo_growth_states
                if x.site_id == winner.owner_id)
            radius_tolerance = max(1e-12*target_radius, 64*np.spacing(target_radius))
            overdue_promotion = (
                winner.event_type == "GROWTH_PROMOTION"
                and winner.crossing_time_s == 0.0
                and committed_growth.radius_m >= target_radius-radius_tolerance)
            if overdue_promotion:
                radius_transition_verified = True
                verified_radius_m = float(committed_growth.radius_m)
                verified_target_radius_m = target_radius
            elif committed_growth.radius_m < target_radius-radius_tolerance:
                # The linear preview did not survive the coupled transport /
                # mechanics transaction.  Treat this accepted state as a new
                # global bracket boundary and rebuild every preview.
                winner = None
                selected_type = None
                growth_packet_boundary = True
            elif committed_growth.radius_m > target_radius+radius_tolerance:
                # Replay from the immutable accepted interval-start state with
                # a smaller horizon.  ``transported`` is purely local and is
                # discarded here, so transport, growth, mechanics and every
                # losing clock are rolled back together.
                radius_retry_count += 1
                radius_retry_total += 1
                if radius_retry_count > 64:
                    raise RuntimeError(
                        "coupled nonlinear radius root replay exceeded 64 brackets")
                segment_retry_cap = 0.5*elapsed
                if segment_retry_cap <= max(
                        np.finfo(float).tiny, 8*np.spacing(elapsed)):
                    raise RuntimeError("coupled nonlinear radius root replay underflow")
                failure_injector = history.get("failure_injector")
                if callable(failure_injector):
                    failure_injector("radius_root_overshoot", current)
                segment_index -= 1
                continue
            else:
                radius_transition_verified = True
                verified_radius_m = float(committed_growth.radius_m)
                verified_target_radius_m = target_radius
            if (radius_transition_verified
                    and winner.event_type != "GROWTH_PROMOTION"):
                failure_injector = history.get("failure_injector")
                if callable(failure_injector):
                    failure_injector(
                        "radius_root_before_topology_publication", current)
                    failure_injector("radius_transition_commit", current)
                from .embryo_growth_v1 import EmbryoGrowthProcessOwner
                transition = ({
                    "GROWTH_STABLE_RADIUS": "R_STABLE",
                    "GROWTH_REMESH_CADENCE": "R_REMESH_CADENCE",
                })[winner.event_type]
                if winner.event_type == "GROWTH_STABLE_RADIUS":
                    transported, _ = _commit_waiting_stabilization_at_radius(
                        transported, winner.owner_id)
                    committed_growth = next(
                        x for x in transported.void_state.embryo_growth_states
                        if x.site_id == winner.owner_id)
                elif winner.event_type == "GROWTH_REMESH_CADENCE":
                    remesh_callback = history.get("commit_embryo_growth_remesh")
                    if not callable(remesh_callback):
                        raise RuntimeError(
                            "resolved radius root requires the existing remesh callback")
                    before_growth = next(
                        x for x in current.void_state.embryo_growth_states
                        if x.site_id == winner.owner_id)
                    transported = remesh_callback(
                        transported, before_growth, committed_growth)
                    committed_growth = next(
                        x for x in transported.void_state.embryo_growth_states
                        if x.site_id == winner.owner_id)
                committed_growth = EmbryoGrowthProcessOwner().commit_radius_transition(
                    committed_growth, event_type=transition)
                transported = replace(transported, void_state=replace(
                    transported.void_state,
                    embryo_growth_states=tuple(
                        committed_growth if x.site_id == winner.owner_id else x
                        for x in transported.void_state.embryo_growth_states)))
        topology_transaction_audit = None
        if winner is not None and winner.event_type == "LIGAMENT_RUPTURE":
            from .voiding_production_v5 import ligament_transaction
            advanced = winner.commit_payload["advanced_state"]
            advanced = replace(
                advanced, vacancy_transport_state_v2=transported.vacancy_transport_state_v2,
                void_state=transported.void_state,
            )
            transported, ligament_audit = ligament_transaction(
                transported,
                _precompleted_directional=(
                    advanced, list(copy.deepcopy(winner.commit_payload["audit"])),
                ),
            )
            topology_transaction_audit = {
                "accepted":bool(ligament_audit.accepted),
                "energy_release_J_per_m":float(ligament_audit.energy_release_J_per_m),
                "hazard_dissipation_J_per_m":float(
                    ligament_audit.hazard_dissipation_J_per_m),
                "energy_margin_J_per_m":float(ligament_audit.energy_margin_J_per_m),
                "reason":str(ligament_audit.reason),
                "event_type":"LIGAMENT_RUPTURE",
                "event_ids":tuple(winner.event_ids),
                "candidate_ids":tuple(winner.candidate_ids),
            }
        elif winner is not None and winner.event_type in {
            "DOWNSTREAM_FIRST_PASSAGE", "CHILD_CONTINUATION",
        }:
            from .voiding_production_v5 import downstream_front_transaction
            advanced = replace(
                winner.commit_payload["advanced_state"],
                vacancy_transport_state_v2=transported.vacancy_transport_state_v2,
                void_state=transported.void_state,
                tip_process_state=transported.tip_process_state,
            )
            source = winner.commit_payload["source"]
            front_process = next((item for owner, item in front_process_advances
                                  if owner == source.get("source_front_id")), None)
            transported, _, _, _ = downstream_front_transaction(
                transported, continuation=bool(winner.commit_payload["continuation"]),
                _precompleted_directional=(
                    advanced, list(copy.deepcopy(winner.commit_payload["audit"])),
                ),
                _precompleted_front_process=front_process,
            )
        elif winner is not None and winner.event_type == "GROWTH_PROMOTION":
            callback = history.get("commit_growth_promotion")
            if not callable(callback):
                raise RuntimeError("growth promotion requires the existing topology commit callback")
            failure_injector = history.get("failure_injector")
            if callable(failure_injector):
                failure_injector(
                    "radius_root_before_topology_publication", current)
            transported = callback(transported, winner)
        elif winner is not None and winner.event_type == "DIRECTIONAL_CLEAVAGE":
            from .crack_network_v11 import ROOT_BRANCH_ID
            from .v12_production_driver import (
                build_physical_cleavage_event_context,execute_event)
            from .voiding_production_v5 import _bundle_from_state, fresh_sharp_front_engine
            failure_injector = history.get("failure_injector")
            if callable(failure_injector):
                failure_injector(
                    "directional_cleavage_before_topology_publication", current)
            candidate_id = winner.candidate_ids[0]
            candidate = next(item for item in transported.competition.candidates
                             if item.candidate_id == candidate_id)
            winner_row = next(
                row for row in winner.commit_payload["audit"]
                if row["candidate_id"] == candidate_id)
            hazard_barrier_J = float(winner_row["hazard_barrier_J"])
            engine = fresh_sharp_front_engine(transported.material, _bundle_from_state(transported))
            start = np.asarray(transported.crack_network.branch(ROOT_BRANCH_ID).tip)
            endpoint = tuple(map(float, start + float(engine.f.da)*np.asarray(candidate.direction_xy)))
            physical_context=build_physical_cleavage_event_context(
                transported,candidate_id=candidate_id,front_id=ROOT_BRANCH_ID,
                source_fingerprint=winner.source_fingerprint,
                event_ids=winner.event_ids,hazard_barrier_J=hazard_barrier_J)
            try:
                transported, topology_transaction_audit = execute_event(
                    transported, endpoint,
                    transaction_identity="global-physical-directional-cleavage",
                    candidate_id=candidate_id,
                    physical_context=physical_context,
                )
            except Exception as error:
                from .v12_production_driver import PhysicalEnergyVeto
                if not isinstance(error,PhysicalEnergyVeto):
                    raise
                veto_audit=dict(error.audit)
                veto_audit.update({
                    "elapsed_time_s":elapsed,"selected_event":selected_type,
                    "selected_owner":(winner.owner_kind,winner.owner_id),
                    "all_losing_clocks_advance_s":elapsed,
                    "requested_segment_horizon_s":segment_horizon,
                    "front_process_advances":tuple(
                        dict(item.audit) for _,item in front_process_advances),
                    "front_process_load_policy":front_process_load_policy,
                })
                raise PhysicalEnergyVeto(error.accepted_state,veto_audit) from None
        events.append({
            "segment_index": segment_index, "elapsed_time_s": elapsed,
            "input_state_fingerprint": segment_input_state_fingerprint,
            "output_state_fingerprint": _accepted_fingerprint(transported),
            "selected_event": selected_type,
            "selected_owner": None if winner is None else (winner.owner_kind, winner.owner_id),
            "tie_group": tuple((item.event_type, item.owner_kind, item.owner_id) for item in tie),
            "eligible_previews": tuple({
                "event_type": item.event_type, "owner_kind": item.owner_kind,
                "owner_id": item.owner_id, "crossing_time_s": item.crossing_time_s,
                "event_ids": item.event_ids, "candidate_ids": item.candidate_ids,
            } for item in previews),
            "all_losing_clocks_advance_s": elapsed,
            "legacy_front_process": tuple(dict(item.audit) for _, item in front_process_advances),
            "front_process_load_policy": front_process_load_policy,
            "directional_owner_advance": directional_owner_advance,
            "topology_transaction": topology_transaction_audit,
            "transport": dict(transport_audit),
            "lifecycle_rates_used": {
                site_id: dict(rates)
                for site_id, rates in lifecycle_rates_by_site.items()
            },
            "selected_lifecycle_preview_rate_s": (
                None if winner is None or winner.event_type not in {
                    "STABILIZATION", "HEALING"} else
                float(winner.commit_payload["rate_s"])),
            "embryo_growth": tuple(embryo_growth_audits),
            "preview_nonmutating": preview_nonmutating,
            "all_owners_previewed": all_owners_previewed,
            "no_growth_owner_advanced_twice": (
                len({item["site_id"] for item in embryo_growth_audits})
                == len(embryo_growth_audits)),
            "growth_packet_global_scheduler_boundary": growth_packet_boundary,
            "complete_preview_set_rebuilt_after_boundary": growth_packet_boundary,
            "nonlinear_radius_replay_count": radius_retry_count,
            "radius_transition_verified": radius_transition_verified,
            "verified_radius_m": verified_radius_m,
            "verified_target_radius_m": verified_target_radius_m,
        })
        current = transported
        remaining = max(remaining-elapsed, 0.0)
        segment_retry_cap = math.inf
        radius_retry_count = 0
        if winner is None and not growth_packet_boundary:
            break
        # A selected event is already consumed by its exact owner advance.  A
        # topology-specific caller can use the frozen preview commit adapters;
        # this interval API records the commit and continues from that state.
        if (elapsed <= np.finfo(float).tiny and remaining > 0.0
                and events[-1]["input_state_fingerprint"]
                == events[-1]["output_state_fingerprint"]):
            raise RuntimeError("zero-time physical event would stall global scheduler")
        maximum_transactions = int(history.get(
            "maximum_transactions_per_call", 0))
        if maximum_transactions < 0:
            raise ValueError("maximum transactions per call must be nonnegative")
        if maximum_transactions and segment_index >= maximum_transactions:
            # ``current`` is a fully committed state here: mechanics,
            # transport, inventories, losing clocks, and the selected winner
            # (if any) have all advanced through the same elapsed time.
            break
    return current, MappingProxyType({
        "schema": SCHEMA, "requested_duration_s": duration,
        "advanced_duration_s": duration-remaining,
        "events": tuple(events),
        "all_owners_previewed": all(item["all_owners_previewed"] for item in events),
        "continuous_emission_advanced": bool(K_by_front),
        "discrete_emission_event_exists": any(
            preview["event_type"] == "CAVITY_LOOP_EMISSION"
            for event in events for preview in event["eligible_previews"]),
        "preview_nonmutating": all(item["preview_nonmutating"] for item in events),
        "no_owner_advanced_twice": all(
            item["no_growth_owner_advanced_twice"] for item in events),
        "nonlinear_radius_replay_total": radius_retry_total,
        "wall_profile_s": {
            "transport": transport_wall_s,
            "scheduler_total": time.perf_counter()-scheduler_wall_start,
        },
    })


def advance_active_coupled_interval(
    accepted_state: Any, duration_s: float, *, temperature_K: float,
    directional_observation_provider,
    mechanical_equilibrium_provider=None,
    transport_linear_solver_provider=None,
    maximum_stagger_iterations: int = 80,
):
    """Advance one event-free production interval with every owner active.

    This is the reusable physical-time operation used after a numerical
    topology handoff.  It first performs the established timed bulk
    constitutive update, then obtains a state-bound directional observation,
    and finally advances transport, growth, the front engine, and all event
    clocks on that same interval.  A structural event inside the proposed
    interval is rejected without publication: callers must localize it and
    repeat the constitutive update for the consumed time instead of carrying a
    future plastic endpoint across changed geometry.
    """
    from .bulk_plasticity_v104 import PlasticityScope, accepted_bulk_load_step

    duration=float(duration_s)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("active coupled interval duration must be positive")
    before=_accepted_fingerprint(accepted_state)
    mechanical,plastic=accepted_bulk_load_step(
        accepted_state,scope=PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO,
        load_factor=1.0,duration_s=duration,temperature_K=float(temperature_K),
        maximum_stagger_iterations=int(maximum_stagger_iterations),
        equilibrium_provider=mechanical_equilibrium_provider)
    observation=directional_observation_provider(mechanical)
    if not isinstance(observation,(tuple,list)) or len(observation)!=4:
        raise TypeError("directional observation provider must return tensor, K, audit, rows")
    tensor,K,drive_audit,load_rows=observation
    active=tuple(mechanical.crack_network.active_tip_ids)
    if len(active)!=1:
        raise RuntimeError("active coupled interval requires exactly one active front")
    front_id=str(active[0])
    history={
        "kind":"hold","temperature_K":float(temperature_K),
        "maximum_events":64,
        "stress_tensor_Pa":np.asarray(tensor,float),
        "K_effective_by_front":{front_id:float(K)},
        "directional_load_rows_by_front":{front_id:load_rows},
    }
    if mechanical_equilibrium_provider is not None:
        history["mechanical_equilibrium_provider"]=mechanical_equilibrium_provider
    if transport_linear_solver_provider is not None:
        history["transport_linear_solver_provider"]=transport_linear_solver_provider
    trial,scheduler=advance_global_physical_interval(
        accepted_state,mechanical,duration,history)
    structural={"DIRECTIONAL_CLEAVAGE","LIGAMENT_RUPTURE",
                "DOWNSTREAM_FIRST_PASSAGE","CHILD_CONTINUATION",
                "GROWTH_PROMOTION"}
    selected=tuple(row["selected_event"] for row in scheduler["events"]
                   if row["selected_event"] is not None)
    interrupted=tuple(item for item in selected if item in structural)
    if interrupted:
        if _accepted_fingerprint(accepted_state)!=before:
            raise RuntimeError("isolated interrupted active interval mutated its input")
        raise RuntimeError(
            "active coupled interval requires event-time constitutive rebasing: "+
            repr(interrupted))
    front_rows=tuple(item for segment in scheduler["events"]
                     for item in segment["legacy_front_process"])
    if not front_rows:
        raise RuntimeError("active coupled interval skipped the front-process owner")
    if any(segment["front_process_load_policy"]=="DIAGNOSTIC_FREEZE"
           for segment in scheduler["events"]):
        raise RuntimeError("active coupled interval used a diagnostic front freeze")
    return trial,MappingProxyType({
        "schema":"unified-2d.active-coupled-interval/1",
        "requested_duration_s":duration,
        "advanced_duration_s":scheduler["advanced_duration_s"],
        "input_state_fingerprint":before,
        "output_state_fingerprint":_accepted_fingerprint(trial),
        "bulk_plasticity":plastic,
        "directional_drive":drive_audit,
        "front_id":front_id,
        "front_process_advances":front_rows,
        "selected_events":selected,
        "scheduler":scheduler,
        "all_physical_owners_active":True,
        "diagnostic_freeze_front_process":False,
    })


class EventAwareCapabilityStop(RuntimeError):
    """No incomplete physical prefix is published when localization fails."""


def advance_event_aware_coupled_interval(
    accepted_state: Any, duration_s: float, *, temperature_K: float,
    directional_observation_provider,
    mechanical_equilibrium_provider=None,
    transport_linear_solver_provider=None,
    provider_transition_manager=None,
    provider_builder=None,
    accepted_topology_qualification_provider=None,
    maximum_stagger_iterations: int = 80,
    functional_test_directional_rate_scale: float = 1.0,
    test_only_functional_qualification: bool = False,
    maximum_localization_replays: int = 12,
):
    """Advance, localize one internal event, decide it, and own the remainder.

    Every trial starts from the same immutable accepted state and recomputes
    the native constitutive endpoint for its actual prefix duration.  Trial
    results are discarded until the event owner and consumed time reproduce
    at the prefix boundary.  Energy veto is a typed terminal physical outcome;
    accepted topology changes alone may rebuild numerical providers.
    """
    from .bulk_plasticity_v104 import PlasticityScope,accepted_bulk_load_step
    from .v12_production_driver import PhysicalEnergyVeto

    requested=float(duration_s)
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError("event-aware interval duration must be positive")
    original_fingerprint=_accepted_fingerprint(accepted_state)
    tolerance=max(8*np.finfo(float).eps*requested,min(TIME_TOLERANCE_S,requested*1e-8))

    def evaluate(horizon):
        mechanical,plastic=accepted_bulk_load_step(
            accepted_state,scope=PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO,
            load_factor=1.0,duration_s=float(horizon),temperature_K=float(temperature_K),
            maximum_stagger_iterations=int(maximum_stagger_iterations),
            equilibrium_provider=mechanical_equilibrium_provider)
        tensor,K,drive,rows=directional_observation_provider(mechanical)
        active=tuple(mechanical.crack_network.active_tip_ids)
        if len(active)>1:
            raise EventAwareCapabilityStop(
                "event-aware interval requires at most one active front")
        front_id=None if not active else str(active[0])
        history={
            "kind":"hold","temperature_K":float(temperature_K),
            "maximum_events":64,"maximum_transactions_per_call":1,
            "stress_tensor_Pa":np.asarray(tensor,float),
            "K_effective_by_front":({} if front_id is None else {front_id:float(K)}),
            "directional_load_rows_by_front":({} if front_id is None else {front_id:rows}),
            "functional_test_directional_rate_scale":float(
                functional_test_directional_rate_scale),
            "test_only_functional_qualification":bool(
                test_only_functional_qualification),
        }
        if mechanical_equilibrium_provider is not None:
            history["mechanical_equilibrium_provider"]=mechanical_equilibrium_provider
        if transport_linear_solver_provider is not None:
            history["transport_linear_solver_provider"]=transport_linear_solver_provider
        try:
            state,audit=advance_global_physical_interval(
                accepted_state,mechanical,float(horizon),history)
        except PhysicalEnergyVeto as veto:
            return {
                "outcome":"ENERGY_VETO","state":veto.accepted_state,
                "consumed_s":float(veto.audit["elapsed_time_s"]),
                "signature":("ENERGY_VETO",str(veto.audit["selected_event"]),
                             tuple(veto.audit["selected_owner"])),
                "event_audit":veto.audit,"scheduler":None,
                "bulk_plasticity":plastic,"directional_drive":drive,
            }
        selected=tuple(row["selected_event"] for row in audit["events"]
                       if row["selected_event"] is not None)
        selected_identities=tuple(
            (item["event_type"],item["owner_kind"],item["owner_id"],
             item["event_ids"],item["candidate_ids"])
            for item in _selected_event_identities(audit)
            if item["decision"] == "ACCEPTED")
        packet=any(row["growth_packet_global_scheduler_boundary"]
                   for row in audit["events"])
        signature=("ACCEPTED_EVENT",selected_identities) if selected else (
            ("GROWTH_PACKET_BOUNDARY",) if packet else ())
        return {
            "outcome":"ACCEPTED_EVENT" if signature else "COMPLETED_NO_EVENT",
            "state":state,"consumed_s":float(audit["advanced_duration_s"]),
            "signature":signature,"event_audit":None,"scheduler":audit,
            "bulk_plasticity":plastic,"directional_drive":drive,
        }

    trial=evaluate(requested)
    trial_count=1
    if trial["outcome"]=="COMPLETED_NO_EVENT":
        return trial["state"],MappingProxyType({
            "schema":"unified-2d.event-aware-coupled-interval/1",
            "outcome":"COMPLETED_NO_EVENT","requested_duration_s":requested,
            "consumed_duration_s":requested,"remaining_duration_s":0.0,
            "total_consumed_duration_s":requested,
            "total_unconsumed_duration_s":0.0,
            "localization_replays":0,"trial_evaluations":trial_count,
            "bulk_plasticity":trial["bulk_plasticity"],
            "directional_drive":trial["directional_drive"],
            "scheduler":trial["scheduler"],"provider_rebuild_count":0,
            "native_topology_transition_count":0,
            "accepted_event_segments":(),
        })
    target=float(trial["consumed_s"])
    signature=trial["signature"]
    if not (0.0 < target <= requested+tolerance):
        raise EventAwareCapabilityStop("event localization produced invalid consumed time")
    localized=None
    low=0.0; high=requested
    for replay_index in range(int(maximum_localization_replays)):
        replay=evaluate(target); trial_count+=1
        if (replay["signature"]==signature
                and abs(float(replay["consumed_s"])-target)<=tolerance):
            localized=replay
            break
        if replay["outcome"]=="COMPLETED_NO_EVENT":
            low=max(low,target)
            target=0.5*(low+high)
        else:
            signature=replay["signature"]
            high=min(high,target)
            target=float(replay["consumed_s"])
            if target <= low+tolerance:
                target=0.5*(low+high)
    if localized is None:
        if _accepted_fingerprint(accepted_state)!=original_fingerprint:
            raise RuntimeError("failed localization mutated its accepted input")
        raise EventAwareCapabilityStop(
            "coupled event-time constitutive replay did not converge")
    consumed=float(localized["consumed_s"])
    remaining=max(requested-consumed,0.0)
    if localized["outcome"]=="ENERGY_VETO":
        return localized["state"],MappingProxyType({
            "schema":"unified-2d.event-aware-coupled-interval/1",
            "outcome":"ENERGY_VETO","terminal":True,
            "requested_duration_s":requested,"consumed_duration_s":consumed,
            "remaining_duration_s":remaining,
            "total_consumed_duration_s":consumed,
            "total_unconsumed_duration_s":remaining,
            "localized_signature":signature,
            "localization_replays":trial_count-1,"trial_evaluations":trial_count,
            "bulk_plasticity":localized["bulk_plasticity"],
            "directional_drive":localized["directional_drive"],
            "energy_decision":localized["event_audit"],
            "provider_rebuild_count":0,"geometry_published":False,
            "native_topology_transition_count":0,
            "accepted_event_segments":(),
            "terminal_event_identity":MappingProxyType({
                "event_type":localized["event_audit"].get("selected_event"),
                "owner":tuple(localized["event_audit"].get("selected_owner", ())),
                "event_ids":tuple(localized["event_audit"].get("event_ids", ())),
                "candidate_ids":tuple(localized["event_audit"].get("candidate_ids", ())),
                "decision":"ENERGY_VETO",
            }),
            "native_veto_policy":"TERMINAL_PENDING_EVENT_AT_ACCEPTED_PREGEOMETRY_STATE",
        })
    boundary=localized["state"]
    topology_qualification_audit=None
    raw_dependency_changed=(numerical_provider_dependency_identity(boundary)!=
                            numerical_provider_dependency_identity(accepted_state))
    if raw_dependency_changed and accepted_topology_qualification_provider is not None:
        boundary,topology_qualification_audit=(
            accepted_topology_qualification_provider(boundary))
    dependency_changed=(numerical_provider_dependency_identity(boundary)!=
                        numerical_provider_dependency_identity(accepted_state))
    rebuild_count=0
    native_transition_count=int(dependency_changed)
    mechanics=mechanical_equilibrium_provider
    transport=transport_linear_solver_provider
    transition_audit=None
    manager_snapshot=_provider_manager_snapshot(provider_transition_manager)
    reduced_active=(mechanics is not None or transport is not None)
    if dependency_changed and reduced_active:
        if provider_transition_manager is None or provider_builder is None:
            raise EventAwareCapabilityStop(
                "accepted operator change requires a reduced-provider transition")
        try:
            mechanics,transport,transition_audit=(
                provider_transition_manager.rebuild_after_native_commit(
                    accepted_state,boundary,provider_builder))
        except Exception:
            _restore_provider_manager(provider_transition_manager,manager_snapshot)
            raise
        rebuild_count=1
    try:
        if remaining <= tolerance:
            final=boundary; remainder_audit=None
        else:
            final,remainder_audit=advance_event_aware_coupled_interval(
                boundary,remaining,temperature_K=temperature_K,
                directional_observation_provider=directional_observation_provider,
                mechanical_equilibrium_provider=mechanics,
                transport_linear_solver_provider=transport,
                provider_transition_manager=provider_transition_manager,
                provider_builder=provider_builder,
                accepted_topology_qualification_provider=(
                    accepted_topology_qualification_provider),
                maximum_stagger_iterations=maximum_stagger_iterations,
                functional_test_directional_rate_scale=functional_test_directional_rate_scale,
                test_only_functional_qualification=test_only_functional_qualification,
                maximum_localization_replays=maximum_localization_replays)
            rebuild_count+=int(remainder_audit.get("provider_rebuild_count",0))
            native_transition_count+=int(
                remainder_audit.get("native_topology_transition_count",0))
    except Exception:
        _restore_provider_manager(provider_transition_manager,manager_snapshot)
        raise
    child_consumed=(0.0 if remainder_audit is None else
                    float(remainder_audit["total_consumed_duration_s"]))
    if remainder_audit is None and remaining <= tolerance:
        total_consumed=requested
        total_unconsumed=0.0
    else:
        total_consumed=min(consumed+child_consumed,requested)
        total_unconsumed=max(requested-total_consumed,0.0)
    child_terminal=bool(remainder_audit and remainder_audit.get("terminal",False))
    event_segments=_selected_event_identities(localized["scheduler"])
    if remainder_audit is not None:
        event_segments+=tuple(remainder_audit.get("accepted_event_segments",()))
    aggregate_trials=trial_count+int(
        0 if remainder_audit is None else remainder_audit.get("trial_evaluations",0))
    aggregate_replays=(trial_count-1)+int(
        0 if remainder_audit is None else remainder_audit.get("localization_replays",0))
    outcome=(str(remainder_audit["outcome"]) if child_terminal else
             "ACCEPTED_EVENT_AND_REMAINDER")
    terminal_identity=(None if not child_terminal else
                       remainder_audit.get("terminal_event_identity"))
    terminal_energy=(None if not child_terminal else
                     _terminal_energy_decision(remainder_audit))
    if abs(requested-total_consumed-total_unconsumed) > tolerance:
        raise RuntimeError("event-aware composed interval violates clock conservation")
    return final,MappingProxyType({
        "schema":"unified-2d.event-aware-coupled-interval/1",
        "outcome":outcome,"terminal":child_terminal,
        "requested_duration_s":requested,"consumed_duration_s":total_consumed,
        "remaining_duration_s":total_unconsumed,
        "total_consumed_duration_s":total_consumed,
        "total_unconsumed_duration_s":total_unconsumed,
        "first_event_consumed_duration_s":consumed,
        "remaining_at_first_event_s":remaining,"localized_signature":signature,
        "localization_replays":aggregate_replays,
        "trial_evaluations":aggregate_trials,
        "bulk_plasticity":localized["bulk_plasticity"],
        "directional_drive":localized["directional_drive"],
        "event_scheduler":localized["scheduler"],
        "provider_rebuild_count":rebuild_count,
        "native_topology_transition_count":native_transition_count,
        "provider_transition":transition_audit,
        "accepted_topology_qualification":topology_qualification_audit,
        "accepted_event_segments":event_segments,
        "terminal_event_identity":terminal_identity,
        "energy_decision":terminal_energy,
        "remainder":remainder_audit,
    })


__all__ = [
    "AcceptedIntervalContinuationV1", "PhysicalEventPreview", "SCHEMA", "advance_directional_preview",
    "commit_directional_preview", "commit_front_engine_emission",
    "commit_downstream_first_passage", "commit_ligament_event",
    "commit_vacancy_birth_hit",
    "make_directional_preview", "preview_front_engine_emission",
    "preview_downstream_first_passage", "preview_ligament_first_passage",
    "preview_healing", "preview_resolved_cavity_growth",
    "preview_stabilization", "preview_subgrid_growth_or_promotion",
    "preview_vacancy_birth_hit",
    "select_physical_event", "advance_global_physical_interval",
    "advance_active_coupled_interval",
    "advance_event_aware_coupled_interval","EventAwareCapabilityStop",
    "mechanical_provider_dependency_identity",
    "transport_provider_dependency_identity",
    "numerical_provider_dependency_identity",
    "controlled_scheduler_ownership_matrix",
]
