"""Candidate-owned hybrid directional drive for the V5 downstream front.

The accepted local-J path is inherited from the V11 exact-topology provider.
Only an invalid local contour opens an isolated, fixed-opening V12 virtual
extension trial.  No trial owns or advances a kinetic clock.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import pickle
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .directional_competition_v11 import CleavageCandidate
from .fem import assemble_mechanics
from .live_topology_kernel_v11 import LiveTopologyRequest, evaluate_exact_topology
from .topology_transaction_v11 import (
    LiveFEMTopologyState,
    TopologyArm,
    apply_v12_production_trial_geometry,
    complete_accepted_state_fingerprint,
    equilibrate_fixed_load_with_production_fem,
)


MODEL_ID = "v5.hybrid-candidate-directional-drive/1"
K_INTERPRETATION = "CANDIDATE_ENERGY_EQUIVALENT_NOT_WILLIAMS_ABSOLUTE_K"
SOURCE_COMMITS = {
    "live_topology_kernel_v11": "0238aae096aa29e79829d3c562383c38f2290ad6",
    "v12_directional_ownership_and_trial_pattern": "b3d0add6cbb0605adaa3e04006fe987961ad6452",
    "v13_isolated_trial_ownership_pattern_read_only": "ca4abfe47765fcdaf0d266bfc0558ecd82d0c64e",
}


def _pickle_hash(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _kinetic_process_payload(state: LiveFEMTopologyState) -> tuple[Any, ...]:
    mechanics_only = {
        "boundary_terminal_context", "crack_representation",
        "latest_causal_support_trials",
    }
    junction = {
        key: value for key, value in state.junction_process_state.items()
        if key not in mechanics_only and not key.startswith("v12_")
    }
    counters = {
        key: value for key, value in state.event_counters.items()
        if key not in {"mesh_generation", "refinement_operation_index"}
    }
    return (
        state.competition, state.tip_process_state, junction,
        state.rng_state, counters, state.void_state,
    )


def accepted_state_identity(state: LiveFEMTopologyState) -> str:
    return "accepted:" + complete_accepted_state_fingerprint(state)


def accepted_stress_field_identity(
    state: LiveFEMTopologyState, accepted_state_id: str,
) -> str:
    """Bind the scalar observations to the accepted tensor field and mesh."""
    _, _, sigma_gp, *_ = assemble_mechanics(
        state.mesh, state.displacement, state.ep_gp, state.rho_gp, state.damage,
        state.elasticity_D, state.material,
        cohesive_network=state.cohesive_network,
    )
    digest = hashlib.sha256()
    digest.update(str(accepted_state_id).encode())
    for value in (state.mesh.nodes, state.mesh.elems, sigma_gp):
        array = np.ascontiguousarray(value)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return "stress:" + digest.hexdigest()


def _ordered_surface_nodes(edges: np.ndarray) -> tuple[int, ...]:
    adjacency: dict[int, list[int]] = {}
    for a, b in np.asarray(edges, dtype=int):
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    if not adjacency or any(len(peers) != 2 for peers in adjacency.values()):
        raise RuntimeError("cavity inventory is not one closed boundary cycle")
    first = min(adjacency)
    ordered = [first]
    previous: int | None = None
    current = first
    while True:
        following = min(node for node in adjacency[current] if node != previous)
        if following == first:
            break
        if following in ordered:
            raise RuntimeError("cavity inventory boundary cycle self-repeats")
        ordered.append(following)
        previous, current = current, following
    if len(ordered) != len(adjacency):
        raise RuntimeError("cavity inventory contains more than one component")
    return tuple(ordered)


def cavity_free_surface_inventory(
    state: LiveFEMTopologyState,
) -> tuple[dict[str, Any], ...]:
    """Extract actual one-owner cavity edges; an absent void gives an empty set."""
    void_state = state.void_state
    if void_state is None or not void_state.cavities:
        return ()
    nodes = np.asarray(state.mesh.nodes, dtype=float)
    elements = np.asarray(state.mesh.elems, dtype=int)
    raw_edges = np.sort(np.concatenate((
        elements[:, (0, 1)], elements[:, (1, 2)], elements[:, (2, 0)],
    )), axis=1)
    unique, counts = np.unique(raw_edges, axis=0, return_counts=True)
    one_owner = unique[counts == 1]
    inventory = []
    for cavity in sorted(void_state.cavities, key=lambda item: item.cavity_id):
        radii = np.linalg.norm(nodes - np.asarray(cavity.center_m, dtype=float), axis=1)
        edges = one_owner[np.all(radii[one_owner] <= float(cavity.radius_m) * 1.02, axis=1)]
        if not len(edges):
            raise RuntimeError(f"cavity {cavity.cavity_id} has no actual free-surface edges")
        ordered = _ordered_surface_nodes(edges)
        inventory.append({
            "surface_id": str(cavity.cavity_id),
            "polygon_xy_m": tuple(tuple(map(float, nodes[node])) for node in ordered),
            "boundary_segments_xy_m": tuple(
                (tuple(map(float, nodes[a])), tuple(map(float, nodes[b])))
                for a, b in edges
            ),
        })
    return tuple(inventory)


@dataclass(frozen=True, order=True)
class ExactMarginalTrialKey:
    accepted_state_id: str
    stress_field_state_id: str
    topology_fingerprint: str
    candidate_id: str
    selected_front_id: str
    process_owner_id: str
    delta_a_hex: str
    local_mesh_level: int

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class ExactMarginalTrialCache:
    """Accepted-generation cache; trials remain observations, never events."""

    def __init__(self) -> None:
        self._entries: dict[ExactMarginalTrialKey, dict[str, Any]] = {}
        self.creation_count = 0
        self.destruction_count = 0

    def create(self, key: ExactMarginalTrialKey, outcome: Mapping[str, Any]) -> None:
        if key in self._entries:
            raise RuntimeError("duplicate exact marginal trial-cache identity")
        self._entries[key] = dict(outcome)
        self.creation_count += 1

    def observe_and_discard(self, key: ExactMarginalTrialKey) -> dict[str, Any]:
        try:
            result = self._entries.pop(key)
        except KeyError as exc:
            raise RuntimeError("exact marginal trial is detached from its accepted state") from exc
        self.destruction_count += 1
        return result

    def require_empty(self) -> None:
        if self._entries:
            raise RuntimeError("ephemeral exact marginal trials survived observation")


def _topology_request(
    state: LiveFEMTopologyState,
    candidates_by_tip: Mapping[str, tuple[CleavageCandidate, ...]],
    *,
    contour_radius_m: float,
    provider_contract_contour_radius_m: float,
) -> LiveTopologyRequest:
    nodes = np.asarray(state.mesh.nodes, dtype=float)
    material = state.material
    return LiveTopologyRequest(
        mesh=state.mesh,
        boundary=state.boundary,
        displacement=state.displacement,
        ep_gp=state.ep_gp,
        rho_gp=state.rho_gp,
        damage=state.damage,
        elasticity_D=state.elasticity_D,
        material=material,
        cohesive_network=state.cohesive_network,
        crack_network=state.crack_network,
        candidates_by_tip=candidates_by_tip,
        mechanical_configuration_fingerprint=MODEL_ID,
        specimen_geometry={
            "x_min_m": float(np.min(nodes[:, 0])),
            "x_max_m": float(np.max(nodes[:, 0])),
            "y_min_m": float(np.min(nodes[:, 1])),
            "y_max_m": float(np.max(nodes[:, 1])),
        },
        boundary_condition_identity="accepted_fixed_opening",
        elastic_constants={
            "E_Pa": float(material.E),
            "nu": float(material.nu),
            "Eprime_Pa": float(material.Eprime),
        },
        cluster_frame={"mode": "explicit_front_candidate_ownership"},
        mpz_station_coordinates_m=(),
        wake_station_coordinates_m=(),
        contour_radius_m=float(contour_radius_m),
        exclude_radius_m=max(
            float(getattr(state.mesh, "hbar_tip", 0.0) or state.mesh.hbar),
            1.0e-12,
        ),
        provider_contract_contour_radius_m=float(provider_contract_contour_radius_m),
        cavity_free_surface_inventory=cavity_free_surface_inventory(state),
    )


def _tip_rows_by_owner(
    state: LiveFEMTopologyState, provider_result: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for provider_tip in provider_result["tips"]:
        xy = tuple(map(float, provider_tip["tip_xy_m"]))
        owners = [
            tip_id for tip_id in state.crack_network.active_tip_ids
            if math.dist(tuple(map(float, state.crack_network.branch(tip_id).tip)), xy)
            <= 1.0e-12
        ]
        if len(owners) != 1 or owners[0] in rows:
            raise RuntimeError("directional provider tip has nonunique physical ownership")
        rows[owners[0]] = provider_tip
    if set(rows) != set(state.crack_network.active_tip_ids):
        raise RuntimeError("directional provider omitted an active physical tip")
    return rows


def _exact_marginal_trial(
    state: LiveFEMTopologyState,
    *,
    branch_id: str,
    candidate: CleavageCandidate,
    delta_a_m: float,
    local_mesh_level: int,
    source_commit: str,
    prepare_support_state: Callable[[LiveFEMTopologyState], LiveFEMTopologyState] | None,
    conform_trial_endpoint: Callable[
        [LiveFEMTopologyState, tuple[float, float]], LiveFEMTopologyState
    ] | None,
) -> dict[str, Any]:
    accepted_before = complete_accepted_state_fingerprint(state)
    process_before = _pickle_hash(_kinetic_process_payload(state))
    base = state.isolated_copy()
    start = tuple(map(float, base.crack_network.branch(branch_id).tip))
    direction = np.asarray(candidate.direction_xy, dtype=float)
    direction /= np.linalg.norm(direction)
    end = tuple(map(float, np.asarray(start) + float(delta_a_m) * direction))
    if conform_trial_endpoint is not None:
        base = conform_trial_endpoint(base, end)
    base = equilibrate_fixed_load_with_production_fem(base)
    arm = TopologyArm(
        candidate.candidate_id,
        branch_id,
        start,
        end,
        float(delta_a_m),
        0.0,
        candidate_direction_xy=tuple(map(float, direction)),
        first_intersection_xy_m=end,
    )
    trial = apply_v12_production_trial_geometry(
        base,
        (arm,),
        source_commit=str(source_commit),
        configuration={
            "model": MODEL_ID,
            "trial_kind": "fixed_void_virtual_candidate_extension",
            "candidate_id": candidate.candidate_id,
            "delta_a_m": float(delta_a_m),
            "local_mesh_level": int(local_mesh_level),
        },
        transaction_identity=(
            f"v5-hybrid-marginal-{branch_id}-{candidate.candidate_id}-"
            f"{float(delta_a_m).hex()}-L{int(local_mesh_level)}"
        ),
        refinement_levels=int(local_mesh_level),
        prepare_support_state=prepare_support_state,
    )
    trial = equilibrate_fixed_load_with_production_fem(trial)
    accepted_after = complete_accepted_state_fingerprint(state)
    process_after = _pickle_hash(_kinetic_process_payload(state))
    if accepted_before != accepted_after or process_before != process_after:
        raise RuntimeError("virtual marginal trial mutated the accepted state")
    if _pickle_hash(base.void_state) != _pickle_hash(trial.void_state):
        raise RuntimeError("virtual marginal trial changed fixed void geometry or inventory")
    if _pickle_hash(_kinetic_process_payload(base)) != _pickle_hash(
        _kinetic_process_payload(trial)
    ):
        raise RuntimeError("virtual marginal trial changed clocks, thresholds, RNG, or process state")
    realized = trial.crack_network.branch(branch_id)
    if math.dist(tuple(map(float, realized.tip)), end) > 1.0e-12:
        raise RuntimeError("virtual marginal trial did not extend only the owning front")
    return {
        "status": "CERTIFIED_EXACT_FIXED_VOID_MARGINAL",
        "G_marginal_J_per_m2": (
            float(base.stored_energy_J_per_m) - float(trial.stored_energy_J_per_m)
        ) / float(delta_a_m),
        "Pi_base_J_per_m": float(base.stored_energy_J_per_m),
        "Pi_trial_J_per_m": float(trial.stored_energy_J_per_m),
        "delta_a_m": float(delta_a_m),
        "local_mesh_level": int(local_mesh_level),
        "owning_front_id": branch_id,
        "candidate_id": candidate.candidate_id,
        "accepted_state_preserved": True,
        "fixed_void_geometry_preserved": True,
        "kinetic_process_state_preserved": True,
        "trial_topology_generation": int(trial.crack_network.geometry_generation),
        "trial_support_transaction_identity": trial.v12_support_state.transaction_identity,
    }


def hybrid_directional_drive_provider(
    state: LiveFEMTopologyState,
    candidates_by_tip: Mapping[str, Sequence[CleavageCandidate]],
    *,
    contour_radius_m: float,
    provider_contract_contour_radius_m: float | None = None,
    marginal_delta_a_m: Sequence[float] | None = None,
    marginal_mesh_levels: Sequence[int] = (2, 3),
    evaluate_overlap_marginals: bool = False,
    source_commit: str = "WORKTREE",
    prepare_support_state: Callable[[LiveFEMTopologyState], LiveFEMTopologyState] | None = None,
    conform_trial_endpoint: Callable[
        [LiveFEMTopologyState, tuple[float, float]], LiveFEMTopologyState
    ] | None = None,
) -> dict[str, Any]:
    """Return candidate-specific local J or an unblended exact marginal G."""
    active = tuple(state.crack_network.active_tip_ids)
    normalized = {
        str(tip): tuple(candidates_by_tip[tip]) for tip in candidates_by_tip
    }
    if set(normalized) != set(active):
        raise RuntimeError("hybrid provider candidates are not owned by every active tip")
    expected_pairs = {
        (tip, candidate.candidate_id)
        for tip, candidates in normalized.items() for candidate in candidates
    }
    if len(expected_pairs) != sum(map(len, normalized.values())):
        raise RuntimeError("duplicate front/candidate ownership pair")
    radius = float(contour_radius_m)
    contract_radius = float(provider_contract_contour_radius_m or radius)
    request = _topology_request(
        state, normalized, contour_radius_m=radius,
        provider_contract_contour_radius_m=contract_radius,
    )
    provider = evaluate_exact_topology(request)
    accepted_id = accepted_state_identity(state)
    stress_id = accepted_stress_field_identity(state, accepted_id)
    topology_id = str(provider["topology_fingerprint"])
    tips = _tip_rows_by_owner(state, provider)
    default_delta = max(
        radius,
        2.0 * float(getattr(state.mesh, "hbar_tip", 0.0) or state.mesh.hbar),
    )
    deltas = tuple(float(value) for value in (
        marginal_delta_a_m or (default_delta, (2.0 / 3.0) * default_delta)
    ))
    levels = tuple(int(value) for value in marginal_mesh_levels)
    if not deltas or any(value <= 0.0 or not math.isfinite(value) for value in deltas):
        raise ValueError("marginal delta-a values must be finite and positive")
    if not levels or any(value < 1 for value in levels):
        raise ValueError("marginal local mesh levels must be positive")
    cache = ExactMarginalTrialCache()
    output_rows = []
    seen_pairs: set[tuple[str, str]] = set()
    for branch_id in active:
        directional = {row["candidate_id"]: row for row in tips[branch_id]["directional"]}
        for candidate in normalized[branch_id]:
            pair = branch_id, candidate.candidate_id
            if pair in seen_pairs or candidate.candidate_id not in directional:
                raise RuntimeError("hybrid provider candidate/tip observation mismatch")
            seen_pairs.add(pair)
            local = directional[candidate.candidate_id]
            local_valid = bool(local["local_contour_valid"])
            marginal_rows = []
            unavailable_reason = None
            if not local_valid or evaluate_overlap_marginals:
                for delta in deltas:
                    for level in levels:
                        key = ExactMarginalTrialKey(
                            accepted_id, stress_id, topology_id,
                            candidate.candidate_id, branch_id, branch_id,
                            float(delta).hex(), int(level),
                        )
                        try:
                            outcome = _exact_marginal_trial(
                                state, branch_id=branch_id, candidate=candidate,
                                delta_a_m=delta, local_mesh_level=level,
                                source_commit=source_commit,
                                prepare_support_state=prepare_support_state,
                                conform_trial_endpoint=conform_trial_endpoint,
                            )
                        except (RuntimeError, ValueError) as exc:
                            outcome = {
                                "status": "DIRECTIONAL_DRIVE_UNAVAILABLE",
                                "reason": f"{type(exc).__name__}:{exc}",
                                "delta_a_m": delta,
                                "local_mesh_level": level,
                            }
                            unavailable_reason = outcome["reason"]
                        cache.create(key, {**outcome, "trial_cache_identity": key.to_dict()})
                        marginal_rows.append(cache.observe_and_discard(key))
            certified = [
                row for row in marginal_rows
                if row["status"] == "CERTIFIED_EXACT_FIXED_VOID_MARGINAL"
            ]
            authoritative_marginal = None
            if certified:
                authoritative_marginal = min(
                    certified,
                    key=lambda row: (float(row["delta_a_m"]), -int(row["local_mesh_level"])),
                )
            available = local_valid or (
                authoritative_marginal is not None and len(certified) == len(marginal_rows)
            )
            if not available:
                G_used = 0.0
                drive_source = "DIRECTIONAL_DRIVE_UNAVAILABLE"
            elif local_valid:
                G_used = max(float(local["J_local_signed_J_per_m2"]), 0.0)
                drive_source = "VALID_NESTED_LOCAL_J"
            else:
                G_used = max(float(authoritative_marginal["G_marginal_J_per_m2"]), 0.0)
                drive_source = "EXACT_FIXED_VOID_VIRTUAL_EXTENSION_MARGINAL_G"
            K_energy = math.sqrt(float(state.material.Eprime) * G_used)
            output_rows.append({
                **local,
                "tip_id": branch_id,
                "candidate_id": candidate.candidate_id,
                "process_owner_id": branch_id,
                "controlling_scalar_tip_id": branch_id,
                "tensor_probe_tip_id": branch_id,
                "accepted_state_id": accepted_id,
                "stress_field_state_id": stress_id,
                "topology_fingerprint": topology_id,
                "G_local_J_per_m2": float(local["J_local_signed_J_per_m2"]),
                "G_local": float(local["J_local_signed_J_per_m2"]),
                "local_contour_valid": local_valid,
                "G_marginal_J_per_m2": (
                    None if authoritative_marginal is None
                    else float(authoritative_marginal["G_marginal_J_per_m2"])
                ),
                "G_marginal": (
                    None if authoritative_marginal is None
                    else float(authoritative_marginal["G_marginal_J_per_m2"])
                ),
                "G_kinetic_used_J_per_m2": G_used,
                "G_kinetic_used": G_used,
                "drive_source": drive_source,
                "K_energy_equivalent_Pa_sqrt_m": K_energy,
                "K_energy_equivalent": K_energy,
                "K_interpretation": K_INTERPRETATION,
                "K_directional_Pa_sqrt_m": K_energy,
                "positive_J_J_per_m2": G_used,
                "marginal_trial_rows": marginal_rows,
                "directional_drive_status": (
                    "AVAILABLE" if available else "DIRECTIONAL_DRIVE_UNAVAILABLE"
                ),
                "directional_drive_unavailable_reason": unavailable_reason,
            })
    if seen_pairs != expected_pairs:
        raise RuntimeError("hybrid provider omitted a front/candidate pair")
    cache.require_empty()
    return {
        "schema": MODEL_ID,
        "source_commits": dict(SOURCE_COMMITS),
        "accepted_state_id": accepted_id,
        "stress_field_state_id": stress_id,
        "topology_fingerprint": topology_id,
        "recoverable_potential_energy_J_per_m": float(
            provider["base_equilibrium"]["recoverable_potential_energy_J_per_m"]
        ),
        "directional": output_rows,
        "local_provider": provider,
        "trial_cache_creation_count": cache.creation_count,
        "trial_cache_destruction_count": cache.destruction_count,
    }


__all__ = [
    "ExactMarginalTrialCache",
    "ExactMarginalTrialKey",
    "K_INTERPRETATION",
    "MODEL_ID",
    "SOURCE_COMMITS",
    "accepted_state_identity",
    "accepted_stress_field_identity",
    "cavity_free_surface_inventory",
    "hybrid_directional_drive_provider",
]
