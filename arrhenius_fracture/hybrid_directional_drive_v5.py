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
MARGINAL_G_MESH_RELATIVE_LIMIT = 0.10
MARGINAL_G_DELTA_A_RELATIVE_LIMIT = 0.10
MARGINAL_DELTA_A_VALUES_M = (3.0e-5, 2.0e-5)
MARGINAL_LOCAL_MESH_LEVELS = (2, 3, 4)
# These are the already-established topology-transaction energy tolerances.
# The directional-G floor is derived from them and the requested delta-a; it is
# therefore dimensionally an energy-release-rate accuracy, not an arbitrary
# absolute scale selected from the observed marginal values.
MARGINAL_ENERGY_RELATIVE_ACCURACY = 1.0e-8
MARGINAL_ENERGY_ABSOLUTE_ACCURACY_J_PER_M = 1.0e-12
SOURCE_COMMITS = {
    "live_topology_kernel_v11": "0238aae096aa29e79829d3c562383c38f2290ad6",
    "v12_directional_ownership_and_trial_pattern": "b3d0add6cbb0605adaa3e04006fe987961ad6452",
    "v13_isolated_trial_ownership_pattern_read_only": "ca4abfe47765fcdaf0d266bfc0558ecd82d0c64e",
}


class MarginalDriveNotCertified(RuntimeError):
    """Expected scientific or numerical unavailability of one marginal trial."""


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
    void_fingerprint: str
    source_commit: str
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


def _canonical_process_owners(
    state: LiveFEMTopologyState,
    active_front_ids: Sequence[str],
    process_owner_by_tip: Mapping[str, str] | None,
) -> dict[str, str]:
    """Validate the accepted runtime's explicit front-to-process ownership."""
    if process_owner_by_tip is None:
        raise RuntimeError("hybrid provider requires an explicit process-owner mapping")
    owners = {str(front): str(owner) for front, owner in process_owner_by_tip.items()}
    if set(owners) != set(active_front_ids):
        raise RuntimeError("process-owner mapping does not cover every active front exactly")
    registry = state.tip_process_state.get("by_branch")
    if not isinstance(registry, Mapping):
        raise RuntimeError("accepted production state has no process-owner registry")
    checkpoint_mapping = state.tip_process_state.get("owner_by_front")
    if checkpoint_mapping is not None and not isinstance(checkpoint_mapping, Mapping):
        raise RuntimeError("checkpoint process-owner mapping is malformed")
    for front_id, owner_id in owners.items():
        if not owner_id or owner_id not in registry:
            raise RuntimeError(f"active front {front_id} has an absent process owner")
        branch_owner = state.crack_network.branch(front_id).local_state.get(
            "front_engine_state_owner"
        )
        if branch_owner is None or str(branch_owner) != owner_id:
            raise RuntimeError(f"active front {front_id} is outside its process-owner region")
        if checkpoint_mapping is not None and str(checkpoint_mapping.get(front_id)) != owner_id:
            raise RuntimeError(f"active front {front_id} disagrees with checkpoint ownership")
    return owners


def _relative_marginal_error(first: float, second: float, floor: float) -> float:
    return abs(float(first) - float(second)) / max(
        abs(float(first)), abs(float(second)), float(floor)
    )


def marginal_convergence_diagnostics(
    marginal_rows: Sequence[Mapping[str, Any]],
    *,
    delta_a_values_m: Sequence[float],
    local_mesh_levels: Sequence[int],
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen complete-family marginal-G certification contract."""
    deltas = tuple(sorted({float(value) for value in delta_a_values_m}))
    levels = tuple(sorted({int(value) for value in local_mesh_levels}))
    if len(deltas) < 2 or len(levels) < 2:
        raise ValueError("marginal certification requires two delta-a values and two mesh levels")
    expected_keys = {(float(delta).hex(), level) for delta in deltas for level in levels}
    observed_keys = [
        (float(row["delta_a_m"]).hex(), int(row["local_mesh_level"]))
        for row in marginal_rows
    ]
    identity_fields = (
        "candidate_id", "owning_front_id", "process_owner_id",
        "accepted_state_id", "stress_field_state_id", "topology_fingerprint",
        "void_fingerprint", "source_commit",
    )
    identity_consistent = all(
        all(row[field] == expected_identity[field] for field in identity_fields)
        for row in marginal_rows
    )
    all_certified = bool(marginal_rows) and all(
        row["status"] == "CERTIFIED_EXACT_FIXED_VOID_MARGINAL"
        for row in marginal_rows
    )
    complete = (
        len(observed_keys) == len(expected_keys)
        and len(set(observed_keys)) == len(observed_keys)
        and set(observed_keys) == expected_keys
    )
    diagnostics: dict[str, Any] = {
        "marginal_trial_count_expected": len(expected_keys),
        "marginal_trial_count_observed": len(marginal_rows),
        "marginal_all_trials_certified": all_certified,
        "marginal_family_identity_consistent": identity_consistent,
        "marginal_mesh_relative_errors_by_delta": {},
        "marginal_delta_a_relative_error_at_finest_mesh": None,
        "marginal_signed_G_consistent": False,
        "marginal_zero_drive_classification": "NOT_EVALUATED",
        "marginal_convergence_passed": False,
        "authoritative_delta_a_m": None,
        "authoritative_mesh_level": None,
        "authoritative_G_marginal_J_per_m2": None,
        "marginal_numerical_floor_J_per_m2": None,
        "marginal_unavailable_reason": None,
    }
    if not complete:
        diagnostics["marginal_unavailable_reason"] = "INCOMPLETE_MARGINAL_TRIAL_FAMILY"
        return diagnostics
    if not identity_consistent:
        raise RuntimeError("marginal trial family mixes accepted ownership identities")
    if not all_certified:
        diagnostics["marginal_unavailable_reason"] = "MARGINAL_TRIAL_NOT_CERTIFIED"
        return diagnostics

    by_key = {
        (float(row["delta_a_m"]).hex(), int(row["local_mesh_level"])): row
        for row in marginal_rows
    }
    energy_scale = max(
        abs(float(row["Pi_base_J_per_m"])) for row in marginal_rows
    )
    energy_scale = max(
        energy_scale,
        max(abs(float(row["Pi_trial_J_per_m"])) for row in marginal_rows),
    )
    energy_accuracy = max(
        MARGINAL_ENERGY_ABSOLUTE_ACCURACY_J_PER_M,
        MARGINAL_ENERGY_RELATIVE_ACCURACY * energy_scale,
    )
    numerical_floor = energy_accuracy / min(deltas)
    diagnostics["marginal_numerical_floor_J_per_m2"] = numerical_floor
    coarse_level, fine_level = levels[-2], levels[-1]
    mesh_errors = {}
    for delta in deltas:
        coarse = float(by_key[(delta.hex(), coarse_level)]["G_marginal_J_per_m2"])
        fine = float(by_key[(delta.hex(), fine_level)]["G_marginal_J_per_m2"])
        if not math.isfinite(coarse) or not math.isfinite(fine):
            raise ValueError("certified marginal G must be finite")
        mesh_errors[delta.hex()] = _relative_marginal_error(coarse, fine, numerical_floor)
    diagnostics["marginal_mesh_relative_errors_by_delta"] = mesh_errors
    small_delta, next_delta = deltas[0], deltas[1]
    small_value = float(by_key[(small_delta.hex(), fine_level)]["G_marginal_J_per_m2"])
    next_value = float(by_key[(next_delta.hex(), fine_level)]["G_marginal_J_per_m2"])
    delta_error = _relative_marginal_error(small_value, next_value, numerical_floor)
    diagnostics["marginal_delta_a_relative_error_at_finest_mesh"] = delta_error
    signed_values = [float(row["G_marginal_J_per_m2"]) for row in marginal_rows]
    if not all(math.isfinite(value) for value in signed_values):
        raise ValueError("certified marginal G must be finite")
    all_numerical_zero = all(abs(value) <= numerical_floor for value in signed_values)
    nonzero_signs = {math.copysign(1.0, value) for value in signed_values if value != 0.0}
    signed_consistent = len(nonzero_signs) <= 1 or all_numerical_zero
    diagnostics["marginal_signed_G_consistent"] = signed_consistent
    if all_numerical_zero:
        diagnostics["marginal_zero_drive_classification"] = (
            "ALL_SIGNED_G_WITHIN_NUMERICAL_ZERO"
        )
    elif small_value <= 0.0:
        diagnostics["marginal_zero_drive_classification"] = "CONVERGED_NONPOSITIVE_G"
    else:
        diagnostics["marginal_zero_drive_classification"] = "POSITIVE_G"

    mesh_passed = all(
        error <= MARGINAL_G_MESH_RELATIVE_LIMIT for error in mesh_errors.values()
    )
    delta_passed = delta_error <= MARGINAL_G_DELTA_A_RELATIVE_LIMIT
    if not mesh_passed:
        reason = "MARGINAL_G_MESH_NOT_CONVERGED"
    elif not signed_consistent:
        reason = "MARGINAL_G_SIGN_INCONSISTENT"
    elif not delta_passed:
        reason = "MARGINAL_G_DELTA_A_NOT_CONVERGED"
    else:
        reason = None
    diagnostics["marginal_unavailable_reason"] = reason
    diagnostics["marginal_convergence_passed"] = reason is None
    if reason is None:
        diagnostics.update({
            "authoritative_delta_a_m": small_delta,
            "authoritative_mesh_level": fine_level,
            "authoritative_G_marginal_J_per_m2": (
                0.0 if all_numerical_zero else small_value
            ),
        })
    return diagnostics


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
    process_owner_id: str,
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
        try:
            base = conform_trial_endpoint(base, end)
        except ValueError as exc:
            if str(exc) == "fixed crack tip is outside the specimen mesh":
                raise MarginalDriveNotCertified(
                    "TRIAL_ENDPOINT_OUTSIDE_PERMITTED_DOMAIN"
                ) from exc
            raise
        except RuntimeError as exc:
            if str(exc) == (
                "v12_support_not_certified: REQUIRES_ACTIVE_TIP_ALIGNMENT_REMESH"
            ):
                raise MarginalDriveNotCertified(
                    "EXACT_V12_SUPPORT_REQUIRES_ACTIVE_TIP_ALIGNMENT"
                ) from exc
            raise
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
    try:
        trial = apply_v12_production_trial_geometry(
            base,
            (arm,),
            source_commit=str(source_commit),
            configuration={
                "model": MODEL_ID,
                "trial_kind": "fixed_void_virtual_candidate_extension",
                "candidate_id": candidate.candidate_id,
                "owning_front_id": branch_id,
                "process_owner_id": process_owner_id,
                "delta_a_m": float(delta_a_m),
                "local_mesh_level": int(local_mesh_level),
            },
            transaction_identity=(
                f"v5-hybrid-marginal-{branch_id}-{process_owner_id}-"
                f"{candidate.candidate_id}-{float(delta_a_m).hex()}-"
                f"L{int(local_mesh_level)}"
            ),
            refinement_levels=int(local_mesh_level),
            prepare_support_state=prepare_support_state,
        )
    except RuntimeError as exc:
        classifications = {
            "production V12 remesh found no physical event support":
                "NO_MECHANICALLY_RESOLVED_TRIAL_NOVELTY",
            "initial V12 support is not certified":
                "EXACT_V12_SUPPORT_NOT_CERTIFIED",
        }
        if str(exc) in classifications:
            raise MarginalDriveNotCertified(classifications[str(exc)]) from exc
        raise
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
        "process_owner_id": process_owner_id,
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
    process_owner_by_tip: Mapping[str, str] | None = None,
    contour_radius_m: float,
    provider_contract_contour_radius_m: float | None = None,
    marginal_delta_a_m: Sequence[float] | None = MARGINAL_DELTA_A_VALUES_M,
    marginal_mesh_levels: Sequence[int] = MARGINAL_LOCAL_MESH_LEVELS,
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
    process_owners = _canonical_process_owners(state, active, process_owner_by_tip)
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
    accepted_before_trials = complete_accepted_state_fingerprint(state)
    process_before_trials = _pickle_hash(_kinetic_process_payload(state))
    void_id = "void:" + _pickle_hash(state.void_state)
    for branch_id in active:
        process_owner_id = process_owners[branch_id]
        directional = {row["candidate_id"]: row for row in tips[branch_id]["directional"]}
        for candidate in normalized[branch_id]:
            pair = branch_id, candidate.candidate_id
            if pair in seen_pairs or candidate.candidate_id not in directional:
                raise RuntimeError("hybrid provider candidate/tip observation mismatch")
            seen_pairs.add(pair)
            local = directional[candidate.candidate_id]
            local_valid = bool(local["local_contour_valid"])
            marginal_rows = []
            expected_identity = {
                "candidate_id": candidate.candidate_id,
                "owning_front_id": branch_id,
                "process_owner_id": process_owner_id,
                "accepted_state_id": accepted_id,
                "stress_field_state_id": stress_id,
                "topology_fingerprint": topology_id,
                "void_fingerprint": void_id,
                "source_commit": str(source_commit),
            }
            marginal_diagnostics = {
                "marginal_trial_count_expected": 0,
                "marginal_trial_count_observed": 0,
                "marginal_all_trials_certified": False,
                "marginal_family_identity_consistent": True,
                "marginal_mesh_relative_errors_by_delta": {},
                "marginal_delta_a_relative_error_at_finest_mesh": None,
                "marginal_signed_G_consistent": False,
                "marginal_zero_drive_classification": "NOT_EVALUATED_LOCAL_J_AUTHORITATIVE",
                "marginal_convergence_passed": False,
                "authoritative_delta_a_m": None,
                "authoritative_mesh_level": None,
                "authoritative_G_marginal_J_per_m2": None,
                "marginal_numerical_floor_J_per_m2": None,
                "marginal_unavailable_reason": None,
            }
            if not local_valid or evaluate_overlap_marginals:
                if len(set(deltas)) < 2 or len(set(levels)) < 2:
                    marginal_diagnostics.update({
                        "marginal_trial_count_expected": len(set(deltas)) * len(set(levels)),
                        "marginal_unavailable_reason": "INCOMPLETE_MARGINAL_TRIAL_FAMILY",
                    })
                else:
                    for delta in deltas:
                        for level in levels:
                            key = ExactMarginalTrialKey(
                                accepted_id, stress_id, topology_id,
                                candidate.candidate_id, branch_id, process_owner_id,
                                void_id, str(source_commit), float(delta).hex(), int(level),
                            )
                            try:
                                outcome = _exact_marginal_trial(
                                    state, branch_id=branch_id,
                                    process_owner_id=process_owner_id,
                                    candidate=candidate, delta_a_m=delta,
                                    local_mesh_level=level, source_commit=source_commit,
                                    prepare_support_state=prepare_support_state,
                                    conform_trial_endpoint=conform_trial_endpoint,
                                )
                            except MarginalDriveNotCertified as exc:
                                outcome = {
                                    "status": "DIRECTIONAL_DRIVE_UNAVAILABLE",
                                    "reason": str(exc),
                                    "delta_a_m": delta,
                                    "local_mesh_level": level,
                                }
                            if (
                                complete_accepted_state_fingerprint(state) != accepted_before_trials
                                or _pickle_hash(_kinetic_process_payload(state)) != process_before_trials
                                or "void:" + _pickle_hash(state.void_state) != void_id
                            ):
                                raise RuntimeError(
                                    "marginal observation mutated accepted or process state"
                                )
                            recorded = {
                                **outcome,
                                **expected_identity,
                                "selected_front_id": branch_id,
                                "trial_cache_identity": key.to_dict(),
                            }
                            cache.create(key, recorded)
                            marginal_rows.append(cache.observe_and_discard(key))
                    marginal_diagnostics = marginal_convergence_diagnostics(
                        marginal_rows,
                        delta_a_values_m=deltas,
                        local_mesh_levels=levels,
                        expected_identity=expected_identity,
                    )
            authoritative_marginal = (
                marginal_diagnostics["authoritative_G_marginal_J_per_m2"]
                if marginal_diagnostics["marginal_convergence_passed"] else None
            )
            available = local_valid or authoritative_marginal is not None
            if not available:
                G_used = 0.0
                drive_source = "MARGINAL_G_NOT_CONVERGED"
            elif local_valid:
                G_used = max(float(local["J_local_signed_J_per_m2"]), 0.0)
                drive_source = "VALID_NESTED_LOCAL_J"
            else:
                G_used = max(float(authoritative_marginal), 0.0)
                drive_source = "EXACT_FIXED_VOID_VIRTUAL_EXTENSION_MARGINAL_G"
            K_energy = math.sqrt(float(state.material.Eprime) * G_used)
            output_rows.append({
                **local,
                "tip_id": branch_id,
                "candidate_id": candidate.candidate_id,
                "process_owner_id": process_owner_id,
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
                    else float(authoritative_marginal)
                ),
                "G_marginal": (
                    None if authoritative_marginal is None
                    else float(authoritative_marginal)
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
                **marginal_diagnostics,
                "directional_drive_status": (
                    "AVAILABLE" if available else "DIRECTIONAL_DRIVE_UNAVAILABLE"
                ),
                "directional_drive_unavailable_reason": (
                    None if available else marginal_diagnostics["marginal_unavailable_reason"]
                ),
                "effective_rate": 0.0 if not available else None,
            })
    if seen_pairs != expected_pairs:
        raise RuntimeError("hybrid provider omitted a front/candidate pair")
    cache.require_empty()
    if (
        complete_accepted_state_fingerprint(state) != accepted_before_trials
        or _pickle_hash(_kinetic_process_payload(state)) != process_before_trials
        or "void:" + _pickle_hash(state.void_state) != void_id
    ):
        raise RuntimeError("hybrid provider changed accepted or process state")
    return {
        "schema": MODEL_ID,
        "source_commits": dict(SOURCE_COMMITS),
        "accepted_state_id": accepted_id,
        "stress_field_state_id": stress_id,
        "topology_fingerprint": topology_id,
        "process_owner_by_tip": process_owners,
        "void_fingerprint": void_id,
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
    "MARGINAL_ENERGY_ABSOLUTE_ACCURACY_J_PER_M",
    "MARGINAL_ENERGY_RELATIVE_ACCURACY",
    "MARGINAL_DELTA_A_VALUES_M",
    "MARGINAL_G_DELTA_A_RELATIVE_LIMIT",
    "MARGINAL_G_MESH_RELATIVE_LIMIT",
    "MARGINAL_LOCAL_MESH_LEVELS",
    "MarginalDriveNotCertified",
    "MODEL_ID",
    "SOURCE_COMMITS",
    "accepted_state_identity",
    "accepted_stress_field_identity",
    "cavity_free_surface_inventory",
    "hybrid_directional_drive_provider",
    "marginal_convergence_diagnostics",
]
