"""Atomic live-FEM topology trials for mechanistic v11 crack branching.

The transaction owns geometry and directional-event consumption.  Trial geometry
is built on an isolated snapshot, re-equilibrated at the accepted displacement
load, and committed only when its actual whole-body potential-energy release pays
the summed hazard-derived dissipation.  It introduces no fracture criterion.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, is_dataclass, replace
import hashlib
import json
import math
import time
from typing import Any, Callable, Iterable, Iterator, Mapping

import numpy as np

from .crack_network_v11 import CrackBranchState, CrackNetworkState
from .crack_backend import SharpWakeBackend
from .causal_sharp_wake_v11 import apply_causal_segment
from .coalescence import segment_intersection_first
from .directional_competition_v11 import (
    CompetingActionProposal,
    DirectionalCompetitionState,
    accept_reservation,
    release_reservation,
    reserve_action,
)


MODEL_ID = "v11.monotonic_tip_only_live_fem_topology_transaction/1"
EQUILIBRIUM_OBSERVABLE_SCHEMA = "unified-2d.accepted-fem-equilibrium-observables/2"
LEGACY_EQUILIBRIUM_OBSERVABLE_SCHEMA = "v6.accepted-fem-equilibrium-observables/1"
REACTION_ABSOLUTE_FLOOR_N_PER_M = 1.0e-12
ENERGY_ABSOLUTE_FLOOR_J_PER_M = 1.0e-18
EQUILIBRIUM_OBSERVABLE_KEYS = (
    "latest_top_reaction_N_per_m",
    "latest_bottom_reaction_N_per_m",
    "latest_reaction_N_per_m",
    "latest_applied_opening_m",
    "latest_compliance_m2_per_N",
    "latest_external_work_J_per_m",
    "latest_stored_recoverable_energy_J_per_m",
    "latest_plastic_eigenstrain_half_work_J_per_m",
    "latest_energy_identity_reference_J_per_m",
    "latest_residual_l2_N_per_m",
    "latest_free_dof_residual_l2_N_per_m",
    "latest_constrained_reaction_l2_N_per_m",
    "latest_top_bottom_reaction_balance",
    "latest_energy_reaction_identity",
)


class EquilibriumObservablesUnavailable(RuntimeError):
    """The accepted FEM state cannot provide certified physical observables."""


def _canonical_observation_fingerprint(value: Any) -> str:
    """Hash numerical ownership without relying on object ids or repr addresses."""
    def normalize(item):
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            return {
                "dtype": array.dtype.str, "shape": array.shape,
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        if is_dataclass(item):
            return normalize({name: getattr(item, name) for name in item.__dataclass_fields__})
        if isinstance(item, Mapping):
            return {str(key): normalize(entry) for key, entry in sorted(
                item.items(), key=lambda pair: str(pair[0]),
            )}
        if isinstance(item, (tuple, list)):
            return [normalize(entry) for entry in item]
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, float) and not math.isfinite(item):
            return {"nonfinite": str(item)}
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        return {"type": type(item).__qualname__, "state": normalize(getattr(item, "__dict__", {}))}

    encoded = json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class EquilibriumObservables(Mapping[str, float]):
    """Physical measurements reassembled from one accepted constitutive state.

    The mapping interface is a backward-compatible view for retained V6
    exporters.  New production code owns this typed object directly; the
    energy ledger is no longer the source of reaction or compliance.
    """

    signed_top_reaction_vector_N_per_m: tuple[float, float]
    signed_bottom_reaction_vector_N_per_m: tuple[float, float]
    top_normal_resultant_N_per_m: float
    bottom_normal_resultant_N_per_m: float
    mean_reaction_magnitude_N_per_m: float
    top_bottom_balance_residual: float
    applied_top_displacement_m: float
    applied_bottom_displacement_m: float
    total_opening_m: float
    free_dof_residual_l2_N_per_m: float
    full_residual_including_reactions_l2_N_per_m: float
    constrained_reaction_l2_N_per_m: float
    recoverable_elastic_energy_J_per_m: float
    total_stored_internal_energy_J_per_m: float
    plastic_eigenstrain_half_work_J_per_m: float
    plastic_internal_dissipation_J_per_m: float | None
    topology_event_dissipation_J_per_m: float | None
    external_reaction_work_J_per_m: float
    compliance_m2_per_N: float
    energy_identity_reference_J_per_m: float
    energy_identity_residual: float
    free_dof_residual_relative: float
    observation_schema: str
    source_state_fingerprint: str
    mesh_fingerprint: str
    material_fingerprint: str

    def _legacy(self) -> dict[str, float | str]:
        return {
            "equilibrium_observable_schema": self.observation_schema,
            "equilibrium_reaction_absolute_floor_N_per_m": REACTION_ABSOLUTE_FLOOR_N_PER_M,
            "equilibrium_energy_absolute_floor_J_per_m": ENERGY_ABSOLUTE_FLOOR_J_PER_M,
            "latest_top_reaction_N_per_m": self.top_normal_resultant_N_per_m,
            "latest_bottom_reaction_N_per_m": self.bottom_normal_resultant_N_per_m,
            "latest_reaction_N_per_m": self.top_normal_resultant_N_per_m,
            "latest_applied_opening_m": self.total_opening_m,
            "latest_compliance_m2_per_N": self.compliance_m2_per_N,
            "latest_external_work_J_per_m": self.external_reaction_work_J_per_m,
            "latest_stored_recoverable_energy_J_per_m": self.recoverable_elastic_energy_J_per_m,
            "latest_plastic_eigenstrain_half_work_J_per_m": self.plastic_eigenstrain_half_work_J_per_m,
            "latest_energy_identity_reference_J_per_m": self.energy_identity_reference_J_per_m,
            "latest_residual_l2_N_per_m": self.full_residual_including_reactions_l2_N_per_m,
            "latest_free_dof_residual_l2_N_per_m": self.free_dof_residual_l2_N_per_m,
            "latest_free_dof_residual_relative": self.free_dof_residual_relative,
            "latest_constrained_reaction_l2_N_per_m": self.constrained_reaction_l2_N_per_m,
            "latest_top_bottom_reaction_balance": self.top_bottom_balance_residual,
            "latest_energy_reaction_identity": self.energy_identity_residual,
        }

    def __getitem__(self, key: str):
        return self._legacy()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._legacy())

    def __len__(self) -> int:
        return len(self._legacy())

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def _accepted_dirichlet_partition(state: "LiveFEMTopologyState") -> tuple[np.ndarray, np.ndarray]:
    """Return the exact constrained/free partition used by ``solve_dirichlet``."""
    prescribed = np.zeros(state.mesh.ndof, dtype=bool)
    prescribed[2 * np.asarray(state.boundary.top_nodes, dtype=int) + 1] = True
    prescribed[2 * np.asarray(state.boundary.bot_nodes, dtype=int) + 1] = True
    prescribed[2 * int(state.boundary.left_bot)] = True
    prescribed[2 * int(state.boundary.left_bot) + 1] = True
    prescribed[2 * int(state.boundary.right_bot)] = True
    return prescribed, ~prescribed


def require_equilibrium_observables(
    state: "LiveFEMTopologyState",
) -> EquilibriumObservables | Mapping[str, float]:
    """Return typed observations, with a read-only legacy-checkpoint adapter."""
    typed = getattr(state, "equilibrium_observables", None)
    if typed is not None:
        if not isinstance(typed, EquilibriumObservables):
            raise EquilibriumObservablesUnavailable("accepted equilibrium observations have the wrong type")
        values = tuple(
            value for value in typed.to_dict().values()
            if isinstance(value, (int, float)) and value is not None
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise EquilibriumObservablesUnavailable("accepted equilibrium observations are nonfinite")
        if typed.mean_reaction_magnitude_N_per_m <= REACTION_ABSOLUTE_FLOOR_N_PER_M:
            raise EquilibriumObservablesUnavailable("accepted reaction is below the declared physical floor")
        if typed.compliance_m2_per_N <= 0.0:
            raise EquilibriumObservablesUnavailable("accepted compliance is not positive")
        return typed
    ledger = state.energy_ledgers
    if ledger.get("equilibrium_observable_schema") not in {
        EQUILIBRIUM_OBSERVABLE_SCHEMA, LEGACY_EQUILIBRIUM_OBSERVABLE_SCHEMA,
    }:
        raise EquilibriumObservablesUnavailable("accepted state has no certified equilibrium-observable ledger")
    missing = [key for key in EQUILIBRIUM_OBSERVABLE_KEYS if key not in ledger]
    if missing:
        raise EquilibriumObservablesUnavailable(
            "accepted equilibrium-observable ledger is incomplete: " + ",".join(missing)
        )
    values = {key: float(ledger[key]) for key in EQUILIBRIUM_OBSERVABLE_KEYS}
    if not all(math.isfinite(value) for value in values.values()):
        raise EquilibriumObservablesUnavailable("accepted equilibrium-observable ledger is nonfinite")
    if abs(values["latest_reaction_N_per_m"]) <= REACTION_ABSOLUTE_FLOOR_N_PER_M:
        raise EquilibriumObservablesUnavailable("accepted top reaction is below the declared physical floor")
    if values["latest_compliance_m2_per_N"] <= 0.0:
        raise EquilibriumObservablesUnavailable("accepted compliance is not positive")
    return values


def _measure_accepted_equilibrium(
    state: "LiveFEMTopologyState", residual: np.ndarray, sigma_gp: np.ndarray,
    stored_energy_J_per_m: float,
) -> EquilibriumObservables:
    """Measure reactions and the plastic-strain-aware production energy identity."""
    prescribed, free = _accepted_dirichlet_partition(state)
    vector = np.asarray(residual, dtype=float)
    displacement = np.asarray(state.displacement, dtype=float)
    top_dofs = 2 * np.asarray(state.boundary.top_nodes, dtype=int) + 1
    bottom_dofs = 2 * np.asarray(state.boundary.bot_nodes, dtype=int) + 1
    top_x_dofs = 2 * np.asarray(state.boundary.top_nodes, dtype=int)
    bottom_x_dofs = 2 * np.asarray(state.boundary.bot_nodes, dtype=int)
    top = float(np.sum(vector[top_dofs]))
    bottom = float(np.sum(vector[bottom_dofs]))
    top_vector = (float(np.sum(vector[top_x_dofs])), top)
    bottom_vector = (float(np.sum(vector[bottom_x_dofs])), bottom)
    top_opening = float(np.mean(displacement[top_dofs]))
    bottom_opening = float(np.mean(displacement[bottom_dofs]))
    opening = top_opening - bottom_opening
    stored = float(stored_energy_J_per_m)
    quantities = np.asarray((top, bottom, opening, stored), dtype=float)
    if not np.all(np.isfinite(vector)) or not np.all(np.isfinite(quantities)):
        raise EquilibriumObservablesUnavailable("accepted equilibrium contains nonfinite values")
    if abs(top) <= REACTION_ABSOLUTE_FLOOR_N_PER_M:
        raise EquilibriumObservablesUnavailable("accepted top reaction is below the declared physical floor")
    if abs(bottom) <= REACTION_ABSOLUTE_FLOOR_N_PER_M:
        raise EquilibriumObservablesUnavailable("accepted bottom reaction is below the declared physical floor")
    if opening <= 0.0:
        raise EquilibriumObservablesUnavailable("accepted applied opening is not positive")
    if stored <= ENERGY_ABSOLUTE_FLOOR_J_PER_M:
        raise EquilibriumObservablesUnavailable(
            "nonzero accepted reaction has no finite stored mechanical energy"
        )
    reaction_scale = 0.5 * (abs(top) + abs(bottom))
    balance = abs(top + bottom) / reaction_scale
    free_norm = float(np.linalg.norm(vector[free]))
    full_norm = float(np.linalg.norm(vector))
    constrained_norm = float(np.linalg.norm(vector[prescribed]))
    free_relative = free_norm / reaction_scale
    boundary_half_work = 0.5 * float(np.dot(displacement[prescribed], vector[prescribed]))
    plastic_half_work = 0.5 * float(np.sum(
        np.asarray(state.ep_gp, dtype=float).T * np.asarray(sigma_gp, dtype=float).T
        * np.asarray(state.mesh.area_e, dtype=float)[:, None]
    ))
    energy_reference = boundary_half_work - plastic_half_work
    energy_identity = abs(stored - energy_reference) / stored
    from .finalization_v3_schema import SCIENTIFIC_ACCEPTANCE_TOLERANCES as limits
    if balance > limits["reaction_balance_relative"]:
        raise EquilibriumObservablesUnavailable("accepted top and bottom reactions do not balance")
    if free_relative > limits["free_residual_relative"]:
        raise EquilibriumObservablesUnavailable("accepted free-DOF residual is above tolerance")
    if energy_identity > limits["energy_reaction_identity_relative"]:
        raise EquilibriumObservablesUnavailable("accepted production energy identity is above tolerance")
    mesh_fingerprint = _canonical_observation_fingerprint({
        "nodes": getattr(state.mesh, "nodes", None),
        "elements": getattr(state.mesh, "elems", None),
    })
    material_fingerprint = _canonical_observation_fingerprint({
        "material": getattr(state, "material", None),
        "elasticity_D": getattr(state, "elasticity_D", None),
    })
    source_fingerprint = _canonical_observation_fingerprint({
        name: getattr(state, name, None) for name in (
            "damage", "displacement", "ep_gp", "rho_gp", "elasticity_D",
            "material", "cohesive_network", "crack_network", "competition",
            "tip_process_state", "junction_process_state", "rng_state",
            "event_counters", "stored_energy_J_per_m", "sharp_wake_model_id",
            "v12_support_state", "void_state", "checkpoint_generation",
        )
    })
    topology_dissipation = (
        float(state.energy_ledgers["hazard_dissipation_J_per_m"])
        if "hazard_dissipation_J_per_m" in getattr(state, "energy_ledgers", {}) else None
    )
    return EquilibriumObservables(
        signed_top_reaction_vector_N_per_m=top_vector,
        signed_bottom_reaction_vector_N_per_m=bottom_vector,
        top_normal_resultant_N_per_m=top,
        bottom_normal_resultant_N_per_m=bottom,
        mean_reaction_magnitude_N_per_m=reaction_scale,
        top_bottom_balance_residual=balance,
        applied_top_displacement_m=top_opening,
        applied_bottom_displacement_m=bottom_opening,
        total_opening_m=opening,
        free_dof_residual_l2_N_per_m=free_norm,
        full_residual_including_reactions_l2_N_per_m=full_norm,
        constrained_reaction_l2_N_per_m=constrained_norm,
        recoverable_elastic_energy_J_per_m=stored,
        total_stored_internal_energy_J_per_m=stored,
        plastic_eigenstrain_half_work_J_per_m=plastic_half_work,
        plastic_internal_dissipation_J_per_m=None,
        topology_event_dissipation_J_per_m=topology_dissipation,
        external_reaction_work_J_per_m=boundary_half_work,
        compliance_m2_per_N=opening / reaction_scale,
        energy_identity_reference_J_per_m=energy_reference,
        energy_identity_residual=energy_identity,
        free_dof_residual_relative=free_relative,
        observation_schema=EQUILIBRIUM_OBSERVABLE_SCHEMA,
        source_state_fingerprint=source_fingerprint,
        mesh_fingerprint=mesh_fingerprint,
        material_fingerprint=material_fingerprint,
    )


class FrozenMapping(dict):
    """Pickle-safe immutable mapping shared by topology siblings."""
    def _immutable(self, *args, **kwargs):
        raise TypeError("accepted topology mapping is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (_make_frozen_mapping, (dict(self),))


def _make_frozen_mapping(values: Mapping[str, Any]) -> FrozenMapping:
    frozen = dict.__new__(FrozenMapping)
    dict.update(frozen, values)
    return frozen


def _freeze(value: Any) -> Any:
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.flags.writeable:
            array = array.copy(); array.setflags(write=False)
        return array
    if isinstance(value, Mapping):
        return _make_frozen_mapping({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _freeze_array_container(value: Any, names: tuple[str, ...]) -> Any:
    changes = {}
    requires_clone = False
    for name in names:
        item = getattr(value, name, None)
        if isinstance(item, np.ndarray):
            changes[name] = _freeze(item)
            requires_clone |= item.flags.writeable
    if not changes:
        return value
    if not requires_clone:
        return value
    if is_dataclass(value):
        return replace(value, **changes)
    clone = copy.copy(value)
    for name, item in changes.items():
        setattr(clone, name, item)
    return clone


@dataclass(frozen=True)
class LiveFEMTopologyState:
    """Complete accepted state crossing a topology transaction boundary."""

    mesh: Any
    boundary: Any
    damage: np.ndarray
    displacement: np.ndarray
    ep_gp: np.ndarray
    rho_gp: np.ndarray
    elasticity_D: np.ndarray
    material: Any
    cohesive_network: Any
    crack_network: CrackNetworkState
    competition: DirectionalCompetitionState
    tip_process_state: Mapping[str, Any]
    junction_process_state: Mapping[str, Any]
    energy_ledgers: Mapping[str, float]
    rng_state: Any
    event_counters: Mapping[str, int]
    stored_energy_J_per_m: float
    sharp_wake_model_id: str = "sharp_wake_causal_v11"
    v12_support_state: Any = None
    void_state: Any = None
    equilibrium_observables: EquilibriumObservables | None = None
    checkpoint_generation: int = 0

    def __post_init__(self) -> None:
        energy = float(self.stored_energy_J_per_m)
        if not math.isfinite(energy):
            raise ValueError("stored FEM energy must be finite")
        object.__setattr__(self, "stored_energy_J_per_m", energy)
        for name in ("damage", "displacement", "ep_gp", "rho_gp", "elasticity_D"):
            source = np.asarray(getattr(self, name), dtype=float)
            array = source if not source.flags.writeable else source.copy()
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "mesh", _freeze_array_container(
            self.mesh, ("nodes", "elems", "area_e", "dNdx_e", "B_e", "element_damage_gp"),
        ))
        object.__setattr__(self, "boundary", _freeze_array_container(
            self.boundary, ("top_nodes", "bot_nodes", "notch_nodes"),
        ))
        object.__setattr__(self, "tip_process_state", _freeze(self.tip_process_state))
        object.__setattr__(self, "junction_process_state", _freeze(self.junction_process_state))
        object.__setattr__(self, "energy_ledgers", _freeze(self.energy_ledgers))
        object.__setattr__(self, "event_counters", _freeze(self.event_counters))
        object.__setattr__(self, "rng_state", _freeze(self.rng_state))
        if self.equilibrium_observables is not None and not isinstance(
            self.equilibrium_observables, EquilibriumObservables
        ):
            raise TypeError("equilibrium_observables must have the accepted typed schema")
        from .sharp_wake_backend_v12 import V11_MODEL_ID, V12_MODEL_ID, select_sharp_wake_model
        selected=select_sharp_wake_model(self.sharp_wake_model_id)
        if selected==V11_MODEL_ID and self.v12_support_state is not None:
            raise ValueError("V11 state may not own V12 support")
        if selected==V12_MODEL_ID and self.v12_support_state is None:
            raise ValueError("V12 state requires authoritative support ownership")
        object.__setattr__(self,"sharp_wake_model_id",selected)

    def isolated_copy(self) -> "LiveFEMTopologyState":
        return LiveFEMTopologyState(
            mesh=self.mesh, boundary=self.boundary,
            damage=self.damage, displacement=self.displacement,
            ep_gp=self.ep_gp, rho_gp=self.rho_gp, elasticity_D=self.elasticity_D,
            material=self.material, cohesive_network=self.cohesive_network,
            crack_network=self.crack_network, competition=self.competition,
            tip_process_state=self.tip_process_state,
            junction_process_state=self.junction_process_state,
            energy_ledgers=self.energy_ledgers, rng_state=self.rng_state,
            event_counters=self.event_counters,
            stored_energy_J_per_m=self.stored_energy_J_per_m,
            sharp_wake_model_id=self.sharp_wake_model_id,
            v12_support_state=self.v12_support_state,
            void_state=self.void_state,
            equilibrium_observables=self.equilibrium_observables,
            checkpoint_generation=self.checkpoint_generation,
        )


@dataclass(frozen=True)
class TopologyArm:
    candidate_id: str
    branch_id: str
    start_xy_m: tuple[float, float]
    end_xy_m: tuple[float, float]
    event_reward_m: float
    hazard_dissipation_J_per_m: float
    event_classification: str = "software_forced_geometry"
    candidate_direction_xy: tuple[float, float] | None = None
    first_intersection_xy_m: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        reward = float(self.event_reward_m)
        dissipation = float(self.hazard_dissipation_J_per_m)
        if reward <= 0.0 or not math.isfinite(reward):
            raise ValueError("each completed event must receive one positive physical reward")
        if dissipation < 0.0 or not math.isfinite(dissipation):
            raise ValueError("hazard-derived dissipation must be finite and nonnegative")
        if math.dist(self.start_xy_m, self.end_xy_m) <= 0.0:
            raise ValueError("topology arm must have positive geometric length")
        classification = str(self.event_classification)
        if classification not in {"software_forced_geometry", "physical_cleavage"}:
            raise ValueError("unknown topology-event classification")
        object.__setattr__(self, "event_classification", classification)
        if classification == "physical_cleavage":
            if dissipation <= 0.0:
                raise ValueError("a physical cleavage event requires nonzero hazard dissipation")
            if self.candidate_direction_xy is None or self.first_intersection_xy_m is None:
                raise ValueError("a physical cleavage event requires direction and first-intersection evidence")
            direction = np.asarray(self.candidate_direction_xy, dtype=float)
            segment = np.asarray(self.end_xy_m, dtype=float) - np.asarray(self.start_xy_m, dtype=float)
            if direction.shape != (2,) or not np.all(np.isfinite(direction)):
                raise ValueError("physical candidate direction must be a finite 2-vector")
            if np.linalg.norm(direction) <= 0.0:
                raise ValueError("physical candidate direction must be nonzero")
            cosine = float(np.dot(direction, segment) / (np.linalg.norm(direction) * np.linalg.norm(segment)))
            if cosine < 1.0 - 1.0e-10:
                raise ValueError("realized arm is not aligned with its selected physical candidate")
            if not np.allclose(self.end_xy_m, self.first_intersection_xy_m, rtol=0.0, atol=1.0e-15):
                raise ValueError("realized endpoint is not the recorded first intersection")


@dataclass(frozen=True)
class TopologyTrialResult:
    accepted: bool
    state: LiveFEMTopologyState
    action_id: str
    energy_release_J_per_m: float
    hazard_dissipation_J_per_m: float
    energy_margin_J_per_m: float
    rejection_reason: str | None
    trial_copy_bytes: int = 0
    trial_copy_wall_time_s: float = 0.0


GeometryTrial = Callable[[LiveFEMTopologyState, tuple[TopologyArm, ...]], LiveFEMTopologyState]
EquilibrateTrial = Callable[[LiveFEMTopologyState], LiveFEMTopologyState]


def equilibrate_fixed_load_with_production_fem(
    state: LiveFEMTopologyState,
) -> LiveFEMTopologyState:
    """Equilibrate directly with the production assembler at fixed opening."""
    from .fem import assemble_mechanics, solve_dirichlet
    from .hazard_energy_event_gate_v10230 import _infer_boundary_opening, _stored_energy

    displacement = np.asarray(state.displacement, dtype=float).copy()
    top, bottom = _infer_boundary_opening(state.boundary, displacement)
    matrix, residual, *_ = assemble_mechanics(
        state.mesh, displacement, state.ep_gp, state.rho_gp, state.damage,
        state.elasticity_D, state.material,
        cohesive_network=state.cohesive_network,
    )
    displacement, _ = solve_dirichlet(
        matrix, residual, displacement, state.boundary, top, bottom,
    )
    _, accepted_residual, sigma_gp, *_ = assemble_mechanics(
        state.mesh, displacement, state.ep_gp, state.rho_gp, state.damage,
        state.elasticity_D, state.material,
        cohesive_network=state.cohesive_network,
    )
    energy = _stored_energy(
        state.mesh, displacement, state.ep_gp, sigma_gp, state.elasticity_D,
    )
    accepted = replace(
        state, displacement=displacement, stored_energy_J_per_m=energy,
        equilibrium_observables=None,
    )
    equilibrium_observables = _measure_accepted_equilibrium(
        accepted, accepted_residual, sigma_gp, energy,
    )
    return replace(
        accepted,
        equilibrium_observables=equilibrium_observables,
        energy_ledgers={**accepted.energy_ledgers, **dict(equilibrium_observables)},
    )


def complete_accepted_state_fingerprint(state: LiveFEMTopologyState) -> str:
    """Hash all accepted production ownership without object identities."""
    from .sharp_wake_backend_v12 import array_fingerprint

    def normalize(value):
        if isinstance(value, np.ndarray):
            return {"array_sha256": array_fingerprint(value)}
        if hasattr(value, "to_dict"):
            return normalize(value.to_dict())
        if is_dataclass(value):
            return normalize({name: getattr(value, name) for name in value.__dataclass_fields__})
        if isinstance(value, Mapping):
            return {str(key): normalize(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
        if isinstance(value, (tuple, list)):
            return [normalize(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return {"nonfinite": str(value)}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return {"type": type(value).__qualname__, "state": normalize(getattr(value, "__dict__", str(value)))}

    payload = {
        "mesh_nodes": state.mesh.nodes,
        "mesh_elems": state.mesh.elems,
        "boundary": state.boundary,
        "damage": state.damage,
        "displacement": state.displacement,
        "ep_gp": state.ep_gp,
        "rho_gp": state.rho_gp,
        "elasticity_D": state.elasticity_D,
        "material": state.material,
        "cohesive_network": state.cohesive_network,
        "crack_network": state.crack_network,
        "competition": state.competition,
        "tip_process_state": state.tip_process_state,
        "junction_process_state": state.junction_process_state,
        "energy_ledgers": state.energy_ledgers,
        "rng_state": state.rng_state,
        "event_counters": state.event_counters,
        "stored_energy_J_per_m": state.stored_energy_J_per_m,
        "sharp_wake_model_id": state.sharp_wake_model_id,
        "v12_support_state": state.v12_support_state,
        "void_state": state.void_state,
        "equilibrium_observables": getattr(state, "equilibrium_observables", None),
        "checkpoint_generation": state.checkpoint_generation,
    }
    encoded = json.dumps(normalize(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def apply_sharp_wake_trial_geometry(
    state: LiveFEMTopologyState,
    arms: tuple[TopologyArm, ...],
    *,
    kill_radius_m: float,
) -> LiveFEMTopologyState:
    """Insert every arm into one copied sharp-wake damage field."""
    backend = SharpWakeBackend()
    current = state
    for index, arm in enumerate(arms):
        result = backend.advance(
            mesh=current.mesh, boundary=current.boundary,
            damage=current.damage, displacement=current.displacement,
            p0=np.asarray(arm.start_xy_m), p1=np.asarray(arm.end_xy_m),
            direction=np.asarray(arm.end_xy_m) - np.asarray(arm.start_xy_m),
            front_id=index, kill_r=float(kill_radius_m),
        )
        if not result.inserted or not math.isclose(
            result.moved, arm.event_reward_m, rel_tol=1.0e-12, abs_tol=1.0e-18
        ):
            raise RuntimeError(f"sharp-wake topology insertion failed: {result.reason}")
        current = replace(
            current, mesh=result.mesh, boundary=result.boundary,
            damage=result.damage, displacement=result.displacement,
        )
    return current


def apply_causal_sharp_wake_trial_geometry(
    state: LiveFEMTopologyState,
    arms: tuple[TopologyArm, ...],
) -> LiveFEMTopologyState:
    """Insert v11 arms with causal P0 support and reject invisible trials."""
    current = state
    audits = []
    for arm in arms:
        current, audit = apply_causal_segment(
            current, np.asarray(arm.start_xy_m), np.asarray(arm.end_xy_m),
        )
        if not audit.mechanically_resolved:
            raise RuntimeError(
                "sharp_wake_trial_not_mechanically_resolved: "
                f"branch={arm.branch_id} candidate={arm.candidate_id}"
            )
        audits.append({
            "branch_id": arm.branch_id,
            "candidate_id": arm.candidate_id,
            **audit.__dict__,
        })
    junction = dict(current.junction_process_state)
    junction["latest_causal_support_trials"] = tuple(audits)
    junction["crack_representation"] = "sharp_wake_causal_v11"
    return replace(current, junction_process_state=junction)


def apply_mechanically_separating_v12_trial_geometry(
    state: LiveFEMTopologyState, arms: tuple[TopologyArm, ...], *,
    source_commit: str, configuration: Mapping[str, Any], transaction_identity: str,
    failure_injector: Callable[[str, LiveFEMTopologyState], None] | None = None,
) -> LiveFEMTopologyState:
    """Realize tentative graph first, then rebuild/certify complete V12 support."""
    from .causal_sharp_wake_v11 import element_damage
    from .mechanically_separating_sharp_wake_v12 import apply_mechanically_separating_graph
    from .sharp_wake_backend_v12 import V12_MODEL_ID, support_state_from_production
    network=state.crack_network
    for arm in arms: network=extend_network_arm(network,arm)
    if failure_injector is not None: failure_injector("graph_edit",replace(state,crack_network=network))
    previous=None
    if state.v12_support_state is not None:
        previous=state.junction_process_state.get("v12_support_record")
    trial,audit=apply_mechanically_separating_graph(state,network,previous_support=previous)
    if failure_injector is not None: failure_injector("support_generation",trial)
    if not audit.certified or audit.mechanically_new_element_count<=0:
        raise RuntimeError("V12 tentative event did not create certified mechanical novelty")
    if failure_injector is not None: failure_injector("support_certification",trial)
    prior=state.v12_support_state.transaction_identity if state.v12_support_state is not None else None
    ownership=support_state_from_production(
        mesh=trial.mesh,crack_network=network,selected_support_elements=audit.selected_element_ids,
        damage_gp=element_damage(trial.mesh,trial.damage),certification_fingerprint=audit.certificate_fingerprint,
        transaction_identity=transaction_identity,previous_accepted_transaction=prior,
        source_commit=source_commit,configuration=configuration,
        checkpoint_generation=state.checkpoint_generation)
    return replace(trial,sharp_wake_model_id=V12_MODEL_ID,v12_support_state=ownership)


def initialize_mechanically_separating_v12(
    state: LiveFEMTopologyState, *, source_commit: str,
    configuration: Mapping[str, Any], transaction_identity: str="v12-initial",
) -> LiveFEMTopologyState:
    """Select V12 only together with certified support for the accepted graph."""
    from .causal_sharp_wake_v11 import element_damage
    from .mechanically_separating_sharp_wake_v12 import apply_mechanically_separating_graph
    from .sharp_wake_backend_v12 import V12_MODEL_ID, support_state_from_production
    trial,audit=apply_mechanically_separating_graph(state,state.crack_network)
    if not audit.certified: raise RuntimeError("initial V12 support is not certified")
    ownership=support_state_from_production(
        mesh=trial.mesh,crack_network=trial.crack_network,
        selected_support_elements=audit.selected_element_ids,
        damage_gp=element_damage(trial.mesh,trial.damage),
        certification_fingerprint=audit.certificate_fingerprint,
        transaction_identity=transaction_identity,previous_accepted_transaction=None,
        source_commit=source_commit,configuration=configuration,
        checkpoint_generation=state.checkpoint_generation)
    return replace(trial,sharp_wake_model_id=V12_MODEL_ID,v12_support_state=ownership)


def remesh_mechanically_separating_v12(
    state: LiveFEMTopologyState, *, mesh, boundary, tentative_network=None,
    transferred_fields: Mapping[str, Any], source_commit: str,
    configuration: Mapping[str, Any], transaction_identity: str,
    failure_injector: Callable[[str, LiveFEMTopologyState], None] | None=None,
) -> LiveFEMTopologyState:
    """Rebuild V12 ownership on a new mesh; never transfer element IDs."""
    required=("damage","displacement","ep_gp","rho_gp","tip_process_state","source_state")
    if any(name not in transferred_fields for name in required):
        raise ValueError("V12 remesh requires every owned structural/history field")
    junction={key:value for key,value in state.junction_process_state.items()
              if key not in ("v12_support_record","v12_graph_support_audit",
                             "v12_accepted_mechanical_fingerprint","v12_trial_mechanical_fingerprint")}
    counters = dict(state.event_counters)
    counters["mesh_generation"] = int(counters.get("mesh_generation", 0)) + 1
    counters["refinement_operation_index"] = int(counters.get("refinement_operation_index", 0)) + 1
    base=replace(state,mesh=mesh,boundary=boundary,
        crack_network=state.crack_network if tentative_network is None else tentative_network,
        damage=np.asarray(transferred_fields["damage"]),
        displacement=np.asarray(transferred_fields["displacement"]),
        ep_gp=np.asarray(transferred_fields["ep_gp"]),rho_gp=np.asarray(transferred_fields["rho_gp"]),
        tip_process_state=transferred_fields["tip_process_state"],
        junction_process_state={**junction, "source_state": transferred_fields["source_state"]},
        event_counters=counters, sharp_wake_model_id="sharp_wake_causal_v11",v12_support_state=None)
    if failure_injector is not None: failure_injector("remesh",base)
    if failure_injector is not None: failure_injector("field_projection",base)
    rebuilt = initialize_mechanically_separating_v12(
        base,source_commit=source_commit,configuration=configuration,
        transaction_identity=transaction_identity)
    if failure_injector is not None: failure_injector("support_rebuild",rebuilt)
    return rebuilt


def apply_v12_production_trial_geometry(
    state: LiveFEMTopologyState,
    arms: tuple[TopologyArm, ...],
    *,
    source_commit: str,
    configuration: Mapping[str, Any],
    transaction_identity: str,
    failure_injector: Callable[[str, LiveFEMTopologyState], None] | None = None,
    refinement_levels: int = 3,
    prepare_support_state: Callable[[LiveFEMTopologyState], LiveFEMTopologyState] | None = None,
) -> LiveFEMTopologyState:
    """Perform graph edit, conforming remesh, physical field transfer and support rebuild."""
    from .adaptive_multitip_mesh_v11 import refine_accepted_state

    network = state.crack_network
    for arm in arms:
        network = extend_network_arm(network, arm)
    graph_state = replace(state, crack_network=network)
    if failure_injector is not None:
        failure_injector("graph_edit", graph_state)

    graph_segments = tuple(
        (a, b) for branch in network.branches for a, b in zip(branch.path, branch.path[1:])
    )
    refined = graph_state
    if refinement_levels < 1:
        raise ValueError("production V12 remesh requires at least one refinement level")
    for level in range(int(refinement_levels)):
        centroids = np.asarray(refined.mesh.nodes)[np.asarray(refined.mesh.elems)].mean(axis=1)
        marked: set[int] = set()
        for start, end in graph_segments:
            a = np.asarray(start, dtype=float)
            b = np.asarray(end, dtype=float)
            delta = b - a
            length2 = float(delta @ delta)
            t = np.clip(((centroids - a) @ delta) / max(length2, 1.0e-300), 0.0, 1.0)
            distance = np.linalg.norm(centroids - (a + t[:, None] * delta), axis=1)
            local_h = np.sqrt(np.maximum(np.asarray(refined.mesh.area_e), 1.0e-300))
            marked.update(np.flatnonzero(distance <= 2.0 * local_h).tolist())
        if not marked:
            raise RuntimeError("production V12 remesh found no physical event support")
        refined, _ = refine_accepted_state(
            refined,
            marked_parent_elements=tuple(sorted(marked)),
            active_tip_ids=network.active_tip_ids,
            generation=int(state.event_counters.get("mesh_generation", 0)) + 1,
            operation_index=int(state.event_counters.get("refinement_operation_index", 0)) + level + 1,
        )
    counters = dict(refined.event_counters)
    counters.update({
        "mesh_generation": int(state.event_counters.get("mesh_generation", 0)) + 1,
        "refinement_operation_index": int(state.event_counters.get("refinement_operation_index", 0)) + 1,
    })
    refined = replace(refined, event_counters=counters)
    # A contact event may explicitly retire its arriving front before the
    # first support certificate. The hook sees the actual refined boundary;
    # it cannot make an active tip eligible for boundary-terminal clipping.
    # Its state remains an isolated trial until the caller's energy/topology
    # transaction commits.
    support_base = state
    if prepare_support_state is not None:
        prepared = prepare_support_state(refined)
        if prepared.mesh is not refined.mesh:
            raise ValueError("support preparation must not replace the refined mesh")
        network = prepared.crack_network
        support_base = replace(prepared, event_counters=state.event_counters)
    return remesh_mechanically_separating_v12(
        support_base,
        mesh=refined.mesh,
        boundary=refined.boundary,
        tentative_network=network,
        transferred_fields={
            "damage": refined.damage,
            "displacement": refined.displacement,
            "ep_gp": refined.ep_gp,
            "rho_gp": refined.rho_gp,
            "tip_process_state": state.tip_process_state,
            "source_state": state.junction_process_state.get("source_state", {}),
        },
        source_commit=source_commit,
        configuration=configuration,
        transaction_identity=transaction_identity,
        failure_injector=failure_injector,
    )


def _replace_branch(network: CrackNetworkState, updated: CrackBranchState) -> CrackNetworkState:
    branches = tuple(updated if item.branch_id == updated.branch_id else item for item in network.branches)
    return replace(network, branches=branches, geometry_generation=network.geometry_generation + 1)


def extend_network_arm(network: CrackNetworkState, arm: TopologyArm) -> CrackNetworkState:
    """Append an accepted segment without splitting any process-zone ledger."""
    branch = network.branch(arm.branch_id)
    if branch.status != "active" or branch.tip != tuple(arm.start_xy_m):
        raise ValueError("arm must extend an active branch from its accepted tip")
    angle = math.atan2(
        arm.end_xy_m[1] - arm.start_xy_m[1],
        arm.end_xy_m[0] - arm.start_xy_m[0],
    )
    local = dict(branch.local_state)
    edges = list(local.get("committed_edges", ()))
    edges.append({
        "start_point_m": list(arm.start_xy_m),
        "end_point_m": list(arm.end_xy_m),
        "branch_id": arm.branch_id,
        "parent_or_junction_id": branch.parent_branch_id,
        "commit_event_id": int(network.geometry_generation + 1),
    })
    local["committed_edges"] = edges
    updated = replace(
        branch,
        path=branch.path + (tuple(arm.end_xy_m),),
        orientation_history_rad=(
            (angle,) if len(branch.path) == 1
            else branch.orientation_history_rad + (angle,)
        ),
        local_state=local,
    )
    return _replace_branch(network, updated)


def clip_arm_at_first_intersection(
    network: CrackNetworkState, arm: TopologyArm
) -> tuple[TopologyArm, str | None]:
    """Clip an incoming arm at its first exact hit without changing either path."""
    p0 = np.asarray(arm.start_xy_m, dtype=float)
    p1 = np.asarray(arm.end_xy_m, dtype=float)
    best: tuple[float, np.ndarray, str] | None = None
    for branch in network.branches:
        for index, (a, b) in enumerate(zip(branch.path, branch.path[1:])):
            if branch.branch_id == arm.branch_id and index == len(branch.path) - 2:
                continue
            hit = segment_intersection_first(p0, p1, np.asarray(a), np.asarray(b))
            if hit is not None and (best is None or hit[0] < best[0]):
                best = hit[0], hit[1], branch.branch_id
    if best is None:
        return arm, None
    _, point, target = best
    clipped = replace(
        arm, end_xy_m=(float(point[0]), float(point[1])),
        event_reward_m=float(np.linalg.norm(point - p0)),
    )
    return clipped, target


def mark_coalesced(network: CrackNetworkState, incoming_branch_id: str, target_branch_id: str) -> CrackNetworkState:
    """Deactivate only the incoming tip and preserve both committed paths."""
    if incoming_branch_id == target_branch_id:
        raise ValueError("a branch cannot coalesce into itself")
    network.branch(target_branch_id)
    incoming = network.branch(incoming_branch_id)
    if incoming.status != "active":
        raise ValueError("only an active incoming branch can coalesce")
    local = dict(incoming.local_state)
    local.update({"coalesced": True, "merge_target_branch_id": target_branch_id})
    return _replace_branch(network, replace(incoming, status="merged", local_state=local))


def execute_topology_trial(
    accepted: LiveFEMTopologyState,
    proposal: CompetingActionProposal,
    arms: Iterable[TopologyArm],
    *,
    apply_trial_geometry: GeometryTrial,
    equilibrate_fixed_load: EquilibrateTrial,
    relative_energy_tolerance: float = 1.0e-8,
    absolute_energy_tolerance_J_per_m: float = 1.0e-12,
    network_geometry_already_realized: bool = False,
    failure_injector: Callable[[str, LiveFEMTopologyState], None] | None = None,
) -> TopologyTrialResult:
    """Reserve, trial and atomically accept/release one one- or two-arm action."""
    trial_arms = tuple(sorted(arms, key=lambda item: item.candidate_id))
    if len(trial_arms) not in (1, 2):
        raise ValueError("v11 topology actions contain exactly one or two arms")
    if tuple(item.candidate_id for item in trial_arms) != proposal.member_candidate_ids:
        raise ValueError("trial arms do not match the proposal's physical candidates")
    rewards = tuple(item.event_reward_m for item in trial_arms)
    reserved = reserve_action(accepted.competition, proposal, event_rewards_m=rewards)
    reserved_state = replace(accepted, competition=reserved)
    copy_start = time.perf_counter()
    isolated = reserved_state.isolated_copy()
    if failure_injector is not None: failure_injector("accepted_snapshot",isolated)
    copy_wall = time.perf_counter() - copy_start
    mechanics = ("damage", "displacement", "ep_gp", "rho_gp", "elasticity_D")
    copy_bytes = sum(
        int(getattr(isolated, name).nbytes)
        for name in mechanics if getattr(isolated, name) is not getattr(reserved_state, name)
    )
    trial = apply_trial_geometry(isolated, trial_arms)
    if failure_injector is not None: failure_injector("field_transfer",trial)
    if network_geometry_already_realized:
        for arm in trial_arms:
            branch = trial.crack_network.branch(arm.branch_id)
            if branch.tip != tuple(arm.end_xy_m):
                raise RuntimeError(
                    "trial network does not contain the exact realized arm endpoint"
                )
    trial = equilibrate_fixed_load(trial)
    if failure_injector is not None: failure_injector("equilibrium",trial)
    released = float(accepted.stored_energy_J_per_m - trial.stored_energy_J_per_m)
    dissipation = math.fsum(item.hazard_dissipation_J_per_m for item in trial_arms)
    tolerance = max(
        float(absolute_energy_tolerance_J_per_m),
        max(float(relative_energy_tolerance), 0.0)
        * max(abs(accepted.stored_energy_J_per_m), abs(trial.stored_energy_J_per_m), dissipation),
    )
    margin = released - dissipation
    if released + tolerance < dissipation:
        # The reservation existed only inside the isolated trial. Discarding that
        # snapshot is the exact release operation and leaves no transactional
        # history in accepted production state.
        release_reservation(reserved, proposal.action_id)
        return TopologyTrialResult(
            False, accepted, proposal.action_id, released, dissipation, margin,
            "insufficient_whole_topology_energy_release", copy_bytes, copy_wall,
        )
    if failure_injector is not None: failure_injector("energy_gate",trial)
    committed_competition = accept_reservation(trial.competition, proposal.action_id)
    committed_network = trial.crack_network
    if not network_geometry_already_realized:
        for arm in trial_arms:
            committed_network = extend_network_arm(committed_network, arm)
    committed = replace(
        trial,
        competition=committed_competition,
        crack_network=committed_network,
        event_counters={
            **trial.event_counters,
            "topology_actions": int(trial.event_counters.get("topology_actions", 0)) + 1,
        },
        energy_ledgers={
            **trial.energy_ledgers,
            "topology_release_J_per_m": float(trial.energy_ledgers.get("topology_release_J_per_m", 0.0)) + released,
            "hazard_dissipation_J_per_m": float(trial.energy_ledgers.get("hazard_dissipation_J_per_m", 0.0)) + dissipation,
        },
    )
    if failure_injector is not None:
        failure_injector("process_state_update",committed)
        failure_injector("topology_verification",committed)
        failure_injector("late_event_veto",committed)
    return TopologyTrialResult(
        True, committed, proposal.action_id, released, dissipation, margin, None,
        copy_bytes, copy_wall,
    )


__all__ = [
    "LiveFEMTopologyState", "MODEL_ID", "TopologyArm", "TopologyTrialResult",
    "apply_sharp_wake_trial_geometry", "apply_causal_sharp_wake_trial_geometry", "apply_mechanically_separating_v12_trial_geometry", "apply_v12_production_trial_geometry", "initialize_mechanically_separating_v12", "remesh_mechanically_separating_v12",
    "clip_arm_at_first_intersection",
    "EQUILIBRIUM_OBSERVABLE_KEYS", "EQUILIBRIUM_OBSERVABLE_SCHEMA",
    "ENERGY_ABSOLUTE_FLOOR_J_PER_M", "EquilibriumObservablesUnavailable",
    "REACTION_ABSOLUTE_FLOOR_N_PER_M", "complete_accepted_state_fingerprint",
    "equilibrate_fixed_load_with_production_fem", "execute_topology_trial",
    "require_equilibrium_observables",
    "extend_network_arm", "mark_coalesced",
]
