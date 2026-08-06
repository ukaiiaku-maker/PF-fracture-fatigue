"""Hybrid v10.2.28-to-v11 live mechanics transition and provider lock."""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import math
import pickle
from typing import Any, Mapping

from .live_topology_kernel_v11 import PROVIDER_ID


PREBRANCH_PROVIDER_ID = "v10.2.28_direct_prescribed_geometry_fem_v1"


@dataclass(frozen=True)
class MechanicsProviderRoutingState:
    initial_mechanics_provider: str = PREBRANCH_PROVIDER_ID
    active_mechanics_provider: str = PREBRANCH_PROVIDER_ID
    transition_step: int | None = None
    transition_state_hash: str | None = None
    transition_parity_results: Mapping[str, Any] | None = None
    topology_fingerprint: str | None = None


def _relative(left: float, right: float) -> float:
    return abs(float(left) - float(right)) / max(abs(float(left)), abs(float(right)), 1.0e-300)


def validate_single_front_transition(
    legacy: Mapping[str, Any], live: Mapping[str, Any],
    *, reaction_energy_rtol: float = 1.0e-6, drive_rtol: float = 1.0e-4,
) -> dict[str, Any]:
    """Require parity without changing any kinetic or competition state."""
    comparisons = {}
    for name in ("reaction_force", "recoverable_potential_energy_J_per_m"):
        residual = _relative(legacy[name], live["base_equilibrium"][name])
        comparisons[name] = residual
        if residual > reaction_energy_rtol:
            raise RuntimeError(f"live-provider transition parity failed for {name}: {residual}")
    legacy_directional = {item["candidate_id"]: item for item in legacy["directional"]}
    live_directional = {item["candidate_id"]: item for item in live["tips"][0]["directional"]}
    if set(legacy_directional) != set(live_directional):
        raise RuntimeError("live-provider transition changed directional candidates")
    for candidate_id in sorted(legacy_directional):
        old = legacy_directional[candidate_id]
        new = live_directional[candidate_id]
        for name in ("signed_J_J_per_m2", "positive_J_J_per_m2", "K_directional_Pa_sqrt_m"):
            if math.copysign(1.0, float(old[name])) != math.copysign(1.0, float(new[name])):
                raise RuntimeError(f"live-provider transition changed sign for {candidate_id} {name}")
            residual = _relative(old[name], new[name])
            comparisons[f"{candidate_id}:{name}"] = residual
            if residual > drive_rtol:
                raise RuntimeError(f"live-provider transition parity failed for {candidate_id} {name}: {residual}")
    comparisons.update({
        "passed": True, "reaction_energy_rtol": reaction_energy_rtol,
        "drive_rtol": drive_rtol, "sign_agreement": True,
    })
    return comparisons


def lock_live_provider(
    routing: MechanicsProviderRoutingState, *, step: int, state_hash: str,
    legacy_result: Mapping[str, Any], live_result: Mapping[str, Any],
    protected_state: Any,
) -> MechanicsProviderRoutingState:
    if routing.active_mechanics_provider != PREBRANCH_PROVIDER_ID:
        raise RuntimeError("mechanics provider transition may occur only once")
    before = pickle.dumps(protected_state, protocol=5)
    parity = validate_single_front_transition(legacy_result, live_result)
    after = pickle.dumps(protected_state, protocol=5)
    if before != after:
        raise RuntimeError("mechanics provider transition mutated protected production state")
    return replace(
        routing, active_mechanics_provider=PROVIDER_ID,
        transition_step=int(step), transition_state_hash=str(state_hash),
        transition_parity_results=copy.deepcopy(parity),
        topology_fingerprint=str(live_result["topology_fingerprint"]),
    )


__all__ = [
    "MechanicsProviderRoutingState", "PREBRANCH_PROVIDER_ID",
    "lock_live_provider", "validate_single_front_transition",
]
