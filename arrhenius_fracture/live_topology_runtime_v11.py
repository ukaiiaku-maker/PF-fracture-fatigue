"""Run-level routing for the irreversible switch to v11 exact-topology mechanics."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .kernel_resolver_v11 import resolve_live_topology_request
from .live_topology_kernel_registry_v11 import (
    MechanicsProviderRoutingState, PREBRANCH_PROVIDER_ID, lock_live_provider,
)
from .live_topology_kernel_v11 import LiveTopologyRequest, PROVIDER_ID


@dataclass(frozen=True)
class LiveTopologyRuntime:
    cache_root: str
    routing: MechanicsProviderRoutingState = MechanicsProviderRoutingState()
    live_fem_solve_count: int = 0
    accepted_provider_state_count: int = 0

    def transition(
        self, *, step: int, state_hash: str, legacy_result: Mapping[str, Any],
        request: LiveTopologyRequest, protected_state: Any,
    ) -> tuple["LiveTopologyRuntime", dict[str, Any]]:
        if self.routing.active_mechanics_provider != PREBRANCH_PROVIDER_ID:
            raise RuntimeError("v11 mechanics transition is already locked")
        live, cache_hit = resolve_live_topology_request(
            request, cache_root=self.cache_root, accepted=True
        )
        routing = lock_live_provider(
            self.routing, step=step, state_hash=state_hash,
            legacy_result=legacy_result, live_result=live,
            protected_state=protected_state,
        )
        return replace(
            self, routing=routing,
            live_fem_solve_count=self.live_fem_solve_count + (0 if cache_hit else 1),
            accepted_provider_state_count=self.accepted_provider_state_count + (0 if cache_hit else 1),
        ), live

    def evaluate_trial(self, request: LiveTopologyRequest) -> tuple["LiveTopologyRuntime", dict[str, Any]]:
        if self.routing.active_mechanics_provider != PROVIDER_ID:
            raise RuntimeError("branch topology trials require the locked v11 live provider")
        result, _ = resolve_live_topology_request(
            request, cache_root=self.cache_root, accepted=False
        )
        return replace(self, live_fem_solve_count=self.live_fem_solve_count + 1), result

    def accept_trial(
        self, request: LiveTopologyRequest, trial_result: Mapping[str, Any]
    ) -> "LiveTopologyRuntime":
        if trial_result.get("topology_fingerprint") is None:
            raise ValueError("trial result lacks exact topology identity")
        accepted, cache_hit = resolve_live_topology_request(
            request, cache_root=self.cache_root, accepted=True
        )
        if accepted["topology_fingerprint"] != trial_result["topology_fingerprint"]:
            raise RuntimeError("accepted topology differs from evaluated trial")
        return replace(
            self,
            routing=replace(
                self.routing,
                topology_fingerprint=str(accepted["topology_fingerprint"]),
            ),
            live_fem_solve_count=self.live_fem_solve_count + (0 if cache_hit else 1),
            accepted_provider_state_count=self.accepted_provider_state_count + (0 if cache_hit else 1),
        )

    def audit_payload(self) -> dict[str, Any]:
        return {
            "initial_mechanics_provider": self.routing.initial_mechanics_provider,
            "active_mechanics_provider": self.routing.active_mechanics_provider,
            "transition_step": self.routing.transition_step,
            "transition_state_hash": self.routing.transition_state_hash,
            "transition_parity_results": self.routing.transition_parity_results,
            "topology_fingerprint": self.routing.topology_fingerprint,
            "live_fem_solve_count": self.live_fem_solve_count,
            "accepted_provider_state_count": self.accepted_provider_state_count,
            "cache_root": str(Path(self.cache_root).resolve()),
            "topology_interpolation": "disabled",
        }


__all__ = ["LiveTopologyRuntime"]
