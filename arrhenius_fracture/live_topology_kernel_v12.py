"""Dynamically sized V12 route to the exact-topology PF mechanics provider.

The inherited V11 implementation iterates over dynamic crack-network and
candidate registries.  Its ``MAXIMUM_FRONTS_SUPPORTED=16`` value is emitted as
metadata but is not enforced by the evaluator.  V12 therefore routes directly
to that evaluator and makes any front limit an explicit operational resource
policy, never a constitutive or topology rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .general_multifront_v12 import ResourcePolicy


SCHEMA = "v12.dynamic-exact-topology-provider/1"
PROVIDER_ID = "v12_dynamic_exact_crack_network_live_fem_v1"


class ProviderResourceLimitReached(RuntimeError):
    pass


@dataclass(frozen=True)
class DynamicExactTopologyProviderV12:
    resource_policy: ResourcePolicy

    def evaluate(self, request: Any) -> dict[str, Any]:
        active_count = len(request.crack_network.active_tip_ids)
        limit = self.resource_policy.front_resource_limit
        if limit is not None and active_count > limit:
            raise ProviderResourceLimitReached(
                "configured_front_resource_limit_reached: "
                f"active={active_count} configured_limit={limit}"
            )
        # Lazy import is essential: source-complete preflight reaches the
        # pre-solve boundary without looking up or starting this provider.
        from .live_topology_kernel_v11 import (
            MAXIMUM_FRONTS_SUPPORTED as V11_REPORTED_LIMIT,
            evaluate_exact_topology,
        )

        result = dict(evaluate_exact_topology(request))
        result.update({
            "schema": SCHEMA,
            "kernel_provider_id": PROVIDER_ID,
            "maximum_fronts_supported": None,
            "operational_front_resource_limit": limit,
            "inherited_v11_reported_limit_not_enforced": V11_REPORTED_LIMIT,
            "cardinality_semantics": "dynamic_registries_operational_resource_policy",
        })
        return result


__all__ = [
    "DynamicExactTopologyProviderV12", "PROVIDER_ID",
    "ProviderResourceLimitReached", "SCHEMA",
]
