"""Pluggable void-formation boundary around the accepted transport owner.

Only the qualified vacancy-clustering mechanism is implemented.  This module
contains no kinetics: every operation delegates to the state-owned V2
transport implementation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import pickle
from types import MappingProxyType
from typing import Any, Mapping


VACANCY_CLUSTERING_V1 = "VACANCY_CLUSTERING_V1"


class VoidFormationMechanism(ABC):
    mechanism_id: str

    @abstractmethod
    def preview(self, accepted_transport_state: Any, **accepted_fields: Any): ...

    @abstractmethod
    def advance(self, accepted_transport_state: Any, **accepted_fields: Any): ...

    @abstractmethod
    def commit(self, original_transport_state: Any, advanced_transport_state: Any,
               *, expected_site_id: str | None = None): ...

    @abstractmethod
    def checkpoint_payload(self, accepted_transport_state: Any) -> Mapping[str, Any]: ...

    @abstractmethod
    def conservation_audit(self, accepted_transport_state: Any) -> Mapping[str, Any]: ...


class VacancyClusteringV1(VoidFormationMechanism):
    """The sole enabled mechanism, backed by actual captured inventory."""

    mechanism_id = VACANCY_CLUSTERING_V1

    def preview(self, accepted_transport_state: Any, **accepted_fields: Any):
        from .vacancy_transport_phase_field_v2 import preview_site_inventory_event
        return preview_site_inventory_event(
            accepted_transport_state, accepted_transport_state.transport_mesh,
            accepted_transport_state.surface_registry,
            row=accepted_transport_state.parameter_row, **accepted_fields,
        )

    def advance(self, accepted_transport_state: Any, **accepted_fields: Any):
        from .vacancy_transport_phase_field_v2 import advance_transport_to_time
        return advance_transport_to_time(
            accepted_transport_state, accepted_transport_state.transport_mesh,
            accepted_transport_state.surface_registry,
            row=accepted_transport_state.parameter_row, **accepted_fields,
        )

    def commit(self, original_transport_state: Any, advanced_transport_state: Any,
               *, expected_site_id: str | None = None):
        from .vacancy_transport_phase_field_v2 import commit_site_inventory_hit
        return commit_site_inventory_hit(
            original_transport_state, advanced_transport_state,
            expected_site_id=expected_site_id,
        )

    def checkpoint_payload(self, accepted_transport_state: Any) -> Mapping[str, Any]:
        payload = pickle.dumps(accepted_transport_state, protocol=5)
        return MappingProxyType({
            "mechanism_id": self.mechanism_id,
            "state_sha256": hashlib.sha256(payload).hexdigest(),
            "parameter_row_id": accepted_transport_state.parameter_row_id,
            "parameter_row_fingerprint": accepted_transport_state.parameter_row_fingerprint,
            "site_ids": tuple(site.site_id for site in accepted_transport_state.site_states),
        })

    def conservation_audit(self, accepted_transport_state: Any) -> Mapping[str, Any]:
        from .vacancy_transport_phase_field_v2 import transport_conservation
        audit = dict(transport_conservation(
            accepted_transport_state,
            accepted_transport_state.transport_mesh.area_e,
        ))
        audit["mechanism_id"] = self.mechanism_id
        audit["inventory_backed_hits"] = all(
            site.reserved_inventory_area_m2 + 1.0e-24 >=
            site.required_embryo_inventory_area_m2*site.hit_count/site.required_hit_count
            for site in accepted_transport_state.site_states
            if site.required_hit_count > 0
        )
        return MappingProxyType(audit)


def active_void_formation_mechanism(mechanism_id: str = VACANCY_CLUSTERING_V1):
    if mechanism_id != VACANCY_CLUSTERING_V1:
        raise ValueError("unimplemented void-formation mechanism: "+str(mechanism_id))
    return VacancyClusteringV1()


__all__ = [
    "VACANCY_CLUSTERING_V1", "VacancyClusteringV1",
    "VoidFormationMechanism", "active_void_formation_mechanism",
]
