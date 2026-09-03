"""Versioned V12 production selection and authoritative support ownership."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from typing import Any
import numpy as np

V11_MODEL_ID="sharp_wake_causal_v11"
V12_MODEL_ID="sharp_wake_mechanically_separating_v12"
DEFAULT_MODEL_ID=V11_MODEL_ID
SCHEMA="v12.sharp-wake-support-state/1"

def select_sharp_wake_model(value: str | None = None) -> str:
    selected=DEFAULT_MODEL_ID if value is None else str(value)
    if selected not in (V11_MODEL_ID,V12_MODEL_ID):
        raise ValueError(f"unsupported sharp_wake_model_id {selected!r}")
    return selected

def array_fingerprint(value: Any) -> str:
    a=np.ascontiguousarray(value)
    return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

@dataclass(frozen=True)
class V12SharpWakeSupportState:
    mesh_geometry_fingerprint: str
    mesh_connectivity_fingerprint: str
    mesh_generation: int
    complete_crack_graph_fingerprint: str
    physical_graph_length_m: float
    certification_arc_fingerprint: str
    selected_support_elements: tuple[int,...]
    accepted_p0_damage_fingerprint: str
    active_tip_identities: tuple[str,...]
    transaction_identity: str
    previous_accepted_transaction: str | None
    source_commit: str
    model_id: str=V12_MODEL_ID
    schema_version: str=SCHEMA

    def __post_init__(self):
        if self.model_id!=V12_MODEL_ID or self.schema_version!=SCHEMA:
            raise ValueError("V12 support state has incompatible identity or schema")
        if self.mesh_generation<0 or not np.isfinite(self.physical_graph_length_m) or self.physical_graph_length_m<0:
            raise ValueError("V12 support state has invalid mesh generation or graph length")
        ids=tuple(int(i) for i in self.selected_support_elements)
        if any(i<0 for i in ids) or len(set(ids))!=len(ids):
            raise ValueError("V12 support element ownership must be unique and nonnegative")
        object.__setattr__(self,"selected_support_elements",ids)

    def provenance_for_remesh(self) -> dict[str,Any]:
        """Return only physical/provenance state; stale element IDs are excluded."""
        return {"model_id":self.model_id,"schema_version":self.schema_version,
                "complete_crack_graph_fingerprint":self.complete_crack_graph_fingerprint,
                "physical_graph_length_m":self.physical_graph_length_m,
                "certification_arc_fingerprint":self.certification_arc_fingerprint,
                "active_tip_identities":self.active_tip_identities,
                "previous_accepted_transaction":self.transaction_identity,
                "source_commit":self.source_commit}

__all__=["DEFAULT_MODEL_ID","SCHEMA","V11_MODEL_ID","V12_MODEL_ID","V12SharpWakeSupportState","array_fingerprint","select_sharp_wake_model"]
