"""Versioned V12 production selection and authoritative support ownership."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
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
    configuration_hash: str
    checkpoint_generation: int
    branch_vertex_lineage_fingerprint: str
    model_id: str=V12_MODEL_ID
    schema_version: str=SCHEMA

    def __post_init__(self):
        if self.model_id!=V12_MODEL_ID or self.schema_version!=SCHEMA:
            raise ValueError("V12 support state has incompatible identity or schema")
        if self.mesh_generation<0 or self.checkpoint_generation<0 or not np.isfinite(self.physical_graph_length_m) or self.physical_graph_length_m<0:
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
                "source_commit":self.source_commit,
                "configuration_hash":self.configuration_hash,
                "checkpoint_generation":self.checkpoint_generation,
                "branch_vertex_lineage_fingerprint":self.branch_vertex_lineage_fingerprint}

def support_state_from_production(*, mesh, crack_network, selected_support_elements,
                                  damage_gp, certification_fingerprint: str,
                                  transaction_identity: str,
                                  previous_accepted_transaction: str | None,
                                  source_commit: str, configuration: Any,
                                  checkpoint_generation: int=0):
    """Construct complete ownership from authoritative graph and current mesh."""
    from .mechanically_separating_sharp_wake_v12 import graph_fingerprint, unique_graph_length
    graph_dict=crack_network.to_dict()
    lineage=json.dumps(graph_dict,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    config=json.dumps(configuration,sort_keys=True,separators=(",",":"),default=str,allow_nan=False).encode()
    return V12SharpWakeSupportState(
        mesh_geometry_fingerprint=array_fingerprint(mesh.nodes),
        mesh_connectivity_fingerprint=array_fingerprint(mesh.elems),
        mesh_generation=int(getattr(mesh,"geometry_generation",getattr(mesh,"generation",0))),
        complete_crack_graph_fingerprint=graph_fingerprint(crack_network),
        physical_graph_length_m=unique_graph_length(crack_network),
        certification_arc_fingerprint=str(certification_fingerprint),
        selected_support_elements=tuple(map(int,selected_support_elements)),
        accepted_p0_damage_fingerprint=array_fingerprint(damage_gp),
        active_tip_identities=tuple(sorted(map(str,crack_network.active_tip_ids))),
        transaction_identity=str(transaction_identity),
        previous_accepted_transaction=previous_accepted_transaction,
        source_commit=str(source_commit),configuration_hash=hashlib.sha256(config).hexdigest(),
        checkpoint_generation=int(checkpoint_generation),
        branch_vertex_lineage_fingerprint=hashlib.sha256(lineage).hexdigest())

__all__=["DEFAULT_MODEL_ID","SCHEMA","V11_MODEL_ID","V12_MODEL_ID","V12SharpWakeSupportState","array_fingerprint","select_sharp_wake_model","support_state_from_production"]
