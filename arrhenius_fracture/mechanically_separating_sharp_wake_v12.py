"""Graph-aware mechanically separating P0 sharp wake (V12 prerequisite).

This module is deliberately not installed into the V11 transaction path.  It
defines a new physical representation and fails closed when a graph edge has no
mesh-resolved interior support from which a separating node-star can be built.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib, math
import numpy as np

from .causal_sharp_wake_v11 import causal_segment_support, element_damage, mechanical_fingerprint

MODEL_ID="sharp_wake_mechanically_separating_v12"

@dataclass(frozen=True)
class GraphSupportAudit:
    selected_element_ids: tuple[int,...]
    newly_selected_element_ids: tuple[int,...]
    selected_area_m2: float
    maximum_normal_support_width_m: float
    active_tip_forward_leakage_m: float
    graph_fingerprint: str
    support_fingerprint: str
    vertex_classes: tuple[tuple[str,str],...]
    segment_partition_invariant_length_m: float
    mesh_size_m: float
    width_over_h: float
    forward_leakage_over_h: float
    unresolved_bridge_node_ids: tuple[int,...]
    certified: bool
    certification_reason: str

def _point_key(point): return f"{float(point[0]):.17g},{float(point[1]):.17g}"
def _digest(value: bytes): return hashlib.sha256(value).hexdigest()

def classify_graph_vertices(network):
    incidence={}; active={tuple(network.branch(i).tip) for i in network.active_tip_ids}
    roots={tuple(b.root) for b in network.branches if b.parent_branch_id is None}
    merged={tuple(b.tip) for b in network.branches if b.status=="merged"}
    for branch in network.branches:
        for a,b in zip(branch.path,branch.path[1:]):
            incidence[tuple(a)]=incidence.get(tuple(a),0)+1; incidence[tuple(b)]=incidence.get(tuple(b),0)+1
    classes={}
    for point,degree in incidence.items():
        if point in active: kind="active_tip"
        elif point in roots: kind="physical_root"
        elif point in merged: kind="merged_vertex"
        elif degree>=3: kind="branch_junction"
        elif degree==2: kind="degree_two_interior"
        else: kind="inactive_terminal"
        classes[point]=kind
    return classes

def _segments(network):
    return tuple(sorted(((tuple(a),tuple(b),branch.branch_id,index) for branch in network.branches
      for index,(a,b) in enumerate(zip(branch.path,branch.path[1:]))),key=lambda v:(v[2],v[3])))

def _mesh_size(mesh):
    triangles=np.asarray(mesh.nodes,float)[np.asarray(mesh.elems,int)]
    edges=np.concatenate((triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,1],triangles[:,0]-triangles[:,2]))
    return float(np.max(np.linalg.norm(edges,axis=1)))

def _unresolved_node_star_bridges(mesh,exact,candidate_nodes,selected):
    """Return exact-support nodes that retain any intact nodal coupling.

    P1 displacement degrees of freedom couple through nodes.  A graph-interior
    node is mechanically closed only when the complete incident element star is
    disabled.  Active-tip nodes are deliberately absent from ``candidate_nodes``
    and therefore retain the sole permitted half-open end of the wake.
    """
    selected_mask=np.zeros(int(mesh.ne),dtype=bool); selected_mask[np.asarray(sorted(selected),int)]=True
    exact_nodes=set(map(int,np.unique(mesh.elems[np.asarray(sorted(exact),int)])))
    required=exact_nodes.intersection(map(int,candidate_nodes))
    unresolved=[]
    for node in sorted(required):
        incident=np.flatnonzero(np.any(mesh.elems==node,axis=1))
        if not np.all(selected_mask[incident]): unresolved.append(node)
    return tuple(unresolved)

def mechanically_separating_graph_support(mesh,network,active_tip_ids=None,previous_support=None,*,tolerance=1e-12):
    """Build deterministic monotone O(h) support from the complete crack graph."""
    active_ids=tuple(sorted(network.active_tip_ids if active_tip_ids is None else active_tip_ids))
    if active_ids!=tuple(sorted(network.active_tip_ids)): raise ValueError("active_tip_ids must match accepted crack network")
    segments=_segments(network)
    if not segments: raise RuntimeError("v12_support_not_certified: crack graph has no edges")
    exact=set(); graph_length=0.; segment_records=[]
    for a,b,branch_id,index in segments:
        p0=np.asarray(a,float); p1=np.asarray(b,float); ids,lengths=causal_segment_support(mesh,p0,p1,tolerance=tolerance)
        if not len(ids): raise RuntimeError(f"v12_support_not_certified: unresolved edge {branch_id}:{index}")
        exact.update(map(int,ids)); graph_length+=float(np.linalg.norm(p1-p0)); segment_records.append((p0,p1,ids))
    # Connectivity closure: every node owned by an exactly intersected element
    # is closed unless it lies at a currently active graph tip. This is defined
    # for aligned and nonaligned meshes and cannot silently reduce to V11.
    active_points=np.asarray([network.branch(i).tip for i in active_ids],float)
    candidate_nodes=np.unique(mesh.elems[np.asarray(sorted(exact),int)])
    scale=max(float(np.sqrt(np.max(mesh.area_e))),1e-300)
    if len(active_points):
        distance=np.min(np.linalg.norm(mesh.nodes[candidate_nodes,None,:]-active_points[None,:,:],axis=2),axis=1)
        candidate_nodes=candidate_nodes[distance>tolerance*max(scale,1.)]
    closure=np.flatnonzero(np.any(np.isin(mesh.elems,candidate_nodes),axis=1))
    selected=set(map(int,closure))|exact
    previous=set() if previous_support is None else set(map(int,np.asarray(previous_support,int)))
    selected|=previous
    selected_ids=np.asarray(sorted(selected),int)
    newly=np.asarray(sorted(selected-previous),int)
    # Conservative active-tip leakage and normal-width audits.
    leakage=0.; widths=[]
    for p0,p1,segment_ids in segment_records:
        direction=p1-p0; length=float(np.linalg.norm(direction)); tangent=direction/length; normal=np.array((-tangent[1],tangent[0]))
        local_nodes=np.unique(mesh.elems[np.asarray(segment_ids,int)])
        if len(active_points):
            distance=np.min(np.linalg.norm(mesh.nodes[local_nodes,None,:]-active_points[None,:,:],axis=2),axis=1)
            local_nodes=local_nodes[distance>tolerance*max(scale,1.)]
        local_closure=np.flatnonzero(np.any(np.isin(mesh.elems,local_nodes),axis=1))
        local_ids=np.union1d(local_closure,segment_ids)
        points=mesh.nodes[mesh.elems[local_ids]].reshape(-1,2); relative=points-p0
        widths.append(float(np.max(np.abs(relative@normal))))
        leakage=max(leakage,float(max(0.,np.max(relative@tangent)-length)))
    classes=classify_graph_vertices(network)
    h=_mesh_size(mesh)
    unresolved=_unresolved_node_star_bridges(mesh,exact,candidate_nodes,selected)
    width=max(widths); width_ratio=width/max(h,1e-300); leakage_ratio=leakage/max(h,1e-300)
    reasons=[]
    if unresolved: reasons.append("INCOMPLETE_INTERIOR_NODE_STAR")
    if not exact.issubset(selected): reasons.append("MISSING_EXACT_GRAPH_SUPPORT")
    # These generous bounds reject global-domain or forward-strip fallbacks
    # while remaining independent of segment angle and endpoint phase.
    if width_ratio>4.0+1e-10: reasons.append("SUPPORT_NOT_O_H")
    if leakage_ratio>3.0+1e-10: reasons.append("ACTIVE_TIP_LEAKAGE_NOT_O_H")
    certified=not reasons
    graph_payload="|".join(f"{bid}:{idx}:{_point_key(a)}>{_point_key(b)}" for a,b,bid,idx in segments).encode()
    support_fp=_digest(np.ascontiguousarray(selected_ids,dtype=np.int64).tobytes())
    audit=GraphSupportAudit(tuple(map(int,selected_ids)),tuple(map(int,newly)),float(np.sum(mesh.area_e[selected_ids])),
      width,leakage,_digest(graph_payload),support_fp,tuple(sorted((_point_key(p),k) for p,k in classes.items())),
      graph_length,h,width_ratio,leakage_ratio,unresolved,certified,
      "CERTIFIED_COMPLETE_INTERIOR_NODE_STARS" if certified else ";".join(reasons))
    if not certified:
        raise RuntimeError(f"v12_support_not_certified: {audit.certification_reason}")
    return selected_ids,audit

def apply_mechanically_separating_graph(state,network,*,previous_support=None):
    before=element_damage(state.mesh,state.damage)
    selected,audit=mechanically_separating_graph_support(state.mesh,network,previous_support=previous_support)
    after=before.copy(); after[selected]=1.; visual=np.asarray(state.damage,float).copy(); visual[np.unique(state.mesh.elems[selected])]=1.
    mesh=replace(state.mesh,element_damage_gp=after)
    junction=dict(state.junction_process_state); junction.update({"crack_representation":MODEL_ID,"v12_graph_support_audit":audit.__dict__,
      "v12_accepted_mechanical_fingerprint":mechanical_fingerprint(state.mesh,before),"v12_trial_mechanical_fingerprint":mechanical_fingerprint(mesh,after)})
    return replace(state,mesh=mesh,damage=visual,crack_network=network,junction_process_state=junction),audit

__all__=["MODEL_ID","GraphSupportAudit","apply_mechanically_separating_graph","classify_graph_vertices","mechanically_separating_graph_support"]
