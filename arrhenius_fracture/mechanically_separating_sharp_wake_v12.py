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
class GraphSupportRecord:
    model_id: str
    mesh_geometry_fingerprint: str
    mesh_connectivity_fingerprint: str
    mesh_generation: int
    crack_graph_fingerprint: str
    accepted_damage_fingerprint: str
    selected_element_ids: tuple[int,...]

@dataclass(frozen=True)
class EdgeCutCertificate:
    segment_id: str
    h_local_max_m: float
    h_local_median_m: float
    support_width_m: float
    support_width_over_h_local: float
    forward_leakage_m: float
    forward_leakage_over_h_local: float
    selected_area_m2: float
    selected_area_over_length_h_local: float
    endpoint_footprint_error_m: float
    positive_seed_count: int
    negative_seed_count: int
    positive_component_labels: tuple[int,...]
    negative_component_labels: tuple[int,...]
    positive_component_count: int
    negative_component_count: int
    tube_boundary_clearance_m: float
    intact_cross_graph_path_exists: bool
    minimum_crossing_path_length: int | None
    bridge_node_ids: tuple[int,...]
    bridge_element_ids: tuple[int,...]
    certificate_fingerprint: str
    sufficient_opposite_side_seeds: bool

@dataclass(frozen=True)
class GraphSupportAudit:
    selected_element_ids: tuple[int,...]
    newly_selected_element_ids: tuple[int,...]
    mechanically_new_element_ids: tuple[int,...]
    mechanically_new_element_count: int
    accepted_damage_fingerprint: str
    trial_damage_fingerprint: str
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
    intact_cross_graph_path_exists: bool
    bridge_node_ids: tuple[int,...]
    bridge_element_ids: tuple[int,...]
    minimum_crossing_path_length: int | None
    positive_side_component_ids: tuple[int,...]
    negative_side_component_ids: tuple[int,...]
    certificate_fingerprint: str
    edge_cut_certificates: tuple[EdgeCutCertificate,...]
    insufficient_seed_segment_ids: tuple[str,...]
    local_h_max_m: float
    local_h_median_m: float
    selected_area_over_unique_graph_length_h_local: float
    endpoint_footprint_error_m: float
    node_star_construction_passed: bool
    independent_separation_certified: bool
    premature_mechanical_coalescence_pairs: tuple[str,...]
    certified: bool
    certification_reason: str

def _point_key(point): return f"{float(point[0]):.17g},{float(point[1]):.17g}"
def _digest(value: bytes): return hashlib.sha256(value).hexdigest()
def _array_digest(value,dtype): return _digest(np.ascontiguousarray(value,dtype=dtype).tobytes())

def classify_graph_vertices(network):
    """Return role sets; coincident graph roles are deliberately not collapsed."""
    incidence={}; roles={}
    def add(point,role): roles.setdefault(tuple(point),set()).add(role)
    for branch in network.branches:
        if branch.parent_branch_id is None: add(branch.root,"physical_root")
        if branch.status=="merged": add(branch.tip,"merged_vertex")
        if branch.branch_id in network.active_tip_ids: add(branch.tip,"active_tip")
        for a,b in zip(branch.path,branch.path[1:]):
            incidence[tuple(a)]=incidence.get(tuple(a),0)+1; incidence[tuple(b)]=incidence.get(tuple(b),0)+1
    for point,degree in incidence.items():
        if degree>=3: add(point,"branch_junction")
        elif degree==2: add(point,"degree_two_interior")
        elif "active_tip" not in roles.get(point,set()): add(point,"inactive_terminal")
    return {point:frozenset(value) for point,value in roles.items()}

def _segments(network):
    return tuple(sorted(((tuple(a),tuple(b),branch.branch_id,index) for branch in network.branches
      for index,(a,b) in enumerate(zip(branch.path,branch.path[1:]))),key=lambda v:(v[2],v[3])))

def graph_fingerprint(network):
    payload="|".join(f"{bid}:{idx}:{_point_key(a)}>{_point_key(b)}" for a,b,bid,idx in _segments(network)).encode()
    return _digest(payload)

def unique_graph_length(network,tolerance=1e-12):
    """Geometric union length, including partially overlapping collinear edges."""
    groups={}
    for a,b,_,_ in _segments(network):
        a=np.asarray(a,float); b=np.asarray(b,float); vector=b-a; length=float(np.linalg.norm(vector)); direction=vector/length
        if direction[0]<-tolerance or (abs(direction[0])<=tolerance and direction[1]<0): direction=-direction
        normal=np.array((-direction[1],direction[0])); offset=float(a@normal)
        key=(round(float(direction[0])/tolerance),round(float(direction[1])/tolerance),round(offset/tolerance))
        lo,hi=sorted((float(a@direction),float(b@direction))); groups.setdefault(key,[]).append((lo,hi))
    total=0.
    for intervals in groups.values():
        intervals.sort(); lo,hi=intervals[0]
        for left,right in intervals[1:]:
            if left<=hi+tolerance: hi=max(hi,right)
            else: total+=hi-lo; lo,hi=left,right
        total+=hi-lo
    return total

def support_record(mesh,network,damage_gp,selected):
    return GraphSupportRecord(MODEL_ID,_array_digest(mesh.nodes,np.float64),_array_digest(mesh.elems,np.int64),
      int(getattr(mesh,"geometry_generation",getattr(mesh,"generation",0))),graph_fingerprint(network),
      _array_digest(damage_gp,np.float64),tuple(map(int,np.asarray(selected,int))))

def _validated_previous(mesh,record,accepted_network,accepted_damage):
    if record is None: return set()
    if isinstance(record,dict): record=GraphSupportRecord(**record)
    if not isinstance(record,GraphSupportRecord): raise TypeError("previous_support must be a GraphSupportRecord")
    expected=support_record(mesh,accepted_network,accepted_damage,record.selected_element_ids)
    for field in ("model_id","mesh_geometry_fingerprint","mesh_connectivity_fingerprint","mesh_generation","crack_graph_fingerprint","accepted_damage_fingerprint"):
        if getattr(record,field)!=getattr(expected,field): raise RuntimeError(f"v12_support_not_certified: STALE_SUPPORT_{field.upper()}")
    ids=np.asarray(record.selected_element_ids,int)
    if np.any(ids<0) or np.any(ids>=mesh.ne): raise RuntimeError("v12_support_not_certified: SUPPORT_ELEMENT_ID_OUT_OF_RANGE")
    if np.any(np.asarray(accepted_damage)[ids]<1.): raise RuntimeError("v12_support_not_certified: SUPPORT_DAMAGE_MISMATCH")
    return set(map(int,ids))

def _mesh_size(mesh):
    triangles=np.asarray(mesh.nodes,float)[np.asarray(mesh.elems,int)]
    edges=np.concatenate((triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,1],triangles[:,0]-triangles[:,2]))
    return float(np.max(np.linalg.norm(edges,axis=1)))

def _element_diameters(mesh,element_ids):
    tri=np.asarray(mesh.nodes,float)[np.asarray(mesh.elems,int)[np.asarray(element_ids,int)]]
    return np.max(np.stack((np.linalg.norm(tri[:,1]-tri[:,0],axis=1),np.linalg.norm(tri[:,2]-tri[:,1],axis=1),np.linalg.norm(tri[:,0]-tri[:,2],axis=1))),axis=0)

def _point_segment_distance(points,p0,p1):
    vector=p1-p0; fraction=np.clip(((points-p0)@vector)/max(float(vector@vector),1e-300),0.,1.)
    return np.linalg.norm(points-(p0+fraction[:,None]*vector),axis=1)

def _path(adjacency,starts,targets):
    parent={int(n):None for n in starts}; queue=list(parent); target_set=set(map(int,targets)); hit=None
    for node in queue:
        if node in target_set: hit=node; break
        for nxt in adjacency.get(node,()):
            if nxt not in parent: parent[nxt]=node; queue.append(nxt)
    if hit is None: return ()
    result=[]
    while hit is not None: result.append(hit); hit=parent[hit]
    return tuple(reversed(result))

def independent_intact_path_certificate(mesh,network,selected,*,tolerance=1e-12):
    """Search the remaining intact element graph for an opposite-side path.

    This verifier is intentionally independent of the node-star construction:
    it only consumes the final selected element IDs, geometry, and crack graph.
    """
    selected=set(map(int,np.asarray(selected,int))); cent=np.mean(mesh.nodes[mesh.elems],axis=1)
    paths=[]; positive_components=set(); negative_components=set(); edge_certificates=[]; insufficient=[]
    for p0,p1,branch_id,index in _segments(network):
        p0=np.asarray(p0,float); p1=np.asarray(p1,float); vector=p1-p0; length=float(np.linalg.norm(vector)); tangent=vector/length; normal=np.array((-tangent[1],tangent[0]))
        exact,_=causal_segment_support(mesh,p0,p1,tolerance=tolerance); h=float(np.max(_element_diameters(mesh,exact)))
        h_median=float(np.median(_element_diameters(mesh,exact)))
        margin=min(2*h,.2*length); rel=cent-p0; axial=rel@tangent; signed=rel@normal
        local=np.flatnonzero((axial>=margin)&(axial<=length-margin)&(np.abs(signed)<=4*h)&~np.isin(np.arange(mesh.ne),tuple(selected)))
        by_node={}
        for eid in local:
            for node in mesh.elems[eid]: by_node.setdefault(int(node),[]).append(int(eid))
        adjacency={int(e):set() for e in local}
        for incident in by_node.values():
            for eid in incident: adjacency[eid].update(other for other in incident if other!=eid)
        positive=[int(e) for e in local if signed[e]>=1.5*h]; negative=[int(e) for e in local if signed[e]<=-1.5*h]
        labels={}; label=0
        for eid in local:
            if int(eid) in labels: continue
            label+=1; labels[int(eid)]=label; queue=[int(eid)]
            for current in queue:
                for nxt in adjacency[current]:
                    if nxt not in labels: labels[nxt]=label; queue.append(nxt)
        positive_labels=tuple(sorted({labels[e] for e in positive})); negative_labels=tuple(sorted({labels[e] for e in negative}))
        sufficient=bool(positive and negative and length>2*h)
        segment_id=f"{branch_id}:{index}"
        if not sufficient: insufficient.append(segment_id)
        found=_path(adjacency,positive,negative)
        if found:
            node_path=[]
            for a,b in zip(found,found[1:]):
                shared=np.intersect1d(mesh.elems[a],mesh.elems[b]); node_path.append(int(shared[0]) if len(shared) else -1)
            paths.append((branch_id,index,found,tuple(node_path)))
        positive_components.update(positive); negative_components.update(negative)
        selected_array=np.asarray(sorted(selected),int); selected_points=mesh.nodes[mesh.elems[selected_array]].reshape(-1,2); selected_rel=selected_points-p0
        selected_axial=selected_rel@tangent; selected_normal=selected_rel@normal
        local_selected=selected_array[np.min(_point_segment_distance(mesh.nodes[mesh.elems[selected_array]].reshape(-1,2),p0,p1).reshape(len(selected_array),3),axis=1)<=2.5*h]
        support_width=float(np.max(np.abs(selected_normal))); forward_leakage=float(max(0.,np.max(selected_axial)-length))
        selected_area=float(np.sum(mesh.area_e[local_selected]))
        clearance=min(max((signed[e] for e in positive),default=-math.inf)-1.5*h,
          abs(min((signed[e] for e in negative),default=math.inf))-1.5*h,
          max(length-2*margin,0.))
        local_payload=(f"{_array_digest(mesh.nodes,np.float64)}|{_array_digest(mesh.elems,np.int64)}|{graph_fingerprint(network)}|{segment_id}|"
          f"{h:.17g}|{margin:.17g}|1.5|4|{','.join(map(str,sorted(selected)))}|{positive_labels}|{negative_labels}").encode()
        edge_certificates.append(EdgeCutCertificate(segment_id,h,h_median,support_width,support_width/h,forward_leakage,forward_leakage/h,
          selected_area,selected_area/max(length*h,1e-300),forward_leakage,len(positive),len(negative),
          positive_labels,negative_labels,len(positive_labels),len(negative_labels),float(clearance),bool(found),len(found) if found else None,
          tuple(node_path) if found else (),tuple(found),_digest(local_payload),sufficient))
    bridge_elements=tuple(sorted({e for _,_,path,_ in paths for e in path})); bridge_nodes=tuple(sorted({n for _,_,_,path in paths for n in path if n>=0}))
    payload=("|".join(c.certificate_fingerprint for c in edge_certificates)).encode()
    return {"intact_cross_graph_path_exists":bool(paths),"minimum_crossing_path_length":min((len(p) for _,_,p,_ in paths),default=None),
      "bridge_node_ids":bridge_nodes,"bridge_element_ids":bridge_elements,"positive_side_component_ids":tuple(sorted(positive_components)),
      "negative_side_component_ids":tuple(sorted(negative_components)),"certificate_fingerprint":_digest(payload),
      "edge_cut_certificates":tuple(edge_certificates),"insufficient_seed_segment_ids":tuple(insufficient)}

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

def mechanically_separating_graph_support(mesh,network,active_tip_ids=None,previous_support=None,*,accepted_network=None,accepted_damage=None,tolerance=1e-12,allow_offgrid_active_tips_for_screen=False):
    """Build deterministic monotone O(h) support from the complete crack graph."""
    active_ids=tuple(sorted(network.active_tip_ids if active_tip_ids is None else active_tip_ids))
    if active_ids!=tuple(sorted(network.active_tip_ids)): raise ValueError("active_tip_ids must match accepted crack network")
    segments=_segments(network)
    if not segments: raise RuntimeError("v12_support_not_certified: crack graph has no edges")
    exact=set(); segment_records=[]
    for a,b,branch_id,index in segments:
        p0=np.asarray(a,float); p1=np.asarray(b,float); ids,lengths=causal_segment_support(mesh,p0,p1,tolerance=tolerance)
        if not len(ids): raise RuntimeError(f"v12_support_not_certified: unresolved edge {branch_id}:{index}")
        exact.update(map(int,ids))
        segment_records.append((p0,p1,ids))
    graph_length=unique_graph_length(network,tolerance)
    # Connectivity closure: every node owned by an exactly intersected element
    # is closed unless it lies at a currently active graph tip. This is defined
    # for aligned and nonaligned meshes and cannot silently reduce to V11.
    classes=classify_graph_vertices(network)
    exempt_points=[point for point,roles in classes.items() if roles==frozenset(("active_tip",))]
    active_points=np.asarray(exempt_points,float)
    if not allow_offgrid_active_tips_for_screen:
        for point in exempt_points:
            if not np.any(np.linalg.norm(np.asarray(mesh.nodes)-np.asarray(point),axis=1)<=tolerance):
                raise RuntimeError("v12_support_not_certified: ACTIVE_TIP_NOT_MESH_VERTEX")
    candidate_nodes=np.unique(mesh.elems[np.asarray(sorted(exact),int)])
    scale=max(float(np.sqrt(np.max(mesh.area_e))),1e-300)
    if len(active_points):
        distance=np.min(np.linalg.norm(mesh.nodes[candidate_nodes,None,:]-active_points[None,:,:],axis=2),axis=1)
        candidate_nodes=candidate_nodes[distance>tolerance*max(scale,1.)]
    closure=np.flatnonzero(np.any(np.isin(mesh.elems,candidate_nodes),axis=1))
    selected=set(map(int,closure))|exact
    if previous_support is not None and (accepted_network is None or accepted_damage is None):
        raise ValueError("accepted_network and accepted_damage are required with previous_support")
    previous=_validated_previous(mesh,previous_support,accepted_network,accepted_damage)
    selected|=previous
    selected_ids=np.asarray(sorted(selected),int)
    newly=np.asarray(sorted(selected-previous),int)
    accepted_field=np.zeros(mesh.ne) if accepted_damage is None else np.asarray(accepted_damage,float)
    mechanically_new=selected_ids[accepted_field[selected_ids]<1.-1e-12]
    trial_field=accepted_field.copy(); trial_field[selected_ids]=1.
    # Conservative active-tip leakage and normal-width audits.
    leakage=0.; widths=[]; local_supports=[]
    for segment_number,(p0,p1,segment_ids) in enumerate(segment_records):
        direction=p1-p0; length=float(np.linalg.norm(direction)); tangent=direction/length; normal=np.array((-tangent[1],tangent[0]))
        local_nodes=np.unique(mesh.elems[np.asarray(segment_ids,int)])
        if len(active_points):
            distance=np.min(np.linalg.norm(mesh.nodes[local_nodes,None,:]-active_points[None,:,:],axis=2),axis=1)
            local_nodes=local_nodes[distance>tolerance*max(scale,1.)]
        local_closure=np.flatnonzero(np.any(np.isin(mesh.elems,local_nodes),axis=1))
        local_ids=np.union1d(local_closure,segment_ids)
        local_supports.append((segment_number,p0,p1,set(map(int,local_ids))))
        points=mesh.nodes[mesh.elems[local_ids]].reshape(-1,2); relative=points-p0
        widths.append(float(np.max(np.abs(relative@normal))))
        leakage=max(leakage,float(max(0.,np.max(relative@tangent)-length)))
    local_h=_element_diameters(mesh,np.asarray(sorted(exact),int)); h=float(np.max(local_h)); h_median=float(np.median(local_h))
    selected_centroids=np.mean(mesh.nodes[mesh.elems[selected_ids]],axis=1)
    graph_distance=np.min(np.stack([_point_segment_distance(selected_centroids,p0,p1) for p0,p1,_ in segment_records]),axis=0)
    maximum_graph_distance=float(np.max(graph_distance))
    unresolved=_unresolved_node_star_bridges(mesh,exact,candidate_nodes,selected)
    width=max(widths); width_ratio=width/max(h,1e-300); leakage_ratio=leakage/max(h,1e-300)
    certificate=independent_intact_path_certificate(mesh,network,selected_ids,tolerance=tolerance)
    premature=[]; centroids=np.mean(mesh.nodes[mesh.elems],axis=1)
    for i,p0,p1,left in local_supports:
        for j,q0,q1,right in local_supports[i+1:]:
            overlap=left&right
            if not overlap: continue
            common=[p for p in (p0,p1) for q in (q0,q1) if np.linalg.norm(p-q)<=tolerance]
            if common:
                junction=np.asarray(common[0]); overlap={eid for eid in overlap if np.linalg.norm(centroids[eid]-junction)>2*h}
            if overlap: premature.append(f"{i}:{j}")
    endpoint_error=leakage
    reasons=[]
    if unresolved: reasons.append("INCOMPLETE_INTERIOR_NODE_STAR")
    if not exact.issubset(selected): reasons.append("MISSING_EXACT_GRAPH_SUPPORT")
    # These generous bounds reject global-domain or forward-strip fallbacks
    # while remaining independent of segment angle and endpoint phase.
    if width_ratio>4.0+1e-10: reasons.append("SUPPORT_NOT_O_H")
    if leakage_ratio>3.0+1e-10: reasons.append("ACTIVE_TIP_LEAKAGE_NOT_O_H")
    if maximum_graph_distance>4*h+1e-10: reasons.append("RETAINED_SUPPORT_NOT_LOCAL")
    if certificate["intact_cross_graph_path_exists"]: reasons.append("INTACT_CROSS_GRAPH_PATH")
    if certificate["insufficient_seed_segment_ids"]: reasons.append("INSUFFICIENT_OPPOSITE_SIDE_SEEDS")
    if premature: reasons.append("PREMATURE_MECHANICAL_COALESCENCE")
    if previous and graph_fingerprint(network)!=graph_fingerprint(accepted_network) and not mechanically_new.size:
        reasons.append("EVENT_BELOW_CERTIFIED_MESH_RESOLUTION")
    certified=not reasons
    graph_payload="|".join(f"{bid}:{idx}:{_point_key(a)}>{_point_key(b)}" for a,b,bid,idx in segments).encode()
    support_fp=_digest(np.ascontiguousarray(selected_ids,dtype=np.int64).tobytes())
    audit=GraphSupportAudit(tuple(map(int,selected_ids)),tuple(map(int,newly)),tuple(map(int,mechanically_new)),len(mechanically_new),
      _array_digest(accepted_field,np.float64),_array_digest(trial_field,np.float64),float(np.sum(mesh.area_e[selected_ids])),
      width,leakage,_digest(graph_payload),support_fp,tuple(sorted((_point_key(p),"+".join(sorted(k))) for p,k in classes.items())),
      graph_length,h,width_ratio,leakage_ratio,unresolved,
      certificate["intact_cross_graph_path_exists"],certificate["bridge_node_ids"],certificate["bridge_element_ids"],certificate["minimum_crossing_path_length"],
      certificate["positive_side_component_ids"],certificate["negative_side_component_ids"],certificate["certificate_fingerprint"],
      certificate["edge_cut_certificates"],certificate["insufficient_seed_segment_ids"],
      h,h_median,float(np.sum(mesh.area_e[selected_ids])/max(graph_length*h,1e-300)),endpoint_error,
      not unresolved,not certificate["intact_cross_graph_path_exists"],tuple(premature),certified,
      "CERTIFIED_INDEPENDENT_INTACT_CUT" if certified else ";".join(reasons))
    if not certified:
        raise RuntimeError(f"v12_support_not_certified: {audit.certification_reason}")
    return selected_ids,audit

def apply_mechanically_separating_graph(state,network,*,previous_support=None):
    before=element_damage(state.mesh,state.damage)
    owned=previous_support
    if owned is None: owned=state.junction_process_state.get("v12_support_record")
    selected,audit=mechanically_separating_graph_support(state.mesh,network,previous_support=owned,
      accepted_network=state.crack_network,accepted_damage=before)
    newly=selected[before[selected]<1.]; audit=replace(audit,newly_selected_element_ids=tuple(map(int,newly)))
    after=before.copy(); after[selected]=1.; visual=np.asarray(state.damage,float).copy(); visual[np.unique(state.mesh.elems[selected])]=1.
    mesh=replace(state.mesh,element_damage_gp=after)
    junction=dict(state.junction_process_state); junction.update({"crack_representation":MODEL_ID,"v12_graph_support_audit":audit.__dict__,
      "v12_support_record":support_record(mesh,network,after,selected).__dict__,
      "v12_accepted_mechanical_fingerprint":mechanical_fingerprint(state.mesh,before),"v12_trial_mechanical_fingerprint":mechanical_fingerprint(mesh,after)})
    return replace(state,mesh=mesh,damage=visual,crack_network=network,junction_process_state=junction),audit

__all__=["MODEL_ID","EdgeCutCertificate","GraphSupportAudit","GraphSupportRecord","apply_mechanically_separating_graph","classify_graph_vertices",
  "graph_fingerprint","independent_intact_path_certificate","mechanically_separating_graph_support","support_record","unique_graph_length"]
