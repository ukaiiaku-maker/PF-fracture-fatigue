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
    signed_tip_footprint_m: float
    forward_overshoot_m: float
    backward_undershoot_m: float
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
class JunctionSectorCertificate:
    junction_id: str
    vertex_roles: tuple[str,...]
    ordered_ray_angles: tuple[float,...]
    sector_ids: tuple[int,...]
    sector_seed_counts: tuple[int,...]
    sector_component_labels: tuple[tuple[int,...],...]
    within_sector_connected: bool
    cross_arm_path_exists: bool
    legal_support_overlap: bool
    junction_certificate_status: str
    junction_certificate_fingerprint: str

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
    positive_seed_element_ids: tuple[int,...]
    negative_seed_element_ids: tuple[int,...]
    certificate_fingerprint: str
    edge_cut_certificates: tuple[EdgeCutCertificate,...]
    insufficient_seed_segment_ids: tuple[str,...]
    local_h_max_m: float
    local_h_median_m: float
    selected_area_over_unique_graph_length_h_local: float
    endpoint_footprint_error_m: float
    active_tip_support_axial_extent_m: float
    active_tip_signed_footprint_m: float
    active_tip_backward_undershoot_m: float
    h_tip_max_m: float
    h_tip_median_m: float
    h_tip_tangent_m: float
    h_tip_normal_m: float
    overshoot_over_h_tip: float
    undershoot_over_h_tip: float
    support_component_ids_edge: tuple[int,...]
    support_component_ids_node: tuple[int,...]
    graph_component_count: int
    graph_to_node_support_component_incidence: tuple[tuple[int,tuple[int,...]],...]
    node_support_to_graph_component_incidence: tuple[tuple[int,tuple[int,...]],...]
    component_incidence_one_to_one: bool
    illegal_support_connection: bool
    minimum_support_component_separation_m: float
    nonadjacent_arc_support_short_circuit_pairs: tuple[str,...]
    junction_sector_certificates: tuple[JunctionSectorCertificate,...]
    node_star_construction_passed: bool
    independent_separation_certified: bool
    premature_mechanical_coalescence_pairs: tuple[str,...]
    certified: bool
    certification_reason: str

def _point_key(point): return f"{float(point[0]):.17g},{float(point[1]):.17g}"
def _tolerance_point_key(point,tolerance): return f"{round(float(point[0])/tolerance)},{round(float(point[1])/tolerance)}"
def _tolerance_scalar_key(value,tolerance): return str(round(float(value)/tolerance))
def _digest(value: bytes): return hashlib.sha256(value).hexdigest()
def _array_digest(value,dtype): return _digest(np.ascontiguousarray(value,dtype=dtype).tobytes())

def classify_graph_vertices(network):
    """Return role sets; coincident graph roles are deliberately not collapsed."""
    incidence={}; directions={}; roles={}
    def add(point,role): roles.setdefault(tuple(point),set()).add(role)
    for branch in network.branches:
        if branch.parent_branch_id is None: add(branch.root,"physical_root")
        if branch.status=="merged": add(branch.tip,"merged_vertex")
        if branch.branch_id in network.active_tip_ids: add(branch.tip,"active_tip")
        for a,b in zip(branch.path,branch.path[1:]):
            incidence[tuple(a)]=incidence.get(tuple(a),0)+1; incidence[tuple(b)]=incidence.get(tuple(b),0)+1
            vector=np.asarray(b,float)-np.asarray(a,float); vector=vector/np.linalg.norm(vector)
            directions.setdefault(tuple(a),[]).append(vector); directions.setdefault(tuple(b),[]).append(-vector)
    for point,degree in incidence.items():
        if degree>=3: add(point,"branch_junction")
        elif degree==2:
            rays=directions[point]; collinear=abs(_cross(rays[0],rays[1]))<=1e-12
            add(point,"degree_two_interior" if collinear else "kink_vertex")
        elif "active_tip" not in roles.get(point,set()): add(point,"inactive_terminal")
    return {point:frozenset(value) for point,value in roles.items()}

def _segments(network):
    return tuple(sorted(((tuple(a),tuple(b),branch.branch_id,index) for branch in network.branches
      for index,(a,b) in enumerate(zip(branch.path,branch.path[1:]))),key=lambda v:(v[2],v[3])))

def certification_arcs(network,tolerance=1e-12):
    """Collapse collinear degree-two history vertices into mechanical arcs."""
    roles=classify_graph_vertices(network); arcs=[]
    for branch in sorted(network.branches,key=lambda item:item.branch_id):
        path=[np.asarray(point,float) for point in branch.path]
        if len(path)<2: continue
        start=path[0]; direction=(path[1]-path[0])/np.linalg.norm(path[1]-path[0]); arc_index=0
        for index in range(1,len(path)):
            end=path[index]; is_last=index==len(path)-1
            if not is_last:
                following=(path[index+1]-end)/np.linalg.norm(path[index+1]-end)
                collinear=abs(direction[0]*following[1]-direction[1]*following[0])<=tolerance and direction@following>0
                merge=roles.get(tuple(end),frozenset())==frozenset(("degree_two_interior",)) and collinear
                if merge: continue
            arcs.append((tuple(start),tuple(end),f"{branch.branch_id}:arc{arc_index}")); arc_index+=1
            if not is_last:
                start=end; direction=(path[index+1]-end)/np.linalg.norm(path[index+1]-end)
    return tuple(arcs)

def certification_arc_fingerprint(network,tolerance=1e-12):
    payload="|".join(f"{arc_id}:{_tolerance_point_key(a,tolerance)}>{_tolerance_point_key(b,tolerance)}"
      for a,b,arc_id in certification_arcs(network,tolerance)).encode()
    return _digest(payload)

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

def _validate_unambiguous_graph_edges(network,tolerance=1e-12):
    segments=_segments(network)
    for index,(a,b,branch_id,_) in enumerate(segments):
        a=np.asarray(a,float); b=np.asarray(b,float); vector=b-a; length=float(np.linalg.norm(vector)); tangent=vector/length
        for c,d,other_branch,_ in segments[index+1:]:
            c=np.asarray(c,float); d=np.asarray(d,float)
            if abs(_cross(tangent,d-c))>tolerance or abs(_cross(tangent,c-a))>tolerance: continue
            lo,hi=sorted((float((c-a)@tangent),float((d-a)@tangent))); overlap=min(length,hi)-max(0.,lo)
            if overlap>tolerance: raise RuntimeError("v12_support_not_certified: DUPLICATE_OR_OVERLAPPING_GRAPH_EDGE")
            if branch_id!=other_branch and (np.linalg.norm(b-c)<=tolerance or np.linalg.norm(a-d)<=tolerance):
                raise RuntimeError("v12_support_not_certified: COLLINEAR_PARENT_CHILD_CONTINUATION_REQUIRES_CANONICAL_BRANCH_SEMANTICS")

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

def _cross(a,b): return float(a[0]*b[1]-a[1]*b[0])

def _segments_intersect(a,b,c,d,tolerance=1e-12):
    ab=b-a; cd=d-c; denominator=_cross(ab,cd)
    if abs(denominator)<=tolerance:
        if abs(_cross(c-a,ab))>tolerance: return False
        scale=max(float(ab@ab),1e-300); lo,hi=sorted((float((c-a)@ab/scale),float((d-a)@ab/scale)))
        return hi>=-tolerance and lo<=1+tolerance
    t=_cross(c-a,cd)/denominator; u=_cross(c-a,ab)/denominator
    return -tolerance<=t<=1+tolerance and -tolerance<=u<=1+tolerance

def _segment_distance(a,b,c,d,tolerance=1e-12):
    if _segments_intersect(a,b,c,d,tolerance): return 0.
    return float(min(_point_segment_distance(a[None,:],c,d)[0],_point_segment_distance(b[None,:],c,d)[0],
      _point_segment_distance(c[None,:],a,b)[0],_point_segment_distance(d[None,:],a,b)[0]))

def _boundary_tube_clearance(mesh,boundary_edges,p0,tangent,normal,margin,length,half_width,tolerance=1e-12):
    corners=[p0+margin*tangent-half_width*normal,p0+(length-margin)*tangent-half_width*normal,
      p0+(length-margin)*tangent+half_width*normal,p0+margin*tangent+half_width*normal]
    sides=tuple(zip(corners,corners[1:]+corners[:1])); clearance=math.inf
    for node_a,node_b in boundary_edges:
        a=np.asarray(mesh.nodes[node_a],float); b=np.asarray(mesh.nodes[node_b],float)
        rel_a=a-p0; rel_b=b-p0
        inside=lambda rel: margin-tolerance<=rel@tangent<=length-margin+tolerance and abs(rel@normal)<=half_width+tolerance
        if inside(rel_a) or inside(rel_b) or any(_segments_intersect(a,b,c,d,tolerance) for c,d in sides): return 0.
        clearance=min(clearance,*( _segment_distance(a,b,c,d,tolerance) for c,d in sides))
    return float(clearance)

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

def _external_boundary_edges(mesh):
    owners={}
    for eid,element in enumerate(mesh.elems):
        for a,b in ((element[0],element[1]),(element[1],element[2]),(element[2],element[0])):
            owners.setdefault(tuple(sorted((int(a),int(b)))),[]).append(eid)
    return np.asarray([edge for edge,value in owners.items() if len(value)==1],int)

def selected_support_components(mesh,selected,*,shared_nodes):
    """Label selected elements under shared-edge or shared-P1-node adjacency."""
    ids=tuple(sorted(map(int,np.asarray(selected,int)))); adjacency={eid:set() for eid in ids}; owners={}
    for eid in ids:
        element=tuple(map(int,mesh.elems[eid]))
        keys=element if shared_nodes==1 else tuple(tuple(sorted(edge)) for edge in ((element[0],element[1]),(element[1],element[2]),(element[2],element[0])))
        for key in keys: owners.setdefault(key,[]).append(eid)
    for incident in owners.values():
        for eid in incident: adjacency[eid].update(other for other in incident if other!=eid)
    labels={}; label=0
    for eid in ids:
        if eid in labels: continue
        label+=1; labels[eid]=label; queue=[eid]
        for current in queue:
            for nxt in adjacency[current]:
                if nxt not in labels: labels[nxt]=label; queue.append(nxt)
    return tuple(labels[eid] for eid in ids)

def graph_component_membership(network,tolerance=1e-12):
    segments=_segments(network); parent=list(range(len(segments)))
    def find(value):
        while parent[value]!=value: parent[value]=parent[parent[value]]; value=parent[value]
        return value
    def union(a,b):
        a=find(a); b=find(b)
        if a!=b: parent[b]=a
    for i,(a,b,_,_) in enumerate(segments):
        for j,(c,d,_,_) in enumerate(segments[:i]):
            if any(np.linalg.norm(np.asarray(p)-np.asarray(q))<=tolerance for p in (a,b) for q in (c,d)): union(i,j)
    roots=sorted({find(i) for i in range(len(segments))}); canonical={root:index+1 for index,root in enumerate(roots)}
    return tuple(canonical[find(i)] for i in range(len(segments)))

def graph_component_count(network,tolerance=1e-12):
    return len(set(graph_component_membership(network,tolerance)))

def _minimum_component_separation(mesh,selected,labels):
    ids=np.asarray(sorted(map(int,np.asarray(selected,int))),int); unique=sorted(set(labels))
    if len(unique)<2: return math.inf
    nodes={label:np.unique(mesh.elems[ids[np.asarray(labels)==label]]) for label in unique}; result=math.inf
    for index,left in enumerate(unique):
        for right in unique[index+1:]:
            a=mesh.nodes[nodes[left]]; b=mesh.nodes[nodes[right]]
            result=min(result,float(np.min(np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2))))
    return result

def junction_sector_certificates(mesh,network,selected,tolerance=1e-12):
    """Certify intact material sectors in an annulus around graph vertices."""
    roles=classify_graph_vertices(network); selected_set=set(map(int,np.asarray(selected,int))); incident={}
    for a,b,_,_ in _segments(network):
        a=np.asarray(a,float); b=np.asarray(b,float); vector=(b-a)/np.linalg.norm(b-a)
        incident.setdefault(tuple(a),[]).append(vector); incident.setdefault(tuple(b),[]).append(-vector)
    results=[]; centroids=np.mean(mesh.nodes[mesh.elems],axis=1)
    for point,rays in sorted(incident.items(),key=lambda item:_point_key(item[0])):
        unique=[]
        for ray in rays:
            if not any(np.linalg.norm(ray-other)<=tolerance for other in unique): unique.append(ray)
        vertex_roles=roles.get(point,frozenset())
        if len(unique)<2 or not ({"kink_vertex","branch_junction","merged_vertex"}&set(vertex_roles) or len(vertex_roles)>1): continue
        center=np.asarray(point,float); exact=set(); incident_supports=[]
        for a,b,_,_ in _segments(network):
            if np.linalg.norm(np.asarray(a)-center)<=tolerance or np.linalg.norm(np.asarray(b)-center)<=tolerance:
                ids,_=causal_segment_support(mesh,np.asarray(a,float),np.asarray(b,float),tolerance=tolerance)
                support=set(map(int,ids)); exact.update(support)
                support_nodes=np.unique(mesh.elems[np.asarray(sorted(support),int)])
                closure=set(map(int,np.flatnonzero(np.any(np.isin(mesh.elems,support_nodes),axis=1))))
                incident_supports.append((support|closure)&selected_set)
        h=float(np.max(_element_diameters(mesh,np.asarray(sorted(exact),int))))
        rel=centroids-center; radius=np.linalg.norm(rel,axis=1); local=[int(e) for e in np.flatnonzero((radius>=.75*h)&(radius<=5.5*h)) if int(e) not in selected_set]
        by_node={}
        for eid in local:
            for node in mesh.elems[eid]: by_node.setdefault(int(node),[]).append(eid)
        adjacency={eid:set() for eid in local}
        for values in by_node.values():
            for eid in values: adjacency[eid].update(other for other in values if other!=eid)
        labels={}; label=0
        for eid in local:
            if eid in labels: continue
            label+=1; labels[eid]=label; queue=[eid]
            for current in queue:
                for nxt in adjacency[current]:
                    if nxt not in labels: labels[nxt]=label; queue.append(nxt)
        angles=tuple(sorted(float(np.mod(np.arctan2(ray[1],ray[0]),2*np.pi)) for ray in unique)); sector_labels=[]; counts=[]
        point_angles=np.mod(np.arctan2(rel[:,1],rel[:,0]),2*np.pi)
        for index,left in enumerate(angles):
            right=angles[(index+1)%len(angles)]; span=(right-left)%(2*np.pi); delta=(point_angles-left)%(2*np.pi)
            seeds=[eid for eid in local if .2*span<delta[eid]<.8*span and 3*h<=radius[eid]<=5*h]
            counts.append(len(seeds)); sector_labels.append(tuple(sorted({labels[eid] for eid in seeds})))
        within=all(count>0 and len(values)==1 for count,values in zip(counts,sector_labels))
        cross=any(set(left)&set(right) for i,left in enumerate(sector_labels) for right in sector_labels[i+1:])
        overlaps=set().union(*(left&right for i,left in enumerate(incident_supports) for right in incident_supports[i+1:]))
        legal_overlap=bool(overlaps) and all(np.linalg.norm(centroids[eid]-center)<=2*h+tolerance for eid in overlaps)
        status="ACCEPTED" if within and not cross and legal_overlap else (
          "MISSING_MATERIAL_SECTOR" if not within else ("CROSS_ARM_INTACT_PATH" if cross else "LEGAL_JUNCTION_OVERLAP_NOT_LOCAL"))
        payload=f"{_tolerance_point_key(center,tolerance)}|{angles}|{counts}|{sector_labels}|{within}|{cross}|{legal_overlap}".encode()
        results.append(JunctionSectorCertificate(_point_key(center),tuple(sorted(vertex_roles)),angles,tuple(range(len(angles))),
          tuple(counts),tuple(sector_labels),within,bool(cross),legal_overlap,status,_digest(payload)))
    return tuple(results)

def independent_intact_path_certificate(mesh,network,selected,*,edge_supports=None,arcs=None,allow_boundary_clip_for_screen=False,tolerance=1e-12):
    """Search the remaining intact element graph for an opposite-side path.

    This verifier is intentionally independent of the node-star construction:
    it only consumes the final selected element IDs, geometry, and crack graph.
    """
    selected=set(map(int,np.asarray(selected,int))); cent=np.mean(mesh.nodes[mesh.elems],axis=1)
    paths=[]; positive_components=set(); negative_components=set(); edge_certificates=[]; insufficient=[]; boundary_edges=_external_boundary_edges(mesh)
    arcs=certification_arcs(network,tolerance) if arcs is None else arcs
    for p0,p1,arc_id in arcs:
        p0=np.asarray(p0,float); p1=np.asarray(p1,float); vector=p1-p0; length=float(np.linalg.norm(vector)); tangent=vector/length; normal=np.array((-tangent[1],tangent[0]))
        exact,_=causal_segment_support(mesh,p0,p1,tolerance=tolerance); h=float(np.max(_element_diameters(mesh,exact)))
        h_median=float(np.median(_element_diameters(mesh,exact)))
        margin=min(2*h,.2*length); rel=cent-p0; axial=rel@tangent; signed=rel@normal
        triangles=mesh.nodes[mesh.elems]; projected_axial=(triangles-p0)@tangent; projected_normal=(triangles-p0)@normal
        geometric_tube=(np.max(projected_axial,axis=1)>=margin)&(np.min(projected_axial,axis=1)<=length-margin)&(np.max(projected_normal,axis=1)>=-3*h)&(np.min(projected_normal,axis=1)<=3*h)
        local=np.flatnonzero(geometric_tube&~np.isin(np.arange(mesh.ne),tuple(selected)))
        by_node={}
        for eid in local:
            for node in mesh.elems[eid]: by_node.setdefault(int(node),[]).append(int(eid))
        adjacency={int(e):set() for e in local}
        for incident in by_node.values():
            for eid in incident: adjacency[eid].update(other for other in incident if other!=eid)
        positive=[int(e) for e in local if np.min(projected_normal[e])>=h]; negative=[int(e) for e in local if np.max(projected_normal[e])<=-h]
        labels={}; label=0
        for eid in local:
            if int(eid) in labels: continue
            label+=1; labels[int(eid)]=label; queue=[int(eid)]
            for current in queue:
                for nxt in adjacency[current]:
                    if nxt not in labels: labels[nxt]=label; queue.append(nxt)
        positive_labels=tuple(sorted({labels[e] for e in positive})); negative_labels=tuple(sorted({labels[e] for e in negative}))
        exterior_clearance=_boundary_tube_clearance(mesh,boundary_edges,p0,tangent,normal,margin,length,3*h,tolerance)
        sufficient=bool(positive and negative and length>2*h and (exterior_clearance>tolerance or allow_boundary_clip_for_screen))
        segment_id=arc_id
        if not sufficient: insufficient.append(segment_id)
        found=_path(adjacency,positive,negative)
        if found:
            node_path=[]
            for a,b in zip(found,found[1:]):
                shared=np.intersect1d(mesh.elems[a],mesh.elems[b]); node_path.append(int(shared[0]) if len(shared) else -1)
            paths.append((arc_id,found,tuple(node_path)))
        positive_components.update(positive); negative_components.update(negative)
        selected_array=np.asarray(sorted(edge_supports[segment_id] if edge_supports is not None else selected),int)
        selected_points=mesh.nodes[mesh.elems[selected_array]].reshape(-1,2); selected_rel=selected_points-p0
        selected_axial=selected_rel@tangent; selected_normal=selected_rel@normal
        support_width=float(np.max(np.abs(selected_normal))); signed_footprint=float(np.max(selected_axial)-length)
        is_active_tip=classify_graph_vertices(network).get(tuple(p1),frozenset())==frozenset(("active_tip",))
        signed_footprint=signed_footprint if is_active_tip else 0.
        forward_leakage=float(max(0.,signed_footprint)); backward_undershoot=float(max(0.,-signed_footprint))
        selected_area=float(np.sum(mesh.area_e[selected_array]))
        clearance=min(max((signed[e] for e in positive),default=-math.inf)-h,
          abs(min((signed[e] for e in negative),default=math.inf))-h,
          max(length-2*margin,0.),exterior_clearance)
        local_payload=(f"{_array_digest(mesh.nodes,np.float64)}|{_array_digest(mesh.elems,np.int64)}|{certification_arc_fingerprint(network,tolerance)}|{segment_id}|"
          f"{_tolerance_scalar_key(h,tolerance)}|{_tolerance_scalar_key(margin,tolerance)}|1|3|{','.join(map(str,sorted(selected)))}|{positive_labels}|{negative_labels}").encode()
        edge_certificates.append(EdgeCutCertificate(segment_id,h,h_median,support_width,support_width/h,forward_leakage,forward_leakage/h,
          selected_area,selected_area/max(length*h,1e-300),abs(signed_footprint),signed_footprint,forward_leakage,backward_undershoot,len(positive),len(negative),
          positive_labels,negative_labels,len(positive_labels),len(negative_labels),float(clearance),bool(found),len(found) if found else None,
          tuple(node_path) if found else (),tuple(found),_digest(local_payload),sufficient))
    bridge_elements=tuple(sorted({e for _,path,_ in paths for e in path})); bridge_nodes=tuple(sorted({n for _,_,path in paths for n in path if n>=0}))
    payload=("|".join(c.certificate_fingerprint for c in edge_certificates)).encode()
    return {"intact_cross_graph_path_exists":bool(paths),"minimum_crossing_path_length":min((len(p) for _,p,_ in paths),default=None),
      "bridge_node_ids":bridge_nodes,"bridge_element_ids":bridge_elements,"positive_seed_element_ids":tuple(sorted(positive_components)),
      "negative_seed_element_ids":tuple(sorted(negative_components)),"certificate_fingerprint":_digest(payload),
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

def mechanically_separating_graph_support(mesh,network,active_tip_ids=None,previous_support=None,*,accepted_network=None,accepted_damage=None,tolerance=1e-12,
    allow_offgrid_active_tips_for_screen=False,return_uncertified_audit_for_screen=False):
    """Build deterministic monotone O(h) support from the complete crack graph."""
    active_ids=tuple(sorted(network.active_tip_ids if active_tip_ids is None else active_tip_ids))
    if active_ids!=tuple(sorted(network.active_tip_ids)): raise ValueError("active_tip_ids must match accepted crack network")
    segments=_segments(network)
    if not segments: raise RuntimeError("v12_support_not_certified: crack graph has no edges")
    _validate_unambiguous_graph_edges(network,tolerance)
    exact=set(); segment_records=[]
    for a,b,branch_id,index in segments:
        p0=np.asarray(a,float); p1=np.asarray(b,float); ids,lengths=causal_segment_support(mesh,p0,p1,tolerance=tolerance)
        if not len(ids): raise RuntimeError(f"v12_support_not_certified: unresolved edge {branch_id}:{index}")
        exact.update(map(int,ids))
        segment_records.append((p0,p1,ids))
    graph_length=unique_graph_length(network,tolerance)
    arcs=certification_arcs(network,tolerance); arc_records=[]
    for a,b,arc_id in arcs:
        p0=np.asarray(a,float); p1=np.asarray(b,float); ids,_=causal_segment_support(mesh,p0,p1,tolerance=tolerance)
        arc_records.append((p0,p1,ids,arc_id))
    # Connectivity closure: every node owned by an exactly intersected element
    # is closed unless it lies at a currently active graph tip. This is defined
    # for aligned and nonaligned meshes and cannot silently reduce to V11.
    classes=classify_graph_vertices(network)
    exempt_points=[point for point,roles in classes.items() if roles==frozenset(("active_tip",))]
    active_points=np.asarray(exempt_points,float)
    if not allow_offgrid_active_tips_for_screen:
        for point in exempt_points:
            if not np.any(np.linalg.norm(np.asarray(mesh.nodes)-np.asarray(point),axis=1)<=tolerance):
                raise RuntimeError("v12_support_not_certified: REQUIRES_ACTIVE_TIP_ALIGNMENT_REMESH")
    candidate_nodes=np.unique(mesh.elems[np.asarray(sorted(exact),int)])
    scale=max(float(np.sqrt(np.max(mesh.area_e))),1e-300)
    if len(active_points):
        distance=np.min(np.linalg.norm(mesh.nodes[candidate_nodes,None,:]-active_points[None,:,:],axis=2),axis=1)
        candidate_nodes=candidate_nodes[distance>tolerance*max(scale,1.)]
    closure=np.flatnonzero(np.any(np.isin(mesh.elems,candidate_nodes),axis=1))
    selected=set(map(int,closure))|exact
    # A turning or branching vertex needs one additional node-star ring so the
    # discrete P1 displacement graph cannot reconnect the two material sectors
    # around the corner. Straight interiors retain the narrower one-ring wake.
    vertex_classes = classify_graph_vertices(network)
    if any({"kink_vertex", "branch_junction", "merged_vertex"} & set(value)
           for value in vertex_classes.values()):
        closure_nodes = np.unique(mesh.elems[np.asarray(sorted(selected), int)])
        selected.update(map(int, np.flatnonzero(np.any(np.isin(mesh.elems, closure_nodes), axis=1))))
    if previous_support is not None and (accepted_network is None or accepted_damage is None):
        raise ValueError("accepted_network and accepted_damage are required with previous_support")
    previous=_validated_previous(mesh,previous_support,accepted_network,accepted_damage)
    selected|=previous
    selected_ids=np.asarray(sorted(selected),int)
    newly=np.asarray(sorted(selected-previous),int)
    accepted_field=np.zeros(mesh.ne) if accepted_damage is None else np.asarray(accepted_damage,float)
    mechanically_new=selected_ids[accepted_field[selected_ids]<1.-1e-12]
    trial_field=accepted_field.copy(); trial_field[selected_ids]=1.
    # Construct arc-local closures, then assign every retained selected element
    # to its nearest physical arc so all certificate metrics cover full support.
    local_supports=[]
    for segment_number,(p0,p1,segment_ids,arc_id) in enumerate(arc_records):
        direction=p1-p0; length=float(np.linalg.norm(direction)); tangent=direction/length; normal=np.array((-tangent[1],tangent[0]))
        local_nodes=np.unique(mesh.elems[np.asarray(segment_ids,int)])
        if len(active_points):
            distance=np.min(np.linalg.norm(mesh.nodes[local_nodes,None,:]-active_points[None,:,:],axis=2),axis=1)
            local_nodes=local_nodes[distance>tolerance*max(scale,1.)]
        local_closure=np.flatnonzero(np.any(np.isin(mesh.elems,local_nodes),axis=1))
        local_ids=np.union1d(local_closure,segment_ids)
        local_supports.append([segment_number,p0,p1,set(map(int,local_ids)),arc_id])
    local_h=_element_diameters(mesh,np.asarray(sorted(exact),int)); h=float(np.max(local_h)); h_median=float(np.median(local_h))
    selected_vertices=mesh.nodes[mesh.elems[selected_ids]]
    edge_distances=np.stack([_point_segment_distance(selected_vertices.reshape(-1,2),p0,p1).reshape(len(selected_ids),3)
      for p0,p1,_,_ in arc_records])
    edge_h=np.asarray([np.max(_element_diameters(mesh,ids)) for _,_,ids,_ in arc_records])
    nearest=np.argmin(edge_distances,axis=0); graph_distance=np.min(edge_distances,axis=0)
    retained_locality_ratio=graph_distance/edge_h[nearest]
    element_nearest=np.argmin(np.min(edge_distances,axis=2),axis=0)
    for eid,arc_index in zip(selected_ids,element_nearest): local_supports[int(arc_index)][3].add(int(eid))
    leakage=0.; undershoot=0.; signed_tip_footprint=0.; axial_extent=0.; widths=[]; tip_h=[]; tip_tangent=[]; tip_normal=[]
    for _,p0,p1,local_ids,_ in local_supports:
        direction=p1-p0; length=float(np.linalg.norm(direction)); tangent=direction/length; normal=np.array((-tangent[1],tangent[0]))
        local_array=np.asarray(sorted(local_ids),int); points=mesh.nodes[mesh.elems[local_array]].reshape(-1,2); relative=points-p0
        widths.append(float(np.max(np.abs(relative@normal))))
        if classes.get(tuple(p1),frozenset())==frozenset(("active_tip",)):
            extent=float(np.max(relative@tangent)); signed=extent-length
            if abs(signed)>abs(signed_tip_footprint): signed_tip_footprint=signed
            axial_extent=max(axial_extent,extent); leakage=max(leakage,max(0.,signed)); undershoot=max(undershoot,max(0.,-signed))
            at_tip=np.flatnonzero(np.any(np.linalg.norm(mesh.nodes[mesh.elems]-p1,axis=2)<=tolerance,axis=1))
            if len(at_tip):
                tip_h.extend(_element_diameters(mesh,at_tip)); tip_points=mesh.nodes[mesh.elems[at_tip]]-p1
                tip_tangent.append(float(np.max(tip_points@tangent)-np.min(tip_points@tangent)))
                tip_normal.append(float(np.max(tip_points@normal)-np.min(tip_points@normal)))
    unresolved=_unresolved_node_star_bridges(mesh,exact,candidate_nodes,selected)
    width=max(widths); width_ratio=width/max(h,1e-300); leakage_ratio=leakage/max(h,1e-300)
    edge_support_map={arc_id:tuple(sorted(values)) for _,_,_,values,arc_id in local_supports}
    explicit_free_surface_root = any(
        branch.local_state.get("source") == "direct_cavity_boundary_tensor"
        for branch in network.branches
    )
    certificate=independent_intact_path_certificate(mesh,network,selected_ids,edge_supports=edge_support_map,
      arcs=arcs,allow_boundary_clip_for_screen=(allow_offgrid_active_tips_for_screen or explicit_free_surface_root),tolerance=tolerance)
    premature=[]; centroids=np.mean(mesh.nodes[mesh.elems],axis=1)
    for i,p0,p1,left,_ in local_supports:
        for j,q0,q1,right,_ in local_supports[i+1:]:
            overlap=left&right
            if not overlap: continue
            common=[p for p in (p0,p1) for q in (q0,q1) if np.linalg.norm(p-q)<=tolerance]
            if common:
                junction=np.asarray(common[0]); overlap={eid for eid in overlap if np.linalg.norm(centroids[eid]-junction)>2*h}
            if overlap: premature.append(f"{i}:{j}")
    endpoint_error=max(leakage,undershoot)
    h_tip_max=float(np.max(tip_h)) if tip_h else math.nan; h_tip_median=float(np.median(tip_h)) if tip_h else math.nan
    h_tip_tangent=float(np.max(tip_tangent)) if tip_tangent else math.nan; h_tip_normal=float(np.max(tip_normal)) if tip_normal else math.nan
    component_edge=selected_support_components(mesh,selected_ids,shared_nodes=2)
    component_node=selected_support_components(mesh,selected_ids,shared_nodes=1)
    segment_component_labels=graph_component_membership(network,tolerance); physical_component_count=len(set(segment_component_labels))
    branch_component={branch_id:label for (_,_,branch_id,_),label in zip(segments,segment_component_labels)}
    arc_component=[branch_component[arc_id.split(":arc",1)[0]] for *_,arc_id in arc_records]
    element_node_label={int(eid):int(label) for eid,label in zip(selected_ids,component_node)}
    graph_to_support={label:set() for label in set(segment_component_labels)}; support_to_graph={label:set() for label in set(component_node)}
    for arc_index,(_,_,_,local_ids,_) in enumerate(local_supports):
        graph_label=arc_component[arc_index]
        for eid in local_ids:
            support_label=element_node_label[eid]; graph_to_support[graph_label].add(support_label); support_to_graph[support_label].add(graph_label)
    graph_incidence=tuple((label,tuple(sorted(values))) for label,values in sorted(graph_to_support.items()))
    support_incidence=tuple((label,tuple(sorted(values))) for label,values in sorted(support_to_graph.items()))
    incidence_one_to_one=all(len(values)==1 for _,values in graph_incidence) and all(len(values)==1 for _,values in support_incidence)
    illegal_connection=not incidence_one_to_one
    component_separation=_minimum_component_separation(mesh,selected_ids,component_node)
    short_circuits=[]
    for left_index,(_,p0,p1,left,_) in enumerate(local_supports):
        left_nodes=set(map(int,np.unique(mesh.elems[np.asarray(sorted(left),int)])))
        for right_index,(_,q0,q1,right,_) in enumerate(local_supports[left_index+1:],left_index+1):
            if arc_component[left_index]!=arc_component[right_index]: continue
            adjacent=any(np.linalg.norm(p-q)<=tolerance for p in (p0,p1) for q in (q0,q1))
            if adjacent: continue
            right_nodes=set(map(int,np.unique(mesh.elems[np.asarray(sorted(right),int)])))
            if left_nodes&right_nodes: short_circuits.append(f"{left_index}:{right_index}")
    junction_certificates=junction_sector_certificates(mesh,network,selected_ids,tolerance)
    reasons=[]
    if unresolved: reasons.append("INCOMPLETE_INTERIOR_NODE_STAR")
    if not exact.issubset(selected): reasons.append("MISSING_EXACT_GRAPH_SUPPORT")
    # These generous bounds reject global-domain or forward-strip fallbacks
    # while remaining independent of segment angle and endpoint phase.
    if width_ratio>4.0+1e-10: reasons.append("SUPPORT_NOT_O_H")
    if leakage_ratio>3.0+1e-10: reasons.append("ACTIVE_TIP_LEAKAGE_NOT_O_H")
    if np.max(retained_locality_ratio)>4.+1e-10: reasons.append("RETAINED_SUPPORT_NOT_LOCAL")
    if certificate["intact_cross_graph_path_exists"]: reasons.append("INTACT_CROSS_GRAPH_PATH")
    if certificate["insufficient_seed_segment_ids"]:
        by_id={edge.segment_id:edge for edge in certificate["edge_cut_certificates"]}
        arc_lengths={arc_id:float(np.linalg.norm(np.asarray(p1)-np.asarray(p0))) for p0,p1,arc_id in arcs}
        if any(arc_lengths[arc_id]<=2*by_id[arc_id].h_local_max_m for arc_id in certificate["insufficient_seed_segment_ids"]):
            reasons.append("CERTIFICATE_ARC_TOO_SHORT")
        else:
            reasons.append("INSUFFICIENT_OPPOSITE_SIDE_SEEDS")
    if premature: reasons.append("PREMATURE_MECHANICAL_COALESCENCE")
    if illegal_connection: reasons.append("DISTINCT_CRACK_COMPONENTS_UNRESOLVED_AT_CURRENT_MESH")
    if short_circuits: reasons.append("NONADJACENT_ARC_SUPPORT_SHORT_CIRCUIT")
    if any(item.junction_certificate_status!="ACCEPTED" for item in junction_certificates): reasons.append("JUNCTION_SECTOR_NOT_CERTIFIED")
    if previous and graph_fingerprint(network)!=graph_fingerprint(accepted_network) and not mechanically_new.size:
        reasons.append("NO_MECHANICALLY_NEW_SUPPORT")
    certified=not reasons
    graph_payload="|".join(f"{bid}:{idx}:{_point_key(a)}>{_point_key(b)}" for a,b,bid,idx in segments).encode()
    support_fp=_digest(np.ascontiguousarray(selected_ids,dtype=np.int64).tobytes())
    audit=GraphSupportAudit(tuple(map(int,selected_ids)),tuple(map(int,newly)),tuple(map(int,mechanically_new)),len(mechanically_new),
      _array_digest(accepted_field,np.float64),_array_digest(trial_field,np.float64),float(np.sum(mesh.area_e[selected_ids])),
      width,leakage,_digest(graph_payload),support_fp,tuple(sorted((_point_key(p),"+".join(sorted(k))) for p,k in classes.items())),
      graph_length,h,width_ratio,leakage_ratio,unresolved,
      certificate["intact_cross_graph_path_exists"],certificate["bridge_node_ids"],certificate["bridge_element_ids"],certificate["minimum_crossing_path_length"],
      certificate["positive_seed_element_ids"],certificate["negative_seed_element_ids"],certificate["certificate_fingerprint"],
      certificate["edge_cut_certificates"],certificate["insufficient_seed_segment_ids"],
      h,h_median,float(np.sum(mesh.area_e[selected_ids])/max(graph_length*h,1e-300)),endpoint_error,
      axial_extent,signed_tip_footprint,undershoot,h_tip_max,h_tip_median,h_tip_tangent,h_tip_normal,
      leakage/max(h_tip_max,1e-300),undershoot/max(h_tip_max,1e-300),
      component_edge,component_node,physical_component_count,graph_incidence,support_incidence,incidence_one_to_one,illegal_connection,component_separation,
      tuple(short_circuits),
      junction_certificates,
      not unresolved,not certificate["intact_cross_graph_path_exists"] and not certificate["insufficient_seed_segment_ids"],tuple(premature),certified,
      "CERTIFIED_INDEPENDENT_INTACT_CUT" if certified else ";".join(reasons))
    if not certified and not return_uncertified_audit_for_screen:
        junction_detail = tuple(
            (item.junction_id, item.junction_certificate_status,
             item.sector_seed_counts, item.within_sector_connected,
             item.cross_arm_path_exists, item.legal_support_overlap)
            for item in audit.junction_sector_certificates
            if item.junction_certificate_status != "ACCEPTED"
        )
        raise RuntimeError(
            f"v12_support_not_certified: {audit.certification_reason}; "
            f"junction_detail={junction_detail!r}"
        )
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

__all__=["MODEL_ID","EdgeCutCertificate","GraphSupportAudit","GraphSupportRecord","JunctionSectorCertificate","apply_mechanically_separating_graph","certification_arc_fingerprint",
  "certification_arcs","classify_graph_vertices","graph_component_count","graph_component_membership","graph_fingerprint","independent_intact_path_certificate",
  "junction_sector_certificates","mechanically_separating_graph_support","selected_support_components","support_record","unique_graph_length"]
