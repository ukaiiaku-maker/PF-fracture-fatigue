"""Conforming traction-free slit reference for V11 crack qualification.

The crack faces have duplicated nodes strictly inside ``(p0, p1)`` while both
endpoints remain shared.  No damage, residual-stiffness, or element-kill law is
used: traction freedom follows from the disconnected face connectivity.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np

from .mesh import BoundaryData, rebuild_tri_mesh
from .voiding_v2 import HoleMesh, solve_static_hole


@dataclass(frozen=True)
class ConformingSlit:
    hole: HoleMesh
    upper_face_edges: np.ndarray
    lower_face_edges: np.ndarray
    parent_node_of_node: np.ndarray | None = None


@dataclass(frozen=True)
class MatchedCrackParent:
    hole: HoleMesh
    p0: tuple[float, float]
    p1: tuple[float, float]
    h_tip_m: float
    geometry_fingerprint: str
    connectivity_fingerprint: str


def _fingerprint(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _axis(parts: list[tuple[float, float, float]]) -> np.ndarray:
    values=[]
    for lo,hi,step in parts:
        count=int(round((hi-lo)/step))
        if count<=0 or not np.isclose(lo+count*step,hi):
            raise ValueError("graded-axis intervals must be exact multiples")
        segment=np.linspace(lo,hi,count+1)
        values.extend(segment if not values else segment[1:])
    return np.asarray(values,float)


def build_matched_crack_parent(
    width_m: float, height_m: float, p0: tuple[float,float], p1: tuple[float,float], h_tip_m: float,
) -> MatchedCrackParent:
    """Locally refined crack-aligned intact parent shared by both representations."""
    h=float(h_tip_m)
    if h not in (25e-6,12.5e-6,6.25e-6):
        raise ValueError("matched screen supports h_tip = 25, 12.5, or 6.25 micrometres")
    # Exact fine corridor around the complete crack and right-tip process patch;
    # geometrically graded tensor bands keep the remote plate inexpensive.
    xs=_axis([(0.,.0004,.0001),(.0004,.0023,h),(.0023,.0031,.0001),(.0031,width_m,.000245)])
    ys=_axis([(-height_m/2,-.001,.00025),(-.001,-.0003,.0001),(-.0003,.0003,h),(.0003,.001,.0001),(.001,height_m/2,.00025)])
    if not any(np.isclose(xs,p0[0])) or not any(np.isclose(xs,p1[0])) or not any(np.isclose(ys,0.)):
        raise ValueError("physical crack must align with parent nodes")
    nx=len(xs)-1; ny=len(ys)-1; gx,gy=np.meshgrid(xs,ys); nodes=np.c_[gx.ravel(),gy.ravel()]
    def node(i,j): return j*(nx+1)+i
    elems=[]
    for j in range(ny):
        for i in range(nx):
            a,b,c,d=node(i,j),node(i+1,j),node(i+1,j+1),node(i,j+1)
            elems.extend(((a,b,c),(a,c,d)) if (i+j)%2==0 else ((a,b,d),(b,c,d)))
    mesh=rebuild_tri_mesh(nodes,np.asarray(elems,int))
    top=np.flatnonzero(np.isclose(nodes[:,1],height_m/2)); bot=np.flatnonzero(np.isclose(nodes[:,1],-height_m/2))
    boundary=BoundaryData(top,bot,node(0,0),node(nx,0),np.empty(0,int))
    hole=HoleMesh(mesh,boundary,(np.nan,np.nan),0.,np.empty((0,2),int),np.empty((0,2),int),np.empty(0,int),
                  {"actual_internal_components":0,"orphan_nodes":0})
    return MatchedCrackParent(hole,p0,p1,h,_fingerprint(nodes),_fingerprint(mesh.elems))


def conforming_slit_from_parent(parent: MatchedCrackParent, p1: tuple[float,float] | None=None) -> ConformingSlit:
    """Duplicate only lower crack-face nodes in an intact matched parent."""
    mesh=parent.hole.mesh; nodes=mesh.nodes.tolist(); tip=parent.p1 if p1 is None else p1
    p0=np.asarray(parent.p0,float); p1a=np.asarray(tip,float)
    line=np.flatnonzero(np.isclose(mesh.nodes[:,1],p0[1]) & (mesh.nodes[:,0]>=p0[0]) & (mesh.nodes[:,0]<=p1a[0]))
    line=line[np.argsort(mesh.nodes[line,0])]
    if len(line)<2 or not np.allclose(mesh.nodes[line[[0,-1]]],[p0,p1a]):
        raise ValueError("slit endpoints must be parent nodes")
    duplicate={int(node):len(nodes)+i for i,node in enumerate(line[1:-1])}
    nodes.extend(mesh.nodes[line[1:-1]].tolist())
    elems=mesh.elems.copy(); cent=mesh.nodes[elems].mean(axis=1)
    below=np.flatnonzero(cent[:,1]<p0[1])
    for ei in below:
        elems[ei]=[duplicate.get(int(node),int(node)) for node in elems[ei]]
    nodes=np.asarray(nodes,float); rebuilt=rebuild_tri_mesh(nodes,elems)
    upper=np.asarray(list(zip(line[:-1],line[1:])),int)
    lower_nodes=np.asarray([duplicate.get(int(node),int(node)) for node in line],int)
    lower=np.asarray(list(zip(lower_nodes[:-1],lower_nodes[1:])),int)
    parent_map=np.r_[np.arange(mesh.nn,dtype=int),line[1:-1]]
    incidence={}
    for elem in elems:
        for a,b in ((elem[0],elem[1]),(elem[1],elem[2]),(elem[2],elem[0])):
            incidence.setdefault(tuple(sorted((int(a),int(b)))),0); incidence[tuple(sorted((int(a),int(b))))]+=1
    face_edges={tuple(sorted(map(int,e))) for e in np.vstack((upper,lower))}
    if not all(incidence.get(edge)==1 for edge in face_edges): raise RuntimeError("derived slit faces are not boundary edges")
    adjacency={}
    for a,b in face_edges:
        adjacency.setdefault(a,set()).add(b); adjacency.setdefault(b,set()).add(a)
    components=0
    while adjacency:
        components+=1; stack=[next(iter(adjacency))]
        while stack:
            node=stack.pop()
            if node not in adjacency: continue
            stack.extend(adjacency.pop(node))
    # Connectivity-normalization proves the slit was derived only by duplicating
    # face nodes: mapping duplicates back exactly recovers the parent elements.
    normalized=parent_map[elems]
    if not np.array_equal(normalized,mesh.elems): raise RuntimeError("non-face connectivity changed")
    boundary=BoundaryData(parent.hole.boundary.top_nodes,parent.hole.boundary.bot_nodes,
                          parent.hole.boundary.left_bot,parent.hole.boundary.right_bot,np.empty(0,int))
    hole=HoleMesh(rebuilt,boundary,(float(np.mean((p0[0],p1a[0]))),-float(np.ptp(nodes[:,1]))),0.,
                  np.vstack((upper,lower)),np.empty((0,2),int),np.empty(0,int),
                  {"actual_internal_components":components,"shared_endpoints":True,"duplicated_interior_face_nodes":len(duplicate),
                   "normalized_parent_connectivity_fingerprint":_fingerprint(normalized)})
    return ConformingSlit(hole,upper,lower,parent_map)


def build_conforming_slit_mesh(
    width_m: float, height_m: float, p0: tuple[float, float], p1: tuple[float, float], h_m: float,
) -> ConformingSlit:
    """Build a structured CST plate with a horizontal, node-conforming slit."""
    if p0[1] != 0.0 or p1[1] != 0.0 or not 0.0 < p0[0] < p1[0] < width_m:
        raise ValueError("oracle requires an interior horizontal slit")
    nx=int(round(width_m/h_m)); ny=int(round(height_m/h_m))
    if nx<4 or ny<4 or not np.isclose(nx*h_m,width_m) or not np.isclose(ny*h_m,height_m):
        raise ValueError("plate dimensions must be integer multiples of h_m")
    xs=np.linspace(0.0,width_m,nx+1); ys=np.linspace(-height_m/2,height_m/2,ny+1)
    i0=int(np.argmin(abs(xs-p0[0]))); i1=int(np.argmin(abs(xs-p1[0]))); j0=int(np.argmin(abs(ys)))
    if not np.isclose(xs[i0],p0[0]) or not np.isclose(xs[i1],p1[0]) or not np.isclose(ys[j0],0.0):
        raise ValueError("p0, p1, and the crack line must be grid aligned")
    gx,gy=np.meshgrid(xs,ys); nodes=np.c_[gx.ravel(),gy.ravel()].tolist()
    def base(i,j): return j*(nx+1)+i
    lower={i:len(nodes)+k for k,i in enumerate(range(i0+1,i1))}
    nodes.extend([[xs[i],0.0] for i in range(i0+1,i1)])
    def node(i,j,side=None):
        if j==j0 and side=="lower" and i in lower: return lower[i]
        return base(i,j)
    elems=[]
    for j in range(ny):
        for i in range(nx):
            side="lower" if j==j0-1 else None
            a,b,c,d=node(i,j,side),node(i+1,j,side),node(i+1,j+1,side),node(i,j+1,side)
            elems.extend(((a,b,c),(a,c,d)) if (i+j)%2==0 else ((a,b,d),(b,c,d)))
    nodes=np.asarray(nodes,float); mesh=rebuild_tri_mesh(nodes,np.asarray(elems,int))
    upper=np.asarray([[node(i,j0),node(i+1,j0)] for i in range(i0,i1)],int)
    lower_edges=np.asarray([[node(i,j0,"lower"),node(i+1,j0,"lower")] for i in range(i0,i1)],int)
    top=np.flatnonzero(np.isclose(nodes[:,1],height_m/2)); bot=np.flatnonzero(np.isclose(nodes[:,1],-height_m/2))
    boundary=BoundaryData(top,bot,base(0,0),base(nx,0),np.empty(0,int))
    # The center is used only by a legacy cavity-traction diagnostic in the
    # shared solve adapter.  Keep it off the slit so that diagnostic never
    # constructs a zero radial normal; oracle face tractions are recovered
    # explicitly with the physical face normal below.
    hole=HoleMesh(mesh,boundary,(float(np.mean((p0[0],p1[0]))),-height_m),0.0,
                  np.vstack((upper,lower_edges)),np.empty((0,2),int),np.empty(0,int),
                  {"actual_internal_components":1,"shared_endpoints":True,
                   "duplicated_interior_face_nodes":len(lower)})
    return ConformingSlit(hole,upper,lower_edges)


def solve_conforming_slit(slit: ConformingSlit, opening_m: float, *, pin_node: int | None=None):
    """Solve the slit using the intact material law and natural free faces."""
    return solve_static_hole(slit.hole,opening_m,rigid_pin_node=pin_node)


def recovered_face_traction_relative(slit: ConformingSlit, result, *, trim_tip_distance_m: float=0.0) -> float:
    """Return the edge-integrated crack-face traction divided by remote stress."""
    mesh=slit.hole.mesh; adjacency={}
    for ei,elem in enumerate(mesh.elems):
        for a,b in ((elem[0],elem[1]),(elem[1],elem[2]),(elem[2],elem[0])):
            adjacency.setdefault(tuple(sorted((int(a),int(b)))),[]).append(ei)
    total=length_sum=0.0
    for edges in (slit.upper_face_edges,slit.lower_face_edges):
        for a,b in edges:
            midpoint=mesh.nodes[[a,b]].mean(axis=0)
            endpoints=mesh.nodes[[edges[0,0],edges[-1,1]]]
            if np.min(np.linalg.norm(endpoints-midpoint,axis=1)) < float(trim_tip_distance_m):
                continue
            ei=adjacency[tuple(sorted((int(a),int(b))))][0]
            s=result.sigma_gp[:,ei]; traction=np.asarray((s[2],s[1]))
            length=float(np.linalg.norm(mesh.nodes[b]-mesh.nodes[a]))
            total+=float(traction@traction)*length; length_sum+=length
    remote=abs(result.reaction_top_N_per_m)/float(np.ptp(mesh.nodes[:,0]))
    return float(np.sqrt(total)/max(remote*np.sqrt(length_sum),1e-300)) if length_sum else np.nan


__all__=["ConformingSlit","MatchedCrackParent","build_conforming_slit_mesh","build_matched_crack_parent",
         "conforming_slit_from_parent","recovered_face_traction_relative","solve_conforming_slit"]
