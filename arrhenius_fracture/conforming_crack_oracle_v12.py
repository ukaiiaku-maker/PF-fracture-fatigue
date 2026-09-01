"""Analysis-only conforming slit oracle for the V12 primal screen.

The node-splitting construction is transplanted from PR #57 commit
8ad7f42.  It deliberately has no dependency on the voiding solver or any
production transaction path.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np

from .mesh import BoundaryData, TriMesh, rebuild_tri_mesh

CONFORMING_ORACLE_SOURCE_COMMIT = "8ad7f42"


def _fingerprint(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


@dataclass(frozen=True)
class MatchedCrackParent:
    mesh: TriMesh
    boundary: BoundaryData
    p0: tuple[float, float]
    p1: tuple[float, float]
    h_tip_m: float
    geometry_fingerprint: str
    connectivity_fingerprint: str


@dataclass(frozen=True)
class ConformingSlit:
    mesh: TriMesh
    boundary: BoundaryData
    upper_face_edges: np.ndarray
    lower_face_edges: np.ndarray
    parent_node_of_node: np.ndarray


def build_matched_crack_parent(width_m, height_m, p0, p1, h_tip_m):
    """Return an exact crack-aligned structured parent mesh."""
    h=float(h_tip_m); nx=int(round(width_m/h)); ny=int(round(height_m/h))
    if h not in (25e-6,12.5e-6,6.25e-6) or not np.isclose(nx*h,width_m) or not np.isclose(ny*h,height_m):
        raise ValueError("domain and declared h_tip must be exactly commensurate")
    xs=np.linspace(0.,width_m,nx+1); ys=np.linspace(-height_m/2,height_m/2,ny+1)
    gx,gy=np.meshgrid(xs,ys); nodes=np.c_[gx.ravel(),gy.ravel()]
    def node(i,j): return j*(nx+1)+i
    elems=[]
    for j in range(ny):
        for i in range(nx):
            a,b,c,d=node(i,j),node(i+1,j),node(i+1,j+1),node(i,j+1)
            elems.extend(((a,b,c),(a,c,d)) if (i+j)%2==0 else ((a,b,d),(b,c,d)))
    mesh=rebuild_tri_mesh(nodes,np.asarray(elems,int),tip_centers=np.asarray(p1))
    line=lambda point: np.any(np.all(np.isclose(nodes,np.asarray(point),atol=1e-14),axis=1))
    if not line(p0) or not line(p1) or not np.isclose(p0[1],0.) or not np.isclose(p1[1],0.):
        raise ValueError("straight-screen slit must be grid aligned")
    bnd=BoundaryData(np.flatnonzero(np.isclose(nodes[:,1],height_m/2)),np.flatnonzero(np.isclose(nodes[:,1],-height_m/2)),node(0,0),node(nx,0),np.empty(0,int))
    return MatchedCrackParent(mesh,bnd,tuple(p0),tuple(p1),h,_fingerprint(nodes),_fingerprint(mesh.elems))


def conforming_slit_from_parent(parent: MatchedCrackParent, p1=None):
    """Duplicate lower-face interior nodes; endpoints remain shared."""
    mesh=parent.mesh; p0=np.asarray(parent.p0); tip=np.asarray(parent.p1 if p1 is None else p1)
    tol=max(1e-14,1e-9*parent.h_tip_m)
    line=np.flatnonzero(np.isclose(mesh.nodes[:,1],p0[1],atol=tol) & (mesh.nodes[:,0]>=p0[0]-tol) & (mesh.nodes[:,0]<=tip[0]+tol))
    line=line[np.argsort(mesh.nodes[line,0])]
    if len(line)<2 or not np.allclose(mesh.nodes[line[[0,-1]]],[p0,tip]): raise ValueError("slit endpoints must be parent nodes")
    nodes=mesh.nodes.tolist(); duplicate={int(n):len(nodes)+i for i,n in enumerate(line[1:-1])}; nodes.extend(mesh.nodes[line[1:-1]].tolist())
    elems=mesh.elems.copy(); below=np.flatnonzero(mesh.nodes[elems].mean(axis=1)[:,1]<p0[1])
    for ei in below: elems[ei]=[duplicate.get(int(n),int(n)) for n in elems[ei]]
    parent_map=np.r_[np.arange(mesh.nn,dtype=int),line[1:-1]]
    if not np.array_equal(parent_map[elems],mesh.elems): raise RuntimeError("non-face connectivity changed")
    rebuilt=rebuild_tri_mesh(np.asarray(nodes),elems,tip_centers=tip)
    lower_nodes=np.asarray([duplicate.get(int(n),int(n)) for n in line])
    upper=np.asarray(list(zip(line[:-1],line[1:])),int); lower=np.asarray(list(zip(lower_nodes[:-1],lower_nodes[1:])),int)
    b=parent.boundary; boundary=BoundaryData(b.top_nodes,b.bot_nodes,b.left_bot,b.right_bot,np.empty(0,int))
    return ConformingSlit(rebuilt,boundary,upper,lower,parent_map)


__all__=["CONFORMING_ORACLE_SOURCE_COMMIT","ConformingSlit","MatchedCrackParent","build_matched_crack_parent","conforming_slit_from_parent"]
