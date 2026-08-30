"""Conforming traction-free slit reference for V11 crack qualification.

The crack faces have duplicated nodes strictly inside ``(p0, p1)`` while both
endpoints remain shared.  No damage, residual-stiffness, or element-kill law is
used: traction freedom follows from the disconnected face connectivity.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .mesh import BoundaryData, rebuild_tri_mesh
from .voiding_v2 import HoleMesh, solve_static_hole


@dataclass(frozen=True)
class ConformingSlit:
    hole: HoleMesh
    upper_face_edges: np.ndarray
    lower_face_edges: np.ndarray


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


def recovered_face_traction_relative(slit: ConformingSlit, result) -> float:
    """Return the edge-integrated crack-face traction divided by remote stress."""
    mesh=slit.hole.mesh; adjacency={}
    for ei,elem in enumerate(mesh.elems):
        for a,b in ((elem[0],elem[1]),(elem[1],elem[2]),(elem[2],elem[0])):
            adjacency.setdefault(tuple(sorted((int(a),int(b)))),[]).append(ei)
    total=length_sum=0.0
    for edges in (slit.upper_face_edges,slit.lower_face_edges):
        for a,b in edges:
            ei=adjacency[tuple(sorted((int(a),int(b))))][0]
            s=result.sigma_gp[:,ei]; traction=np.asarray((s[2],s[1]))
            length=float(np.linalg.norm(mesh.nodes[b]-mesh.nodes[a]))
            total+=float(traction@traction)*length; length_sum+=length
    remote=abs(result.reaction_top_N_per_m)/float(np.ptp(mesh.nodes[:,0]))
    return float(np.sqrt(total)/max(remote*np.sqrt(length_sum),1e-300))


__all__=["ConformingSlit","build_conforming_slit_mesh","recovered_face_traction_relative","solve_conforming_slit"]
