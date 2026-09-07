"""V5 body-fitted explicit-cavity FEM adapter derived from PR #57.

This module is default-off and intentionally stops at prescribed static cavity
mechanics.  It contains no stochastic production-driver coupling and no claims
of material calibration or validation.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from scipy.sparse.linalg import spsolve
from .config import ElasticProperties, GeometryConfig
from .fem import assemble_mechanics, plane_strain_D, solve_dirichlet, stress_state
from .mesh import BoundaryData, TriMesh, rebuild_tri_mesh

SCHEMA = "v12.explicit-cavity-static/1"


def _segments_intersect_open_disk(a: np.ndarray, b: np.ndarray, center: np.ndarray, radius: float) -> bool:
    ab = b-a
    t = float(np.clip(np.dot(center-a, ab)/max(np.dot(ab, ab), 1e-300), 0.0, 1.0))
    return float(np.linalg.norm(a+t*ab-center)) < radius*(1.0-1e-10)


def _point_in_triangle(p: np.ndarray, tri: np.ndarray) -> bool:
    v0, v1, v2 = tri[2]-tri[0], tri[1]-tri[0], p-tri[0]
    d00, d01, d02 = v0@v0, v0@v1, v0@v2
    d11, d12 = v1@v1, v1@v2
    den = d00*d11-d01*d01
    if abs(den) < 1e-300:
        return False
    u, v = (d11*d02-d01*d12)/den, (d00*d12-d01*d02)/den
    return u >= 0 and v >= 0 and u+v <= 1


def triangle_intersects_open_disk(tri: np.ndarray, center: Sequence[float], radius: float) -> bool:
    c = np.asarray(center, float)
    if np.any(np.linalg.norm(tri-c, axis=1) < radius*(1-1e-10)):
        return True
    if _point_in_triangle(c, tri):
        return True
    return any(_segments_intersect_open_disk(tri[i], tri[(i+1)%3], c, radius) for i in range(3))


def _edge_counts(elems: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.sort(np.vstack((elems[:,[0,1]], elems[:,[1,2]], elems[:,[2,0]])), axis=1)
    return np.unique(edges, axis=0, return_counts=True)


def _components(edges: np.ndarray) -> list[np.ndarray]:
    adjacency: dict[int, set[int]] = {}
    for a,b in edges:
        adjacency.setdefault(int(a),set()).add(int(b)); adjacency.setdefault(int(b),set()).add(int(a))
    out=[]
    while adjacency:
        seed=next(iter(adjacency)); stack=[seed]; nodes=[]
        while stack:
            n=stack.pop()
            if n not in adjacency: continue
            nbr=adjacency.pop(n); nodes.append(n); stack.extend(nbr)
        out.append(np.array(sorted(nodes),int))
    return out


def cavity_edge_traction_geometry(a, b, cavity_interior_point, stress_tensor):
    """Return orientation-independent edge geometry and analytic traction."""
    a=np.asarray(a,float); b=np.asarray(b,float); center=np.asarray(cavity_interior_point,float)
    if tuple(a) > tuple(b): a,b=b,a
    edge=b-a; length=float(np.linalg.norm(edge))
    if not length>0.0: raise ValueError("cavity edge must have positive length")
    tangent=edge/length; normal=np.array((tangent[1],-tangent[0])); midpoint=.5*(a+b); radial=midpoint-center
    if float(normal@radial)<0.0: normal=-normal
    radial_unit=radial/max(float(np.linalg.norm(radial)),1e-300)
    traction=np.asarray(stress_tensor,float)@normal
    return {"canonical_edge_tangent":tangent,"cavity_outward_into_solid_normal":normal,
            "solid_domain_outward_into_cavity_normal":-normal,"edge_length_m":length,
            "center_radial_consistency":float(normal@radial_unit),"traction":traction,
            "normal_traction":float(traction@normal),"tangential_traction":float(traction@tangent)}


@dataclass(frozen=True)
class HoleMesh:
    mesh: TriMesh
    boundary: BoundaryData
    center_m: tuple[float,float]
    radius_m: float
    cavity_edges: np.ndarray
    exterior_edges: np.ndarray
    prescribed_polygon_nodes: np.ndarray
    validation: Mapping[str, Any]


def _replace_split_edge(edges: np.ndarray, edge: tuple[int, int], new: int) -> np.ndarray:
    """Replace a boundary edge by its two conforming children, if present."""
    edge = tuple(sorted(edge))
    out = []
    found = False
    for raw in np.asarray(edges, dtype=int).reshape((-1, 2)):
        if tuple(sorted(map(int, raw))) == edge:
            out.extend(((edge[0], new), (new, edge[1])))
            found = True
        else:
            out.append(tuple(map(int, raw)))
    return np.asarray(out, dtype=int).reshape((-1, 2)) if found else np.asarray(edges, dtype=int)


def _final_mesh_validation(hole: HoleMesh) -> dict[str, Any]:
    """Recompute geometry/topology/quality facts from the realized mesh."""
    mesh = hole.mesh
    edges, counts = _edge_counts(mesh.elems)
    boundary_edges = edges[counts == 1]
    components = _components(boundary_edges)
    cavity_nodes = set(map(int, np.asarray(hole.cavity_edges).ravel()))
    degrees = {node: 0 for node in cavity_nodes}
    for a, b in np.asarray(hole.cavity_edges, dtype=int).reshape((-1, 2)):
        degrees[int(a)] += 1; degrees[int(b)] += 1
    tri = mesh.nodes[mesh.elems]
    side = np.linalg.norm(tri[:, [1, 2, 0]] - tri[:, [0, 1, 2]], axis=2)
    area = np.asarray(mesh.area_e, dtype=float)
    quality = 4.0 * np.sqrt(3.0) * area / np.maximum(np.sum(side ** 2, axis=1), 1e-300)
    cavity_lengths = (np.linalg.norm(mesh.nodes[hole.cavity_edges[:, 1]] -
                                     mesh.nodes[hole.cavity_edges[:, 0]], axis=1)
                      if len(hole.cavity_edges) else np.empty(0))
    validation = dict(hole.validation)
    validation.update({
        "actual_boundary_components": len(components),
        "actual_internal_components": 1 if cavity_nodes else 0,
        "cavity_cycle": bool(cavity_nodes) and all(value == 2 for value in degrees.values()),
        "triangle_disk_intersections": int(sum(triangle_intersects_open_disk(t, hole.center_m, hole.radius_m)
                                                   for t in tri)) if hole.radius_m else 0,
        "orphan_nodes": int(mesh.nn - len(np.unique(mesh.elems))),
        "minimum_quality": float(np.min(quality)),
        "maximum_aspect_ratio": float(np.max(side.max(axis=1) / side.min(axis=1))),
        "local_edge_min_m": float(np.min(cavity_lengths)) if len(cavity_lengths) else math.nan,
        "local_edge_max_m": float(np.max(cavity_lengths)) if len(cavity_lengths) else math.nan,
    })
    return validation


def conform_crack_path(hole: HoleMesh, crack_path_m: Sequence[Sequence[float]],
                       *, tolerance_m: float = 1.0e-12) -> tuple[HoleMesh, Mapping[str, Any]]:
    """Insert every fixed laboratory-frame crack vertex into a ``HoleMesh``.

    Boundary edge registries and displacement boundary node sets are updated
    when an inserted vertex splits an edge.  The first vertex must lie on the
    actual exterior boundary; subsequent vertices must lie in the material or
    on its boundary.  The returned audit records element ancestry and the
    post-insertion geometry generation.
    """
    requested = np.asarray(crack_path_m, dtype=float)
    if requested.ndim != 2 or requested.shape[0] < 2 or requested.shape[1] != 2:
        raise ValueError("crack_path_m must contain at least root and tip")
    if not np.all(np.isfinite(requested)):
        raise ValueError("crack_path_m must be finite")
    current = hole
    records = []
    generation = int(current.validation.get("geometry_generation", 0))
    for path_index, point in enumerate(requested):
        mesh = current.mesh
        distances = np.linalg.norm(mesh.nodes - point, axis=1)
        nearest = int(np.argmin(distances))
        split_edge = None
        parents: list[int] = []
        if float(distances[nearest]) <= tolerance_m:
            node = nearest
        else:
            owners = []
            for index, ids in enumerate(np.asarray(mesh.elems, dtype=int)):
                tri = mesh.nodes[ids]
                matrix = np.column_stack((tri[1] - tri[0], tri[2] - tri[0]))
                if abs(float(np.linalg.det(matrix))) <= 1e-24:
                    continue
                uv = np.linalg.solve(matrix, point - tri[0])
                if uv[0] >= -tolerance_m and uv[1] >= -tolerance_m and uv.sum() <= 1.0 + tolerance_m:
                    owners.append((index, uv))
            if not owners:
                raise ValueError(f"crack path point {path_index} is outside the specimen mesh")
            strict = [(index, uv) for index, uv in owners
                      if uv[0] > 1e-10 and uv[1] > 1e-10 and uv.sum() < 1.0 - 1e-10]
            source = np.asarray(mesh.elems, dtype=int)
            node = mesh.nn
            replacements = []
            removed = set()
            if strict:
                owner = strict[0][0]; parents = [owner]
                a, b, c = map(int, source[owner]); removed.add(owner)
                replacements.extend(((a, b, node), (b, c, node), (c, a, node)))
            else:
                candidates = []
                for owner, _ in owners:
                    ids = source[owner]
                    for u, v in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
                        delta = mesh.nodes[v] - mesh.nodes[u]
                        fraction = float((point - mesh.nodes[u]) @ delta / max(delta @ delta, 1e-300))
                        distance = abs(float(delta[0] * (point - mesh.nodes[u])[1] -
                                             delta[1] * (point - mesh.nodes[u])[0])) / math.sqrt(max(delta @ delta, 1e-300))
                        if -tolerance_m <= fraction <= 1 + tolerance_m and distance <= tolerance_m:
                            candidates.append(tuple(sorted((int(u), int(v)))))
                if not candidates:
                    raise ValueError(f"cannot conform crack path point {path_index}")
                split_edge = min(candidates)
                for owner, ids in enumerate(source):
                    if not set(split_edge).issubset(map(int, ids)): continue
                    parents.append(owner); removed.add(owner)
                    u, v = split_edge; w = next(int(value) for value in ids if int(value) not in split_edge)
                    original = mesh.nodes[ids]
                    oa, ob = original[1] - original[0], original[2] - original[0]
                    original_sign = oa[0] * ob[1] - oa[1] * ob[0]
                    for candidate in ((u, node, w), (node, v, w)):
                        xyz = np.vstack(tuple(point if q == node else mesh.nodes[q] for q in candidate))
                        ca, cb = xyz[1] - xyz[0], xyz[2] - xyz[0]
                        sign = ca[0] * cb[1] - ca[1] * cb[0]
                        replacements.append(candidate if sign * original_sign > 0 else (candidate[0], candidate[2], candidate[1]))
            elems = np.vstack((np.delete(source, sorted(removed), axis=0), replacements))
            rebuilt = rebuild_tri_mesh(np.vstack((mesh.nodes, point)), elems, tip_centers=requested[-1])
            exterior = current.exterior_edges
            cavity = current.cavity_edges
            boundary = current.boundary
            polygon = current.prescribed_polygon_nodes
            if split_edge is not None:
                exterior = _replace_split_edge(exterior, split_edge, node)
                cavity = _replace_split_edge(cavity, split_edge, node)
                if tuple(sorted(split_edge)) in {tuple(sorted(map(int, e))) for e in current.cavity_edges}:
                    polygon = np.append(polygon, node)
                top = np.asarray(boundary.top_nodes)
                bot = np.asarray(boundary.bot_nodes)
                if set(split_edge).issubset(map(int, top)): top = np.append(top, node)
                if set(split_edge).issubset(map(int, bot)): bot = np.append(bot, node)
                boundary = BoundaryData(np.unique(top), np.unique(bot), boundary.left_bot,
                                        boundary.right_bot, boundary.notch_nodes)
            current = replace(current, mesh=rebuilt, boundary=boundary, cavity_edges=cavity,
                              exterior_edges=exterior, prescribed_polygon_nodes=polygon)
            generation += 1
        realized = current.mesh.nodes[node]
        records.append({"path_index": path_index, "node_id": int(node),
                        "requested_m": point.tolist(), "realized_m": realized.tolist(),
                        "error_m": float(np.linalg.norm(realized - point)),
                        "parent_element_ids": parents,
                        "split_boundary_edge": None if split_edge is None else list(split_edge)})
    exterior_nodes = set(map(int, np.asarray(current.exterior_edges).ravel()))
    if records[0]["node_id"] not in exterior_nodes:
        raise ValueError("crack root is not on the actual exterior boundary component")
    validation = _final_mesh_validation(current)
    validation["geometry_generation"] = generation
    current = replace(current, validation=validation)
    audit = {"mode": "V3_FIXED_LABORATORY_GEOMETRY", "geometry_generation": generation,
             "path_records": records, "root_boundary_component": "exterior_traction_free",
             "maximum_requested_realized_error_m": max(r["error_m"] for r in records),
             "post_insertion_validation": validation}
    return current, audit


def build_solid_plate_mesh(width_m: float, height_m: float, h_m: float) -> HoleMesh:
    """Deterministic no-hole control using the same production CST mesh type."""
    nx=max(2,int(math.ceil(width_m/h_m))); ny=max(2,int(math.ceil(height_m/h_m)))
    xs=np.linspace(0,width_m,nx+1); ys=np.linspace(-height_m/2,height_m/2,ny+1)
    gx,gy=np.meshgrid(xs,ys); nodes=np.c_[gx.ravel(),gy.ravel()]
    elems=[]
    def node(i,j): return j*(nx+1)+i
    for j in range(ny):
        for i in range(nx):
            a,b,c,d=node(i,j),node(i+1,j),node(i+1,j+1),node(i,j+1)
            elems.extend(((a,b,c),(a,c,d)) if (i+j)%2==0 else ((a,b,d),(b,c,d)))
    mesh=rebuild_tri_mesh(nodes,np.asarray(elems,int))
    top=np.where(np.isclose(nodes[:,1],height_m/2))[0]; bot=np.where(np.isclose(nodes[:,1],-height_m/2))[0]
    boundary=BoundaryData(top,bot,node(0,0),node(nx,0),np.array([],int))
    return HoleMesh(mesh,boundary,(math.nan,math.nan),0.0,np.empty((0,2),int),
                    np.empty((0,2),int),np.empty(0,int),
                    {"actual_internal_components":0,"orphan_nodes":0})


def build_explicit_hole_mesh(
    width_m: float, height_m: float, center_m: tuple[float,float], radius_m: float,
    far_h_m: float, boundary_segments: int, *, radial_layers_override: Optional[int]=None,
) -> HoleMesh:
    """Build, prune, and derive the actual cavity boundary from retained connectivity."""
    cx,cy=center_m
    if boundary_segments < 16 or boundary_segments % 8:
        raise ValueError("boundary_segments must be a multiple of eight and at least 16")
    if not (radius_m < cx < width_m-radius_m and -height_m/2+radius_m < cy < height_m/2-radius_m):
        raise ValueError("cavity must lie strictly inside the plate")
    # Body-fitted polar-to-rectangle mapping.  Every angular ray begins on the
    # prescribed cavity and ends on the exterior rectangle, so the cavity cycle
    # is actual retained connectivity rather than a synthetic edge list.
    theta=2*np.pi*np.arange(boundary_segments)/boundary_segments
    direction=np.c_[np.cos(theta),np.sin(theta)]
    distances=[]
    for dx,dy in direction:
        candidates=[]
        if dx > 1e-14: candidates.append((width_m-cx)/dx)
        if dx < -1e-14: candidates.append((0-cx)/dx)
        if dy > 1e-14: candidates.append((height_m/2-cy)/dy)
        if dy < -1e-14: candidates.append((-height_m/2-cy)/dy)
        distances.append(min(v for v in candidates if v > 0))
    distances=np.asarray(distances)
    polygon_radius=radius_m/math.cos(math.pi/boundary_segments)
    radial_layers=(max(4,int(radial_layers_override)) if radial_layers_override is not None else
                   max(4,int(math.ceil((float(distances.max())-polygon_radius)/far_h_m))))
    nodes=[]
    for j in range(radial_layers+1):
        s=j/radial_layers
        # Quadratic grading makes the first radial layer comparable to the
        # cavity chord and expands smoothly toward the exterior rectangle.
        blend=s*(0.3+0.7*s)
        radius=polygon_radius+blend*(distances-polygon_radius)
        nodes.extend(np.c_[cx+radius*direction[:,0],cy+radius*direction[:,1]])
    nodes=np.asarray(nodes,float)
    elems=[]
    for j in range(radial_layers):
        base=j*boundary_segments; nxt=(j+1)*boundary_segments
        for i in range(boundary_segments):
            ip=(i+1)%boundary_segments
            a,b,c,d=base+i,base+ip,nxt+ip,nxt+i
            # Alternation avoids a systematic handed bias in symmetric probes.
            elems.extend(((a,b,c),(a,c,d)) if (i+j)%2==0 else ((a,b,d),(b,c,d)))
    elems=np.asarray(elems,int)
    signed=(nodes[elems[:,1],0]-nodes[elems[:,0],0])*(nodes[elems[:,2],1]-nodes[elems[:,0],1])-\
           (nodes[elems[:,1],1]-nodes[elems[:,0],1])*(nodes[elems[:,2],0]-nodes[elems[:,0],0])
    negative=signed<0
    elems[negative,1],elems[negative,2]=elems[negative,2].copy(),elems[negative,1].copy()
    mesh=rebuild_tri_mesh(nodes,elems,tip_centers=np.asarray(center_m))
    edges,counts=_edge_counts(elems); boundary_edges=edges[counts==1]
    components=_components(boundary_edges)
    internal=[]; exterior=[]
    tol=max(far_h_m*0.1,1e-12)
    for component in components:
        xy=nodes[component]
        on_outer=np.any((np.abs(xy[:,0])<tol)|(np.abs(xy[:,0]-width_m)<tol)|
                        (np.abs(xy[:,1]+height_m/2)<tol)|(np.abs(xy[:,1]-height_m/2)<tol))
        (exterior if on_outer else internal).append(component)
    if len(internal)!=1:
        raise RuntimeError(f"expected one actual cavity boundary component, found {len(internal)}")
    cavity_nodes=set(map(int,internal[0])); cavity_edges=np.array(
        [edge for edge in boundary_edges if int(edge[0]) in cavity_nodes and int(edge[1]) in cavity_nodes],int)
    exterior_nodes=set(int(v) for comp in exterior for v in comp); exterior_edges=np.array(
        [edge for edge in boundary_edges if int(edge[0]) in exterior_nodes and int(edge[1]) in exterior_nodes],int)
    actual_xy=nodes[np.array(sorted(cavity_nodes))]
    prescribed_xy=nodes[np.arange(boundary_segments)]
    pairwise=np.linalg.norm(actual_xy[:,None,:]-prescribed_xy[None,:,:],axis=2)
    radius_error=float(max(np.max(np.min(pairwise,axis=1)),np.max(np.min(pairwise,axis=0))))
    exact_node_set=(cavity_nodes==set(range(boundary_segments)))
    degrees={n:0 for n in cavity_nodes}
    for a,b in cavity_edges: degrees[int(a)]+=1; degrees[int(b)]+=1
    crossings=sum(triangle_intersects_open_disk(nodes[e],center_m,radius_m) for e in elems)
    orphan=int(len(nodes)-len(np.unique(elems)))
    lengths=np.linalg.norm(nodes[cavity_edges[:,1]]-nodes[cavity_edges[:,0]],axis=1)
    tri=nodes[elems]; side=np.linalg.norm(tri[:,[1,2,0]]-tri[:,[0,1,2]],axis=2)
    avec=tri[:,1]-tri[:,0]; bvec=tri[:,2]-tri[:,0]
    area=np.abs(avec[:,0]*bvec[:,1]-avec[:,1]*bvec[:,0])/2
    quality=4*np.sqrt(3)*area/np.maximum(np.sum(side**2,axis=1),1e-300)
    validation={"actual_internal_components":len(internal),"cavity_cycle":all(v==2 for v in degrees.values()),
                "triangle_disk_intersections":int(crossings),"orphan_nodes":orphan,
                "polygon_bidirectional_Hausdorff_m":radius_error,"polygon_exact_node_set_match":exact_node_set,
                "polygon_match_max_radius_error_m":radius_error,"minimum_quality":float(quality.min()),
                "maximum_aspect_ratio":float((side.max(axis=1)/side.min(axis=1)).max()),
                "minimum_angle_deg":float(np.degrees(np.arccos(np.clip(
                    (side[:,0]**2+side[:,2]**2-side[:,1]**2)/(2*side[:,0]*side[:,2]),-1,1))).min()),
                "local_edge_min_m":float(lengths.min()),"local_edge_max_m":float(lengths.max())}
    validation["radial_layers"]=radial_layers
    x,y=nodes[:,0],nodes[:,1]
    top=np.where(np.isclose(y,height_m/2,atol=tol))[0]; bot=np.where(np.isclose(y,-height_m/2,atol=tol))[0]
    lb=int(np.argmin(x*x+(y+height_m/2)**2)); rb=int(np.argmin((x-width_m)**2+(y+height_m/2)**2))
    boundary=BoundaryData(top,bot,lb,rb,np.array([],int))
    return HoleMesh(mesh,boundary,center_m,radius_m,cavity_edges,exterior_edges,
                    np.arange(boundary_segments,dtype=int),validation)


def fill_explicit_hole_mesh(hole: HoleMesh) -> HoleMesh:
    """Fill only the cavity patch to make a discretization-matched control."""
    if len(hole.cavity_edges) < 3:
        raise ValueError("an explicit cavity cycle is required")
    nodes=np.vstack((hole.mesh.nodes,np.asarray(hole.center_m,float)))
    center=len(nodes)-1
    adjacency: dict[int,list[int]]={}
    for a,b in hole.cavity_edges:
        adjacency.setdefault(int(a),[]).append(int(b)); adjacency.setdefault(int(b),[]).append(int(a))
    if any(len(v)!=2 for v in adjacency.values()):
        raise ValueError("cavity boundary is not a simple cycle")
    start=min(adjacency); ordered=[start]; previous=None; current=start
    while True:
        candidates=[n for n in adjacency[current] if n!=previous]
        nxt=candidates[0]
        if nxt==start: break
        ordered.append(nxt); previous,current=current,nxt
        if len(ordered)>len(adjacency): raise ValueError("invalid cavity cycle")
    xy=nodes[np.asarray(ordered)]
    signed=.5*np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1))
    if signed < 0: ordered=ordered[:1]+ordered[:0:-1]
    fan=np.asarray([(center,ordered[i],ordered[(i+1)%len(ordered)]) for i in range(len(ordered))],int)
    mesh=rebuild_tri_mesh(nodes,np.vstack((hole.mesh.elems,fan)))
    return HoleMesh(mesh,hole.boundary,(math.nan,math.nan),0.0,np.empty((0,2),int),
                    hole.exterior_edges,np.empty(0,int),
                    {"actual_internal_components":0,"orphan_nodes":0,
                     "matched_parent_nodes":hole.mesh.nn,"matched_parent_elements":hole.mesh.ne})


@dataclass(frozen=True)
class StaticFEMResult:
    displacement: np.ndarray
    sigma_gp: np.ndarray
    reaction_top_N_per_m: float
    reaction_bottom_N_per_m: float
    stored_energy_J_per_m: float
    compliance_m2_per_N: float
    free_residual_norm_N_per_m: float
    traction_l2_normalized: float
    hoop_stress_concentration: float
    symmetry_error: float
    crack_tip_sigma_yy_Pa: float = math.nan
    weak_cavity_residual_relative: float = math.nan
    mirror_sigma_xx_relative: float = math.nan
    mirror_sigma_yy_relative: float = math.nan
    mirror_sigma_xy_antisym_relative: float = math.nan
    conditioning_diagonal_ratio: float = math.nan
    killed_element_energy_J_per_m: float = math.nan
    traction_normal_l2_normalized: float = math.nan
    traction_tangential_l2_normalized: float = math.nan
    traction_resultant_normalized: tuple[float, float] = (math.nan, math.nan)
    traction_moment_normalized: float = math.nan
    traction_l2_dimensional_Pa_sqrt_m: float = math.nan
    traction_normal_l2_dimensional_Pa_sqrt_m: float = math.nan
    traction_tangential_l2_dimensional_Pa_sqrt_m: float = math.nan
    nominal_remote_stress_Pa: float = math.nan
    cavity_perimeter_m: float = math.nan
    cavity_edge_traction_records: tuple[Mapping[str, Any], ...] = ()


def solve_static_hole(hole: HoleMesh, opening_m: float, mat: Optional[ElasticProperties]=None,
                      *, crack_tip_m: Optional[tuple[float,float]]=None,
                      wake_half_width_m: float=0.0,
                      symmetric_rigid_constraint: bool=True,
                      element_kill_mask: Optional[np.ndarray]=None,
                      rigid_pin_node: Optional[int]=None,
                      residual_stiffness_kappa: float=1e-6) -> StaticFEMResult:
    """Use the unmodified production CST assembly and displacement solver."""
    mat=mat or ElasticProperties(E=210e9,nu=0.3)
    mesh=hole.mesh
    if crack_tip_m is not None:
        cent=mesh.nodes[mesh.elems].mean(axis=1)
        killed=(np.asarray(element_kill_mask,bool) if element_kill_mask is not None else
                ((cent[:,0] <= crack_tip_m[0]) &
                 (np.abs(cent[:,1]-crack_tip_m[1]) <= wake_half_width_m)))
        if killed.shape!=(mesh.ne,): raise ValueError("element_kill_mask must have one entry per element")
        # The V11 production assembler's authoritative P0 sharp-wake channel.
        mesh=replace(mesh,element_damage_gp=killed.astype(float))
    D=plane_strain_D(mat)
    u=np.zeros(mesh.ndof); ep=np.zeros((3,mesh.ne)); rho=np.zeros(mesh.ne); damage=np.zeros(mesh.nn)
    kappa=float(residual_stiffness_kappa) if crack_tip_m is not None else 0.0
    if not np.isfinite(kappa) or kappa<0.0: raise ValueError("residual_stiffness_kappa must be finite and nonnegative")
    K,R,*_=assemble_mechanics(mesh,u,ep,rho,damage,D,mat,kappa=kappa)
    prescribed=np.zeros(mesh.ndof,bool)
    prescribed[2*hole.boundary.top_nodes+1]=True; prescribed[2*hole.boundary.bot_nodes+1]=True
    if symmetric_rigid_constraint:
        mid=(int(rigid_pin_node) if rigid_pin_node is not None else
             int(np.argmin((mesh.nodes[:,0]-np.mean(mesh.nodes[:,0]))**2+mesh.nodes[:,1]**2)))
        if not 0<=mid<mesh.nn: raise ValueError("rigid_pin_node is outside the mesh")
        prescribed[2*mid]=True
        u_pres=np.zeros(mesh.ndof); u_pres[2*hole.boundary.top_nodes+1]=opening_m/2
        u_pres[2*hole.boundary.bot_nodes+1]=-opening_m/2
        free=~prescribed; u[free]=spsolve(K[np.ix_(free,free)],-R[free]-K[np.ix_(free,prescribed)]@u_pres[prescribed])
        u[prescribed]=u_pres[prescribed]
    else:
        u,reaction=solve_dirichlet(K,R,u,hole.boundary,opening_m/2,-opening_m/2)
        free=~prescribed
    K2,R2,*_=assemble_mechanics(mesh,u,ep,rho,damage,D,mat,kappa=kappa)
    sigma,*_,psi=stress_state(mesh,u,ep,damage,D,mat,kappa=kappa)
    residual=R2.copy()
    if not symmetric_rigid_constraint:
        prescribed[2*hole.boundary.left_bot:2*hole.boundary.left_bot+2]=True
        prescribed[2*hole.boundary.right_bot]=True
    free_norm=float(np.linalg.norm(residual[~prescribed]))
    top=float(np.sum(residual[2*hole.boundary.top_nodes+1])); bottom=float(np.sum(residual[2*hole.boundary.bot_nodes+1]))
    energy=float(0.5*u@(K2@u))
    diagonal=np.abs(K.diagonal()[free]); positive=diagonal[diagonal>0]
    conditioning_proxy=float(np.max(positive)/np.min(positive)) if len(positive) else math.inf
    killed_energy=(float(np.sum(psi[killed]*mesh.area_e[killed])) if crack_tip_m is not None else math.nan)
    compliance=float(opening_m/max(abs(top),1e-300))
    # Boundary traction from the unique adjacent CST, integrated edgewise.
    edge_to_elem={}
    for ei,elem in enumerate(mesh.elems):
        for edge in (tuple(sorted((elem[0],elem[1]))),tuple(sorted((elem[1],elem[2]))),tuple(sorted((elem[2],elem[0])))):
            edge_to_elem.setdefault(edge,[]).append(ei)
    t2=0.0; tn2=0.0; tt2=0.0; resultant=np.zeros(2); moment=0.0
    hoop=[]; weighted=[]; perimeter=0.0; edge_records=[]
    c=np.asarray(hole.center_m)
    for a,b in hole.cavity_edges:
        xy=mesh.nodes[[a,b]]; midpoint=xy.mean(axis=0)
        owners=edge_to_elem.get(tuple(sorted((int(a),int(b)))),())
        if len(owners)!=1: raise RuntimeError("CAVITY_BOUNDARY_EDGE_OWNER_COUNT_NOT_ONE")
        ei=owners[0]
        S=np.array([[sigma[0,ei],sigma[2,ei]],[sigma[2,ei],sigma[1,ei]]])
        geometry=cavity_edge_traction_geometry(xy[0],xy[1],c,S)
        tangent=geometry["canonical_edge_tangent"]; normal=geometry["cavity_outward_into_solid_normal"]
        length=geometry["edge_length_m"]; perimeter+=length
        traction=geometry["traction"]; normal_component=geometry["normal_traction"]; tangential_component=geometry["tangential_traction"]
        t2+=float(traction@traction)*length; tn2+=normal_component**2*length; tt2+=tangential_component**2*length
        force=traction*length; resultant+=force
        arm=midpoint-c; moment+=float(arm[0]*force[1]-arm[1]*force[0])
        edge_records.append({"edge_node_ids":(int(a),int(b)),"edge_endpoints_m":tuple(map(tuple,xy)),
          "canonical_edge_tangent":tuple(map(float,tangent)),
          "cavity_outward_into_solid_normal":tuple(map(float,normal)),
          "solid_domain_outward_into_cavity_normal":tuple(map(float,-normal)),
          "center_radial_consistency":geometry["center_radial_consistency"],
          "adjacent_solid_element_count":len(owners),"adjacent_element_id":int(ei),"edge_length_m":length,
          "traction_Pa":tuple(map(float,traction)),"normal_traction_Pa":normal_component,
          "tangential_traction_Pa":tangential_component})
        hoop.append(float(tangent@S@tangent)); weighted.append(length)
    remote=abs(top)/max(float(np.ptp(mesh.nodes[:,0])),1e-300)
    traction_norm=(math.sqrt(t2)/max(remote*math.sqrt(perimeter),1e-300)
                   if perimeter > 0 else math.nan)
    traction_normal=(math.sqrt(tn2)/max(remote*math.sqrt(perimeter),1e-300) if perimeter else math.nan)
    traction_tangential=(math.sqrt(tt2)/max(remote*math.sqrt(perimeter),1e-300) if perimeter else math.nan)
    resultant_normalized=(tuple(map(float,resultant/max(remote*perimeter,1e-300))) if perimeter else (math.nan,math.nan))
    moment_normalized=(float(moment/max(remote*perimeter*max(hole.radius_m,1e-300),1e-300)) if perimeter else math.nan)
    hoop_sc=(float(max(hoop)/max(remote,1e-300)) if hoop else math.nan)
    # Mirror-pair hoop samples after sorting by |x-cx|, y sign.
    symmetry=float(abs(top+bottom)/max(abs(top),1e-300))
    cavity_nodes=np.unique(hole.cavity_edges) if len(hole.cavity_edges) else np.empty(0,int)
    cavity_dofs=np.ravel(np.column_stack((2*cavity_nodes,2*cavity_nodes+1))) if len(cavity_nodes) else np.empty(0,int)
    weak_cavity=(float(np.linalg.norm(residual[cavity_dofs]))/max(abs(top),1e-300)
                 if len(cavity_dofs) else math.nan)
    cent=mesh.nodes[mesh.elems].mean(axis=1)
    try:
        from scipy.spatial import cKDTree
        mirrored=cent.copy(); mirrored[:,1]=2*float(np.mean(mesh.nodes[:,1]))-mirrored[:,1]
        distance,pair=cKDTree(cent).query(mirrored)
        scale=max(float(np.ptp(mesh.nodes[:,1])),1e-300)
        valid=distance<=1e-6*scale
        def mirror_error(component: int, sign: float) -> float:
            lhs=sigma[component,valid]; rhs=sign*sigma[component,pair[valid]]
            return float(np.linalg.norm(lhs-rhs)/max(np.linalg.norm(lhs),1e-300))
        mirror_xx,mirror_yy,mirror_xy=mirror_error(0,1),mirror_error(1,1),mirror_error(2,-1)
    except Exception:
        mirror_xx=mirror_yy=mirror_xy=math.nan
    tip_sigma=math.nan
    if crack_tip_m is not None:
        cent=mesh.nodes[mesh.elems].mean(axis=1)
        ahead=(cent[:,0]>=crack_tip_m[0])
        candidates=np.where(ahead)[0]
        if len(candidates):
            local=candidates[np.argmin(np.linalg.norm(cent[candidates]-np.asarray(crack_tip_m),axis=1))]
            tip_sigma=float(sigma[1,local])
    return StaticFEMResult(u,sigma,top,bottom,energy,compliance,free_norm,traction_norm,
                           hoop_sc,symmetry,tip_sigma,weak_cavity,mirror_xx,mirror_yy,mirror_xy,
                           conditioning_proxy,killed_energy,traction_normal,traction_tangential,
                           resultant_normalized,moment_normalized,math.sqrt(t2),math.sqrt(tn2),math.sqrt(tt2),
                           remote,perimeter,tuple(edge_records))


__all__ = ["HoleMesh","StaticFEMResult","build_explicit_hole_mesh","cavity_edge_traction_geometry",
           "build_solid_plate_mesh","conform_crack_path","fill_explicit_hole_mesh",
           "solve_static_hole","triangle_intersects_open_disk"]
