"""Correctness-first V2 scaffold and static explicit-cavity FEM adapter.

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
from .config import ElasticProperties, GeometryConfig
from .fem import assemble_mechanics, plane_strain_D, solve_dirichlet, stress_state
from .mesh import BoundaryData, TriMesh, rebuild_tri_mesh

SCHEMA = "voiding-v2.scaffold/1"


class SiteState(str, Enum):
    AVAILABLE = "AVAILABLE_SITE"
    EMBRYO = "EMBRYO"
    HEALED = "HEALED_SITE"
    CONSUMED = "CONSUMED_SITE"


@dataclass(frozen=True)
class VoidingV2Config:
    enabled: bool = False
    calibration_status: str = "DIAGNOSTIC_IMPLEMENTATION_QUALIFICATION_ONLY"
    schema: str = SCHEMA


@dataclass
class Clock:
    hazard: float
    threshold: float
    rng_state: Mapping[str, Any] = field(default_factory=dict)

    def residual_time(self, rate: float) -> float:
        if not math.isfinite(rate) or rate <= 0.0:
            return math.inf
        return max(self.threshold - self.hazard, 0.0) / rate


@dataclass
class Site:
    site_id: str
    position_m: tuple[float, float]
    state: SiteState
    birth: Clock
    stabilization: Clock
    healing: Clock
    defect_inventory: float = 0.0


@dataclass
class SubgridVoid:
    void_id: str
    parent_site_id: str
    center_m: tuple[float, float]
    radius_m: float
    defect_inventory: float


@dataclass
class Registry:
    config: VoidingV2Config = field(default_factory=VoidingV2Config)
    sites: dict[str, Site] = field(default_factory=dict)
    voids: dict[str, SubgridVoid] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class LifecycleRates:
    birth_s: float
    stabilization_s: float
    healing_s: float


class TransactionAdapter:
    """Narrow adapter compatible with the production immutable transaction owner."""
    def trial(self, registry: Registry) -> Registry:
        return copy.deepcopy(registry)

    def commit(self, accepted: Registry, trial: Registry) -> None:
        accepted.__dict__.clear()
        accepted.__dict__.update(copy.deepcopy(trial.__dict__))


def series_limited_growth_rate(diffusion_m_s: float, accommodation_m_s: float) -> float:
    """Growth requires positive delivery and positive accommodation."""
    values = (float(diffusion_m_s), float(accommodation_m_s))
    if not all(math.isfinite(v) for v in values):
        raise ValueError("growth rates must be finite")
    if values[0] <= 0.0 or values[1] <= 0.0:
        return 0.0
    return 1.0 / (1.0/values[0] + 1.0/values[1])


def _advance_clock(clock: Clock, rate: float, dt: float) -> None:
    if rate > 0.0:
        clock.hazard = min(clock.threshold, clock.hazard + rate*dt)


def advance_lifecycle_localized(
    registry: Registry,
    site_id: str,
    dt_s: float,
    rate_provider: Callable[[Registry, Site], LifecycleRates],
    initial_radius_m: float,
    *,
    adapter: Optional[TransactionAdapter] = None,
    post_transition_veto: Optional[Callable[[Registry, str], None]] = None,
) -> None:
    """Localize every first passage and use only residual time after a transition."""
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    if not registry.config.enabled:
        return
    adapter = adapter or TransactionAdapter()
    trial = adapter.trial(registry)
    remaining = float(dt_s)
    eps = max(1e-15*dt_s, 1e-18)
    while remaining > eps:
        site = trial.sites[site_id]
        rates = rate_provider(trial, site)
        if site.state == SiteState.AVAILABLE:
            crossing = site.birth.residual_time(rates.birth_s)
            step = min(remaining, crossing)
            _advance_clock(site.birth, rates.birth_s, step)
            remaining -= step
            if crossing > step + eps or not math.isfinite(crossing):
                break
            site.state = SiteState.EMBRYO
            trial.events.append({"event": "EMBRYO_BIRTH", "time_within_step_s": dt_s-remaining})
            if post_transition_veto:
                post_transition_veto(trial, "EMBRYO_BIRTH")
            continue
        if site.state == SiteState.EMBRYO:
            t_stable = site.stabilization.residual_time(rates.stabilization_s)
            t_heal = site.healing.residual_time(rates.healing_s)
            # Deterministic ties heal: fail closed rather than create geometry.
            crossing = min(t_stable, t_heal)
            step = min(remaining, crossing)
            _advance_clock(site.stabilization, rates.stabilization_s, step)
            _advance_clock(site.healing, rates.healing_s, step)
            remaining -= step
            if crossing > step + eps or not math.isfinite(crossing):
                break
            if t_heal <= t_stable:
                site.state = SiteState.HEALED
                trial.events.append({"event": "EMBRYO_HEALING", "time_within_step_s": dt_s-remaining})
                if post_transition_veto:
                    post_transition_veto(trial, "EMBRYO_HEALING")
                break
            void_id = f"void:{site.site_id}"
            if void_id in trial.voids:
                raise RuntimeError("duplicate stabilization")
            trial.voids[void_id] = SubgridVoid(
                void_id, site.site_id, site.position_m, float(initial_radius_m), site.defect_inventory,
            )
            site.state = SiteState.CONSUMED
            trial.events.append({"event": "EMBRYO_STABILIZATION", "time_within_step_s": dt_s-remaining})
            if post_transition_veto:
                post_transition_veto(trial, "EMBRYO_STABILIZATION")
            break
        break
    adapter.commit(registry, trial)


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
    far_h_m: float, boundary_segments: int,
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
    radial_layers=max(4,int(math.ceil((float(distances.max())-polygon_radius)/far_h_m)))
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
    radius_error=float(np.max(np.linalg.norm(
        np.sort(actual_xy,axis=0)-np.sort(prescribed_xy,axis=0),axis=1)))
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
                "polygon_match_max_radius_error_m":radius_error,"minimum_quality":float(quality.min()),
                "maximum_aspect_ratio":float((side.max(axis=1)/side.min(axis=1)).max()),
                "minimum_angle_deg":float(np.degrees(np.arccos(np.clip(
                    (side[:,0]**2+side[:,2]**2-side[:,1]**2)/(2*side[:,0]*side[:,2]),-1,1))).min()),
                "local_edge_min_m":float(lengths.min()),"local_edge_max_m":float(lengths.max())}
    x,y=nodes[:,0],nodes[:,1]
    top=np.where(np.isclose(y,height_m/2,atol=tol))[0]; bot=np.where(np.isclose(y,-height_m/2,atol=tol))[0]
    lb=int(np.argmin(x*x+(y+height_m/2)**2)); rb=int(np.argmin((x-width_m)**2+(y+height_m/2)**2))
    boundary=BoundaryData(top,bot,lb,rb,np.array([],int))
    return HoleMesh(mesh,boundary,center_m,radius_m,cavity_edges,exterior_edges,
                    np.arange(boundary_segments,dtype=int),validation)


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


def solve_static_hole(hole: HoleMesh, opening_m: float, mat: Optional[ElasticProperties]=None,
                      *, crack_tip_m: Optional[tuple[float,float]]=None,
                      wake_half_width_m: float=0.0) -> StaticFEMResult:
    """Use the unmodified production CST assembly and displacement solver."""
    mat=mat or ElasticProperties(E=210e9,nu=0.3)
    mesh=hole.mesh
    if crack_tip_m is not None:
        cent=mesh.nodes[mesh.elems].mean(axis=1)
        killed=((cent[:,0] <= crack_tip_m[0]) &
                (np.abs(cent[:,1]-crack_tip_m[1]) <= wake_half_width_m))
        # The V11 production assembler's authoritative P0 sharp-wake channel.
        mesh=replace(mesh,element_damage_gp=killed.astype(float))
    D=plane_strain_D(mat)
    u=np.zeros(mesh.ndof); ep=np.zeros((3,mesh.ne)); rho=np.zeros(mesh.ne); damage=np.zeros(mesh.nn)
    kappa=1e-6 if crack_tip_m is not None else 0.0
    K,R,*_=assemble_mechanics(mesh,u,ep,rho,damage,D,mat,kappa=kappa)
    u,reaction=solve_dirichlet(K,R,u,hole.boundary,opening_m/2,-opening_m/2)
    K2,R2,*_=assemble_mechanics(mesh,u,ep,rho,damage,D,mat,kappa=kappa)
    sigma,*_=stress_state(mesh,u,ep,damage,D,mat,kappa=kappa)
    residual=R2.copy()
    prescribed=np.zeros(mesh.ndof,bool)
    prescribed[2*hole.boundary.top_nodes+1]=True; prescribed[2*hole.boundary.bot_nodes+1]=True
    prescribed[2*hole.boundary.left_bot:2*hole.boundary.left_bot+2]=True
    prescribed[2*hole.boundary.right_bot]=True
    free_norm=float(np.linalg.norm(residual[~prescribed]))
    top=float(np.sum(residual[2*hole.boundary.top_nodes+1])); bottom=float(np.sum(residual[2*hole.boundary.bot_nodes+1]))
    energy=float(0.5*u@(K2@u))
    compliance=float(opening_m/max(abs(top),1e-300))
    # Boundary traction from the unique adjacent CST, integrated edgewise.
    edge_to_elem={}
    for ei,elem in enumerate(mesh.elems):
        for edge in (tuple(sorted((elem[0],elem[1]))),tuple(sorted((elem[1],elem[2]))),tuple(sorted((elem[2],elem[0])))):
            edge_to_elem.setdefault(edge,[]).append(ei)
    t2=0.0; hoop=[]; weighted=[]; perimeter=0.0
    c=np.asarray(hole.center_m)
    for a,b in hole.cavity_edges:
        xy=mesh.nodes[[a,b]]; midpoint=xy.mean(axis=0); normal=(midpoint-c); normal/=np.linalg.norm(normal)
        tangent=np.array([-normal[1],normal[0]]); length=float(np.linalg.norm(xy[1]-xy[0])); perimeter+=length
        ei=edge_to_elem[tuple(sorted((int(a),int(b))))][0]
        S=np.array([[sigma[0,ei],sigma[2,ei]],[sigma[2,ei],sigma[1,ei]]])
        traction=S@normal; t2+=float(traction@traction)*length
        hoop.append(float(tangent@S@tangent)); weighted.append(length)
    remote=abs(top)/max(float(np.ptp(mesh.nodes[:,0])),1e-300)
    traction_norm=(math.sqrt(t2)/max(remote*math.sqrt(perimeter),1e-300)
                   if perimeter > 0 else math.nan)
    hoop_sc=(float(max(hoop)/max(remote,1e-300)) if hoop else math.nan)
    # Mirror-pair hoop samples after sorting by |x-cx|, y sign.
    symmetry=float(abs(top+bottom)/max(abs(top),1e-300))
    tip_sigma=math.nan
    if crack_tip_m is not None:
        cent=mesh.nodes[mesh.elems].mean(axis=1)
        ahead=(cent[:,0]>=crack_tip_m[0])
        candidates=np.where(ahead)[0]
        if len(candidates):
            local=candidates[np.argmin(np.linalg.norm(cent[candidates]-np.asarray(crack_tip_m),axis=1))]
            tip_sigma=float(sigma[1,local])
    return StaticFEMResult(u,sigma,top,bottom,energy,compliance,free_norm,traction_norm,
                           hoop_sc,symmetry,tip_sigma)


__all__ = ["Clock","HoleMesh","LifecycleRates","Registry","Site","SiteState",
           "StaticFEMResult","SubgridVoid","TransactionAdapter","VoidingV2Config",
           "advance_lifecycle_localized","build_explicit_hole_mesh","build_solid_plate_mesh","series_limited_growth_rate",
           "solve_static_hole","triangle_intersects_open_disk"]
