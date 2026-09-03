"""Default-off, backend-neutral V12 one-void state contract."""
from __future__ import annotations
from dataclasses import dataclass,field,replace
import copy,math
import numpy as np
from enum import Enum
from typing import Any,Mapping

SCHEMA="v12.stateful-voiding/1"
class VoidingState(str,Enum):
 AVAILABLE_SITE="AVAILABLE_SITE"; EMBRYO="EMBRYO"; HEALED_SITE="HEALED_SITE"
 CONSUMED_SITE="CONSUMED_SITE"; STABLE_SUBGRID_VOID="STABLE_SUBGRID_VOID"
 RESOLVED_VOID="RESOLVED_VOID"; CONNECTED_VOID="CONNECTED_VOID"
 DOWNSTREAM_FRONT_ACTIVE="DOWNSTREAM_FRONT_ACTIVE"; MERGED_OR_CONSUMED="MERGED_OR_CONSUMED"

@dataclass(frozen=True)
class VoidingV3Config:
 voiding_enabled: bool=False
 schema: str=SCHEMA

@dataclass(frozen=True)
class VoidingV3State:
 sites: tuple[Any,...]=()
 cavities: tuple[Any,...]=()
 rng_state: Mapping[str,Any]=field(default_factory=dict)
 thresholds: Mapping[str,float]=field(default_factory=dict)
 growth_state: Mapping[str,Any]=field(default_factory=dict)
 geometry_lineage: Mapping[str,Any]=field(default_factory=dict)
 length_ledgers: Mapping[str,float]=field(default_factory=dict)
 schema: str=SCHEMA

@dataclass(frozen=True)
class FirstPassageClock:
 hazard: float
 threshold: float
 def residual_time(self,rate):
  return math.inf if rate<=0 else max(self.threshold-self.hazard,0.)/rate

@dataclass(frozen=True)
class SiteRecord:
 site_id: str; center_m: tuple[float,float]; state: VoidingState
 hits: int; required_hits: int; birth: FirstPassageClock
 stabilization: FirstPassageClock; healing: FirstPassageClock
 inventory: float=0.

@dataclass(frozen=True)
class CavityRecord:
 cavity_id: str; parent_site_id: str; center_m: tuple[float,float]
 radius_m: float; inventory: float; state: VoidingState
 lineage: tuple[str,...]=(); population_identity: str=""

def series_limited_growth_rate(diffusion_m_s,accommodation_m_s):
 values=(float(diffusion_m_s),float(accommodation_m_s))
 if not all(math.isfinite(v) for v in values): raise ValueError("growth channels must be finite")
 return 0. if min(values)<=0 else 1./(1./values[0]+1./values[1])

def signed_growth_rate(diffusion_m_s,accommodation_m_s,chemical_potential_drive):
 if chemical_potential_drive<0: return -series_limited_growth_rate(abs(diffusion_m_s),abs(accommodation_m_s))
 return series_limited_growth_rate(diffusion_m_s,accommodation_m_s)

def advance_site_localized(site: SiteRecord,dt_s: float,*,birth_rate: float,
 stabilization_rate: float,healing_rate: float):
 if dt_s<=0 or not math.isfinite(dt_s): raise ValueError("dt must be positive finite")
 current=site; remaining=float(dt_s); events=[]
 while remaining>1e-15*dt_s:
  if current.state==VoidingState.AVAILABLE_SITE:
   crossing=current.birth.residual_time(birth_rate); step=min(remaining,crossing)
   clock=replace(current.birth,hazard=min(current.birth.threshold,current.birth.hazard+max(birth_rate,0)*step))
   current=replace(current,birth=clock); remaining-=step
   if crossing>step or not math.isfinite(crossing): break
   hits=current.hits+1
   if hits<current.required_hits:
    current=replace(current,hits=hits,birth=FirstPassageClock(0.,current.birth.threshold)); events.append("BIRTH_HIT"); continue
   current=replace(current,hits=hits,state=VoidingState.EMBRYO); events.append("EMBRYO"); continue
  if current.state==VoidingState.EMBRYO:
   ts=current.stabilization.residual_time(stabilization_rate); th=current.healing.residual_time(healing_rate)
   crossing=min(ts,th); step=min(remaining,crossing); remaining-=step
   s=replace(current.stabilization,hazard=min(current.stabilization.threshold,current.stabilization.hazard+max(stabilization_rate,0)*step))
   h=replace(current.healing,hazard=min(current.healing.threshold,current.healing.hazard+max(healing_rate,0)*step))
   current=replace(current,stabilization=s,healing=h)
   if crossing>step or not math.isfinite(crossing): break
   current=replace(current,state=VoidingState.HEALED_SITE if th<=ts else VoidingState.CONSUMED_SITE)
   events.append("HEALED" if th<=ts else "STABILIZED"); break
  break
 return current,tuple(events),remaining

def stabilize_cavity(site: SiteRecord,radius_m: float):
 if site.state!=VoidingState.CONSUMED_SITE: raise ValueError("site is not stabilized")
 return CavityRecord("void:"+site.site_id,site.site_id,site.center_m,radius_m,site.inventory,
                     VoidingState.STABLE_SUBGRID_VOID,(site.site_id,),"population:"+site.site_id)

def grow_cavity(cavity:CavityRecord,dt_s:float,*,diffusion_m_s:float,accommodation_m_s:float,chemical_potential_drive:float):
 rate=signed_growth_rate(diffusion_m_s,accommodation_m_s,chemical_potential_drive)
 radius=cavity.radius_m+rate*dt_s
 if radius<=0: raise ValueError("growth step consumed cavity")
 return replace(cavity,radius_m=radius,inventory=cavity.inventory+4*math.pi*cavity.radius_m**2*rate*dt_s)

def promote_cavity(cavity:CavityRecord,*,minimum_radius_m:float,minimum_ligament_m:float,ligament_m:float):
 if cavity.state!=VoidingState.STABLE_SUBGRID_VOID or cavity.radius_m<minimum_radius_m or ligament_m<minimum_ligament_m:
  raise ValueError("cavity is not numerically resolvable")
 return replace(cavity,state=VoidingState.RESOLVED_VOID)

@dataclass(frozen=True)
class CrackVoidCandidate:
 candidate_id:str; start_m:tuple[float,float]; direction:tuple[float,float]
 cavity_id:str; barrier_id:str; threshold_id:str

def first_cavity_intersection(candidate:CrackVoidCandidate,cavity:CavityRecord):
 p=np.asarray(candidate.start_m,float); d=np.asarray(candidate.direction,float); d/=np.linalg.norm(d); c=np.asarray(cavity.center_m,float)
 b=float(d@(p-c)); disc=b*b-(float((p-c)@(p-c))-cavity.radius_m**2)
 if disc<0:return None
 roots=[-b-math.sqrt(disc),-b+math.sqrt(disc)]; positive=[r for r in roots if r>0]
 return None if not positive else tuple(p+min(positive)*d)

def _connect_candidate_geometry(candidate:CrackVoidCandidate,cavity:CavityRecord,*,existing_barrier_id:str):
 if candidate.barrier_id!=existing_barrier_id: raise ValueError("ligament must use existing cleavage barrier")
 hit=first_cavity_intersection(candidate,cavity)
 if hit is None:return None,cavity,{"classification":"MISS"}
 fractured=math.dist(candidate.start_m,hit); connected=replace(cavity,state=VoidingState.CONNECTED_VOID,lineage=cavity.lineage+(candidate.candidate_id,))
 ledger={"classification":"HIT","fractured_ligament_increment":fractured,
         "free_void_span_increment":2*cavity.radius_m,"active_front_coordinate_increment":fractured,
         "projected_fractured_length":fractured,"projected_free_span":2*cavity.radius_m,
         "projected_front_advance":fractured,"total_connected_free_surface_extent":fractured+2*cavity.radius_m}
 return hit,connected,ledger

def _make_downstream_front(cavity:CavityRecord,*,candidate_id:str,direction:tuple[float,float],renewed_r_tip_m:float):
 if cavity.state!=VoidingState.CONNECTED_VOID: raise ValueError("downstream front requires connected cavity")
 if renewed_r_tip_m<=0: raise ValueError("renewed analytical tip radius must be positive")
 return replace(cavity,state=VoidingState.DOWNSTREAM_FRONT_ACTIVE,
                lineage=cavity.lineage+(candidate_id,)),{"candidate_id":candidate_id,"direction":direction,
                "r_tip_m":renewed_r_tip_m,"source":"direct_cavity_boundary_tensor"}

def connect_crack_to_void(state:VoidingV3State,candidate:CrackVoidCandidate,*,existing_barrier_id:str,failure_stage=None):
 accepted=state; trial=copy.deepcopy(state)
 if candidate.barrier_id!=existing_barrier_id: raise ValueError("ligament must use existing cleavage barrier")
 cavity=next(c for c in trial.cavities if c.cavity_id==candidate.cavity_id)
 hit=first_cavity_intersection(candidate,cavity)
 if hit is None:return accepted,{"accepted":False,"reason":"MISS"}
 for stage in ("geometry","remesh","equilibrium","topology","late_veto"):
  if failure_stage==stage:return accepted,{"accepted":False,"reason":"INJECTED_"+stage.upper()}
 length=math.dist(candidate.start_m,hit); connected=replace(cavity,state=VoidingState.CONNECTED_VOID,lineage=cavity.lineage+(candidate.candidate_id,))
 cavities=tuple(connected if c.cavity_id==cavity.cavity_id else c for c in trial.cavities)
 ledger=dict(trial.length_ledgers); ledger.update({"fractured_ligament_increment":length,"free_void_span_increment":2*cavity.radius_m,"active_front_coordinate_increment":0.,"projected_fractured_length":length,"projected_free_span":2*cavity.radius_m,"projected_front_advance":0.,"total_connected_free_surface_extent":length+2*cavity.radius_m})
 return replace(trial,cavities=cavities,length_ledgers=ledger),{"accepted":True,"hit_m":hit,"downstream_tip_created":False}

def activate_downstream_front(state:VoidingV3State,cavity_id:str,front_id:str,*,renewed_r_tip_m:float=1e-8):
 cavity=next(c for c in state.cavities if c.cavity_id==cavity_id)
 if cavity.state!=VoidingState.CONNECTED_VOID: raise ValueError("cavity is not connected")
 updated=replace(cavity,state=VoidingState.DOWNSTREAM_FRONT_ACTIVE,lineage=cavity.lineage+(front_id,))
 return replace(state,cavities=tuple(updated if c.cavity_id==cavity_id else c for c in state.cavities),
                geometry_lineage={**state.geometry_lineage,"downstream_front_id":front_id,
                                  "r_tip_m":renewed_r_tip_m,
                                  "source":"direct_cavity_boundary_tensor"})

def initialize_voiding(config: VoidingV3Config, *, rng_state=None):
 """Disabled capability creates and advances no physical or stochastic state."""
 if not config.voiding_enabled: return None
 return VoidingV3State(rng_state={} if rng_state is None else rng_state)

def attach_disabled_voiding_manifest(physical_manifest: Mapping[str,Any]):
 """Capability metadata is separate and leaves the physical manifest untouched."""
 return dict(physical_manifest),{"voiding_schema":SCHEMA,"voiding_enabled":False}

__all__=["SCHEMA","CavityRecord","CrackVoidCandidate","FirstPassageClock","SiteRecord","VoidingState","VoidingV3Config","VoidingV3State","activate_downstream_front","advance_site_localized","attach_disabled_voiding_manifest","connect_crack_to_void","first_cavity_intersection","grow_cavity","initialize_voiding","promote_cavity","series_limited_growth_rate","signed_growth_rate","stabilize_cavity"]
