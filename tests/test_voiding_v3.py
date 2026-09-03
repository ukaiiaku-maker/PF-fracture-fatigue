from copy import deepcopy
from dataclasses import replace
from arrhenius_fracture.voiding_v3 import *
import pytest

def test_default_off_instantiates_nothing_and_does_not_touch_rng():
 config=VoidingV3Config(); rng={"state":[3,1,4]}; before=deepcopy(rng)
 assert initialize_voiding(config,rng_state=rng) is None
 assert rng==before

def test_disabled_metadata_is_separate_from_physical_artifact():
 physical={"events":[1,2],"graph":"abc","energy":3.0}; before=deepcopy(physical)
 result,capability=attach_disabled_voiding_manifest(physical)
 assert result==before==physical and capability["voiding_enabled"] is False

def test_contract_has_all_required_typed_states():
 assert {item.value for item in VoidingState}=={"AVAILABLE_SITE","EMBRYO","HEALED_SITE","CONSUMED_SITE","STABLE_SUBGRID_VOID","RESOLVED_VOID","CONNECTED_VOID","DOWNSTREAM_FRONT_ACTIVE","MERGED_OR_CONSUMED"}

def site():
 return SiteRecord("s",(0.,0.),VoidingState.AVAILABLE_SITE,0,2,
  FirstPassageClock(0.,.25),FirstPassageClock(0.,.5),FirstPassageClock(0.,10.),1.)

def test_multihit_localized_lifecycle_and_partition():
 one,events,_=advance_site_localized(site(),1.,birth_rate=1.,stabilization_rate=1.,healing_rate=1.)
 split,_,_=advance_site_localized(site(),.4,birth_rate=1.,stabilization_rate=1.,healing_rate=1.)
 split,_,_=advance_site_localized(split,.6,birth_rate=1.,stabilization_rate=1.,healing_rate=1.)
 assert one==split and events==("BIRTH_HIT","EMBRYO","STABILIZED")
 cavity=stabilize_cavity(one,1e-8)
 assert cavity.state==VoidingState.STABLE_SUBGRID_VOID

def test_healing_competes_and_creates_no_cavity():
 s=replace(site(),required_hits=1,healing=FirstPassageClock(0.,.1))
 healed,events,_=advance_site_localized(s,1.,birth_rate=1.,stabilization_rate=1.,healing_rate=1.)
 assert healed.state==VoidingState.HEALED_SITE and events[-1]=="HEALED"
 with pytest.raises(ValueError): stabilize_cavity(healed,1e-8)

def test_series_growth_sign_inventory_and_promotion_identity():
 assert series_limited_growth_rate(2.,3.)==pytest.approx(1.2)
 assert series_limited_growth_rate(-2.,3.)==0.
 c=CavityRecord("v","s",(0.,0.),1.,2.,VoidingState.STABLE_SUBGRID_VOID,("s",),"pop:s")
 grown=grow_cavity(c,.1,diffusion_m_s=2.,accommodation_m_s=3.,chemical_potential_drive=1.)
 shrunk=grow_cavity(c,.1,diffusion_m_s=2.,accommodation_m_s=3.,chemical_potential_drive=-1.)
 assert grown.radius_m>c.radius_m>shrunk.radius_m
 promoted=promote_cavity(grown,minimum_radius_m=1.,minimum_ligament_m=.2,ligament_m=.3)
 assert promoted.population_identity==c.population_identity and promoted.state==VoidingState.RESOLVED_VOID

def test_ligament_hit_miss_ledgers_and_separate_downstream_activation():
 c=CavityRecord("v","s",(2.,0.),.5,1.,VoidingState.RESOLVED_VOID,("s",),"pop:s")
 state=VoidingV3State(cavities=(c,))
 miss=CrackVoidCandidate("miss",(0.,0.),(1.,1.),"v","cleavage","t1")
 unchanged,info=connect_crack_to_void(state,miss,existing_barrier_id="cleavage")
 assert unchanged is state and info["reason"]=="MISS"
 hit=CrackVoidCandidate("hit",(0.,0.),(1.,0.),"v","cleavage","t2")
 connected,info=connect_crack_to_void(state,hit,existing_barrier_id="cleavage")
 assert info["hit_m"]==pytest.approx((1.5,0.)) and connected.length_ledgers["fractured_ligament_increment"]==pytest.approx(1.5)
 assert connected.cavities[0].state==VoidingState.CONNECTED_VOID and not info["downstream_tip_created"]
 active=activate_downstream_front(connected,"v","downstream")
 assert active.cavities[0].state==VoidingState.DOWNSTREAM_FRONT_ACTIVE
 assert active.geometry_lineage["source"]=="direct_cavity_boundary_tensor"

@pytest.mark.parametrize("stage",("geometry","remesh","equilibrium","topology","late_veto"))
def test_crack_void_injected_failures_are_exact_rollback(stage):
 c=CavityRecord("v","s",(2.,0.),.5,1.,VoidingState.RESOLVED_VOID,("s",),"pop:s")
 state=VoidingV3State(cavities=(c,),rng_state={"x":[1,2]},thresholds={"t":.4})
 candidate=CrackVoidCandidate("hit",(0.,0.),(1.,0.),"v","cleavage","t")
 result,info=connect_crack_to_void(state,candidate,existing_barrier_id="cleavage",failure_stage=stage)
 assert result is state and info["reason"]=="INJECTED_"+stage.upper()

def test_complete_deterministic_one_void_sequence():
 s=site(); s,events,_=advance_site_localized(s,1.,birth_rate=1.,stabilization_rate=1.,healing_rate=0.)
 cavity=stabilize_cavity(s,1e-8); cavity=grow_cavity(cavity,1.,diffusion_m_s=2e-8,accommodation_m_s=2e-8,chemical_potential_drive=1.)
 cavity=promote_cavity(cavity,minimum_radius_m=1e-8,minimum_ligament_m=1e-8,ligament_m=5e-8)
 state=VoidingV3State(sites=(s,),cavities=(cavity,),geometry_lineage={"events":events})
 candidate=CrackVoidCandidate("ligament",(-1e-7,0.),(1.,0.),cavity.cavity_id,"cleavage","threshold")
 state,info=connect_crack_to_void(state,candidate,existing_barrier_id="cleavage")
 assert info["accepted"] and not info["downstream_tip_created"]
 state=activate_downstream_front(state,cavity.cavity_id,"front:new")
 state=replace(state,geometry_lineage={**state.geometry_lineage,"continued_front_events":1})
 assert state.cavities[0].state==VoidingState.DOWNSTREAM_FRONT_ACTIVE
 assert state.geometry_lineage["continued_front_events"]==1
