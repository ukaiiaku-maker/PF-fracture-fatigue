#!/usr/bin/env python3
"""Replacement V5 broad campaign with fail-closed, non-aliased evidence."""
from __future__ import annotations

from dataclasses import asdict, replace
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.voiding_production_v5 import (
    ROOT_BRANCH_ID, _complete_next_clock, build_production_void_state,
    cavity_boundary_tensor, crack_tip_tensor, deterministic_trajectory,
    downstream_front_transaction, equilibrate_fixed_load_with_production_fem,
    ligament_transaction, local_site_tensor, natural_trajectory, observables,
)
from arrhenius_fracture.voiding_v5 import (
    Cavity2D, ProductionVoidState, VoidPhase, VoidingConfig, advance_site,
    arrhenius_rates, create_subgrid_cavity, grow_cavity_from_rate, promote_cavity,
    update_cavity_growth,
)

PARTITIONS=(1,2,4,8,16)
CONTROL_NAMES=("centered","positive_offset","negative_offset","short_ligament",
 "long_ligament","diffusion_limited","accommodation_limited","embryo_healing",
 "downstream_zero_drive","delayed_downstream","fixed_mesh_oblique",
 "local_remesh_refinement")
RESTART_STAGES=("available_site","incomplete_first_hit","between_birth_hits","embryo",
 "stable_subgrid_cavity","before_promotion","after_promotion","before_ligament",
 "connected_before_downstream","downstream_front_before_continuation")

def canonical(value):
    if isinstance(value,np.ndarray): return canonical(value.tolist())
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,float) and not math.isfinite(value): return "infinity" if value>0 else "-infinity"
    if isinstance(value,dict): return {str(k):canonical(v) for k,v in sorted(value.items())}
    if isinstance(value,(tuple,list)): return [canonical(v) for v in value]
    return value

def digest(value):
    return hashlib.sha256(json.dumps(canonical(value),sort_keys=True,separators=(",",":")).encode()).hexdigest()

def equivalent_values(left,right):
    if isinstance(left,(int,float)) and isinstance(right,(int,float)) and not isinstance(left,bool) and not isinstance(right,bool):
        return math.isclose(float(left),float(right),rel_tol=1e-12,abs_tol=1e-15)
    if isinstance(left,dict) and isinstance(right,dict):
        return left.keys()==right.keys() and all(equivalent_values(left[key],right[key]) for key in left)
    if isinstance(left,(list,tuple)) and isinstance(right,(list,tuple)):
        return len(left)==len(right) and all(equivalent_values(a,b) for a,b in zip(left,right))
    return left==right

def partition_projection(value):
    """Remove only subdivision-count and interval-local audit coordinates."""
    if isinstance(value,dict):
        return {key:partition_projection(item) for key,item in value.items()
                if key not in {"geometry_generation","action_before","action_increment"}}
    if isinstance(value,list): return [partition_projection(item) for item in value]
    return value

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(canonical(value),indent=2,sort_keys=True)+"\n")
    temporary.replace(path)

def lifecycle_to_embryo(seed=3621):
    state,_=build_production_void_state(stochastic=True,seed=seed)
    cfg=VoidingConfig(enabled=True,promotion_radius_m=5e-5)
    rates=arrhenius_rates(cfg,temperature_K=900.,stress_tensor_Pa=np.eye(2)*1e9)
    void=state.void_state; events=[]
    for _ in range(2):
        site=void.sites[0]
        dt=max(site.birth.threshold-site.birth.accumulated,0.)/(rates["birth_s"]*site.candidate_weight)
        void,new=advance_site(void,"site-1",dt,rates=rates); events.extend(new)
    return replace(state,void_state=void),rates,events

def clock_partition(base,tensor,n,source):
    state=base
    _,preview=_complete_next_clock(state,tensor,**source)
    finite=[row["crossing_time_s"] for row in preview if math.isfinite(row["crossing_time_s"])]
    total=min(finite) if finite else 1.; audits=[]
    for _ in range(n):
        state,audit=_complete_next_clock(state,tensor,maximum_advance_duration_s=total/n,**source)
        audits.extend(audit)
    return state,audits,total

def transition_partition_rows():
    rows=[]
    for n in PARTITIONS:
        embryo,rates,birth_events=lifecycle_to_embryo(7100)
        rows.append({"execution_id":f"partition-birth-{n}","transition":"birth_hits","partitions":n,
          "events":birth_events,"state_measurement":canonical(asdict(embryo.void_state)),"state":digest(asdict(embryo.void_state)),"passed":len(birth_events)==2})
        site=embryo.void_state.sites[0]; total=site.stabilization.threshold/rates["stabilization_s"]
        void=embryo.void_state; events=[]
        for _ in range(n): void,new=advance_site(void,"site-1",total/n,rates=rates); events.extend(new)
        rows.append({"execution_id":f"partition-stabilization-{n}","transition":"stabilization","partitions":n,
          "events":events,"state_measurement":canonical(asdict(void)),"state":digest(asdict(void)),"passed":void.sites[0].phase==VoidPhase.STABLE_SUBGRID_VOID})
        heal={**rates,"stabilization_s":0.}; site=embryo.void_state.sites[0]; total=site.healing.threshold/heal["healing_s"]
        void=embryo.void_state; events=[]
        for _ in range(n): void,new=advance_site(void,"site-1",total/n,rates=heal); events.extend(new)
        rows.append({"execution_id":f"partition-healing-{n}","transition":"healing","partitions":n,
          "events":events,"state_measurement":canonical(asdict(void)),"state":digest(asdict(void)),"passed":void.sites[0].phase==VoidPhase.HEALED_SITE})
        area=math.pi*(2.5e-5)**2
        cavity=Cavity2D("partition-void","site-1",(0.,0.),2.5e-5,area,area,VoidPhase.STABLE_SUBGRID_VOID)
        grown=cavity
        for _ in range(n): grown=grow_cavity_from_rate(grown,rates=rates,dt_s=1e-6/n,radial_growth_scale_m=1e-9)
        rows.append({"execution_id":f"partition-growth-{n}","transition":"subgrid_growth","partitions":n,
          "radius_m":grown.radius_m,"state_measurement":canonical(asdict(grown)),"state":digest(asdict(grown)),"passed":grown.radius_m>cavity.radius_m})
        promoted_source=ProductionVoidState((),(replace(grown,radius_m=5e-5,area_m2=math.pi*(5e-5)**2,
          inventory_area_m2=math.pi*(5e-5)**2),))
        promoted=promote_cavity(promoted_source,"partition-void",5e-5)
        rows.append({"execution_id":f"partition-promotion-{n}","transition":"promotion","partitions":n,
          "state_measurement":canonical(asdict(promoted)),"state":digest(asdict(promoted)),"passed":promoted.cavities[0].phase==VoidPhase.RESOLVED_VOID})
        pre,_=deterministic_trajectory(stop_before_ligament=True)
        tensor,elements=crack_tip_tensor(pre,branch_id=ROOT_BRANCH_ID)
        advanced,audit,total=clock_partition(pre,tensor,n,{"source_kind":"sharp_front","source_front_id":ROOT_BRANCH_ID,
          "source_position_m":pre.crack_network.branch(ROOT_BRANCH_ID).tip,
          "source_probe_identity":{"kind":"crack_tip_tensor","element_ids":list(elements)}})
        rows.append({"execution_id":f"partition-ligament-{n}","transition":"ligament_first_passage","partitions":n,
          "duration_s":total,"clock_measurement":canonical(asdict(advanced.competition)),"clock":digest(asdict(advanced.competition)),"winner_count":sum(bool(x["winner"]) for x in audit),"passed":True})
        connected,_=ligament_transaction(pre); tensor,elements=cavity_boundary_tensor(connected)
        advanced,audit,total=clock_partition(connected,tensor,n,{"source_kind":"cavity_surface",
          "source_cavity_id":connected.void_state.cavities[0].cavity_id,"source_boundary_site_id":"connection_exit",
          "source_position_m":connected.void_state.cavities[0].connection_exit_m,
          "source_probe_identity":{"kind":"direct_cavity_boundary_tensor","element_ids":list(elements)}})
        rows.append({"execution_id":f"partition-downstream-{n}","transition":"downstream_first_passage","partitions":n,
          "duration_s":total,"clock_measurement":canonical(asdict(advanced.competition)),"clock":digest(asdict(advanced.competition)),"winner_count":sum(bool(x["winner"]) for x in audit),"passed":True})
        child,result,_,_=downstream_front_transaction(connected)
        if result is None:
            rows.append({"execution_id":f"partition-child-{n}","transition":"child_continuation","partitions":n,
                         "passed":False,"classification":"NO_CHILD_FIRST_PASSAGE"})
        else:
            tensor,elements=crack_tip_tensor(child,branch_id="void-front-1")
            advanced,audit,total=clock_partition(child,tensor,n,{"source_kind":"sharp_front","source_front_id":"void-front-1",
              "source_position_m":child.crack_network.branch("void-front-1").tip,
              "source_probe_identity":{"kind":"child_crack_tip_tensor","element_ids":list(elements)}})
            rows.append({"execution_id":f"partition-child-{n}","transition":"child_continuation","partitions":n,
              "duration_s":total,"clock_measurement":canonical(asdict(advanced.competition)),"clock":digest(asdict(advanced.competition)),"winner_count":sum(bool(x["winner"]) for x in audit),"passed":True})
    by={}
    for row in rows: by.setdefault(row["transition"],[]).append(row)
    for group in by.values():
        reference=group[0]
        for row in group:
            if "clock_measurement" in row:
                equivalent=equivalent_values(partition_projection(row["clock_measurement"]),
                  partition_projection(reference["clock_measurement"])) and row["winner_count"]==reference["winner_count"]
            else:
                equivalent=equivalent_values(partition_projection(row.get("state_measurement")),
                  partition_projection(reference.get("state_measurement"))) and row.get("events")==reference.get("events")
            row["partition_equivalent"]=bool(row["passed"] and equivalent)
            row["comparison_projection_excludes"]=["geometry_generation","pending_event.action_before","pending_event.action_increment"]
    return rows

def controlled_rows():
    specs={"centered":{"kind":"trajectory","center":(7e-4,0.)},"positive_offset":{"kind":"connection","center":(7e-4,1e-5)},
      "negative_offset":{"kind":"connection","center":(7e-4,-1e-5)},"short_ligament":{"kind":"connection","center":(6.75e-4,0.)},
      "long_ligament":{"kind":"connection","center":(7.35e-4,0.)},"diffusion_limited":{"kind":"rate","channel":"vacancy_transport_s"},
      "accommodation_limited":{"kind":"rate","channel":"plastic_accommodation_s"},"embryo_healing":{"kind":"healing"},
      "downstream_zero_drive":{"kind":"connection","center":(7e-4,-1e-5),"zero":True},
      "delayed_downstream":{"kind":"delayed","center":(7e-4,-1e-5)},
      "fixed_mesh_oblique":{"kind":"connection","center":(7e-4,0.),"theta":30.},
      "local_remesh_refinement":{"kind":"preligament","center":(7e-4,0.),"theta":45.}}
    rows=[]
    for index,name in enumerate(CONTROL_NAMES):
        spec=specs[name]; row={"execution_id":f"controlled-{index:02d}-{digest(spec)[:12]}","case":name,"input":spec,"input_hash":digest(spec)}
        try:
            if spec["kind"] in ("connection","preligament"):
                crack_path=None
                if name in ("positive_offset","negative_offset","downstream_zero_drive"):
                    crack_path=((0.,0.),(0.0005725993004046688,0.))
                if name=="fixed_mesh_oblique":
                    angle=math.radians(spec["theta"]); direction=np.asarray((math.cos(angle),math.sin(angle)))
                    center=np.asarray(spec["center"]); tip=center-1.5e-4*direction
                    root=np.asarray((0.,tip[1]-tip[0]*direction[1]/direction[0])); crack_path=(tuple(root),tuple(tip))
                pre,trace=deterministic_trajectory(stop_before_ligament=True,cavity_center_m=spec["center"],
                  crack_path_m=crack_path,cleavage_theta_deg=spec.get("theta",0.))
                final=pre if spec["kind"]=="preligament" else ligament_transaction(pre)[0]
                row.update({"trace":[x["operation"] for x in trace],"terminal":observables(final,name),
                            "terminal_fingerprint":complete_accepted_state_fingerprint(final),"passed":True})
            elif spec["kind"]=="trajectory":
                final,trace=deterministic_trajectory(cavity_center_m=spec["center"])
                row.update({"trace":[x["operation"] for x in trace],"terminal":observables(final,name),
                            "terminal_fingerprint":complete_accepted_state_fingerprint(final),"passed":True})
            elif spec["kind"]=="rate":
                rates=arrhenius_rates(VoidingConfig(enabled=True),temperature_K=900.,stress_tensor_Pa=np.eye(2)*1e9)
                limited={**rates,spec["channel"]:rates[spec["channel"]]*1e-3}
                area=math.pi*(2.5e-5)**2
                cavity=Cavity2D("controlled","site-1",(0.,0.),2.5e-5,area,area,VoidPhase.STABLE_SUBGRID_VOID)
                final=grow_cavity_from_rate(cavity,rates=limited,dt_s=1e-6,radial_growth_scale_m=1e-9)
                row.update({"trace":["source_native_rates",spec["channel"],"series_growth"],"terminal":asdict(final),
                            "terminal_fingerprint":digest(asdict(final)),"passed":final.radius_m>cavity.radius_m})
            elif spec["kind"]=="healing":
                embryo,rates,_=lifecycle_to_embryo(8200); site=embryo.void_state.sites[0]
                heal={**rates,"stabilization_s":0.}; void,events=advance_site(embryo.void_state,"site-1",site.healing.threshold/heal["healing_s"],rates=heal)
                row.update({"trace":events,"terminal":asdict(void.sites[0]),"terminal_fingerprint":digest(asdict(void)),
                            "passed":void.sites[0].phase==VoidPhase.HEALED_SITE})
            else:
                pre,trace=deterministic_trajectory(stop_before_ligament=True,cavity_center_m=spec["center"]); connected,_=ligament_transaction(pre)
                delayed,audit=_complete_next_clock(connected,np.zeros((2,2)),source_kind="cavity_surface",
                  source_cavity_id=connected.void_state.cavities[0].cavity_id,source_boundary_site_id="connection_exit",
                  source_position_m=connected.void_state.cavities[0].connection_exit_m,
                  source_probe_identity={"kind":"prospective_zero_drive"},maximum_advance_duration_s=5.)
                row.update({"trace":[x["operation"] for x in trace]+["zero_drive_delay"],"terminal":observables(delayed,name),
                  "terminal_fingerprint":complete_accepted_state_fingerprint(delayed),"clock_status":delayed.junction_process_state["latest_directional_clock_status"],
                  "passed":not any(x["winner"] for x in audit)})
        except Exception as error: row.update({"passed":False,"failure":f"{type(error).__name__}: {error}"})
        rows.append(row)
    return rows

def restart_rows(out):
    trace=[]; deterministic_trajectory(state_trace=trace)
    captured=dict(trace)
    sources={"available_site":captured["available_site"],"between_birth_hits":captured["multi_hit_1"],
      "embryo":captured["multi_hit_2"],"stable_subgrid_cavity":captured["stabilization"],
      "before_promotion":captured["subgrid_growth"],"after_promotion":captured["geometric_promotion"],
      "before_ligament":captured["resolved_growth"],"connected_before_downstream":captured["ligament_rupture"],
      "downstream_front_before_continuation":captured["new_graph_front"]}
    initial=sources["available_site"]; cfg=VoidingConfig(enabled=True,promotion_radius_m=5e-5)
    tensor=local_site_tensor(initial); rates=arrhenius_rates(cfg,temperature_K=900.,stress_tensor_Pa=tensor)
    site=initial.void_state.sites[0]; half=0.5*site.birth.threshold/(rates["birth_s"]*site.candidate_weight)
    void,_=advance_site(initial.void_state,"site-1",half,rates=rates)
    sources["incomplete_first_hit"]=equilibrate_fixed_load_with_production_fem(replace(initial,void_state=void))
    def resume(stage,state):
        if stage in ("available_site","incomplete_first_hit","between_birth_hits"):
            rates=arrhenius_rates(cfg,temperature_K=900.,stress_tensor_Pa=local_site_tensor(state)); site=state.void_state.sites[0]
            dt=max(site.birth.threshold-site.birth.accumulated,0.)/(rates["birth_s"]*site.candidate_weight)
            void,_=advance_site(state.void_state,"site-1",dt,rates=rates); return equilibrate_fixed_load_with_production_fem(replace(state,void_state=void))
        if stage=="embryo":
            rates=arrhenius_rates(cfg,temperature_K=900.,stress_tensor_Pa=local_site_tensor(state)); site=state.void_state.sites[0]
            void,_=advance_site(state.void_state,"site-1",site.stabilization.threshold/rates["stabilization_s"],rates=rates)
            return equilibrate_fixed_load_with_production_fem(replace(state,void_state=void))
        if stage=="stable_subgrid_cavity":
            void=create_subgrid_cavity(state.void_state,"site-1",2.5e-5); return equilibrate_fixed_load_with_production_fem(replace(state,void_state=void))
        if stage=="before_promotion":
            cavity=state.void_state.cavities[0]; return replace(state,void_state=promote_cavity(state.void_state,cavity.cavity_id,cfg.promotion_radius_m))
        if stage=="after_promotion":
            rates=arrhenius_rates(cfg,temperature_K=900.,stress_tensor_Pa=cavity_boundary_tensor(state)[0]); cavity=state.void_state.cavities[0]
            void=update_cavity_growth(state.void_state,cavity.cavity_id,rates=rates,dt_s=1e-9,radial_growth_scale_m=cfg.radial_growth_scale_m)
            return equilibrate_fixed_load_with_production_fem(replace(state,void_state=void))
        if stage=="before_ligament": return ligament_transaction(state)[0]
        if stage=="connected_before_downstream": return downstream_front_transaction(state)[0]
        if stage=="downstream_front_before_continuation": return downstream_front_transaction(state,continuation=True)[0]
        raise ValueError(stage)
    rows=[]
    for index,stage in enumerate(RESTART_STAGES):
        if stage not in sources:
            rows.append({"execution_id":f"restart-{index:02d}","stage":stage,"passed":False,"classification":"STAGE_CAPTURE_NOT_IMPLEMENTED"}); continue
        state=sources[stage]; path=out/"checkpoints"/(stage+".json"); write_checkpoint(state,path); restored=restore_checkpoint(path)
        uninterrupted=resume(stage,state); restarted=resume(stage,restored)
        same=complete_accepted_state_fingerprint(uninterrupted)==complete_accepted_state_fingerprint(restarted)
        rows.append({"execution_id":f"restart-{index:02d}","stage":stage,"checkpoint_fingerprint":complete_accepted_state_fingerprint(state),
          "restored_checkpoint_fingerprint":complete_accepted_state_fingerprint(restored),
          "uninterrupted_fingerprint":complete_accepted_state_fingerprint(uninterrupted),
          "restored_fingerprint":complete_accepted_state_fingerprint(restarted),"passed":same,
          "classification":"STAGE_SPECIFIC_CONTINUATION_EXACT"})
    return rows

def natural_rows():
    rows=[]
    for index in range(32):
        seed=12000+index
        try:
            final,trace=natural_trajectory(seed=seed,steps=16); site=final.void_state.sites[0]
            classification="NO_BIRTH_WITHIN_WINDOW" if site.hits==0 else ("INCOMPLETE_MULTI_HIT" if site.phase==VoidPhase.AVAILABLE_SITE else site.phase.value)
            rows.append({"execution_id":f"natural-{index:02d}-{seed}","seed":seed,"steps":16,"terminal_classification":classification,
              "event_count":sum(len(x["events"]) for x in trace),"cavity_count":len(final.void_state.cavities),
              "maximum_radius_m":max((c.radius_m for c in final.void_state.cavities),default=0.),
              "terminal_fingerprint":complete_accepted_state_fingerprint(final),"passed":True})
        except Exception as error: rows.append({"execution_id":f"natural-{index:02d}-{seed}","seed":seed,"passed":False,
          "terminal_classification":"NUMERICAL_FAILURE","failure":f"{type(error).__name__}: {error}"})
    return rows

def run_static(out):
    target=out/"static_mechanics"; completed=subprocess.run([sys.executable,str(ROOT/"scripts/qualify_crack_void_static_v5.py"),str(target)],cwd=ROOT,text=True,capture_output=True)
    payload=json.loads((target/"case_rows.json").read_text()) if (target/"case_rows.json").exists() else {}
    rows=payload.get("rows",[])
    return {"execution_id":"static-matrix-"+digest([row.get("case") for row in rows])[:12],"returncode":completed.returncode,
            "case_count":len(rows),"all_passed":bool(rows) and all(row.get("passed") for row in rows),"stdout":completed.stdout,"stderr":completed.stderr}

def neutrality_rows(out,base_worktree,head_worktree):
    if not base_worktree or not head_worktree:
        return [{"execution_id":f"neutrality-{name}","trajectory":name,"passed":False,
          "classification":"CLEAN_WORKTREE_PATHS_NOT_SUPPLIED"} for name in ("monotonic","oblique","checkpoint_restart","unload_reload")]
    probe=ROOT/"scripts/run_v5_disabled_neutrality_probe.py"; outputs=[]
    for label,worktree in (("base",Path(base_worktree)),("head",Path(head_worktree))):
        target=out/"neutrality_raw"/(label+".json")
        completed=subprocess.run([sys.executable,str(probe),str(target)],cwd=worktree,text=True,capture_output=True)
        if completed.returncode: raise RuntimeError(f"neutrality {label} failed: {completed.stderr}")
        outputs.append(json.loads(target.read_text()))
    base_rows={row["trajectory"]:row for row in outputs[0]["rows"]}; head_rows={row["trajectory"]:row for row in outputs[1]["rows"]}
    rows=[]
    for name in ("monotonic","oblique","checkpoint_restart","unload_reload"):
        equal=base_rows[name]==head_rows[name]
        rows.append({"execution_id":f"neutrality-{name}","trajectory":name,"base_measurement":base_rows[name],
          "v5_disabled_measurement":head_rows[name],"passed":equal,"classification":"EXACT_PHYSICAL_EQUALITY" if equal else "PHYSICAL_DIFFERENCE"})
    return rows

def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("out",nargs="?",default=str(ROOT/"artifacts/voiding_v5_finalization_v2"))
    parser.add_argument("--implementation-sha",default=os.environ.get("VOIDING_V5_IMPLEMENTATION_SHA"))
    parser.add_argument("--neutrality-base-worktree",default=os.environ.get("V5_NEUTRALITY_BASE_WORKTREE"))
    parser.add_argument("--neutrality-head-worktree",default=os.environ.get("V5_NEUTRALITY_HEAD_WORKTREE")); args=parser.parse_args(argv)
    out=Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=True); head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    transitions=transition_partition_rows(); controlled=controlled_rows(); restarts=restart_rows(out); natural=natural_rows(); static=run_static(out)
    conservation=[]
    for row in controlled:
        terminal=row.get("terminal",{}); ledgers=terminal.get("length_ledgers")
        if not ledgers: continue
        physical_error=abs(ledgers["physical_active_front_travel_m"]-ledgers["fractured_ligament_length_m"]
          -ledgers["ordinary_crack_fractured_length_m"]-ledgers["traversed_void_free_span_m"])
        projected_error=abs(ledgers["projected_front_advance_m"]-ledgers["projected_fractured_length_m"]
          -ledgers["projected_free_span_m"])
        inventory_error=abs(terminal["consumed_defect_inventory_area_m2"]-terminal["cavity_area_m2"])
        conservation.append({"execution_id":"conservation-"+row["execution_id"],"source_execution_id":row["execution_id"],
          "physical_length_identity_error_m":physical_error,"projected_length_identity_error_m":projected_error,
          "inventory_identity_error_m2":inventory_error,"tolerance_m":1e-15,"tolerance_m2":1e-24,
          "raw_length_ledgers":ledgers,"cavity_area_m2":terminal["cavity_area_m2"],
          "consumed_inventory_area_m2":terminal["consumed_defect_inventory_area_m2"],
          "passed":physical_error<=1e-15 and projected_error<=1e-15 and inventory_error<=1e-24})
    rollback=[]; pre,_=deterministic_trajectory(stop_before_ligament=True)
    for index,stage in enumerate(("graph_edit","remesh","field_projection","support_rebuild","equilibrium","energy_gate","process_state_update",
      "topology_verification","late_event_veto","intersection_alignment","connected_surface_certification")):
        before=complete_accepted_state_fingerprint(pre); message=""
        try: ligament_transaction(pre,failure_stage=stage)
        except RuntimeError as error: message=str(error)
        rollback.append({"execution_id":f"rollback-{index:02d}","stage":stage,"exception":message,
          "restored_exactly":before==complete_accepted_state_fingerprint(pre),"passed":message=="injected:"+stage})
    neutrality=neutrality_rows(out,args.neutrality_base_worktree,args.neutrality_head_worktree)
    all_execution_ids=[x["execution_id"] for rows in (transitions,controlled,restarts,natural,rollback,neutrality,conservation) for x in rows]
    gates={"TRANSITION_PARTITIONS":all(x["passed"] and x["partition_equivalent"] for x in transitions),
      "STAGE_RESTART_CONTINUATION":all(x["passed"] and "CONTINUATION_NOT_EXECUTED" not in x["classification"] for x in restarts),
      "TWELVE_DISTINCT_CONTROLLED":len({x["input_hash"] for x in controlled})==12 and all(x["passed"] for x in controlled),
      "STATIC_MECHANICS":static["all_passed"],"NATURAL_32":len(natural)>=32 and all(x["passed"] for x in natural),
      "DISABLED_NEUTRALITY":all(x["passed"] for x in neutrality),"ATOMIC_ROLLBACK":all(x["passed"] for x in rollback),
      "EVIDENCE_ONTOLOGY":len(all_execution_ids)==len(set(all_execution_ids)),
      "LENGTH_INVENTORY_CONSERVATION":bool(conservation) and all(x["passed"] for x in conservation)}
    datasets={"transition_partitions":transitions,"controlled_trajectories":controlled,"restart_matrix":restarts,
      "natural_seed_ensemble":natural,"rollback_matrix":rollback,"disabled_neutrality":neutrality,
      "length_inventory_conservation":conservation}
    for name,rows in datasets.items(): write_json(out/(name+".json"),rows)
    write_json(out/"static_mechanics_summary.json",static)
    manifest={"schema":"v12.voiding-v5-finalization-v2/1","implementation_sha":args.implementation_sha or head,
      "counts":{name:len(rows) for name,rows in datasets.items()},"gates":gates,"decision":"PASS" if all(gates.values()) else "BLOCKED",
      "limitations":[key for key,value in gates.items() if not value]}
    write_json(out/"campaign_manifest.json",manifest)
    hashes={path.relative_to(out).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out.rglob("*"))
            if path.is_file() and path.name!="sha256_manifest.json"}; write_json(out/"sha256_manifest.json",hashes)
    print(json.dumps(manifest,indent=2,sort_keys=True)); return 0 if manifest["decision"]=="PASS" else 2

if __name__=="__main__": raise SystemExit(main())
