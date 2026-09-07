#!/usr/bin/env python3
"""Execute actual closure attempts and retain complete checkpoint-backed endpoints."""
from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from arrhenius_fracture.closure_lifecycle_evidence import (
    SCHEMA, PARTITIONS, PRECURSORS, CFG, advance_transition, resume_to_guard,
    conservation, transition_occurred, load_state, validate_lifecycle,
    lifecycle_decision,
)
from arrhenius_fracture.closure_mechanics_evidence import canonical_data
from arrhenius_fracture.finalization_v3_schema import FROZEN_CASE_REGISTRY, canonical_hash
from arrhenius_fracture.finalization_v3_closure_schema import validate_closure_evidence
from arrhenius_fracture.checkpoint_v11 import write_checkpoint, restore_checkpoint
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint
from arrhenius_fracture.voiding_production_v5 import (
    deterministic_trajectory, build_production_void_state, local_site_tensor, cavity_boundary_tensor,
    _complete_next_clock, _geometry, remesh_cavity, ligament_transaction,
    advance_disabled_v5_stage2,
)
from arrhenius_fracture.voiding_v5 import advance_site, arrhenius_rates, VoidPhase, update_cavity_growth
from arrhenius_fracture.sharp_wake_backend_v12 import V12_MODEL_ID
from arrhenius_fracture.v12_production_driver import build_loaded_state, execute_event


def write_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(canonical_data(payload),sort_keys=True,indent=2,allow_nan=False)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("output",type=Path)
    args=parser.parse_args(); out=args.output
    if out.exists() and any(out.iterdir()): raise ValueError("refusing to overwrite evidence")
    if subprocess.check_output(("git","status","--porcelain"),cwd=ROOT,text=True).strip():
        raise RuntimeError("lifecycle evidence requires a clean committed implementation")
    sha=subprocess.check_output(("git","rev-parse","HEAD"),cwd=ROOT,text=True).strip()
    rows=[]; trace=[]
    terminal,_=deterministic_trajectory(state_trace=trace)
    captured=dict(trace); available=captured["available_site"]
    def checkpoint(state):
        relative="checkpoints/"+fingerprint(state)+".json"
        if not (out/relative).exists(): write_checkpoint(state,out/relative)
        return relative
    def record(dataset,case,before,after,configuration,operations,**extra):
        cfg=canonical_data(configuration)
        row={"dataset":dataset,"case_identity":case,"partition_count":cfg.get("partition_count"),
             "execution_id":f"{dataset}:{case}:{cfg.get('partition_count')}:{len(rows)}:{sha}",
             "executed_code_sha":sha,"input_configuration":cfg,"input_hash":canonical_hash(cfg),
             "initial_checkpoint":checkpoint(before),"terminal_checkpoint":checkpoint(after),
             "initial_fingerprint":fingerprint(before),"terminal_fingerprint":fingerprint(after),
             "actual_operations":canonical_data(operations),"conservation":conservation(after,before),**extra}
        rows.append(row); write_json(out/"rows"/(str(len(rows))+".json"),row)
        return row
    for name in FROZEN_CASE_REGISTRY["transitions"]:
        for partitions in PARTITIONS:
            print(f"Actual transition attempt {name}/{partitions}",flush=True)
            precursor=PRECURSORS[name]; before=captured.get(precursor,terminal)
            initial_capture=restore_checkpoint(out/checkpoint(before))
            after=before; operations=[]; error=None
            try: after,_=advance_transition(before,name,partitions,operations=operations)
            except Exception as exc: error={"type":type(exc).__name__,"message":str(exc)}
            record("transitions",name,initial_capture,after,{"partition_count":partitions,"requested_predecessor":precursor,
                "predecessor_available":precursor in captured,"fixed_opening_m":4e-7,"seed":3621},operations,
                transition_occurred=transition_occurred(name,initial_capture,after),failure=error)
    # Each restart continues, not just one step, to the attainable blocked
    # terminal. That reproduction is separate from the required continued-front gate.
    restart_labels={"available_site":"available_site","between_birth_hits":"multi_hit_1","embryo":"multi_hit_2",
        "stable_subgrid_cavity":"subgrid_void","before_promotion":"subgrid_growth","after_promotion":"geometric_promotion",
        "before_ligament":"resolved_growth","connected_before_downstream":"ligament_rupture",
        "downstream_child_before_continuation":"new_graph_front","zero_drive_connected":"ligament_rupture"}
    for stage in FROZEN_CASE_REGISTRY["restarts"]:
        print("Actual restart attempt "+stage,flush=True)
        before=captured.get(restart_labels.get(stage),available if stage=="incomplete_first_hit" else terminal)
        stage_available=restart_labels.get(stage) in captured or stage=="incomplete_first_hit"
        if stage=="incomplete_first_hit":
            rates=arrhenius_rates(CFG,temperature_K=900.,stress_tensor_Pa=local_site_tensor(before))
            site=before.void_state.sites[0]; dt=.5*site.birth.crossing_time(rates["birth_s"]*site.candidate_weight)
            voids,_=advance_site(before.void_state,site.site_id,dt,rates=rates); before=replace(before,void_state=voids)
        if stage=="zero_drive_connected": before=load_state(before,0.)
        restored=restore_checkpoint(out/checkpoint(before)); operations=[]; resumed_ops=[]; after=before; resumed=restored; error=None
        try:
            after=resume_to_guard(before,operations); resumed=resume_to_guard(restored,resumed_ops)
        except Exception as exc: error={"type":type(exc).__name__,"message":str(exc)}
        replay_path=checkpoint(resumed)
        record("restarts",stage,before,after,{"expected_terminal":"DOWNSTREAM_FRONT_CONTINUED",
            "requested_stage_available":stage_available,"reload_policy":"retain_accepted_load"},operations,
            restored_terminal_checkpoint=replay_path,restart_exact=fingerprint(after)==fingerprint(resumed),
            continued_front_terminal_reached=after.void_state.cavities and after.void_state.cavities[0].phase==VoidPhase.DOWNSTREAM_FRONT_ACTIVE,
            failure=error)
    # All controlled rows use actual FEM loading and state updates. Limiter
    # cases never substitute detached rate calculations for a growth history.
    specs={"centered":((7e-4,0.),4e-7),"positive_offset":((7e-4,1e-5),4e-7),
        "negative_offset":((7e-4,-1e-5),4e-7),"short_ligament":((6.75e-4,0.),4e-7),
        "long_ligament":((7.35e-4,0.),4e-7),"diffusion_limited":((7e-4,0.),6e-7),
        "accommodation_limited":((7e-4,0.),1e-7),"embryo_healing":((7e-4,0.),-4e-7),
        "downstream_zero_drive":((7e-4,-1e-5),0.),"delayed_downstream":((7e-4,-1e-5),8e-7),
        "fixed_mesh_oblique":((7e-4,0.),4e-7),"local_remesh_refinement":((7e-4,0.),4e-7)}
    for case in FROZEN_CASE_REGISTRY["controlled"]:
        print("Actual controlled history "+case,flush=True)
        center,opening=specs[case]; local_trace=[]; ops=[]; before=available; after=available; error=None
        path=((0.,0.),(.0005725993004046688,0.))
        if case=="fixed_mesh_oblique":
            direction=np.asarray((np.cos(np.pi/6),np.sin(np.pi/6)))
            tip=np.asarray(center)-1.5e-4*direction
            path=((0.,float(tip[1]-tip[0]*direction[1]/direction[0])),tuple(map(float,tip)))
        try:
            angle=30. if case=="fixed_mesh_oblique" else 0.
            before,_=deterministic_trajectory(stop_before_ligament=True,cavity_center_m=center,
                crack_path_m=path,cleavage_theta_deg=angle,state_trace=local_trace)
            if case=="embryo_healing":
                before=dict(local_trace)["multi_hit_2"]; before=load_state(before,opening)
                after,_=advance_transition(before,"healing",1,operations=ops)
            elif case in ("diffusion_limited","accommodation_limited"):
                before=load_state(dict(local_trace)["subgrid_void"],opening); after=before
                for _ in range(8):
                    rates=arrhenius_rates(CFG,temperature_K=900.,stress_tensor_Pa=local_site_tensor(after))
                    voids=update_cavity_growth(after.void_state,after.void_state.cavities[0].cavity_id,
                        rates=rates,dt_s=1e-10,radial_growth_scale_m=CFG.radial_growth_scale_m)
                    after=replace(after,void_state=voids)
                    after=load_state(after,opening)
                    ops.append({"api":"accepted_load_growth_interval","duration_s":1e-10,"opening_m":opening,"rates":rates})
            else:
                after=before
                if case=="local_remesh_refinement":
                    cavity=after.void_state.cavities[0]
                    hole,_=_geometry(radius_m=cavity.radius_m,center_m=cavity.center_m,boundary_segments=64,radial_layers=24)
                    from arrhenius_fracture.voiding_production_v5 import _grow_hole_boundary
                    hole=_grow_hole_boundary(hole,cavity.radius_m,crack_path_m=after.crack_network.branches[0].path)
                    audit=[]; after=remesh_cavity(after,hole,after.void_state,"controlled-refinement",audit)
                    ops.append({"api":"actual_local_remesh","operations":audit})
                elif case in ("downstream_zero_drive","delayed_downstream"):
                    after,_=advance_transition(after,"ligament",1,operations=ops)
                    after=load_state(after,0.); after,_=advance_transition(after,"downstream_child",1,operations=ops)
                    if case=="delayed_downstream":
                        after=load_state(after,opening); after,_=advance_transition(after,"downstream_child",1,operations=ops)
                else: after=resume_to_guard(after,ops)
        except Exception as exc: error={"type":type(exc).__name__,"message":str(exc)}
        record("controlled",case,before,after,{"center_m":center,"loading_opening_m":opening,
            "fixed_crack_path_m":path,"history_kind":case},ops,
            actual_preparation_stages=[name for name,_ in local_trace],
            actual_preparation_checkpoints={name:checkpoint(state) for name,state in local_trace},failure=error)
    for case in FROZEN_CASE_REGISTRY["neutrality"]:
        print("Actual V12 versus disabled V5 "+case,flush=True)
        base=build_loaded_state(V12_MODEL_ID); disabled=build_loaded_state(V12_MODEL_ID)
        before=disabled; ops=[]; error=None
        if case=="checkpoint_restart": disabled=restore_checkpoint(out/checkpoint(disabled))
        try:
            if case=="unload_reload":
                for opening in (0.,4e-7): base=load_state(base,opening); disabled=load_state(disabled,opening)
                ops.append({"api":"V12_and_disabled_accepted_unload_reload"})
            else:
                end=(5.25e-4,2.5e-5) if case=="fixed_mesh_oblique" else (5.25e-4,0.)
                base,event_a=execute_event(base,end,transaction_identity="closure-neutrality:"+case)
                disabled,event_b=advance_disabled_v5_stage2(disabled,end,transaction_identity="closure-neutrality:"+case)
                ops.append({"api":"V12_execute_event_vs_disabled_V5_dispatch","base_event":event_a,"disabled_event":event_b})
        except Exception as exc: error={"type":type(exc).__name__,"message":str(exc)}
        record("neutrality",case,before,disabled,{"base_model":V12_MODEL_ID,"v5_model":V12_MODEL_ID,"voiding_enabled":False},ops,
            base_terminal_checkpoint=checkpoint(base),exact_neutrality=fingerprint(base)==fingerprint(disabled),failure=error)
    # Natural seeds: same physical duration, actual source-native stress, RNG,
    # and midpoint restart under all five timestep partitions.
    for seed in range(12000,12032):
        for partitions in PARTITIONS:
            before,_=build_production_void_state(stochastic=True,seed=seed)
            after=before; ops=[]; restart_exact=True
            for interval in range(16*partitions):
                rates=arrhenius_rates(CFG,temperature_K=900.,stress_tensor_Pa=local_site_tensor(after))
                voids,events=advance_site(after.void_state,"site-1",1e-12/partitions,rates=rates)
                after=replace(after,void_state=voids)
                ops.append({"api":"advance_site","duration_s":1e-12/partitions,"events":events,"rates":rates})
                if interval==8*partitions-1:
                    restored=restore_checkpoint(out/checkpoint(after)); restart_exact &= fingerprint(restored)==fingerprint(after)
                    replay=restored
                elif interval>=8*partitions:
                    replay_rates=arrhenius_rates(CFG,temperature_K=900.,stress_tensor_Pa=local_site_tensor(replay))
                    replay_voids,_=advance_site(replay.void_state,"site-1",1e-12/partitions,rates=replay_rates)
                    replay=replace(replay,void_state=replay_voids)
            restart_exact &= fingerprint(after)==fingerprint(replay)
            record("natural",str(seed),before,after,{"seed":seed,"partition_count":partitions,"duration_s":16e-12},ops,
                midpoint_restart_exact=restart_exact,restarted_terminal_checkpoint=checkpoint(replay))
    # Real rollback injections for stages reachable without unqualified events.
    for stage in ("field_projection","support_rebuild","equilibrium"):
        before=captured["subgrid_growth"]; after=before; ops=[]; error=None
        initial_capture=restore_checkpoint(out/checkpoint(before))
        cavity=before.void_state.cavities[0]; hole,_=_geometry(radius_m=cavity.radius_m,center_m=cavity.center_m)
        try: after=remesh_cavity(before,hole,before.void_state,"rollback-promotion",ops,failure_stage=stage)
        except Exception as exc: error={"type":type(exc).__name__,"message":str(exc)}
        record("rollback","promotion:"+stage,initial_capture,after,{"failure_stage":stage},ops,failure=error,
            restored_exactly=fingerprint(initial_capture)==fingerprint(after))
    for stage in ("graph_edit","remesh","field_projection","support_rebuild","equilibrium","energy_gate",
                  "connected_surface_certification","dormant_support_rebuild"):
        before=captured["resolved_growth"]; after=before; ops=[]; error=None
        initial_capture=restore_checkpoint(out/checkpoint(before))
        try: after,_=ligament_transaction(before,failure_stage=stage,operation_log=ops)
        except Exception as exc: error={"type":type(exc).__name__,"message":str(exc)}
        record("rollback","ligament:"+stage,initial_capture,after,{"failure_stage":stage},ops,failure=error,
            restored_exactly=fingerprint(initial_capture)==fingerprint(after))
    class Sources:
        def __getitem__(self,key): return restore_checkpoint(out/key)
    payload={"schema":SCHEMA,"executed_code_sha":sha,"rows":rows,
             "decision":lifecycle_decision(rows,Sources())}
    validation=validate_closure_evidence(payload,Sources(),executed_code_sha=sha)
    write_json(out/"lifecycle_rows.json",payload); write_json(out/"ontology_validation.json",validation)
    write_json(out/"sha256_manifest.json",{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob("*")) if p.is_file()})
    print(json.dumps(validation),flush=True)


if __name__=="__main__":main()
