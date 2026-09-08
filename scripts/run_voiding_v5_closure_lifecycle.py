#!/usr/bin/env python3
"""Execute actual closure attempts and retain complete checkpoint-backed endpoints."""
from dataclasses import replace
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from arrhenius_fracture.closure_lifecycle_evidence import (
    SCHEMA, PARTITIONS, PRECURSORS, CFG, advance_transition, resume_to_guard,
    conservation, transition_occurred, load_state, validate_lifecycle,
    lifecycle_decision,stagewise_topology,natural_terminal_measurements,build_healing_predecessor,CONTROLLED_HEALING_SEED,
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
from arrhenius_fracture.voiding_lifecycle_driver_v5 import (
    advance_production_void_interval,NATURAL_WINDOW_S,NATURAL_SEEDS,
)
from arrhenius_fracture.closure_rollback_matrix_v5 import rollback_attempts


def write_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(canonical_data(payload),sort_keys=True,indent=2,allow_nan=False)+"\n")


class ProgressTrace(list):
    def append(self,item):
        print('Captured accepted lifecycle stage '+item[0],flush=True)
        super().append(item)


def resume_attempt(state,operations,*,common_restart_protocol,legacy_reload=False):
    """Capture one independently executed continuation, including accepted progress."""
    try:
        if legacy_reload:
            state=load_state(state,8e-7)
            operations.append({'api':'accepted_tensile_reload','opening_m':8e-7})
        return resume_to_guard(state,operations,common_restart_protocol=common_restart_protocol),None
    except Exception as exc:
        return getattr(exc,'accepted_lifecycle_state',state),{'type':type(exc).__name__,'message':str(exc)}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("output",type=Path)
    parser.add_argument('--qualified-fine-history',action='store_true',
        help='Execute source-qualified 512/192 transition/restart histories; natural window/seed registry stays unchanged')
    parser.add_argument('--phases-7-8-only',action='store_true',
        help='Development execution of all transitions, restarts and rollback; not a full closure campaign')
    parser.add_argument('--section',choices=('all','transitions','restarts','controlled','neutrality','natural','rollback'),default='all')
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--shard-count',type=int,default=1)
    parser.add_argument('--common-restart-protocol',choices=('v1','v2'),default='v2')
    args=parser.parse_args(); out=args.output
    if not 0<=args.shard_index<args.shard_count:raise ValueError('invalid lifecycle shard')
    if args.shard_count>1 and args.section in ('all','rollback','neutrality'):
        raise ValueError('select a case-shardable lifecycle section')
    if args.phases_7_8_only and args.section!='all':raise ValueError('development subset and section are exclusive')
    def selected(section,registry):
        if args.section not in ('all',section):return ()
        if args.phases_7_8_only and section not in ('transitions','restarts','rollback'):return ()
        return tuple(case for i,case in enumerate(registry) if i%args.shard_count==args.shard_index)
    if out.exists() and any(out.iterdir()): raise ValueError("refusing to overwrite evidence")
    if subprocess.check_output(("git","status","--porcelain"),cwd=ROOT,text=True).strip():
        raise RuntimeError("lifecycle evidence requires a clean committed implementation")
    sha=subprocess.check_output(("git","rev-parse","HEAD"),cwd=ROOT,text=True).strip()
    rows=[]; trace=ProgressTrace()
    fine_options = dict(boundary_segments=512,radial_layers=192,
        crack_path_m=((0.,0.),(.0005725993004046688,0.)),qualify_source=True) if args.qualified_fine_history and args.section not in ('natural','neutrality','controlled') else {}
    if args.qualified_fine_history and args.section=='restarts':fine_options['common_restart_protocol']=args.common_restart_protocol
    preparation_failure=None
    try:
        terminal,trajectory_history=deterministic_trajectory(state_trace=trace,**fine_options)
    except Exception as exc:
        if not trace: raise
        terminal=trace[-1][1]; trajectory_history=[]
        preparation_failure={'type':type(exc).__name__,'message':str(exc)}
    captured=dict(trace); available=captured["available_site"]
    captured['healing_peer'],healing_preparation=build_healing_predecessor()
    def checkpoint(state):
        relative="checkpoints/"+fingerprint(state)+".json"
        if not (out/relative).exists(): write_checkpoint(state,out/relative,compression='gzip' if args.qualified_fine_history else None)
        return relative
    def record(dataset,case,before,after,configuration,operations,**extra):
        cfg=canonical_data(configuration)
        row={"dataset":dataset,"case_identity":case,"partition_count":cfg.get("partition_count"),
             "execution_id":f"{dataset}:{case}:{cfg.get('partition_count')}:{len(rows)}:{sha}",
             "executed_code_sha":sha,"input_configuration":cfg,"input_hash":canonical_hash(cfg),
             "initial_checkpoint":checkpoint(before),"terminal_checkpoint":checkpoint(after),
             "initial_fingerprint":fingerprint(before),"terminal_fingerprint":fingerprint(after),
             "actual_operations":canonical_data(operations),"conservation":conservation(after,before),
             'stagewise_topology':stagewise_topology(after),**canonical_data(extra)}
        rows.append(row); write_json(out/"rows"/(str(len(rows))+".json"),row)
        return row
    write_json(out/'preparation.json',{'executed_code_sha':sha,'failure':preparation_failure,
        'accepted_stage_checkpoints':{name:checkpoint(state) for name,state in trace},
        'actual_trajectory_history':trajectory_history})
    for index,((previous_name,before),(name,after)) in enumerate(zip(trace,trace[1:])):
        if args.section not in ('all','transitions') or args.shard_index!=0:continue
        record('stagewise',name,before,after,{'stage_index':index,'previous_stage':previous_name,
            'measurement_kind':'DERIVED_ACCEPTED_TRAJECTORY_STAGE_NOT_NEW_BASE_EXECUTION'},
            [{'api':'deterministic_trajectory','captured_stage':name,'actual_trajectory_history':trajectory_history}])
    for name in selected('transitions',FROZEN_CASE_REGISTRY["transitions"]):
        for partitions in PARTITIONS:
            print(f"Actual transition attempt {name}/{partitions}",flush=True)
            precursor='source_resolution_attempt' if args.qualified_fine_history and name=='downstream_child' else PRECURSORS[name]
            before=captured.get(precursor,terminal)
            initial_capture=restore_checkpoint(out/checkpoint(before))
            after=before; operations=[]; error=None
            try: after,_=advance_transition(before,name,partitions,operations=operations)
            except Exception as exc: error={"type":type(exc).__name__,"message":str(exc)}
            record("transitions",name,initial_capture,after,{"partition_count":partitions,"requested_predecessor":precursor,
                "predecessor_available":precursor in captured,"fixed_opening_m":-4e-7 if name=='healing' else 4e-7,
                "seed":CONTROLLED_HEALING_SEED if name=='healing' else 3621},operations,
                predecessor_preparation_operations=healing_preparation if name=='healing' else [],
                transition_occurred=transition_occurred(name,initial_capture,after),failure=error)
    # Each restart continues, not just one step, to the attainable blocked
    # terminal. That reproduction is separate from the required continued-front gate.
    restart_labels={"available_site":"available_site","between_birth_hits":"multi_hit_1","embryo":"multi_hit_2",
        "stable_subgrid_cavity":"subgrid_void","before_promotion":"subgrid_growth","after_promotion":"geometric_promotion",
        "before_ligament":"resolved_growth","connected_before_downstream":"ligament_rupture",
        "downstream_child_before_continuation":"new_graph_front","zero_drive_connected":"ligament_rupture"}
    restart_captured=captured;restart_terminal=terminal
    if args.qualified_fine_history and selected('restarts',FROZEN_CASE_REGISTRY['restarts']):
        restart_labels['zero_drive_connected']='zero_drive_connected'
        if args.section!='restarts':
            restart_trace=ProgressTrace()
            try:
                restart_terminal,_=deterministic_trajectory(state_trace=restart_trace,
                    boundary_segments=512,radial_layers=192,crack_path_m=((0.,0.),(.0005725993004046688,0.)),
                    qualify_source=True,common_restart_protocol=args.common_restart_protocol)
            except Exception as exc:
                if not restart_trace:raise
                restart_terminal=restart_trace[-1][1]
                print('Common restart reference rejected: '+type(exc).__name__+': '+str(exc),flush=True)
            restart_captured=dict(restart_trace)
    for stage in selected('restarts',FROZEN_CASE_REGISTRY["restarts"]):
        print("Actual restart attempt "+stage,flush=True)
        before=restart_captured.get(restart_labels.get(stage),restart_captured['available_site'] if stage=="incomplete_first_hit" else restart_terminal)
        stage_available=restart_labels.get(stage) in restart_captured or stage=="incomplete_first_hit"
        if stage=="incomplete_first_hit":
            rates=arrhenius_rates(CFG,temperature_K=900.,stress_tensor_Pa=local_site_tensor(before))
            site=before.void_state.sites[0]; dt=site.birth.crossing_time_exact(rates["birth_s"]*site.candidate_weight)/2
            voids,_=advance_site(before.void_state,site.site_id,dt,rates=rates); before=replace(before,void_state=voids)
        if stage=="zero_drive_connected" and not args.qualified_fine_history: before=load_state(before,0.)
        restored=restore_checkpoint(out/checkpoint(before)); operations=[]; resumed_ops=[]; after=before; resumed=restored; error=None
        after,direct_failure=resume_attempt(after,operations,common_restart_protocol=args.common_restart_protocol if args.qualified_fine_history else False,
            legacy_reload=stage=='zero_drive_connected' and not args.qualified_fine_history)
        resumed,replay_failure=resume_attempt(resumed,resumed_ops,common_restart_protocol=args.common_restart_protocol if args.qualified_fine_history else False,
            legacy_reload=stage=='zero_drive_connected' and not args.qualified_fine_history)
        error=direct_failure or replay_failure
        replay_path=checkpoint(resumed)
        record("restarts",stage,before,after,{"expected_terminal":"DOWNSTREAM_FRONT_CONTINUED",
            "requested_stage_available":stage_available,
            'common_terminal_protocol':'v5.common-terminal-restart-load/'+args.common_restart_protocol[1:] if args.qualified_fine_history else 'RETAINED_LEGACY_DIFFERENT_LOAD_HISTORIES',
            "reload_policy":("common_4e-7_compressive_minus4e-7_zero_drive_16us_"+('4e-7' if args.common_restart_protocol=='v2' else '8e-7')) if args.qualified_fine_history else "tensile_8e-7" if stage=='zero_drive_connected' else "retain_accepted_load"},operations,
            restored_terminal_checkpoint=replay_path,restart_exact=fingerprint(after)==fingerprint(resumed),
            restarted_operations=resumed_ops,subsequent_history_exact=canonical_data(operations)==canonical_data(resumed_ops),
            continued_front_terminal_reached=stage_available and any(op.get('api')=='child_tip_continuation'
                and op.get('accepted',False) for op in operations),
            failure=error,direct_failure=direct_failure,replay_failure=replay_failure)
    # All controlled rows use actual FEM loading and state updates. Limiter
    # cases never substitute detached rate calculations for a growth history.
    specs={"centered":((7e-4,0.),4e-7),"positive_offset":((7e-4,1e-5),4e-7),
        "negative_offset":((7e-4,-1e-5),4e-7),"short_ligament":((6.75e-4,0.),4e-7),
        "long_ligament":((7.35e-4,0.),4e-7),"diffusion_limited":((7e-4,0.),6e-7),
        "accommodation_limited":((7e-4,0.),1e-7),"embryo_healing":((7e-4,0.),-4e-7),
        "downstream_zero_drive":((7e-4,-1e-5),0.),"delayed_downstream":((7e-4,-1e-5),8e-7),
        "fixed_mesh_oblique":((7e-4,0.),4e-7),"local_remesh_refinement":((7e-4,0.),4e-7)}
    for case in selected('controlled',FROZEN_CASE_REGISTRY["controlled"]):
        print("Actual controlled history "+case,flush=True)
        center,opening=specs[case]; local_trace=[]; ops=[]; before=available; after=available; error=None
        path=((0.,0.),(.0005725993004046688,0.))
        if case=="fixed_mesh_oblique":
            direction=np.asarray((np.cos(np.pi/6),np.sin(np.pi/6)))
            tip=np.asarray(center)-1.5e-4*direction
            path=((0.,float(tip[1]-tip[0]*direction[1]/direction[0])),tuple(map(float,tip)))
        try:
            angle=30. if case=="fixed_mesh_oblique" else 0.
            if case!='embryo_healing':
                before,_=deterministic_trajectory(stop_before_ligament=True,cavity_center_m=center,
                    crack_path_m=path,cleavage_theta_deg=angle,state_trace=local_trace,
                    boundary_segments=512 if args.qualified_fine_history and case!='local_remesh_refinement' else 32,
                    radial_layers=192 if args.qualified_fine_history and case!='local_remesh_refinement' else 12)
            if case=="embryo_healing":
                before,preparation=build_healing_predecessor();ops.extend(preparation)
                path=before.crack_network.branches[0].path
                local_trace.append(('stochastic_healing_predecessor',before))
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
                    after=resume_to_guard(after,ops)
                elif case in ("downstream_zero_drive","delayed_downstream"):
                    after,_=advance_transition(after,"ligament",1,operations=ops)
                    after=load_state(after,0.); after,_=advance_transition(after,"downstream_child",1,operations=ops)
                    if case=="delayed_downstream":
                        after=load_state(after,opening); after,_=advance_transition(after,"downstream_child",1,operations=ops)
                else: after=resume_to_guard(after,ops)
        except Exception as exc:
            error={"type":type(exc).__name__,"message":str(exc)}
            after=getattr(exc,'accepted_lifecycle_state',after)
            if after is available and local_trace:
                # A failed preparation still owns its actual accepted history.
                # Never substitute the unrelated centered campaign precursor.
                before=local_trace[0][1]
                after=local_trace[-1][1]
        record("controlled",case,before,after,{"center_m":center,"loading_opening_m":opening,
            "fixed_crack_path_m":path,"history_kind":case,'cleavage_theta_deg':angle,
            'seed':CONTROLLED_HEALING_SEED if case=='embryo_healing' else 3621,
            'remesh_boundary_segments':64 if case=='local_remesh_refinement' else 32,
            'remesh_radial_layers':24 if case=='local_remesh_refinement' else 12,
            'preparation_boundary_segments':512 if args.qualified_fine_history and case not in ('local_remesh_refinement','embryo_healing') else 32,
            'preparation_radial_layers':192 if args.qualified_fine_history and case not in ('local_remesh_refinement','embryo_healing') else 12,
            'accepted_opening_history_m':([4e-7,0.,opening] if case=='delayed_downstream'
                else [4e-7,0.] if case=='downstream_zero_drive' else [4e-7,opening]),
            'growth_interval_count':8 if case in ('diffusion_limited','accommodation_limited') else None,
            'growth_interval_duration_s':1e-10 if case in ('diffusion_limited','accommodation_limited') else None},ops,
            actual_preparation_stages=[name for name,_ in local_trace],
            actual_preparation_checkpoints={name:checkpoint(state) for name,state in local_trace},failure=error)
    for case in selected('neutrality',FROZEN_CASE_REGISTRY["neutrality"]):
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
    for seed in selected('natural',NATURAL_SEEDS):
        for partitions in PARTITIONS:
            print(f'Actual natural lifecycle {seed}/{partitions}',flush=True)
            before,_=build_production_void_state(stochastic=True,seed=seed)
            after=before; ops=[]; replay_ops=[]; failure=None; elapsed_intervals=[]; cache={}
            # Two half-windows make the restart point physical and common to
            # all partitions, rather than an arbitrary event-index checkpoint.
            for half in range(2):
                for interval in range(partitions):
                    after,trace,result=advance_production_void_interval(after,NATURAL_WINDOW_S/(2*partitions),
                        config=CFG,refinement_attempt_cache=cache)
                    ops.extend(trace);elapsed_intervals.append(result['elapsed_duration_s'])
                    if result['failure'] is not None: failure=result['failure'];break
                if half==0:
                    midpoint=after; replay=restore_checkpoint(out/checkpoint(midpoint))
                if failure is not None: break
            replay_failure=None
            if failure is None:
                for interval in range(partitions):
                    replay,trace,result=advance_production_void_interval(replay,NATURAL_WINDOW_S/(2*partitions),
                        config=CFG,refinement_attempt_cache={})
                    replay_ops.extend(trace)
                    if result['failure'] is not None: replay_failure=result['failure'];break
            phase=after.void_state.cavities[0].phase if after.void_state.cavities else after.void_state.sites[0].phase
            record("natural",str(seed),before,after,{"seed":seed,"partition_count":partitions,
                "duration_s":NATURAL_WINDOW_S,'opening_m':4e-7,'temperature_K':900.,
                'driver':'advance_production_void_interval','restart_time_s':NATURAL_WINDOW_S/2},ops,
                elapsed_physical_time_s=math.fsum(elapsed_intervals),failure=failure,replay_failure=replay_failure,
                terminal_measurements=natural_terminal_measurements(after),
                terminal_classification=phase.value,restarted_operations=replay_ops,
                midpoint_restart_exact=fingerprint(after)==fingerprint(replay),
                restarted_terminal_checkpoint=checkpoint(replay))
    # Real rollback injections for stages reachable without unqualified events.
    for stage in selected('rollback',("field_projection","support_rebuild","equilibrium")):
        before=captured["subgrid_growth"]; after=before; ops=[]; error=None
        initial_capture=restore_checkpoint(out/checkpoint(before))
        cavity=before.void_state.cavities[0]; hole,_=_geometry(radius_m=cavity.radius_m,center_m=cavity.center_m)
        try: after=remesh_cavity(before,hole,before.void_state,"rollback-promotion",ops,failure_stage=stage)
        except Exception as exc: error={"type":type(exc).__name__,"message":str(exc)}
        record("rollback","promotion:"+stage,initial_capture,after,{"failure_stage":stage,'intended_stage_reached':stage in ops},ops,failure=error,
            restored_exactly=fingerprint(initial_capture)==fingerprint(after))
    for stage in selected('rollback',("graph_edit","remesh","field_projection","support_rebuild","equilibrium","energy_gate",
                  "connected_surface_certification","dormant_support_rebuild")):
        before=captured["resolved_growth"]; after=before; ops=[]; error=None
        initial_capture=restore_checkpoint(out/checkpoint(before))
        try: after,_=ligament_transaction(before,failure_stage=stage,operation_log=ops)
        except Exception as exc: error={"type":type(exc).__name__,"message":str(exc)}
        record("rollback","ligament:"+stage,initial_capture,after,{"failure_stage":stage,'intended_stage_reached':stage in ops},ops,failure=error,
            restored_exactly=fingerprint(initial_capture)==fingerprint(after))
    for stage,before,after,configuration,ops,error,restored in (rollback_attempts(captured,terminal,out/'rollback_checkpoint.json')
            if args.section in ('all','rollback') else ()):
        print('Actual lifecycle rollback '+stage,flush=True)
        record('rollback','lifecycle:'+stage,before,after,configuration,ops,failure=error,restored_exactly=restored)
    class Sources:
        def __getitem__(self,key): return restore_checkpoint(out/key)
    payload={"schema":SCHEMA,"executed_code_sha":sha,"rows":rows,
             "preparation_failure":preparation_failure,
             "execution_section":args.section,"shard_index":args.shard_index,"shard_count":args.shard_count,
             "decision":lifecycle_decision(rows,Sources())}
    if args.phases_7_8_only or args.section!='all':
        payload['schema']='v5.source-resolution-development-phases-7-8/1' if args.phases_7_8_only else 'v5.source-resolution-lifecycle-shard/1'
        validation={'full_closure_ontology':'NOT_APPLICABLE_PARTIAL_DEVELOPMENT_EXECUTION',
            'transition_count':sum(r['dataset']=='transitions' for r in rows),
            'restart_count':sum(r['dataset']=='restarts' for r in rows),
            'rollback_count':sum(r['dataset']=='rollback' for r in rows)}
    else:
        validation=validate_closure_evidence(payload,Sources(),executed_code_sha=sha)
    write_json(out/"lifecycle_rows.json",payload); write_json(out/"ontology_validation.json",validation)
    write_json(out/"sha256_manifest.json",{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob("*")) if p.is_file()})
    print(json.dumps(validation),flush=True)


if __name__=="__main__":main()
