#!/usr/bin/env python3
"""Two provenance-locked single-front continuations; never enables V13 marks."""
import argparse
import copy
from dataclasses import replace
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import traceback

from scripts.run_v13_clean_parents import FirstParentCapture
from scripts.run_pf_current_source_multifront_field_atlas_v12 import ROOT, ROWS, atomic_json, sha256, campaign_environment
from scripts.v13_value_fingerprint import physical_state_fingerprint as fp
from arrhenius_fracture.sharp_front_v11_branching import _capture_shared_engine

SOURCE_ROOT=ROOT/"analysis_outputs/v13_physical_companion_qualification"
RECORD_COMMIT="d10c3ebd8a3513edec5624faebdc7faf8656d106"
CASES=("Peak_1000K","weakT_1000K")


def certify(case):
    if case not in CASES:
        raise ValueError("only two selected continuations are authorized")
    path=SOURCE_ROOT/"physical_parents"/case/"parent/parent_record.json"
    frozen=subprocess.check_output(["git","show",f"{RECORD_COMMIT}:{path.relative_to(ROOT)}"])
    if path.read_bytes()!=frozen:
        raise RuntimeError("parent record differs from immutable d10c3eb")
    r=json.loads(frozen)
    source=path.parent/"accepted_single.json"
    manifest=json.loads(source.read_text())
    state_path=source.parent/manifest["state_file"]
    if sha256(state_path)!=r["accepted_single_checkpoint_sha256"] or manifest["state_sha256"]!=r["accepted_single_checkpoint_sha256"]:
        raise RuntimeError("clean parent checkpoint hash mismatch")
    if not r["fresh_initialization"] or r["historical_process_state_used"] or r["event_ordinal"]!=1:
        raise RuntimeError("not a first canonical clean parent")
    return source,manifest,r


class LaterCapture(FirstParentCapture):
    def __init__(self,output,source_record):
        super().__init__(output)
        self.source_record=source_record
        self.milestones=[25.,50.,100.]
        self.last_event_count=1

    def begin(self,state,engine,physical_time,opening,runtime):
        from arrhenius_fracture.current_source_runtime_bindings import runtime_binding_inventory, callable_id
        from arrhenius_fracture.persistent_site_source_v10221 import _persistent_emit
        if callable_id(engine.mpz._emit)!=callable_id(_persistent_emit):
            raise RuntimeError("continuation lost intended persistent-source law")
        if state.crack_network.branching_enabled or len(state.crack_network.active_tip_ids)!=1:
            raise RuntimeError("later parent must remain branch-disabled")
        self.pre=(state,copy.deepcopy(_capture_shared_engine(engine)),physical_time,opening,copy.deepcopy(runtime))
        self.bindings=runtime_binding_inventory(engine)

    def accept(self,*,checkpoint,context,result,solved_pre_event,pre_event_sigma,args,cfg):
        from arrhenius_fracture.branch_checkpoint_v11 import write_branch_checkpoint
        from arrhenius_fracture.sharp_front_v11_branching import _hash,_mesh_identity
        self.interval_count+=1
        extension=checkpoint.projected_extension_m*1e6
        count=len(checkpoint.state.competition.consumed_event_ids)
        atomic_json(self.output/"progress.json",{"step":context.step,"forward_extension_um":extension,
            "event_count":count,"time_s":checkpoint.physical_time_s,"opening_m":checkpoint.accepted_load})
        selected=[t for t in result.trials if t.selected]
        if not selected:
            return False
        if len(selected)!=1 or selected[0].proposal.action_type!='one_arm':
            raise RuntimeError("branch-disabled later parent selected non-single event")
        # Save the next event and first event at/after each sparse checkpoint.
        reached=[m for m in self.milestones if extension>=m]
        save=count==2 or bool(reached)
        self.milestones=[m for m in self.milestones if m not in reached]
        if not save:
            return False
        folder=self.output/f"events/event{count:05d}"/"parent"
        state,process,t,U,runtime=self.pre
        pre=replace(checkpoint,state=state,shared_process_state=process,physical_time_s=t,accepted_load=U,
            provider_runtime=runtime,boundary_condition_state={"opening_m":U},mesh_identity=_mesh_identity(state.mesh),
            topology_fingerprint=_hash(state.crack_network),front_competitions={tip:state.competition for tip in state.crack_network.active_tip_ids},
            branch_clusters=(),projected_extension_m=max(b.tip[0] for b in state.crack_network.branches)-cfg.geometry.a0,
            physical_extension_m=state.crack_network.total_physical_crack_length_m-cfg.geometry.a0,termination_reason=None)
        a=write_branch_checkpoint(pre,folder/"pre_cleavage.json")
        b=write_branch_checkpoint(checkpoint,folder/"accepted_single.json")
        payload={"context":context,"pre_cleavage_checkpoint":pre,"accepted_single_checkpoint":checkpoint,
            "solved_pre_event_state":solved_pre_event,"pre_event_sigma":pre_event_sigma,
            "canonical_result":result,"args":vars(args),"configuration":cfg}
        raw=pickle.dumps(payload,protocol=5)
        (folder/"event_context.pkl").write_bytes(raw)
        proposal=selected[0].proposal
        atomic_json(folder/"parent_record.json",{"schema":"v13.later-clean-parent/1","fresh_initialization":False,
            "clean_history":True,"source_first_parent_checkpoint_sha256":self.source_record["accepted_single_checkpoint_sha256"],
            "winning_candidate_id":proposal.member_candidate_ids[0],"event_id":proposal.member_event_ids[0],
            "event_ordinal":proposal.member_event_ordinals[0],"raw_first_passage_s":proposal.completion_times_s[0],
            "accepted_endpoint_s":checkpoint.physical_time_s,"accepted_opening_m":checkpoint.accepted_load,
            "pre_cleavage_checkpoint_sha256":a["state_sha256"],"accepted_single_checkpoint_sha256":b["state_sha256"],
            "accepted_process_sha256":fp(checkpoint.shared_process_state),"event_context_sha256":hashlib.sha256(raw).hexdigest(),
            "accepted_event_ids":list(checkpoint.state.competition.consumed_event_ids),"forward_extension_um":extension,
            "milestones_reached_um":reached,"branching_enabled":False,"installed_bindings":self.bindings})
        return False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',choices=CASES,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    root=args.output_root/args.case
    root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(root).free < 1024**3:
        raise RuntimeError('less than 1 GiB durable free space; no continuation launched')
    with (root/'launch_claim.json').open('x') as stream:
        json.dump({'pid':os.getpid(),'case':args.case,'branching_enabled':False},stream)
    source,manifest,record=certify(args.case)
    launch=json.loads((source.parent.parent/'launch.json').read_text())
    family=Path(launch['family_validation']['family_validation']['family'])
    os.environ.update(campaign_environment(family))
    for key in ('V11_BRANCH_RESTART_CHECKPOINT','PF_V5_4_1_RESTORE_ONLY_SENTINEL_OUT'):
        os.environ.pop(key,None)
    decision=json.loads((args.output_root.parent/'clock_audit.json').read_text())
    if decision['mechanism_decision']!='SEPARATE_COOPERATIVE_PAIR_TRANSITION_REQUIRED_FOR_APPRECIABLE_BRANCHING':
        raise RuntimeError('mechanism decision differs from this bounded continuation plan')
    initial=root/'input_checkpoint'
    initial.mkdir()
    step=int(manifest['event_counters']['accepted_steps'])
    destination=initial/f'step{step:07d}.json'
    shutil.copyfile(source,destination)
    shutil.copyfile(source.parent/manifest['state_file'],initial/manifest['state_file'])
    argv=list(launch['arguments'])
    argv[argv.index('--out')+1]=str(root)
    argv[argv.index('--target-crack-extension-um')+1]='100'
    argv[argv.index('--steps')+1]=str(step+1000)
    argv.extend(['--v11-restart-checkpoint',str(destination)])
    from arrhenius_fracture import sharp_front_v11_branching as production, sharp_front_v10_2_27 as paper
    old=(production.run_2d,production.require_uncontaminated_replay_checkpoint,
         paper.DEFAULT_REGISTRY,paper.SELECTION_RECORD,paper.VALID_OPTIONS)
    def clean_guard(actual_step,**kwargs):
        certify(args.case)
        if actual_step!=step:
            raise RuntimeError('clean restart step identity mismatch')
        # The historical step-287 exclusion is not applicable to this exact
        # post-restoration fresh lineage. No arbitrary checkpoint is exempted.
    production.require_uncontaminated_replay_checkpoint=clean_guard
    production.run_2d=partial(old[0],parent_capture=LaterCapture(root,record))
    paper.DEFAULT_REGISTRY=ROOT/'runtime_inputs/pf_current_source_branching/pf_v2_four_class_pf_transfer_registry.csv'
    paper.SELECTION_RECORD=ROOT/'runtime_inputs/pf_current_source_branching/pf_v2_four_class_pf_transfer_selection.json'
    paper.VALID_OPTIONS={alias:canonical for canonical,alias in ROWS.values()}
    atomic_json(root/'launch.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_parent_record':record,'arguments':argv,'family_validation':launch['family_validation'],
        'target_um':100,'branching_enabled':False,'no_reseed':True,'source_first_parent_record_commit':RECORD_COMMIT})
    try:
        production.main(argv)
        terminal=json.loads((root/'checkpoint/latest.json').read_text())
        atomic_json(root/'terminal.json',{'status':'TERMINATED','reason':terminal['termination_reason']})
    except Exception as exc:
        atomic_json(root/'terminal.json',{'status':'STOPPED','reason':str(exc),'type':type(exc).__name__,
            'traceback':traceback.format_exc(),'automatic_retry':False})
        raise
    finally:
        production.run_2d,production.require_uncontaminated_replay_checkpoint,paper.DEFAULT_REGISTRY,paper.SELECTION_RECORD,paper.VALID_OPTIONS=old


if __name__=='__main__':
    main()
