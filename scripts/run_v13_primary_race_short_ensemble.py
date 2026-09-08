"""Preregistered 16-case first-branch gate; two workers, no retry or seed screen."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from scripts.run_pf_current_source_multifront_field_atlas_v12 import (
    ROOT, ROWS, common_arguments, campaign_environment, validate_inputs, registry_row, atomic_json, sha256)
from arrhenius_fracture.branch_checkpoint_v11 import restore_branch_checkpoint
from arrhenius_fracture.conditional_branch_mark_v13 import BOUNDARY

OUT=ROOT/'analysis_outputs/v13_primary_continuation_race'
PLAN=OUT/'short_ensemble_plan.json'
CASES=[f'{material}_{temperature}K_seed{seed}' for seed in (3621,3622,3623,3624)
       for material in ('Peak','weakT') for temperature in (300,1000)]


def prepare():
    gate=json.loads((OUT/'frozen_summary.json').read_text())
    verification=json.loads((OUT/'verification.json').read_text())
    if not gate['later_pattern_gate_passed'] or verification['status']!='PASS':
        raise RuntimeError('frozen pattern and source parity gates must pass')
    old=json.loads((ROOT/'analysis_outputs/v13_physical_companion_qualification/preregistered_plan.json').read_text())
    original_launch=json.loads((ROOT/'analysis_outputs/v13_physical_companion_qualification/physical_parents/Peak_1000K/launch.json').read_text())
    plan={'schema':'v13.primary-race-short-ensemble/1','cases':CASES,'seeds':[3621,3622,3623,3624],
        'materials':['Peak','weakT'],'temperatures_K':[300,1000],'common_random_numbers':True,
        'seed_screen':False,'maximum_workers':2,'target_forward_um':125.,
        'daughter_growth_after_birth_um':20.,'daughter_total_length_stop_um':25.,
        'physical_event_length_um':5.,'maximum_accepted_intervals':2000,
        'family':original_launch['family_validation']['family_validation']['family'],'family_sha256':old['family_sha256'],
        'mechanical_configuration_sha256':old['mechanical_configuration_sha256'],
        'default_off_option':'--v13-inherited-primary-race','parent_maximum_fronts':1,
        'post_birth_maximum_fronts':2,'boundary':BOUNDARY,'new_material_parameters':False,
        'frozen_summary_sha256':sha256(OUT/'frozen_summary.json'),
        'verification_sha256':sha256(OUT/'verification.json'),
        'reuse_of_archived_different_source_trajectories':False,
        'terminal_rules':['first branch plus 20 um daughter growth','125 um maximum forward reach','existing legitimate numerical/model-domain gate'],
        'incidence_is_not_calibrated_probability':True}
    if PLAN.exists() and json.loads(PLAN.read_text())!=plan:
        raise RuntimeError('preregistered plan differs; no overwrite')
    atomic_json(PLAN,plan)


def terminal_record(folder,reason,status):
    path=folder/'checkpoint/latest.json'
    rows=[json.loads(line) for line in (folder/'v13_primary_race.jsonl').read_text().splitlines()] if (folder/'v13_primary_race.jsonl').exists() else []
    branches=[r for r in rows if r['outcome']=='PAIR_ACCEPTED']
    record={'status':status,'reason':reason,'boundary':BOUNDARY,'branch_count':len(branches),
        'first_branch':branches[0] if branches else None,'automatic_retry':False}
    if path.exists():
        checkpoint=restore_branch_checkpoint(path)
        state=checkpoint.state
        daughters=[sum(math.dist(a,b) for a,b in zip(c.path,c.path[1:])) for c in state.crack_network.branches if c.generation>0]
        record.update(last_checkpoint=str(path),last_checkpoint_manifest_sha256=sha256(path),
            last_checkpoint_state_sha256=sha256(path.parent/json.loads(path.read_text())['state_file']),
            forward_extension_um=checkpoint.projected_extension_m*1e6,
            accepted_time_s=checkpoint.physical_time_s,accepted_opening_m=checkpoint.accepted_load,
            active_front_count=len(state.crack_network.active_tip_ids),
            daughter_growth_after_birth_um=max(daughters,default=5e-6)*1e6-5 if daughters else None,
            branch_clock_bookkeeping='separate V13 mark ledger; canonical event counters unchanged')
        # A single final portable field record; no image sequence or field campaign.
        np.savez_compressed(folder/'final_accepted_fields.npz',
            coordinates=np.asarray(state.mesh.nodes),elements=np.asarray(state.mesh.elems),displacement=state.displacement,
            damage=state.damage,ep_gp=state.ep_gp,rho_gp=state.rho_gp)
        record['final_fields_sha256']=sha256(folder/'final_accepted_fields.npz')
    atomic_json(folder/'terminal.json',record)


def worker(case):
    plan=json.loads(PLAN.read_text())
    if case not in plan['cases']:
        raise ValueError('unregistered case')
    folder=OUT/'short_ensemble'/case
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'launch_claim.json').open('x') as stream:
        json.dump({'pid':os.getpid(),'case':case},stream)
    material,temperature,seed=case.split('_')
    temperature=int(temperature[:-1]);seed=int(seed[4:])
    family=Path(plan['family'])
    validation=validate_inputs(family)
    if sha256(family)!=plan['family_sha256']:
        raise RuntimeError('family identity changed')
    env=campaign_environment(family)
    env.update(CLEAVAGE_HAZARD_SEED=str(seed),PF_QUALIFIED_DAUGHTER_STOP_UM='25')
    os.environ.update(env)
    for key in ('V11_BRANCH_RESTART_CHECKPOINT','PF_V5_4_1_RESTORE_ONLY_SENTINEL_OUT'):
        os.environ.pop(key,None)
    canonical,alias=ROWS[material]
    argv=common_arguments(alias,temperature,family,folder)
    argv[argv.index('--steps')+1]='2000'
    argv[argv.index('--target-crack-extension-um')+1]='125'
    argv+=['--maximum-fronts','1','--v13-inherited-primary-race']
    row,row_hash=registry_row(canonical)
    from arrhenius_fracture import sharp_front_v11_branching as production, sharp_front_v10_2_27 as paper
    paper.DEFAULT_REGISTRY=ROOT/'runtime_inputs/pf_current_source_branching/pf_v2_four_class_pf_transfer_registry.csv'
    paper.SELECTION_RECORD=ROOT/'runtime_inputs/pf_current_source_branching/pf_v2_four_class_pf_transfer_selection.json'
    paper.VALID_OPTIONS={alias:c for c,alias in ROWS.values()}
    atomic_json(folder/'launch.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'case':case,'seed':seed,'arguments':argv,'material_row':row,'material_row_sha256':row_hash,
        'family_validation':validation,'plan_sha256':sha256(PLAN),'fresh_initialization':True,'boundary':BOUNDARY})
    try:
        production.main(argv)
        manifest=json.loads((folder/'checkpoint/latest.json').read_text())
        terminal_record(folder,manifest['termination_reason'],'TERMINATED')
    except Exception as exc:
        atomic_json(folder/'exception.json',{'type':type(exc).__name__,'reason':str(exc),'traceback':traceback.format_exc()})
        # Preserve evidence without silently calling an unknown exception a
        # legitimate physical stop. Queue pauses only for software/unclassified errors.
        known=any(token in str(exc) for token in ('opening_scale_not_resolved_above_probe_uncertainty',
            'directional adaptive stepping reached its minimum fraction','family endpoint','outside qualified',
            'Newton failed','equilibrium did not converge','nonpositive_opening'))
        terminal_record(folder,str(exc),'EXISTING_GATE_STOP' if known else 'SOFTWARE_OR_UNCLASSIFIED_STOP')
        if not known:
            raise


def queue():
    if not PLAN.exists():
        raise RuntimeError('preregister before launching')
    root=OUT/'short_ensemble';root.mkdir(exist_ok=True)
    with (root/'queue_claim.json').open('x') as stream:
        json.dump({'pid':os.getpid(),'maximum_workers':2},stream)
    pending=list(CASES);active={}
    if any((root/c/'launch_claim.json').exists() for c in pending):
        raise RuntimeError('existing case launch claim; no duplicate/restart')
    while pending or active:
        while pending and len(active)<2:
            if shutil.disk_usage(root).free<1024**3:
                raise RuntimeError('less than 1 GiB durable space; no next worker launched')
            case=pending.pop(0);folder=root/case;folder.mkdir(exist_ok=True)
            log=(folder/'worker.log').open('a')
            env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONHASHSEED='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                MPLCONFIGDIR='/tmp/pf-current-source-v13-parent-mpl')
            active[case]=(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--case',case],env=env,stdout=log,stderr=subprocess.STDOUT),log)
        for case,(process,log) in list(active.items()):
            if process.poll() is not None:
                log.close();del active[case]
                terminal=root/case/'terminal.json'
                if not terminal.exists() or json.loads(terminal.read_text())['status']=='SOFTWARE_OR_UNCLASSIFIED_STOP':
                    # Let an already-running peer finish; never kill its accepted work.
                    atomic_json(root/'queue_pause.json',{'case':case,'reason':'software_or_unclassified_stop','pending':pending})
                    pending=[]
        atomic_json(root/'queue_status.json',{'active':{c:p.pid for c,(p,_) in active.items()},'pending':pending,
            'completed':[c for c in CASES if (root/c/'terminal.json').exists()]})
        if active:
            time.sleep(5)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=CASES);parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    prepare() if args.prepare else worker(args.case) if args.case else queue()
