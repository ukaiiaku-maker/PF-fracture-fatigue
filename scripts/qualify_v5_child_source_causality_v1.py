#!/usr/bin/env python3
"""Controlled source interventions through the real downstream transaction.

These are causal software interventions, not newly equilibrated physical loads.
They do not qualify a substituted tensor as a production cavity source.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture import voiding_production_v5 as production
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint,write_checkpoint
from arrhenius_fracture.closure_mechanics_evidence import canonical_data
from arrhenius_fracture.closure_lifecycle_evidence import conservation,stagewise_topology
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_pair',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('refusing to overwrite evidence')
    args.output.mkdir(parents=True)
    manifest=json.loads((args.source_pair/'sha256_manifest.json').read_text())
    actual={str(p.relative_to(args.source_pair)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.source_pair.rglob('*') if p.is_file() and p.name!='sha256_manifest.json'}
    if actual!=manifest:raise ValueError('source-pair manifest mismatch')
    qualified=restore_checkpoint(args.source_pair/'qualified_or_preserved_state.json')
    child=restore_checkpoint(args.source_pair/'child_or_rejected_state.json')
    rows=[]
    def attempt(name,before,continuation,probe,factor):
        print('Actual source-causality transaction '+name,flush=True)
        write_checkpoint(before,args.output/(name+'_initial.json'),compression='gzip')
        original=getattr(production,probe);calls=[]
        def intervention(*a,**kw):
            tensor,ids=original(*a,**kw);calls.append({'element_ids':list(ids),'original_tensor_Pa':tensor.tolist()})
            return factor*tensor,ids
        after=before;event=None;audit={};failure=None;operations=[]
        try:
            with patch.object(production,probe,intervention):
                after,event,operations,audit=production.downstream_front_transaction(before,continuation=continuation)
        except Exception as exc:failure={'type':type(exc).__name__,'message':str(exc)}
        write_checkpoint(after,args.output/(name+'.json'),compression='gzip')
        row={'case':name,'continuation':continuation,'intervention':{'probe':probe,'factor':factor},
            'probe_calls':calls,'initial_fingerprint':fingerprint(before),'terminal_fingerprint':fingerprint(after),
            'accepted':bool(event is not None and event.accepted),'audit':audit,'operations':operations,'failure':failure,
            'topology':stagewise_topology(after),'conservation':conservation(after,before)}
        rows.append(row)
        (args.output/'partial_rows.json').write_text(json.dumps(canonical_data(rows),sort_keys=True,indent=2,allow_nan=False)+'\n')
        return row
    changed_cavity=attempt('initial_cavity_source_changed',qualified,False,'cavity_boundary_tensor',1.01)
    baseline=attempt('ordinary_child_continuation',child,True,'crack_tip_tensor',1.)
    irrelevant=attempt('changed_cavity_during_continuation',child,True,'cavity_boundary_tensor',1.01)
    changed_tip=attempt('changed_child_tip_drive',child,True,'crack_tip_tensor',2.)
    zero=attempt('zero_child_tip_drive',child,True,'crack_tip_tensor',0.)
    # Accepted mappings deliberately reject mutation, including after deepcopy.
    tip={**child.tip_process_state,'by_branch':{key:dict(value) for key,value in child.tip_process_state['by_branch'].items()}}
    tip['by_branch']['void-front-1']['r_tip_m']*=2
    network=replace(child.crack_network,branches=tuple(replace(branch,local_state={**branch.local_state,
        'r_tip_m':tip['by_branch']['void-front-1']['r_tip_m']}) if branch.branch_id=='void-front-1' else branch
        for branch in child.crack_network.branches))
    altered=replace(child,tip_process_state=tip,crack_network=network)
    radius=attempt('changed_child_owned_radius',altered,True,'crack_tip_tensor',1.)
    radius['radius_intervention']={'factor':2.,'owned_process_state_and_branch_local_state_changed_together':True,
        'R_void_unchanged':altered.void_state.cavities[0].radius_m==child.void_state.cavities[0].radius_m}
    def rates(row):return [x['effective_rate_s'] for x in row['audit'].get('cleavage',[])]
    gates={
        'changed_cavity_source_rejected_without_recognition_as_qualified':not changed_cavity['accepted']
            and changed_cavity['audit'].get('status')=='UNQUALIFIED_CAVITY_SOURCE_TENSOR',
        'actual_ordinary_child_continuation':baseline['accepted'],
        'cavity_probe_cannot_influence_continuation':baseline['accepted'] and irrelevant['accepted']
            and not irrelevant['probe_calls'] and baseline['terminal_fingerprint']==irrelevant['terminal_fingerprint'],
        'child_tensor_changes_actual_continuation_rate':baseline['accepted'] and changed_tip['accepted']
            and bool(rates(baseline)) and rates(baseline)!=rates(changed_tip),
        'zero_child_tip_drive_creates_no_event':not zero['accepted'] and zero['failure'] is None
            and zero['audit'].get('status')=='NO_KINETICALLY_ACTIVE_CANDIDATE',
        'owned_radius_causally_enters_continuation_law':bool(rates(baseline)) and rates(radius)!=rates(baseline),
        'radius_separate_from_void_radius':child.tip_process_state['by_branch']['void-front-1']['r_tip_m']
            !=child.void_state.cavities[0].radius_m,
    }
    result={'schema':'v5.child-source-causal-interventions/1',
        'executed_code_sha':subprocess.check_output(('git','rev-parse','HEAD'),cwd=ROOT,text=True).strip(),
        'worktree_status':subprocess.check_output(('git','status','--porcelain'),cwd=ROOT,text=True),
        'source_pair_manifest_sha256':hashlib.sha256((args.source_pair/'sha256_manifest.json').read_bytes()).hexdigest(),
        'scope':'SOFTWARE_CAUSAL_INTERVENTIONS_NOT_EQUILIBRATED_LOAD_QUALIFICATION',
        'rows':rows,'gates':gates,'passed':all(gates.values())}
    (args.output/'report.json').write_text(json.dumps(canonical_data(result),sort_keys=True,indent=2,allow_nan=False)+'\n')
    (args.output/'sha256_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.output.iterdir()) if p.is_file()},sort_keys=True,indent=2)+'\n')
    print(json.dumps(gates),flush=True)


if __name__=='__main__':main()
