#!/usr/bin/env python3
"""New actual quality-constrained static solves, raw and recovered values retained."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from arrhenius_fracture.static_numerical_family_v1 import REGISTRY, GROUPS, SCHEMA, classify
from arrhenius_fracture.crack_void_mechanics_v5 import solve_crack_void_case
from arrhenius_fracture.closure_mechanics_evidence import measurements, canonical_data
from arrhenius_fracture.closure_static_evidence import validate_solver_capture, source_fingerprints
from qualify_cavity_boundary_patch_recovery_v1 import recover_arcs


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('output',type=Path)
    parser.add_argument('--sentinel',action='store_true')
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--shard-count',type=int,default=1)
    args = parser.parse_args()
    if not 0<=args.shard_index<args.shard_count:raise ValueError('invalid static shard')
    if args.sentinel and args.shard_count!=1:raise ValueError('sentinel cannot be sharded')
    if args.output.exists(): raise ValueError('refusing to overwrite evidence')
    status = subprocess.check_output(('git','status','--porcelain'),cwd=ROOT,text=True)
    if status.strip() and not args.sentinel: raise ValueError('full static qualification requires clean committed code')
    sha = subprocess.check_output(('git','rev-parse','HEAD'),cwd=ROOT,text=True).strip()
    groups = {'centered:{mesh}':GROUPS['centered:{mesh}']} if args.sentinel else GROUPS
    selected = {key for levels in groups.values() for key in levels.values()}
    selected = {key for i,key in enumerate(sorted(selected)) if i%args.shard_count==args.shard_index}
    (args.output/'sources').mkdir(parents=True); rows = {}
    for key,cfg in REGISTRY.items():
        if key not in selected: continue
        print('Static fine-family solve '+str(len(rows)+1)+'/'+str(len(selected))+' '+key[:12],flush=True)
        row = {'case_id':key,'input_configuration':cfg,'executed_code_sha':sha}
        try:
            solved = solve_crack_void_case(**cfg); raw = solved['source_capture']
            np.savez_compressed(args.output/'sources'/(key+'.npz'),**raw)
            row.update(source_fingerprints=source_fingerprints(raw), measurements=measurements(raw,cfg),
                solver_validation=canonical_data(validate_solver_capture(raw,cfg)),
                geometry_conformity_audit=solved['geometry_conformity_audit'],
                recovery=recover_arcs(raw,cfg['cavity_center_m']) if cfg['cavity_enabled'] else [])
        except Exception as exc: row['failure']={'type':type(exc).__name__,'message':str(exc)}
        rows[key]=row
        (args.output/(key+'.json')).write_text(json.dumps(canonical_data(row),sort_keys=True,indent=2,allow_nan=False)+'\n')
    decision=classify(rows,groups) if args.shard_count==1 else {'passed':False,'classification':'INCOMPLETE_SHARD_REQUIRES_FULL_REGISTRY_ASSEMBLY'}
    result={'schema':SCHEMA,'executed_code_sha':sha,'worktree_status':status,'sentinel_only':args.sentinel,
        'shard_index':args.shard_index,'shard_count':args.shard_count,
        'groups':groups,'rows':rows,'decision':decision, 'historical_v3_predicates':'UNCHANGED_RETAINED_683_OF_791'}
    (args.output/'report.json').write_text(json.dumps(canonical_data(result),sort_keys=True,indent=2,allow_nan=False)+'\n')
    (args.output/'sha256_manifest.json').write_text(json.dumps({str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.output.rglob('*')) if p.is_file()},sort_keys=True,indent=2)+'\n')
    print(json.dumps({'passed':result['decision']['passed'],'solves':len(rows)}),flush=True)


if __name__=='__main__': main()
