#!/usr/bin/env python3
"""Rerun only the five inexpensive physical stages after exact-duration repair."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint,write_checkpoint
from arrhenius_fracture.closure_lifecycle_evidence import advance_transition,transition_occurred,conservation
from arrhenius_fracture.closure_mechanics_evidence import canonical_data
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('development_rows',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--stages',nargs='+',choices=('birth_hit_1','birth_hit_2','stabilization','healing','subgrid_growth','child_continuation'),
        default=('birth_hit_1','birth_hit_2','stabilization','healing','subgrid_growth'))
    args=parser.parse_args()
    if args.output.exists():raise ValueError('refusing to overwrite sentinels')
    status=subprocess.check_output(('git','status','--porcelain'),cwd=ROOT,text=True)
    if status.strip():raise ValueError('site sentinel requires clean implementation')
    args.output.mkdir(parents=True);rows=[]
    source_rows=[(p,json.loads(p.read_text())) for p in (args.development_rows/'rows').glob('*.json')]
    for name in args.stages:
        path,row=next((p,r) for p,r in source_rows if r['dataset']=='transitions' and r['case_identity']==name and r['partition_count']==1)
        before=restore_checkpoint(args.development_rows/row['initial_checkpoint']);reference=None
        for count in (1,2,4,8,16):
            print('Actual exact-duration stage '+name+'/'+str(count),flush=True)
            operations=[];after,error=before,None
            try:after,_=advance_transition(before,name,count,operations=operations)
            except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)}
            identity=fingerprint(after)
            if reference is None:reference=identity
            filename=name+'_'+str(count)+'.json';write_checkpoint(after,args.output/filename,compression='gzip')
            checks=conservation(after,before)
            rows.append({'case':name,'partition_count':count,'initial_fingerprint':fingerprint(before),
                'source_row_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_initial_checkpoint':row['initial_checkpoint'],
                'operations':operations,'failure':error,'terminal_checkpoint':filename,'terminal_fingerprint':identity,
                'conservation':checks,'passed':error is None and identity==reference and transition_occurred(name,before,after) and checks['passed']})
    report={'schema':'v5.exact-physical-stage-partition-sentinels/1',
        'executed_code_sha':subprocess.check_output(('git','rev-parse','HEAD'),cwd=ROOT,text=True).strip(),
        'rows':rows,'passed':all(r['passed'] for r in rows),'not_full_closure_campaign':True}
    (args.output/'report.json').write_text(json.dumps(canonical_data(report),sort_keys=True,indent=2,allow_nan=False)+'\n')
    (args.output/'sha256_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.output.iterdir()) if p.is_file()},sort_keys=True,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'cases':len(rows)}),flush=True)


if __name__=='__main__':main()
