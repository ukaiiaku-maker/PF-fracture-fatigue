#!/usr/bin/env python3
"""Sparse frozen checks only after both branch-disabled continuations terminate."""
import argparse
import json
from pathlib import Path
import subprocess
import traceback

from scripts.run_v13_later_clean_parents import CASES
from scripts.qualify_v13_physical_companions import evaluate_parent
from scripts.run_pf_current_source_multifront_field_atlas_v12 import atomic_json
from scripts.qualify_v13_process_restore import safe_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    parents=args.root/'later_parents'
    if any(not (parents/case/'terminal.json').exists() for case in CASES):
        raise RuntimeError('continuations still active; no extra heavy worker launched')
    plan=json.loads(args.plan.read_text())
    rows=[]
    for case in CASES:
        for event in sorted((parents/case/'events').glob('event*')):
            try:
                result=evaluate_parent(event.name,event.parent,args.root/'later_companions'/case,plan,
                    later_launch_path=parents/case/'launch.json')
            except Exception as exc:
                result={'case':event.name,'status':'FROZEN_COMPANION_STOP','reason':str(exc),
                    'type':type(exc).__name__,'traceback':traceback.format_exc(),'automatic_retry':False}
                atomic_json(args.root/'later_companions'/case/event.name/'stop.json',result)
            result['material_case']=case
            rows.append(result)
            print(case,event.name,result['status'],flush=True)
    atomic_json(args.root/'later_companions/summary.json',safe_json({'cases':rows,
        'producer_code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'branching_enabled_in_parent':False,'short_ensemble_launched':False}))


if __name__=='__main__':
    main()
