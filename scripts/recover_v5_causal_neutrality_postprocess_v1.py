#!/usr/bin/env python3
"""Recover only the report stage of an exact failed final-campaign shard.

The original physical A/B rows and failed execution reports remain retained.
This tool accepts only the known missing-ripgrep postprocessing failure, runs no
physical worker, and emits a separately identified replacement for assembly.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from qualify_v5_disabled_causal_neutrality_v1 import postprocess
from validate_v5_source_resolution_evidence_v1 import verify_inventory

SCHEMA='v5.causal-neutrality-postprocess-recovery/1'
HISTORICAL_SHA='326e3f5973ef623781ab8568798c495a3f68238c'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def require_failed_execution(report,implementation_sha):
    expected=[{'operation':'causal_neutrality','script':'qualify_v5_disabled_causal_neutrality_v1.py','returncode':1}]
    if report.get('executed_code_sha')!=implementation_sha or report.get('execution_completed') is not False:
        raise ValueError('not the exact incomplete implementation execution')
    if report.get('phase')!='causal-neutrality' or report.get('section')!='all' or report.get('operations')!=expected:
        raise ValueError('not the known causal-neutrality postprocessor failure')
    if not report.get('clean_exact_head_at_end'):
        raise ValueError('failed shard did not retain a clean exact head')
    return report


def write_inventory(root):
    manifest={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*'))
        if p.is_file() and p!=root/'sha256_manifest.json'}
    (root/'sha256_manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')


def recover(source,output,implementation_sha,repair_sha,source_run_id,artifact_name,artifact_digest):
    for value,label in ((implementation_sha,'implementation'),(repair_sha,'repair')):
        if not re.fullmatch('[0-9a-f]{40}',value):raise ValueError(label+' SHA must be exact')
    if output.exists():raise ValueError('refusing to overwrite recovered shard')
    sides={}
    for side in ('a','b'):
        original=source/side;verify_inventory(original)
        execution=require_failed_execution(json.loads((original/'execution.json').read_text()),implementation_sha)
        evidence=original/'evidence'
        if (evidence/'report.json').exists() or (evidence/'sha256_manifest.json').exists():
            raise ValueError('failed shard unexpectedly contains completed postprocessing')
        for mode,expected_sha in (('base',HISTORICAL_SHA),('disabled',implementation_sha)):
            rows=json.loads((evidence/mode/'rows.json').read_text())
            if rows.get('sha')!=expected_sha or len(rows.get('rows',()))!=4:
                raise ValueError('incomplete or wrong physical neutrality rows')
        target=output/side;shutil.copytree(original,target)
        shutil.copy2(target/'execution.json',target/'original_execution.json')
        postprocess(target/'evidence',ROOT,implementation_sha)
        recovered={**execution,'execution_completed':True,'postprocessing_recovery':{
            'schema':SCHEMA,'physical_execution_sha':implementation_sha,'repair_sha':repair_sha,
            'source_run_id':str(source_run_id),'source_artifact_name':artifact_name,
            'source_artifact_digest':artifact_digest,'physical_workers_rerun':False,
            'original_failed_execution_retained':'original_execution.json',
            'failure_identity':'FileNotFoundError: rg during consumer occurrence indexing'}}
        recovered['operations']=[*execution['operations'],{'operation':'causal_neutrality_postprocess_recovery',
            'script':'recover_v5_causal_neutrality_postprocess_v1.py','returncode':0}]
        (target/'execution.json').write_text(json.dumps(recovered,sort_keys=True,indent=2)+'\n')
        write_inventory(target);verify_inventory(target);sides[side]=target
    for relative in ('evidence/base/rows.json','evidence/disabled/rows.json','evidence/report.json'):
        if digest(sides['a']/relative)!=digest(sides['b']/relative):raise ValueError('recovered A/B evidence differs: '+relative)
    report={'schema':SCHEMA,'implementation_sha':implementation_sha,'repair_sha':repair_sha,
        'source_run_id':str(source_run_id),'source_artifact_name':artifact_name,'source_artifact_digest':artifact_digest,
        'physical_workers_rerun':False,'a_b_physical_rows_exact':True,'a_b_recovered_reports_exact':True,
        'classification':'RECOVERED_POSTPROCESS_ONLY'}
    (output/'recovery.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(report,sort_keys=True),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--implementation-sha',required=True);parser.add_argument('--repair-sha',required=True)
    parser.add_argument('--source-run-id',required=True);parser.add_argument('--artifact-name',required=True)
    parser.add_argument('--artifact-digest',required=True);args=parser.parse_args()
    recover(args.source,args.output,args.implementation_sha,args.repair_sha,args.source_run_id,args.artifact_name,args.artifact_digest)
