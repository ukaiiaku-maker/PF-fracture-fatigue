#!/usr/bin/env python3
"""Run the frozen exact-head audit and retain a known scientific blocker."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


EXPECTED='natural internal-stage independent production replay mismatch'


def git(root,*args):
    return subprocess.check_output(('git','-C',str(root),*args),text=True).strip()


def classify(implementation,evidence,publication,sha,output):
    if git(implementation,'rev-parse','HEAD')!=sha or git(implementation,'status','--porcelain'):
        raise ValueError('exact implementation worker is not clean at the requested SHA')
    paired=json.loads((evidence/'paired_comparison.json').read_text())
    if (paired['executed_code_sha']!=sha or not paired['exact_recursive_comparison']
            or paired['classification']!='PASS'):
        raise ValueError('exact recursive A/B comparison did not pass before exact-head audit')
    validator=implementation/'scripts/validate_v5_complete_published_campaign_v1.py'
    command=(sys.executable,str(validator),str(evidence),str(publication),sha,str(output)+'.raw')
    completed=subprocess.run(command,text=True,capture_output=True)
    if git(implementation,'rev-parse','HEAD')!=sha or git(implementation,'status','--porcelain'):
        raise ValueError('exact implementation worker changed during reconstruction')
    if completed.returncode==0:
        result=json.loads(Path(str(output)+'.raw').read_text())
        result.update({'classification':'PASS','exact_head_audit_completed':True,
            'validator_returncode':0,'validator_sha256':hashlib.sha256(validator.read_bytes()).hexdigest()})
    else:
        transcript=completed.stdout+'\n'+completed.stderr
        if EXPECTED not in transcript:
            raise RuntimeError('unexpected exact-head validator failure:\n'+transcript[-4000:])
        failure_line=next(line.strip() for line in reversed(transcript.splitlines()) if EXPECTED in line)
        result={'schema':'v5.exact-publication-head-complete-evidence-audit/2',
            'classification':'EXECUTED_BLOCKED','exact_head_audit_completed':True,
            'implementation_sha':sha,'publication_sha':sha,'exact_recursive_evidence_comparison':True,
            'evidence_valid':False,'scientific_decision':'BLOCKED','predicate_relaxed':False,
            'clean_implementation_worker_at_end':True,'validator_returncode':completed.returncode,
            'validator_sha256':hashlib.sha256(validator.read_bytes()).hexdigest(),
            'failure':{'type':'ValueError','identity':EXPECTED,'message':failure_line}}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('implementation',type=Path);parser.add_argument('evidence',type=Path)
    parser.add_argument('publication',type=Path);parser.add_argument('sha');parser.add_argument('output',type=Path)
    args=parser.parse_args()
    report=classify(args.implementation,args.evidence,args.publication,args.sha,args.output)
    print(json.dumps({'classification':report['classification'],
        'exact_head_audit_completed':report['exact_head_audit_completed']},sort_keys=True))
