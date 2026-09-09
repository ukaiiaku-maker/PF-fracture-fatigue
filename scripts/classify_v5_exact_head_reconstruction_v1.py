#!/usr/bin/env python3
"""Run the frozen exact-head audit and retain a known scientific blocker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


EXPECTED='natural internal-stage independent production replay mismatch'
LEDGER_SCHEMA_EXTENSION='published complete scientific classification does not recompute'


def git(root,*args):
    return subprocess.check_output(('git','-C',str(root),*args),text=True).strip()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')


def manifest(root):
    return {str(path.relative_to(root)):digest(path) for path in sorted(root.rglob('*'))
        if path.is_file() and path!=root/'sha256_manifest.json'}


def exact_ledger(implementation,evidence):
    program=('import json,sys; from pathlib import Path; '
        'sys.path.insert(0,str(Path.cwd()/"scripts")); '
        'from assemble_v5_source_resolution_campaign_v1 import derive_ledger; '
        'print(json.dumps(derive_ledger(Path(sys.argv[1])),sort_keys=True))')
    completed=subprocess.run((sys.executable,'-c',program,str(evidence)),cwd=implementation,
        text=True,capture_output=True)
    if completed.returncode:
        raise RuntimeError('exact-head ledger reconstruction failed:\n'+completed.stderr[-4000:])
    return json.loads(completed.stdout)


def project_extended_ledger(implementation,source,destination):
    """Make an inode-cheap projection understood by the frozen implementation.

    The final evidence is not changed.  Only the repair-added ontology field and
    its matching mandatory gate may be removed, and the frozen implementation
    must independently derive every remaining ledger byte semantically.
    """
    shutil.copytree(source,destination,copy_function=os.link)
    records=[]
    for side in ('a','b'):
        original=json.loads((source/side/'scientific_ledger.json').read_text())
        ontology=original.get('lifecycle_evidence_ontology')
        gate=original.get('mandatory_scientific_gates',{}).get('complete_lifecycle_evidence_ontology')
        if ontology!=json.loads((source/side/'lifecycle/ontology_validation.json').read_text()):
            raise ValueError('extended ledger ontology is not bound to its source record')
        if gate is not (ontology.get('valid') is True):
            raise ValueError('extended ledger ontology gate is inconsistent')
        projected=json.loads(json.dumps(original))
        projected.pop('lifecycle_evidence_ontology')
        projected['mandatory_scientific_gates'].pop('complete_lifecycle_evidence_ontology')
        expected=exact_ledger(implementation,source/side)
        if canonical(projected)!=canonical(expected):
            raise ValueError('final ledger differs from exact-head ledger beyond the registered ontology extension')
        target=destination/side/'scientific_ledger.json'
        target.unlink();write(target,expected)
        side_manifest=destination/side/'sha256_manifest.json';side_manifest.unlink()
        write(side_manifest,manifest(destination/side))
        records.append({'side':side,'removed_field':'lifecycle_evidence_ontology',
            'removed_gate':'complete_lifecycle_evidence_ontology',
            'original_ledger_sha256':digest(source/side/'scientific_ledger.json'),
            'projected_ledger_sha256':digest(target)})
    root_manifest=destination/'sha256_manifest.json';root_manifest.unlink()
    write(root_manifest,manifest(destination))
    return records


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
        if LEDGER_SCHEMA_EXTENSION in transcript:
            with tempfile.TemporaryDirectory(prefix='v5-exact-ledger-projection-') as temporary:
                projection=Path(temporary)/'evidence'
                projection_records=project_extended_ledger(implementation,evidence,projection)
                projected_output=Path(temporary)/'audit.json'
                projected=subprocess.run((sys.executable,str(validator),str(projection),str(publication),
                    sha,str(projected_output)),text=True,capture_output=True)
                if projected.returncode:
                    raise RuntimeError('exact-head compatibility projection failed:\n'
                        +(projected.stdout+'\n'+projected.stderr)[-4000:])
                frozen_result=json.loads(projected_output.read_text())
            result={'schema':'v5.exact-publication-head-complete-evidence-audit/3',
                'classification':'PASS_WITH_REGISTERED_LEDGER_SCHEMA_EXTENSION',
                'exact_head_audit_completed':True,'implementation_sha':sha,'publication_sha':sha,
                'exact_recursive_evidence_comparison':True,'evidence_valid':True,
                'scientific_decision':frozen_result['scientific_decision'],'predicate_relaxed':False,
                'clean_implementation_worker_at_end':True,
                'validator_returncode_on_extended_ledger':completed.returncode,
                'validator_returncode_on_lossless_projection':projected.returncode,
                'validator_sha256':hashlib.sha256(validator.read_bytes()).hexdigest(),
                'registered_extension':{'identity':LEDGER_SCHEMA_EXTENSION,
                    'projection_kind':'REMOVE_ONLY_REPAIR_ADDED_ONTOLOGY_LEDGER_FIELD_AND_GATE',
                    'records':projection_records},'frozen_validator_result':frozen_result}
        elif EXPECTED in transcript:
            failure_line=next(line.strip() for line in reversed(transcript.splitlines()) if EXPECTED in line)
            result={'schema':'v5.exact-publication-head-complete-evidence-audit/2',
                'classification':'EXECUTED_BLOCKED','exact_head_audit_completed':True,
                'implementation_sha':sha,'publication_sha':sha,'exact_recursive_evidence_comparison':True,
                'evidence_valid':False,'scientific_decision':'BLOCKED','predicate_relaxed':False,
                'clean_implementation_worker_at_end':True,'validator_returncode':completed.returncode,
                'validator_sha256':hashlib.sha256(validator.read_bytes()).hexdigest(),
                'failure':{'type':'ValueError','identity':EXPECTED,'message':failure_line}}
        else:
            raise RuntimeError('unexpected exact-head validator failure:\n'+transcript[-4000:])
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
