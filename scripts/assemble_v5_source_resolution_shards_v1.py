#!/usr/bin/env python3
"""Assemble disjoint executed shards; never count an assembly as a new solve."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.closure_mechanics_evidence import canonical_data
from arrhenius_fracture.static_numerical_family_v1 import REGISTRY,GROUPS,SCHEMA as STATIC_SCHEMA,classify
from arrhenius_fracture.closure_lifecycle_evidence import SCHEMA as LIFECYCLE_SCHEMA,lifecycle_decision
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
from arrhenius_fracture.finalization_v3_closure_schema import validate_closure_evidence
from validate_v5_source_resolution_evidence_v1 import verify_inventory,validate_static


def write(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(canonical_data(payload),sort_keys=True,indent=2,allow_nan=False)+'\n')


def copy_owned(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).digest()!=hashlib.sha256(source.read_bytes()).digest():
            raise ValueError('conflicting owned checkpoint/source '+str(destination))
    else:shutil.copy2(source,destination)


def assemble(kind,inputs,output):
    if output.exists():raise ValueError('refusing to overwrite assembly')
    if len({p.resolve() for p in inputs})!=len(inputs):raise ValueError('aliased shard directories')
    filename='report.json' if kind=='static' else 'lifecycle_rows.json'
    reports=[];provenance=[]
    for path in inputs:
        inventory=verify_inventory(path);report=json.loads((path/filename).read_text())
        reports.append(report)
        provenance.append({'section':report.get('execution_section','static'),'shard_index':report['shard_index'],
            'shard_count':report['shard_count'],'source_manifest_sha256':hashlib.sha256((path/'sha256_manifest.json').read_bytes()).hexdigest()})
    shas={r['executed_code_sha'] for r in reports}
    if len(shas)!=1:raise ValueError('mixed implementation shards')
    sections=('static',) if kind=='static' else ('transitions','restarts','controlled','neutrality','natural','rollback')
    for section in sections:
        peers=[r for r in reports if r.get('execution_section','static')==section]
        if not peers or len({r['shard_count'] for r in peers})!=1:raise ValueError('missing/inconsistent section '+section)
        count=peers[0]['shard_count']
        if sorted(r['shard_index'] for r in peers)!=list(range(count)):raise ValueError('incomplete/duplicate shards '+section)
    if any(r.get('execution_section','static') not in sections for r in reports):raise ValueError('unexpected section')
    output.mkdir(parents=True)
    if kind=='static':
        rows={}
        for path,report in zip(inputs,reports):
            if report['schema']!=STATIC_SCHEMA or report['sentinel_only'] or report['worktree_status'].strip():
                raise ValueError('invalid static shard provenance')
            for key,row in report['rows'].items():
                if key in rows:raise ValueError('aliased static execution')
                rows[key]=row
                if (path/'sources'/(key+'.npz')).exists():copy_owned(path/'sources'/(key+'.npz'),output/'sources'/(key+'.npz'))
                write(output/(key+'.json'),row)
        if set(rows)!=set(REGISTRY):raise ValueError('incomplete static registry')
        payload={'schema':STATIC_SCHEMA,'executed_code_sha':next(iter(shas)),'worktree_status':'','sentinel_only':False,
            'groups':GROUPS,'rows':rows,'decision':classify(rows),'shard_provenance':sorted(provenance,key=lambda p:p['shard_index']),
            'historical_v3_predicates':'UNCHANGED_RETAINED_683_OF_791'}
    else:
        rows=[];identities=set()
        for path,report in zip(inputs,reports):
            if report['schema']!='v5.source-resolution-lifecycle-shard/1':raise ValueError('wrong lifecycle shard schema')
            for row in report['rows']:
                identity=(row['dataset'],row['case_identity'],row['partition_count'])
                if identity in identities:raise ValueError('aliased lifecycle execution')
                identities.add(identity);rows.append(row)
            for source in path.rglob('*'):
                if source.is_file() and (source.relative_to(path).parts[0]=='checkpoints' or source.name.startswith('rollback_checkpoint')):
                    copy_owned(source,output/source.relative_to(path))
        rows.sort(key=lambda r:(r['dataset'],r['case_identity'],r['partition_count'] or 0))
        class Sources:
            def __getitem__(self,key):return restore_checkpoint(output/key)
        payload={'schema':LIFECYCLE_SCHEMA,'executed_code_sha':next(iter(shas)),'rows':rows,
            'preparation_failures':[r['preparation_failure'] for r in reports if r.get('preparation_failure')],
            'shard_provenance':sorted(provenance,key=lambda p:(p['section'],p['shard_index'])),
            'decision':lifecycle_decision(rows,Sources())}
        for index,row in enumerate(rows):write(output/'rows'/(str(index+1)+'.json'),row)
        write(output/'ontology_validation.json',validate_closure_evidence(payload,Sources(),executed_code_sha=payload['executed_code_sha']))
    write(output/filename,payload)
    write(output/'sha256_manifest.json',{str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob('*')) if p.is_file()})
    return validate_static(output) if kind=='static' else {'valid':True,'actual_state_rows':len(rows),'decision':payload['decision']['decision']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('kind',choices=('static','lifecycle'))
    parser.add_argument('output',type=Path);parser.add_argument('shards',nargs='+',type=Path)
    args=parser.parse_args();print(json.dumps(assemble(args.kind,args.shards,args.output),sort_keys=True))
