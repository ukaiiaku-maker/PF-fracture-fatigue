#!/usr/bin/env python3
"""Strict source reconstruction of one complete development lifecycle section.

Trusted authorized CI outputs only: inventory validation precedes unpickling.
This is not a new physical execution or a substitute for the final full ontology.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import platform
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from validate_v5_source_resolution_evidence_v1 import verify_inventory,same
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
from arrhenius_fracture.closure_lifecycle_evidence import validate_lifecycle_rows,lifecycle_decision,PARTITIONS
from arrhenius_fracture.closure_rollback_matrix_v5 import ROLLBACK_STAGES
from arrhenius_fracture.finalization_v3_schema import FROZEN_CASE_REGISTRY


def expected_registry(section):
    if section=='transitions':return {(case,p) for case in FROZEN_CASE_REGISTRY[section] for p in PARTITIONS}
    if section=='restarts':return {(case,None) for case in FROZEN_CASE_REGISTRY[section]}
    if section!='rollback':raise ValueError('unsupported development section')
    cases={'lifecycle:'+name for name in ROLLBACK_STAGES}
    cases.update('promotion:'+name for name in ('field_projection','support_rebuild','equilibrium'))
    cases.update('ligament:'+name for name in ('graph_edit','remesh','field_projection','support_rebuild',
        'equilibrium','energy_gate','connected_surface_certification','dormant_support_rebuild'))
    return {(case,None) for case in cases}


def require_registry(section,rows):
    if len({r['execution_id'] for r in rows})!=len(rows):raise ValueError('aliased development execution IDs')
    selected=[r for r in rows if r['dataset']==section]
    observed=[(r['case_identity'],r['partition_count']) for r in selected]
    if len(observed)!=len(set(observed)) or set(observed)!=expected_registry(section):
        raise ValueError('incomplete or aliased development '+section+' registry')
    if any(r['dataset'] not in (section,'stagewise') for r in rows):raise ValueError('unrelated section rows')
    if section!='transitions' and len(selected)!=len(rows):raise ValueError('unexpected stagewise rows')
    return selected


def validate(section,inputs):
    if len({p.resolve() for p in inputs})!=len(inputs):raise ValueError('aliased shard directories')
    rows=[];reports=[];heads=set();owned={};hashes={};provenance=[]
    for directory in inputs:
        outer=verify_inventory(directory);execution=json.loads((directory/'execution.json').read_text())
        if not execution['execution_completed'] or not execution['clean_exact_head_at_end']:
            raise ValueError('incomplete development execution')
        root=directory/'evidence';inventory=verify_inventory(root)
        report=json.loads((root/'lifecycle_rows.json').read_text())
        if report['schema']!='v5.source-resolution-lifecycle-shard/1' or report['execution_section']!=section:
            raise ValueError('wrong development section identity')
        if execution['executed_code_sha']!=report['executed_code_sha']:raise ValueError('mixed producer identity')
        heads.add(report['executed_code_sha']);reports.append(report);rows.extend(report['rows'])
        provenance.append({'shard_index':report['shard_index'],'shard_count':report['shard_count'],
            'source_manifest_sha256':outer['evidence/sha256_manifest.json']})
        for key,digest in inventory.items():
            if key.startswith('checkpoints/'):
                if key in hashes and hashes[key]!=digest:raise ValueError('conflicting owned checkpoint '+key)
                hashes[key]=digest;owned[key]=root/key
    if len(heads)!=1 or len({r['shard_count'] for r in reports})!=1:raise ValueError('mixed implementation/shard counts')
    if sorted(r['shard_index'] for r in reports)!=list(range(reports[0]['shard_count'])):
        raise ValueError('missing or duplicate development shards')
    selected=require_registry(section,rows);sha=next(iter(heads))
    class Sources:
        @lru_cache(maxsize=2)
        def __getitem__(self,key):
            if key not in owned:raise ValueError('checkpoint not in verified owned inventory')
            return restore_checkpoint(owned[key])
    sources=Sources()
    for row in rows:
        validate_lifecycle_rows([row],sources,executed_code_sha=sha)
        print('Reconstructed '+row['dataset']+':'+row['case_identity']+':'+str(row['partition_count']),flush=True)
    for report in reports:same(lifecycle_decision(report['rows'],sources),report['decision'],'development decision mismatch')
    decision=lifecycle_decision(rows,sources)
    if section=='restarts':
        passed=decision['required_continued_front_restart_terminal'] and decision['all_restart_stages_reach_identical_complete_terminal']
        passed &= all(r['restart_exact'] and r['failure'] is None and r.get('direct_failure') is None
            and r.get('replay_failure') is None for r in selected)
        successes=sum(r['restart_exact'] and r['continued_front_terminal_reached'] and r['subsequent_history_exact']
            and r['failure'] is None and r['conservation']['passed'] and r['stagewise_topology']['passed'] for r in selected)
    else:
        gates=decision['transition_partitions' if section=='transitions' else 'rollback_attempts']
        successes=sum(r['passed'] for r in gates);passed=all(r['passed'] for r in gates)
    passed &= decision['stagewise_topology_and_conservation']
    return {'schema':'v5.development-lifecycle-source-reconstruction/1',
        'record_kind':'READ_ONLY_RECONSTRUCTION_NOT_NEW_PHYSICAL_EXECUTION','section':section,
        'source_sha':sha,'verification_code_sha':subprocess.check_output(('git','rev-parse','HEAD'),cwd=ROOT,text=True).strip(),
        'python_version':platform.python_version(),'valid':True,'complete_section_registry':True,
        'complete_final_campaign':False,'actual_section_cases':len(selected),'actual_source_rows':len(rows),
        'passed_cases':successes,'science_passed':bool(passed),'decision':decision,
        'checkpoint_assembly_conflicts':False,'source_provenance':sorted(provenance,key=lambda r:r['shard_index'])}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('section',choices=('transitions','restarts','rollback'))
    parser.add_argument('output',type=Path);parser.add_argument('shards',nargs='+',type=Path);args=parser.parse_args()
    result=validate(args.section,args.shards);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('decision','source_provenance')}),flush=True)
