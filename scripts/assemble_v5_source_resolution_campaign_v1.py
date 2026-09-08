#!/usr/bin/env python3
"""Join the one full independently executed A/B campaign and derive its ledger."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from assemble_v5_source_resolution_shards_v1 import assemble,write
from validate_v5_source_resolution_evidence_v1 import verify_inventory,validate_positive,validate_causality
from v5_source_resolution_campaign_matrix_v1 import matrix


def read(root,name):return json.loads((root/name).read_text())


def derive_ledger(root):
    static=read(root/'static','report.json');life=read(root/'lifecycle','lifecycle_rows.json');d=life['decision']
    positive=read(root/'source/positive','result.json');causality=read(root/'source/causality','report.json')
    coarse=read(root/'source/production','production_32_12.json')
    neutrality=read(root/'causal-neutrality/evidence','report.json')
    rollbacks=d['rollback_attempts'];restarts=[r for r in life['rows'] if r['dataset']=='restarts']
    gates={
        'qualified_production_source':positive.get('qualified',False) and positive.get('preexisting_clocks_preserved',False),
        'coarse_unqualified_negative_retained':coarse.get('guard_preserves_complete_state',False)
            and not coarse.get('event_transaction_created',True) and not coarse.get('child_created',True),
        'actual_downstream_child':positive.get('child',{}).get('accepted',False),
        'ordinary_child_continuation_and_source_causality':causality['passed'],
        'complete_static_numerical_families':static['decision']['passed'] and len(static['rows'])==96,
        'all_45_transition_partitions':len(d['transition_partitions'])==45 and all(r['passed'] for r in d['transition_partitions']),
        'all_11_common_terminal_restarts':len(restarts)==11 and d['required_continued_front_restart_terminal']
            and d['all_restart_stages_reach_identical_complete_terminal'] and all(r['restart_exact'] and r['failure'] is None for r in restarts),
        'all_12_controlled_histories':len(d['controlled_histories'])==12 and all(r['passed'] for r in d['controlled_histories']),
        'disabled_future_causal_neutrality':neutrality['passed'],
        'all_28_lifecycle_rollback_stages':sum(r['case_identity'].startswith('lifecycle:') for r in rollbacks)==28
            and all(r['passed'] for r in rollbacks),
        'natural_32_seed_160_partition_restart_peers':len(d['natural_partitions_restart'])==160 and all(r['passed'] for r in d['natural_partitions_restart']),
        'stagewise_topology_and_conservation':d['stagewise_topology_and_conservation'] and all(r['actual_transition'] for r in d['transition_partitions']),
        'same_head_disabled_dispatch':len(d['V12_disabled_neutrality'])==4 and all(r['passed'] for r in d['V12_disabled_neutrality']),
    }
    counts=lambda rows:{'passed':sum(bool(r['passed']) for r in rows),'total':len(rows)}
    return {'schema':'v5.source-resolution-final-scientific-ledger/1','record_kind':'DERIVED_NOT_ADDITIONAL_EXECUTION',
        'executed_code_sha':life['executed_code_sha'],'mandatory_scientific_gates':gates,
        'scientific_decision':'PASS' if all(gates.values()) else 'BLOCKED','release_candidate_authorized':False,
        'static_families':counts(static['decision']['families']),'static_derivatives':counts(static['decision']['derivatives']),
        'transition_partitions':counts(d['transition_partitions']),'controlled_histories':counts(d['controlled_histories']),
        'rollback':counts(rollbacks),'natural_partitions_restart':counts(d['natural_partitions_restart']),
        'restart_exact':sum(r['restart_exact'] for r in restarts),'common_terminal_exact':d['all_restart_stages_reach_identical_complete_terminal'],
        'source_causality_gates':causality['gates'],'historical_raw_full_state_identity':'FAIL_RETAINED',
        'historical_static_result':'UNCHANGED_683_OF_791','mission_terminal':False,
        'remaining_publication_requirements':['exact-head CI terminal result','repository-wide inherited failure classification','final PR63 ledger']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('shards',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('refusing to overwrite campaign')
    specs=matrix();shas=set()
    for spec in specs:
        for side in ('a','b'):
            path=args.shards/spec['id']/side;verify_inventory(path);execution=read(path,'execution.json')
            if not execution['execution_completed']:raise ValueError('phase execution incomplete '+spec['id']+'/'+side)
            for key in ('phase','section'):
                if execution[key]!=spec[key]:raise ValueError('wrong phase identity')
            if execution['shard_index']!=spec['index'] or execution['shard_count']!=spec['count']:raise ValueError('wrong shard identity')
            shas.add(execution['executed_code_sha'])
    if len(shas)!=1:raise ValueError('mixed exact implementation heads')
    for side in ('a','b'):
        destination=args.output/side;destination.mkdir(parents=True)
        for kind in ('static','lifecycle'):
            sources=[args.shards[s['id']]/side/'evidence' for s in specs if s['phase']==kind]
            assemble(kind,sources,destination/kind)
        for kind in ('source','recovery','causal-neutrality'):
            shutil.copytree(args.shards/kind/side,destination/kind)
        validate_positive(destination/'source/positive')
        validate_causality(destination/'source/causality')
        write(destination/'scientific_ledger.json',derive_ledger(destination))
        write(destination/'sha256_manifest.json',{str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(destination.rglob('*')) if p.is_file()})
    a,b=(verify_inventory(args.output/side) for side in ('a','b'))
    comparison={'schema':'v5.source-resolution-final-paired-comparison/1','exact_recursive_comparison':a==b,
        'file_count_per_side':[len(a),len(b)],'executed_code_sha':next(iter(shas)),
        'classification':'PASS' if a==b else 'FAIL','independent_execution_record':'DEDICATED_WORKFLOW_DISJOINT_A_AND_B_INVOCATIONS'}
    write(args.output/'paired_comparison.json',comparison)
    write(args.output/'sha256_manifest.json',{str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.output.rglob('*')) if p.is_file()})
    print(json.dumps(comparison),flush=True)
    if a!=b:raise ValueError('full A/B recursive inventory mismatch')


if __name__=='__main__':main()
