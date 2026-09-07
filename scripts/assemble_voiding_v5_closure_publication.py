#!/usr/bin/env python3
"""Package already independently executed A/B bundles and derive their ledger.

This command performs no physical runs and does not claim that byte comparison
proves independent execution. The caller retains the source-run/CI record.
It never overwrites an existing publication directory or historical bundle.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
FAMILIES=('traction','mechanics','production','lifecycle','neutrality')


def inventory(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'sha256_manifest.json'}


def verified(root):
    if any(p.is_symlink() for p in root.rglob('*')): raise ValueError('bundle contains symlink')
    observed=inventory(root)
    if observed!=json.loads((root/'sha256_manifest.json').read_text()):
        raise ValueError('source bundle manifest mismatch: '+str(root))
    if any(p.stat().st_size>=100*1024*1024 for p in root.rglob('*') if p.is_file()):
        raise ValueError('bundle exceeds hosting object-size limit')
    return {**observed,'sha256_manifest.json':hashlib.sha256((root/'sha256_manifest.json').read_bytes()).hexdigest()}


def read(root,name): return json.loads((root/name).read_text())


def regression(path):
    tests=list(ET.parse(path).getroot().iter('testcase'))
    failed=[t for t in tests if t.find('failure') is not None or t.find('error') is not None]
    skipped=[t for t in tests if t.find('skipped') is not None]
    return {'tests':len(tests),'passed':len(tests)-len(failed)-len(skipped),'failed':len(failed),'skipped':len(skipped),
        'failure_ids':[t.attrib['classname'].replace('.','/')+'.py::'+t.attrib['name'] for t in failed],
        'xml_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def ledger(bundle,spec):
    traction=read(bundle/'traction','traction_convergence.json')
    mechanics=read(bundle/'mechanics','mechanics_matrix.json')
    production=read(bundle/'production','transfer_manifest.json')
    lifecycle=read(bundle/'lifecycle','lifecycle_rows.json');decision=lifecycle['decision'];rows=lifecycle['rows']
    neutrality=read(bundle/'neutrality','neutrality_comparison.json')
    rollback=decision['rollback_attempts'];natural=[r for r in rows if r['dataset']=='natural']
    full=regression(Path(spec['full_regression_xml']));focused=regression(Path(spec['focused_regression_xml']))
    baseline=read(ROOT/'artifacts/voiding_v5_semantic_hardening','general_ci_inheritance.json')['base']['failure_ids']
    full['inherited_failure_ids']=sorted(set(full['failure_ids'])&set(baseline))
    full['noninherited_failure_ids']=sorted(set(full['failure_ids'])-set(baseline))
    products=[read(bundle/'production',f'production_{n}_{r}.json') for n,r in production['physical_registry']]
    source_peers=[r['source_refinement_peer'] for r in products if 'source_refinement_peer' in r]
    static_pass=all(r['predicate_result'] for r in mechanics['derived_rows'])
    source_pass=bool(production['production_resolution_cavity_tensor_qualified'] and source_peers
        and any(p.get('qualified') and p.get('event_accepted') for p in source_peers))
    gates={
        'source_tensor_transfer':source_pass,'complete_static_mechanics':static_pass,
        'all_45_physical_transition_partitions':len(decision['transition_partitions'])==45 and all(r['passed'] for r in decision['transition_partitions']),
        'all_11_common_terminal_restarts':len([r for r in rows if r['dataset']=='restarts'])==11
            and decision['required_continued_front_restart_terminal'] and decision['all_restart_stages_reach_identical_complete_terminal'],
        'all_12_controlled_histories':len(decision['controlled_histories'])==12 and all(r['passed'] for r in decision['controlled_histories']),
        'clean_historical_V12_disabled_neutrality':neutrality['decision']=='PASS',
        'complete_22_lifecycle_rollback_stages':sum(r['case_identity'].startswith('lifecycle:') for r in rollback)==22
            and all(r['passed'] for r in rollback if r['case_identity'].startswith('lifecycle:')),
        'natural_seed_partition_restart_rng':len(natural)==160 and all(r['passed'] for r in decision['natural_partitions_restart']),
        'stagewise_conservation_and_topology':decision['stagewise_topology_and_conservation'],
        'focused_regressions':focused['failed']==0 and focused['tests']>0,
    }
    return {'schema':'v12.complete-closure-derived-ledger/1','record_kind':'DERIVED_AUDIT_NOT_ADDITIONAL_PHYSICAL_EXECUTIONS',
        'audit_code_sha':subprocess.check_output(('git','rev-parse','HEAD'),cwd=ROOT,text=True).strip(),
        'source_implementation_shas':{'traction':traction['executed_code_sha'],'mechanics':mechanics['executed_code_sha'],
            'production':production['executed_code_sha'],'lifecycle':lifecycle['executed_code_sha'],
            'neutrality_base':neutrality['base_sha'],'neutrality_current':neutrality['current_sha']},
        'mandatory_scientific_gates':gates,'scientific_decision':'PASS' if all(gates.values()) else 'BLOCKED',
        'campaign_execution_counts':{'static_unique_solves':len(mechanics['base_rows']),'production_resolutions':len(products),
            'lifecycle_datasets':dict(Counter(r['dataset'] for r in rows))},
        'static_predicates':{'passed':sum(r['predicate_result'] for r in mechanics['derived_rows']),
            'failed':sum(not r['predicate_result'] for r in mechanics['derived_rows'])},
        'source_refinement_peers':source_peers,'production_comparisons':production['comparisons'],
        'lifecycle_decision':decision,'natural_phase_counts':dict(Counter(r['terminal_classification'] for r in natural if r['partition_count']==1)),
        'natural_elapsed_window_completed':sum(r['failure'] is None for r in natural),
        'historical_neutrality_decision':neutrality['decision'],
        'full_regressions':{**full,'implementation_sha':spec['full_regression_sha'],'classification':'NOT_GREEN' if full['failed'] else 'GREEN'},
        'focused_regressions':{**focused,'implementation_sha':spec['focused_regression_sha']},
        'exact_head_ci':'EXTERNAL_TERMINAL_RUN_RECORD_REQUIRED_SEPARATELY',
        'mission_terminal':False,'no_rc_or_merge':True}


def write(path,payload):
    path.write_text(json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('specification',type=Path)
    parser.add_argument('output',type=Path);args=parser.parse_args();spec=read(args.specification.parent,args.specification.name)
    if args.output.exists(): raise ValueError('refusing to overwrite publication directory')
    for family in FAMILIES:
        a,b=(Path(spec[family][side]).resolve() for side in ('a','b'))
        if a==b: raise ValueError('A and B must be different executed source directories')
        if verified(a)!=verified(b): raise ValueError('nonidentical A/B inventory: '+family)
    for side in ('a','b'):
        destination=args.output/side;destination.mkdir(parents=True)
        for family in FAMILIES: shutil.copytree(spec[family][side],destination/family)
        write(destination/'closure_ledger.json',ledger(destination,spec))
        # Only these top-level manifests omit themselves; nested source
        # manifests remain part of the complete publication inventory.
        write(destination/'sha256_manifest.json',{str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(destination.rglob('*')) if p.is_file()})
    a,b=args.output/'a',args.output/'b'
    comparison={str(p.relative_to(a)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.rglob('*')) if p.is_file()}
    peer={str(p.relative_to(b)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(b.rglob('*')) if p.is_file()}
    if comparison!=peer: raise ValueError('assembled complete A/B differs')
    write(args.output/'paired_comparison.json',{'exact_recursive_comparison':True,'file_count_per_side':len(comparison),
        'meaning':'BYTE_COMPARISON_OF_SEPARATELY_EXECUTED_SOURCE_BUNDLES_NOT_A_NEW_EXECUTION',
        'scientific_decision':read(a,'closure_ledger.json')['scientific_decision']})
    print(json.dumps(read(args.output,'paired_comparison.json')))


if __name__=='__main__':main()
