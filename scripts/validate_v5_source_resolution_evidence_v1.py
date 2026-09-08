#!/usr/bin/env python3
"""Recompute new source-resolution predicates from hash-verified owned sources.

Checkpoint payloads must be trusted local/authorized CI outputs. Hash validation
protects integrity, not authenticity of arbitrary pickle input.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.finalization_v3_schema import canonical_hash
from arrhenius_fracture.closure_mechanics_evidence import canonical_data,measurements
from arrhenius_fracture.closure_static_evidence import validate_solver_capture,source_fingerprints
from arrhenius_fracture.static_numerical_family_v1 import REGISTRY,GROUPS,SCHEMA,classify
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint
from arrhenius_fracture.closure_lifecycle_evidence import conservation,stagewise_topology
from arrhenius_fracture.voiding_production_v5 import _qualified_cavity_source,cavity_source_resolution_metrics
from qualify_cavity_boundary_patch_recovery_v1 import recover_arcs


def verify_inventory(root):
    root=Path(root)
    if any(p.is_symlink() for p in root.rglob('*')):raise ValueError('symlink in evidence')
    actual={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'sha256_manifest.json'}
    if actual!=json.loads((root/'sha256_manifest.json').read_text()):raise ValueError('evidence inventory/hash mismatch')
    return actual


def same(a,b,message):
    if canonical_hash(canonical_data(a))!=canonical_hash(canonical_data(b)):raise ValueError(message)


def validate_static(root):
    verify_inventory(root);report=json.loads((root/'report.json').read_text())
    if report['schema']!=SCHEMA or report['sentinel_only'] or report['worktree_status'].strip():
        raise ValueError('not a clean full prospective static qualification')
    same(report['groups'],GROUPS,'static family registry mismatch')
    if set(report['rows'])!=set(REGISTRY):raise ValueError('static solve registry incomplete')
    for key,cfg in REGISTRY.items():
        row=report['rows'][key]
        same(row['input_configuration'],cfg,'static input mismatch')
        if row['executed_code_sha']!=report['executed_code_sha']:raise ValueError('static source implementation mismatch')
        if 'failure' in row:
            if not row['failure'].get('type') or not row['failure'].get('message'):raise ValueError('empty failed-solve record')
            continue
        with np.load(root/'sources'/(key+'.npz'),allow_pickle=False) as source:raw=dict(source)
        same(source_fingerprints(raw),row['source_fingerprints'],'raw static source mismatch')
        same(validate_solver_capture(raw,cfg),row['solver_validation'],'independent static equilibrium mismatch')
        same(measurements(raw,cfg),row['measurements'],'static mechanics measurements do not recompute')
        same(recover_arcs(raw,cfg['cavity_center_m']) if cfg['cavity_enabled'] else [],row['recovery'],
            'static recovery stencil/tensor mismatch')
    same(classify(report['rows']),report['decision'],'static classification mismatch')
    return {'valid':True,'unique_physical_solves':len(REGISTRY),'science_passed':report['decision']['passed']}


def validate_positive(root):
    verify_inventory(root);report=json.loads((root/'result.json').read_text())
    if report['worktree_status'].strip():raise ValueError('source pair not from clean implementation')
    before=restore_checkpoint(root/'initial_unqualified_state.json')
    negative=restore_checkpoint(root/'negative_guarded_state.json')
    if fingerprint(before)!=report['initial_state_fingerprint']:raise ValueError('initial source mismatch')
    same(report['negative']['full_state_unchanged'],fingerprint(negative)==fingerprint(before),'negative full-state claim mismatch')
    same(report['negative']['event_created'],False,'unqualified negative created an event')
    qualified=restore_checkpoint(root/'qualified_or_preserved_state.json')
    positive=_qualified_cavity_source(qualified,cavity_source_resolution_metrics(qualified)['tensor_Pa'])
    same(report.get('qualified',False),positive,'source qualification does not recompute')
    same(report['preexisting_clocks_preserved'],before.competition==qualified.competition and before.rng_state==qualified.rng_state,
        'source transfer clock/RNG claim mismatch')
    last=qualified
    for name,filename in (('child','child_or_rejected_state.json'),('continuation','continued_or_rejected_state.json')):
        if name not in report:continue
        after=restore_checkpoint(root/filename);row=report[name]
        same(fingerprint(after),row['fingerprint'],'accepted event checkpoint mismatch')
        same(conservation(after,last),row['conservation'],'event conservation mismatch')
        same(stagewise_topology(after),row['topology'],'event independent topology mismatch')
        if row['accepted']:
            if after.crack_network.active_tip_ids!=('void-front-1',):raise ValueError('accepted event lacks sole active child')
            if name=='child' and len(after.crack_network.branches)!=len(last.crack_network.branches)+1:
                raise ValueError('child event multiplicity mismatch')
            if name=='continuation' and after.crack_network.branch('void-front-1').tip==last.crack_network.branch('void-front-1').tip:
                raise ValueError('continued event did not advance its child')
        last=after
    return {'valid':True,'source_qualified':positive,'actual_child':report.get('child',{}).get('accepted',False),
        'actual_continuation':report.get('continuation',{}).get('accepted',False)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('kind',choices=('static','positive'))
    parser.add_argument('directory',type=Path);args=parser.parse_args()
    print(json.dumps((validate_static if args.kind=='static' else validate_positive)(args.directory),sort_keys=True))
