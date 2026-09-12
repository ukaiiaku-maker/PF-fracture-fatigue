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


def validate_causality(root):
    """Recalculate intervention tensors/rates and graph/topology from owned states."""
    from arrhenius_fracture.voiding_production_v5 import crack_tip_tensor,cavity_boundary_tensor,directional_clock_rates
    verify_inventory(root);report=json.loads((root/'report.json').read_text())
    if report['worktree_status'].strip():raise ValueError('causality not from clean implementation')
    expected=('initial_cavity_source_changed','ordinary_child_continuation','changed_cavity_during_continuation',
        'changed_child_tip_drive','zero_child_tip_drive','changed_child_owned_radius')
    rows={row['case']:row for row in report['rows']}
    if len(report['rows'])!=6 or set(rows)!=set(expected):raise ValueError('incomplete actual source-intervention registry')
    states={};computed_rates={}
    for name in expected:
        row=rows[name];before=restore_checkpoint(root/(name+'_initial.json'));after=restore_checkpoint(root/(name+'.json'))
        states[name]=before
        same(fingerprint(before),row['initial_fingerprint'],'intervention initial ownership mismatch')
        same(fingerprint(after),row['terminal_fingerprint'],'intervention terminal ownership mismatch')
        same(stagewise_topology(after),row['topology'],'intervention topology does not recompute')
        same(conservation(after,before),row['conservation'],'intervention conservation does not recompute')
        def graph(state):return [(b.branch_id,b.path,b.status) for b in state.crack_network.branches]
        same(row['accepted'],canonical_data(graph(before))!=canonical_data(graph(after)),'intervention event classification mismatch')
        if row['continuation']:
            tensor,ids=crack_tip_tensor(before,branch_id='void-front-1')
        else:
            cavity=before.void_state.cavities[0]
            node=int(np.argmin(np.linalg.norm(before.mesh.nodes-np.asarray(cavity.connection_exit_m),axis=1)))
            tensor,ids=cavity_boundary_tensor(before,boundary_node=node)
        calls=row['probe_calls'];intervention=row['intervention']
        relevant=(intervention['probe']=='crack_tip_tensor')==row['continuation']
        if relevant:
            if not calls:raise ValueError('source intervention never sampled its declared probe')
            same(calls[0]['original_tensor_Pa'],tensor,'intervention probe is not source-native')
            same(calls[0]['element_ids'],list(ids),'intervention tensor stencil mismatch')
            tensor=intervention['factor']*tensor
        elif calls:raise ValueError('continuation sampled its forbidden cavity source')
        same(row['audit']['tensor_Pa'],tensor,'event tensor does not match actual intervention')
        rates=directional_clock_rates(before,tensor)
        computed_rates[name]=[r['effective_rate_s'] for r in rates]
        if row['continuation']:
            for key in ('candidate_id','effective_rate_s','hazard_barrier_J'):
                same([r[key] for r in row['audit']['cleavage']],[r[key] for r in rates],
                    'actual cleavage law does not reproduce intervention '+key)
    baseline=rows['ordinary_child_continuation'];irrelevant=rows['changed_cavity_during_continuation']
    zero=rows['zero_child_tip_drive'];changed=rows['initial_cavity_source_changed']
    child=states['ordinary_child_continuation'];altered=states['changed_child_owned_radius']
    radius=child.tip_process_state['by_branch']['void-front-1']['r_tip_m']
    if altered.tip_process_state['by_branch']['void-front-1']['r_tip_m']!=2*radius:
        raise ValueError('radius intervention did not change its process owner')
    if altered.crack_network.branch('void-front-1').local_state['r_tip_m']!=2*radius:
        raise ValueError('radius intervention did not change its branch owner')
    same(altered.void_state,child.void_state,'radius intervention changed the cavity')
    gates={
        'changed_cavity_source_rejected_without_recognition_as_qualified':not changed['accepted']
            and changed['audit'].get('status')=='UNQUALIFIED_CAVITY_SOURCE_TENSOR',
        'actual_ordinary_child_continuation':baseline['accepted'],
        'cavity_probe_cannot_influence_continuation':baseline['accepted'] and irrelevant['accepted']
            and not irrelevant['probe_calls'] and baseline['terminal_fingerprint']==irrelevant['terminal_fingerprint'],
        'child_tensor_changes_actual_continuation_rate':baseline['accepted'] and rows['changed_child_tip_drive']['accepted']
            and bool(computed_rates['ordinary_child_continuation'])
            and computed_rates['ordinary_child_continuation']!=computed_rates['changed_child_tip_drive'],
        'zero_child_tip_drive_creates_no_event':not zero['accepted'] and zero['failure'] is None
            and zero['audit'].get('status')=='NO_KINETICALLY_ACTIVE_CANDIDATE',
        'owned_radius_causally_enters_continuation_law':bool(computed_rates['ordinary_child_continuation'])
            and computed_rates['changed_child_owned_radius']!=computed_rates['ordinary_child_continuation'],
        'radius_separate_from_void_radius':radius!=child.void_state.cavities[0].radius_m}
    same(gates,report['gates'],'source-causality predicates do not recompute')
    same(all(gates.values()),report['passed'],'source-causality classification mismatch')
    return {'valid':True,'actual_interventions':6,'science_passed':all(gates.values())}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('kind',choices=('static','positive','causality'))
    parser.add_argument('directory',type=Path);args=parser.parse_args()
    print(json.dumps({'static':validate_static,'positive':validate_positive,'causality':validate_causality}[args.kind](args.directory),sort_keys=True))
