#!/usr/bin/env python3
"""Small actual negative/positive source and child sentinel, never broad replay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.closure_mechanics_evidence import canonical_data
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint
from arrhenius_fracture.voiding_production_v5 import refine_downstream_source, downstream_front_transaction, cavity_source_resolution_metrics
from arrhenius_fracture.closure_lifecycle_evidence import conservation, stagewise_topology


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_bundle', type=Path); parser.add_argument('output', type=Path)
    parser.add_argument('--segments', type=int, default=512); parser.add_argument('--layers', type=int, default=192)
    parser.add_argument('--common-restart-reload',action='store_true',
        help='Bounded development sentinel for the frozen compressive/8e-7 common restart protocol')
    args = parser.parse_args(); source, output = args.source_bundle, args.output
    if output.exists(): raise ValueError('refusing to overwrite source sentinel')
    expected = json.loads((source/'sha256_manifest.json').read_text())
    actual = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file() and p.name != 'sha256_manifest.json'}
    if actual != expected: raise ValueError('trusted source inventory/hash mismatch')
    state = restore_checkpoint(source/f'checkpoints/production_{args.segments}_{args.layers}_connected.json')
    original = fingerprint(state); output.mkdir(parents=True)
    write_checkpoint(state,output/'initial_unqualified_state.json',compression='gzip')
    report = {'schema': 'v5.source-resolution-positive-sentinel/1',
        'executed_code_sha': subprocess.check_output(('git', 'rev-parse', 'HEAD'), cwd=ROOT, text=True).strip(),
        'worktree_status': subprocess.check_output(('git', 'status', '--short'), cwd=ROOT, text=True),
        'source_case': [args.segments, args.layers], 'initial_state_fingerprint': original,
        'source_bundle_manifest_sha256': hashlib.sha256((source/'sha256_manifest.json').read_bytes()).hexdigest()}
    def retain():
        (output/'result.json').write_text(json.dumps(canonical_data(report), sort_keys=True, indent=2, allow_nan=False)+'\n')
    print('Actual unqualified negative-control event attempt', flush=True)
    negative, event, operations, audit = downstream_front_transaction(state)
    write_checkpoint(negative,output/'negative_guarded_state.json',compression='gzip')
    report['negative'] = {'full_state_unchanged': fingerprint(negative) == original,
        'event_created': event is not None, 'operations': operations, 'source_audit': audit}
    retain()
    if args.common_restart_reload:
        from arrhenius_fracture.closure_lifecycle_evidence import prepare_common_restart_reload
        load_operations=[]
        state=prepare_common_restart_reload(state,load_operations)
        report['common_restart_load_operations']=load_operations
        write_checkpoint(state,output/'accepted_common_reload_state.json',compression='gzip')
        retain()
    source_initial=fingerprint(state)
    print('Quality-valid parent, actual ring refinement, frozen source/time certification', flush=True)
    try:
        qualified, proof = refine_downstream_source(state, max_refinement_levels=2 if args.common_restart_reload else 1,
            refinement_region='complete_cavity_ring', quality_improvement='constrained_v1')
        report['qualification'] = proof; report['qualified'] = proof['status'] == 'SOURCE_TENSOR_QUALIFIED'
        report['preexisting_clocks_preserved'] = qualified.competition == state.competition and qualified.rng_state == state.rng_state
        write_checkpoint(qualified, output/'qualified_or_preserved_state.json',compression='gzip'); retain()
        if report['qualified']:
            print('Attempting real downstream first passage and child transaction', flush=True)
            child, event, operations, audit = downstream_front_transaction(qualified)
            report['child'] = {'accepted': event is not None and event.accepted, 'source_audit': audit,
                'operations': operations, 'fingerprint': fingerprint(child), 'active_graph_tips': child.crack_network.active_tip_ids,
                'active_support_tips': child.v12_support_state.active_tip_identities,
                'conservation': conservation(child, qualified), 'topology': stagewise_topology(child)}
            write_checkpoint(child, output/'child_or_rejected_state.json',compression='gzip'); retain()
            if report['child']['accepted']:
                print('Attempting ordinary child-tip continuation', flush=True)
                continued, event, operations, audit = downstream_front_transaction(child, continuation=True)
                report['continuation'] = {'accepted': event is not None and event.accepted,
                    'operations': operations, 'source_audit': audit, 'fingerprint': fingerprint(continued),
                    'conservation': conservation(continued, child), 'topology': stagewise_topology(continued)}
                write_checkpoint(continued, output/'continued_or_rejected_state.json',compression='gzip')
    except Exception as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    report['original_caller_unchanged'] = fingerprint(state) == source_initial
    retain()
    (output/'sha256_manifest.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir()) if p.is_file()}, sort_keys=True, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('qualified', 'failure', 'original_caller_unchanged') if k in report}), flush=True)


if __name__ == '__main__': main()
