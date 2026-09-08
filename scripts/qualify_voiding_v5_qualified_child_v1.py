#!/usr/bin/env python3
"""Actual event attempt from a retained source-qualified checkpoint, no replay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from arrhenius_fracture import voiding_production_v5 as p
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.closure_mechanics_evidence import canonical_data
from arrhenius_fracture.closure_lifecycle_evidence import conservation, stagewise_topology
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path); parser.add_argument('output', type=Path)
    parser.add_argument('--continuation', action='store_true')
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('refusing to overwrite evidence')
    before = restore_checkpoint(args.checkpoint); original = fingerprint(before)
    args.output.mkdir(parents=True)
    report = {'schema': 'v5.qualified-source-real-child-sentinel/1',
        'implementation_sha': subprocess.check_output(('git', 'rev-parse', 'HEAD'), cwd=ROOT, text=True).strip(),
        'worktree_status': subprocess.check_output(('git', 'status', '--short'), cwd=ROOT, text=True),
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'input_fingerprint': original, 'continuation': args.continuation, 'operations': []}
    def retain():
        (args.output/'result.json').write_text(json.dumps(canonical_data(report), sort_keys=True, indent=2, allow_nan=False)+'\n')
    original_prepare = p._refresh_downstream_boundary_context
    def observe(trial):
        prepared = original_prepare(trial)
        write_checkpoint(prepared, args.output/'UNACCEPTED_support_trial.json')
        report['unaccepted_trial_retained'] = True; retain()
        return prepared
    try:
        with patch.object(p, '_refresh_downstream_boundary_context', observe):
            after, event, operations, audit = p.downstream_front_transaction(before, continuation=args.continuation,
                operation_log=report['operations'])
        report.update(accepted=event is not None and event.accepted, source_audit=audit,
            result_fingerprint=fingerprint(after), conservation=conservation(after, before),
            topology=stagewise_topology(after), active_graph_tips=after.crack_network.active_tip_ids,
            active_support_tips=after.v12_support_state.active_tip_identities)
        write_checkpoint(after, args.output/'accepted_or_preserved_state.json')
    except Exception as exc:
        report.update(accepted=False, failure={'type': type(exc).__name__, 'message': str(exc)})
    report['caller_unchanged'] = fingerprint(before) == original; retain()
    (args.output/'sha256_manifest.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.output.iterdir()) if p.is_file()}, sort_keys=True, indent=2)+'\n')
    print(json.dumps({'accepted': report['accepted'], 'caller_unchanged': report['caller_unchanged'],
        'failure': report.get('failure', {}).get('message', '')[:300]}), flush=True)


if __name__ == '__main__': main()
