"""Single authorized step-419 replay/continuation; original artifacts immutable."""
import argparse
from dataclasses import asdict
import inspect
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import traceback

from scripts.run_v13_primary_race_short_ensemble import (
    OUT, PLAN, CASES, case_folder, terminal_record, ROOT, ROWS,
    campaign_environment, atomic_json, sha256)

CASE = 'Peak_300K_seed3624'
EXPECTED = '3209c19513c44557a222aed5b8b026f71202fd8ab7655f24fc573229ac8c27cf'
DEST = OUT / 'tip_local_recovery'


class FrozenAcceptedStop(Exception):
    """Diagnostic sentinel after the first atomic accepted checkpoint."""


def load_frozen_interval():
    """Read the forensic pickle, binding only known installed MPZ functions.

    Production checkpoints use explicit field serialization. The diagnostic
    caller-frame capture also contains a live engine with instance-bound
    methods, whose function names are not class attributes. Resolve those
    names through the existing closed registry; no constructor or RNG draw.
    """
    from types import MethodType
    from arrhenius_fracture.current_source_runtime_bindings import BINDING_REGISTRY
    functions = {f.__name__: f for bindings in BINDING_REGISTRY.values() for f in bindings.values()}
    def known_getattr(obj, name):
        if type(obj).__name__ == 'UnifiedMPZState' and name in functions:
            return MethodType(functions[name], obj)
        return getattr(obj, name)
    class Reader(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'builtins' and name == 'getattr':
                return known_getattr
            return super().find_class(module, name)
    return Reader(io.BytesIO((DEST/'frozen_failure/failed_interval.pkl').read_bytes())).load()


def freeze():
    DEST.mkdir(exist_ok=True)
    target = DEST / 'input_hashes.json'
    if target.exists():
        verify()
        return
    paths = []
    completed = []
    for case in CASES:
        folder = case_folder(case)
        if (folder / 'terminal.json').exists():
            if json.loads((folder / 'terminal.json').read_text())['status'] == 'TERMINATED':
                completed.append(case)
            paths.extend(p for p in folder.rglob('*') if p.is_file())
    assert len(completed) == 13
    paths += [OUT / 'marked_event_readonly_audit/prebranch_opportunity_table.json', PLAN]
    atomic_json(target, {'completed_cases': completed, 'files': {str(p): sha256(p) for p in sorted(set(paths))}})


def verify():
    record = json.loads((DEST / 'input_hashes.json').read_text())
    changed = [p for p, h in record['files'].items() if sha256(Path(p)) != h]
    if changed:
        raise RuntimeError('immutable input changed: ' + repr(changed))
    return len(record['files'])


def run(mode):
    freeze()
    if mode == 'continuation':
        gate = json.loads((DEST/'qualification.json').read_text())
        if gate['status'] != 'PASS' or gate['narrow_regression_count'] != 10:
            raise RuntimeError('frozen qualification has not passed')
        if any(sha256(ROOT/name) != h for name,h in gate['production_source_sha256'].items()):
            raise RuntimeError('production source changed after frozen qualification')
    original = OUT / 'short_ensemble' / CASE
    source = original / 'checkpoint/latest.json'
    manifest = json.loads(source.read_text())
    assert manifest['event_counters']['accepted_steps'] == 419
    assert sha256(source.parent / manifest['state_file']) == EXPECTED
    folder = DEST / mode
    folder.mkdir()
    atomic_json(folder / 'launch_claim.json', {'pid': os.getpid(), 'mode': mode, 'source_checkpoint_sha256': EXPECTED})
    inputs = folder / 'input_checkpoint'
    inputs.mkdir()
    cp = inputs / 'step0000419.json'
    shutil.copyfile(source, cp)
    shutil.copyfile(source.parent / manifest['state_file'], inputs / manifest['state_file'])
    shutil.copyfile(original / 'v13_primary_race.jsonl', folder / 'v13_primary_race.jsonl')
    launch = json.loads((original / 'launch.json').read_text())
    family = Path(json.loads(PLAN.read_text())['family'])
    os.environ.update(campaign_environment(family))
    os.environ.update(CLEAVAGE_HAZARD_SEED='3624', PF_QUALIFIED_DAUGHTER_STOP_UM='25')
    from arrhenius_fracture import sharp_front_v11_branching as production, sharp_front_v10_2_27 as paper
    def guard(actual_step, **kwargs):
        if actual_step != 419 or sha256(inputs / manifest['state_file']) != EXPECTED:
            raise RuntimeError('recovery is restricted to sealed step 419')
    production.require_uncontaminated_replay_checkpoint = guard
    argv = list(launch['arguments'])
    argv.remove('--v13-inherited-primary-race')
    argv[argv.index('--out') + 1] = str(folder)
    argv[argv.index('--maximum-fronts') + 1] = '2'
    argv += ['--v11-restart-checkpoint', str(cp)]
    paper.DEFAULT_REGISTRY = ROOT / 'runtime_inputs/pf_current_source_branching/pf_v2_four_class_pf_transfer_registry.csv'
    paper.SELECTION_RECORD = ROOT / 'runtime_inputs/pf_current_source_branching/pf_v2_four_class_pf_transfer_selection.json'
    paper.VALID_OPTIONS = {alias: c for c, alias in ROWS.values()}
    atomic_json(folder / 'launch.json', dict(launch, arguments=argv, fresh_initialization=False,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_checkpoint_sha256=EXPECTED, original_source_commit=launch['source_commit'],
        reseeded=False, regenerated_parent=False, mode=mode))
    if mode == 'frozen_failure':
        original_owner = production.selected_event_owner
        def capture(proposal, observations):
            # Diagnostic-only interception, immediately before the unchanged guard.
            frame = inspect.currentframe().f_back
            loop = frame.f_back
            payload = {k: loop.f_locals[k] for k in ('interval_state', 'diagnostics', 'context', 'selected')}
            payload.update({k: frame.f_locals[k] for k in (
                'pre_event_state', 'selected_state', 'trial_realized_arm_lengths', 'trial_clusters',
                'engine', 'controlling', 'info')})
            payload.update(proposal=proposal, observations=observations)
            (folder / 'failed_interval.pkl').write_bytes(pickle.dumps(payload, protocol=5))
            net0 = payload['pre_event_state'].crack_network
            net1 = payload['selected_state'].crack_network
            pre = {b.branch_id: b for b in net0.branches}
            post = {b.branch_id: b for b in net1.branches}
            by = {o.candidate_id: o.tip_id for o in observations}
            members = [dict(candidate_id=c, event_id=e, event_ordinal=n, completion_time_s=t,
                pre_event_front_id=by[c], full_event_identity=[by[c], c, n],
                process_owner_id=payload['pre_event_state'].junction_process_state['cluster'].cluster_id,
                branch_opportunity_id=[by[c], pre[by[c]].parent_branch_id, pre[by[c]].initiation_event])
                for c,e,n,t in zip(proposal.member_candidate_ids, proposal.member_event_ids,
                    proposal.member_event_ordinals, proposal.completion_times_s)]
            changed = [b for b in pre.keys() & post.keys() if pre[b] != post[b]]
            record = dict(proposal=asdict(proposal), members=members,
                context=asdict(payload['context']), events_proposed_for_consumption=list(proposal.member_event_ids),
                proposed_renewal_owner_tip_ids=sorted({m['pre_event_front_id'] for m in members}),
                exact_pre_topology=asdict(net0), exact_post_topology=asdict(net1),
                created_front_ids=sorted(post.keys()-pre.keys()), continued_front_ids=sorted(net1.active_tip_ids),
                retired_front_ids=sorted(set(net0.active_tip_ids)-set(net1.active_tip_ids)),
                coalesced_front_ids=[b for b in changed if post[b].local_state.get('coalesced', False)],
                structurally_affected_tip_ids=sorted(set(changed) | (post.keys()-pre.keys())),
                frozen_payload_sha256=sha256(folder / 'failed_interval.pkl'))
            try:
                original_owner(proposal, observations)
            except RuntimeError as exc:
                record['reproduced_failure'] = str(exc)
                atomic_json(folder / 'failed_proposal.json', record)
                raise
            raise RuntimeError('frozen replay did not reproduce expected owner guard')
        production.selected_event_owner = capture
    if mode == 'frozen_corrected':
        original_write = production.write_branch_checkpoint
        original_renew = production.apply_post_interval_event_renewal
        renewals = []
        def renew(*args, **kwargs):
            renewals.append(dict(kwargs))
            return original_renew(*args, **kwargs)
        def write(*args, **kwargs):
            result = original_write(*args, **kwargs)
            path = folder / 'checkpoint/latest.json'
            if path.exists() and json.loads(path.read_text())['event_counters']['accepted_steps'] == 420:
                atomic_json(folder / 'renewal_calls.json', renewals)
                raise FrozenAcceptedStop('first corrected interval accepted; frozen diagnostic ends')
            return result
        production.apply_post_interval_event_renewal = renew
        production.write_branch_checkpoint = write
    try:
        production.main(argv)
        if mode == 'frozen_failure':
            raise RuntimeError('frozen replay unexpectedly completed')
        terminal_record(folder, json.loads((folder / 'checkpoint/latest.json').read_text())['termination_reason'], 'TERMINATED')
    except FrozenAcceptedStop:
        atomic_json(folder / 'frozen_result.json', {'status': 'FIRST_CORRECTED_INTERVAL_ACCEPTED', 'accepted_step': 420})
    except Exception as exc:
        atomic_json(folder / 'exception.json', {'type': type(exc).__name__, 'reason': str(exc), 'traceback': traceback.format_exc()})
        if mode == 'frozen_failure' and str(exc) == 'one topology transaction spans multiple pre-event tips':
            pass
        else:
            if mode == 'continuation':
                known = any(token in str(exc) for token in (
                    'opening_scale_not_resolved_above_probe_uncertainty',
                    'directional adaptive stepping reached its minimum fraction',
                    'family endpoint', 'outside qualified', 'Newton failed',
                    'equilibrium did not converge', 'nonpositive_opening'))
                terminal_record(folder, str(exc), 'EXISTING_GATE_STOP' if known else 'SOFTWARE_OR_UNCLASSIFIED_STOP')
                if known:
                    return
            raise
    finally:
        atomic_json(folder / 'immutable_verification.json', {'unchanged_file_count': verify()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['frozen_failure', 'frozen_corrected', 'continuation', 'verify'])
    args = parser.parse_args()
    print(verify()) if args.mode == 'verify' else run(args.mode)
