#!/usr/bin/env python3
"""Audited causal comparison beside retained historical raw identity failure."""
import argparse
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.future_causal_state_fingerprint_v1 import (
    components as future_causal_components,
    fingerprint as future_causal_fingerprint,
)

from qualify_v5_disabled_neutrality_clean import (
    normalize, differences, write_json, clean_head, CASES, STAGE_II_V3_BASE,
)

DIAGNOSTICS = ('latest_free_dof_residual_l2_N_per_m', 'latest_constrained_reaction_l2_N_per_m',
               'latest_top_bottom_reaction_balance', 'latest_energy_reaction_identity')


def consumer_occurrences(repository,keys):
    """Keep every consumer hit in deterministic file/line order."""
    occurrences={}
    for key in keys:
        result=subprocess.run(('rg','-n',key,'arrhenius_fracture'),cwd=repository,text=True,capture_output=True)
        if result.returncode not in (0,1):raise RuntimeError('consumer audit search failed: '+result.stderr)
        occurrences[key]=sorted(result.stdout.splitlines())
    return occurrences


def causal(components):
    return future_causal_components(components)


def worker(repository, output, mode):
    sha = clean_head(repository); sys.path.insert(0, str(repository))
    import math
    import numpy as np
    from arrhenius_fracture.v12_production_driver import build_loaded_state, execute_event
    from arrhenius_fracture.sharp_wake_backend_v12 import V12_MODEL_ID
    from arrhenius_fracture.checkpoint_v11 import write_checkpoint, restore_checkpoint
    from arrhenius_fracture.topology_transaction_v11 import equilibrate_fixed_load_with_production_fem as equilibrate
    from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint
    from arrhenius_fracture.directional_competition_v11 import preview_directional_interval, commit_directional_interval
    from arrhenius_fracture.fem import assemble_mechanics
    from arrhenius_fracture.sharp_front import FrontEngine, FrontConfig, default_cleavage_barrier, default_emission_barrier
    if mode == 'disabled':
        from arrhenius_fracture.voiding_production_v5 import advance_disabled_v5_stage2 as event
    else: event = execute_event
    def components(state): return {field.name: normalize(getattr(state, field.name)) for field in fields(state)}
    def future(state, index):
        _, _, stress, *_ = assemble_mechanics(state.mesh, state.displacement, state.ep_gp, state.rho_gp,
            state.damage, state.elasticity_D, state.material, cohesive_network=state.cohesive_network)
        tip = np.asarray(state.crack_network.branches[0].tip)
        centroids = state.mesh.nodes[state.mesh.elems].mean(axis=1)
        intact = np.flatnonzero(np.asarray(state.mesh.element_damage_gp) == 0)
        eid = int(intact[np.argmin(np.linalg.norm(centroids[intact]-tip, axis=1))])
        tensor = np.array(((stress[0,eid], stress[2,eid]), (stress[2,eid], stress[1,eid])))
        engine = FrontEngine(FrontConfig(), default_cleavage_barrier(), default_emission_barrier(state.material.b),
                             state.material.G, state.material.nu, state.material.b)
        candidates = {c.candidate_id: c for c in state.competition.candidates}
        rates = []
        for h in state.competition.hazard_states:
            candidate = candidates[h.candidate_id]; n = np.asarray(candidate.normal_xy)
            opening = float(n@tensor@n)
            rate = 0. if opening <= 0 else engine.lambda_cleave(opening, 900.)[0]
            rates.append(float(rate))
        times = [(h.current_threshold_action-h.action)/rate if rate > 0 else math.inf
                 for h, rate in zip(state.competition.hazard_states, rates)]
        dt = min(times)
        if not math.isfinite(dt) or dt <= 0: raise RuntimeError('NO_POSITIVE_FUTURE_SOURCE_PASSAGE')
        advanced = tuple(commit_directional_interval(h, preview_directional_interval(h, lambda_per_s=rate,
            start_time_s=float(index), duration_s=dt)) for h, rate in zip(state.competition.hazard_states, rates))
        state = replace(state, competition=replace(state.competition, hazard_states=advanced))
        pending = state.competition.pending_events
        if not pending: raise RuntimeError('NO_ACTUAL_FUTURE_THRESHOLD_COMPLETION')
        candidate = candidates[pending[0].candidate_id]
        direction = np.asarray(candidate.direction_xy); normal = np.asarray((-direction[1],direction[0]))
        offset = state.mesh.nodes-tip; axial = offset@direction
        eligible = np.flatnonzero((np.abs(offset@normal) <= 1e-12)&(axial > 1e-8)
            &(state.mesh.nodes[:,0] < 1e-3-1e-12)&(np.abs(state.mesh.nodes[:,1]) < 5e-4-1e-12))
        if not len(eligible): raise RuntimeError('NO_MESH_ALIGNED_FUTURE_DIRECTION_ENDPOINT')
        endpoint = tuple(state.mesh.nodes[int(eligible[np.argmin(np.abs(axial[eligible]-25e-6))])])
        after, audit = event(state, endpoint, transaction_identity='causal-future:'+str(index))
        return after, {'candidate_id': candidate.candidate_id, 'event_ids': [e.event_id for e in pending],
            'source_tensor_Pa': tensor.tolist(), 'source_element_id': eid, 'rates': rates, 'duration_s': dt,
            'endpoint_m': endpoint, 'operations': audit['operations'],
            'energy_release_J_per_m': audit['energy_release_J_per_m'],
            'hazard_dissipation_J_per_m': audit['hazard_dissipation_J_per_m']}
    rows = []
    for name, cfg in CASES.items():
        state = build_loaded_state(V12_MODEL_ID, seed=cfg['seed']); history = []; snapshots = []; failure = None
        restart_exact = False
        try:
            if name == 'unload_reload':
                for opening in cfg['openings_m']:
                    u = state.displacement.copy(); u[2*np.asarray(state.boundary.top_nodes)+1] = opening/2
                    u[2*np.asarray(state.boundary.bot_nodes)+1] = -opening/2
                    state = equilibrate(replace(state, displacement=u))
                state, _ = event(state, (5.25e-4, 0.), transaction_identity='causal-initial:'+name)
            else: state, _ = event(state, tuple(cfg['endpoint_m']), transaction_identity='causal-initial:'+name)
            snapshots.append(components(state))
            write_checkpoint(state, output/'checkpoints'/(name+'_restart.json'))
            restored = restore_checkpoint(output/'checkpoints'/(name+'_restart.json'))
            direct, a = future(state, 1); resumed, b = future(restored, 1)
            restart_exact = fingerprint(direct) == fingerprint(resumed) and a == b
            state = direct; history.append(a); snapshots.append(components(state))
        except Exception as exc: failure = {'type': type(exc).__name__, 'message': str(exc)}
        write_checkpoint(state, output/'checkpoints'/(name+'_terminal.json'))
        rows.append({'case_id': name, 'configuration': cfg, 'raw_states': snapshots, 'history': normalize(history),
            'failure': failure, 'restart_exact': restart_exact, 'terminal_components': components(state)})
    write_json(output/'rows.json', {'sha': sha, 'rows': rows})
    if clean_head(repository) != sha: raise RuntimeError('worker implementation changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path); parser.add_argument('--base-worktree', type=Path)
    parser.add_argument('--worker', choices=('base', 'disabled')); parser.add_argument('--repository', type=Path)
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    if args.output.exists(): raise ValueError('refusing to overwrite evidence')
    if args.worker: worker(args.repository, args.output, args.worker); return
    if clean_head(args.base_worktree) != STAGE_II_V3_BASE: raise ValueError('wrong historical base')
    current = clean_head(root)
    for mode, repository in (('base', args.base_worktree), ('disabled', root)):
        subprocess.run((sys.executable, str(Path(__file__).resolve()), str(args.output/mode),
            '--worker', mode, '--repository', str(repository)), cwd=repository, check=True)
    a = json.loads((args.output/'base/rows.json').read_text()); b = json.loads((args.output/'disabled/rows.json').read_text())
    rows = []
    for first, second in zip(a['rows'], b['rows']):
        pairs = [(causal(x), causal(y)) for x, y in zip(first['raw_states'], second['raw_states'])]
        equal = len(first['raw_states']) == len(second['raw_states']) == 2 and all(x[0] == y[0] for x, y in pairs)
        history_equal = len(first['history']) == len(second['history']) == 1 and first['history'] == second['history']
        rows.append({'case_id': first['case_id'], 'causal_states_exact': equal, 'histories_exact': history_equal,
            'observed_causal_states_exact':bool(pairs) and len(first['raw_states'])==len(second['raw_states'])
                and all(x[0]==y[0] for x,y in pairs),
            'observed_histories_exact':bool(first['history']) and first['history']==second['history'],
            'observed_future_event_counts':[len(first['history']),len(second['history'])],
            'causal_fingerprints': [(future_causal_fingerprint(x), future_causal_fingerprint(y))
                                    for x, y in zip(first['raw_states'], second['raw_states'])],
            'separate_audit_provenance': [(x[1], y[1]) for x, y in pairs],
            'raw_differences': differences(first['terminal_components'], second['terminal_components']),
            'causal_differences': differences(causal(first['terminal_components'])[0], causal(second['terminal_components'])[0]),
            'base_failure': first['failure'], 'current_failure': second['failure'],
            'passed': equal and history_equal and first['restart_exact'] and second['restart_exact']
                and first['failure'] is None and second['failure'] is None})
    keys = (*DIAGNOSTICS, 'v12_boundary_terminal_certificates', 'boundary_terminal_certificates', 'source_commit', 'void_state')
    occurrences = consumer_occurrences(root,keys)
    report = {'schema': 'v5.disabled-future-causal-neutrality/2', 'implementation_sha': current,
        'historical_raw_full_state_identity': 'FAIL_RETAINED', 'rows': rows, 'passed': all(r['passed'] for r in rows),
        'consumer_occurrences': occurrences,
        'protocol_sha256': hashlib.sha256((root/'docs/V5_DISABLED_CAUSAL_NEUTRALITY_V2.md').read_bytes()).hexdigest()}
    write_json(args.output/'report.json', report)
    write_json(args.output/'sha256_manifest.json', {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.output.rglob('*')) if p.is_file()})
    print(json.dumps({'passed': report['passed'], 'cases': len(rows)}), flush=True)


if __name__ == '__main__': main()
