"""Eight archived states; fixed-opening primary increment FEM, never a trajectory."""
import copy
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import pickle
from types import SimpleNamespace

from arrhenius_fracture.inherited_primary_race_v13 import (
    ClockProjection, apply_primary_continuation_race, exponential_ensemble_probability, BOUNDARY)
from arrhenius_fracture.sharp_front_v11_branching import _request, _capture_shared_engine
from arrhenius_fracture.topology_transaction_v11 import TopologyArm, extend_network_arm, apply_causal_sharp_wake_trial_geometry
from scripts.v13_frozen_support import initialized_engine
from scripts.v13_value_fingerprint import physical_state_fingerprint as fp
from scripts.run_pf_current_source_multifront_field_atlas_v12 import atomic_json, sha256, campaign_environment
from scripts.audit_v13_inherited_clocks import balance

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'analysis_outputs/v13_clock_and_pair_mechanism'
OUT = ROOT/'analysis_outputs/v13_primary_continuation_race'


def evaluate(saved):
    case, event = saved['material_case'], saved['case']
    folder = SOURCE/'later_parents'/case/'events'/event/'parent'
    record = saved['clean_parent_record']
    context_file = folder/'event_context.pkl'
    if sha256(context_file) != record['event_context_sha256']:
        raise RuntimeError('source context hash mismatch')
    payload = pickle.loads(context_file.read_bytes())
    single = payload['accepted_single_checkpoint']
    baseline = single.state
    before = fp((baseline, single.shared_process_state))
    pair_path = SOURCE/'later_companions'/case/event/'exact_pair_result.pkl'
    cached_pair = pickle.loads(pair_path.read_bytes())
    trial, parent = cached_pair['trial'], cached_pair['parent']
    # Pickles have separate object identities; verify equality before rebinding
    # immutable parent invariants for the same-state transaction fixture.
    invariant_names = ('competition', 'rng_state', 'tip_process_state', 'event_counters')
    for name in invariant_names:
        if fp(getattr(trial.state, name)) != fp(getattr(baseline, name)):
            raise RuntimeError('cached exact pair parent invariant differs: '+name)
    trial = replace(trial, state=replace(trial.state, **{n:getattr(baseline,n) for n in invariant_names}))
    primary_id = record['winning_candidate_id']
    candidate = next(c for c in baseline.competition.candidates if c.candidate_id == primary_id)
    companion = saved['companions'][0]
    if companion['post_primary_process_sha256'] != fp(single.shared_process_state):
        raise RuntimeError('cached companion process identity mismatch')
    out = OUT/case/event
    out.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(**payload['args'])
    launch = json.loads((SOURCE/'later_parents'/case/'launch.json').read_text())
    family = Path(launch['family_validation']['family_validation']['family'])
    os.environ.update(campaign_environment(family))
    with initialized_engine(single.shared_process_state) as engine:
        process_before = fp(_capture_shared_engine(engine))
        tip_id = baseline.crack_network.active_tip_ids[0]
        start = baseline.crack_network.branch(tip_id).tip
        da = math.dist(payload['solved_pre_event_state'].crack_network.branch(tip_id).tip, start)
        arm = TopologyArm(primary_id, tip_id, start,
            tuple(start[k]+da*candidate.direction_xy[k] for k in (0,1)), da, 0.)
        ephemeral = replace(baseline.isolated_copy(), crack_network=extend_network_arm(baseline.crack_network, arm))
        ephemeral = apply_causal_sharp_wake_trial_geometry(ephemeral, (arm,))
        request = replace(_request(ephemeral, baseline.competition.candidates, args=args,
            cfg=payload['configuration'], runtime_step=payload['context'].step, cluster=None),
            cluster_frame={'mode':'candidate_marginal_kinetic_drive'})
        key = fp((request, before))
        cache = out/'primary_fixed_mechanics.json'
        if cache.exists():
            mechanics = json.loads(cache.read_text())
            if mechanics['request_sha256'] != key:
                raise RuntimeError('primary fixed mechanics cache identity mismatch')
        else:
            runtime = replace(single.provider_runtime, cache_root=str(out/'primary_trial_cache'))
            _, live = runtime.evaluate_trial(request)
            energy = float(live['base_equilibrium']['recoverable_potential_energy_J_per_m'])
            G = (baseline.stored_energy_J_per_m-energy)/da
            mechanics = {'request_sha256':key, 'G_primary_discrete_J_per_m2':G,
                'K_primary_discrete_Pa_sqrt_m':math.sqrt(baseline.material.Eprime*max(0.,G)),
                'baseline_energy_J_per_m':baseline.stored_energy_J_per_m,
                'trial_energy_J_per_m':energy, 'da_m':da,
                'scope':'fixed_opening_native_discrete_increment_not_remote_K_or_continuum_G',
                'topology_fingerprint':live['topology_fingerprint']}
            atomic_json(cache, mechanics)
        if mechanics['G_primary_discrete_J_per_m2'] <= 0:
            raise RuntimeError('nonpositive primary continuation marginal drive')
        # Production prefers valid local J. Match the exact accepted provider
        # result, including its displacement, rather than assuming invalidity.
        import numpy as np
        accepted_live = None
        for manifest_path in (SOURCE/'later_parents'/case/'live_kernel_cache').glob('*/manifest.json'):
            manifest = json.loads(manifest_path.read_text())
            if manifest['topology_fingerprint'] != single.provider_runtime.routing.topology_fingerprint:
                continue
            live_path = manifest_path.parent/'provider_state.pkl'
            if sha256(live_path) != manifest['state_sha256']:
                raise RuntimeError('accepted provider cache hash mismatch')
            live = pickle.loads(live_path.read_bytes())
            if np.array_equal(live['base_equilibrium']['displacement'], baseline.displacement):
                accepted_live = live
                break
        if accepted_live is None:
            raise RuntimeError('no exact accepted post-primary provider result; do not infer local J validity')
        local = next(d for tip in accepted_live['tips'] for d in tip['directional'] if d['candidate_id']==primary_id)
        G_used = local['J_local_signed_J_per_m2'] if local['local_contour_valid'] else mechanics['G_primary_discrete_J_per_m2']
        mechanics = dict(mechanics, production_primary_local_J_valid=local['local_contour_valid'],
            production_primary_signed_local_J_J_per_m2=local['J_local_signed_J_per_m2'],
            production_primary_kinetic_J_J_per_m2=max(0.,G_used), accepted_provider_sha256=manifest['state_sha256'])
        preview = copy.deepcopy(engine)
        effective, raw, barrier = preview.lambda_cleave(preview.sigma_tip(
            math.sqrt(baseline.material.Eprime*max(0.,G_used))/math.sqrt(candidate.gamma_rel)), float(args.temperatures[0]))
        tau = float(engine.f.tau_c)
        hazards = {h.candidate_id:h for h in baseline.competition.hazard_states}
        clocks = {}
        for cid, rate in ((primary_id,effective),(companion['candidate_id'],companion['effective_parent_law_rate_diagnostic_per_s'])):
            h = hazards[cid]
            if h.pending_events:
                raise RuntimeError('pending event requires explicit queue arbitration; not inferred')
            clocks[cid] = ClockProjection(cid,h.action,h.current_threshold_action,h.completed_event_count+1,
                f'v11-directional-threshold|{h.threshold_seed}|{cid}|{h.completed_event_count+1}',rate,
                record['accepted_single_checkpoint_sha256'])
        primary, secondary = clocks[primary_id], clocks[companion['candidate_id']]
        projected = apply_primary_continuation_race(parent=parent, baseline_single_state=baseline,
            primary=primary, companions=(secondary,), tau_c=tau, exact_pair_trial=lambda *_:trial, enabled=True)
        # Raw-crossing alternative uses both clocks at that same time origin.
        old = next(h for h in payload['pre_cleavage_checkpoint'].state.competition.hazard_states if h.candidate_id==primary_id)
        raw_primary = replace(primary,action=old.current_threshold_action,observation_identity='common_raw_crossing')
        raw_secondary = replace(secondary,action=saved['later_nonwinner_residual']['H_at_primary_crossing'],observation_identity='common_raw_crossing')
        raw_result = apply_primary_continuation_race(parent=parent, baseline_single_state=baseline,
            primary=raw_primary,companions=(raw_secondary,),tau_c=tau,exact_pair_trial=lambda *_:trial,enabled=True)
        if fp(_capture_shared_engine(engine)) != process_before or fp((baseline,single.shared_process_state)) != before:
            raise RuntimeError('read-only race mutated accepted process or parent')
        result = {'case':case,'event':event,'forward_extension_um':record['forward_extension_um'],
            'primary':asdict(primary),'companion':asdict(secondary),
            'T_i_next_s':primary.completion_s,'T_j_s':secondary.completion_s,'tau_c_s':tau,
            'raw_crossing_T_i_next_s':raw_primary.completion_s,'raw_crossing_T_j_s':raw_secondary.completion_s,
            'raw_crossing_outcome':raw_result.disposition,'outcome':projected.disposition,
            'raw_crossing_action_roundoff_bound':saved['later_nonwinner_residual']['action_roundoff_bound_from_absolute_time'],
            'primary_raw_rate_per_s':raw,'primary_barrier_J':barrier,
            'companion_raw_rate_per_s':companion['raw_arrival_per_s'],
            'raw_rate_balance':balance(raw,companion['raw_arrival_per_s']),
            'effective_rate_balance':balance(effective,secondary.effective_rate_per_s),
            'exact_pair_admissible':trial.accepted,'exact_pair_margin_J_per_m':trial.energy_margin_J_per_m,
            'parent_checkpoint_sha256':record['accepted_single_checkpoint_sha256'],
            'source_context_sha256':sha256(context_file),'cached_pair_sha256':sha256(pair_path),
            'process_sha256':process_before,'baseline_rng_sha256':fp(baseline.rng_state),
            'engine_rng_sha256':fp(engine._hazard_rng),'parent_invariants_unchanged':True,
            'fallback_object_exact':projected.state is baseline if not projected.disposition=='PAIR_ACCEPTED' else None,
            'global_time_increment_s':0.,'new_branch_rng_draws':0,'mechanics':mechanics,'boundary':BOUNDARY}
        atomic_json(out/'race.json',result)
        print(case,event,result['outcome'],primary.completion_s,secondary.completion_s,flush=True)
        return result


def main():
    summary_path = SOURCE/'later_companions/summary.json'
    source_hash = sha256(summary_path)
    rows = [evaluate(item) for item in json.loads(summary_path.read_text())['cases']]
    first_path = SOURCE/'clock_audit.json'
    first = []
    for item in json.loads(first_path.read_text())['cases']:
        nonwinner = next(c for c in item['candidates'] if not c['winner'])
        first.append({'case':item['case'], 'T_j_accepted_endpoint_s':nonwinner['predicted_remaining_post_s_from_endpoint'],
            'tau_c_s':item['tau_c_s'],'cannot_beat_expiry':nonwinner['predicted_remaining_post_s_from_endpoint']>item['tau_c_s'],
            'source_context_sha256':item['event_context_sha256']})
    gate = (len(first)==8 and all(r['cannot_beat_expiry'] for r in first) and
        len(rows)==8 and all(r['outcome']!='PAIR_ACCEPTED' for r in rows if r['event']=='event00002')
        and all(any(r['outcome']=='PAIR_ACCEPTED' for r in rows if r['case']==c) for c in ('Peak_1000K','weakT_1000K'))
        and all(any(r['outcome']!='PAIR_ACCEPTED' for r in rows if r['case']==c and r['event']!='event00002') for c in ('Peak_1000K','weakT_1000K')))
    atomic_json(OUT/'frozen_summary.json',{'cases':rows,'first_event_controls':first,'first_audit_sha256':sha256(first_path),
        'later_pattern_gate_passed':gate,
        'source_summary_sha256':source_hash,'equal_saturated_ideal_ensemble_probability':exponential_ensemble_probability(1e6,1e6,1e-6),
        'production_registered':False,'short_ensemble_launched':False,'boundary':BOUNDARY})


if __name__=='__main__':
    main()
