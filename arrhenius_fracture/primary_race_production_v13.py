"""Default-off conditional mark after the untouched canonical single-arm step.

Only the first bifurcation is in scope. Post-birth growth uses the existing
two-front driver. No extra process step, event renewal, time or RNG draw.
"""
import copy
from dataclasses import asdict, dataclass, replace
import math
from types import SimpleNamespace

import numpy as np

from .conditional_branch_mark_v13 import ParentCleavageEvent, BranchMark, BOUNDARY
from .inherited_primary_race_v13 import ClockProjection, apply_primary_continuation_race
from .marked_topology_trial_v13 import trial_conditional_pair
from .topology_transaction_v11 import TopologyArm, extend_network_arm, apply_causal_sharp_wake_trial_geometry
from .fem import assemble_mechanics
from .anisotropic_emission_v10174 import probe_tensor_ahead, build_front_drive, require_admissible_tensor_drive
from .hazard_energy_event_gate_v10230 import hazard_resistance_J_per_m2


@dataclass(frozen=True)
class ProductionRaceResult:
    state: object
    cluster: object
    runtime: object
    record: dict


def evaluate_production_mark(*, checkpoint, selected, solved_pre_event, engine, args, cfg, context, accepted_live):
    from .sharp_front_v11_branching import _request, _realized_trial_network, _capture_shared_engine
    from scripts.v13_value_fingerprint import physical_state_fingerprint as fp
    base = checkpoint.state
    initial_hash = fp((base, _capture_shared_engine(engine)))
    pid = selected.proposal.member_candidate_ids[0]
    cid = next(c.candidate_id for c in base.competition.candidates if c.candidate_id != pid)
    candidates = {c.candidate_id:c for c in base.competition.candidates}
    if len(candidates)!=2 or len(base.crack_network.active_tip_ids)!=1:
        raise ValueError('bounded first-bifurcation option requires the qualified two-plane parent')
    hazards = {h.candidate_id:h for h in base.competition.hazard_states}
    identity = fp(base.competition)
    process_identity = fp(checkpoint.shared_process_state)
    parent = ParentCleavageEvent(('v13-short',base.crack_network.active_tip_ids[0]),
        base.crack_network.active_tip_ids[0], pid, selected.proposal.member_event_ids[0],
        selected.proposal.member_event_ordinals[0], checkpoint.physical_time_s,
        checkpoint.accepted_load, identity, process_identity, identity, fp(engine._hazard_rng))
    record = {'boundary':BOUNDARY,'step':context.step,'parent_event_id':parent.primary_event_id,
        'primary_candidate_id':pid,'companion_candidate_id':cid,
        'accepted_time_s':checkpoint.physical_time_s,'accepted_opening_m':checkpoint.accepted_load,
        'parent_forward_extension_um':checkpoint.projected_extension_m*1e6,
        'parent_competition_sha256':identity,'parent_process_sha256':process_identity,
        'parent_engine_rng_sha256':fp(engine._hazard_rng),'tau_c_s':engine.f.tau_c,
        'clock_time_origin':'canonical_accepted_endpoint','new_branch_rng_draws':0,'global_time_increment_s':0.,
        'process_steps_by_mark':0,'event_renewals_by_mark':0}
    def finish(state=base,cluster=None,runtime=checkpoint.provider_runtime,**details):
        if fp((base,_capture_shared_engine(engine)))!=initial_hash:
            raise RuntimeError('conditional mark mutated canonical parent or continuous-emission engine')
        return ProductionRaceResult(state,cluster,runtime,dict(record,**details))
    tip_id = base.crack_network.active_tip_ids[0]
    junction = solved_pre_event.crack_network.branch(tip_id).tip
    endpoint = base.crack_network.branch(tip_id).tip
    da = math.dist(junction,endpoint)
    sigma = assemble_mechanics(base.mesh,base.displacement,base.ep_gp,base.rho_gp,base.damage,
        base.elasticity_D,base.material,cohesive_network=base.cohesive_network)[2]
    probe = probe_tensor_ahead(base.mesh,sigma,base.damage,np.asarray(junction),
        np.asarray(candidates[cid].direction_xy),engine.anisotropic_cfg)
    drive = build_front_drive(base.mesh,sigma,base.damage,np.asarray(junction),engine.anisotropic_cfg)
    try:
        require_admissible_tensor_drive(drive)
    except RuntimeError as exc:
        return finish(outcome='SINGLE_INADMISSIBLE_COMPANION_OBSERVATION',reason=str(exc))
    normal = np.asarray(candidates[cid].normal_xy)
    if not probe['reliable']:
        return finish(outcome='SINGLE_INADMISSIBLE_COMPANION_OBSERVATION',reason='unreliable_candidate_tensor')
    sigma_nn = float(normal@np.asarray(probe['tensor'])@normal)
    if sigma_nn<=0:
        return finish(outcome='SINGLE_INADMISSIBLE_COMPANION_OBSERVATION',reason='nonpositive_candidate_normal_traction')
    ledger = dict(base.energy_ledgers)
    for key in ('topology_release_J_per_m','hazard_dissipation_J_per_m'):
        ledger[key]=solved_pre_event.energy_ledgers.get(key,0.)
    pair_start = replace(solved_pre_event,tip_process_state=base.tip_process_state,energy_ledgers=ledger)
    proposal = SimpleNamespace(action_type='two_arm',member_candidate_ids=tuple(sorted((pid,cid))))
    network,cluster,pairs = _realized_trial_network(pair_start,proposal,base.competition.candidates,da,None)
    arms = tuple(replace(a,end_xy_m=endpoint) if a.candidate_id==pid else a for _,a in pairs)
    def geometry(state, actual_arms):
        realized=network
        for arm in actual_arms:
            realized=extend_network_arm(realized,arm)
        return apply_causal_sharp_wake_trial_geometry(replace(state,crack_network=realized,
            junction_process_state={'cluster':cluster,'crack_representation':'sharp_wake_causal_v11'}),actual_arms)
    pair_state=geometry(pair_start.isolated_copy(),arms)
    request=_request(pair_state,base.competition.candidates,args=args,cfg=cfg,runtime_step=context.step,cluster=cluster)
    runtime,live=checkpoint.provider_runtime.evaluate_trial(request)
    pair_energy=float(live['base_equilibrium']['recoverable_potential_energy_J_per_m'])
    G_comp=(base.stored_energy_J_per_m-pair_energy)/da
    if G_comp<=0:
        return finish(outcome='SINGLE_INADMISSIBLE_COMPANION_OBSERVATION',reason='nonpositive_marginal_energy')
    def rate(candidate,G):
        preview=copy.deepcopy(engine)
        return preview.lambda_cleave(preview.sigma_tip(math.sqrt(base.material.Eprime*G/candidate.gamma_rel)),float(args.temperatures[0]))
    eff_j,raw_j,barrier_j=rate(candidates[cid],G_comp)
    def project(candidate_id,effective):
        h=hazards[candidate_id]
        if h.pending_events:
            pending=min(h.pending_events,key=lambda p:(p.completion_time_s,p.event_ordinal))
            return ClockProjection(candidate_id,h.action,pending.action_after,pending.event_ordinal,
                f'v11-directional-threshold|{h.threshold_seed}|{candidate_id}|{pending.event_ordinal}',
                effective,identity,pending.event_id)
        return ClockProjection(candidate_id,h.action,h.current_threshold_action,h.completed_event_count+1,
            f'v11-directional-threshold|{h.threshold_seed}|{candidate_id}|{h.completed_event_count+1}',effective,identity)
    companion=project(cid,eff_j)
    record.update(companion=asdict(companion),T_j_s=companion.completion_s,
        companion_raw_rate_per_s=raw_j,companion_G_discrete_J_per_m2=G_comp,sigma_nn_Pa=sigma_nn)
    if companion.completion_s>=engine.f.tau_c:
        return finish(outcome='SINGLE_CORRELATION_WINDOW_EXPIRED',primary_rate_evaluated=False)
    local=next(d for tip in accepted_live['tips'] for d in tip['directional'] if d['candidate_id']==pid)
    G_primary=local['J_local_signed_J_per_m2']
    if not local['local_contour_valid']:
        direction=candidates[pid].direction_xy
        arm=TopologyArm(pid,tip_id,endpoint,tuple(endpoint[k]+da*direction[k] for k in (0,1)),da,0.)
        state=apply_causal_sharp_wake_trial_geometry(replace(base.isolated_copy(),
            crack_network=extend_network_arm(base.crack_network,arm)),(arm,))
        req=replace(_request(state,base.competition.candidates,args=args,cfg=cfg,runtime_step=context.step,cluster=None),
            cluster_frame={'mode':'candidate_marginal_kinetic_drive'})
        runtime,primary_live=runtime.evaluate_trial(req)
        G_primary=(base.stored_energy_J_per_m-float(primary_live['base_equilibrium']['recoverable_potential_energy_J_per_m']))/da
    eff_i,raw_i,barrier_i=rate(candidates[pid],max(0.,G_primary))
    primary=project(pid,eff_i)
    record.update(primary=asdict(primary),T_i_next_s=primary.completion_s,primary_raw_rate_per_s=raw_i,
        primary_G_kinetic_J_per_m2=G_primary,primary_local_J_valid=local['local_contour_valid'])
    cost=hazard_resistance_J_per_m2(barrier_J=barrier_j,cooperative_hits=engine.f.m_hits,
        burgers_vector_m=engine.b,gamma_relative=candidates[cid].gamma_rel)*da
    arms=tuple(replace(a,hazard_dissipation_J_per_m=selected.result.hazard_dissipation_J_per_m if a.candidate_id==pid else cost) for a in arms)
    def equilibrium(state):
        return replace(state,displacement=np.asarray(live['base_equilibrium']['displacement']),stored_energy_J_per_m=pair_energy)
    trial=trial_conditional_pair(pre_event_state=pair_start,baseline_single_state=base,parent=parent,
        mark=BranchMark(cid,None,()),arms=arms,apply_trial_geometry=geometry,
        equilibrate_fixed_load=equilibrium,network_geometry_already_realized=True)
    marked=apply_primary_continuation_race(parent=parent,baseline_single_state=base,primary=primary,
        companions=(companion,),tau_c=engine.f.tau_c,exact_pair_trial=lambda *_:trial,enabled=True)
    if marked.state is base:
        return finish(outcome=marked.disposition,pair_margin_J_per_m=trial.energy_margin_J_per_m)
    cluster=replace(cluster,shared_process_state={**cluster.shared_process_state,
        'birth_step':context.step,'birth_extension_m':checkpoint.physical_extension_m,
        'conditional_mark_model':'inherited_primary_race_v13'})
    runtime=runtime.accept_trial(request,live)
    return finish(state=marked.state,cluster=cluster,runtime=runtime,outcome=marked.disposition,
        pair_margin_J_per_m=trial.energy_margin_J_per_m,
        branch_junction_xy_m=junction,primary_endpoint_xy_m=endpoint,
        daughter_initial_length_m=da,branch_clock_rng_identity_preserved=True)
