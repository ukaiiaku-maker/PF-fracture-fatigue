"""Bounded physical-time integration of the existing one-void production APIs.

No rate, threshold or mesh-quality tolerance is altered. Each first passage
is localized from the owned clock; geometric transitions use actual remesh
and equilibrium. An unavailable downstream source remains a real no-event
interval. A failed trial retains the last accepted stage, with elapsed time
reported separately from the requested interval.
"""
from dataclasses import replace
import math
import numpy as np
from .canonical_kinetic_time_v1 import AcceptedTime, exact

from .voiding_v5 import (
    VoidPhase,VoidingConfig,advance_site,arrhenius_rates,create_subgrid_cavity,
    update_cavity_growth,promote_cavity,
)
from .voiding_production_v5 import (
    local_site_tensor,cavity_boundary_tensor,crack_tip_tensor,directional_clock_rates,
    _complete_next_clock,_geometry,_grow_hole_boundary,remesh_cavity,
    ligament_transaction,downstream_front_transaction,refine_downstream_source,
)
from .topology_transaction_v11 import equilibrate_fixed_load_with_production_fem as equilibrate

NATURAL_WINDOW_S=16e-6
NATURAL_SEEDS=tuple(range(12000,12032))


def advance_production_void_interval(state,dt_s,*,temperature_K=900.,config=None,
                                     refinement_levels=3,refinement_attempt_cache=None):
    config=VoidingConfig(enabled=True,promotion_radius_m=5e-5) if config is None else config
    duration=float(dt_s)
    if not math.isfinite(duration) or duration<0.: raise ValueError('finite nonnegative physical duration required')
    if state.void_state is None: raise ValueError('enabled production void state required')
    duration_exact=exact(dt_s)
    remaining=duration_exact;elapsed=0.;operations=[];error=None;accepted=state
    initial_time=float(state.junction_process_state.get('production_time_s',0.))
    initial_clock=state.junction_process_state.get('canonical_accepted_time_v1', AcceptedTime.from_seconds(initial_time))
    elapsed_exact=exact(0)
    cache={} if refinement_attempt_cache is None else refinement_attempt_cache
    def commit(trial,step,operation):
        nonlocal accepted,remaining,elapsed,elapsed_exact
        previous=accepted
        elapsed_exact+=exact(step);elapsed=float(elapsed_exact)
        remaining=max(duration_exact-elapsed_exact,0)
        physical_clock=initial_clock.advance(elapsed_exact)
        accepted=replace(trial,junction_process_state={**trial.junction_process_state,
            'production_time_s':physical_clock.seconds(), 'canonical_accepted_time_v1':physical_clock})
        from .closure_lifecycle_evidence import conservation,stagewise_topology
        from .topology_transaction_v11 import complete_accepted_state_fingerprint
        operations.append({**operation,'duration_s':float(step),'physical_time_s':physical_clock.seconds(),
            'temperature_K':temperature_K,'accepted_pre_interval_fingerprint':complete_accepted_state_fingerprint(previous),
            'accepted_post_interval_fingerprint':complete_accepted_state_fingerprint(accepted),
            'stagewise_conservation':conservation(accepted,previous),'stagewise_topology':stagewise_topology(accepted)})
    for _ in range(64):
        if remaining<=0.: break
        try:
            voids=accepted.void_state;site=voids.sites[0]
            if site.phase==VoidPhase.HEALED_SITE:
                commit(accepted,remaining,{'api':'healed_site_residence','events':[]});continue
            if site.phase in (VoidPhase.AVAILABLE_SITE,VoidPhase.EMBRYO):
                rates=arrhenius_rates(config,temperature_K=temperature_K,stress_tensor_Pa=local_site_tensor(accepted))
                def crossing_time(clock,rate):
                    value=clock.crossing_time_exact(rate)
                    return math.inf if value is None else value
                crossing=(crossing_time(site.birth,rates['birth_s']*site.candidate_weight)
                    if site.phase==VoidPhase.AVAILABLE_SITE else min(crossing_time(site.stabilization,rates['stabilization_s']),
                        crossing_time(site.healing,rates['healing_s'])))
                step=min(remaining,crossing)
                if step<=0.: raise RuntimeError('NONPOSITIVE_OWNED_SITE_PASSAGE_INTERVAL')
                updated,events=advance_site(voids,site.site_id,step,rates=rates)
                commit(replace(accepted,void_state=updated),step,{'api':'advance_site','events':list(events),'rates':rates})
                continue
            if not voids.cavities:
                updated=create_subgrid_cavity(voids,site.site_id,2.5e-5)
                commit(equilibrate(replace(accepted,void_state=updated)),0.,
                       {'api':'create_subgrid_cavity','radius_m':2.5e-5,'events':['INITIAL_CAVITY_SEED_INVENTORY_DEBIT']})
                continue
            cavity=voids.cavities[0]
            if cavity.phase in (VoidPhase.STABLE_SUBGRID_VOID,VoidPhase.RESOLVED_VOID):
                target=config.promotion_radius_m if cavity.phase==VoidPhase.STABLE_SUBGRID_VOID else 5.5e-5
                if cavity.phase==VoidPhase.STABLE_SUBGRID_VOID and cavity.radius_m>=target:
                    updated=promote_cavity(voids,cavity.cavity_id,target)
                    hole,_=_geometry(radius_m=cavity.radius_m,center_m=cavity.center_m)
                    hole=_grow_hole_boundary(hole,cavity.radius_m,crack_path_m=accepted.crack_network.branches[0].path)
                    trace=[];trial=remesh_cavity(accepted,hole,updated,'natural-promotion',trace)
                    commit(trial,0.,{'api':'promotion_remesh','operations':trace,'events':['GEOMETRIC_PROMOTION']});continue
                if cavity.radius_m<target:
                    tensor=local_site_tensor(accepted) if cavity.phase==VoidPhase.STABLE_SUBGRID_VOID else cavity_boundary_tensor(accepted)[0]
                    rates=arrhenius_rates(config,temperature_K=temperature_K,stress_tensor_Pa=tensor)
                    from .voiding_v5 import growth_time_to_radius_exact
                    crossing=growth_time_to_radius_exact(voids,cavity.cavity_id,target,rates=rates,
                        radial_growth_scale_m=config.radial_growth_scale_m)
                    crossing=math.inf if crossing is None else crossing
                    step=min(remaining,crossing)
                    updated=update_cavity_growth(voids,cavity.cavity_id,rates=rates,dt_s=step,
                                                radial_growth_scale_m=config.radial_growth_scale_m)
                    # The resolved-growth target is a physical remesh event.
                    # Intermediate API partitions advance the exact owned
                    # radius integral but must not manufacture extra meshes,
                    # equilibrium solves, or source tensors.  Commit the one
                    # represented-geometry update when the target is reached.
                    if (cavity.phase==VoidPhase.RESOLVED_VOID
                            and updated.cavities[0].radius_m>=target):
                        hole,_=_geometry(radius_m=cavity.radius_m,center_m=cavity.center_m)
                        hole=_grow_hole_boundary(hole,updated.cavities[0].radius_m,crack_path_m=accepted.crack_network.branches[0].path)
                        trace=[];trial=remesh_cavity(accepted,hole,updated,'natural-resolved-growth',trace)
                    else:
                        # Subgrid bookkeeping and unresolved increments toward
                        # the next represented resolved geometry are not inputs
                        # to assemble_mechanics.  With unchanged mesh/u/ep/rho/
                        # damage/material/boundary, another solve would add only
                        # caller-subdivision-dependent state and roundoff.
                        trace=[];trial=replace(accepted,void_state=updated)
                    commit(trial,step,{'api':'state_owned_growth','rates':rates,'operations':trace});continue
                root=accepted.crack_network.branches[0]
                tensor,ids=crack_tip_tensor(accepted,branch_id=root.branch_id)
                rates=directional_clock_rates(accepted,tensor);crossing=min(r['crossing_time_s'] for r in rates)
                if crossing>remaining:
                    trial,audit=_complete_next_clock(accepted,tensor,maximum_advance_duration_s=remaining,
                        source_kind='sharp_front',source_front_id=root.branch_id,source_position_m=root.tip,
                        source_probe_identity={'kind':'crack_tip_tensor','element_ids':list(ids)})
                    commit(trial,remaining,{'api':'ligament_hazard_interval','audit':audit,'events':[]})
                else:
                    trace=[];trial,result=ligament_transaction(accepted,operation_log=trace)
                    commit(trial,crossing,{'api':'ligament_transaction','accepted':result.accepted,'operations':trace})
                continue
            if cavity.phase==VoidPhase.CONNECTED_VOID:
                # Cache only failed refinement diagnostics for an identical
                # mechanical/candidate source. It cannot authorize activity.
                from .sharp_wake_backend_v12 import array_fingerprint
                key=(array_fingerprint(accepted.mesh.nodes),array_fingerprint(accepted.mesh.elems),
                    array_fingerprint(accepted.displacement),array_fingerprint(accepted.ep_gp),
                    array_fingerprint(accepted.rho_gp),array_fingerprint(accepted.damage),
                    tuple(c.candidate_id for c in accepted.competition.candidates))
                if key not in cache:
                    trial,audit=refine_downstream_source(accepted,max_refinement_levels=refinement_levels)
                    operations.append({'api':'refine_downstream_source','audit':audit,'duration_s':0.})
                    if trial is accepted: cache[key]=audit
                    accepted=trial
                else:
                    operations.append({'api':'reuse_failed_refinement_diagnostic','audit':cache[key],'duration_s':0.})
                position=np.asarray(cavity.connection_exit_m)
                node=int(np.argmin(np.linalg.norm(accepted.mesh.nodes-position,axis=1)))
                tensor,ids=cavity_boundary_tensor(accepted,boundary_node=node)
                raw=directional_clock_rates(accepted,tensor)
                crossing=min(r['crossing_time_s'] for r in raw)
                if not math.isfinite(crossing) or crossing>remaining:
                    trial,audit=_complete_next_clock(accepted,tensor,source_kind='cavity_surface',
                        source_cavity_id=cavity.cavity_id,source_boundary_site_id='connection_exit',
                        source_position_m=tuple(position),source_probe_identity={'kind':'direct_cavity_boundary_tensor',
                        'boundary_node_id':node,'element_ids':list(ids)},maximum_advance_duration_s=remaining)
                    commit(trial,remaining,{'api':'downstream_hazard_interval','audit':audit,'events':[]})
                else:
                    trial,result,trace,audit=downstream_front_transaction(accepted)
                    if result is None:
                        commit(trial,remaining,{'api':'unavailable_downstream_interval','audit':audit,'events':[]})
                    else:
                        commit(trial,crossing,{'api':'downstream_front_transaction','operations':trace,'audit':audit,'accepted':result.accepted})
                continue
            if cavity.phase==VoidPhase.DOWNSTREAM_FRONT_ACTIVE:
                child=accepted.crack_network.branch(accepted.crack_network.active_tip_ids[0])
                tensor,_=crack_tip_tensor(accepted,branch_id=child.branch_id)
                crossing=min(r['crossing_time_s'] for r in directional_clock_rates(accepted,tensor))
                if crossing>remaining:
                    trial,audit=_complete_next_clock(accepted,tensor,source_kind='sharp_front',
                        source_front_id=child.branch_id,source_position_m=child.tip,maximum_advance_duration_s=remaining)
                    commit(trial,remaining,{'api':'child_tip_hazard_interval','audit':audit})
                else:
                    trial,result,trace,audit=downstream_front_transaction(accepted,continuation=True)
                    commit(trial,crossing,{'api':'child_tip_continuation','operations':trace,'audit':audit,'accepted':result is not None and result.accepted})
                continue
            raise RuntimeError('UNSUPPORTED_NATURAL_PHASE:'+cavity.phase.value)
        except Exception as exc:
            error={'type':type(exc).__name__,'message':str(exc)};break
    else: error={'type':'RuntimeError','message':'BOUNDED_PHYSICAL_INTERVAL_STAGE_LIMIT'}
    return accepted,operations,{'requested_duration_s':duration,'elapsed_duration_s':elapsed,'failure':error}
