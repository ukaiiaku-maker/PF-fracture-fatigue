"""Frozen real-stage rollback attempts; unreached injections are failures."""
from dataclasses import replace
from .closure_lifecycle_evidence import CFG,load_state,advance_transition,build_healing_predecessor
from .voiding_v5 import advance_site,arrhenius_rates,create_subgrid_cavity,update_cavity_growth,promote_cavity
from .voiding_production_v5 import (
    local_site_tensor,_geometry,_grow_hole_boundary,remesh_cavity,ligament_transaction,
    downstream_front_transaction,refine_downstream_source,
)
from .checkpoint_v11 import write_checkpoint,restore_checkpoint
from .topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint

ROLLBACK_STAGES=(
    'first_hit_threshold_renewal','second_hit_embryo_transition','stabilization','healing',
    'state_owned_growth','initial_inventory_debit','inventory_return_under_shrinkage',
    'promotion_criterion','explicit_cavity_creation','promotion_remesh','field_transfer',
    'equilibrium','resolved_growth_remesh','ligament_hazard_completion','ligament_graph_edit',
    'cavity_connection','downstream_source_refinement','downstream_threshold_completion',
    'child_creation','child_support_rebuild','child_tip_continuation','checkpoint_write',
)


def rollback_attempts(captured,terminal,checkpoint_path):
    for stage in ROLLBACK_STAGES:
        before=captured.get('ligament_rupture',terminal);ops=[];error=None;hook=stage;identity=fingerprint(before)
        def inject(name,trial):
            ops.append(name)
            if name==hook: raise RuntimeError('injected:'+name)
        try:
            if stage in ROLLBACK_STAGES[:4]:
                label={'first_hit_threshold_renewal':'available_site','second_hit_embryo_transition':'multi_hit_1',
                    'stabilization':'multi_hit_2','healing':'multi_hit_2'}[stage]
                before=captured[label]
                if stage=='healing': before,preparation=build_healing_predecessor();ops.extend(preparation)
                rates=arrhenius_rates(CFG,temperature_K=900.,stress_tensor_Pa=local_site_tensor(before))
                site=before.void_state.sites[0]
                if stage in ROLLBACK_STAGES[:2]: duration=site.birth.crossing_time(rates['birth_s']*site.candidate_weight)
                else: duration=getattr(site,stage).crossing_time(rates[stage+'_s'])
                identity=fingerprint(before)
                advance_site(before.void_state,site.site_id,duration,rates=rates,failure_injector=inject)
            elif stage=='initial_inventory_debit':
                before,_=advance_transition(captured['multi_hit_2'],'stabilization',1)
                identity=fingerprint(before)
                create_subgrid_cavity(before.void_state,'site-1',2.5e-5,failure_injector=inject)
            elif stage in ('state_owned_growth','inventory_return_under_shrinkage'):
                before=captured['subgrid_void']
                if stage=='inventory_return_under_shrinkage': before=load_state(before,-4e-7)
                identity=fingerprint(before)
                rates=arrhenius_rates(CFG,temperature_K=900.,stress_tensor_Pa=local_site_tensor(before))
                # A prospectively imposed negative chemical-potential segment
                # exercises the production inventory return, without rate substitution.
                update_cavity_growth(before.void_state,before.void_state.cavities[0].cavity_id,
                    rates=rates,dt_s=1. if stage=='inventory_return_under_shrinkage' else 1e-10,
                    radial_growth_scale_m=CFG.radial_growth_scale_m,
                    chemical_potential_drive_J=-1e-20 if stage=='inventory_return_under_shrinkage' else 1e-20,
                    failure_injector=inject)
            elif stage=='promotion_criterion':
                before=captured['subgrid_growth'];identity=fingerprint(before)
                promote_cavity(before.void_state,before.void_state.cavities[0].cavity_id,5e-5,failure_injector=inject)
            elif stage in ('explicit_cavity_creation','promotion_remesh','field_transfer','equilibrium','resolved_growth_remesh'):
                before=captured['geometric_promotion' if stage=='resolved_growth_remesh' else 'subgrid_growth']
                cavity=before.void_state.cavities[0];identity=fingerprint(before)
                hole,_=_geometry(radius_m=cavity.radius_m,center_m=cavity.center_m)
                hole=_grow_hole_boundary(hole,cavity.radius_m,crack_path_m=before.crack_network.branches[0].path)
                hook={'promotion_remesh':'remesh','field_transfer':'field_projection','resolved_growth_remesh':'remesh'}.get(stage,stage)
                voids=before.void_state
                if stage!='resolved_growth_remesh': voids=promote_cavity(voids,cavity.cavity_id,5e-5)
                remesh_cavity(before,hole,voids,'resolved-growth' if stage=='resolved_growth_remesh' else 'promotion',ops,failure_stage=hook)
            elif stage in ('ligament_hazard_completion','ligament_graph_edit','cavity_connection'):
                before=captured['resolved_growth'];identity=fingerprint(before)
                hook={'ligament_graph_edit':'graph_edit','cavity_connection':'connected_surface_certification'}.get(stage,stage)
                ligament_transaction(before,failure_stage=hook,operation_log=ops)
            elif stage=='downstream_source_refinement':
                identity=fingerprint(before)
                refine_downstream_source(before,max_refinement_levels=1,failure_stage=hook,operation_log=ops)
            elif stage=='checkpoint_write':
                before=captured['available_site'];identity=fingerprint(before)
                write_checkpoint(before,checkpoint_path)
                write_checkpoint(replace(before,checkpoint_generation=before.checkpoint_generation+1),
                    checkpoint_path,failure_injector=inject)
            else:
                if stage=='child_tip_continuation': before=captured.get('new_graph_front',terminal)
                identity=fingerprint(before)
                hook={'downstream_threshold_completion':'downstream_threshold_completion','child_creation':'downstream_child_activation',
                    'child_support_rebuild':'support_rebuild','child_tip_continuation':'graph_edit'}[stage]
                downstream_front_transaction(before,continuation=stage=='child_tip_continuation',failure_stage=hook,operation_log=ops)
        except Exception as exc: error={'type':type(exc).__name__,'message':str(exc)}
        after=before
        restored=fingerprint(after)==identity
        if stage=='checkpoint_write': restored &= fingerprint(restore_checkpoint(checkpoint_path))==identity
        yield stage,before,after,{'failure_stage':hook,'intended_stage_reached':hook in ops,
            'duration_s':1. if stage=='inventory_return_under_shrinkage' else None,
            'chemical_potential_drive_J':-1e-20 if stage=='inventory_return_under_shrinkage' else None},ops,error,restored
