"""Actual-state closure attempts, including fail-closed prerequisites.

This contract separates evidence validity from a successful scientific gate.
It never invents an unavailable child checkpoint or a successful operation
trace. Every reported endpoint must resolve to a complete accepted FEM state.
"""
from dataclasses import replace
import math
import numpy as np

from .finalization_v3_schema import FROZEN_CASE_REGISTRY, canonical_hash, SCIENTIFIC_ACCEPTANCE_TOLERANCES as LIMITS
from .topology_transaction_v11 import (
    LiveFEMTopologyState, complete_accepted_state_fingerprint as fingerprint,
    equilibrate_fixed_load_with_production_fem as equilibrate,
)
from .voiding_production_v5 import (
    _geometry, _grow_hole_boundary, _complete_next_clock,
    local_site_tensor, cavity_boundary_tensor, crack_tip_tensor, directional_clock_rates,
    ligament_transaction, downstream_front_transaction, remesh_cavity,
)
from .voiding_v5 import (
    VoidingConfig, VoidPhase, advance_site, arrhenius_rates, create_subgrid_cavity,
    update_cavity_growth, promote_cavity,
)

SCHEMA = "v12.voiding-v5-closure-actual-lifecycle/7"
PARTITIONS = (1,2,4,8,16)
CFG = VoidingConfig(enabled=True, promotion_radius_m=5e-5)
PRECURSORS = {"birth_hit_1": "available_site", "birth_hit_2": "multi_hit_1",
              "stabilization": "multi_hit_2", "healing": "healing_peer",
              "subgrid_growth": "subgrid_void", "promotion": "subgrid_growth",
              "ligament": "resolved_growth", "downstream_child": "ligament_rupture",
              "child_continuation": "new_graph_front"}
_TOPOLOGY_MEASUREMENTS = {}
CONTROLLED_HEALING_SEED=12010


def build_healing_predecessor():
    from .voiding_production_v5 import build_production_void_state
    state,_=build_production_void_state(stochastic=True,seed=CONTROLLED_HEALING_SEED)
    ops=[]
    for stage in ('birth_hit_1','birth_hit_2'):
        state,_=advance_transition(state,stage,1,operations=ops)
    state=load_state(state,-4e-7)
    ops.append({'api':'accepted_compressive_reload','opening_m':-4e-7,'seed':CONTROLLED_HEALING_SEED})
    return state,ops


def natural_terminal_measurements(state):
    """Decode the actual accepted seed state, never infer events from rates."""
    from .closure_mechanics_evidence import canonical_data
    voids=state.void_state;site=voids.sites[0]
    clocks={name:{'integrated_hazard':getattr(site,name).accumulated,
        'threshold':getattr(site,name).threshold,
        'threshold_margin':getattr(site,name).threshold-getattr(site,name).accumulated}
        for name in ('birth','stabilization','healing')}
    return {'site_id':site.site_id,'site_phase':site.phase.value,'birth_hit_count':site.hits,
        'clocks':clocks,'void_rng_identity':canonical_hash(dict(voids.rng_state)),
        'global_rng_identity':canonical_hash(canonical_data(state.rng_state)),
        'event_history':list(voids.event_history),'cavity_count':len(voids.cavities),
        'cavities':[{'cavity_id':c.cavity_id,'radius_m':c.radius_m,'phase':c.phase.value,
            'geometry_generation':c.geometry_generation,'entry_m':c.connection_entry_m,'exit_m':c.connection_exit_m}
            for c in voids.cavities],
        'active_tip_ids':list(state.crack_network.active_tip_ids),
        'accepted_topology_actions':state.event_counters.get('topology_actions',0),
        'directional_event_provenance':state.junction_process_state.get('directional_event_provenance',{}),
        'length_ledgers':dict(voids.length_ledgers)}


def load_state(state, opening_m):
    u = state.displacement.copy()
    u[2*np.asarray(state.boundary.top_nodes)+1] = opening_m/2
    u[2*np.asarray(state.boundary.bot_nodes)+1] = -opening_m/2
    return equilibrate(replace(state, displacement=u))


def advance_transition(state, name, partitions, *, operations=None, config=CFG):
    """Execute a real stage from its accepted predecessor at a fixed load."""
    from .canonical_kinetic_time_v1 import exact,packed
    operations = [] if operations is None else operations
    initial = state
    if name in ("birth_hit_1", "birth_hit_2", "stabilization", "healing"):
        rates = arrhenius_rates(config, temperature_K=900., stress_tensor_Pa=local_site_tensor(state))
        site = state.void_state.sites[0]
        clock = site.birth if name.startswith("birth") else getattr(site, name)
        channel = "birth_s" if name.startswith("birth") else name+"_s"
        effective = rates[channel]*(site.candidate_weight if name.startswith("birth") else 1.)
        total = clock.crossing_time_exact(effective)
        if total is None: raise RuntimeError("NO_KINETICALLY_ACTIVE_CANDIDATE")
        for part in range(partitions):
            # All partitions end at the same exact analytic first passage.
            # Re-rounding a remaining crossing can otherwise integrate a tiny
            # extra interval into the newly renewed clock in only some peers.
            dt = total/partitions
            voids, events = advance_site(state.void_state, site.site_id, dt, rates=rates)
            state = replace(state, void_state=voids)
            operations.append({"api": "advance_site", "duration_s": float(dt), "duration_exact_s":packed(dt),
                "rates": rates, "events": list(events)})
        return equilibrate(state), operations
    if name == "subgrid_growth":
        rates = arrhenius_rates(config, temperature_K=900., stress_tensor_Pa=local_site_tensor(state))
        cavity = state.void_state.cavities[0]
        from .voiding_v5 import growth_time_to_radius_exact
        total = growth_time_to_radius_exact(state.void_state,cavity.cavity_id,5e-5,rates=rates,
            radial_growth_scale_m=config.radial_growth_scale_m)
        if total is None:raise RuntimeError('NO_KINETICALLY_ACTIVE_GROWTH_CHANNEL')
        for part in range(partitions):
            dt = total/partitions
            state = replace(state, void_state=update_cavity_growth(state.void_state, cavity.cavity_id,
                rates=rates, dt_s=dt, radial_growth_scale_m=config.radial_growth_scale_m))
            operations.append({"api": "update_cavity_growth", "duration_s": float(dt), "duration_exact_s":packed(dt), "rates": rates})
        return equilibrate(state), operations
    if name == "promotion":
        cavity = state.void_state.cavities[0]
        voids = promote_cavity(state.void_state, cavity.cavity_id, config.promotion_radius_m)
        n,layers = state.junction_process_state.get('production_mesh_resolution',(32,12))
        hole, _ = _geometry(radius_m=cavity.radius_m, center_m=cavity.center_m,
                            boundary_segments=n,radial_layers=layers)
        hole = _grow_hole_boundary(hole, cavity.radius_m, crack_path_m=state.crack_network.branches[0].path)
        trace = []
        state = remesh_cavity(state, hole, voids, "promotion", trace)
        operations.append({"api": "remesh_cavity", "operations": trace})
        return state, operations
    if name == "ligament":
        tensor, ids = crack_tip_tensor(state, branch_id=state.crack_network.branches[0].branch_id)
        rates = directional_clock_rates(state,tensor)
        total = min(r["crossing_time_s"] for r in rates)
        if not math.isfinite(total): raise RuntimeError("NO_KINETICALLY_ACTIVE_CANDIDATE")
        # Leave the final actual crossing to the real topology transaction.
        for _ in range(partitions-1):
            state, audit = _complete_next_clock(state,tensor,maximum_advance_duration_s=total/partitions,
                source_kind="sharp_front",source_front_id=state.crack_network.branches[0].branch_id,
                source_position_m=state.crack_network.branches[0].tip,
                source_probe_identity={"kind":"crack_tip_tensor","element_ids":list(ids)})
            if any(r["winner"] for r in audit):
                raise RuntimeError("unexpected early threshold crossing in pretransaction partition")
            operations.append({"api": "_complete_next_clock", "duration_s":total/partitions,"audit":audit})
        trace = []
        state, trial = ligament_transaction(state,operation_log=trace)
        operations.append({"api":"ligament_transaction", "accepted":trial.accepted,"operations":trace})
        return state, operations
    if name in ("downstream_child", "child_continuation"):
        # Transition partition qualification localizes a real first passage;
        # the independently frozen natural ensemble still has its 16 us limit.
        # Long physical waiting times are reported, never labeled rapid events.
        from .voiding_production_v5 import _qualified_cavity_source,refine_downstream_source
        continuation = name == 'child_continuation'
        if continuation:
            if not state.crack_network.active_tip_ids: raise RuntimeError('NO_ACTIVE_DOWNSTREAM_CHILD')
            child = state.crack_network.branch(state.crack_network.active_tip_ids[0])
            tensor,ids = crack_tip_tensor(state,branch_id=child.branch_id)
            source = dict(source_kind='sharp_front',source_front_id=child.branch_id,source_position_m=child.tip,
                source_probe_identity={'kind':'child_crack_tip_tensor','element_ids':list(ids)})
        else:
            cavity = state.void_state.cavities[0]
            node = int(np.argmin(np.linalg.norm(state.mesh.nodes-np.asarray(cavity.connection_exit_m),axis=1)))
            tensor,ids = cavity_boundary_tensor(state,boundary_node=node)
            if not any(row['effective_rate_s'] > 0 for row in directional_clock_rates(state,tensor)):
                from .voiding_lifecycle_driver_v5 import NATURAL_WINDOW_S
                state,audit = _complete_next_clock(state,tensor,source_kind='cavity_surface',
                    maximum_advance_duration_s=NATURAL_WINDOW_S)
                operations.append({'api':'zero_drive_connected_interval','duration_s':NATURAL_WINDOW_S,
                    'audit':audit,'accepted':False})
                return state,operations
            if not _qualified_cavity_source(state,tensor):
                state,audit = refine_downstream_source(state,max_refinement_levels=(2 if
                    state.junction_process_state.get('common_terminal_restart_protocol_v1') else 1),
                    refinement_region='complete_cavity_ring',quality_improvement='constrained_v1')
                operations.append({'api':'refine_downstream_source','audit':audit,'duration_s':0.})
                node = int(np.argmin(np.linalg.norm(state.mesh.nodes-np.asarray(cavity.connection_exit_m),axis=1)))
                tensor,ids = cavity_boundary_tensor(state,boundary_node=node)
            if not _qualified_cavity_source(state,tensor): raise RuntimeError('UNQUALIFIED_CAVITY_SOURCE_TENSOR')
            source = dict(source_kind='cavity_surface',source_cavity_id=cavity.cavity_id,source_boundary_site_id='connection_exit',
                source_position_m=cavity.connection_exit_m,source_probe_identity={'kind':'direct_cavity_boundary_tensor',
                    'boundary_node_id':node,'element_ids':list(ids)})
        total = min(row['crossing_time_s'] for row in directional_clock_rates(state,tensor))
        if not math.isfinite(total): raise RuntimeError('NO_KINETICALLY_ACTIVE_CANDIDATE')
        for _ in range(partitions-1):
            state,audit = _complete_next_clock(state,tensor,maximum_advance_duration_s=total/partitions,**source)
            if any(row['winner'] for row in audit): raise RuntimeError('EARLY_PARTITION_THRESHOLD_COMPLETION')
            operations.append({'api':'source_owned_hazard_interval','duration_s':audit[0]['common_advance_duration_s'],'audit':audit})
        state,result,trace,audit = downstream_front_transaction(state,continuation=continuation)
        operations.append({'api':'child_tip_continuation' if continuation else 'downstream_front_transaction',
            'duration_s':audit['cleavage'][0]['common_advance_duration_s'],'operations':trace,'audit':audit,
            'accepted':result is not None and result.accepted})
        return state, operations
    raise ValueError(name)


def transition_occurred(name, before, after):
    old,new = before.void_state.sites[0],after.void_state.sites[0]
    if name == "birth_hit_1": return new.hits == old.hits+1 and new.phase == VoidPhase.AVAILABLE_SITE
    if name == "birth_hit_2": return new.hits == old.hits+1 and new.phase == VoidPhase.EMBRYO
    if name == "stabilization": return old.phase == VoidPhase.EMBRYO and new.phase == VoidPhase.STABLE_SUBGRID_VOID
    if name == "healing": return old.phase == VoidPhase.EMBRYO and new.phase == VoidPhase.HEALED_SITE
    if not before.void_state.cavities or not after.void_state.cavities: return False
    a,b = before.void_state.cavities[0],after.void_state.cavities[0]
    if name == "subgrid_growth": return b.radius_m > a.radius_m and after.void_state.consumed_defect_inventory_area_m2 > before.void_state.consumed_defect_inventory_area_m2
    if name == "promotion": return a.phase == VoidPhase.STABLE_SUBGRID_VOID and b.phase == VoidPhase.RESOLVED_VOID and before.mesh.ne != after.mesh.ne
    if name == "ligament":
        def consumed(state):
            return {key for key,value in state.junction_process_state.get("directional_event_provenance",{}).items()
                    if value.get("status")=="CONSUMED_AT_OWNED_SOURCE"}
        # The new cavity source correctly owns a fresh competition. The root
        # event's consumed identity is retained in the source-provenance ledger,
        # not in that fresh competition's empty consumed-event tuple.
        return (a.phase == VoidPhase.RESOLVED_VOID and b.phase == VoidPhase.CONNECTED_VOID
                and not after.crack_network.active_tip_ids and bool(consumed(after)-consumed(before)))
    if name == "downstream_child": return len(after.crack_network.branches)==len(before.crack_network.branches)+1 and b.phase == VoidPhase.DOWNSTREAM_FRONT_ACTIVE and len(after.crack_network.active_tip_ids)==1
    if name == "child_continuation": return len(before.crack_network.active_tip_ids)==1 and a.phase == VoidPhase.DOWNSTREAM_FRONT_ACTIVE and after.crack_network.total_physical_crack_length_m > before.crack_network.total_physical_crack_length_m
    raise ValueError(name)


def stagewise_topology(state):
    from .mechanically_separating_sharp_wake_v12 import independent_intact_path_certificate
    from .voiding_production_v5 import cavity_free_surface_certificate,crack_void_connection_certificate
    from .closure_mechanics_evidence import canonical_data
    from .sharp_wake_backend_v12 import array_fingerprint
    selected=state.v12_support_state.selected_support_elements
    source_key=canonical_hash(canonical_data({'nodes':array_fingerprint(state.mesh.nodes),
        'elements':array_fingerprint(state.mesh.elems),'selected':selected,
        'network':state.crack_network.to_dict(),'cavities':() if state.void_state is None else state.void_state.cavities,
        'boundary_context':state.junction_process_state.get('boundary_terminal_context',{})}))
    if source_key in _TOPOLOGY_MEASUREMENTS: return _TOPOLOGY_MEASUREMENTS[source_key]
    certificate=independent_intact_path_certificate(state.mesh,state.crack_network,selected,
        boundary_terminal_context=state.junction_process_state.get('boundary_terminal_context',{}))
    passed=not certificate['intact_cross_graph_path_exists'] and not certificate['insufficient_seed_segment_ids']
    result={'independent_cut':canonical_data(certificate),'passed':bool(passed)}
    if state.void_state and state.void_state.cavities:
        cavity=state.void_state.cavities[0]
        if cavity.phase in (VoidPhase.RESOLVED_VOID,VoidPhase.CONNECTED_VOID,VoidPhase.DOWNSTREAM_FRONT_ACTIVE):
            cycle=cavity_free_surface_certificate(state)
            result['actual_cavity_cycle']=cycle;result['passed'] &= cycle['passed']
        if cavity.connection_entry_m is not None:
            root=state.crack_network.branches[0]
            combined=crack_void_connection_certificate(state,branch_id=root.branch_id,
                cavity_id=cavity.cavity_id,intended_intersection=cavity.connection_entry_m)
            result['actual_component_incidence']=canonical_data(combined)
            result['passed'] &= combined['passed']
    result['topology_source_fingerprint']=source_key
    _TOPOLOGY_MEASUREMENTS[source_key]=result
    return result


def conservation(state, initial):
    voids = state.void_state
    if voids is None:
        return {"void_inventory": "NOT_APPLICABLE_DISABLED", "passed": initial.void_state is None
            and tuple(state.crack_network.active_tip_ids)==tuple(state.v12_support_state.active_tip_identities)}
    total = voids.available_defect_inventory_area_m2+voids.consumed_defect_inventory_area_m2
    original = initial.void_state.available_defect_inventory_area_m2+initial.void_state.consumed_defect_inventory_area_m2
    inventory_error = abs(total-original)
    radius_error = max((abs(c.area_m2-math.pi*c.radius_m**2) for c in voids.cavities),default=0.)
    active = tuple(state.crack_network.active_tip_ids)
    support = tuple(state.v12_support_state.active_tip_identities)
    ledger = voids.length_ledgers
    added = state.crack_network.total_physical_crack_length_m-initial.crack_network.total_physical_crack_length_m
    expected = (ledger["fractured_ligament_length_m"]+ledger["ordinary_crack_fractured_length_m"]
        -initial.void_state.length_ledgers["fractured_ligament_length_m"]-initial.void_state.length_ledgers["ordinary_crack_fractured_length_m"])
    length_error = abs(added-expected)
    physical_error=abs(ledger['physical_active_front_travel_m']-
        ledger['fractured_ligament_length_m']-ledger['ordinary_crack_fractured_length_m']-
        ledger['traversed_void_free_span_m'])
    projected_error=abs(ledger['projected_front_advance_m']-
        ledger['projected_fractured_length_m']-ledger['projected_free_span_m'])
    chord_error=projected_chord_error=0.
    chord=None
    if voids.cavities and voids.cavities[0].connection_entry_m is not None:
        cavity=voids.cavities[0]
        delta=np.asarray(cavity.connection_exit_m)-np.asarray(cavity.connection_entry_m)
        chord={'entry_m':list(cavity.connection_entry_m),'exit_m':list(cavity.connection_exit_m),
               'physical_length_m':float(np.linalg.norm(delta)),
               'reporting_direction':[1.,0.],'projected_length_m':float(delta[0])}
        chord_error=abs(ledger['connected_void_free_span_m']-chord['physical_length_m'])
        projected_chord_error=abs(ledger['projected_connected_void_free_span_m']-chord['projected_length_m'])
    hazards=state.competition.hazard_states
    candidates=[c.candidate_id for c in state.competition.candidates]
    ownership=(candidates==[h.candidate_id for h in hazards] and len(set(candidates))==len(candidates)
               and all(h.current_threshold_action>0. and h.action>=0. for h in hazards))
    connected_dormant=not voids.cavities or voids.cavities[0].phase!=VoidPhase.CONNECTED_VOID or not active
    errors=(length_error,physical_error,projected_error,chord_error,projected_chord_error)
    return {"inventory_error_m2":inventory_error,"cavity_area_identity_error_m2":radius_error,
            "length_error_m":length_error,"active_tip_ids":list(active),"support_tip_ids":list(support),
            'physical_travel_identity_error_m':physical_error,'projected_travel_identity_error_m':projected_error,
            'actual_cavity_chord':chord,'connected_chord_error_m':chord_error,
            'projected_connected_chord_error_m':projected_chord_error,
            'candidate_threshold_ownership':ownership,'connected_state_has_no_active_front':connected_dormant,
            "single_void":len(voids.cavities)<=1,
            "passed":inventory_error<=LIMITS['inventory_identity_abs_m2'] and radius_error<=LIMITS['inventory_identity_abs_m2']
                and max(errors)<=LIMITS['length_identity_abs_m'] and active==support and len(active)<=1
                and len(voids.cavities)<=1 and ownership and connected_dormant}


def prepare_common_restart_reload(state,operations,state_trace=None,accepted=None,protocol_version=1):
    """One accepted load protocol shared by all eleven checkpoint positions."""
    from .voiding_lifecycle_driver_v5 import NATURAL_WINDOW_S
    if protocol_version not in (1,2):raise ValueError('unregistered common restart protocol')
    key='common_terminal_restart_protocol_v'+str(protocol_version)
    other='common_terminal_restart_protocol_v'+str(3-protocol_version)
    if other in state.junction_process_state:raise ValueError('cannot mix accepted common restart protocols')
    reload_opening=8e-7 if protocol_version==1 else 4e-7
    if state.void_state.cavities[0].phase!=VoidPhase.CONNECTED_VOID:
        raise ValueError('common restart reload requires the accepted connected state')
    position=state.junction_process_state.get(key)
    if position is None:
        state=load_state(state,-4e-7)
        state=replace(state,junction_process_state={**state.junction_process_state,key:'COMPRESSIVE_LOAD_ACCEPTED'})
        if accepted is not None:accepted[0]=state
        operations.append({'api':'common_restart_compressive_load','opening_m':-4e-7})
        if state_trace is not None:state_trace.append(('zero_drive_connected',state))
        position='COMPRESSIVE_LOAD_ACCEPTED'
    if position=='COMPRESSIVE_LOAD_ACCEPTED':
        cavity=state.void_state.cavities[0]
        node=int(np.argmin(np.linalg.norm(state.mesh.nodes-np.asarray(cavity.connection_exit_m),axis=1)))
        tensor,_=cavity_boundary_tensor(state,boundary_node=node)
        if any(row['effective_rate_s']>0 for row in directional_clock_rates(state,tensor)):
            raise RuntimeError('COMMON_RESTART_DORMANT_LOAD_HAS_POSITIVE_SOURCE_DRIVE')
        state,audit=_complete_next_clock(state,tensor,source_kind='cavity_surface',maximum_advance_duration_s=NATURAL_WINDOW_S)
        state=replace(state,junction_process_state={**state.junction_process_state,key:'ZERO_INTERVAL_ACCEPTED'})
        if accepted is not None:accepted[0]=state
        operations.append({'api':'common_restart_dormant_interval','duration_s':NATURAL_WINDOW_S,'audit':audit})
        position='ZERO_INTERVAL_ACCEPTED'
    if position=='ZERO_INTERVAL_ACCEPTED':
        state=load_state(state,reload_opening)
        state=replace(state,junction_process_state={**state.junction_process_state,key:'TENSILE_RELOAD_ACCEPTED'})
        if accepted is not None:accepted[0]=state
        operations.append({'api':'common_restart_tensile_reload','opening_m':reload_opening})
        position='TENSILE_RELOAD_ACCEPTED'
    if position!='TENSILE_RELOAD_ACCEPTED':raise ValueError('unrecognized common restart protocol position')
    return state


def resume_to_guard(state, operations, *, common_restart_protocol=False):
    """Resume the actual accepted lifecycle to its attainable terminal state."""
    accepted=[state]
    try:
        return _resume_to_guard(state,operations,common_restart_protocol,accepted)
    except Exception as error:
        error.accepted_lifecycle_state=accepted[0]
        raise


def _resume_to_guard(state,operations,common_restart_protocol,accepted):
    def keep(value):
        accepted[0]=value
        return value
    while state.void_state.sites[0].phase == VoidPhase.AVAILABLE_SITE:
        name = "birth_hit_1" if state.void_state.sites[0].hits == 0 else "birth_hit_2"
        previous = fingerprint(state)
        state,_ = advance_transition(state,name,1,operations=operations)
        keep(state)
        if fingerprint(state)==previous: raise RuntimeError("birth resume made no progress")
    if state.void_state.sites[0].phase == VoidPhase.EMBRYO:
        state,_ = advance_transition(state,"stabilization",1,operations=operations)
        keep(state)
    if state.void_state.sites[0].phase == VoidPhase.HEALED_SITE: return state
    if not state.void_state.cavities:
        state = equilibrate(replace(state,void_state=create_subgrid_cavity(state.void_state,"site-1",2.5e-5)))
        keep(state)
        operations.append({"api":"create_subgrid_cavity","radius_m":2.5e-5})
    cavity = state.void_state.cavities[0]
    if cavity.radius_m < 5e-5:
        state,_ = advance_transition(state,"subgrid_growth",1,operations=operations)
        keep(state)
    if state.void_state.cavities[0].phase == VoidPhase.STABLE_SUBGRID_VOID:
        state,_ = advance_transition(state,"promotion",1,operations=operations)
        keep(state)
    cavity = state.void_state.cavities[0]
    if cavity.phase == VoidPhase.RESOLVED_VOID and cavity.radius_m < 5.5e-5:
        n,layers = state.junction_process_state.get('production_mesh_resolution',(32,12))
        hole,_ = _geometry(radius_m=cavity.radius_m,center_m=cavity.center_m,boundary_segments=n,radial_layers=layers)
        hole = _grow_hole_boundary(hole,5.5e-5,crack_path_m=state.crack_network.branches[0].path)
        rates = arrhenius_rates(CFG,temperature_K=900.,stress_tensor_Pa=cavity_boundary_tensor(state)[0])
        from .voiding_v5 import growth_time_to_radius_exact
        from .canonical_kinetic_time_v1 import packed
        dt = growth_time_to_radius_exact(state.void_state,cavity.cavity_id,5.5e-5,rates=rates,
            radial_growth_scale_m=CFG.radial_growth_scale_m)
        if dt is None:raise RuntimeError('NO_KINETICALLY_ACTIVE_GROWTH_CHANNEL')
        voids = update_cavity_growth(state.void_state,cavity.cavity_id,rates=rates,dt_s=dt,
                                     radial_growth_scale_m=CFG.radial_growth_scale_m)
        trace=[]; state=remesh_cavity(state,hole,voids,"resolved-growth",trace)
        keep(state)
        operations.append({"api":"resolved_growth_remesh","duration_s":float(dt),'duration_exact_s':packed(dt),"rates":rates,"operations":trace})
    if state.void_state.cavities[0].phase == VoidPhase.RESOLVED_VOID:
        state,_ = advance_transition(state,"ligament",1,operations=operations)
        keep(state)
    if state.void_state.cavities[0].phase == VoidPhase.CONNECTED_VOID:
        if common_restart_protocol:state=prepare_common_restart_reload(state,operations,accepted=accepted,
            protocol_version=2 if common_restart_protocol=='v2' else 1)
        state,_ = advance_transition(state,"downstream_child",1,operations=operations)
        keep(state)
    if state.void_state.cavities[0].phase == VoidPhase.DOWNSTREAM_FRONT_ACTIVE:
        state,_ = advance_transition(state,"child_continuation",1,operations=operations)
        keep(state)
    return state


def validate_controlled_inputs(row, sources):
    """Bind physical inputs and failed preparation endpoints to owned states."""
    from .closure_mechanics_evidence import canonical_data
    cfg=row['input_configuration']; before=sources[row['initial_checkpoint']]
    if canonical_data(before.crack_network.branches[0].path)!=cfg['fixed_crack_path_m']:
        raise ValueError('controlled fixed crack does not match accepted source')
    for reference in [row['initial_checkpoint'],row['terminal_checkpoint'],
                      *row['actual_preparation_checkpoints'].values()]:
        if canonical_data(sources[reference].void_state.sites[0].center_m)!=cfg['center_m']:
            raise ValueError('controlled boundary site does not match physical input')
    stages=row['actual_preparation_stages']
    if len(set(stages))!=len(stages) or set(row['actual_preparation_checkpoints'])!=set(stages):
        raise ValueError('controlled preparation registry mismatch')
    if row['failure'] and not row['actual_operations'] and stages:
        if row['terminal_checkpoint']!=row['actual_preparation_checkpoints'][stages[-1]]:
            raise ValueError('failed preparation lost its last accepted state')


def continued_front_terminal(state):
    """Read the actual child path, lineage and source-owned accepted event."""
    if state.void_state is None or not state.void_state.cavities:return False
    cavity=state.void_state.cavities[0]
    if cavity.phase!=VoidPhase.DOWNSTREAM_FRONT_ACTIVE or 'CONTINUED_EVENT' not in cavity.lineage:return False
    if state.crack_network.active_tip_ids!=('void-front-1',):return False
    if state.v12_support_state.active_tip_identities!=('void-front-1',):return False
    if len(state.crack_network.branch('void-front-1').path)<3:return False
    return any(row.get('event')=='CONTINUED_ACCEPTED_EVENT' and row.get('source_kind')=='sharp_front'
        and row.get('source_front_id')=='void-front-1' for row in state.void_state.event_history)


def validate_lifecycle(payload, sources, *, executed_code_sha):
    if payload["schema"] != SCHEMA or payload["executed_code_sha"] != executed_code_sha:
        raise ValueError("lifecycle source/schema identity")
    rows = payload["rows"]
    if len({r["execution_id"] for r in rows}) != len(rows): raise ValueError("aliased lifecycle execution")
    expected = {(case,p) for case in FROZEN_CASE_REGISTRY["transitions"] for p in PARTITIONS}
    transitions = [r for r in rows if r["dataset"]=="transitions"]
    if len(transitions)!=45 or {(r["case_identity"],r["partition_count"]) for r in transitions} != expected:
        raise ValueError("actual transition registry mismatch")
    for dataset in ("restarts","controlled","neutrality"):
        observed = [r["case_identity"] for r in rows if r["dataset"]==dataset]
        if sorted(observed) != sorted(FROZEN_CASE_REGISTRY[dataset]):
            raise ValueError("actual "+dataset+" registry mismatch")
    natural=[r for r in rows if r["dataset"]=="natural"]
    if len(natural)!=160 or {(int(r["case_identity"]),r["partition_count"]) for r in natural} != {
        (seed,p) for seed in range(12000,12032) for p in PARTITIONS}:
        raise ValueError("natural seed/partition registry mismatch")
    from .closure_rollback_matrix_v5 import ROLLBACK_STAGES
    observed=[r['case_identity'] for r in rows if r['dataset']=='rollback' and r['case_identity'].startswith('lifecycle:')]
    if sorted(observed)!=sorted('lifecycle:'+name for name in ROLLBACK_STAGES):
        raise ValueError('complete lifecycle rollback registry mismatch')
    validate_lifecycle_rows(rows,sources,executed_code_sha=executed_code_sha)
    if payload["decision"] != lifecycle_decision(rows,sources):
        raise ValueError("lifecycle decision differs from actual checkpoint comparisons")
    return {"valid":True,"actual_state_rows":len(rows),"transition_attempts":len(transitions),
            "successful_transition_attempts":sum(r["transition_occurred"] for r in transitions)}


def validate_lifecycle_rows(rows,sources,*,executed_code_sha):
    """Shared source-bound reconstruction, without claiming a complete registry.

    The full validator above still requires every final registry. Development
    adapters must independently enforce their complete selected section.
    """
    if len({r['execution_id'] for r in rows})!=len(rows):raise ValueError('aliased lifecycle execution')
    for row in rows:
        before, after = sources[row["initial_checkpoint"]],sources[row["terminal_checkpoint"]]
        if not isinstance(before,LiveFEMTopologyState) or not isinstance(after,LiveFEMTopologyState):
            raise ValueError("not complete accepted FEM captures")
        if row["initial_fingerprint"] != fingerprint(before) or row["terminal_fingerprint"] != fingerprint(after):
            raise ValueError("actual accepted checkpoint mismatch")
        if row["executed_code_sha"] != executed_code_sha or row["input_hash"] != canonical_hash(row["input_configuration"]):
            raise ValueError("actual execution input/source mismatch")
        if row["conservation"] != conservation(after,before): raise ValueError("stagewise conservation recomputation")
        if row['dataset']=='controlled':
            validate_controlled_inputs(row,sources)
        if row["dataset"] == "transitions":
            occurred = transition_occurred(row["case_identity"],before,after)
            if row["transition_occurred"] != occurred: raise ValueError("transition falsely claimed")
            if occurred:
                name=row["case_identity"]; ops=row["actual_operations"]
                if name in ("birth_hit_1","birth_hit_2","stabilization","healing"):
                    if len(ops)!=row["partition_count"] or any(op["api"]!="advance_site" for op in ops):
                        raise ValueError("missing actual site intervals")
                    expected_events=[r["event"] for r in after.void_state.event_history[len(before.void_state.event_history):]]
                    if [event for op in ops for event in op["events"]]!=expected_events:
                        raise ValueError("site operation trace is not the actual event history")
                required={"promotion":("remesh","field_projection","support_rebuild","equilibrium"),
                    "ligament":("accepted_snapshot","graph_edit","remesh","field_projection","support_rebuild",
                                "connected_surface_certification","equilibrium","energy_gate","topology_verification")}.get(name)
                if required:
                    trace=ops[-1]["operations"]
                    if any(trace.count(token)!=1 for token in required) or [trace.index(t) for t in required] != sorted(trace.index(t) for t in required):
                        raise ValueError("missing/reordered/duplicated actual topology operations")
        if row["dataset"] == "restarts":
            restarted=sources[row["restored_terminal_checkpoint"]]
            if row["restart_exact"] != (fingerprint(after)==fingerprint(restarted)):
                raise ValueError("restart peer mismatch")
            if row['subsequent_history_exact'] != (row['actual_operations']==row['restarted_operations']):
                raise ValueError('subsequent restart history mismatch')
            reached=row['input_configuration']['requested_stage_available'] and continued_front_terminal(after)
            if row['continued_front_terminal_reached']!=reached:
                raise ValueError('common continued terminal is not owned by the actual source state')
        if row["dataset"] == "natural":
            from .closure_mechanics_evidence import canonical_data
            if row['terminal_measurements']!=canonical_data(natural_terminal_measurements(after)):
                raise ValueError('natural seed terminal measurements do not match accepted state')
            restarted=sources[row["restarted_terminal_checkpoint"]]
            if row["midpoint_restart_exact"] != (fingerprint(after)==fingerprint(restarted)):
                raise ValueError("natural restart mismatch")
            # Independently replay actual accepted intervals: every internal
            # stage's measured conservation/topology must arise from the same
            # source state, not a caller-authored list of PASS labels.
            from .voiding_lifecycle_driver_v5 import advance_production_void_interval,NATURAL_WINDOW_S
            from .closure_mechanics_evidence import canonical_data
            replay=before;replayed_ops=[];cache={};accepted_intervals=[];failure=None
            for _ in range(2*row['partition_count']):
                replay,trace,result=advance_production_void_interval(replay,NATURAL_WINDOW_S/(2*row['partition_count']),
                    config=CFG,refinement_attempt_cache=cache)
                replayed_ops.extend(trace);accepted_intervals.append(result['elapsed_duration_s'])
                if result['failure'] is not None: failure=result['failure'];break
            replay_checks={
                'operations_exact':canonical_data(replayed_ops)==row['actual_operations'],
                'terminal_fingerprint_exact':fingerprint(replay)==fingerprint(after),
                'failure_exact':failure==row['failure'],
                'elapsed_physical_time_exact':math.fsum(accepted_intervals)==row['elapsed_physical_time_s'],
            }
            if not all(replay_checks.values()):
                raise ValueError('natural internal-stage independent production replay mismatch: '
                    f"case_identity={row['case_identity']} partition_count={row['partition_count']} "
                    f"checks={replay_checks}")
        if row["dataset"] == "neutrality":
            base=sources[row["base_terminal_checkpoint"]]
            if row["exact_neutrality"] != (fingerprint(after)==fingerprint(base)):
                raise ValueError("V12 neutrality source comparison mismatch")
        if row["dataset"] == "rollback":
            if row["restored_exactly"] != (fingerprint(after)==fingerprint(before)):
                raise ValueError("rollback source comparison mismatch")
        if row.get('stagewise_topology')!=stagewise_topology(after):
            raise ValueError('stagewise topology does not recompute from accepted source')
    return {"valid":True,"actual_state_rows":len(rows),"complete_final_registry_checked":False}


def lifecycle_decision(rows,sources):
    """Recompute success separately from valid evidence of a failed attempt."""
    transitions=[r for r in rows if r["dataset"]=="transitions"]
    partition=[]
    for row in transitions:
        reference=next(r for r in transitions if r["case_identity"]==row["case_identity"] and r["partition_count"]==1)
        elapsed=lambda r:math.fsum(float(op.get("duration_s",0.)) for op in r["actual_operations"])
        duration,reference_duration=elapsed(row),elapsed(reference)
        time_error=abs(duration-reference_duration)/max(abs(reference_duration),1e-300)
        if row["case_identity"] in ("ligament","downstream_child","child_continuation"):
            times=[]
            for r in (row,reference):
                before=sources[r["initial_checkpoint"]]; after=sources[r["terminal_checkpoint"]]
                times.append(after.junction_process_state.get("production_time_s",0.)-before.junction_process_state.get("production_time_s",0.))
            time_error=abs(times[0]-times[1])/max(abs(times[1]),1e-300)
        partition.append({"execution_id":row["execution_id"],"reference_execution_id":reference["execution_id"],
            "actual_transition":row["transition_occurred"],
            "event_time_relative_error":time_error,"frozen_event_time_limit":1e-12,
            "complete_terminal_exact":row["terminal_fingerprint"]==reference["terminal_fingerprint"],
            "passed":bool(row["transition_occurred"] and time_error<=1e-12 and row["terminal_fingerprint"]==reference["terminal_fingerprint"] and row["conservation"]["passed"])})
    restarts=[r for r in rows if r["dataset"]=="restarts"]
    natural=[]
    for row in [r for r in rows if r["dataset"]=="natural"]:
        ref=next(r for r in rows if r["dataset"]=="natural" and r["case_identity"]==row["case_identity"] and r["partition_count"]==1)
        natural.append({"execution_id":row["execution_id"],"partition_exact":row["terminal_fingerprint"]==ref["terminal_fingerprint"],
            "midpoint_restart_exact":row["midpoint_restart_exact"],
            "passed":row["terminal_fingerprint"]==ref["terminal_fingerprint"] and row["midpoint_restart_exact"]
                and row.get('failure') is None and row.get('replay_failure') is None
                and row.get('elapsed_physical_time_s')==row['input_configuration'].get('duration_s')
                and all(op.get('stagewise_conservation',{}).get('passed',False)
                    and op.get('stagewise_topology',{}).get('passed',False)
                    for op in row['actual_operations'] if op.get('duration_s',0.)>0. or op.get('api')=='promotion_remesh')
                and row["conservation"]["passed"]})
    controlled=[]
    controlled_v2=[]
    expected_v2={
        'accommodation_limited':'STABLE_SUBGRID_VOID_WITH_PLASTIC_ACCOMMODATION_MINIMUM_ALL_INTERVALS',
        'centered':'DOWNSTREAM_FRONT_ACTIVE',
        'delayed_downstream':'DOWNSTREAM_FRONT_ACTIVE_AFTER_DORMANT_INTERVAL_AND_RELOAD',
        'diffusion_limited':'STABLE_SUBGRID_VOID_WITH_VACANCY_TRANSPORT_MINIMUM_ALL_INTERVALS',
        'downstream_zero_drive':'CONNECTED_VOID_ZERO_DOWNSTREAM_DRIVE_WITH_QUALIFIED_SOURCE',
        'embryo_healing':'HEALED_SITE',
        'fixed_mesh_oblique':'CONNECTED_VOID_ZERO_DOWNSTREAM_DRIVE',
        'local_remesh_refinement':'DOWNSTREAM_FRONT_ACTIVE_AFTER_QUALIFIED_LOCAL_REFINEMENT',
        'long_ligament':'DOWNSTREAM_FRONT_ACTIVE',
        'negative_offset':'CONNECTED_VOID_ZERO_DOWNSTREAM_DRIVE_WITH_QUALIFIED_SOURCE',
        'positive_offset':'CONNECTED_VOID_ZERO_DOWNSTREAM_DRIVE_WITH_QUALIFIED_SOURCE',
        'short_ligament':'DOWNSTREAM_FRONT_ACTIVE',
    }
    for row in [r for r in rows if r["dataset"]=="controlled"]:
        state=sources[row["terminal_checkpoint"]]; case=row["case_identity"]
        phase=state.void_state.sites[0].phase if not state.void_state.cavities else state.void_state.cavities[0].phase
        success=False
        if case=="embryo_healing": success=phase==VoidPhase.HEALED_SITE
        elif case in ("diffusion_limited","accommodation_limited"):
            channel="vacancy_transport_s" if case=="diffusion_limited" else "plastic_accommodation_s"
            histories=[op for op in row["actual_operations"] if op.get("api")=="accepted_load_growth_interval"]
            success=bool(histories) and all(op["rates"][channel]==min(op["rates"][k] for k in
                ("surface_reaction_s","vacancy_transport_s","plastic_accommodation_s")) for op in histories)
        elif case=="local_remesh_refinement":
            success=any(op.get("api")=="refine_downstream_source" for op in row["actual_operations"])
            success &= phase==VoidPhase.DOWNSTREAM_FRONT_ACTIVE
        elif case in ("positive_offset","negative_offset","short_ligament","long_ligament","downstream_zero_drive"):
            success=phase==VoidPhase.CONNECTED_VOID and state.junction_process_state.get("latest_crack_void_connection_certificate",{}).get("passed",False)
            if case in ("positive_offset","negative_offset","downstream_zero_drive") and success:
                from .closure_production_evidence import candidate_measurements
                # A valid positive-drive connection is not the frozen
                # CONNECTED_VOID_ZERO_DOWNSTREAM_DRIVE terminal. Use the
                # owned exit probe, not an unrelated maximum surface tensor.
                success &= all(r["rates_before_resolution_guard"]["effective_rate_s"]==0.
                               for r in candidate_measurements(state))
        else:
            success=phase==VoidPhase.DOWNSTREAM_FRONT_ACTIVE and bool(state.crack_network.active_tip_ids)
        controlled.append({"case_identity":case,"actual_phase":phase.value,
            "passed":bool(success and row["failure"] is None and row["conservation"]["passed"])})
        operations=row.get('actual_operations',[])
        source_statuses=[op.get('audit',{}).get('status') for op in operations
            if isinstance(op.get('audit'),dict) and op.get('audit',{}).get('status')]
        zero_rows=[entry for op in operations if op.get('api')=='zero_drive_connected_interval'
            for entry in op.get('audit',[]) if isinstance(entry,dict)]
        growth=[op for op in operations if op.get('api')=='accepted_load_growth_interval']
        actual=phase.value
        protocol=False
        if phase==VoidPhase.HEALED_SITE:
            actual='HEALED_SITE';protocol=True
        elif case in ('diffusion_limited','accommodation_limited') and growth:
            channels=('surface_reaction_s','vacancy_transport_s','plastic_accommodation_s')
            minima=[min(channels,key=lambda key:op['rates'][key]) for op in growth]
            # Channel keys carry the dimensional ``_s`` suffix.  It is not
            # part of the controlled-history taxonomy and previously leaked
            # into only the accommodation enum.
            limiter=minima[0][:-2] if minima[0].endswith('_s') else minima[0]
            actual=('STABLE_SUBGRID_VOID_WITH_'+limiter.upper()+'_MINIMUM_ALL_INTERVALS'
                    if len(set(minima))==1 else 'STABLE_SUBGRID_VOID_WITH_MIXED_RATE_MINIMA')
            protocol=len(growth)==8
        elif phase==VoidPhase.DOWNSTREAM_FRONT_ACTIVE:
            if case=='delayed_downstream':
                actual='DOWNSTREAM_FRONT_ACTIVE_AFTER_DORMANT_INTERVAL_AND_RELOAD'
                protocol=bool(zero_rows) and all(entry.get('effective_rate_s')==0. for entry in zero_rows)
            elif case=='local_remesh_refinement':
                actual='DOWNSTREAM_FRONT_ACTIVE_AFTER_QUALIFIED_LOCAL_REFINEMENT'
                protocol=(any(op.get('api')=='actual_local_remesh' for op in operations)
                    and 'SOURCE_TENSOR_QUALIFIED' in source_statuses)
            else:
                actual='DOWNSTREAM_FRONT_ACTIVE';protocol=True
        elif phase==VoidPhase.CONNECTED_VOID:
            if 'SOURCE_TENSOR_UNQUALIFIED' in source_statuses or (row.get('failure') or {}).get('message')=='UNQUALIFIED_CAVITY_SOURCE_TENSOR':
                actual='CONNECTED_VOID_SOURCE_UNQUALIFIED';protocol=True
            elif zero_rows and all(entry.get('effective_rate_s')==0. and not entry.get('winner')
                                   for entry in zero_rows):
                actual=('CONNECTED_VOID_ZERO_DOWNSTREAM_DRIVE_WITH_QUALIFIED_SOURCE'
                        if case in ('positive_offset','negative_offset','downstream_zero_drive')
                        else 'CONNECTED_VOID_ZERO_DOWNSTREAM_DRIVE')
                protocol=True
            else: actual='CONNECTED_VOID_UNCLASSIFIED_SOURCE_STATE'
        expected=expected_v2[case]
        controlled_v2.append({'case_identity':case,'prospective_expected_classification':expected,
            'final_source_resolved_classification':actual,'source_qualification_statuses':source_statuses,
            'zero_drive_candidate_count':len(zero_rows),'protocol_complete':protocol,
            'failure':row.get('failure'),'passed':bool(actual==expected and protocol
                and row.get('failure') is None and row['conservation']['passed'])})
    rollback=[{"case_identity":r["case_identity"],"passed":r["restored_exactly"] and bool(r["failure"])
        and r['input_configuration'].get('intended_stage_reached',False)
        and r["failure"]["message"]=='injected:'+r['input_configuration']['failure_stage']}
        for r in rows if r["dataset"]=="rollback"]
    neutrality=[{"case_identity":r["case_identity"],"passed":r["exact_neutrality"] and r["failure"] is None}
        for r in rows if r["dataset"]=="neutrality"]
    continued_restarts=bool(restarts) and all(bool(r["continued_front_terminal_reached"])
        and r.get('subsequent_history_exact',False) and r['input_configuration']['requested_stage_available'] for r in restarts)
    same_terminal=bool(restarts) and len({r["terminal_fingerprint"] for r in restarts})==1
    downstream_rollback=all(any(r['case_identity']=='lifecycle:'+stage and r['passed'] for r in rollback)
        for stage in ('downstream_source_refinement','downstream_threshold_completion','child_creation',
                      'child_support_rebuild','child_tip_continuation'))
    complete=(all(r["passed"] for r in partition+natural+controlled_v2+rollback+neutrality)
              and all(r.get('stagewise_topology',{}).get('passed',False) and r['conservation']['passed'] for r in rows)
              and continued_restarts and same_terminal and downstream_rollback)
    return {"transition_partitions":partition,"natural_partitions_restart":natural,
        "controlled_histories":controlled,"controlled_histories_v2":controlled_v2,
        "rollback_attempts":rollback,"V12_disabled_neutrality":neutrality,
        "restart_to_attainable_terminal_exact":all(r["restart_exact"] for r in restarts),
        "all_restart_stages_reach_identical_complete_terminal":same_terminal,
        "required_continued_front_restart_terminal":continued_restarts,
        'stagewise_topology_and_conservation':all(r.get('stagewise_topology',{}).get('passed',False)
            and r['conservation']['passed'] for r in rows),
        "downstream_lifecycle_rollback": "PASS" if downstream_rollback else "NOT_EXERCISED_SOURCE_RESOLUTION_PREREQUISITE",
        "decision":"PASS" if complete else "BLOCKED"}
