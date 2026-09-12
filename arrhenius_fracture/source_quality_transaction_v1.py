"""Quality repair and prospective source/time certification, no forced event."""
from dataclasses import replace
import math
import numpy as np

from .quality_constrained_mesh_v1 import constrained_quality_mesh, production_geometry_constraints
from .source_resolution_protocol_v1 import ALLOCATED_LOG_RATE_BUDGET, ALLOCATED_WAITING_TIME_RELATIVE_ERROR
from .finalization_v3_schema import SCIENTIFIC_ACCEPTANCE_TOLERANCES as LIMITS

PROOF_SCHEMA = 'v5.quality-constrained-source-refinement/1'


def source_transfer_budget(current, previous, current_tensor, previous_tensor, *, temperature_K=900.):
    """Identical owned clocks, stable finite-positive rate comparisons only."""
    from .voiding_production_v5 import directional_clock_rates
    from .sharp_front import KB
    if current.competition != previous.competition or current.rng_state != previous.rng_state:
        return {'passed': False, 'failure': 'CLOCK_CANDIDATE_OR_RNG_IDENTITY_CHANGED'}
    a, b = current.void_state.cavities[0], previous.void_state.cavities[0]
    if (a.cavity_id, a.connection_exit_m, a.connection_direction_xy) != (b.cavity_id, b.connection_exit_m, b.connection_direction_xy):
        return {'passed': False, 'failure': 'OWNED_PHYSICAL_SOURCE_CHANGED'}
    first = directional_clock_rates(current, current_tensor, temperature_K=temperature_K)
    second = directional_clock_rates(previous, previous_tensor, temperature_K=temperature_K)
    rows = []
    for p, q in zip(first, second):
        rate, reference = p['effective_rate_s'], q['effective_rate_s']
        if p['candidate_id'] != q['candidate_id']: return {'passed': False, 'failure': 'CANDIDATE_ORDER_CHANGED'}
        if min(rate, reference) <= np.finfo(float).tiny or not math.isfinite(rate+reference):
            rows.append({'candidate_id': p['candidate_id'], 'classification': 'INACTIVE_OR_UNDERFLOW_REFERENCE',
                         'passed': rate == reference == 0., 'positive': False})
            continue
        log_error = abs(math.log(rate)-math.log(reference))
        barrier_error = abs(p['hazard_barrier_J']-q['hazard_barrier_J'])/(KB*temperature_K)
        crossing_error = abs(p['crossing_time_s']-q['crossing_time_s'])/max(abs(q['crossing_time_s']), np.finfo(float).tiny)
        rows.append({'candidate_id': p['candidate_id'], 'classification': 'POSITIVE_SAME_OWNED_THRESHOLD',
            'log_rate_error': log_error, 'effective_barrier_error_over_kBT': barrier_error,
            'crossing_time_relative_error': crossing_error, 'positive': True,
            'passed': log_error <= ALLOCATED_LOG_RATE_BUDGET and barrier_error <= ALLOCATED_LOG_RATE_BUDGET
                      and crossing_error <= ALLOCATED_WAITING_TIME_RELATIVE_ERROR})
    return {'schema': 'v5.prospective-source-time-budget/1', 'temperature_K': temperature_K,
        'allocated_log_rate_limit': ALLOCATED_LOG_RATE_BUDGET,
        'allocated_waiting_time_relative_limit': ALLOCATED_WAITING_TIME_RELATIVE_ERROR,
        'rows': rows, 'passed': bool(rows and all(r['passed'] for r in rows)),
        'positive_candidate_exists': any(r['positive'] for r in rows)}


def repair_connected_quality(state, *, strategy='flips', failure_stage=None, operation_log=None):
    """Isolated state transfer/support/equilibrium trial with frozen checks."""
    from . import voiding_production_v5 as p
    from .closure_lifecycle_evidence import conservation, stagewise_topology
    from .topology_transaction_v11 import complete_accepted_state_fingerprint as fingerprint
    if state.crack_network.active_tip_ids or state.void_state.cavities[0].phase != p.VoidPhase.CONNECTED_VOID:
        raise ValueError('quality source repair requires dormant connected state')
    original = fingerprint(state); cavity = state.void_state.cavities[0]
    before = p.observables(state, 'quality_repair_before'); old_metrics = p.cavity_source_resolution_metrics(state)
    fixed, edges = production_geometry_constraints(state)
    mesh, geometry = constrained_quality_mesh(state.mesh, fixed_nodes=fixed, protected_edges=edges, strategy=strategy)
    operations = [] if operation_log is None else operation_log
    def inject(stage):
        operations.append(stage)
        if stage == failure_stage: raise RuntimeError('injected:'+stage)
    inject('source_quality_geometry')
    if not geometry['geometrical_quality_target_passed']:
        return state, {'accepted': False, 'geometry': geometry, 'failure': 'GLOBAL_QUALITY_TARGET_UNRESOLVED'}
    fields = p._project_fields(state, mesh)
    trial = replace(state, mesh=mesh, damage=fields['damage'], displacement=fields['displacement'],
                    ep_gp=fields['ep_gp'], rho_gp=fields['rho_gp'])
    inject('source_quality_field_transfer')
    trial = p._prepare_connected_ligament_support(trial, entry=cavity.connection_entry_m,
        exit_point=cavity.connection_exit_m, direction=cavity.connection_direction_xy, new_connection=False)
    fields = {name: getattr(trial, name) for name in ('damage', 'displacement', 'ep_gp', 'rho_gp', 'tip_process_state')}
    fields['source_state'] = trial.junction_process_state.get('source_state', {})
    trial = p.remesh_mechanically_separating_v12(trial, mesh=trial.mesh, boundary=trial.boundary,
        transferred_fields=fields, source_commit=p._head(), configuration={'event': 'SOURCE_QUALITY_REPAIR_V1'},
        transaction_identity='source-quality-repair-v1')
    inject('source_quality_support_rebuild')
    trial = p.equilibrate_fixed_load_with_production_fem(trial)
    inject('source_quality_equilibrium')
    after = p.observables(trial, 'quality_repair_after'); metrics = p.cavity_source_resolution_metrics(trial)
    errors = {key: abs(after[key]-before[key])/max(abs(before[key]), 1e-300)
              for key in ('reaction_N_per_m', 'compliance_m2_per_N', 'energy_J_per_m')}
    tensor_error = float(np.linalg.norm(np.asarray(metrics['tensor_Pa'])-old_metrics['tensor_Pa'])/max(np.linalg.norm(old_metrics['tensor_Pa']), 1e-300))
    kinetic_transfer = source_transfer_budget(trial,state,metrics['tensor_Pa'],old_metrics['tensor_Pa'])
    accounting = conservation(trial, state); topology = stagewise_topology(trial)
    checks = {'quality': metrics['minimum_quality'] >= .05,
        'reaction': errors['reaction_N_per_m'] <= LIMITS['static_mesh_reaction_relative'],
        'compliance': errors['compliance_m2_per_N'] <= LIMITS['static_mesh_reaction_relative'],
        'energy': errors['energy_J_per_m'] <= LIMITS['static_mesh_energy_relative'],
        'fixed_source_tensor': tensor_error <= LIMITS['tensor_probe_relative'],
        'repair_barrier_rate_waiting_time_budget': kinetic_transfer['passed'],
        'free_residual': after['free_dof_residual_l2_N_per_m'] <= LIMITS['free_residual_relative']*max(after['constrained_reaction_l2_N_per_m'], 1e-300),
        'reaction_balance': after['top_bottom_reaction_balance'] <= LIMITS['reaction_balance_relative'],
        'energy_identity': after['energy_reaction_identity'] <= LIMITS['energy_reaction_identity_relative'],
        'void_physical_state': trial.void_state == state.void_state,
        'graph_geometry': trial.crack_network == state.crack_network,
        'clocks_rng_candidates': trial.competition == state.competition and trial.rng_state == state.rng_state,
        'topology': topology['passed'], 'accounting': accounting['passed'], 'caller_unchanged': fingerprint(state) == original}
    audit = {'schema': 'v5.source-quality-transaction/1', 'accepted': all(checks.values()),
        'checks': checks, 'geometry': geometry, 'relative_errors': errors, 'tensor_relative_error': tensor_error,
        'before': before, 'after': after, 'source_metrics': metrics, 'topology': topology, 'accounting': accounting,
        'source_transfer_budget': kinetic_transfer,
        'operations': operations}
    if not audit['accepted']: return state, audit
    # Invalidate any prior numerical authority; bind fresh source provenance,
    # but keep every clock unavailable until a separate refinement proof passes.
    junction = dict(trial.junction_process_state); junction.pop('cavity_source_resolution_proof', None)
    old_source = junction.get('active_event_source', {})
    probe = {'kind': 'direct_cavity_boundary_tensor', 'boundary_node_id': metrics['boundary_node_id'], 'element_ids': metrics['probe_element_ids']}
    source = {**old_source, **p._source_identity(trial, metrics['tensor_Pa'], source_kind='cavity_surface',
        source_cavity_id=cavity.cavity_id, source_boundary_site_id='connection_exit',
        source_position_m=cavity.connection_exit_m, source_probe_identity=probe)}
    source['candidate_source_states'] = tuple({**row, 'source_probe_identity': probe,
        'source_tensor_fingerprint': p._tensor_fingerprint(metrics['tensor_Pa']),
        'source_mesh_generation': int(trial.event_counters.get('mesh_generation', 0)),
        'source_geometry_generation': int(trial.crack_network.geometry_generation),
        'geometry_status': 'GEOMETRICALLY_VALID_KINETICALLY_DORMANT',
        'instantaneous_status': 'UNQUALIFIED_CAVITY_SOURCE_TENSOR', 'effective_rate_s': 0., 'crossing_time_s': math.inf}
        for row in old_source.get('candidate_source_states', ()))
    junction.update({'active_event_source': source, 'source_quality_repair_v1': audit})
    inject('source_quality_acceptance')
    return replace(trial, junction_process_state=junction), audit
