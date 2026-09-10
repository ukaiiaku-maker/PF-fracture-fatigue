"""Account for every first passage, including energy-admitted zero advance."""
import math
import numpy as np


def validate_first_passage_history(geometry,kinetic,energy,stochastic,seed,engine_id):
    history=stochastic['hazard_threshold_history'];count=int(stochastic['hazard_event_index'])
    if len(history)!=count:raise ValueError('incomplete first-passage threshold history')
    rng=np.random.default_rng(np.random.SeedSequence([seed,engine_id]))
    expected=[max(float(rng.exponential(1.0)),1e-12) for _ in range(count+1)]
    if any(not math.isclose(float(a),b,rel_tol=1e-12) for a,b in zip(history,expected)):
        raise ValueError('first-passage threshold RNG provenance changed')
    if not math.isclose(float(stochastic['hazard_threshold_action']),expected[-1],rel_tol=1e-12):
        raise ValueError('next threshold RNG provenance changed')
    if stochastic['rng_state']!=rng.bit_generator.state:raise ValueError('terminal RNG state changed')
    passages=[r for r in kinetic if r['fired'] or r.get('hazard_energy_gate_attempt_consumed',False)]
    if len(passages)!=count or len(energy)!=count:raise ValueError('missing first-passage or energy attempt records')
    indexed={int(e['hazard_event_index']):e for e in geometry}
    if len(indexed)!=len(geometry) or any(i<0 or i>=count for i in indexed):raise ValueError('invalid geometry threshold index')
    zero=0
    for i,(record,gate) in enumerate(zip(passages,energy)):
        if not record['event_localized'] or not record['coupled_hazard_event_restart']:
            raise ValueError('first passage not localized or restarted')
        if i in indexed:
            event=indexed[i]
            if not record['fired'] or not gate.get('inserted',False) or gate['committed_event_length_m']<=0:
                raise ValueError('geometry event lacks admitted positive energy attempt')
            if not math.isclose(float(event['threshold_action']),expected[i],rel_tol=1e-12):
                raise ValueError('geometry event threshold index mismatch')
        else:
            if record['fired'] or not record.get('hazard_energy_gate_zero_length_attempt',False) or not record.get('hazard_energy_gate_attempt_consumed',False):
                raise ValueError('threshold gap lacks a consumed zero-length first passage')
            if gate.get('inserted',False) or gate['committed_event_length_m']!=0 or gate['energy_admissible_event_length_m']!=0 or record['energy_gated_event_advance_m']!=0 or not gate.get('arrest_reason'):
                raise ValueError('zero-length first passage advanced geometry or lacks energy evidence')
            zero+=1
    return zero


def first_passage_attempt_rows(geometry,kinetic,energy,stochastic,frequency_Hz):
    passages=[r for r in kinetic if r['fired'] or r.get('hazard_energy_gate_attempt_consumed',False)]
    indexed={int(e['hazard_event_index']):e for e in geometry};rows=[];extension=0.
    for i,(record,gate,threshold) in enumerate(zip(passages,energy,stochastic['hazard_threshold_history'])):
        event=indexed.get(i)
        if event:extension+=event['x1']-event['x0']
        action=event['event_transaction_audit']['hazard_action_completed'] if event else None
        rows.append(dict(hazard_event_index=i,threshold_action=threshold,physical_hazard_action=action,
            action_source='checked_event_transaction' if event else 'complete_action_not_persisted_for_zero_length_attempt',
            partial_block_hazard_action=record['physical_hazard_action_block'],cycles_total=record['time_s']*frequency_Hz,
            projected_extension_m=extension,geometry_advanced=event is not None,
            proposed_length_m=gate['stochastic_proposed_event_length_m'],energy_admitted_length_m=gate['energy_admissible_event_length_m'],
            committed_length_m=gate['committed_event_length_m'],event_K_Pa_sqrt_m=gate['event_K_Pa_sqrt_m'],energy_gate_reason=gate['arrest_reason']))
    return rows
