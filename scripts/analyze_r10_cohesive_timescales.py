#!/usr/bin/env python3
"""Cross opening and native source timescales for kinetic-CZM acquisition.

This is an acquisition diagnostic, not a substitute for developed fatigue.
The source exposure is a short, actual native state update; its local rate is
never multiplied by the site count because that multiplicity is already owned
by the persistent-source implementation.
"""
from __future__ import annotations

import argparse,csv,json,math,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.cohesive_arrhenius import COHESIVE_BARRIER_ONLY,COHESIVE_KINETIC_CZM
from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.paris_slope_r9 import configure_stochastic_engine
from arrhenius_fracture.reduced_persistent_reference_r2 import runtime_receipt
from arrhenius_fracture.retained_material_cohesive_r10 import build_cohesive_current_engine,load_cohesive_binding
from run_r10_kinetic_czm_coupled import mechanics,phase_landscape,probe_loads

def atomic(path,payload):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+'\n');tmp.replace(path)

def stable_receipt(value):
    if isinstance(value,dict):return {k:stable_receipt(v) for k,v in value.items() if k!='object_id'}
    if isinstance(value,list):return [stable_receipt(v) for v in value]
    return value

def locate(registries,candidate):
    for path in registries:
        with path.open(newline='') as stream:
            if any(row['candidate_id']==candidate for row in csv.DictReader(stream)):return path
    raise KeyError(candidate)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--registries',type=Path,nargs='+',required=True)
    p.add_argument('--candidates',nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--observed-run',type=Path)
    p.add_argument('--phase-count',type=int,default=12);p.add_argument('--micro-exposure-s',type=float,default=1e-9)
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
    curves=[];summaries=[];identities={}
    for candidate in a.candidates:
        registry=locate(a.registries,candidate)
        kinetic=load_cohesive_binding(registry,candidate,execution_identity=COHESIVE_KINETIC_CZM)
        barrier=kinetic.manifest.cleavage;mech=mechanics(barrier,400e9/(1-.22**2))
        rows,picked=probe_loads(kinetic,barrier,mech['environment_stiffness_Pa_per_m'],300.,.1,1000.,a.phase_count)
        for row in rows:curves.append({'candidate_id':candidate,**row})
        eligible=[x for x in rows if not x['spinodal_reached'] and x['action_per_cycle']>0]
        if not eligible:
            summaries.append({'candidate_id':candidate,'disposition':'NO_CONNECTED_POSITIVE_ACTION_IN_DOMAIN'});continue
        # Target order-one opening action, while retaining the exact dense curve.
        selected=min(eligible,key=lambda x:abs(math.log10(x['action_per_cycle'])-math.log10(1.5)))
        K=float(selected['Kmax_MPa_sqrt_m']);action=float(selected['action_per_cycle'])
        engine=configure_stochastic_engine(kinetic,kernel_family_path=R4_KERNEL_FAMILY,seed=20261017)
        threshold=float(engine.hazard_threshold_action);wait_cycles=threshold/action;wait_s=wait_cycles/1000.
        land,rate,traction,free,imposed=phase_landscape(engine,barrier,K*1e6,300.,mech['environment_stiffness_Pa_per_m'],{'branch':'INTACT'},kinetic.reduced_config.cleavage_hits,kinetic.reduced_config.cleavage_tau_s)
        engine.mpz._reversible_transport_K_signed_Pa_sqrt_m=K*1e6
        before_a=float(getattr(engine.mpz,'signed_source_activations_total',0.));before_l=float(getattr(engine.mpz,'signed_line_content_emitted_total',0.))
        step=engine._plastic_half_step(a.micro_exposure_s,300.,traction)
        da=float(getattr(engine.mpz,'signed_source_activations_total',0.))-before_a
        dl=float(getattr(engine.mpz,'signed_line_content_emitted_total',0.))-before_l
        projected=da/a.micro_exposure_s*wait_s
        disposition=('OPENING_BURST_BEFORE_DEVELOPED_STATE' if min(x['action_per_cycle'] for x in eligible)>=10. else
            'OPENING_DOMINATED_OVER_INITIAL_WAIT__ADVANCING_CONTROL_ONLY' if projected<1e-2 else
            'SOURCE_AND_OPENING_TIMESCALES_BOTH_ACTIVE__ADVANCING_RUN_REQUIRED')
        summaries.append({'candidate_id':candidate,'registry':str(registry),'selected_Kmax_MPa_sqrt_m':K,
            'selected_action_per_cycle':action,'seeded_threshold_action':threshold,
            'initial_opening_wait_cycles':wait_cycles,'initial_opening_wait_s':wait_s,
            'peak_load_cooperative_opening_rate_s-1':rate,'micro_exposure_s':a.micro_exposure_s,
            'source_activations_in_micro_exposure':da,'signed_lines_emitted_in_micro_exposure':dl,
            'aggregate_source_hazard_s-1':float(getattr(engine.mpz,'continuum_source_last_aggregate_hazard_s',0.)),
            'mobile_count_after_micro_exposure':float(engine.mpz.mobile_count),
            'retained_count_after_micro_exposure':float(engine.mpz.retained_count),
            'initial_rate_source_activations_over_opening_wait':projected,
            'source_step':step,'disposition':disposition,
            'multiplicity_handling':'NATIVE_AGGREGATE_ALREADY_INCLUDES_PERSISTENT_SITE_MULTIPLICITY__NO_SECOND_MULTIPLICATION',
            'scope':'DENSE_12_PHASE_INITIAL_CURVE_PLUS_NATIVE_MICRO_EXPOSURE__NOT_DEVELOPED_FATIGUE'})
        identities[candidate]={}
        for identity in (COHESIVE_BARRIER_ONLY,COHESIVE_KINETIC_CZM):
            binding=load_cohesive_binding(registry,candidate,execution_identity=identity)
            identities[candidate][identity]=stable_receipt(runtime_receipt(build_cohesive_current_engine(binding,kernel_family_path=R4_KERNEL_FAMILY),stage='r10_cohesive_timescale_acquisition'))
    with (a.out/'dense_opening_curves.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,curves[0].keys(),lineterminator='\n');w.writeheader();w.writerows(curves)
    observed=[]
    if a.observed_run:
        for path in sorted(a.observed_run.glob('Kmax_*/*/result.json')):
            result=json.loads(path.read_text());last=(result['events'][-1] if result.get('events') else {})
            observed.append({'path':str(path),'candidate_id':result['candidate_id'],'arm':result['arm'],
                'events':len(result.get('events',[])),'cycles_to_last_event':last.get('cycles'),
                'extension_m':result.get('final_extension_m'),'mobile_count_at_event':last.get('mobile_count'),
                'retained_count_at_event':last.get('retained_count'),
                'source_activations_at_event':last.get('source_activations_total'),
                'signed_line_content_at_event':last.get('signed_line_content_emitted_total'),
                'interpretation':'OPENING_BURST_CONTROL__NOT_A_DEVELOPED_FATIGUE_RATE'})
    payload={'schema':'r10.cohesive-timescale-acquisition/1','phase_count':a.phase_count,
        'frequency_Hz':1000.,'load_ratio':.1,'summaries':summaries,
        'runtime_identities':identities,
        'kinetic_opening_consumer':{'stress_coordinate':'UNCAPPED_ELASTIC_K_FIELD_AT_RETAINED_RADIUS',
            'legacy_30_GPa_source_cap_applied':False,
            'landscape':'FINITE_COMPLIANCE_CONNECTED_EXPONENTIAL_COHESIVE_POTENTIAL'},
        'observed_advancing_controls':observed,
        'interpretation':'Twelve-phase curves select exposure scales only; they do not establish long-time inactivity or developed Paris response.'}
    atomic(a.out/'timescale_acquisition.json',payload);print(json.dumps(payload,indent=2,sort_keys=True,default=str));return 0

if __name__=='__main__':raise SystemExit(main())
