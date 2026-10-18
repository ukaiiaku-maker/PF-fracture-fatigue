#!/usr/bin/env python3
"""Accepted-state recurrence and renewal tracing for R10 opening alternatives."""
from __future__ import annotations
import argparse,copy,csv,hashlib,json,math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.cohesive_arrhenius import (BILINEAR,SMOOTH_EXPONENTIAL,
    COHESIVE_KINETIC_CZM,cohesive_shape_response)
from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.fatigue_v1 import FatigueWaveform
from arrhenius_fracture.opening_alternatives_r10 import OpeningAlternative,evaluate_opening
from arrhenius_fracture.persistent_site_high_cycle_state_v10230 import serialize_active_state
from arrhenius_fracture.retained_material_cohesive_r10 import load_cohesive_binding,build_cohesive_current_engine
from scripts.run_r10_global_mechanism_search import canonical_hash
from scripts.run_r10_opening_alternatives import thermal_from,mechanics

SCHEMA="r10.opening-renewal-recurrence/2"

def atomic(path,payload):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+'\n');tmp.replace(path)

def row_for(path,candidate):
    with path.open(newline='') as f:return next(r for r in csv.DictReader(f) if r['candidate_id']==candidate)

def sums(mpz):
    names=('mobile_positive','mobile_negative','retained_positive','retained_negative',
      'accumulated_slip_positive','accumulated_slip_negative','returned_slip_positive','returned_slip_negative',
      'wake_mobile_positive','wake_mobile_negative','wake_retained_positive','wake_retained_negative',
      'wake_slip_positive','wake_slip_negative')
    return {name:float(np.sum(np.asarray(getattr(mpz,name,0.),float))) for name in names}

def snapshot(engine,alternative,thermal,kenv,K,T,branch,action,threshold,cycle,phase,label,patch_m,
             opening_phase_duration_s):
    stress=max(float(engine.sigma_tip(max(float(K),0.))),0.)
    opening=evaluate_opening(alternative,stress/kenv,kenv,thermal,T,branch,
                             phase_duration_s=opening_phase_duration_s)
    point=opening.get('connected');sep=0. if point is None else float(point['delta_m'])
    traction,tangent,work=cohesive_shape_response(alternative.cohesive_shape,sep,thermal,T)
    mpz=engine.mpz
    return {'label':label,'physical_time_s':float(engine.t),'cycle':cycle,'phase':phase,
      'K_Pa_sqrt_m':float(K),'Kmax_convention':'Kmax; DeltaK=(1-R)Kmax with R=0.1',
      'crack_position_m':float(engine.checkpoint_advance_total_m),'path_coordinate_m':float(engine.a_adv),
      'material_cell_id':int(round(float(engine.checkpoint_advance_total_m)/patch_m)),
      'environment_imposed_opening_m':stress/kenv,'environment_stiffness_Pa_per_m':kenv,
      'cohesive_separation_m':sep,'cohesive_traction_Pa':float(traction),
      'cohesive_tangent_Pa_per_m':float(tangent),'combined_stability_margin_Pa_per_m':float(tangent)+kenv,
      'branch_state':opening['state'],'kinetic_validity':opening['kinetic_validity'],
      'barrier_kBT':opening.get('barrier_kBT'),'mfpt_s':opening.get('mfpt_s'),
      'opening_action':action,'pending_threshold':threshold,
      'tip_radius_m':float(engine.r_eff()),'K_shield_Pa_sqrt_m':float(engine.K_shield()),
      'source_backstress_Pa':float(getattr(mpz,'continuum_source_last_sigma_back_Pa',0.)),
      'tip_source_activity_mean':float(np.mean(getattr(mpz,'tip_source_activity',np.ones(1)))),
      'mobile_count':float(mpz.mobile_count),'retained_count':float(mpz.retained_count),
      'signed_state_sums':sums(mpz),'cohesive_storage_J_per_m':float(work)*patch_m,
      'connected_new_area_m2_per_m':patch_m,'surface_energy_if_completed_J_per_m':thermal.gamma(T)*patch_m,
      'defect_work_ledger_J_per_m':float(engine.W_emit),
      'external_work_J_per_m':None,
      'external_work_status':'UNAVAILABLE_IN_REDUCED_FIXED_K_CONDENSATION',
      'opening':opening}

def commit(engine,patch_m):
    wake=engine.mpz.advance(patch_m);engine.a_adv=float(engine.a_adv)+patch_m
    engine.n_adv=int(engine.n_adv)+1
    engine.checkpoint_advance_total_m=float(engine.checkpoint_advance_total_m)+patch_m
    return wake

def clone_engine_state(engine):
    """Clone mutable constitutive ownership without copying immutable kernels."""
    trial=copy.copy(engine);trial.mpz=copy.deepcopy(engine.mpz)
    if hasattr(engine,'_hazard_rng'):
      trial._hazard_rng=copy.deepcopy(engine._hazard_rng)
    return trial

def plastic_prefix(engine,duration_s,T,K,transport_substeps=1):
    trial=clone_engine_state(engine);duration=max(float(duration_s),0.)
    steps=max(int(transport_substeps),1);subduration=duration/steps
    for _ in range(steps):
      if subduration<=0.:break
      sig=max(float(trial.sigma_tip(max(float(K),0.))),0.)
      trial._plastic_half_step(.5*subduration,T,sig)
      sig=max(float(trial.sigma_tip(max(float(K),0.))),0.)
      trial._plastic_half_step(.5*subduration,T,sig)
      trial.t=float(trial.t)+subduration
    return trial

def classify(events,cycle_rows,path_endpoint_hit,unsupported,numeric_failure,max_cycles):
    event_cycles=sorted({int(math.floor(float(e['cycle'])+1e-12)) for e in events})
    if numeric_failure:return 'NUMERICAL_FAILURE'
    if path_endpoint_hit:return 'OBSERVATION_PATH_ENDPOINT_DURING_CASCADE'
    if len(event_cycles)>=2:return 'SUSTAINED_RECURRENT_GROWTH'
    residuals=[r['cycle_state_relative_change'] for r in cycle_rows[-8:] if r['event_count_cycle']==0]
    if len(residuals)>=8 and max(residuals)<1e-8:
        return 'SUPPORTED_CYCLIC_SHAKEDOWN_AFTER_BURST' if events else 'SUPPORTED_CYCLIC_SHAKEDOWN_NO_ADVANCE'
    if unsupported and not events:return 'UNSUPPORTED_FIRST_PASSAGE_CROSSOVER'
    if events:return 'STABLE_FINITE_FIRST_EXCURSION_BURST_AT_EXPOSURE_ENDPOINT'
    return 'PHYSICAL_EXPOSURE_ENDPOINT_NO_ADVANCE' if max_cycles>0 else 'RESOURCE_STOP'

def run_case(registry,candidate,shape,mode,Kmax,T,cycles,plastic_phases,opening_phases,patch_m,seed,
             path_limit_m=None,transport_substeps=1):
    if opening_phases%plastic_phases:raise ValueError('opening phases must be divisible by plastic phases')
    if int(transport_substeps)<1:raise ValueError('transport substeps must be positive')
    row=row_for(registry,candidate);c=json.loads(row['cohesive_parameter_vector'])
    binding=load_cohesive_binding(registry,candidate,execution_identity=COHESIVE_KINETIC_CZM)
    engine=build_cohesive_current_engine(binding,kernel_family_path=R4_KERNEL_FAMILY)
    thermal=thermal_from(c);_,kenv=mechanics(thermal,T)
    alt=OpeningAlternative(mode,shape,'quasistatic_mechanical' if mode=='cohesive_athermal' else 'time_dependent_first_passage',
      None if mode=='cohesive_athermal' else float(c['activation_area_m2']),None if mode=='cohesive_athermal' else float(c['attempt_frequency_s']))
    wave=FatigueWaveform(Kmax=Kmax*1e6,R=.1,frequency_Hz=1000.);period=wave.period_s
    rng=np.random.default_rng(seed);threshold=float(rng.exponential());action=0.;branch={'branch':'BONDED','delta_m':0.}
    events=[];transitions=[];cycle_rows=[];unsupported=False;domain_hit=False;numeric_failure=None
    process_support=float(engine.f.L_pz)
    domain=(process_support if path_limit_m is None else float(path_limit_m))
    if domain<=0.:raise ValueError('cohesive path limit must be positive')
    sub=opening_phases//plastic_phases
    try:
      for cycle in range(cycles):
        cycle_start=serialize_active_state(engine).vector;events_start=len(events)
        for ip in range(plastic_phases):
          p_mid=(ip+.5)/plastic_phases;Kmid=float(wave.K_phase(2*math.pi*p_mid));dt=period/plastic_phases
          for j in range(sub):
            phase=(ip+(j+.5)/sub)/plastic_phases
            K=float(wave.K_phase(2*math.pi*phase));odt=period/opening_phases
            phase_engine_start=clone_engine_state(engine)
            engine=plastic_prefix(phase_engine_start,odt,T,K,transport_substeps)
            snap=snapshot(engine,alt,thermal,kenv,K,T,branch,action,threshold,cycle,phase,'PHASE_STATE',patch_m,odt)
            opening=snap['opening'];branch=opening['connected_state']
            validity=opening['kinetic_validity'];unsupported|=validity in {
              'CROSSOVER_REQUIRES_FIRST_PASSAGE','NONADIABATIC_DISTRIBUTION_PROPAGATION_REQUIRED',
              'STATIC_SADDLE_MFPT_HAZARD_APPROXIMATION'}
            # A frozen saddle-arrival mean remains diagnostic until survival
            # and committed separation are qualified.
            rate=(opening.get('elementary_rate_s-1') if
                  validity=='QUALIFIED_HIGH_BARRIER' else None)
            increment=0. if rate is None else float(rate)*odt
            # Branch loss begins transit on this material cell; it is not yet
            # geometry-owned crack advance.
            mechanical=opening['state']=='SEPARATION_COMPLETED'
            action_start=action
            thermal_fire=mode=='cohesive_mixed' and action_start+increment>=threshold
            trial_endpoint_action=action_start+increment
            action=trial_endpoint_action
            if mechanical or thermal_fire:
              trigger='MECHANICAL_COMPLETION' if mechanical else 'COMMON_COORDINATE_FIRST_PASSAGE'
              if thermal_fire and increment>0:
                fraction=np.clip((threshold-action_start)/increment,0.,1.)
                # Recompute the accepted constitutive prefix instead of
                # publishing the full-phase trial state.  Within this declared
                # piecewise-constant load interval, rate and plastic state are
                # both evaluated from the same prefix propagator.
                lo=0.;hi=1.;accepted_engine=None
                for _ in range(32):
                  mid=.5*(lo+hi);trial=plastic_prefix(phase_engine_start,mid*odt,T,K,transport_substeps)
                  trial_snap=snapshot(trial,alt,thermal,kenv,K,T,branch,action_start,
                    threshold,cycle,phase,'EVENT_LOCALIZATION_TRIAL',patch_m,odt)
                  trial_open=trial_snap['opening'];trial_rate=(trial_open.get('elementary_rate_s-1')
                    if trial_open['kinetic_validity']=='QUALIFIED_HIGH_BARRIER' else None)
                  trial_action=action_start+(0. if trial_rate is None else float(trial_rate)*mid*odt)
                  if trial_action>=threshold:hi=mid;accepted_engine=trial
                  else:lo=mid
                fraction=hi
                engine=(accepted_engine if accepted_engine is not None else
                        plastic_prefix(phase_engine_start,fraction*odt,T,K,transport_substeps))
                action=threshold
              else:fraction=1.
              event_phase=phase+(fraction-.5)/opening_phases
              accepted_time=(cycle+event_phase)*period
              remaining_phase_time=max((1.-fraction)*odt,0.)
              while True:
                before=snapshot(engine,alt,thermal,kenv,K,T,branch,action,threshold,cycle,event_phase,'PRE_SEPARATION',patch_m,odt)
                for state_row in (before,):
                  state_row['operator_split_state_time_s']=state_row['physical_time_s']
                  state_row['physical_time_s']=accepted_time
                if float(engine.checkpoint_advance_total_m)+patch_m>domain+1e-15:
                    domain_hit=True;break
                wake=commit(engine,patch_m)
                after=snapshot(engine,alt,thermal,kenv,K,T,{'branch':'BONDED','delta_m':0.},0.,threshold,cycle,event_phase,'POST_REMAP',patch_m,odt)
                reeq=snapshot(engine,alt,thermal,kenv,K,T,after['opening']['connected_state'],0.,threshold,cycle,event_phase,'POST_REEQUILIBRATION',patch_m,odt)
                for state_row in (after,reeq):
                  state_row['operator_split_state_time_s']=state_row['physical_time_s']
                  state_row['physical_time_s']=accepted_time
                event={'event_index':len(events)+1,'cycle':accepted_time/period,'trigger':trigger,
                  'accepted_event_time_s':accepted_time,'accepted_action':action,
                  'discarded_trial_endpoint_action':trial_endpoint_action,
                  'remaining_phase_time_after_event_s':remaining_phase_time,
                  'advance_m':patch_m,'extension_m':float(engine.checkpoint_advance_total_m),
                  'pre':before,'post_remap':after,'post_reequilibrated':reeq,'wake_transaction':wake,
                  'delta_tip_radius_m':after['tip_radius_m']-before['tip_radius_m'],
                  'delta_K_shield_Pa_sqrt_m':after['K_shield_Pa_sqrt_m']-before['K_shield_Pa_sqrt_m'],
                  'delta_mobile_count':after['mobile_count']-before['mobile_count'],
                  'delta_retained_count':after['retained_count']-before['retained_count']}
                events.append(event);transitions.extend([before,after,reeq])
                branch=reeq['opening']['connected_state'];action=0.;threshold=float(rng.exponential())
                next_open=reeq['opening'];next_rate=(next_open.get('elementary_rate_s-1')
                  if next_open['kinetic_validity']=='QUALIFIED_HIGH_BARRIER' else None)
                mechanical=next_open['state']=='SEPARATION_COMPLETED'
                next_wait=(math.inf if next_rate is None or float(next_rate)<=0. else
                           threshold/float(next_rate))
                thermal_fire=(mode=='cohesive_mixed' and next_wait<=remaining_phase_time)
                if not (mechanical or thermal_fire):break
                if thermal_fire:
                  engine=plastic_prefix(engine,next_wait,T,K,transport_substeps)
                  accepted_time+=next_wait;remaining_phase_time-=next_wait
                  event_phase=accepted_time/period-cycle;action=threshold
                  trial_endpoint_action=float(next_rate)*(remaining_phase_time+next_wait)
                trigger='MECHANICAL_COMPLETION' if mechanical else 'COMMON_COORDINATE_FIRST_PASSAGE'
              if not domain_hit and not mechanical and next_rate is not None and not thermal_fire:
                action=float(next_rate)*remaining_phase_time
              if not domain_hit and remaining_phase_time>0.:
                engine=plastic_prefix(engine,remaining_phase_time,T,K,transport_substeps)
              if domain_hit:break
          if domain_hit:break
        end=serialize_active_state(engine).vector
        denom=max(float(np.linalg.norm(cycle_start)),1.)
        cycle_rows.append({'cycle':cycle+1,'physical_time_s':float(engine.t),'event_count_cycle':len(events)-events_start,
          'cumulative_events':len(events),'extension_m':float(engine.checkpoint_advance_total_m),
          'cycle_state_relative_change':float(np.linalg.norm(end-cycle_start)/denom),
          'mobile_count':float(engine.mpz.mobile_count),'retained_count':float(engine.mpz.retained_count),
          'tip_radius_m':float(engine.r_eff()),'K_shield_Pa_sqrt_m':float(engine.K_shield()),
          'source_backstress_Pa':float(getattr(engine.mpz,'continuum_source_last_sigma_back_Pa',0.))})
        if domain_hit:break
    except Exception as exc:
      numeric_failure=f'{type(exc).__name__}: {exc}'
    stop=classify(events,cycle_rows,domain_hit,unsupported,numeric_failure,cycles)
    reduced_support_exceeded=any(float(r['tip_radius_m'])>process_support for r in cycle_rows)
    if reduced_support_exceeded and numeric_failure is None:
      stop='REDUCED_MECHANICS_SUPPORT_CENSOR'
    return {'schema':SCHEMA,'candidate_id':candidate,'shape':shape,'mode':mode,'temperature_K':T,
      'Kmax_MPa_sqrt_m':Kmax,'DeltaK_MPa_sqrt_m':.9*Kmax,'R':.1,'frequency_Hz':1000.,
      'plastic_phase_count':plastic_phases,'opening_phase_count':opening_phases,'patch_m':patch_m,
      'transport_substeps_per_opening_interval':int(transport_substeps),
      'process_zone_support_m':process_support,'cohesive_path_observation_limit_m':domain,
      'path_support_kind':('LOCAL_SUPPORT_ONLY' if path_limit_m is None else
        'MOVING_MPZ_WINDOW_WITH_SEPARATE_COHESIVE_PATH_OBSERVATION_LIMIT'),
      'reduced_mechanics_support_exceeded':reduced_support_exceeded,
      'stop_category':stop,'numeric_failure':numeric_failure,
      'events':events,'transition_states':transitions,'cycle_history':cycle_rows,
      'unsupported_crossover_encountered':unsupported,'legacy_cooperative_clock_used':False,
      'same_complete_material_sha256':row['complete_row_sha256'],'paired_plasticity_sha256':row['parent_complete_row_sha256'],
      'rng_state':rng.bit_generator.state,'pending_threshold':threshold,'pending_action':action}

def trace(args):
    result=run_case(args.registry,args.candidate_id,args.shape,args.mode,args.Kmax,args.temperature_K,
      args.cycles,args.plastic_phases,args.opening_phases,args.patch_um*1e-6,args.seed,
      None if args.path_limit_um is None else args.path_limit_um*1e-6,args.transport_substeps)
    args.out.mkdir(parents=True,exist_ok=True);atomic(args.out/'result.json',result)
    print(json.dumps({'candidate':args.candidate_id,'shape':args.shape,'mode':args.mode,
      'events':len(result['events']),'cycles':len(result['cycle_history']),'stop':result['stop_category'],
      'extension_m':result['cycle_history'][-1]['extension_m'] if result['cycle_history'] else 0.},indent=2));return 0

def crossed_registry(args):
    with args.source.open(newline='') as f:rows={r['candidate_id']:r for r in csv.DictReader(f)}
    specs=[('COH27231_PL27231','R10D_COH_27231','R10D_COH_27231'),
      ('COH27231_PL30745','R10D_COH_27231','R10D_COH_30745'),('COH27231_PL31630','R10D_COH_27231','R10D_COH_31630'),
      ('COH27231_PL11267','R10D_COH_27231','R10D_COH_11267'),
      ('COH27231_PL26045','R10D_COH_27231','R10D_COH_26045'),
      ('COH27231_PL15732','R10D_COH_27231','R10D_COH_15732'),
      ('COH27231_PL19672','R10D_COH_27231','R10D_COH_19672'),
      ('COH30745_PL27231','R10D_COH_30745','R10D_COH_27231'),('COH00167_PL27231','R10D_COH_00167','R10D_COH_27231'),
      ('COH31630_PL31630','R10D_COH_31630','R10D_COH_31630')]
    out=[]
    for label,coh_id,pl_id in specs:
      coh=rows[coh_id];pl=rows[pl_id];r=dict(coh);r['candidate_id']='R10E_'+label
      r['parent_candidate_id']=pl['parent_candidate_id'];r['parent_complete_row_sha256']=pl['parent_complete_row_sha256']
      r['parent_complete_parameter_vector']=pl['parent_complete_parameter_vector'];r['paired_plasticity_sha256']=pl['parent_complete_row_sha256']
      r['cross_design_kind']=('BASE' if coh_id==pl_id else 'FIXED_COHESION_VARY_PLASTICITY' if coh_id=='R10D_COH_27231' else 'FIXED_PLASTICITY_VARY_COHESION')
      parent=json.loads(pl['parent_complete_parameter_vector'])
      metadata={'candidate_id','option_key','material_class','role','mechanism_summary',
        'validation_status','search_source_id','coordinate_source_Tref_K',
        'coordinate_rereference_only'}
      consumed={k:v for k,v in parent.items() if k not in metadata and not k.startswith('cleave_')}
      r['plasticity_physics_sha256']=canonical_hash(consumed)
      complete={'family':r['family'],'candidate_id':r['candidate_id'],'parent_sha256':r['parent_complete_row_sha256'],'cohesive_sha256':r['cohesive_parameter_sha256']}
      r['complete_row_sha256']=canonical_hash(complete);out.append(r)
    fields=[]
    for r in out:
      for k in r:
       if k not in fields:fields.append(k)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('w',newline='') as f:w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(out)
    print(json.dumps({'rows':len(out),'out':str(args.out)},indent=2));return 0

def main(argv=None):
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('trace');t.add_argument('--registry',type=Path,required=True);t.add_argument('--candidate-id',required=True);t.add_argument('--shape',choices=(SMOOTH_EXPONENTIAL,BILINEAR),required=True);t.add_argument('--mode',choices=('cohesive_athermal','cohesive_mixed'),required=True);t.add_argument('--Kmax',type=float,required=True);t.add_argument('--temperature-K',type=float,default=300.);t.add_argument('--cycles',type=int,default=32);t.add_argument('--plastic-phases',type=int,default=8);t.add_argument('--opening-phases',type=int,default=64);t.add_argument('--transport-substeps',type=int,default=1);t.add_argument('--patch-um',type=float,default=1.);t.add_argument('--path-limit-um',type=float,default=None);t.add_argument('--seed',type=int,default=20261017);t.add_argument('--out',type=Path,required=True);t.set_defaults(func=trace)
    c=sub.add_parser('crossed-registry');c.add_argument('--source',type=Path,required=True);c.add_argument('--out',type=Path,required=True);c.set_defaults(func=crossed_registry)
    a=p.parse_args(argv);return a.func(a)
if __name__=='__main__':raise SystemExit(main())
