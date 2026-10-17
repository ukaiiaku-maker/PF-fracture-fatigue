#!/usr/bin/env python3
"""Advancing finite-compliance kinetic-CZM comparison with retained R10 state."""
from __future__ import annotations

import argparse,csv,json,math
from pathlib import Path
import sys
from types import MethodType
import numpy as np
from scipy.special import gammainc

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.cohesive_arrhenius import COHESIVE_KINETIC_CZM
from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.fatigue_v1 import FatigueWaveform
from arrhenius_fracture.paris_slope_r9 import configure_stochastic_engine
from arrhenius_fracture.retained_material_cohesive_r10 import load_cohesive_binding

def atomic(path,payload):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+'\n');tmp.replace(path)

def cooperative(raw,hits,tau):return float(gammainc(hits,min(max(raw,0.)*tau,1e12))/tau)

def mechanics(barrier,Eprime):
    gamma=barrier.thermal.gamma(300.);peak=barrier.thermal.t_hat(300.)
    zone=Eprime*gamma/peak**2
    kenv=Eprime/(2.*zone)
    return {'Eprime_Pa':Eprime,'condensed_half_height_m':zone,'environment_stiffness_Pa_per_m':kenv,
        'derivation':'k_env=Eprime/(2*l_cz), l_cz=Eprime*Gamma/t_hat^2','fit_coordinate':False}

def phase_values(waveform,n):
    phi=2*math.pi*(np.arange(n)+.5)/n
    return np.asarray(waveform.K_phase(phi),float)

def phase_landscape(engine,barrier,K,T,kenv,connected,hits,tau):
    free=max(float(engine.sigma_tip(max(K,0.))),0.)
    imposed=free/kenv
    landscape=barrier.rate_from_environment(imposed,kenv,T,connected)
    point=landscape.get('connected')
    separation=0. if point is None else float(point['delta_m'])
    traction=float(barrier.thermal.gamma(T)/barrier.thermal.delta0_m*(separation/barrier.thermal.delta0_m)*math.exp(-separation/barrier.thermal.delta0_m))
    rate=cooperative(landscape['elementary_rate_s-1'],hits,tau)
    return landscape,rate,traction,free,imposed

def advance_override_phase(engine,K,T,dt,traction,rate):
    """One constant-rate Strang phase with exact threshold localization."""
    if hasattr(engine,'_synchronize_driver_checkpoint_length'):
        engine._synchronize_driver_checkpoint_length()
    event_length=float(getattr(engine,'avalanche_event_advance_m',engine.f.da))
    threshold=max(float(engine.hazard_threshold_action),1e-300)
    progress=max(float(rate),0.)/threshold
    remaining_progress=max(1.-float(engine.B),0.)
    h=min(float(dt),remaining_progress/progress) if progress>0 else float(dt)
    h=max(min(h,float(dt)),0.)
    engine.mpz._reversible_transport_K_signed_Pa_sqrt_m=float(K)
    engine.mpz._reversible_tip_radius_m=float(engine.r_eff())
    engine._plastic_half_step(.5*h,T,traction)
    dB=min(progress*h,remaining_progress);dH=dB*threshold;da=event_length*engine.tip_cfg.velocity_scale*dB
    if da>0:engine.mpz.advance(da)
    engine._plastic_half_step(.5*h,T,traction)
    engine.B+=dB;engine.hazard_action_current+=dH;engine.micro_advance_total_m+=da;engine.t+=h
    engine.hazard_last_progress_rate_s=progress
    fired=engine.B>=1.-1e-10;descriptor=None
    if fired:
        engine.B=max(engine.B-1.,0.);engine.a_adv+=event_length;engine.checkpoint_advance_total_m+=event_length;engine.n_adv+=1
        engine.hazard_last_completed_threshold=threshold;engine.hazard_last_completed_action=float(engine.hazard_action_current)
        engine.hazard_threshold_history.append(threshold);event_index=int(engine.hazard_event_index);engine.hazard_event_index+=1
        engine.hazard_action_current=0.;engine.hazard_threshold_action=engine._draw_threshold()
        descriptor={'event_advance_m':event_length,'event_length_factor':float(getattr(engine,'avalanche_event_length_factor',1.)),
            'threshold_action':threshold,'hazard_seed':int(engine.hazard_cfg.seed),'hazard_event_index':event_index,
            'geometry_subsegment_fraction':float(engine.avalanche_cfg.geometry_subsegment_fraction)}
        if hasattr(engine,'avalanche_event_length_history'):engine.avalanche_event_length_history.append(event_length)
        if hasattr(engine,'_set_current_event_length'):engine._set_current_event_length()
    return {'fired':fired,'dt_consumed':h,'dt_unused':max(float(dt)-h,0.),'descriptor':descriptor,
        'physical_hazard_action_step':dH,'dB':dB,'da':da,
        'integration_identity':'PHASE_RESOLVED_CONSTANT_COHESIVE_RATE_STRANG_V1'}

def probe_loads(binding,barrier,kenv,T,R,freq,phases):
    engine=configure_stochastic_engine(binding,kernel_family_path=R4_KERNEL_FAMILY,seed=20261017)
    radius=max(float(engine.r_eff()),1e-30)
    peak_K=barrier.thermal.t_hat(T)*math.sqrt(2*math.pi*radius)/1e6
    grid=np.linspace(max(.05,.35*peak_K),1.25*peak_K,721);rows=[]
    for Kmax in grid:
        w=FatigueWaveform(Kmax=Kmax*1e6,R=R,frequency_Hz=freq);state={'branch':'INTACT'};action=0.;spin=False
        for K in phase_values(w,phases):
            free=min(max(float(K),0.)/math.sqrt(2*math.pi*radius),float(engine.f.sigma_cap))
            land=barrier.rate_from_environment(free/kenv,kenv,T,state)
            rate=cooperative(land['elementary_rate_s-1'],binding.reduced_config.cleavage_hits,binding.reduced_config.cleavage_tau_s)
            state=land['connected_state'];action+=rate*w.period_s/phases
            spin=spin or land['state']=='INTACT_BRANCH_LOST_TO_SEPARATED'
        rows.append({'Kmax_MPa_sqrt_m':Kmax,'action_per_cycle':action,'spinodal_reached':spin})
    targets=(.80,1.50,3.00)
    picked=[]
    for target in targets:
        eligible=[r for r in rows if not r['spinodal_reached'] and r['action_per_cycle']>0]
        row=min(eligible,key=lambda r:abs(math.log10(r['action_per_cycle'])-math.log10(target)))
        if row['Kmax_MPa_sqrt_m'] not in [x['Kmax_MPa_sqrt_m'] for x in picked]:picked.append(row)
    return rows,picked

def run_arm(binding,barrier,mechanics_receipt,Kmax,arm,seed,target_events,target_extension,maximum_cycles,phases,T,R,freq):
    engine=configure_stochastic_engine(binding,kernel_family_path=R4_KERNEL_FAMILY,seed=seed)
    if arm=='OPENING_REFERENCE':
        engine._plastic_half_step=MethodType(lambda self,dt,temperature_K,opening_stress_Pa:{'frozen_dislocation_feedback':1.0},engine)
    waveform=FatigueWaveform(Kmax=Kmax*1e6,R=R,frequency_Hz=freq);values=phase_values(waveform,phases)
    connected={'branch':'INTACT'};events=[];history=[];energy=[];cycle=0.;phase=0;remaining=0.
    kenv=mechanics_receipt['environment_stiffness_Pa_per_m'];dt_phase=waveform.period_s/phases
    while (len(events)<target_events or float(engine.checkpoint_advance_total_m)<target_extension) and cycle<maximum_cycles:
        if remaining<=0.:remaining=dt_phase
        K=float(values[phase]);before_cycle=cycle+phase/phases+(dt_phase-remaining)/waveform.period_s
        land,rate,traction,free,imposed=phase_landscape(engine,barrier,K,T,kenv,connected,binding.reduced_config.cleavage_hits,binding.reduced_config.cleavage_tau_s)
        before_state=dict(land['connected_state']);before_density=(None if land.get('connected') is None else float(land['connected']['energy_J_m2']))
        result=advance_override_phase(engine,max(K,0.),T,remaining,traction,rate)
        descriptor=result['descriptor']
        consumed=float(result['dt_consumed']);remaining=max(remaining-consumed,0.);event_cycle=before_cycle+consumed/waveform.period_s
        if remaining<=max(64*np.spacing(dt_phase),1e-15):
            remaining=0.;phase+=1
            if phase==phases:phase=0;cycle=math.floor(event_cycle+1e-12)+1.
        connected=before_state
        history.append({'cycles':event_cycle,'phase_index':phase,'K_Pa_sqrt_m':K,'free_tip_stress_Pa':free,
            'cohesive_traction_Pa':traction,'separation_m':0. if land.get('connected') is None else land['connected']['delta_m'],
            'landscape_state':land['state'],'barrier_J':land['barrier_J'],'elementary_rate_s-1':land['elementary_rate_s-1'],
            'cooperative_rate_s-1':rate,'mobile_count':float(engine.mpz.mobile_count),'retained_count':float(engine.mpz.retained_count),
            'K_shield_Pa_sqrt_m':float(engine.K_shield()),'tip_radius_m':float(engine.r_eff()),'fired':bool(result.get('fired'))})
        if result.get('fired'):
            if descriptor is None:raise RuntimeError('accepted kinetic event lacks geometry descriptor')
            advance=float(descriptor['event_advance_m']);gamma=barrier.thermal.gamma(T)
            pre=gamma if before_density is None else before_density
            surface=gamma*advance;stored=pre*advance
            ledger={'event_index':len(events)+1,'surface_free_energy_after_J_per_m':surface,
                'cohesive_plus_environment_before_J_per_m':stored,
                'recoverable_release_J_per_m':max(stored-surface,0.),'thermal_activation_exchange_J_per_m':max(surface-stored,0.),
                'activation_barrier_surface_charge_J_per_m':0.,'balance_residual_J_per_m':stored+max(surface-stored,0.)-surface-max(stored-surface,0.),
                'surface_energy_owner':'KINETIC_CZM_BROKEN_PATCH','event_advance_m':advance}
            energy.append(ledger)
            events.append({'event_index':len(events)+1,'cycles':event_cycle,'extension_m':float(engine.checkpoint_advance_total_m),
                'event_advance_m':advance,'phase':event_cycle%1.,'K_Pa_sqrt_m':K,'threshold_action':descriptor['threshold_action'],
                'mobile_count':float(engine.mpz.mobile_count),'retained_count':float(engine.mpz.retained_count),
                'K_shield_Pa_sqrt_m':float(engine.K_shield()),'tip_radius_m':float(engine.r_eff()),
                'unused_phase_time_s':remaining,'remaining_waveform_continued':remaining>0.})
            print(json.dumps({'kinetic_czm_event':len(events),'arm':arm,'Kmax_MPa_sqrt_m':Kmax,
                'cycles':event_cycle,'extension_m':float(engine.checkpoint_advance_total_m),
                'mobile_count':float(engine.mpz.mobile_count)}),flush=True)
            connected={'branch':'INTACT'}
    status='developed_target_completed' if len(events)>=target_events and float(engine.checkpoint_advance_total_m)>=target_extension else 'maximum_cycles_resource_stop'
    return {'schema':'r10.kinetic-czm-coupled-advancing/1','status':status,'arm':arm,'execution_identity':COHESIVE_KINETIC_CZM,
        'candidate_id':binding.optimized_row['candidate_id'],'Kmax_MPa_sqrt_m':Kmax,'DeltaK_MPa_sqrt_m':(1-R)*Kmax,
        'temperature_K':T,'R':R,'frequency_Hz':freq,'phase_count':phases,'seed':seed,'events':events,'history':history,
        'energy_ledger':energy,'final_cycles':history[-1]['cycles'] if history else 0.,'final_extension_m':float(engine.checkpoint_advance_total_m),
        'target_events':target_events,'target_extension_m':target_extension,'mechanics_condensation':mechanics_receipt,
        'dislocation_feedback_frozen':arm=='OPENING_REFERENCE','cohesive_rupture_state_frozen':False,
        'legacy_scalar_opening_attenuation_applied_to_cohesive_traction':False,
        'resolved_cohesive_redistribution_owner':'FINITE_COMPLIANCE_CONNECTED_LANDSCAPE',
        'unresolved_source_geometry_owner':'RETAINED_R10_ENGINE_RADIUS_BACKSTRESS_WAKE'}

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,required=True);p.add_argument('--candidate-id',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--loads');p.add_argument('--target-events',type=int,default=15);p.add_argument('--target-extension-um',type=float,default=70.);p.add_argument('--maximum-cycles',type=float,default=1e6);p.add_argument('--phase-count',type=int,default=8);p.add_argument('--seed',type=int,default=20261017);a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
    binding=load_cohesive_binding(a.registry,a.candidate_id,execution_identity=COHESIVE_KINETIC_CZM);barrier=binding.manifest.cleavage
    Eprime=400e9/(1-.22**2);mech=mechanics(barrier,Eprime)
    probe,picked=probe_loads(binding,barrier,mech['environment_stiffness_Pa_per_m'],300.,.1,1000.,a.phase_count)
    loads=[float(x) for x in a.loads.split(',')] if a.loads else [r['Kmax_MPa_sqrt_m'] for r in picked]
    with (a.out/'load_probe.csv').open('w',newline='') as f:w=csv.DictWriter(f,probe[0].keys(),lineterminator='\n');w.writeheader();w.writerows(probe)
    results=[]
    for K in loads:
        for arm in ('FULL_DISLOCATION_STATE','OPENING_REFERENCE'):
            case=a.out/f'Kmax_{K:.6f}'/arm;case.mkdir(parents=True,exist_ok=True)
            result_path=case/'result.json'
            if result_path.exists():
                prior=json.loads(result_path.read_text())
                reusable=(prior.get('status')=='developed_target_completed' and
                    prior.get('candidate_id')==a.candidate_id and prior.get('arm')==arm and
                    int(prior.get('phase_count',-1))==a.phase_count and
                    len(prior.get('events',[]))>=a.target_events and
                    float(prior.get('final_extension_m',0.))>=a.target_extension_um*1e-6)
                if reusable:
                    results.append(prior)
                    print(json.dumps({'Kmax':K,'arm':arm,'status':'resumed_completed_result',
                        'events':len(prior['events']),'cycles':prior['final_cycles'],
                        'extension_m':prior['final_extension_m']}),flush=True)
                    continue
            result=run_arm(binding,barrier,mech,K,arm,a.seed,a.target_events,a.target_extension_um*1e-6,a.maximum_cycles,a.phase_count,300.,.1,1000.)
            atomic(result_path,result);results.append(result)
            print(json.dumps({'Kmax':K,'arm':arm,'status':result['status'],'events':len(result['events']),'cycles':result['final_cycles'],'extension_m':result['final_extension_m']}),flush=True)
    atomic(a.out/'campaign_summary.json',{'candidate_id':a.candidate_id,'same_complete_material_sha256':binding.provenance['complete_row_sha256'],
        'execution_identity':COHESIVE_KINETIC_CZM,'loads':loads,'mechanics_condensation':mech,
        'results':[{'Kmax_MPa_sqrt_m':r['Kmax_MPa_sqrt_m'],'arm':r['arm'],'status':r['status'],'events':len(r['events']),'cycles':r['final_cycles'],'extension_m':r['final_extension_m']} for r in results]})
    return 0
if __name__=='__main__':raise SystemExit(main())
