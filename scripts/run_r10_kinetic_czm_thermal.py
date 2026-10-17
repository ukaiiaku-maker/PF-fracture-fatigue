#!/usr/bin/env python3
"""Same-material monotonic thermal panel for the finite-compliance kinetic CZM."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from types import MethodType

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.cohesive_arrhenius import COHESIVE_KINETIC_CZM
from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.paris_slope_r9 import configure_stochastic_engine
from arrhenius_fracture.retained_material_cohesive_r10 import load_cohesive_binding
from scripts.run_r10_kinetic_czm_coupled import advance_override_phase,mechanics,phase_landscape


def atomic(path,payload):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');tmp.replace(path)


def run(binding,barrier,mech,T,arm,seed,Kdot,dK,Kmax):
    engine=configure_stochastic_engine(binding,kernel_family_path=R4_KERNEL_FAMILY,seed=seed)
    if arm=='OPENING_REFERENCE':
        engine._plastic_half_step=MethodType(lambda self,dt,temperature_K,opening_stress_Pa:{'frozen_dislocation_feedback':1.0},engine)
    connected={'branch':'INTACT'};time=0.;samples=[];event=None;steps=0
    initial={'mobile_count':float(engine.mpz.mobile_count),'retained_count':float(engine.mpz.retained_count),
             'K_shield_Pa_sqrt_m':float(engine.K_shield()),'tip_radius_m':float(engine.r_eff()),
             'hazard_threshold_action':float(engine.hazard_threshold_action)}
    K=0.05
    while K<=Kmax+1e-12:
        steps+=1;mid=K+0.5*dK;dt=dK/Kdot
        land,rate,traction,free,imposed=phase_landscape(engine,barrier,mid*1e6,T,
            mech['environment_stiffness_Pa_per_m'],connected,binding.reduced_config.cleavage_hits,
            binding.reduced_config.cleavage_tau_s)
        before_B=float(engine.B);before_time=time
        result=advance_override_phase(engine,mid*1e6,T,dt,traction,rate)
        consumed=float(result['dt_consumed']);time+=consumed
        connected=land['connected_state']
        row={'K_mid_MPa_sqrt_m':mid,'time_s':time,'hazard_fraction':float(engine.B),
             'hazard_action_J0':float(engine.hazard_action_current),'landscape_state':land['state'],
             'barrier_eV':land['barrier_eV'],'cooperative_rate_s-1':rate,
             'separation_m':0. if land.get('connected') is None else float(land['connected']['delta_m']),
             'cohesive_traction_Pa':traction,'free_tip_stress_Pa':free,
             'mobile_count':float(engine.mpz.mobile_count),'retained_count':float(engine.mpz.retained_count),
             'K_shield_Pa_sqrt_m':float(engine.K_shield()),'tip_radius_m':float(engine.r_eff())}
        if steps==1 or steps%20==0 or result['fired']:samples.append(row)
        if result['fired']:
            fraction=consumed/dt
            event={'K_event_MPa_sqrt_m':K+fraction*dK,'bracket_low_MPa_sqrt_m':K,
                   'bracket_high_MPa_sqrt_m':K+dK,'root_resolution_MPa_sqrt_m':dK,
                   'time_s':time,'completed_threshold_action':float(engine.hazard_last_completed_threshold),
                   'completed_action':float(engine.hazard_last_completed_action),
                   'landscape_state':land['state'],'terminal_rate_s-1':rate,
                   'terminal_hazard_fraction_before_step':before_B,
                   'terminal_step_start_time_s':before_time,'event_advance_m':result['descriptor']['event_advance_m']}
            break
        K+=dK
    status='KINETIC_EVENT' if event else 'PHYSICAL_K_BOUND_CENSOR'
    return {'schema':'r10.kinetic-czm-thermal-ramp/1','candidate_id':binding.optimized_row['candidate_id'],
        'execution_identity':COHESIVE_KINETIC_CZM,'arm':arm,'temperature_K':T,'status':status,
        'Kdot_MPa_sqrt_m_per_s':Kdot,'dK_MPa_sqrt_m':dK,'K_bound_MPa_sqrt_m':Kmax,
        'initialization':initial,'event':event,'terminal_action_fraction':float(engine.B),
        'terminal_time_s':time,'steps':steps,'samples':samples,
        'mechanics_condensation':mech,'same_complete_material_sha256':binding.provenance['complete_row_sha256'],
        'dislocation_feedback_frozen':arm=='OPENING_REFERENCE',
        'censor_definition':('event threshold localized within declared K bracket' if event else
            'physical imposed K bound reached with terminal action retained; not a numerical failure')}


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,required=True);p.add_argument('--candidate-id',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--Kdot',type=float,default=.005)
    p.add_argument('--dK',type=float,default=.05);p.add_argument('--Kmax',type=float,default=80.);p.add_argument('--seed',type=int,default=20261017)
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
    binding=load_cohesive_binding(a.registry,a.candidate_id,execution_identity=COHESIVE_KINETIC_CZM);barrier=binding.manifest.cleavage
    mech=mechanics(barrier,400e9/(1-.22**2));summaries=[]
    for T in (300.,450.,600.,750.,900.,1050.,1200.):
        for arm in ('FULL_DISLOCATION_STATE','OPENING_REFERENCE'):
            path=a.out/f'T_{int(T):04d}K'/arm/'result.json';path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():result=json.loads(path.read_text())
            else:result=run(binding,barrier,mech,T,arm,a.seed,a.Kdot,a.dK,a.Kmax);atomic(path,result)
            summaries.append({k:result[k] for k in ('temperature_K','arm','status','terminal_time_s','steps')}
                | {'K_event_MPa_sqrt_m':None if result['event'] is None else result['event']['K_event_MPa_sqrt_m'],
                   'terminal_action_fraction':result['terminal_action_fraction']})
            print(json.dumps(summaries[-1],sort_keys=True),flush=True)
    with (a.out/'thermal_panel.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,summaries[0].keys(),lineterminator='\n');w.writeheader();w.writerows(summaries)
    atomic(a.out/'panel_summary.json',{'candidate_id':a.candidate_id,'same_complete_material_sha256':binding.provenance['complete_row_sha256'],
        'mechanics_condensation':mech,'Kdot_MPa_sqrt_m_per_s':a.Kdot,'dK_MPa_sqrt_m':a.dK,
        'cases':summaries,'classification_scope':'FINITE_COMPLIANCE_REDUCED_KINETIC_CZM_THERMAL_PANEL'})
    return 0


if __name__=='__main__':raise SystemExit(main())
