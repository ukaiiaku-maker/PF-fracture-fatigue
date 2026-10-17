#!/usr/bin/env python3
"""Execute a mechanism-balanced state-aware cohesive acquisition block."""
from __future__ import annotations

import argparse,csv,hashlib,json,math
from pathlib import Path
import sys
import numpy as np
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.fatigue_v1 import FatigueWaveform
from arrhenius_fracture.paris_slope_r9 import _NativePhaseController
from arrhenius_fracture.persistent_site_poincare_v10230 import one_cycle_map
from arrhenius_fracture.retained_material_cohesive_r10 import load_cohesive_binding,build_cohesive_current_engine
from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from scripts.run_r10_cohesive_arrhenius_search import generate,SEED
from scripts.run_r10_global_mechanism_search import read_sources,canonical_hash,signed

LOADS=(4.0,6.0,9.0,13.5,20.25)

def write_csv(path,rows):
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(rows)

def mutate_dislocations(parent,u):
    row=dict(parent)
    row['emit_G00_eV']=float(row['emit_G00_eV'])*math.exp(.50*signed(u[0]))
    shared=math.exp(.42*signed(u[1]));row['emit_sigc0_GPa']=float(row['emit_sigc0_GPa'])*shared
    row['emit_exp_a']=float(row['emit_exp_a'])*math.exp(.75*signed(u[2]))
    row['emit_exp_n']=float(row['emit_exp_n'])*math.exp(.45*signed(u[3]))
    hscale=math.exp(.48*signed(u[4]))
    row['peierls_H0_eV']=float(row['peierls_H0_eV'])*hscale
    row['taylor_H0_eV']=float(row['taylor_H0_eV'])*hscale*math.exp(.20*signed(u[5]))
    entropy=22.*signed(u[6])
    row['peierls_activation_entropy_kB']=float(row['peierls_activation_entropy_kB'])+entropy
    row['taylor_activation_entropy_kB']=float(row['taylor_activation_entropy_kB'])+entropy
    row['peierls_exp_a']=float(row['peierls_exp_a'])*math.exp(.55*signed(u[7]))
    row['taylor_exp_a']=float(row['taylor_exp_a'])*math.exp(.55*signed(u[8]))
    row['peierls_exp_n']=float(row['peierls_exp_n'])*math.exp(.32*signed(u[9]))
    row['taylor_exp_n']=float(row['taylor_exp_n'])*math.exp(.32*signed(u[10]))
    row['cohesive_r10_dislocation_coordinate_status']='AUTHORIZED_COUPLED_RELEASE'
    return row

def valid_transport(row):
    from arrhenius_fracture.material_manifest import KB_EV_PER_K
    for T in (300.,450.,600.,750.,900.,1050.,1200.):
        for p in ('peierls','taylor'):
            H=float(row[f'{p}_H0_eV'])-float(row[f'{p}_activation_entropy_kB'])*KB_EV_PER_K*(T-300.)
            if H<=.05:return False
    return True

def registry_row(item,cid,parent,kind,base):
    c=item['cohesive'];ph=canonical_hash(parent);ch=canonical_hash(c)
    return {'candidate_id':cid,'family':'D_COHESIVE_ARRHENIUS_EXP_PLASTICITY',
        'complete_row_sha256':canonical_hash({'family':'D_COHESIVE_ARRHENIUS_EXP_PLASTICITY','candidate_id':cid,'parent_sha256':ph,'cohesive_sha256':ch}),
        'parent_candidate_id':item['parent_candidate_id'],'parent_complete_row_sha256':ph,
        'cohesive_parameter_sha256':ch,'cleavage_hits':3.0,'cleavage_tau_s':1e-6,
        'mechanism_acquisition_stratum':base['mechanism_acquisition_stratum'],
        'thermal_acquisition_stratum':base['thermal_acquisition_stratum'],
        'state_acquisition_kind':kind,
        'cohesive_parameter_vector':json.dumps(c,sort_keys=True,separators=(',',':')),
        'parent_complete_parameter_vector':json.dumps(parent,sort_keys=True,separators=(',',':'))}

def select_controls(allocation,count_per_stratum):
    chosen=[]
    for mechanism in ('OPENING_DIRECTED','MIXED_DIRECTED','PLASTICITY_DIRECTED'):
        pool=[r for r in allocation if r['mechanism_acquisition_stratum']==mechanism]
        thermal={}
        for row in pool:thermal.setdefault(row['thermal_acquisition_stratum'],[]).append(row)
        for rows in thermal.values():rows.sort(key=lambda r:(abs(float(r['A0_slope_300K_median'])-4.),r['candidate_id']))
        labels=sorted(thermal);cursor={label:0 for label in labels}
        while len([r for r in chosen if r['mechanism_acquisition_stratum']==mechanism])<count_per_stratum:
            advanced=False
            for label in labels:
                index=cursor[label]
                if index<len(thermal[label]):
                    chosen.append(thermal[label][index]);cursor[label]+=1;advanced=True
                    if len([r for r in chosen if r['mechanism_acquisition_stratum']==mechanism])>=count_per_stratum:break
            if not advanced:break
    return chosen

def evaluate(registry,candidate,cycles,phase_count):
    binding=load_cohesive_binding(registry,candidate)
    rows=[]
    for K in LOADS:
        engine=build_cohesive_current_engine(binding,kernel_family_path=R4_KERNEL_FAMILY)
        waveform=FatigueWaveform(Kmax=K*1e6,R=.1,frequency_Hz=1000.)
        controller=_NativePhaseController(phase_count)
        actions=[];mobile=[];retained=[];shield=[]
        for _ in range(cycles):
            result=one_cycle_map(engine,controller,waveform,300.,commit=True)
            engine=result.engine;actions.append(result.hazard_action_per_cycle)
            mobile.append(float(engine.mpz.mobile_count));retained.append(float(engine.mpz.retained_count));shield.append(float(engine.K_shield()))
        a0=actions[0];developed=float(np.mean(actions[-min(4,len(actions)):]))
        rows.append({'candidate_id':candidate,'Kmax_MPa_sqrt_m':K,'DeltaK_MPa_sqrt_m':.9*K,
            'cycles_evolved':cycles,'A0_action_per_cycle':a0,'developed_action_per_cycle':developed,
            'A0_rate_m_per_cycle':5e-6*a0,'developed_rate_m_per_cycle':5e-6*developed,
            'rate_factor_full_over_A0':developed/max(a0,1e-300),
            'hazard_drift_last_over_first':actions[-1]/max(actions[0],1e-300),
            'mobile_count_final':mobile[-1],'retained_count_final':retained[-1],
            'K_shield_final_Pa_sqrt_m':shield[-1],
            'action_history_json':json.dumps(actions,separators=(',',':'))})
    full=np.array([r['developed_rate_m_per_cycle'] for r in rows]);a0=np.array([r['A0_rate_m_per_cycle'] for r in rows]);dk=.9*np.array(LOADS)
    fs=np.diff(np.log(np.maximum(full,1e-300)))/np.diff(np.log(dk));zs=np.diff(np.log(np.maximum(a0,1e-300)))/np.diff(np.log(dk))
    contrast=float(np.max(np.abs(fs-zs)));factor=float(np.max([r['rate_factor_full_over_A0'] for r in rows]))
    mechanism=('STATE_INFORMATIVE' if contrast>=.5 or factor>=1.5 or factor<=2/3 else 'OPENING_INSENSITIVE_SCREEN')
    summary={'candidate_id':candidate,'state_aware_loads':len(rows),'cycles_per_load':cycles,
        'A0_adjacent_slopes':zs.tolist(),'state_aware_adjacent_slopes':fs.tolist(),
        'maximum_absolute_slope_change':contrast,'maximum_rate_factor':factor,
        'all_adjacent_primary_target':bool(np.all((fs>=3.4)&(fs<=4.6)) and math.log10(full[-1]/full[0])>=2),
        'mechanism_screen_disposition':mechanism}
    return rows,summary

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--allocation',type=Path,required=True);p.add_argument('--original-registry',type=Path,required=True);p.add_argument('--parent-registry',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--count-per-stratum',type=int,default=3);p.add_argument('--cycles',type=int,default=12);p.add_argument('--phase-count',type=int,default=32);p.add_argument('--exclude-registry',type=Path);a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
    with a.allocation.open(newline='') as f:allocation=list(csv.DictReader(f))
    if a.exclude_registry and a.exclude_registry.exists():
        with a.exclude_registry.open(newline='') as f:
            excluded={r['candidate_id'].removesuffix('_J') for r in csv.DictReader(f)}
        allocation=[r for r in allocation if r['candidate_id'] not in excluded]
    selected=select_controls(allocation,a.count_per_stratum);sources=read_sources(a.original_registry,a.parent_registry)
    points=qmc.Sobol(d=5,scramble=True,seed=SEED).random_base2(15)
    joint=qmc.Sobol(d=11,scramble=True,seed=20261017).random_base2(
        int(math.ceil(math.log2(max(len(selected),1)))))[:len(selected)]
    registry=[]
    for j,base in enumerate(selected):
        index=int(base['candidate_id'].rsplit('_',1)[1]);item=generate(index,points[index],sources)
        registry.append(registry_row(item,item['candidate_id'],item['parent'],'FROZEN_PARENT_CONTROL',base))
        mutated=mutate_dislocations(item['parent'],joint[j])
        if valid_transport(mutated):registry.append(registry_row(item,item['candidate_id']+'_J',mutated,'JOINT_RELEASED_DISLOCATION',base))
    registry_path=a.out/'state_aware_registry.csv';write_csv(registry_path,registry)
    details=[];summaries=[]
    for i,row in enumerate(registry,1):
        d,s=evaluate(registry_path,row['candidate_id'],a.cycles,a.phase_count);details.extend(d);summaries.append({**s,'state_acquisition_kind':row['state_acquisition_kind'],'complete_row_sha256':row['complete_row_sha256']})
        print(json.dumps({'state_aware_completed':i,'total':len(registry),'candidate_id':row['candidate_id'],'disposition':s['mechanism_screen_disposition']}),flush=True)
    write_csv(a.out/'state_aware_evaluations.csv',details);write_csv(a.out/'state_aware_dispositions.csv',summaries)
    informative=[s for s in summaries if s['mechanism_screen_disposition']=='STATE_INFORMATIVE']
    payload={'schema':'r10.cohesive-state-aware-acquisition/1','allocated_source_rows':len(allocation),
        'executed_complete_materials':len(registry),'executed_load_conditions':len(details),
        'cycles_per_load':a.cycles,'phase_count':a.phase_count,'informative_count':len(informative),
        'joint_released_count':sum(r['state_acquisition_kind']=='JOINT_RELEASED_DISLOCATION' for r in registry),
        'frozen_control_count':sum(r['state_acquisition_kind']=='FROZEN_PARENT_CONTROL' for r in registry),
        'qualified_primary_slope_count':sum(s['all_adjacent_primary_target'] for s in summaries),
        'registry_sha256':hashlib.sha256(registry_path.read_bytes()).hexdigest()}
    (a.out/'state_aware_summary.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps(payload,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
