#!/usr/bin/env python3
"""Execute the state-informed cyclic/ramp tier for retained R10 V2 materials."""
from __future__ import annotations
import argparse,csv,json,math
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.cohesive_arrhenius import BILINEAR,SMOOTH_EXPONENTIAL,FAMILY,OPENING_ALTERNATIVES_FAMILY
from arrhenius_fracture.coupled_opening_reduction_r10 import (bilinear_accessibility_coordinates,
    coupled_cycle_reduction,coupled_monotonic_ramp)
from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.opening_alternatives_r10 import OpeningAlternative
from arrhenius_fracture.retained_material_cohesive_r10 import load_cohesive_binding,build_cohesive_current_engine
from scripts.run_r10_global_mechanism_search import canonical_hash
from scripts.run_r10_opening_alternatives import critical_K,mechanics,thermal_from
from scripts.run_r10_opening_recurrence import run_case

TEMPERATURES=(300.,450.,600.,750.,900.,1050.,1200.)

def atomic(path,payload):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+'\n');tmp.replace(path)

def write_csv(path,rows):
 fields=[]
 for row in rows:
  for key in row:
   if key not in fields:fields.append(key)
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(rows)

def prepare(a):
 with a.source.open(newline='') as f:rows=list(csv.DictReader(f))
 # Preserve physical-vector diversity from both completed native blocks.
 preferred=('R10F_V2_01617','R10F_V2_03832','R10F_V2_07211','R10F_V2_02913','R10F_V2_03963','R10F_V2_01620')
 base={r['source_material_candidate_id']:r for r in rows}
 out=[]
 for source in preferred:
  original=base[source]
  for shape in (SMOOTH_EXPONENTIAL,BILINEAR):
   for mode in ('cohesive_athermal','cohesive_mixed'):
    r=dict(original);code=('A' if mode=='cohesive_athermal' else 'M')+('S' if shape==SMOOTH_EXPONENTIAL else 'B')
    r['candidate_id']=source+'__R'+code;r['selected_shape']=shape;r['selected_mode']=mode
    r['source_complete_row_sha256']=original['complete_row_sha256']
    r['complete_row_sha256']=canonical_hash({'family':FAMILY,'source_complete_row_sha256':original['complete_row_sha256'],
      'additive_family':OPENING_ALTERNATIVES_FAMILY,'shape':shape,'mode':mode,'reduction_binding':1})
    out.append(r)
 write_csv(a.out,out);print(json.dumps({'paired_rows':len(out),'physical_vectors':len(preferred),'shapes':2,'modes':2},indent=2));return 0

def arm(row,c):
 mode=row['selected_mode'];return OpeningAlternative(mode,row['selected_shape'],
  'quasistatic_mechanical' if mode=='cohesive_athermal' else 'time_dependent_first_passage',
  None if mode=='cohesive_athermal' else float(c['activation_area_m2']),
  None if mode=='cohesive_athermal' else float(c['attempt_frequency_s']))

def reference_load(row,T=300.,phases=8):
 c=json.loads(row['cohesive_parameter_vector']);thermal=thermal_from(c);zone,kenv=mechanics(thermal,T)
 if row['selected_shape']==BILINEAR:
  d=bilinear_accessibility_coordinates(thermal,T,kenv,1e-6,phases)
  return d['discrete_waveform_Kmax_for_branch_loss_MPa_sqrt_m'],d
 continuous=critical_K(thermal,T,SMOOTH_EXPONENTIAL);hmax=.55+.45*math.cos(math.pi/phases)
 return continuous/hmax,{'occupied_bonded_branch_loss_K_MPa_sqrt_m':continuous,
   'discrete_waveform_Kmax_for_branch_loss_MPa_sqrt_m':continuous/hmax,'waveform_midpoint_hmax':hmax,
   'static_completed_branch_K_MPa_sqrt_m':None}

def cycle_task(task):
 registry=Path(task['registry']);row=task['row'];binding=load_cohesive_binding(registry,row['candidate_id'],execution_identity='COHESIVE_KINETIC_CZM')
 engine=build_cohesive_current_engine(binding,kernel_family_path=R4_KERNEL_FAMILY)
 c=json.loads(row['cohesive_parameter_vector']);thermal=thermal_from(c);_,kenv=mechanics(thermal,task['T'])
 result=coupled_cycle_reduction(engine,arm(row,c),thermal,kenv,task['K'],task['T'],task['cycles'],task['phases'],1e-6,120e-6)
 result.update({'candidate_id':row['candidate_id'],'source_material_candidate_id':row['source_material_candidate_id'],
  'shape':row['selected_shape'],'mode':row['selected_mode'],'search_design':row['search_design'],
  'complete_row_sha256':row['complete_row_sha256'],'reference_Kmax_MPa_sqrt_m':task['Kref'],
  'reference_factor':task['K']/task['Kref']});return task,result

def reduced(a):
 with a.registry.open(newline='') as f:rows=list(csv.DictReader(f))
 factors=[float(x) for x in a.load_factors.split(',')];tasks=[]
 for row in rows:
  Kref,_=reference_load(row,300.,a.phases)
  for factor in factors:tasks.append({'registry':str(a.registry),'row':row,'T':300.,'Kref':Kref,'K':factor*Kref,'cycles':a.cycles,'phases':a.phases})
 summaries=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  futures=[pool.submit(cycle_task,t) for t in tasks]
  for future in as_completed(futures):
   task,result=future.result();path=a.out/result['candidate_id']/f"K_{result['Kmax_MPa_sqrt_m']:.8g}.json";atomic(path,result)
   summaries.append({k:result.get(k) for k in ('candidate_id','source_material_candidate_id','shape','mode','search_design','Kmax_MPa_sqrt_m','DeltaK_MPa_sqrt_m','reference_factor','cycles_completed','event_cycle_count','developed_rate_m_per_cycle','outcome','unsupported_crossover')})
   print(json.dumps(summaries[-1]),flush=True)
 write_csv(a.out/'reduced_case_dispositions.csv',summaries);return 0

def ramp_task(task):
 registry=Path(task['registry']);row=task['row'];binding=load_cohesive_binding(registry,row['candidate_id'],execution_identity='COHESIVE_KINETIC_CZM')
 engine=build_cohesive_current_engine(binding,kernel_family_path=R4_KERNEL_FAMILY)
 c=json.loads(row['cohesive_parameter_vector']);thermal=thermal_from(c);_,kenv=mechanics(thermal,task['T'])
 result=coupled_monotonic_ramp(engine,arm(row,c),thermal,kenv,task['T'],task['Kdot'],task['dK'],task['Kmax'])
 result.update({'candidate_id':row['candidate_id'],'source_material_candidate_id':row['source_material_candidate_id'],
  'shape':row['selected_shape'],'mode':row['selected_mode'],'complete_row_sha256':row['complete_row_sha256']});return task,result

def ramps(a):
 with a.registry.open(newline='') as f:allrows=list(csv.DictReader(f))
 wanted=set(a.candidate_ids.split(','));rows=[r for r in allrows if r['candidate_id'] in wanted];tasks=[]
 for row in rows:
  for T in TEMPERATURES:tasks.append({'registry':str(a.registry),'row':row,'T':T,'Kdot':a.Kdot,'dK':a.dK,'Kmax':a.Kmax})
 summaries=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  futures=[pool.submit(ramp_task,t) for t in tasks]
  for future in as_completed(futures):
   task,result=future.result();path=a.out/result['candidate_id']/f"T_{int(task['T']):04d}K.json";atomic(path,result)
   summaries.append({k:result.get(k) for k in ('candidate_id','source_material_candidate_id','shape','mode','temperature_K','status','occupied_branch_loss_K_MPa_sqrt_m','first_committed_separation_K_MPa_sqrt_m','unsupported_crossover')});print(json.dumps(summaries[-1]),flush=True)
 write_csv(a.out/'ramp_dispositions.csv',summaries);return 0

def native_task(task):
 row=task['row'];result=run_case(Path(task['registry']),row['candidate_id'],row['selected_shape'],
  row['selected_mode'],task['K'],300.,task['cycles'],task['plastic_phases'],task['opening_phases'],
  1e-6,task['seed'],120e-6,1)
 result.update({'source_material_candidate_id':row['source_material_candidate_id'],
  'search_design':row['search_design'],'reference_Kmax_MPa_sqrt_m':task['Kref'],
  'reference_factor':task['K']/task['Kref']});return task,result

def native(a):
 with a.registry.open(newline='') as f:allrows=list(csv.DictReader(f))
 wanted=set(a.candidate_ids.split(','));rows=[r for r in allrows if r['candidate_id'] in wanted]
 factors=[float(x) for x in a.load_factors.split(',')];tasks=[]
 for i,row in enumerate(rows):
  Kref,_=reference_load(row,300.,a.opening_phases)
  for j,factor in enumerate(factors):tasks.append({'registry':str(a.registry),'row':row,'Kref':Kref,
   'K':factor*Kref,'cycles':a.cycles,'plastic_phases':a.plastic_phases,
   'opening_phases':a.opening_phases,'seed':a.seed+1000*i+j})
 summaries=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  futures=[pool.submit(native_task,t) for t in tasks]
  for future in as_completed(futures):
   task,result=future.result();path=a.out/result['candidate_id']/f"K_{result['Kmax_MPa_sqrt_m']:.8g}.json";atomic(path,result)
   cycles=result['cycle_history'];summaries.append({'candidate_id':result['candidate_id'],
    'source_material_candidate_id':result['source_material_candidate_id'],'shape':result['shape'],
    'mode':result['mode'],'search_design':result['search_design'],'Kmax_MPa_sqrt_m':result['Kmax_MPa_sqrt_m'],
    'DeltaK_MPa_sqrt_m':result['DeltaK_MPa_sqrt_m'],'reference_factor':result['reference_factor'],
    'cycles_completed':len(cycles),'events':len(result['events']),
    'extension_m':cycles[-1]['extension_m'] if cycles else 0.,'stop_category':result['stop_category'],
    'unsupported_crossover':result['unsupported_crossover_encountered']});print(json.dumps(summaries[-1]),flush=True)
 write_csv(a.out/'native_case_dispositions.csv',summaries);return 0

def main(argv=None):
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='stage',required=True)
 q=sub.add_parser('prepare');q.add_argument('--source',type=Path,required=True);q.add_argument('--out',type=Path,required=True);q.set_defaults(func=prepare)
 q=sub.add_parser('reduced');q.add_argument('--registry',type=Path,required=True);q.add_argument('--out',type=Path,required=True);q.add_argument('--load-factors',default='.72,.84,.94,1.02,1.12,1.28');q.add_argument('--cycles',type=int,default=32);q.add_argument('--phases',type=int,default=8);q.add_argument('--workers',type=int,default=4);q.set_defaults(func=reduced)
 q=sub.add_parser('ramps');q.add_argument('--registry',type=Path,required=True);q.add_argument('--candidate-ids',required=True);q.add_argument('--out',type=Path,required=True);q.add_argument('--Kdot',type=float,default=.005);q.add_argument('--dK',type=float,default=1.);q.add_argument('--Kmax',type=float,default=200.);q.add_argument('--workers',type=int,default=4);q.set_defaults(func=ramps)
 q=sub.add_parser('native');q.add_argument('--registry',type=Path,required=True);q.add_argument('--candidate-ids',required=True);q.add_argument('--out',type=Path,required=True);q.add_argument('--load-factors',default='.84,.96,1.06,1.20');q.add_argument('--cycles',type=int,default=24);q.add_argument('--plastic-phases',type=int,default=4);q.add_argument('--opening-phases',type=int,default=8);q.add_argument('--workers',type=int,default=4);q.add_argument('--seed',type=int,default=20261018);q.set_defaults(func=native)
 a=p.parse_args(argv);return a.func(a)
if __name__=='__main__':raise SystemExit(main())
