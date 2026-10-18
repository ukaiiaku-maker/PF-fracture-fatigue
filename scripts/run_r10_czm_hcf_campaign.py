#!/usr/bin/env python3
"""Map current-CZM timescales, select logarithmic exposures, and execute them."""
from __future__ import annotations
import argparse,csv,json,math
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

from arrhenius_fracture.cohesive_arrhenius import COHESIVE_KINETIC_CZM
from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.czm_high_cycle_adapter_r10 import (CZMPhaseController,install_czm_owner,opening_snapshot)
from arrhenius_fracture.fatigue_v1 import FatigueWaveform
from arrhenius_fracture.opening_alternatives_r10 import OpeningAlternative
from arrhenius_fracture.persistent_site_poincare_v10230 import one_cycle_map
from arrhenius_fracture.retained_material_cohesive_r10 import load_cohesive_binding,build_cohesive_current_engine
from scripts.run_r10_coupled_opening_search import reference_load
from scripts.run_r10_czm_hcf import run_case
from scripts.run_r10_opening_alternatives import thermal_from,mechanics

SOURCES=("R10F_V2_01617","R10F_V2_02913","R10F_V2_03963","R10F_V2_01620")

def write_csv(path,rows):
 rows=list(rows);fields=[]
 for row in rows:
  for key in row:
   if key not in fields:fields.append(key)
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(rows)

def map_task(task):
 registry=Path(task['registry']);row=task['row'];factor=task['factor'];T=300.;phases=task['phases']
 c=json.loads(row['cohesive_parameter_vector']);thermal=thermal_from(c);_,kenv=mechanics(thermal,T)
 Kref,_=reference_load(row,T,phases);Kmax=factor*Kref
 alt=OpeningAlternative(row['selected_mode'],row['selected_shape'],
  'quasistatic_mechanical' if row['selected_mode']=='cohesive_athermal' else 'time_dependent_first_passage',
  None if row['selected_mode']=='cohesive_athermal' else float(c['activation_area_m2']),
  None if row['selected_mode']=='cohesive_athermal' else float(c['attempt_frequency_s']))
 engine=build_cohesive_current_engine(load_cohesive_binding(registry,row['candidate_id'],execution_identity=COHESIVE_KINETIC_CZM),kernel_family_path=R4_KERNEL_FAMILY)
 install_czm_owner(engine,alt,thermal,kenv,phases,task['seed']);wave=FatigueWaveform(Kmax*1e6,.1,1000.)
 controller=CZMPhaseController(phases);cycle=one_cycle_map(engine,controller,wave,T)
 successor=one_cycle_map(cycle.engine,controller,wave,T)
 change=float(np.linalg.norm(successor.state_end.vector-cycle.state_end.vector)/max(np.linalg.norm(cycle.state_end.vector),1.))
 peak=opening_snapshot(engine,wave.Kmax,T);action=float(cycle.hazard_action_per_cycle)
 return {"candidate_id":row['candidate_id'],"source_material_candidate_id":row['source_material_candidate_id'],
  "search_design":row['search_design'],"shape":row['selected_shape'],"mode":row['selected_mode'],
  "reference_factor":factor,"Kmax_MPa_sqrt_m":Kmax,"DeltaK_MPa_sqrt_m":.9*Kmax,
  "peak_branch_state":peak['branch_state'],"peak_kinetic_validity":peak['kinetic_validity'],
  "peak_barrier_kBT":peak['barrier_kBT'],"opening_action_per_cycle":action,
  "unit_action_cycles":None if action<=0. else 1./action,"plastic_state_relative_change":change,
  "plastic_state_timescale_cycles":None if change<=0. else 1./change,
  "unsupported":engine._r10_czm_unsupported}

def map_stage(a):
 with a.registry.open(newline='') as f:rows=[r for r in csv.DictReader(f) if r['source_material_candidate_id'] in SOURCES]
 factors=[float(x) for x in a.factors.split(',')];tasks=[]
 for i,row in enumerate(rows):
  for j,factor in enumerate(factors):tasks.append({'registry':str(a.registry),'row':row,'factor':factor,'phases':a.phases,'seed':a.seed+100*i+j})
 out=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  futures=[pool.submit(map_task,t) for t in tasks]
  for future in as_completed(futures):out.append(future.result());print(json.dumps(out[-1]),flush=True)
 out.sort(key=lambda r:(r['candidate_id'],r['reference_factor']));write_csv(a.out/'amplitude_timescale_map.csv',out)
 # Assign historical logarithmic exposure targets across physical backgrounds.
 targets={"R10F_V2_01617":1e6,"R10F_V2_02913":1e4,"R10F_V2_03963":1e6,"R10F_V2_01620":1e4}
 selected=[]
 for candidate in sorted({r['candidate_id'] for r in out}):
  arm=[r for r in out if r['candidate_id']==candidate and not r['unsupported'] and r['peak_branch_state']!='SEPARATION_COMPLETED']
  target=targets[arm[0]['source_material_candidate_id']]
  def coordinate(r):
   wait=r['unit_action_cycles'];state=r['plastic_state_timescale_cycles']
   return wait if wait is not None and math.isfinite(float(wait)) else state
  eligible=[r for r in arm if coordinate(r) is not None and coordinate(r)>0]
  chosen=min(eligible,key=lambda r:abs(math.log10(coordinate(r))-math.log10(target))) if eligible else min(arm,key=lambda r:abs(r['reference_factor']-.65))
  selected.append({**chosen,"exposure_target_cycles":target,
   "selection_coordinate_cycles":coordinate(chosen),
   "selection_basis":"CURRENT_CZM_UNIT_ACTION" if chosen['unit_action_cycles'] is not None else
                     "CURRENT_FULL_STATE_PLASTIC_TIMESCALE" if chosen['plastic_state_timescale_cycles'] is not None else
                     "BONDED_LOWER_AMPLITUDE_COVERAGE_FALLBACK"})
 write_csv(a.out/'selected_hcf_cases.csv',selected);return 0

def execute_task(task):
 return run_case(Path(task['registry']),task['candidate_id'],float(task['reference_factor']),
  float(task['exposure_target_cycles']),Path(task['out']),int(task['seed']),int(task['phases']),max_wall_s=float(task['max_wall_s']))

def execute_stage(a):
 with a.selection.open(newline='') as f:rows=list(csv.DictReader(f))
 if a.source_ids:
  requested={x.strip() for x in a.source_ids.split(',') if x.strip()}
  rows=[r for r in rows if r['source_material_candidate_id'] in requested]
 tasks=[]
 for i,row in enumerate(rows):
  if a.horizon_override is not None:row={**row,'exposure_target_cycles':a.horizon_override}
  tasks.append({**row,'registry':str(a.registry),'out':str(a.out/row['candidate_id']/f"f_{float(row['reference_factor']):.4f}"),
                'seed':a.seed+i,'phases':a.phases,'max_wall_s':a.max_wall_s})
 summaries=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  futures=[pool.submit(execute_task,t) for t in tasks]
  for future in as_completed(futures):
   d=future.result();summaries.append({k:d[k] for k in ('candidate_id','source_material_candidate_id','shape','mode','Kmax_MPa_sqrt_m','DeltaK_MPa_sqrt_m','reference_factor','exposure_target_cycles','accepted_cycles','accepted_seconds','physical_event_count','advance_m','first_event_cycle','inter_event_cycles','developed_window','developed_rate_m_per_cycle','disposition','recurrence_outcome','wall_seconds')});print(json.dumps(summaries[-1]),flush=True)
 write_csv(a.out/'hcf_case_summary.csv',summaries);return 0

def main(argv=None):
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='stage',required=True)
 q=sub.add_parser('map');q.add_argument('--registry',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
 q.add_argument('--factors',default='.30,.40,.50,.60,.70,.78,.84,.89,.93');q.add_argument('--phases',type=int,default=16);q.add_argument('--workers',type=int,default=4);q.add_argument('--seed',type=int,default=20261019);q.set_defaults(func=map_stage)
 q=sub.add_parser('execute');q.add_argument('--registry',type=Path,required=True);q.add_argument('--selection',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
 q.add_argument('--phases',type=int,default=16);q.add_argument('--workers',type=int,default=4);q.add_argument('--seed',type=int,default=20261100);q.add_argument('--max-wall-s',type=float,default=900.)
 q.add_argument('--source-ids',help='Optional comma-separated source-material filter for a continuation.')
 q.add_argument('--horizon-override',type=float,help='Optional accepted-cycle target for the filtered continuation.')
 q.set_defaults(func=execute_stage)
 a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True);return a.func(a)
if __name__=='__main__':raise SystemExit(main())
