#!/usr/bin/env python3
"""Execute long-exposure HCF/VHCF cases with the current CZM opening owner."""
from __future__ import annotations
import argparse,copy,csv,json,math,os,time
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.cohesive_arrhenius import COHESIVE_KINETIC_CZM
from arrhenius_fracture.coupled_opening_reduction_r10 import score_recurrence
from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.czm_high_cycle_adapter_r10 import (CZMPhaseController,
    commit_pending_event,install_czm_owner,opening_snapshot,qualify_accelerated_overlap)
from arrhenius_fracture.fatigue_v1 import FatigueWaveform
from arrhenius_fracture.developed_fatigue_r3 import _restore_geometry
from arrhenius_fracture.opening_alternatives_r10 import OpeningAlternative
from arrhenius_fracture.persistent_site_high_cycle_engine_v10230 import integrate_state_coupled_waveform
from arrhenius_fracture.persistent_site_high_cycle_checkpoint_v10230 import restore_checkpoint,write_checkpoint
from arrhenius_fracture.persistent_site_high_cycle_state_v10230 import serialize_active_state
from arrhenius_fracture.persistent_site_poincare_v10230 import one_cycle_map
from arrhenius_fracture.retained_material_cohesive_r10 import load_cohesive_binding,build_cohesive_current_engine
from scripts.run_r10_coupled_opening_search import reference_load
from scripts.run_r10_opening_alternatives import thermal_from,mechanics

SCHEMA="r10.czm-owned-hcf-trajectory/1"

def unresolved_post_escape(history,events):
 return (not events and any(row.get('branch_state')=='POST_ESCAPE_TRANSIT'
                            for row in history))

def atomic(path,payload):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+'\n');tmp.replace(path)

def row_for(path,candidate):
 with path.open(newline='') as f:return next(r for r in csv.DictReader(f) if r['candidate_id']==candidate)

def phase_values(wave,phases):
 x=(np.arange(phases,dtype=float)+.5)*2*math.pi/phases
 return np.asarray(wave.K_phase(x),float)

def state_row(engine,wave,T,cycles,kind):
 snap=serialize_active_state(engine,waveform=wave,temperature_K=T)
 return {"kind":kind,"accepted_cycles":float(cycles),"accepted_seconds":float(cycles/wave.frequency_Hz),
  "extension_m":float(engine.checkpoint_advance_total_m),"active_state_norm":float(np.linalg.norm(snap.vector)),
  **opening_snapshot(engine,wave.Kmax,T)}

def save_checkpoint(directory,engine,wave,T,progress):
 previous=os.environ.get('V10230_HIGH_CYCLE_CHECKPOINT_DIR')
 os.environ['V10230_HIGH_CYCLE_CHECKPOINT_DIR']=str(directory)
 try:write_checkpoint(engine,waveform=wave,temperature_K=T,reason='r10_czm_hcf_block',metadata=progress)
 finally:
  if previous is None:os.environ.pop('V10230_HIGH_CYCLE_CHECKPOINT_DIR',None)
  else:os.environ['V10230_HIGH_CYCLE_CHECKPOINT_DIR']=previous

def exact_to_next_cycle(engine,wave,T,phases,cycles,events,patch_m,path_limit_m):
 values=phase_values(wave,phases);tol=1e-12
 target=math.floor(cycles+tol)+1.
 while cycles<target-tol and engine.checkpoint_advance_total_m<path_limit_m-1e-15:
  phase_pos=(cycles-math.floor(cycles))*phases;index=min(int(math.floor(phase_pos)),phases-1)
  phase_end=(index+1)/phases;duration=(phase_end-(cycles-math.floor(cycles)))*wave.period_s
  result=engine._integrate_coupled(float(values[index]),T,duration)
  consumed=float(result["dt_consumed"]);cycles+=consumed*wave.frequency_Hz
  if result["fired"]:
   event=commit_pending_event(engine,patch_m);event.update({"cycle":cycles,"phase":cycles%1.,
    "K_Pa_sqrt_m":float(values[index]),"event_index":len(events)+1})
   events.append(event)
   drain_mechanical_successors(engine,float(values[index]),T,cycles,events,patch_m,path_limit_m)
  elif consumed<=0.:raise RuntimeError("exact waveform remainder made no progress")
 return cycles

def drain_mechanical_successors(engine,K,T,cycles,events,patch_m,path_limit_m):
 """Commit remapped cells that are already complete at the same accepted phase."""
 while engine.checkpoint_advance_total_m<path_limit_m-1e-15:
  snap=opening_snapshot(engine,K,T)
  if snap["branch_state"]!="SEPARATION_COMPLETED":break
  opening=engine._r10_czm_last_opening
  engine._r10_czm_pending_event={"threshold_action":None,
   "trigger":"MECHANICAL_SUCCESSOR_COMPLETION","pre_opening":opening,
   "accepted_duration_s":0.,"unused_duration_s":0.,"stochastic_threshold":False}
  event=commit_pending_event(engine,patch_m);event.update({"cycle":cycles,"phase":cycles%1.,
   "K_Pa_sqrt_m":K,"event_index":len(events)+1});events.append(event)

def run_case(registry,candidate,factor,horizon,out,seed=20261019,phases=16,T=300.,R=.1,
             frequency=1000.,patch_m=1e-6,path_limit_m=120e-6,max_wall_s=900.,
             plastic_feedback=True):
 row=row_for(registry,candidate);c=json.loads(row['cohesive_parameter_vector'])
 thermal=thermal_from(c);_,kenv=mechanics(thermal,T);Kref,_=reference_load(row,T,phases);Kmax=factor*Kref
 alt=OpeningAlternative(row['selected_mode'],row['selected_shape'],
  'quasistatic_mechanical' if row['selected_mode']=='cohesive_athermal' else 'time_dependent_first_passage',
  None if row['selected_mode']=='cohesive_athermal' else float(c['activation_area_m2']),
  None if row['selected_mode']=='cohesive_athermal' else float(c['attempt_frequency_s']))
 binding=load_cohesive_binding(registry,candidate,execution_identity=COHESIVE_KINETIC_CZM)
 engine=build_cohesive_current_engine(binding,kernel_family_path=R4_KERNEL_FAMILY)
 install_czm_owner(engine,alt,thermal,kenv,phases,seed,frequency_Hz=frequency,
                   plastic_feedback=plastic_feedback)
 wave=FatigueWaveform(Kmax*1e6,R,frequency);controller=CZMPhaseController(phases)
 predictor_cycle=one_cycle_map(engine,controller,wave,T)
 predictor_successor=one_cycle_map(predictor_cycle.engine,controller,wave,T)
 initial_change=float(np.linalg.norm(predictor_successor.state_end.vector-predictor_cycle.state_end.vector)/
                     max(np.linalg.norm(predictor_cycle.state_end.vector),1.))
 action=float(predictor_cycle.hazard_action_per_cycle);threshold=float(engine.hazard_threshold_action)
 predictor={"initial_action_per_cycle":action,
  "initial_first_event_cycles":None if action<=0. else threshold/action,
  "initial_state_relative_change_per_cycle":initial_change,
  "state_timescale_cycles":None if initial_change<=0. else 1./initial_change,
  "selector":"HISTORICAL_LOG_EXPOSURE_TARGETS_WITH_CURRENT_CZM_ACTION_AND_PLASTIC_STATE_TIMESCALE"}
 overlap=qualify_accelerated_overlap(engine,controller,wave,T)
 progress_path=out/'progress.json';checkpoint_dir=out/'checkpoint'
 if progress_path.is_file() and (checkpoint_dir/'high_cycle_live_checkpoint.json').is_file():
  saved=json.loads(progress_path.read_text())
  same_protocol=(math.isclose(float(saved.get('temperature_K',T)),T,rel_tol=0.,abs_tol=1e-12) and
                 math.isclose(float(saved.get('R',R)),R,rel_tol=0.,abs_tol=1e-14) and
                 math.isclose(float(saved.get('frequency_Hz',frequency)),frequency,rel_tol=0.,abs_tol=1e-12) and
                 bool(saved.get('plastic_feedback',True))==bool(plastic_feedback))
  if (saved['candidate_id']!=candidate or
      not math.isclose(float(saved['factor']),factor,rel_tol=0.,abs_tol=1e-14) or not same_protocol):
   raise RuntimeError('restart identity differs from requested CZM HCF case')
  _restore_geometry(engine,checkpoint_dir);restore_checkpoint(engine,checkpoint_dir)
  cycles=float(saved['accepted_cycles']);events=list(saved['events']);history=list(saved['history']);blocks=list(saved['blocks'])
  prior_cpu_seconds=float(saved.get('cpu_seconds',0.))
  engine._r10_czm_unsupported=bool(saved.get('unsupported',False))
  engine._r10_czm_unsupported_validities=set(saved.get('unsupported_validities',[]))
 else:
  cycles=0.;events=[];history=[state_row(engine,wave,T,0.,"INITIAL")];blocks=[]
  prior_cpu_seconds=0.
 start=time.monotonic();cpu_start=time.process_time()
 targets=sorted(set([x for x in (1e2,1e4,1e6,1e8,1e10,1e12,float(horizon)) if x<=horizon]))
 resource=False;numeric=None
 post_escape_stop=unresolved_post_escape(history,events)
 try:
  while (not post_escape_stop and cycles<horizon-1e-9 and
         engine.checkpoint_advance_total_m<path_limit_m-1e-15):
   if time.monotonic()-start>max_wall_s:resource=True;break
   if cycles<1.-1e-12 or cycles-math.floor(cycles)>1e-12:
    cycles=exact_to_next_cycle(engine,wave,T,phases,cycles,events,patch_m,path_limit_m);continue
   target=next((x for x in targets if x>cycles+1e-9),float(horizon));requested=min(target-cycles,horizon-cycles)
   before=copy.deepcopy(engine);before_cycle=cycles
   result=integrate_state_coupled_waveform(engine,controller,wave,T,requested)
   consumed=float(result.get('coupled_hazard_cycles_consumed',0.))
   if consumed<=0.:
    engine=before;resource=True;break
   cycles+=consumed
   blocks.append({"start_cycle":before_cycle,"requested_cycles":requested,"accepted_cycles":consumed,
    "accepted_seconds":consumed/frequency,"fired":bool(result.get('fired')),
    "work_budget_exhausted":bool(result.get('coupled_hazard_work_budget_exhausted')),
    "wall_seconds":float(result.get('coupled_hazard_wall_seconds',0.)),
    "modes":result.get('coupled_hazard_modes',[]),"partial_return":bool(result.get('coupled_hazard_partial_return'))})
   if result.get('fired'):
    event=commit_pending_event(engine,patch_m);event.update({"cycle":cycles,"phase":cycles%1.,
      "K_Pa_sqrt_m":float(phase_values(wave,phases)[min(int((cycles%1.)*phases),phases-1)]),
      "event_index":len(events)+1});events.append(event)
    drain_mechanical_successors(engine,event["K_Pa_sqrt_m"],T,cycles,events,patch_m,path_limit_m)
   history.append(state_row(engine,wave,T,cycles,"EVENT" if result.get('fired') else "EXPOSURE_CHECKPOINT"))
   progress={"candidate_id":candidate,"factor":factor,"temperature_K":T,"R":R,
             "frequency_Hz":frequency,"horizon":horizon,"accepted_cycles":cycles,
             "cpu_seconds":prior_cpu_seconds+time.process_time()-cpu_start,
             "plastic_feedback":bool(plastic_feedback),
             "events":events,"history":history,"blocks":blocks,"unsupported":bool(engine._r10_czm_unsupported),
             "unsupported_validities":sorted(engine._r10_czm_unsupported_validities)}
   atomic(progress_path,progress);save_checkpoint(checkpoint_dir,engine,wave,T,progress)
   if result.get('coupled_hazard_work_budget_exhausted') and consumed<requested-1e-9:
    resource=True;break
 except Exception as exc:numeric=f'{type(exc).__name__}: {exc}'
 scored=score_recurrence(events,path_limit_m,bool(engine._r10_czm_unsupported))
 post_escape_unresolved=unresolved_post_escape(history,events)
 event_cycles=[float(e['cycle']) for e in events]
 waits=([event_cycles[0]] if event_cycles else [])+[event_cycles[i]-event_cycles[i-1] for i in range(1,len(event_cycles))]
 if numeric:disposition="NUMERICAL_STOP"
 elif resource:disposition="RESOURCE_STOP"
 elif engine._r10_czm_unsupported or post_escape_unresolved:disposition="UNRESOLVED_OPENING_PHYSICS"
 elif scored['developed_rate_m_per_cycle'] is not None:disposition="DEVELOPED_CYCLIC_GROWTH"
 elif scored['observation_target_hit']:disposition="TARGET_TRUNCATED_CASCADE"
 elif events:disposition="FINITE_BURST_WITH_EXPOSURE"
 elif cycles>=horizon-1e-8:disposition="PHYSICAL_RUNOUT_TO_DECLARED_HORIZON"
 else:disposition="INSUFFICIENT_EXPOSURE"
 result={"schema":SCHEMA,"candidate_id":candidate,"source_material_candidate_id":row['source_material_candidate_id'],
  "shape":row['selected_shape'],"mode":row['selected_mode'],"search_design":row['search_design'],
  "Kmax_MPa_sqrt_m":Kmax,"Kmin_MPa_sqrt_m":R*Kmax,"DeltaK_MPa_sqrt_m":(1-R)*Kmax,
  "reference_factor":factor,"temperature_K":T,"R":R,"frequency_Hz":frequency,"control":"FIXED_REMOTE_K",
  "exposure_target_cycles":horizon,"accepted_cycles":cycles,"accepted_seconds":cycles/frequency,
  "events":events,"physical_event_count":len(events),"advance_m":float(engine.checkpoint_advance_total_m),
  "first_event_cycle":event_cycles[0] if event_cycles else None,"inter_event_cycles":waits[1:],
  "all_waiting_intervals_cycles":waits,"developed_window":scored['developed_window'],
  "developed_rate_m_per_cycle":scored['developed_rate_m_per_cycle'],"disposition":disposition,
  "recurrence_outcome":"UNRESOLVED_POST_ESCAPE_TRANSIT" if post_escape_unresolved else scored['outcome'],
  "unsupported_validities":sorted(engine._r10_czm_unsupported_validities |
    ({'POST_ESCAPE_TRANSIT_PROPAGATION_REQUIRED'} if post_escape_unresolved else set())),
  "numeric_failure":numeric,"resource_stop":resource,"predictor":predictor,"overlap":overlap,
  "history":history,"acceleration_blocks":blocks,"phase_count":phases,"patch_m":patch_m,
  "observation_path_m":path_limit_m,"legacy_cooperative_clock_used":False,
  "opening_owner":"CURRENT_ATHERMAL_OR_MIXED_CZM","plasticity_owner":"CURRENT_EXP_FLOOR_FULL_STATE",
  "plastic_feedback_reference":"EVOLVING_FULL_STATE" if plastic_feedback else
                               "FROZEN_INITIAL_PLASTIC_STATE",
  "restart_checkpoint":str(checkpoint_dir/'high_cycle_live_checkpoint.json'),
  "cpu_seconds":prior_cpu_seconds+time.process_time()-cpu_start,
  "wall_seconds":time.monotonic()-start}
 atomic(out/'result.json',result);return result

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,required=True);p.add_argument('--candidate-id',required=True)
 p.add_argument('--factor',type=float,required=True);p.add_argument('--horizon',type=float,required=True)
 p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=20261019)
 p.add_argument('--phases',type=int,default=16);p.add_argument('--max-wall-s',type=float,default=900.)
 p.add_argument('--temperature-K',type=float,default=300.);p.add_argument('--R',type=float,default=.1)
 p.add_argument('--frequency-Hz',type=float,default=1000.);p.add_argument('--patch-um',type=float,default=1.)
 p.add_argument('--frozen-initial-plastic-state',action='store_true')
 p.add_argument('--path-limit-um',type=float,default=120.)
 a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
 result=run_case(a.registry,a.candidate_id,a.factor,a.horizon,a.out,a.seed,a.phases,
  T=a.temperature_K,R=a.R,frequency=a.frequency_Hz,patch_m=a.patch_um*1e-6,
  path_limit_m=a.path_limit_um*1e-6,max_wall_s=a.max_wall_s,
  plastic_feedback=not a.frozen_initial_plastic_state)
 print(json.dumps({k:result[k] for k in ('candidate_id','Kmax_MPa_sqrt_m','accepted_cycles','accepted_seconds','physical_event_count','advance_m','disposition','developed_rate_m_per_cycle','wall_seconds')},indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
