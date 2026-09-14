"""Run one checkpointed V2.1 Tier-1 300 K production fatigue case."""
from __future__ import annotations
import argparse,json,math,os,signal,sys,time
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.complete_corrected_thermodynamic_joint_search_v10230 as v2
import scripts.run_corrected_joint_search_v2_1_transfer_v10230 as run
from arrhenius_fracture.fatigue_v1 import FatigueWaveform
from arrhenius_fracture.persistent_site_high_cycle_checkpoint_v10230 import restore_checkpoint
from arrhenius_fracture.persistent_site_high_cycle_engine_v10230_v5 import MODEL_ID,integrate_state_coupled_waveform
from arrhenius_fracture.stochastic_avalanche_tip import clear_pending_geometry_events
from scripts.run_thermodynamic_joint_search_v10230 import surface_action

OUT=run.OUT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('candidate_id');ap.add_argument('role');ap.add_argument('fraction',type=float);ap.add_argument('--max-wall-seconds',type=int,default=1800);a=ap.parse_args()
    result_path=OUT/'fatigue_case_results'/a.candidate_id/f'fraction_{a.fraction:.2f}.json'
    if result_path.is_file():print(json.dumps({'status':'SKIP_COMPLETE','path':str(result_path)}));return
    v2.ROWS=v2.source_rows();v2.ACTIVE=v2.active_stresses(v2.ROWS)
    params=pd.read_csv(run.V2_DIR/'candidate_parameter_rows.csv').set_index('candidate_id')
    p=run.indexed_candidate(params,a.candidate_id)
    fracture=pd.read_csv(OUT/'production_1d_fracture_results.csv')
    onset=float(fracture[(fracture.candidate_id==a.candidate_id)&(fracture.temperature_K==300)&(fracture.Kdot==run.RATE)].iloc[0].K_onset_MPa_sqrt_m)
    Kmax=a.fraction*onset;checkpoint=OUT/'fatigue_live_checkpoints'/'exact_wrapper_nphase48'/a.candidate_id/f'fraction_{a.fraction:.2f}'
    checkpoint.mkdir(parents=True,exist_ok=True);os.environ['V10230_HIGH_CYCLE_CHECKPOINT_DIR']=str(checkpoint.resolve());os.environ['V10230_HIGH_CYCLE_CHECKPOINT_MIN_SECONDS']='30'
    os.environ.update({'V10230_PERIODIC_MAX_ITERATIONS':'24','V10230_HIGH_CYCLE_MAX_MODE_OPERATIONS':'256','V10230_FORWARD_MAX_ACCEPTED_SEGMENTS':'4096','V10230_FORWARD_MAX_TRIAL_INTEGRATIONS':'32768','V10230_DMD_CHAIN_MAX_SEGMENTS':'128'})
    clear_pending_geometry_events();eng,controller,cleavage,parent=run.prepare_fatigue_engine(p)
    start=0.;restored=False
    if (checkpoint/'high_cycle_live_checkpoint.json').is_file():
        run.repair_pre_event_fixed_quantum_checkpoint(checkpoint);payload=restore_checkpoint(eng,checkpoint);start=float(payload.get('cycles_from_engine_time') or 0.);restored=True
        eng.f.da=7e-9;eng.avalanche_cfg.mode='fixed';eng.avalanche_base_checkpoint_m=7e-9;eng.avalanche_checkpoint_synchronized=True;eng.avalanche_event_length_factor=1.;eng.avalanche_event_advance_m=7e-9
    waveform=FatigueWaveform(Kmax=Kmax*1e6,R=.1,frequency_Hz=1000.);initial=float(eng.hazard_threshold_action);t0=time.monotonic();res={};failure=''
    def timeout(_signum,_frame):raise TimeoutError(f'watchdog timeout after {a.max_wall_seconds} seconds')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(a.max_wall_seconds)
    try:
        res=integrate_state_coupled_waveform(eng,controller,waveform,300.,max(1e10-start,0.));cycles=start+float(res.get('coupled_hazard_cycles_consumed',0.));fired=bool(res.get('fired',False))
        status='PRODUCTION_1D_STATE_CLOSURE_FAILED' if fired else 'CENSORED_AT_EXISTING_1E10_CYCLE_LIMIT'
        if fired:failure='first passage reached; geometry-dependent post-passage energy transaction requires prohibited 2-D mechanics'
    except Exception as exc:
        cycles=float(eng.t)*1000.;fired=False;status='PRODUCTION_WATCHDOG_TIMEOUT' if isinstance(exc,TimeoutError) else 'PRODUCTION_NUMERICAL_FAILURE';failure=f'{type(exc).__name__}: {exc}'
    finally:signal.alarm(0)
    parent_frame=v2.physical_controls()[p['parent_id']]['frame'];parent_rate=float(np.exp(np.interp(np.log(max(Kmax,1e-12)),np.log(parent_frame.Kmax.to_numpy(float)),np.log(parent_frame.physical_rate.to_numpy(float)))))
    record={'candidate_id':a.candidate_id,'production_role':a.role,'temperature_K':300.,'R':.1,'frequency_Hz':1000.,'hazard_seed':1001721,'K_onset_300K_MPa_sqrt_m':onset,'Kmax_fraction':a.fraction,'Kmax_MPa_sqrt_m':Kmax,'DeltaK_MPa_sqrt_m':.9*Kmax,'cycle_limit':1e10,'cycles_consumed':cycles,'accepted_event_count':0,'projected_extension_m':0.,'developed_da_dN':None,'censor_status':status,'failure_reason':failure,'initial_threshold':initial,'exact_cycle_integrated_opening_hazard':surface_action(cleavage,Kmax,300.),'corrected_analytical_da_dN':7e-9*surface_action(cleavage,Kmax,300.),'parent_target_da_dN_log_interpolated':parent_rate,'production_high_cycle_model_id':MODEL_ID,'no_hidden_fatigue_floor':True,'no_bulk_hazard':True,'fixed_event_quantum_m':7e-9,'target_extension_m':1.5e-8,'maximum_accepted_events':3,'live_checkpoint_directory':str(checkpoint),'checkpoint_restored':restored,'wall_seconds':time.monotonic()-t0,'two_dimensional_mechanics_executed':False,'event':None}
    if fired:
        record['event']={'candidate_id':a.candidate_id,'production_role':a.role,'Kmax_fraction':a.fraction,'Kmax_MPa_sqrt_m':Kmax,'DeltaK_MPa_sqrt_m':.9*Kmax,'event_index':0,'event_cycle':cycles,'first_passage_threshold':float(res.get('hazard_threshold_completed_action',initial)),'physical_hazard_action':float(res.get('hazard_action_completed',initial)),'proposed_event_size_m':float(res.get('stochastic_event_proposed_advance_m',7e-9)),'energy_admissible_event_size_m':None,'accepted_event_size_m':0.,'energy_gate_status':'PRODUCTION_1D_STATE_CLOSURE_FAILED',**v2.snapshot_summary(eng)}
        eng.restore_geometry_veto()
    result_path.parent.mkdir(parents=True,exist_ok=True);tmp=result_path.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n');os.replace(tmp,result_path);print(json.dumps({'status':status,'candidate_id':a.candidate_id,'fraction':a.fraction,'cycles':cycles,'path':str(result_path)},indent=2))
if __name__=='__main__':main()
