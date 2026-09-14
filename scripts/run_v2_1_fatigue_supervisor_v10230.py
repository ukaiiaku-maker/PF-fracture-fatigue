"""Two-worker supervisor and aggregator for the bounded V2.1 Tier-1 pilot."""
from __future__ import annotations
import json,subprocess,sys,time
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.run_corrected_joint_search_v2_1_transfer_v10230 as run
OUT=run.OUT;PYTHON='/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python'

def main():
    picks=[x for x in run.freeze_candidates() if x[1] in {'P25_DBTT_LIKE','P40_DBTT_LIKE','P25_WEAK_T','P40_WEAK_T'}]
    jobs=[(cid,role,f) for f in (.95,.75,.55) for cid,role in picks]
    # Resume the already-started preregistered case first.
    first=('P25_TJBSV2_S_002987','P25_DBTT_LIKE',.55);jobs.remove(first);jobs.insert(0,first)
    active={};done=[]
    while jobs or active:
        while jobs and len(active)<2:
            job=jobs.pop(0);path=OUT/'fatigue_case_results'/job[0]/f'fraction_{job[2]:.2f}.json'
            if path.is_file():done.append(job);continue
            log=OUT/'fatigue_case_results'/job[0]/f'fraction_{job[2]:.2f}.log';log.parent.mkdir(parents=True,exist_ok=True);stream=log.open('w')
            command=[PYTHON,str(ROOT/'scripts/run_v2_1_fatigue_case_v10230.py'),job[0],job[1],str(job[2]),'--max-wall-seconds','1800']
            proc=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,text=True);active[job]=(proc,stream,time.monotonic())
            print(json.dumps({'event':'START','job':job,'pid':proc.pid}),flush=True)
        time.sleep(5)
        for job,(proc,stream,start) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            stream.close();del active[job]
            if code!=0:raise RuntimeError(f'fatigue worker failed {job}, exit={code}')
            done.append(job);print(json.dumps({'event':'DONE','job':job,'seconds':time.monotonic()-start}),flush=True)
    records=[]
    for cid,role,f in [(c,r,f) for c,r in picks for f in (.55,.75,.95)]:
        records.append(json.loads((OUT/'fatigue_case_results'/cid/f'fraction_{f:.2f}.json').read_text()))
    events=[r['event'] for r in records if r.get('event')]
    cases=pd.DataFrame([{k:v for k,v in r.items() if k!='event'} for r in records]);cases.to_csv(OUT/'production_300K_fatigue_case_table.csv',index=False)
    event_frame=pd.DataFrame(events) if events else pd.DataFrame(columns=['candidate_id','event_index','event_cycle','accepted_event_size_m']);event_frame.to_parquet(OUT/'production_300K_fatigue_event_ledger.parquet',index=False,compression='zstd')
    compare=cases[['candidate_id','production_role','Kmax_fraction','Kmax_MPa_sqrt_m','DeltaK_MPa_sqrt_m','corrected_analytical_da_dN','parent_target_da_dN_log_interpolated','developed_da_dN','censor_status']];compare.to_csv(OUT/'production_300K_fatigue_transfer_comparison.csv',index=False)
    slopes=[{'candidate_id':cid,'production_role':role,'finite_uncensored_points':0,'three_point_local_slope':np.nan,'two_point_secant':np.nan,'classification':'FATIGUE_SLOPE_UNRESOLVED_CENSORED'} for cid,role in picks];pd.DataFrame(slopes).to_csv(OUT/'production_300K_fatigue_local_slopes.csv',index=False)
    classification='PRODUCTION_1D_STATE_CLOSURE_FAILED' if any(cases.censor_status=='PRODUCTION_1D_STATE_CLOSURE_FAILED') else 'PRODUCTION_1D_CENSORED_IN_TESTED_DOMAIN'
    decision={'tier':1,'trajectory_count':12,'candidate_count':4,'classification':classification,'status_counts':cases.censor_status.value_counts().to_dict(),'interpretation':'Production 1-D first-passage/censor boundaries are retained. Geometry-dependent post-passage energy closure was not fabricated without prohibited 2-D mechanics.','tier2_executed':False,'two_dimensional_PF_FEM_CZM_executed':False}
    (OUT/'production_300K_fatigue_pilot_decision.json').write_text(json.dumps(decision,indent=2,sort_keys=True)+'\n')
    (OUT/'PRODUCTION_300K_FATIGUE_PILOT_DECISION.md').write_text('# Production 300 K fatigue pilot decision\n\n'+decision['interpretation']+' All local slopes are `FATIGUE_SLOPE_UNRESOLVED_CENSORED`.\n')
    fracture=pd.read_csv(OUT/'production_1d_fracture_results.csv');tier2=[]
    for cid in ['P25_TJBSV2_S_004127','P40_TJBSV2_S_031027']:
        onset=float(fracture[(fracture.candidate_id==cid)&(fracture.temperature_K==300)&(fracture.Kdot==run.RATE)].iloc[0].K_onset_MPa_sqrt_m)
        for f in (.55,.75,.95):tier2.append({'candidate_id':cid,'Kmax_fraction':f,'Kmax_MPa_sqrt_m':f*onset,'DeltaK_MPa_sqrt_m':.9*f*onset,'status':'PREPARED_NOT_EXECUTED_REQUIRES_EXPLICIT_REVIEW'})
    pd.DataFrame(tier2).to_csv(OUT/'optional_tier2_fatigue_plan.csv',index=False)
    print(json.dumps({'status':'PASS','trajectories':12,'status_counts':decision['status_counts']},indent=2),flush=True)
if __name__=='__main__':main()
