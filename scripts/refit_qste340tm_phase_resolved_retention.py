#!/usr/bin/env python3
"""Joint refits using birth-phase-resolved retention timing."""
from __future__ import annotations
import argparse,csv,json,math,time
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.refit_qste340tm_capacity_policy import DATA,OUT,POLICIES,LOWER,UPPER,build_records,evaluate,metrics,pack,unpack

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--max-nfev',type=int,default=150);ap.add_argument('--policy',choices=[*POLICIES,'all'],default='all');args=ap.parse_args();records=build_records(DATA)
 rv=np.array([float(r['R']) for r in records]);obs=np.log10(np.array([float(r['observed_dadn_m_per_cycle']) for r in records]));weights=np.empty(len(records))
 for r in sorted(set(rv)):weights[rv==r]=1/math.sqrt(np.sum(rv==r))
 control=json.load(open(DATA/'model_conditional/stateful_exp_floor/fit.json'));historical=pack(control['fit']['parameters'])
 wanted=list(POLICIES) if args.policy=='all' else [args.policy]
 for policy in wanted:
  old=json.load(open(OUT/f'fit_{policy}.json'));warm=np.array(old['parameters_vector'])
  def residual(x):
   s=evaluate(x,records,policy,1,retention_timing='birth_phase_resolved');e=(np.log10(s['prediction'])-obs)*weights
   return np.r_[e,max(0.,.25-float(np.min(s['raw_opening_barrier_over_kBT'])))*10]
  runs=[]
  with ThreadPool(4) as pool:
   start_vectors=[('capacity_matched_warm',warm),('historical_control',historical)]
   phase_10k=OUT/'fit_fixed_10000_phase_resolved.json'
   if policy=='uncapped' and phase_10k.exists():
    start_vectors=[('corrected_fixed_10000_transfer',np.array(json.loads(phase_10k.read_text())['parameters_vector'])),('capacity_matched_warm',warm)]
   for name,x0 in start_vectors:
    print(f'[{policy}/phase] {name}',flush=True);t=time.time();sol=least_squares(residual,x0,bounds=(LOWER,UPPER),max_nfev=args.max_nfev,x_scale='jac',ftol=1e-10,xtol=1e-10,gtol=1e-10,workers=pool.map)
    st=evaluate(sol.x,records,policy,1,retention_timing='birth_phase_resolved');m=metrics(records,st);runs.append((m['overall_rmse_log10_rate'],name,sol,st,m,time.time()-t));print(f'  RMSE={m["overall_rmse_log10_rate"]:.6f} nfev={sol.nfev} success={sol.success}',flush=True)
   _,name,sol,st,m,_=min(runs,key=lambda z:z[0]);print(f'[{policy}/phase] refine {name}',flush=True)
   sol=least_squares(residual,sol.x,bounds=(LOWER,UPPER),max_nfev=args.max_nfev,x_scale='jac',ftol=2e-12,xtol=2e-12,gtol=2e-12,workers=pool.map)
  st=evaluate(sol.x,records,policy,1,retention_timing='birth_phase_resolved');fr=evaluate(sol.x,records,policy,1,evolve=False,retention_timing='birth_phase_resolved')
  payload={'schema':'qste340tm-birth-phase-retention-joint-refit/1','policy':policy,'fixed_capacity':POLICIES[policy],'retention_timing':'birth_phase_resolved','parameters_vector':sol.x.tolist(),'parameters':unpack(sol.x),'metrics':metrics(records,st),'frozen_metrics':metrics(records,fr),
           'optimizer':{'success':bool(sol.success),'message':str(sol.message),'nfev':int(sol.nfev),'cost':float(sol.cost),'optimality':float(sol.optimality)},
           'multi_start':[{'start':n,'metrics':mm,'optimizer':{'success':bool(s.success),'message':str(s.message),'nfev':int(s.nfev),'elapsed_s':e}} for _,n,s,ss,mm,e in runs],
           'state':{k:v.tolist() for k,v in st.items()},'frozen_prediction':fr['prediction'].tolist()}
  (OUT/f'fit_{policy}_phase_resolved.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
  fields=list(records[0])+['predicted_dadn_m_per_cycle','frozen_dadn_m_per_cycle','residual_log10']+list(payload['state'])
  with (OUT/f'predictions_{policy}_phase_resolved.csv').open('w',newline='',encoding='utf-8') as h:
   w=csv.DictWriter(h,fieldnames=fields,lineterminator='\n');w.writeheader()
   for i,row in enumerate(records):
    pred=payload['state']['prediction'][i];extra={'predicted_dadn_m_per_cycle':pred,'frozen_dadn_m_per_cycle':payload['frozen_prediction'][i],'residual_log10':math.log10(pred/float(row['observed_dadn_m_per_cycle']))};extra.update({k:v[i] for k,v in payload['state'].items()});w.writerow(dict(row)|extra)
  print(json.dumps({'policy':policy,'metrics':payload['metrics'],'optimizer':payload['optimizer']},indent=2),flush=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
