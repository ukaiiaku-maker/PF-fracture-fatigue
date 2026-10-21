#!/usr/bin/env python3
"""Fresh leave-one-R-out optimizations for the converged fixed-1000 policy."""
from __future__ import annotations
import argparse,json,math
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
import sys,time
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.refit_qste340tm_capacity_policy import DATA,OUT,LOWER,UPPER,build_records,evaluate,metrics,pack,unpack

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--max-nfev',type=int,default=160);args=ap.parse_args()
 records=build_records(DATA);rv=np.array([float(r['R']) for r in records]);obs=np.log10(np.array([float(r['observed_dadn_m_per_cycle']) for r in records]))
 control=json.load(open(DATA/'model_conditional/stateful_exp_floor/fit.json'));full=json.load(open(OUT/'fit_fixed_1000.json'))
 historical=pack(control['fit']['parameters']);fullx=np.array(full['parameters_vector']);results=[]
 for held in (.1,.3,.5,.7):
  train=rv!=held;weights=np.zeros(len(records))
  for r in sorted(set(rv[train])):weights[rv==r]=1/math.sqrt(np.sum(rv==r))
  def residual(x):
   st=evaluate(x,records,'fixed_1000',1);e=(np.log10(st['prediction'])-obs)*weights
   return np.r_[e[train],max(0.,.25-float(np.min(st['raw_opening_barrier_over_kBT'])))*10]
  runs=[]
  with ThreadPool(4) as pool:
   for name,x0 in [('historical_control',historical),('complete_data_warm_start',fullx)]:
    print(f'[holdout R={held}] {name}',flush=True);t=time.time()
    sol=least_squares(residual,x0,bounds=(LOWER,UPPER),max_nfev=args.max_nfev,x_scale='jac',ftol=1e-10,xtol=1e-10,gtol=1e-10,workers=pool.map)
    st=evaluate(sol.x,records,'fixed_1000',1);m=metrics(records,st)
    train_rmse=float(np.sqrt(np.mean(np.log10(st['prediction'][train]/(10**obs[train]))**2)))
    runs.append((train_rmse,name,sol,st,m,time.time()-t));print(f'  train={train_rmse:.6f} held={m["block_metrics"][str(held)]["rmse_log10_rate"]:.6f} nfev={sol.nfev} success={sol.success}',flush=True)
  tr,name,sol,st,m,elapsed=min(runs,key=lambda z:z[0])
  results.append({'held_out_R':held,'selected_start':name,'training_rmse_log10_rate':tr,'held_out_metrics':m['block_metrics'][str(held)],'all_metrics':m,
                  'parameters':unpack(sol.x),
                  'optimizer':{'success':bool(sol.success),'message':str(sol.message),'nfev':int(sol.nfev),'cost':float(sol.cost),'optimality':float(sol.optimality),'elapsed_s':elapsed},
                  'candidate_starts':[{'start':n,'training_rmse':t,'heldout_rmse':mm['block_metrics'][str(held)]['rmse_log10_rate'],'success':bool(s.success),'nfev':int(s.nfev)} for t,n,s,ss,mm,e in runs]})
 (OUT/'leave_one_R_out_fixed_1000.json').write_text(json.dumps({'schema':'qste340tm-fixed-1000-fresh-complete-R-holdouts/1','results':results},indent=2,sort_keys=True)+'\n')
 print(json.dumps([{'held_out_R':r['held_out_R'],'training_rmse':r['training_rmse_log10_rate'],'heldout_rmse':r['held_out_metrics']['rmse_log10_rate'],'optimizer':r['optimizer']} for r in results],indent=2))
 return 0
if __name__=='__main__':raise SystemExit(main())
