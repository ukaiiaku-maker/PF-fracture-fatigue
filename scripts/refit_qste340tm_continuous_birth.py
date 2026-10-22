#!/usr/bin/env python3
"""Final QSTE340TM refit with exact frozen-panel continuous-birth survival."""
from __future__ import annotations
import argparse,csv,json,math,time
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import differential_evolution,least_squares
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.refit_qste340tm_capacity_policy import DATA,OUT,LOWER,UPPER,build_records,evaluate,metrics,unpack

MODE='continuous_birth_exact'; POLICY='uncapped'; OUTDIR=OUT/'continuous_birth_exact'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--max-nfev',type=int,default=240);ap.add_argument('--global-iterations',type=int,default=20);args=ap.parse_args();OUTDIR.mkdir(parents=True,exist_ok=True)
 records=build_records(DATA);rv=np.array([float(r['R']) for r in records]);obs=np.log10(np.array([float(r['observed_dadn_m_per_cycle']) for r in records]));weights=np.empty(len(records))
 for r in sorted(set(rv)):weights[rv==r]=1/math.sqrt(np.sum(rv==r))
 lo=LOWER.copy();hi=UPPER.copy();hi[9]=math.log1p(3000.)
 prior=json.load(open(OUT/'fit_uncapped_phase_resolved.json'));base=np.array(prior['parameters_vector'])
 def residual(x,mask=None):
  st=evaluate(x,records,POLICY,1,retention_timing=MODE);e=(np.log10(st['prediction'])-obs)*weights
  if mask is not None:e=e[mask]
  return np.r_[e,max(0.,.25-float(np.min(st['raw_opening_barrier_over_kBT'])))*10]
 starts=[]
 for label,path in [('uncapped_phase',OUT/'fit_uncapped_phase_resolved.json'),('fixed10000_phase',OUT/'fit_fixed_10000_phase_resolved.json'),('uncapped_legacy',OUT/'fit_uncapped.json')]:
  x=np.array(json.load(open(path))['parameters_vector'][:10]);x[9]=math.log1p(min(max(math.expm1(x[9]),0.),3000.));starts.append((label,x))
 for n0 in (0.,100.,500.,1500.,3000.):
  x=base.copy();x[9]=math.log1p(n0);starts.append((f'initial_{n0:g}',x))
 runs=[]
 with ThreadPool(4) as pool:
  print('[continuous] bounded global acquisition',flush=True)
  de=differential_evolution(lambda x:float(np.dot(residual(x),residual(x))),list(zip(lo,hi)),seed=20261021,popsize=5,maxiter=args.global_iterations,polish=False,workers=pool.map,updating='deferred',tol=1e-5)
  starts.append(('bounded_global',de.x))
  scored=sorted((float(np.dot(residual(x),residual(x))),name,x) for name,x in starts)
  (OUTDIR/'start_scores.json').write_text(json.dumps([{'start':n,'weighted_sse':v,'parameters':unpack(x)} for v,n,x in scored],indent=2,sort_keys=True)+'\n')
  for _,name,x0 in scored[:5]:
   print('[continuous]',name,flush=True);t=time.time();sol=least_squares(residual,x0,bounds=(lo,hi),max_nfev=args.max_nfev,x_scale='jac',ftol=1e-10,xtol=1e-10,gtol=1e-10,workers=pool.map)
   st=evaluate(sol.x,records,POLICY,1,retention_timing=MODE);m=metrics(records,st);runs.append((m['overall_rmse_log10_rate'],name,sol,st,m,time.time()-t));print(f'  RMSE={m["overall_rmse_log10_rate"]:.6f} n0={unpack(sol.x)["initial_retained_count"]:.3f} nfev={sol.nfev} success={sol.success}',flush=True)
   (OUTDIR/f'checkpoint_{name}.json').write_text(json.dumps({'start':name,'parameters_vector':sol.x.tolist(),'parameters':unpack(sol.x),'metrics':m,'optimizer':{'success':bool(sol.success),'message':str(sol.message),'nfev':int(sol.nfev)}},indent=2,sort_keys=True)+'\n')
  _,name,sol,_,_,_=min(runs,key=lambda z:z[0]);print('[continuous] refine',name,flush=True);sol=least_squares(residual,sol.x,bounds=(lo,hi),max_nfev=args.max_nfev,x_scale='jac',ftol=2e-12,xtol=2e-12,gtol=2e-12,workers=pool.map)
 state=evaluate(sol.x,records,POLICY,1,retention_timing=MODE);frozen=evaluate(sol.x,records,POLICY,1,evolve=False,retention_timing=MODE)
 convergence={'waveform':{},'outer_history':{}}
 for n in (32,64,128,256,512):convergence['waveform'][str(n)]=metrics(records,evaluate(sol.x,records,POLICY,1,retention_timing=MODE,waveform_steps=n))
 for n in (1,2,4,8):convergence['outer_history'][str(n)]=metrics(records,evaluate(sol.x,records,POLICY,n,retention_timing=MODE,waveform_steps=128))
 payload={'schema':'qste340tm-continuous-birth-exact-joint-refit/1','policy':POLICY,'retention_timing':MODE,'initial_state_profile_range':[0,3000],
          'parameters_vector':sol.x.tolist(),'parameters':unpack(sol.x),'metrics':metrics(records,state),'frozen_metrics':metrics(records,frozen),'convergence':convergence,
          'optimizer':{'success':bool(sol.success),'message':str(sol.message),'nfev':int(sol.nfev),'cost':float(sol.cost),'optimality':float(sol.optimality)},
          'multi_start':[{'start':n,'metrics':m,'parameters':unpack(s.x),'optimizer':{'success':bool(s.success),'message':str(s.message),'nfev':int(s.nfev),'elapsed_s':e}} for _,n,s,ss,m,e in runs],
          'state':{k:v.tolist() for k,v in state.items()},'frozen_prediction':frozen['prediction'].tolist()}
 (OUTDIR/'fit_uncapped_continuous_birth.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
 fields=list(records[0])+['predicted_dadn_m_per_cycle','frozen_dadn_m_per_cycle','residual_log10']+list(payload['state'])
 with (OUTDIR/'predictions_uncapped_continuous_birth.csv').open('w',newline='',encoding='utf-8') as h:
  w=csv.DictWriter(h,fieldnames=fields,lineterminator='\n');w.writeheader()
  for i,row in enumerate(records):
   pred=payload['state']['prediction'][i];extra={'predicted_dadn_m_per_cycle':pred,'frozen_dadn_m_per_cycle':payload['frozen_prediction'][i],'residual_log10':math.log10(pred/float(row['observed_dadn_m_per_cycle']))};extra.update({k:v[i] for k,v in payload['state'].items()});w.writerow(dict(row)|extra)
 print(json.dumps({'metrics':payload['metrics'],'parameters':payload['parameters'],'optimizer':payload['optimizer'],'convergence':{a:{k:v['overall_rmse_log10_rate'] for k,v in b.items()} for a,b in convergence.items()}},indent=2))
 return 0
if __name__=='__main__':raise SystemExit(main())
