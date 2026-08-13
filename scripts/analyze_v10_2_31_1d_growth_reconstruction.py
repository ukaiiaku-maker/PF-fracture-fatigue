#!/usr/bin/env python3
"""Retrospective growth-law fitting gate for sparse v10.2.31 2-D validation."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

def models():
    return {
      "linear":(lambda n,a0,v:a0+v*n,[0,1e-12]),
      "power":(lambda n,a0,c,p:a0+c*np.maximum(n,0)**p,[0,1e-12,1]),
      "exponential":(lambda n,a0,A,tau:a0+A*np.expm1(np.clip(n/tau,-700,700)),[0,1e-6,1]),
    }
def fit(name,n,a):
    fn,p0=models()[name]; scale=max(float(n[-1]),1.0); x=n/scale
    try:
      popt,cov=curve_fit(fn,x,a,p0=p0,maxfev=50000,bounds=([-np.inf,0,0][:len(p0)],[np.inf,np.inf,np.inf][:len(p0)]))
      pred=fn(x,*popt); rss=float(np.sum((a-pred)**2)); k=len(popt); q=len(a)
      return {"ok":True,"rss":rss,"aic":q*math.log(max(rss/q,1e-300))+2*k,"bic":q*math.log(max(rss/q,1e-300))+k*math.log(q),"parameters":popt.tolist(),"covariance":cov.tolist(),"scale_cycles":scale}
    except Exception as exc:return {"ok":False,"error":str(exc)}
def crossing(name,result,target):
    if not result.get("ok"):return None
    fn,_=models()[name]; p=result["parameters"]; scale=result["scale_cycles"]
    lo,hi=0.0,1.0
    while fn(hi,*p)<target and hi<1e30: hi*=10
    if fn(hi,*p)<target:return None
    for _ in range(150):
      mid=(lo+hi)/2
      if fn(mid,*p)>=target:hi=mid
      else:lo=mid
    return hi*scale
def main():
 p=argparse.ArgumentParser();p.add_argument("--events",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
 rows=list(csv.DictReader(a.events.open())); groups={}
 for r in rows:groups.setdefault((r["class"],r["case"]),[]).append(r)
 output=[]
 for (cls,case),rr in groups.items():
  rr.sort(key=lambda r:int(r["event_index"])); n=np.array([float(r["cycles"]) for r in rr]); ext=np.array([float(r["cumulative_extension_m"]) for r in rr])
  if ext[-1]<100e-6:continue
  truth={"N50":float(np.interp(50e-6,ext,n)),"N100":float(np.interp(100e-6,ext,n))}
  for count in (5,6,8,10,len(rr)):
   if count>len(rr):continue
   nn=n[:count];aa=ext[:count]
   for name in models():
    f=fit(name,nn,aa); n50=crossing(name,f,50e-6);n100=crossing(name,f,100e-6)
    output.append({"class":cls,"case":case,"events_used":count,"events_total":len(rr),"model":name,"fit_ok":f.get("ok"),"aic":f.get("aic"),"bic":f.get("bic"),"predicted_N50":n50,"actual_N50":truth["N50"],"N50_relative_error":None if n50 is None else abs(n50/truth["N50"]-1),"predicted_N100":n100,"actual_N100":truth["N100"],"N100_relative_error":None if n100 is None else abs(n100/truth["N100"]-1),"fit":json.dumps(f,sort_keys=True)})
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
 summary={}
 for cls in "ABCD":
  summary[cls]={}
  for name in models():
   vals=[r for r in output if r["class"]==cls and r["model"]==name and r["events_used"] in (6,8,10) and r["N100_relative_error"] is not None]
   summary[cls][name]={"samples":len(vals),"median_N100_relative_error":None if not vals else float(np.median([r["N100_relative_error"] for r in vals])),"fraction_within_10pct":None if not vals else float(np.mean([r["N100_relative_error"]<=.1 for r in vals]))}
 (a.out.with_suffix(".summary.json")).write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");print(json.dumps(summary,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
