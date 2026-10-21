#!/usr/bin/env python3
"""Release only the supported EXP-floor exponent after capacity comparison."""
from __future__ import annotations
import argparse, csv, json, math, time
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.refit_qste340tm_capacity_policy import (
    DATA, OUT, LOWER, UPPER, build_records, evaluate, metrics, unpack,
)


def main() -> int:
    ap=argparse.ArgumentParser();ap.add_argument("--max-nfev",type=int,default=220);ap.add_argument("--resume",action="store_true");args=ap.parse_args()
    records=build_records(DATA); base=json.loads((OUT/"fit_uncapped.json").read_text())
    obs=np.log10(np.array([float(r["observed_dadn_m_per_cycle"]) for r in records]));rv=np.array([float(r["R"]) for r in records])
    weights=np.empty(len(records))
    for r in sorted(set(rv)):weights[rv==r]=1/math.sqrt(np.sum(rv==r))
    lo=np.r_[LOWER,math.log(.25)];hi=np.r_[UPPER,math.log(4.)]
    def residual(x):
        s=evaluate(x,records,"uncapped",1);e=(np.log10(s["prediction"])-obs)*weights
        return np.r_[e,max(0.,.25-float(np.min(s["raw_opening_barrier_over_kBT"])))*10]
    starts=[]
    prior_path=OUT/"fit_uncapped_curvature.json"
    if args.resume and prior_path.exists():
        starts=[("resume_curvature_best",np.asarray(json.loads(prior_path.read_text())["parameters_vector"],dtype=float))]
    else:
        for n in (.5,1.,2.5):starts.append((f"EXP_n_{n:g}",np.r_[base["parameters_vector"],math.log(n)]))
    runs=[]
    with ThreadPool(4) as pool:
        for name,x0 in starts:
            print(f"[curvature] optimizing {name}",flush=True);t=time.time()
            sol=least_squares(residual,x0,bounds=(lo,hi),max_nfev=args.max_nfev,x_scale="jac",ftol=1e-10,xtol=1e-10,gtol=1e-10,workers=pool.map)
            st=evaluate(sol.x,records,"uncapped",1);m=metrics(records,st)
            runs.append((m["overall_rmse_log10_rate"],name,sol,st,m,time.time()-t))
            print(f"[curvature] {name}: RMSE={m['overall_rmse_log10_rate']:.6f}, nfev={sol.nfev}, success={sol.success}",flush=True)
        _,name,best,_,_,_=min(runs,key=lambda z:z[0]);print(f"[curvature] refining {name}",flush=True)
        best=least_squares(residual,best.x,bounds=(lo,hi),max_nfev=args.max_nfev,x_scale="jac",ftol=2e-12,xtol=2e-12,gtol=2e-12,workers=pool.map)
    state=evaluate(best.x,records,"uncapped",1);frozen=evaluate(best.x,records,"uncapped",1,evolve=False)
    payload={"schema":"qste340tm-uncapped-released-exp-floor-curvature/1","policy":"uncapped","released_coordinate":"shared EXP_floor exponent n; alpha fixed",
             "parameters_vector":best.x.tolist(),"parameters":unpack(best.x),"metrics":metrics(records,state),"frozen_metrics":metrics(records,frozen),
             "optimizer":{"success":bool(best.success),"status":int(best.status),"message":str(best.message),"nfev":int(best.nfev),"cost":float(best.cost),"optimality":float(best.optimality)},
             "multi_start":[{"start":name,"metrics":m,"optimizer":{"success":bool(sol.success),"nfev":int(sol.nfev),"message":str(sol.message),"elapsed_s":elapsed}} for _,name,sol,st,m,elapsed in runs],
             "state":{k:v.tolist() for k,v in state.items()},"frozen_prediction":frozen["prediction"].tolist()}
    (OUT/"fit_uncapped_curvature.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    fields=list(records[0])+["predicted_dadn_m_per_cycle","frozen_dadn_m_per_cycle","residual_log10"]+list(payload["state"])
    with (OUT/"predictions_uncapped_curvature.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields,lineterminator="\n");w.writeheader()
        for i,row in enumerate(records):
            pred=payload["state"]["prediction"][i];extra={"predicted_dadn_m_per_cycle":pred,"frozen_dadn_m_per_cycle":payload["frozen_prediction"][i],"residual_log10":math.log10(pred/float(row["observed_dadn_m_per_cycle"]))}
            extra.update({k:v[i] for k,v in payload["state"].items()});w.writerow(dict(row)|extra)
    print(json.dumps({"metrics":payload["metrics"],"parameters":payload["parameters"],"optimizer":payload["optimizer"]},indent=2))
    return 0
if __name__=="__main__":raise SystemExit(main())
