#!/usr/bin/env python3
"""Matched 20-start no-bulk control for the signed-bulk 1-D comparison."""
from __future__ import annotations

import argparse, json, math, time
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution, least_squares

import scripts.refit_qste340tm_emission_owned_blunting as base
import scripts.refit_qste340tm_resolved_sloping_barrier as slope
from scripts.refit_qste340tm_restored_native import build_records
from scripts.refit_qste340tm_signed_bulk_1d import write_predictions

SOURCE = slope.OUT / "fit.json"
OUT = base.OUT / "resolved_increment_1d/no_bulk_matched_refit_20261028"


def profile(candidate, records, phases, step, seed):
    state=slope.state_trajectory(candidate,records,phases,step)
    fixed=np.asarray(candidate,float).copy(); indices=np.arange(5)
    def assemble(values):
        out=fixed.copy();out[indices]=values;return out
    def objective(values):
        r=slope.objective_residual(assemble(values),records,state)
        return float(r@r)
    result=differential_evolution(objective,
        list(zip(slope.LOWER[indices],slope.UPPER[indices])),seed=seed,
        popsize=7,maxiter=32,polish=True,tol=1e-6)
    return float(result.fun),assemble(result.x)


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--support-per-R",type=int,default=16)
    # Match the accepted signed-bulk acquisition exactly.  Eight phases was
    # used by an earlier provisional control and is not the fair comparator.
    ap.add_argument("--fit-phases",type=int,default=6)
    ap.add_argument("--final-phases",type=int,default=48)
    ap.add_argument("--max-nfev",type=int,default=90)
    ap.add_argument("--advection-step-um",type=float,default=.25)
    ns=ap.parse_args(argv);OUT.mkdir(parents=True,exist_ok=True);started=time.time()
    parent=json.loads(SOURCE.read_text());control=np.asarray(parent["parameters_vector"],float)
    records=build_records("index_rank",premeasurement=True)
    support=base.select_support(records,ns.support_per_R);step=ns.advection_step_um*1e-6
    residual=slope.CachedFullResidual(support,ns.fit_phases,step)
    starts=slope.mechanical_bank(control,count=20,seed=20261028)
    profiled=[];profiles=[]
    for index,(label,start) in enumerate(starts):
        score,vector=profile(start,support,ns.fit_phases,step,20261028+index)
        profiles.append({"label":label,"opening_profile_SSE":score})
        profiled.append((score,label,vector));print(f"[profile {label}] SSE={score:.8g}",flush=True)
    fitted=[]
    for _,label,start in sorted(profiled)[:4]:
        solution=least_squares(residual,start,bounds=(slope.LOWER,slope.UPPER),
            max_nfev=ns.max_nfev,x_scale="jac",ftol=2e-6,xtol=2e-6,gtol=2e-6)
        score=float(residual(solution.x)@residual(solution.x));fitted.append((score,label,solution))
        print(f"[{label}] SSE={score:.8g} nfev={solution.nfev}",flush=True)
    _,label,solution=min(fitted,key=lambda item:item[0])
    full_score,vector=profile(solution.x,records,ns.final_phases,step,20261128)
    state=slope.state_trajectory(vector,records,ns.final_phases,step)
    prediction,raw,_=slope.prediction(vector,state);metrics=slope.metrics(records,prediction,state)
    payload={"schema":"qste340tm-no-bulk-matched-refit/1","status":"executed",
        "comparison_contract":{"mechanical_bank_count":20,"mechanical_bank_seed":20261028,
            "opening_profile":"DE popsize=7 maxiter=32 polish=true",
            "local_refinement_count":4,"fit_phases":ns.fit_phases,
            "final_phases":ns.final_phases,"support_per_R":ns.support_per_R,
            "same_as_signed_bulk":bool(ns.fit_phases==6 and ns.final_phases==48
                                        and ns.support_per_R==16)},
        "source_fit":str(SOURCE),"selected_start":label,"parameters_vector":vector.tolist(),
        "parameters":slope.unpack(vector),"metrics":metrics,"support_points":len(support),
        "total_points":len(records),"fit_phases":ns.fit_phases,"final_phases":ns.final_phases,
        "advection_step_m":step,"elapsed_s":time.time()-started,
        "multi_start":[{"label":l,"SSE":s,"nfev":int(sol.nfev)} for s,l,sol in sorted(fitted)],
        "opening_profiles":profiles,"full_data_opening_profile_SSE":full_score,
        "barrier_over_kBT_range":[float(np.min(raw)),float(np.max(raw))]}
    (OUT/"fit.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    write_predictions(OUT/"predictions.csv",records,prediction,state)
    print(json.dumps({"output":str(OUT),"RMSE":metrics["overall_rmse_log10_rate"]},indent=2))
    return 0

if __name__=="__main__":raise SystemExit(main())
