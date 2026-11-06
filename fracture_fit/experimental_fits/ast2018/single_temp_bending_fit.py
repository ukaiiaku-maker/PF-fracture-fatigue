"""Bounded conditional native 1-D fit of early F3c at 21 C/rate 0.5.

The nominal beam field, geometry, compliance, entropies and all other material
parameters are fixed. Every native submission is recorded before another is
started. Unsupported measured ordinates are never extrapolated into a score.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import AstAssumptions
from .mechanics import observe_like_ast, post_event_J_curve
from .objective import _quadrature_weights, physically_admissible
from .shared_material_v9 import seed_evaluator
from .shared_material_v10 import NAMES, material_from_coordinates
from .spatial import predict_spatial

SOURCE_MATERIAL=Path("runs/ast2018_shared_floor_v12_enthalpy_checked/selected_conditional_material.json")
TARGET_ID="F3c_T21_r0.5"
BOUNDS=((.415,.495),(1.23,1.39),(.875,1.025))
MAX_SEARCH_CALLS=30
SEARCH_WALL_S=900.


def target_rows(data):
    rows=sorted((r for r in data.J_curves if r["curve_id"]==TARGET_ID and
        r["sample_for_fit"].lower() in ("true","1","yes") and
        0<float(r["crack_extension_um"])<=.5),
        key=lambda r:float(r["crack_extension_um"]))
    if len(rows)<5:raise ValueError("insufficient measured early F3c support")
    x=np.array([float(r["crack_extension_um"])*1e-6 for r in rows])
    y=np.array([float(r["J_N_per_m"]) for r in rows])
    if np.any(np.diff(x)<=0):raise ValueError("duplicate/nonmonotone fit abscissae")
    return rows,x,y,_quadrature_weights(x)


def material_at(seed,center,coordinates):
    gf,ge,hp=coordinates
    if any(not lo<=v<=hi for v,(lo,hi) in zip(coordinates,BOUNDS)):
        raise ValueError("isothermal coordinates outside declared bounds")
    material=replace(seed,
        cleavage=replace(seed.cleavage,Gf_ref_eV=seed.cleavage.Gf_ref_eV+gf-center[0]),
        emission=replace(seed.emission,G00_eV=seed.emission.G00_eV+ge-center[1]),
        peierls=replace(seed.peierls,H0_eV=hp),candidate_id="ast2018_21C_bending_trial")
    if not physically_admissible(material):
        raise ValueError("trial material violates active surface bounds")
    return material


def score(observed,target_x,target_y,weights):
    x,y=post_event_J_curve(observed)
    supported=target_x<=x[-1]+1e-15
    if not np.all(supported):
        return {"supported_points":int(np.sum(supported)),
                "total_points":len(target_x),"maximum_supported_extension_um":float(x[-1]*1e6),
                "support_shortfall_um":float(max(target_x[-1]-x[-1],0)*1e6),
                "weighted_rms_N_per_m":None,"level_N_per_m":None,
                "shape_rms_N_per_m":None,"objective":None}
    pred=np.interp(target_x,x,y)
    error=pred-target_y
    level=float(np.dot(weights,error))
    shape=float(np.sqrt(np.dot(weights,(error-level)**2)))
    rms=float(np.sqrt(np.dot(weights,error**2)))
    return {"supported_points":len(target_x),"total_points":len(target_x),
            "maximum_supported_extension_um":float(x[-1]*1e6),
            "support_shortfall_um":0.,"weighted_rms_N_per_m":rms,
            "level_N_per_m":level,"shape_rms_N_per_m":shape,
            "objective":float(np.sqrt(level*level+2*shape*shape)/20.)}


def _jsonable(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(type(value).__name__)


def _save_prediction(path,pred,observed):
    path.write_text(json.dumps({"status":pred.status,
        "numerical_failure":pred.numerical_failure,
        "history":[asdict(p) for p in pred.history],
        "observations":[asdict(p) for p in observed],"events":pred.events},
        indent=2,default=_jsonable)+"\n")


def run(output:Path,shape_path:Path):
    output.mkdir(parents=True,exist_ok=True)
    figures=output/"final_figures";figures.mkdir(exist_ok=True)
    primary,seed=seed_evaluator()
    selected=json.loads(SOURCE_MATERIAL.read_text())
    coordinates=[selected["coordinates"][name] for name in NAMES]
    base=material_from_coordinates(seed,coordinates)
    center=(float(selected["coordinates"]["Gf_open_294_eV"]),
            float(selected["coordinates"]["G0_emit_294_eV"]),
            float(base.peierls.H0_eV))
    condition=primary.ev.data.condition("Tp21C_rate0p5")
    assumptions=AstAssumptions(target_extension_m=.48e-6,
        nominal_K_step_MPa_sqrt_m=.1,maximum_nominal_K_MPa_sqrt_m=20.,
        loading_mode="nominal_bending_shape",bending_opening_shape_path=str(shape_path))
    specimen=assumptions.specimen(condition)
    physics=assumptions.physics(primary.ev.base_physics,specimen)
    raw,x,y,w=target_rows(primary.ev.data)
    (output/"selected_experimental_points.csv").write_text(
        "extension_um,J_N_per_m,quadrature_weight\n"+"".join(
            f"{xi*1e6:.12g},{yi:.12g},{wi:.12g}\n" for xi,yi,wi in zip(x,y,w)))
    configuration={"condition_id":"Tp21C_rate0p5","measured_curve":TARGET_ID,
        "fit_window_extension_um":[float(x[0]*1e6),float(x[-1]*1e6)],
        "selected_points":len(x),"objective":"sqrt(level^2+2*shape_rms^2)/20 N/m; full support required",
        "coordinates":["opening_floor_G_294_eV","emission_G0_294_eV","peierls_H0_eV"],
        "bounds":BOUNDS,"starting_vector":center,"max_search_calls":MAX_SEARCH_CALLS,
        "search_wall_s":SEARCH_WALL_S,"assumptions":assumptions.as_dict(),
        "source_material":str(SOURCE_MATERIAL),
        "source_material_sha256":hashlib.sha256(SOURCE_MATERIAL.read_bytes()).hexdigest(),
        "elastic_field_table":str(shape_path),
        "elastic_field_sha256":hashlib.sha256(shape_path.read_bytes()).hexdigest(),
        "native_receipt":primary.ev.receipt,
        "linked_loading_trace":False}
    (output/"fit_configuration.json").write_text(json.dumps(configuration,indent=2,default=_jsonable)+"\n")
    ledger=output/"evaluation_ledger.jsonl"
    if ledger.exists():
        raise FileExistsError("use a fresh output folder; existing trials must not be overwritten")
    seen={};best=None;best_vector=None
    start=time.monotonic()
    def evaluate(vector,stage):
        nonlocal best,best_vector
        vector=tuple(round(float(v),8) for v in vector)
        if vector in seen:return seen[vector]
        if len(seen)>=MAX_SEARCH_CALLS or time.monotonic()-start>SEARCH_WALL_S:
            return None
        tick=time.monotonic()
        row={"index":len(seen),"stage":stage,"coordinates":vector}
        try:
            material=material_at(base,center,vector)
            pred=predict_spatial(primary.ev.spatial,material,physics,specimen,
                                 294.15,.5,assumptions)
            observed=observe_like_ast(pred.history,specimen)
            m=score(observed,x,y,w)
            row.update(m,status=pred.status,numerical_failure=pred.numerical_failure,
                events=len(pred.events),load_steps=pred.load_steps,
                final_extension_um=(pred.history[-1].a_m-specimen.a0_m)*1e6,
                final_u_um=pred.history[-1].imposed_u_m*1e6,
                final_J_N_per_m=observed[-1].J_continuous_N_per_m)
            if row["index"]==0:_save_prediction(output/"seed_history.json",pred,observed)
            if row["objective"] is not None and pred.status=="complete" and (
                best is None or row["objective"]<best["objective"]):
                best=row;best_vector=vector
                _save_prediction(output/"search_best_history.json",pred,observed)
        except Exception as exc:
            row.update(status="evaluation_error",error=f"{type(exc).__name__}: {exc}",
                       supported_points=0,total_points=len(x),objective=None)
        row["runtime_s"]=time.monotonic()-tick
        with ledger.open("a") as stream:
            stream.write(json.dumps(row,default=_jsonable)+"\n");stream.flush()
        seen[vector]=row
        return row

    evaluate(center,"seed")
    # Resolve the narrow opening-floor rate sensitivity before moving the
    # emission and glide coordinates. The steps are declared, not adaptive
    # material priors or geometry multipliers.
    for gf in (.425,.435,.445,.455,.465,.475,.485):
        evaluate((gf,center[1],center[2]),"floor_scan")
    if best_vector is not None:
        for ge in (1.245,1.275,1.305,1.335,1.365):
            evaluate((best_vector[0],ge,best_vector[2]),"emission_scan")
    if best_vector is not None:
        for hp in (.89,.92,.95,.98,1.01):
            evaluate((best_vector[0],best_vector[1],hp),"glide_scan")
    if best_vector is not None:
        for step in (.0075,.003):
            for axis in (0,1,2):
                for sign in (-1,1):
                    candidate=list(best_vector);candidate[axis]+=sign*step*(3 if axis==1 else 1)
                    if BOUNDS[axis][0]<=candidate[axis]<=BOUNDS[axis][1]:
                        evaluate(candidate,f"local_{step:g}")
    # Confirm the selected vector with finer load increments; a change in
    # objective is reported, never silently folded into the search score.
    confirmations={}
    if best_vector is not None:
        for step in (.05,.025):
            fine=replace(assumptions,nominal_K_step_MPa_sqrt_m=step)
            material=material_at(base,center,best_vector)
            tick=time.monotonic()
            pred=predict_spatial(primary.ev.spatial,material,physics,specimen,
                                 294.15,.5,fine)
            observed=observe_like_ast(pred.history,specimen)
            metric=score(observed,x,y,w)
            confirmations[str(step)]={**metric,"status":pred.status,
                "runtime_s":time.monotonic()-tick,"events":len(pred.events),
                "numerical_failure":pred.numerical_failure}
            _save_prediction(output/f"best_step_{step:g}_history.json",pred,observed)
    summary={"status":"supported_candidate_with_residual" if best_vector is not None else
                    "no_supported_candidate_within_budget",
             "condition_id":"Tp21C_rate0p5","fit_window_um":[x[0]*1e6,x[-1]*1e6],
             "search_submissions":len(seen),"search_elapsed_s":time.monotonic()-start,
             "best_vector":best_vector,"best_search_row":best,
             "confirmations":confirmations,
             "evaluation_status_counts":{s:sum(r.get("status")==s for r in seen.values())
                 for s in set(r.get("status") for r in seen.values())}}
    (output/"fit_summary.json").write_text(json.dumps(summary,indent=2,default=_jsonable)+"\n")
    if best_vector is not None:
        _figures(output,figures,x,y,w)
    return summary


def _figures(output,figures,x,y,w):
    seed=json.loads((output/"seed_history.json").read_text())
    fitted=json.loads((output/"best_step_0.025_history.json").read_text())
    fig,ax=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    ax[0].plot(x*1e6,y,"ko",label="F3c measured")
    for data,label in ((seed,"v12 seed"),(fitted,"bending fit")):
        aa,jj=post_event_J_curve(data["observations"])
        ax[0].plot(aa*1e6,jj,"o-",ms=3,label=label)
        residual=np.where(x<=aa[-1]+1e-15,np.interp(x,aa,jj)-y,np.nan)
        ax[1].plot(x*1e6,residual,"o-",ms=3,label=label)
    ax[0].set(xlabel="crack extension (µm)",ylabel="continuous J (N/m)",
              title="F3c early resistance")
    ax[1].axhline(0,color="k",lw=.8)
    ax[1].set(xlabel="crack extension (µm)",ylabel="model − measured J (N/m)",
              title="Residuals at measured ordinates")
    for a in ax:a.legend(fontsize=8)
    fig.savefig(figures/"early_J_fit_and_residuals.png",dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for data,label in ((seed,"v12 seed"),(fitted,"bending fit")):
        h=data["history"]
        ax[0].plot([p["imposed_u_m"]*1e6 for p in h],
                   [p["force_N"]*1e3 for p in h],label=label)
        ax[1].plot([(p["a_m"]-2.25e-6)*1e6 for p in h],
                   [p["plastic_u_m"]*1e6 for p in h],label=label)
    ax[0].set(xlabel="imposed displacement (µm)",ylabel="force (mN)",
              title="Predicted loading; no linked F3c trace")
    ax[1].set(xlabel="crack extension (µm)",ylabel="plastic displacement (µm)",
              title="Mechanism response")
    for a in ax:a.legend(fontsize=8)
    fig.savefig(figures/"loading_and_plastic_history.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--shape",type=Path,required=True)
    args=p.parse_args()
    run(args.output,args.shape)
