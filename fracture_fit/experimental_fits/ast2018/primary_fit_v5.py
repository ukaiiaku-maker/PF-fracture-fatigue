"""Conditional linked F2a/F3a fit with the conserved v4 Arrhenius model."""
from __future__ import annotations

import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import qmc

from .single_condition import Evaluator, from_isothermal
from .objective import physically_admissible
from .single_condition_search import SOURCE, DATA, bounds, admissible
from .single_condition_ledger import SubmissionLedger
from .reduced import predict_reduced
from .spatial import predict_spatial
from .mechanics import observe_like_ast

RUN=Path("runs/ast2018_primary_fit_v5")
U_END=2.919786e-6
TSTAR=294.15
RATE=.2
TOL_K=.15
TOL_S=.04
FREE=(0,1,6,2,3)
MAX_REDUCED=384
MAX_SPATIAL=6
WALL_LIMIT_S=1800
ANALYTICAL_STOP_S=1480


def _group(rows,xkey,ykey,xscale=1.):
    pairs=sorted((float(row[xkey])*xscale,float(row[ykey])) for row in rows)
    xs=np.unique([pair[0] for pair in pairs])
    ys=np.array([np.median([v for xx,v in pairs if xx==x]) for x in xs])
    return xs,ys


def _weights(x):
    widths=np.empty_like(x)
    widths[0]=.5*(x[1]-x[0]);widths[-1]=.5*(x[-1]-x[-2])
    widths[1:-1]=.5*(x[2:]-x[:-2])
    return widths/np.sum(widths)


def _huber(x):
    y=np.abs(x)
    return np.where(y<=1,.5*y*y,y-.5)


class PrimaryEvaluator:
    def __init__(self,slope=None,geometry=None,step=.05):
        self.ev=Evaluator(SOURCE,DATA,"floor",step=step,
                          protocol="line_content_reconciliation_v4")
        data=self.ev.data
        loading=[r for r in data.K_displacement if
                 r["curve_id"]=="Ast2018_F2a_T21_rate0.2" and r["branch"]=="loading"]
        stiffness=[r for r in data.stiffness if r["curve_id"]=="F3a_T21_r0.2"]
        self.uK,self.K=_group(loading,"displacement_um","K_MPa_sqrt_m",1e-6)
        self.uS,self.S=_group(stiffness,"displacement_um","relative_stiffness",1e-6)
        self.wK=_weights(self.uK);self.wS=_weights(self.uS)
        first=[r for r in sorted(loading,key=lambda r:float(r["displacement_um"]))
               if float(r["K_MPa_sqrt_m"])<=2.5][:5]
        early_u=np.array([float(r["displacement_um"]) for r in first])
        early_K=np.array([float(r["K_MPa_sqrt_m"]) for r in first])
        self.slope=float(early_u@early_K/(early_u@early_u)) if slope is None else float(slope)
        specimen=self.ev.assumptions.specimen(data.condition("Tp21C_rate0p2"))
        if geometry is not None:
            specimen=replace(specimen,L_m=geometry[0]*1e-6,a0_m=geometry[1]*1e-6)
        C0=specimen.geometry_factor(specimen.a0_m)/(self.slope*1e12)
        self.specimen=replace(specimen,initial_compliance_m_per_N=C0,
                              instrument_compliance_m_per_N=0.)
        self.physics=self.ev.assumptions.physics(self.ev.base_physics,self.specimen)
        self.step=step
        self.initial_slope_rows=[{"u_um":float(u),"K_MPa_sqrt_m":float(k),
                                  "residual_MPa_sqrt_m":float(k-self.slope*u)}
                                 for u,k in zip(early_u,early_K)]
        self.initial_slope_rms=float(np.sqrt(np.mean((early_K-self.slope*early_u)**2)))
        self.old_vector=json.loads(Path("runs/ast2018_21c_rate0p5_extension_stop_v2/fit_summary.json")
                                   .read_text())["winner_isothermal"]
        self.box=bounds(self.ev)
        self.window_edges={"K":np.linspace(self.uK[0],self.uK[-1],4),
                           "S":np.linspace(self.uS[0],self.uS[-1],4)}

    def full_vector(self,y):
        x=np.asarray(self.old_vector,float).copy()
        x[list(FREE)]=np.asarray(y,float)
        return x

    def free_vector(self,x):return np.asarray(x,float)[list(FREE)]

    def valid(self,x):return admissible(self.ev,x)

    def _curve_metrics(self,grid,observed,weights,model_u,model_q,tol,edges):
        supported=grid<=model_u[-1]+1e-15
        fraction=float(np.sum(weights[supported]))
        if np.any(supported):
            r=np.interp(grid[supported],model_u,model_q)-observed[supported]
            w=weights[supported]/np.sum(weights[supported])
            bias=float(np.dot(w,r))
            rms=float(np.sqrt(np.dot(w,r*r)))
            centered=float(np.sqrt(np.dot(w,(r-bias)**2)))
            loss=(bias/tol)**2+2*float(np.dot(w,_huber((r-bias)/tol)))
        else:
            bias=rms=centered=1e6;loss=1e5
        windows=[];changes=[]
        for i in range(3):
            mask=supported & (grid>=edges[i]) & (grid<=edges[i+1])
            if np.count_nonzero(mask)<2:
                windows.append(None);changes.append(None);continue
            pred=np.interp(grid[mask],model_u,model_q)
            obs=observed[mask]
            ww=weights[mask]/np.sum(weights[mask])
            windows.append(float(np.sqrt(np.dot(ww,(pred-obs)**2))))
            changes.append(float((pred[-1]-pred[0])-(obs[-1]-obs[0])))
        return {"bias":bias,"rms":rms,"centered_rms":centered,"loss":loss,
                "support_fraction":fraction,"window_rms":windows,
                "window_change_errors":changes}

    def run(self,x,level="reduced",save=None,material=None):
        t0=time.monotonic()
        x=np.asarray(x,float)
        row={"x":x.tolist(),"free_y":self.free_vector(x).tolist(),
             "level":level,"slope_MPa_sqrt_m_per_um":self.slope,
             "initial_compliance_m_per_N":self.specimen.initial_compliance_m_per_N,
             "nominal_displacement_rate_m_per_s":self.specimen.initial_displacement_rate(RATE)}
        try:
            if material is None:
                if not self.valid(x):raise ValueError("inadmissible candidate")
                material=from_isothermal(self.ev.material,x,self.ev.completion)
            elif not physically_admissible(material):
                raise ValueError("inadmissible supplied material")
            predictor=predict_reduced if level=="reduced" else predict_spatial
            module=self.ev.zero if level=="reduced" else self.ev.spatial
            result=predictor(module,material,self.physics,self.specimen,TSTAR,RATE,
                             self.ev.assumptions,deterministic_threshold=True,
                             required_u_m=U_END,required_extension_m=0.)
            observed=observe_like_ast(result.history,self.specimen)
            model_u=np.array([p.u_m for p in observed])
            model_K=np.array([p.K_nominal_MPa_sqrt_m for p in observed])
            model_S=np.array([p.relative_stiffness for p in observed])
            K=self._curve_metrics(self.uK,self.K,self.wK,model_u,model_K,TOL_K,
                                  self.window_edges["K"])
            S=self._curve_metrics(self.uS,self.S,self.wS,model_u,model_S,TOL_S,
                                  self.window_edges["S"])
            full=(result.status=="complete" and model_u[-1]>=U_END-1e-11 and
                  K["support_fraction"]>=1-1e-12 and S["support_fraction"]>=1-1e-12)
            support_penalty=1000*((1-K["support_fraction"])**2+
                                  (1-S["support_fraction"])**2)
            if not full:support_penalty+=100
            loss=K["loss"]+S["loss"]+support_penalty
            acceptable=(full and K["rms"]<=TOL_K and S["rms"]<=TOL_S and
                        abs(K["bias"])<=.10 and abs(S["bias"])<=.025 and
                        all(v is not None and abs(v)<=.20 for v in K["window_change_errors"]) and
                        all(v is not None and abs(v)<=.06 for v in S["window_change_errors"]))
            last=observed[-1]
            row.update(status=result.status,numerical_failure=result.numerical_failure,
                       full_support=full,acceptable=acceptable,loss=loss,
                       support_penalty=support_penalty,K_metrics=K,S_metrics=S,
                       events=len(result.events),load_steps=result.load_steps,
                       final_u_m=float(model_u[-1]),final_extension_m=last.da_m,
                       final_J_N_per_m=last.J_continuous_N_per_m,
                       final_J_el_N_per_m=(last.K_el_current_MPa_sqrt_m*1e6)**2/
                                        self.specimen.Eprime_Pa,
                       final_plastic_u_m=last.plastic_u_compliance_m,
                       maximum_nominal_K_MPa_sqrt_m=float(np.max(model_K)))
            if result.events:
                event=result.events[-1]
                row["last_event"]={key:event.get(key) for key in (
                    "cumulative_activations","cumulative_emitted_line",
                    "cumulative_escaped_line","cumulative_glide_line_distance_m",
                    "cumulative_birth_plastic_u_m","cumulative_transport_plastic_u_m",
                    "cleavage_effective_rate_s","emission_effective_activations_s",
                    "peierls_effective_encounter_s","taylor_effective_release_s")}
            if save is not None:
                save=Path(save);save.mkdir(parents=True,exist_ok=True)
                if result.final_state is not None:
                    np.savez_compressed(save/"final_state.npz",**result.final_state)
                    row["final_tip_radius_m"]=float(result.final_state["tip_radius_m"])
                    row["final_state_file"]="final_state.npz"
                for name,items in (("mechanical_history.csv",[asdict(v) for v in result.history]),
                                   ("ast_observer.csv",[asdict(v) for v in observed])):
                    with (save/name).open("w",newline="") as handle:
                        writer=csv.DictWriter(handle,fieldnames=list(items[0]))
                        writer.writeheader();writer.writerows(items)
                (save/"events.json").write_text(json.dumps(result.events,indent=2)+"\n")
        except Exception as exc:
            row.update(status="numerical_failure",error=type(exc).__name__+": "+str(exc),
                       full_support=False,acceptable=False,loss=1e6)
        row["runtime_s"]=time.monotonic()-t0
        return row


def _hashes():
    names=("primary_fit_v5.py","line_profile.py","reduced.py","spatial.py",
           "mechanics.py","single_condition.py")
    return {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in names}


def _append(path,row):
    with path.open("a") as handle:
        handle.write(json.dumps(row,allow_nan=False)+"\n")
        handle.flush()


def submit(ledger,evaluator,x,level,stage,save=False):
    key=(f"v5:{level}:{evaluator.slope:.12g}:{evaluator.step:.12g}:"
         f"L={evaluator.specimen.L_m:.12g}:a0={evaluator.specimen.a0_m:.12g}:")+",".join(f"{v:.12g}" for v in x)
    reservation=ledger.reserve(level,key,stage)
    eid=reservation["evaluation_id"]
    destination=RUN/"histories"/eid if save else None
    row=evaluator.run(x,level,destination)
    row.update(evaluation_id=eid,stage=stage,key=key)
    _append(RUN/"evaluations.jsonl",row)
    ledger.complete(eid,"error" if row.get("error") else row.get("status","unknown"))
    return row


def _distinct(rows,count=8):
    valid=[r for r in rows if r.get("full_support") and math.isfinite(r.get("loss",math.inf))]
    valid.sort(key=lambda r:r["loss"])
    selected=[]
    for row in valid:
        y=np.asarray(row["free_y"])
        if all(np.linalg.norm((y-np.asarray(s["free_y"])) /
                              np.maximum(np.asarray([6.,1e10,5.,5.,1e10]),1))>.03
               for s in selected):selected.append(row)
        if len(selected)>=count:break
    return selected


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    evaluator=PrimaryEvaluator()
    ledger=SubmissionLedger(RUN,MAX_REDUCED,MAX_SPATIAL,WALL_LIMIT_S)
    if ledger.audit()["unresolved_ids"]:
        raise RuntimeError("unresolved submission; inspect before continuing")
    box=evaluator.box
    config={"native_source":SOURCE,"native_commit":evaluator.ev.receipt["commit"],
        "source_hashes":_hashes(),"condition":"Tp21C_rate0p2",
        "primary_curves":["Ast2018_F2a_T21_rate0.2/loading","F3a_T21_r0.2"],
        "K_support_m":[float(evaluator.uK[0]),float(evaluator.uK[-1])],
        "S_support_m":[float(evaluator.uS[0]),float(evaluator.uS[-1])],
        "fixed_displacement_end_m":U_END,"initial_slope_rows":evaluator.initial_slope_rows,
        "initial_slope_MPa_sqrt_m_per_um":evaluator.slope,
        "initial_slope_rms_MPa_sqrt_m":evaluator.initial_slope_rms,
        "C0_m_per_N":evaluator.specimen.initial_compliance_m_per_N,
        "nominal_speed_m_per_s":evaluator.specimen.initial_displacement_rate(RATE),
        "instrument_compliance_m_per_N":0.,"specimen":asdict(evaluator.specimen),
        "physics":asdict(evaluator.physics),"material_seed":asdict(evaluator.ev.material),
        "thermal_completion":evaluator.ev.completion,"old_vector":evaluator.old_vector,
        "full_bounds":box.tolist(),"free_order":["opening_G0_eV","opening_sigma_c_Pa",
            "opening_Gf_eV","emission_G0_eV","emission_sigma_c_Pa"],
        "free_indices":list(FREE),"fixed_indices":[4,5],
        "objective":{"tol_K_MPa_sqrt_m":TOL_K,"tol_relative_stiffness":TOL_S,
            "bias_K_limit":.10,"bias_S_limit":.025,"change_K_limit":.20,
            "change_S_limit":.06,"window_edges_m":{k:v.tolist() for k,v in evaluator.window_edges.items()},
            "support_penalty":"1000*(uncovered_K^2+uncovered_S^2)+100 if incomplete",
            "quadrature":"trapezoid over observed displacement; median duplicate abscissae",
            "huber_transition":1},
        "limits":{"reduced":MAX_REDUCED,"spatial":MAX_SPATIAL,"wall_s":WALL_LIMIT_S}}
    (RUN/"configuration.json").write_text(json.dumps(config,indent=2,default=float)+"\n")
    rows=[json.loads(s) for s in (RUN/"evaluations.jsonl").read_text().splitlines()] if (RUN/"evaluations.jsonl").exists() else []
    known={r["key"] for r in rows}
    rng=np.random.default_rng(20182511)
    old=np.asarray(evaluator.old_vector,float)
    seeds=[old]
    for g0,sc,gf,eg,es in ((5.,4e9,1.4,1.1,2.5e9),
                            (6.,8e9,1.8,.8,3e9),
                            (3.,2e9,1.2,.6,5e9),
                            (5.,10e9,2.,1.5,2.5e9),
                            (2.5,6e9,1.,.5,8e9),
                            (6.,4e9,2.,1.8,8e9)):
        seeds.append(evaluator.full_vector([g0,sc,gf,eg,es]))
    sobol=qmc.Sobol(5,scramble=True,seed=20182512).random_base2(6)
    free_box=box[list(FREE)]
    for u in sobol:
        seeds.append(evaluator.full_vector(free_box[:,0]+u*(free_box[:,1]-free_box[:,0])))
    def elapsed():
        first=ledger.started_unix_s()
        return 0. if first is None else time.time()-first
    def try_one(x,stage):
        nonlocal rows
        if not evaluator.valid(x):return None
        key=f"v5:reduced:{evaluator.slope:.12g}:{evaluator.step:.12g}:"+",".join(f"{v:.12g}" for v in x)
        if key in known or ledger.remaining("reduced")<=0 or elapsed()>=ANALYTICAL_STOP_S:return None
        known.add(key)
        row=submit(ledger,evaluator,x,"reduced",stage)
        rows.append(row)
        if len(rows)%10==0:
            best=min(rows,key=lambda r:r.get("loss",math.inf))
            print("progress",len(rows),round(elapsed()),"best",best["evaluation_id"],
                  round(best["loss"],4),"K",best.get("K_metrics",{}).get("rms"),
                  "S",best.get("S_metrics",{}).get("rms"),flush=True)
        return row
    for x in seeds:
        if len(rows)>=64 or elapsed()>=ANALYTICAL_STOP_S:break
        try_one(x,"initial_design")
    width=free_box[:,1]-free_box[:,0]
    iteration=0
    while ledger.remaining("reduced")>80 and elapsed()<ANALYTICAL_STOP_S:
        valid=[r for r in rows if r.get("full_support")]
        if valid and any(r.get("acceptable") for r in valid):break
        parents=_distinct(rows,12)
        if not parents:
            y=free_box[:,0]+rng.random(5)*width
        else:
            parent=parents[iteration%len(parents)]
            y=np.asarray(parent["free_y"]).copy()
            scale=(.22 if iteration<60 else .11 if iteration<160 else .055)
            dims=rng.choice(5,size=int(rng.integers(1,4)),replace=False)
            y[dims]+=rng.normal(size=len(dims))*scale*width[dims]
            y=np.clip(y,free_box[:,0],free_box[:,1])
            y[2]=min(y[2],y[0])
        try_one(evaluator.full_vector(y),"adaptive")
        iteration+=1
        if iteration>2000:break
    (RUN/"exploration_checkpoint.json").write_text(json.dumps({
        "reduced_submissions":MAX_REDUCED-ledger.remaining("reduced"),
        "elapsed_s":elapsed(),"acceptable_count":sum(r.get("acceptable",False) for r in rows),
        "best_ids":[r["evaluation_id"] for r in sorted(rows,key=lambda r:r.get("loss",math.inf))[:10]]},indent=2)+"\n")
    shortlist=_distinct(rows,3)
    if not shortlist:
        shortlist=sorted(rows,key=lambda r:r.get("loss",math.inf))[:3]
    for candidate in shortlist:
        if ledger.remaining("spatial")<=0 or elapsed()>WALL_LIMIT_S-120:break
        x=np.asarray(candidate["x"])
        row=submit(ledger,evaluator,x,"spatial","nominal_confirmation",save=True)
        print("confirm",row["evaluation_id"],candidate["evaluation_id"],row.get("loss"),
              row.get("K_metrics",{}).get("rms"),row.get("S_metrics",{}).get("rms"),flush=True)
    (RUN/"ledger_audit.json").write_text(json.dumps(ledger.audit(),indent=2)+"\n")


if __name__=="__main__":main()
