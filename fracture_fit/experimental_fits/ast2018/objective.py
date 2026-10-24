"""Block-normalized working objective for the supported Ast observables.

The scales are analyst tolerances, not inferred SDs. Every selected residual
slot remains present when a candidate terminates early or fails numerically.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from math import sqrt
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar

from .mechanics import continuous_onset_K, observe_like_ast
from .reduced import predict_reduced


PARAMETERS = (
    ("cleavage.G00_eV",0.3,7.0),
    ("cleavage.gT_eV_per_K",-0.015,0.015),
    ("cleavage.sigc0_Pa",1.5e9,15e9),
    ("emission.G00_eV",0.2,5.0),
    ("emission.gT_eV_per_K",-0.015,0.015),
    ("emission.sigc0_Pa",1.5e9,15e9),
    ("peierls.H0_eV",0.3,5.0),
)


def normalized_start(candidate) -> np.ndarray:
    vals = [candidate.cleavage.G00_eV,candidate.cleavage.gT_eV_per_K,
            candidate.cleavage.sigc0_Pa,candidate.emission.G00_eV,
            candidate.emission.gT_eV_per_K,candidate.emission.sigc0_Pa,
            candidate.peierls.H0_eV]
    return np.array([(v-lo)/(hi-lo) for v,(_,lo,hi) in zip(vals,PARAMETERS)])


def material_from_normalized(base, x):
    if len(x) != len(PARAMETERS) or not np.all(np.isfinite(x)) or np.any(np.asarray(x)<0) or np.any(np.asarray(x)>1):
        raise ValueError("parameter vector outside recorded bounds")
    values = [lo+(hi-lo)*float(v) for v,(_,lo,hi) in zip(x,PARAMETERS)]
    return replace(base,candidate_id="ast2018_trial",
        cleavage=replace(base.cleavage,G00_eV=values[0],gT_eV_per_K=values[1],
                         sigc0_Pa=values[2]),
        emission=replace(base.emission,G00_eV=values[3],gT_eV_per_K=values[4],
                         sigc0_Pa=values[5]),
        peierls=replace(base.peierls,H0_eV=values[6]))


def physically_admissible(candidate) -> bool:
    """Reject optimizer steps that rely on the source's numerical lower clips."""
    surfaces=(candidate.cleavage,candidate.emission,
              candidate.peierls.surface(candidate.emission),
              candidate.taylor.surface(candidate.emission))
    for surface in surfaces:
        for T in (183.15,773.15):
            if (surface.G00_eV+surface.gT_eV_per_K*(T-surface.Tref_K)<=1e-4 or
                surface.sigc0_Pa+surface.sT_Pa_per_K*(T-surface.Tref_K)<=1e6):
                return False
    return True


def _select_quantiles(rows: list[dict], n: int, key: str) -> list[dict]:
    if len(rows) <= n:
        return rows
    ordered = sorted(rows,key=lambda r:float(r[key]))
    indices = np.unique(np.round(np.linspace(0,len(ordered)-1,n)).astype(int))
    return [ordered[i] for i in indices]


def _quadrature_weights(x: np.ndarray) -> np.ndarray:
    if len(x)==1:
        return np.ones(1)
    x = np.maximum.accumulate(x)
    w = np.empty(len(x))
    w[0] = max(x[1]-x[0],1e-12)/2
    w[-1] = max(x[-1]-x[-2],1e-12)/2
    if len(x)>2:
        w[1:-1] = np.maximum(x[2:]-x[:-2],1e-12)/2
    return w/max(w.sum(),1e-30)


def robust_quadrature_residual(raw:np.ndarray,weights:np.ndarray,
                               family_fraction:float)->np.ndarray:
    """Map normalized pseudo-Huber block loss exactly to linear least squares."""
    raw=np.asarray(raw,dtype=float);weights=np.asarray(weights,dtype=float)
    if raw.shape!=weights.shape or np.any(weights<0) or family_fraction<=0:
        raise ValueError("invalid fixed quadrature")
    transformed=np.sign(raw)*np.sqrt(2*(np.hypot(1,raw)-1))
    return sqrt(family_fraction)*np.sqrt(weights)*transformed


@dataclass(frozen=True)
class Family:
    name: str
    curve_id: str
    values: tuple[dict, ...]
    weights: tuple[float, ...]


class WorkingObjective:
    def __init__(self,data,scenario,assumptions,zero,base_candidate,base_physics,
                 *, curve_points=25,trace_points=25,stiffness_points=20,
                 predictor=predict_reduced):
        self.data=data;self.scenario=scenario;self.assumptions=assumptions
        self.zero=zero;self.base_candidate=base_candidate;self.base_physics=base_physics
        self.predictor=predictor
        grouped: dict[str,list[Family]] = defaultdict(list)
        for row in data.toughness_scenario(scenario):
            if row["observable"]=="Kq_Jcontinuous_onset":
                grouped[row["condition_id"]].append(Family("onset",row["measurement_id"],
                    (row,),(1.0,)))
        curves=defaultdict(list)
        for row in data.J_curves:
            if row["sample_for_fit"].lower() in ("true","1","yes"):
                curves[row["curve_id"]].append(row)
        for cid,rs in curves.items():
            selected=_select_quantiles(rs,curve_points,"J_N_per_m")
            selected=sorted(selected,key=lambda r:float(r["J_N_per_m"]))
            weights=_quadrature_weights(np.array([float(r["J_N_per_m"]) for r in selected]))
            grouped[selected[0]["condition_id"]].append(Family("J_curve",cid,tuple(selected),tuple(weights)))
        curves=defaultdict(list)
        for row in data.K_displacement:
            if row["fit_role"]=="loading_history_candidate":
                curves[row["curve_id"]].append(row)
        for cid,rs in curves.items():
            selected=_select_quantiles(rs,trace_points,"displacement_um")
            selected=sorted(selected,key=lambda r:float(r["displacement_um"]))
            weights=_quadrature_weights(np.array([float(r["displacement_um"]) for r in selected]))
            grouped[selected[0]["condition_id"]].append(Family("nominal_K_trace",cid,tuple(selected),tuple(weights)))
        curves=defaultdict(list)
        for row in data.stiffness: curves[row["curve_id"]].append(row)
        for cid,rs in curves.items():
            selected=_select_quantiles(rs,stiffness_points,"displacement_nm")
            selected=sorted(selected,key=lambda r:float(r["displacement_nm"]))
            weights=_quadrature_weights(np.array([float(r["displacement_nm"]) for r in selected]))
            grouped[selected[0]["condition_id"]].append(Family("relative_stiffness",cid,tuple(selected),tuple(weights)))
        self.families={key:tuple(value) for key,value in grouped.items()}
        self.condition_ids=tuple(r["condition_id"] for r in data.conditions)
        self.n_residuals=sum(len(f.values) for ff in self.families.values() for f in ff)
        self.calls=0
        self.case_calls=0
        self.last_detail=[]
        self.progress_callback: Callable[[str,str,int],None]|None=None
        self.case_callback=None

    def case_targets(self,condition_id:str)->tuple[float,float]:
        families=self.families.get(condition_id,())
        all_curve_support=[float(r["crack_extension_nm"])*1e-9+.04e-6
                           for r in self.data.J_curves if r["condition_id"]==condition_id]
        target=max([.22e-6]+all_curve_support)
        required_u=max([0.0]+[max(float(r["displacement_um"]) for r in f.values)*1e-6+
                              (.3e-6 if f.values[0]["figure"]=="2a" else 0.0)
                              for f in families if f.name=="nominal_K_trace"]+
                       [max(float(r["displacement_nm"]) for r in f.values)*1e-9
                        for f in families if f.name=="relative_stiffness"])
        return target,required_u

    def evaluate(self,x,*,detail=False):
        self.calls+=1
        candidate=material_from_normalized(self.base_candidate,x)
        if not physically_admissible(candidate):
            penalty=np.full(self.n_residuals,50.0)
            return (penalty,[]) if detail else penalty
        residuals=[]; details=[]
        for condition_id in self.condition_ids:
            families=self.families.get(condition_id,())
            if not families: continue
            condition=self.data.condition(condition_id)
            specimen=self.assumptions.specimen(condition)
            physics=self.assumptions.physics(self.base_physics,specimen)
            target,required_u=self.case_targets(condition_id)
            result=self.predictor(self.zero,candidate,physics,specimen,
                    float(condition["temperature_K"]),
                    float(condition["initial_elastic_Kdot_MPa_sqrt_m_per_s"]),
                    self.assumptions,deterministic_threshold=True,
                    required_u_m=required_u,required_extension_m=target)
            self.case_calls+=1
            if self.progress_callback is not None:
                self.progress_callback(condition_id,result.status,self.case_calls)
            try: observed=observe_like_ast(result.history,specimen)
            except Exception: observed=[]
            if self.case_callback is not None:
                self.case_callback(condition_id,result,observed,specimen)
            for f in families:
                raw=self._family_residual(f,result,observed,specimen)
                # Pseudo-Huber block cost is sum_i q_i*(sqrt(1+r_i²)-1).
                # Robustifying after multiplication by sqrt(q) would change
                # the objective when the same raster line is sampled densely.
                scaled=robust_quadrature_residual(raw,np.asarray(f.weights),
                                                  1/len(families))
                residuals.extend(scaled.tolist())
                details.append({"condition_id":condition_id,"family":f.name,
                                "curve_id":f.curve_id,"status":result.status,
                                "rms_standardized":float(np.sqrt(np.mean(raw**2))),
                                "block_cost":float(.5*np.sum(scaled**2)),
                                "n_points":len(f.values)})
        if len(residuals)!=self.n_residuals:
            raise RuntimeError("frozen residual mask changed")
        self.last_detail=details
        return (np.array(residuals),details) if detail else np.array(residuals)

    def _family_residual(self,f:Family,result,observed,specimen):
        n=len(f.values)
        if result.status.startswith("numerical") or not observed:
            return np.full(n,50.0)
        if f.name=="onset":
            prediction=continuous_onset_K(observed,specimen)
            if not np.isfinite(prediction):
                gap=max(.2e-6-observed[-1].da_m,0)
                return np.array([10+gap/50e-9])
            target=float(f.values[0]["value"])
            return np.array([(prediction-target)/max(.2,.05*target)])
        if f.name=="J_curve":
            points=[p for p in observed if p.event]
            if not points:
                return np.full(n,20.0)
            J=np.array([p.J_continuous_N_per_m for p in points]);a=np.array([p.da_m for p in points])
            target_J=np.array([float(r["J_N_per_m"]) for r in f.values])
            target_a=np.array([float(r["crack_extension_nm"])*1e-9 for r in f.values])
            if np.all(np.diff(J)>-1e-9):
                pred=np.interp(target_J,J,a)
                residual=(pred-target_a)/50e-9
                beyond=target_J>max(J)
                residual[beyond]+=10+(target_J[beyond]-max(J))/max(10,.05*max(J))
                return residual
            # Ordered parametric curve distance for nonmonotone J histories.
            distance=np.hypot((target_J[:,None]-J[None,:])/max(10,.05*max(target_J)),
                              (target_a[:,None]-a[None,:])/50e-9)
            residual=np.min(distance,axis=1)
            residual[target_J>max(J)]+=10
            return residual
        if f.name=="nominal_K_trace":
            x=np.array([float(r["displacement_um"])*1e-6 for r in f.values])
            y=np.array([float(r["K_MPa_sqrt_m"]) for r in f.values])
            model_x=np.array([p.u_m for p in observed]);model_y=np.array([p.K_nominal_MPa_sqrt_m for p in observed])
            scale=.1
            def cost(shift):
                xp=x+shift
                pred=np.interp(xp,model_x,model_y)
                return float(np.mean(((pred-y)/scale)**2)+.01*(shift/.3e-6)**2)
            shift=0.0
            if f.values[0]["figure"]=="2a":
                shift=float(minimize_scalar(cost,bounds=(-.3e-6,.3e-6),method="bounded",
                                            options={"xatol":1e-10}).x)
            xp=x+shift
            pred=np.interp(xp,model_x,model_y)
            residual=(pred-y)/scale
            residual[xp>max(model_x)]+=10+(xp[xp>max(model_x)]-max(model_x))/0.1e-6
            residual[xp<0]+=10+(-xp[xp<0])/0.1e-6
            return residual
        if f.name=="relative_stiffness":
            x=np.array([float(r["displacement_nm"])*1e-9 for r in f.values])
            y=np.array([float(r["relative_stiffness"]) for r in f.values])
            model_x=np.array([p.u_m for p in observed]);model_y=np.array([p.relative_stiffness for p in observed])
            pred=np.interp(x,model_x,model_y)
            residual=(pred-y)/.03
            residual[x>max(model_x)]+=10+(x[x>max(model_x)]-max(model_x))/0.1e-6
            return residual
        raise ValueError(f.name)
