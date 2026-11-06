"""Fixed-mask, family-budgeted working objective for the supported Ast observables.

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

from .mechanics import continuous_onset_K, observe_like_ast, post_event_J_curve
from .opening_surface import IndependentFloorSurface, FiniteCutoffOpeningSurface
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
    if isinstance(candidate.cleavage,(IndependentFloorSurface,FiniteCutoffOpeningSurface)):
        try: candidate.cleavage.validate()
        except ValueError: return False
    surfaces=(candidate.cleavage,candidate.emission,
              candidate.peierls.surface(candidate.emission),
              candidate.taylor.surface(candidate.emission))
    for surface in surfaces:
        if isinstance(surface,(IndependentFloorSurface,FiniteCutoffOpeningSurface)):
            continue
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


def paired_working_residual(data_error:np.ndarray,coverage:np.ndarray,
                            weights:np.ndarray,family_fraction:float)->np.ndarray:
    """Independent robust slots; coverage cannot cancel a signed data error.

    Each slot has the same fixed quadrature weight. The data contribution keeps
    its previous normalization; nonnegative coverage is an additional cost.
    """
    data_error=np.asarray(data_error,dtype=float)
    coverage=np.asarray(coverage,dtype=float)
    weights=np.asarray(weights,dtype=float)
    if data_error.shape!=coverage.shape or data_error.shape!=weights.shape or np.any(coverage<0):
        raise ValueError("invalid paired residual slots")
    return robust_quadrature_residual(np.concatenate((data_error,coverage)),
                                      np.concatenate((weights,weights)),family_fraction)


def _domain_coverage(x:np.ndarray,lo:float,hi:float,scale:float)->np.ndarray:
    """Continuous, uncertainty-scaled distance outside measured model support."""
    if scale<=0 or hi<lo:
        raise ValueError("valid domain and positive working scale required")
    gap=np.maximum(np.maximum(lo-x,x-hi),0)
    return gap/scale


def _supported_data_residual(pred:np.ndarray,target:np.ndarray,
                             coverage:np.ndarray,scale:float)->np.ndarray:
    """Keep fixed slots without calling flat interpolation a prediction."""
    return np.where(coverage<=0,(pred-target)/scale,0.0)


def J_curve_working_errors(target_a,target_J,model_a,model_J,weights,
                           J_scale,extension_scale,shape_weight=2.0):
    """One nonduplicated level/shape residual and continuous support shortfall.

    The centered and mean parts of the same physical J errors are reweighted
    in one vector. They are not appended as extra independent observations.
    """
    x=np.asarray(target_a,dtype=float);y=np.asarray(target_J,dtype=float)
    a=np.asarray(model_a,dtype=float);J=np.asarray(model_J,dtype=float)
    q=np.asarray(weights,dtype=float)
    if (not len(x) or len(a)==0 or np.any(x<0) or np.any(np.diff(a)<=0) or
        x.shape!=y.shape or x.shape!=q.shape or a.shape!=J.shape or
        J_scale<=0 or extension_scale<=0 or shape_weight<=0):
        raise ValueError("invalid J curve correspondence")
    q=q/q.sum()
    supported=x<=a[-1]+1e-15
    # Outside modeled crack-growth support there is no predicted J value.
    # Preserve those fixed slots through the separate coverage residual.
    physical=np.where(supported,(np.interp(x,a,J)-y)/J_scale,0.0)
    supported_weight=q*supported
    supported_weight_sum=float(np.sum(supported_weight))
    level=(float(np.dot(supported_weight,physical)/supported_weight_sum)
           if supported_weight_sum>0 else float("nan"))
    data=np.zeros_like(physical)
    if supported_weight_sum>0:
        data[supported]=np.sqrt(shape_weight)*(physical[supported]-level)+level
    coverage=np.maximum(x-a[-1],0.0)/extension_scale
    return data,coverage,{"level_standardized":level if np.isfinite(level) else None,
        "shape_rms_standardized":(float(np.sqrt(np.dot(supported_weight,
             (physical-np.nan_to_num(level))**2)/supported_weight_sum))
             if supported_weight_sum>0 else None),
        "raw_J_rms_N_per_m":(float(np.sqrt(np.dot(supported_weight,
             (physical*J_scale)**2)/supported_weight_sum))
             if supported_weight_sum>0 else None),
        "observed_max_extension_m":float(np.max(x)),
        "model_max_extension_m":float(a[-1]),
        "support_shortfall":float(np.max(coverage)),
        "supported_points":int(np.count_nonzero(supported)),
        "unsupported_points":int(np.count_nonzero(~supported))}


@dataclass(frozen=True)
class Family:
    name: str
    curve_id: str
    values: tuple[dict, ...]
    weights: tuple[float, ...]


class WorkingObjective:
    def __init__(self,data,scenario,assumptions,zero,base_candidate,base_physics,
                 *, curve_points=25,trace_points=25,stiffness_points=20,
                 predictor=predict_reduced,candidate_factory=None,
                 admissibility=None,family_budget="equal",
                 condition_subset=None,block_weight_policy="family_global",
                 specimen_factory=None,family_filter=None):
        self.data=data;self.scenario=scenario;self.assumptions=assumptions
        self.zero=zero;self.base_candidate=base_candidate;self.base_physics=base_physics
        self.predictor=predictor
        self.specimen_factory=specimen_factory or assumptions.specimen
        self.candidate_factory=candidate_factory or material_from_normalized
        self.admissibility=admissibility or physically_admissible
        budgets={"equal":{"J_curve":.25,"onset":.25,
                          "nominal_K_trace":.25,"relative_stiffness":.25},
                 "fracture_priority":{"J_curve":.40,"onset":.30,
                                      "nominal_K_trace":.20,"relative_stiffness":.10}}
        if family_budget not in budgets:
            raise ValueError("unknown declared observable-family budget")
        self.family_budget_name=family_budget
        self.family_budget=budgets[family_budget]
        if block_weight_policy not in ("family_global","rate_condition"):
            raise ValueError("unknown block weight policy")
        self.block_weight_policy=block_weight_policy
        self.J_shape_weight=1.0 if family_budget=="equal" else 2.0
        self.onset_contrast_weight=1.0 if family_budget=="equal" else 2.0
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
            # Negative/zero digitized starts are uncertain origins, not healing.
            # Fit reliable positive extension nodes with a fixed physical mask.
            usable=[r for r in rs if float(r["crack_extension_nm"])>0]
            selected=_select_quantiles(usable,curve_points,"crack_extension_nm")
            selected=sorted(selected,key=lambda r:float(r["crack_extension_nm"]))
            weights=_quadrature_weights(np.array([float(r["crack_extension_nm"]) for r in selected]))
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
        allowed=(set(condition_subset) if condition_subset is not None else
                 {r["condition_id"] for r in data.conditions})
        if not allowed.issubset({r["condition_id"] for r in data.conditions}):
            raise ValueError("unknown condition in screening subset")
        self.families={key:tuple(f for f in value if family_filter is None or
                                    family_filter(key,f))
                       for key,value in grouped.items() if key in allowed}
        self.condition_ids=tuple(r["condition_id"] for r in data.conditions
                                 if r["condition_id"] in allowed)
        self.family_curve_counts={name:sum(f.name==name for fs in self.families.values()
                                           for f in fs) for name in self.family_budget}
        rates={r["condition_id"]:r.get("initial_elastic_Kdot_MPa_sqrt_m_per_s","unspecified")
               for r in data.conditions}
        active=[cid for cid in self.condition_ids if self.families.get(cid)]
        rate_counts={rate:sum(rates[cid]==rate for cid in active)
                     for rate in {rates[cid] for cid in active}}
        self.block_fractions={
            (cid,f.curve_id):1.0/(len(rate_counts)*rate_counts[rates[cid]]*
                                   len(self.families[cid]))
            for cid in active for f in self.families[cid]}
        self.n_residuals=2*sum(len(f.values) for ff in self.families.values() for f in ff)
        self.calls=0
        self.case_calls=0
        self.last_detail=[]
        self.progress_callback: Callable[[str,str,int],None]|None=None
        self.case_callback=None

    def case_targets(self,condition_id:str)->tuple[float,float]:
        families=self.families.get(condition_id,())
        # Loading/curve records provide a finite experimental displacement
        # horizon. Never keep loading only to force observed crack extension.
        has_loading_window=any(f.name in ("nominal_K_trace",
                                          "relative_stiffness") for f in families)
        if has_loading_window:
            target=0.0
        else:
            J_extensions=[max(float(r["crack_extension_nm"]) for r in f.values)*1e-9
                          for f in families if f.name=="J_curve"]
            # A resistance-only specimen has no measured displacement horizon.
            # Run to its observed extension, with the existing nominal-K cap.
            target=max(J_extensions) if J_extensions else .22e-6
        required_u=max([0.0]+[max(float(r["displacement_um"]) for r in f.values)*1e-6+
                              (.3e-6 if f.values[0]["figure"]=="2a" else 0.0)
                              for f in families if f.name=="nominal_K_trace"]+
                       [max(float(r["displacement_nm"]) for r in f.values)*1e-9
                        for f in families if f.name=="relative_stiffness"])
        return target,required_u

    @staticmethod
    def _J_scales(f:Family)->tuple[float,float]:
        energies=np.array([float(row["J_N_per_m"]) for row in f.values])
        vertical=[float(row.get("J_digitization_pm_N_per_m",0) or 0)
                  for row in f.values]
        horizontal=[float(row.get("crack_extension_digitization_pm_nm",0) or 0)
                    for row in f.values]
        J_scale=max(20.0,.05*float(np.ptp(energies)),float(np.median(vertical)))
        extension_scale=max(50e-9,float(np.median(horizontal))*1e-9)
        return J_scale,extension_scale

    @staticmethod
    def _J_extension_scale(f:Family)->float:
        return WorkingObjective._J_scales(f)[1]

    def evaluate(self,x,*,detail=False):
        self.calls+=1
        candidate=self.candidate_factory(self.base_candidate,x)
        if not self.admissibility(candidate):
            penalty=np.full(self.n_residuals,50.0)
            return (penalty,[]) if detail else penalty
        blocks=[]
        for condition_id in self.condition_ids:
            families=self.families.get(condition_id,())
            if not families: continue
            condition=self.data.condition(condition_id)
            specimen=self.specimen_factory(condition)
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
                blocks.append((condition_id,f,result,raw,observed,specimen))
        # The onset contrast operator acts on the same 23 observations, not
        # extra pseudo-observations. Centering emphasizes temperature/rate
        # differences while preserving one level direction and the fixed mask.
        onsets=[(index,raw) for index,(_,f,_,raw,_,_) in enumerate(blocks)
                if f.name=="onset" and raw[1]==0]
        if onsets:
            mean=float(np.mean([raw[0] for _,raw in onsets]))
            for index,raw in onsets:
                transformed=raw.copy()
                transformed[0]=np.sqrt(self.onset_contrast_weight)*(raw[0]-mean)+mean
                condition_id,f,result,_,observed,specimen=blocks[index]
                blocks[index]=(condition_id,f,result,transformed,observed,specimen)
        residuals=[];details=[]
        for condition_id,f,result,raw,observed,specimen in blocks:
                # Pseudo-Huber loss is applied before quadrature weighting so
                # dense raster sampling does not create new experiments.
                n=len(f.values)
                family_fraction=(self.block_fractions[(condition_id,f.curve_id)]
                                 if self.block_weight_policy=="rate_condition" else
                                 self.family_budget[f.name]/self.family_curve_counts[f.name])
                scaled=paired_working_residual(raw[:n],raw[n:],np.asarray(f.weights),
                                               family_fraction)
                residuals.extend(scaled.tolist())
                record={"condition_id":condition_id,"family":f.name,
                                "curve_id":f.curve_id,"status":result.status,
                                "rms_standardized":(float(np.sqrt(np.mean(raw[:n][raw[n:]<=0]**2)))
                                    if np.any(raw[n:]<=0) else None),
                                "uncovered_points":int(np.count_nonzero(raw[n:])),
                                "supported_points":int(np.count_nonzero(raw[n:]<=0)),
                                "observation_supported":bool(np.all(raw[n:]<=0)),
                                "misfit_defined":bool(np.any(raw[n:]<=0)),
                                "data_cost":float(.5*np.sum(scaled[:n]**2)),
                                "coverage_cost":float(.5*np.sum(scaled[n:]**2)),
                                "block_cost":float(.5*np.sum(scaled**2)),
                                "n_points":len(f.values),
                                "family_budget":self.family_budget[f.name],
                                "curve_weight":family_fraction}
                if f.name=="J_curve":
                    observed_max=max(float(row["crack_extension_nm"])*1e-9
                                     for row in f.values)
                    model_max=float(result.history[-1].a_m-
                                    self.assumptions.a0_m) if result.history else 0.0
                    extension_scale=self._J_extension_scale(f)
                    record.update(observed_max_extension_m=observed_max,
                                  model_max_extension_m=model_max,
                                  extension_working_scale_m=extension_scale,
                                  support_shortfall=max(observed_max-model_max,0.0)/
                                                    extension_scale,
                                  n_events=len(result.events))
                    if observed:
                        a,J=post_event_J_curve(observed)
                        target_a=np.array([float(row["crack_extension_nm"])*1e-9
                                           for row in f.values])
                        target_J=np.array([float(row["J_N_per_m"]) for row in f.values])
                        J_scale,_=self._J_scales(f)
                        _,_,shape=J_curve_working_errors(target_a,target_J,a,J,
                              np.asarray(f.weights),J_scale,extension_scale,
                              self.J_shape_weight)
                        record.update(shape)
                        record.update(observed_J_rise_N_per_m=float(target_J[-1]-target_J[0]),
                                      predicted_J_rise_N_per_m=(float(np.interp(target_a[-1],a,J)-
                                                                      np.interp(target_a[0],a,J))
                                         if target_a[-1]<=a[-1]+1e-15 else None))
                if f.name=="onset":
                    prediction=continuous_onset_K(observed,specimen) if observed else float("nan")
                    record.update(observed_onset_K_MPa_sqrt_m=float(f.values[0]["value"]),
                                  predicted_onset_K_MPa_sqrt_m=(float(prediction) if
                                  np.isfinite(prediction) else None),
                                  temperature_C=float(f.values[0]["temperature_C"]),
                                  initial_rate=float(f.values[0]["initial_elastic_Kdot_MPa_sqrt_m_per_s"]))
                details.append(record)
        if len(residuals)!=self.n_residuals:
            raise RuntimeError("frozen residual mask changed")
        self.last_detail=details
        return (np.array(residuals),details) if detail else np.array(residuals)

    def _family_residual(self,f:Family,result,observed,specimen):
        n=len(f.values)
        if result.status.startswith("numerical") or not observed:
            return np.concatenate((np.zeros(n),np.full(n,50.0)))
        if f.name=="onset":
            prediction=continuous_onset_K(observed,specimen)
            if not np.isfinite(prediction):
                gap=max(.2e-6-observed[-1].da_m,0)
                return np.array([0.0,gap/50e-9])
            target=float(f.values[0]["value"])
            return np.array([(prediction-target)/max(.2,.05*target),0.0])
        if f.name=="J_curve":
            a,J=post_event_J_curve(observed)
            target_J=np.array([float(r["J_N_per_m"]) for r in f.values])
            target_a=np.array([float(r["crack_extension_nm"])*1e-9 for r in f.values])
            J_scale,extension_scale=WorkingObjective._J_scales(f)
            shape_weight=getattr(self,"J_shape_weight",2.0)
            data,coverage,_=J_curve_working_errors(target_a,target_J,a,J,
                    np.asarray(f.weights),J_scale,extension_scale,shape_weight)
            return np.concatenate((data,coverage))
        if f.name=="nominal_K_trace":
            x=np.array([float(r["displacement_um"])*1e-6 for r in f.values])
            y=np.array([float(r["K_MPa_sqrt_m"]) for r in f.values])
            model_x=np.array([p.u_m for p in observed]);model_y=np.array([p.K_nominal_MPa_sqrt_m for p in observed])
            scale=.1
            def residual_at(shift):
                xp=x+shift
                pred=np.interp(xp,model_x,model_y)
                coverage=_domain_coverage(xp,float(model_x[0]),float(model_x[-1]),.1e-6)
                data=_supported_data_residual(pred,y,coverage,scale)
                return data,coverage
            def cost(shift):
                data,coverage=residual_at(shift)
                scaled=paired_working_residual(data,coverage,np.asarray(f.weights),1.0)
                return float(.5*np.sum(scaled**2))
            shift=0.0
            if f.values[0]["figure"]=="2a":
                shift=float(minimize_scalar(cost,bounds=(-.3e-6,.3e-6),method="bounded",
                                            options={"xatol":1e-10}).x)
            data,coverage=residual_at(shift)
            return np.concatenate((data,coverage))
        if f.name=="relative_stiffness":
            x=np.array([float(r["displacement_nm"])*1e-9 for r in f.values])
            y=np.array([float(r["relative_stiffness"]) for r in f.values])
            model_x=np.array([p.u_m for p in observed]);model_y=np.array([p.relative_stiffness for p in observed])
            pred=np.interp(x,model_x,model_y)
            coverage=_domain_coverage(x,float(model_x[0]),float(model_x[-1]),.1e-6)
            data=_supported_data_residual(pred,y,coverage,.03)
            return np.concatenate((data,coverage))
        raise ValueError(f.name)
