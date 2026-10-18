"""Mechanical and mixed cohesive opening modes for the additive R10 family.

The legacy cooperative Arrhenius clock remains in ``cohesive_arrhenius``.  This
module deliberately does not import or evaluate that completion law.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.linalg import eigh_tridiagonal

from .cohesive_arrhenius import (
    BILINEAR, COMPACT_INDEPENDENT, SMOOTH_EXPONENTIAL, BilinearCohesiveLaw,
    CompactIndependentCohesiveLaw, CohesiveThermalLaw,
    EV_TO_J, cohesive_shape_response, finite_compliance_landscape,
    mechanical_completion,
)
from .material_manifest import KB_EV_PER_K

BRANCH_STATES = frozenset({
    "BONDED_STABLE", "BONDED_METASTABLE", "STABLE_SOFTENING",
    "BONDED_BRANCH_LOST", "POST_ESCAPE_TRANSIT", "SEPARATION_COMPLETED",
    "CLOSED_CONTACT", "NUMERICALLY_UNRESOLVED",
})


@dataclass(frozen=True)
class OpeningAlternative:
    opening_mode: str
    cohesive_shape: str
    dynamics: str
    activation_area_m2: float | None = None
    attempt_frequency_s: float | None = None
    high_barrier_min_kBT: float = 5.0

    def __post_init__(self):
        if self.opening_mode not in {"cohesive_athermal", "cohesive_mixed"}:
            raise ValueError("opening_mode must be cohesive_athermal or cohesive_mixed")
        if self.cohesive_shape not in {SMOOTH_EXPONENTIAL, BILINEAR,
                                       COMPACT_INDEPENDENT}:
            raise ValueError("unsupported cohesive shape")
        expected = ({"quasistatic_mechanical"} if self.opening_mode == "cohesive_athermal"
                    else {"qualified_high_barrier_hazard","time_dependent_first_passage"})
        if self.dynamics not in expected:
            raise ValueError(f"{self.opening_mode} requires one of {sorted(expected)}")
        if self.opening_mode == "cohesive_mixed":
            if not self.activation_area_m2 or not self.attempt_frequency_s:
                raise ValueError("mixed opening requires area and attempt frequency")


def _root_landscape(imposed: float, kenv: float, thermal: CohesiveThermalLaw,
                    temperature: float, shape: str) -> list[dict[str, float | str]]:
    gamma = thermal.gamma(temperature)
    if shape == BILINEAR:
        law=BilinearCohesiveLaw(thermal);dp=law.peak_separation(temperature)
        df=law.final_separation(temperature);peak=thermal.t_hat(temperature)
        candidates=[]
        d1=kenv*imposed/(peak/dp+kenv)
        if 0.0<=d1<=dp:candidates.append(d1)
        denom=kenv-peak/(df-dp)
        if abs(denom)>np.finfo(float).eps*max(kenv,peak/(df-dp)):
            d2=(kenv*imposed-peak*df/(df-dp))/denom
            if dp<=d2<=df:candidates.append(d2)
        if imposed>=df:candidates.append(imposed)
        roots=sorted(set(candidates))
    else:
        compact_law = (CompactIndependentCohesiveLaw(thermal)
                       if shape == COMPACT_INDEPENDENT else None)
        compact_span = (compact_law.final_separation(temperature)
                        if compact_law is not None else 25.0*thermal.delta0_m)
        span = max(compact_span*1.05, imposed+10.0*thermal.delta0_m)
        if (compact_law is not None and thermal.compact_variant in
                {"rounded_plateau","two_scale_tail"}):
            dc=compact_law.final_separation(temperature)
            rise=thermal.compact_rise_fraction*dc
            fall=(thermal.compact_fall_fraction if thermal.compact_variant=="rounded_plateau"
                  else thermal.compact_drop_fraction)*dc
            grid=np.unique(np.r_[np.linspace(0.,rise,129),
                np.linspace(rise,fall,257),np.linspace(fall,dc,129),
                np.linspace(dc,span,65)])
        else:
            grid = np.linspace(0.0, span, 257)

    def residual(d):
        t = float(cohesive_shape_response(shape,d,thermal,temperature)[0])
        return t + kenv*(d-imposed)

    if shape != BILINEAR:
        vals = np.array([residual(float(d)) for d in grid])
        roots = []
        for i in range(len(grid)-1):
            if vals[i] == 0.0:
                roots.append(float(grid[i]))
            if vals[i]*vals[i+1] < 0.0:
                roots.append(float(brentq(residual,float(grid[i]),float(grid[i+1]))))
        if vals[-1] == 0.0:
            roots.append(float(grid[-1]))
        roots = sorted({round(x, 18) for x in roots})
    out=[]
    for d in roots:
        _, tangent, work = cohesive_shape_response(shape,d,thermal,temperature)
        curvature=float(tangent)+kenv
        energy=float(work)+0.5*kenv*(imposed-d)**2
        out.append({"delta_m":d,"energy_J_m2":energy,
                    "curvature_Pa_per_m":curvature,
                    "kind":"MINIMUM" if curvature>0.0 else "SADDLE"})
    return out


def connected_landscape(imposed_delta_m: float, environment_stiffness_Pa_per_m: float,
                        thermal: CohesiveThermalLaw, temperature_K: float,
                        shape: str, prior: dict[str, Any] | None = None) -> dict[str, Any]:
    """Follow a connected cohesive equilibrium and expose branch loss explicitly."""
    imposed=float(imposed_delta_m);kenv=float(environment_stiffness_Pa_per_m)
    if kenv <= 0.0:
        raise ValueError("environment stiffness must be positive")
    if imposed < 0.0:
        return {"state":"CLOSED_CONTACT","connected":None,"barrier_J_m2":None,
                "stationary":[],"connected_state":{"branch":"CLOSED","delta_m":0.0}}
    if shape==SMOOTH_EXPONENTIAL:
        old_branch=(prior or {}).get("branch","BONDED")
        legacy={"branch":"SEPARATED" if old_branch in {"TRANSIT","SEPARATED"} else "INTACT"}
        raw=finite_compliance_landscape(imposed,kenv,thermal,temperature_K,legacy)
        connected=raw.get("connected")
        completion=(mechanical_completion(shape,float(connected["delta_m"]),thermal,temperature_K)
                    if connected is not None else None)
        mapping={"METASTABLE":"BONDED_METASTABLE","NO_ESCAPE_SADDLE":"BONDED_STABLE",
                 "CONVEX_CONNECTED":"BONDED_STABLE",
                 "INTACT_BRANCH_LOST_TO_SEPARATED":"BONDED_BRANCH_LOST",
                 "SEPARATED_BRANCH":"POST_ESCAPE_TRANSIT",
                 "NUMERICAL_ROOT_FAILURE":"NUMERICALLY_UNRESOLVED"}
        state=mapping[raw["state"]]
        if completion and completion["complete"]:state="SEPARATION_COMPLETED"
        branch=("SEPARATED" if state=="SEPARATION_COMPLETED" else "TRANSIT" if state in
                {"BONDED_BRANCH_LOST","POST_ESCAPE_TRANSIT"} else "METASTABLE" if
                state=="BONDED_METASTABLE" else "BONDED")
        return {"state":state,"connected":connected,"barrier_J_m2":raw["barrier_J_m2"],
                "stationary":raw["stationary"],"completion":completion,
                "connected_state":{"branch":branch,"delta_m":None if connected is None else connected["delta_m"]},
                "native_landscape_state":raw["state"]}
    points=_root_landscape(imposed,kenv,thermal,temperature_K,shape)
    minima=[p for p in points if p["kind"]=="MINIMUM"]
    saddles=[p for p in points if p["kind"]=="SADDLE"]
    if not minima:
        return {"state":"NUMERICALLY_UNRESOLVED","connected":None,
                "barrier_J_m2":None,"stationary":points,
                "connected_state":{"branch":"UNRESOLVED","delta_m":None}}
    prior=prior or {"branch":"BONDED","delta_m":0.0}
    prior_d=float(prior.get("delta_m") or 0.0)
    connected=min(minima,key=lambda p:abs(float(p["delta_m"])-prior_d))
    peak_d=(thermal.delta0_m if shape==SMOOTH_EXPONENTIAL else
            BilinearCohesiveLaw(thermal).peak_separation(temperature_K)
            if shape==BILINEAR else
            CompactIndependentCohesiveLaw(thermal).peak_separation(temperature_K))
    lost=(prior.get("branch") in {"BONDED","METASTABLE"} and
          float(connected["delta_m"])>peak_d and prior_d<=peak_d and len(minima)==1)
    completion=mechanical_completion(shape,float(connected["delta_m"]),thermal,temperature_K)
    later=[p for p in saddles if float(p["delta_m"])>float(connected["delta_m"])]
    barrier=(float(later[0]["energy_J_m2"])-float(connected["energy_J_m2"])
             if later else None)
    if completion["complete"]:
        state="SEPARATION_COMPLETED";branch="SEPARATED"
    elif lost:
        state="BONDED_BRANCH_LOST";branch="TRANSIT"
    elif prior.get("branch")=="TRANSIT":
        state="POST_ESCAPE_TRANSIT";branch="TRANSIT"
    elif barrier is not None and barrier > 0.0:
        state="BONDED_METASTABLE";branch="METASTABLE"
    elif float(connected["delta_m"])>peak_d:
        state="STABLE_SOFTENING";branch="BONDED"
    else:
        state="BONDED_STABLE";branch="BONDED"
    return {"state":state,"connected":connected,"barrier_J_m2":barrier,
            "stationary":points,"completion":completion,
            "connected_state":{"branch":branch,"delta_m":connected["delta_m"]}}


def overdamped_mfpt(alternative: OpeningAlternative, imposed_delta_m: float,
                    environment_stiffness_Pa_per_m: float,
                    thermal: CohesiveThermalLaw, temperature_K: float,
                    connected: dict[str, Any], saddle: dict[str, Any],
                    grid_points: int = 1025) -> dict[str, float]:
    """Static reflecting/absorbing MFPT on the same cohesive coordinate.

    The friction is not fitted: the declared microscopic attempt frequency owns
    the harmonic relaxation, ``zeta=A*k0/nu0``.  Cyclic use is allowed only
    when the caller establishes adiabatic separation from its phase duration.
    """
    area=float(alternative.activation_area_m2);nu=float(alternative.attempt_frequency_s)
    k0=(CompactIndependentCohesiveLaw(thermal).initial_stiffness(temperature_K)
        if alternative.cohesive_shape == COMPACT_INDEPENDENT
        else thermal.initial_stiffness(temperature_K))
    zeta=area*k0/nu
    x0=float(connected["delta_m"]);xb=float(saddle["delta_m"])
    if not (0.0<=x0<xb):
        raise ValueError("MFPT requires an ordered connected minimum and saddle")
    x=np.linspace(0.0,xb,max(int(grid_points),129))
    _,_,psi=cohesive_shape_response(alternative.cohesive_shape,x,thermal,temperature_K)
    U=area*(np.asarray(psi)+0.5*environment_stiffness_Pa_per_m*(imposed_delta_m-x)**2)
    beta=1.0/(KB_EV_PER_K*EV_TO_J*temperature_K);shift=float(np.min(U))
    em=np.exp(np.clip(-beta*(U-shift),-700.,700.))
    inner=np.r_[0.,np.cumsum(0.5*(em[1:]+em[:-1])*np.diff(x))]
    ep=np.exp(np.clip(beta*(U-shift),-700.,700.))
    integrand=ep*inner
    start=int(np.searchsorted(x,x0));integral=float(np.trapezoid(integrand[start:],x[start:]))
    mfpt=zeta*beta*integral
    relaxation=zeta/(area*max(k0+environment_stiffness_Pa_per_m,1e-300))
    return {"mfpt_s":mfpt,"mfpt_rate_s-1":0.0 if mfpt<=0.0 else 1.0/mfpt,
            "friction_J_s_m-2":zeta,"harmonic_relaxation_s":relaxation,
            "reflecting_boundary_m":0.0,"absorbing_boundary_m":xb,
            "grid_points":len(x)}


def frozen_commitment_density_reference(
    alternative: OpeningAlternative, imposed_delta_m: float,
    environment_stiffness_Pa_per_m: float,
    thermal: CohesiveThermalLaw, temperature_K: float,
    landscape: dict[str, Any], duration_s: float,
    grid_points: int = 129,
) -> dict[str, Any]:
    """Frozen-landscape overdamped density reference to committed separation.

    The left boundary is reflecting contact.  The right boundary is the
    existing geometry-owned cohesive completion cutoff, not the saddle.  The
    initial density is the local Boltzmann distribution in the connected basin
    through its first later saddle.  This is a reference calculation, not a
    claim that a changing cyclic landscape is frozen.
    """
    if alternative.opening_mode != "cohesive_mixed":
        raise ValueError("density reference requires the mixed opening mode")
    connected=landscape.get("connected")
    if connected is None:
        raise ValueError("density reference requires a connected minimum")
    completion=mechanical_completion(
        alternative.cohesive_shape,0.,thermal,temperature_K)
    cutoff=float(completion["cutoff_m"]);d0=thermal.delta0_m
    n=max(int(grid_points),33);y=np.linspace(0.,cutoff/d0,n);h=float(y[1]-y[0])
    x=y*d0;area=float(alternative.activation_area_m2)
    _,_,psi=cohesive_shape_response(
        alternative.cohesive_shape,x,thermal,temperature_K)
    U=area*(np.asarray(psi)+0.5*environment_stiffness_Pa_per_m*
            (imposed_delta_m-x)**2)
    kBT=KB_EV_PER_K*EV_TO_J*temperature_K;V=(U-float(np.min(U)))/kBT
    k0=(CompactIndependentCohesiveLaw(thermal).initial_stiffness(temperature_K)
        if alternative.cohesive_shape == COMPACT_INDEPENDENT
        else thermal.initial_stiffness(temperature_K))
    zeta=area*k0/float(alternative.attempt_frequency_s)
    diffusion_y=kBT/(zeta*d0*d0)
    # The last node is absorbing; the transient chain contains nodes 0..n-2.
    m=n-1;forward=np.zeros(m);backward=np.zeros(m)
    for i in range(m):
        if i+1<n:
            forward[i]=diffusion_y/h**2*math.exp(float(np.clip(
                -0.5*(V[i+1]-V[i]),-80.,80.)))
        if i>0:
            backward[i]=diffusion_y/h**2*math.exp(float(np.clip(
                -0.5*(V[i-1]-V[i]),-80.,80.)))
    diagonal=-(forward+backward)
    offdiag=np.sqrt(forward[:-1]*backward[1:])
    eigenvalues,eigenvectors=eigh_tridiagonal(diagonal,offdiag)
    saddles=[p for p in landscape.get("stationary",[]) if p.get("kind")=="SADDLE"
             and float(p["delta_m"])>float(connected["delta_m"])]
    basin_end=float(saddles[0]["delta_m"])/d0 if saddles else float(connected["delta_m"])/d0
    initial=np.where(y[:m]<=basin_end+1e-14,np.exp(np.clip(-V[:m],-700.,700.)),0.)
    if float(np.sum(initial))<=0.:
        initial[int(np.argmin(abs(y[:m]-float(connected["delta_m"])/d0)))]=1.
    initial/=float(np.sum(initial))
    # Q = D S D^-1, with D=sqrt(local reversible Boltzmann weights).
    pi=np.exp(np.clip(-V[:m],-700.,700.));sqrt_pi=np.sqrt(pi)
    transformed=initial/np.maximum(sqrt_pi,1e-300)
    coefficients=eigenvectors.T@transformed
    evolved=sqrt_pi*(eigenvectors@(np.exp(np.clip(eigenvalues*duration_s,-745.,0.))*coefficients))
    evolved=np.maximum(np.real(evolved),0.);survival=float(np.clip(np.sum(evolved),0.,1.))
    spectral_scale=max(float(np.max(np.abs(eigenvalues))),1.)
    mean_resolved=bool(float(np.max(eigenvalues)) < -1e-13*spectral_scale)
    mean=(float(np.sum(sqrt_pi*(eigenvectors@((-1./eigenvalues)*coefficients))))
          if mean_resolved else None)
    return {"reference_kind":"FROZEN_COMMITMENT_DENSITY",
            "initialization":"LOCAL_BOLTZMANN_CONNECTED_BASIN_THROUGH_FIRST_SADDLE",
            "reflecting_boundary_m":0.,"absorbing_commitment_boundary_m":cutoff,
            "friction_J_s_m-2":zeta,"prefactor_relation":"zeta=A*k0/nu0",
            "duration_s":float(duration_s),"survival_probability":survival,
            "committed_probability":1.-survival,"mean_commitment_time_s":mean,
            "mean_commitment_time_numerically_resolved":mean_resolved,
            "grid_points":n,"saddle_is_absorbing":False}


def evaluate_opening(alternative: OpeningAlternative, imposed_delta_m: float,
                     environment_stiffness_Pa_per_m: float,
                     thermal: CohesiveThermalLaw, temperature_K: float,
                     prior: dict[str, Any] | None = None,
                     phase_duration_s: float | None = None) -> dict[str, Any]:
    landscape=connected_landscape(imposed_delta_m,environment_stiffness_Pa_per_m,
                                  thermal,temperature_K,
                                  alternative.cohesive_shape,prior)
    result={**landscape,"opening_mode":alternative.opening_mode,
            "cohesive_shape":alternative.cohesive_shape,
            "dynamics":alternative.dynamics,"elementary_rate_s-1":None,
            "kinetic_validity":"NOT_APPLICABLE_ATHERMAL"}
    if alternative.opening_mode=="cohesive_athermal":
        return result
    density=landscape.get("barrier_J_m2")
    if landscape["state"] in {"BONDED_BRANCH_LOST","POST_ESCAPE_TRANSIT",
                               "SEPARATION_COMPLETED"}:
        result["kinetic_validity"]="MECHANICAL_CONTINUATION"
        return result
    if density is None or density <= 0.0:
        result["kinetic_validity"]="NO_METASTABLE_ESCAPE_PATH"
        return result
    barrier_J=float(density)*float(alternative.activation_area_m2)
    barrier_kBT=barrier_J/(KB_EV_PER_K*EV_TO_J*temperature_K)
    result.update({"barrier_J":barrier_J,"barrier_eV":barrier_J/EV_TO_J,
                   "barrier_kBT":barrier_kBT})
    if barrier_kBT < alternative.high_barrier_min_kBT:
        if alternative.dynamics != "time_dependent_first_passage" or phase_duration_s is None:
            result["kinetic_validity"]="CROSSOVER_REQUIRES_FIRST_PASSAGE"
            return result
        connected=landscape.get("connected")
        later=[p for p in landscape.get("stationary",[]) if p.get("kind")=="SADDLE"
               and connected is not None and float(p["delta_m"])>float(connected["delta_m"])]
        if connected is None or not later:
            result["kinetic_validity"]="CROSSOVER_REQUIRES_FIRST_PASSAGE"
            return result
        mfpt=overdamped_mfpt(alternative,imposed_delta_m,environment_stiffness_Pa_per_m,
                             thermal,temperature_K,connected,later[0])
        ratio=mfpt["harmonic_relaxation_s"]/max(float(phase_duration_s),1e-300)
        result.update(mfpt);result["adiabatic_relaxation_to_phase_ratio"]=ratio
        if ratio>1e-3:
            result["kinetic_validity"]="NONADIABATIC_DISTRIBUTION_PROPAGATION_REQUIRED"
            return result
        result["elementary_rate_s-1"]=mfpt["mfpt_rate_s-1"]
        result["kinetic_validity"]="STATIC_SADDLE_MFPT_HAZARD_APPROXIMATION"
        return result
    result["elementary_rate_s-1"]=(float(alternative.attempt_frequency_s)
        *math.exp(max(-barrier_kBT,-700.0)))
    result["kinetic_validity"]="QUALIFIED_HIGH_BARRIER"
    return result


__all__=["BRANCH_STATES","OpeningAlternative","connected_landscape",
         "evaluate_opening","frozen_commitment_density_reference","overdamped_mfpt"]
