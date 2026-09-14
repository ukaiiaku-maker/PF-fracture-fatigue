"""Sign-explicit analysis operators for the corrected v10.2.30 joint search.

This module is analysis-only.  It reuses the immutable v1 300 K reference
surface and does not modify a production entry point or material row.
"""
from __future__ import annotations
from dataclasses import asdict
import math
from typing import Mapping
import numpy as np
from scipy.special import gammainc

from arrhenius_fracture.material_manifest import KB_EV_PER_K
from scripts.thermodynamic_joint_barrier_v10230 import (
    Basis,EV_J,TR_K,ThermodynamicSurface,candidate_surface_from_parameters,
)

ENTROPY_REPRESENTATION="DIRECT_FREE_ENERGY_SURFACE_NO_PREFACTOR_DOUBLE_COUNTING"
HITS=3.2732414351776242
TAU_S=6.992153587194454e-7

def barrier_temperature_derivative_over_kB(surface,stress_Pa,temperature_K=TR_K):
    """Return (partial G/partial T)_sigma/k_B, explicitly opposite to S/k_B."""
    return -np.asarray(surface.entropy_kB(stress_Pa,temperature_K),dtype=float)

def sign_explicit_fields(surface,stress_Pa,temperature_K=TR_K):
    entropy=np.asarray(surface.entropy_kB(stress_Pa,temperature_K),dtype=float)
    derivative=barrier_temperature_derivative_over_kB(surface,stress_Pa,temperature_K)
    return {
        "activation_entropy_over_kB":entropy,
        "barrier_temperature_derivative_over_kB":derivative,
        "entropy_representation":ENTROPY_REPRESENTATION,
    }

def corrected_entropy_from_legacy(active_kB:float,stratum:str)->float:
    """Map v1 Sobol coordinates to the preregistered negative prior."""
    active=float(active_kB)
    if "PREFERRED" in str(stratum):
        return -50.0+0.5*(min(max(active,20.0),60.0)-20.0)
    return -30.0+1.5*min(max(active,0.0),20.0)

def paired_parameters(legacy:Mapping,corrected_candidate_id:str)->dict:
    p=dict(legacy)
    p.update(
        candidate_id=corrected_candidate_id,
        legacy_candidate_id=str(legacy["candidate_id"]),
        entropy_pair_id=f"PAIR_{legacy['candidate_id']}",
        legacy_emission_entropy_over_kB=float(legacy["emission_entropy_active_kB"]),
        corrected_emission_entropy_over_kB=corrected_entropy_from_legacy(
            legacy["emission_entropy_active_kB"],legacy.get("emission_stratum","")
        ),
        legacy_emission_barrier_temperature_derivative_over_kB=-float(legacy["emission_entropy_active_kB"]),
        entropy_representation=ENTROPY_REPRESENTATION,
        all_nonentropy_coordinates_equal=True,
        Tref_surface_equal=True,
    )
    p["emission_entropy_active_kB"]=p["corrected_emission_entropy_over_kB"]
    p["emission_entropy_infinity_kB"]=-abs(float(legacy["emission_entropy_infinity_kB"]))
    if "emission_entropy_zero_kB" in p and p["emission_entropy_zero_kB"] is not None:
        try:p["emission_entropy_zero_kB"]=-float(p["emission_entropy_zero_kB"])
        except (TypeError,ValueError):pass
    p["emission_stratum"]="NEGATIVE_PRIMARY_PRIOR" if -50<=p["emission_entropy_active_kB"]<=-30 else "NEGATIVE_TO_ZERO_CONTROL"
    return p

def serialize_surface(surface:ThermodynamicSurface)->dict:
    data=asdict(surface);data["bases"]=[asdict(x) for x in surface.bases]
    return data

def deserialize_surface(data:Mapping)->ThermodynamicSurface:
    d=dict(data);d["bases"]=tuple(Basis(**x) for x in d["bases"])
    for key in ("reference_amplitudes_eV","entropy_amplitudes_kB","heat_capacity_amplitudes_kB"):
        d[key]=tuple(d[key])
    return ThermodynamicSurface(**d)

def exact_Tref_equal(a,b,stress_Pa,tolerance_eV=2e-12):
    return bool(np.max(abs(np.asarray(a.G_eV(stress_Pa,TR_K))-np.asarray(b.G_eV(stress_Pa,TR_K))))<=tolerance_eV)

def finite_difference_audit(surface,stress_Pa,temperature_K):
    stress=np.asarray(stress_Pa,float);T=float(temperature_K);dt=.02;ds=2e4
    entropy_fd=-(surface.G_eV(stress,T+dt)-surface.G_eV(stress,T-dt))/(2*dt*KB_EV_PER_K)
    h=np.where(stress==0.,1.,ds);lo=np.maximum(stress-h,0.);hi=stress+h
    volume_fd=-(surface.G_eV(hi,T)-surface.G_eV(lo,T))/(hi-lo)*EV_J
    ss=stress[stress>ds]
    dV=(surface.activation_volume_m3(ss,T+dt)-surface.activation_volume_m3(ss,T-dt))/(2*dt)
    dS=(surface.entropy_kB(ss+ds,T)-surface.entropy_kB(ss-ds,T))/(2*ds)*KB_EV_PER_K*EV_J
    return {
        "entropy_max_abs_kB":float(np.max(abs(entropy_fd-surface.entropy_kB(stress,T)))),
        "volume_max_abs_m3":float(np.max(abs(volume_fd-surface.activation_volume_m3(stress,T)))),
        "maxwell_max_abs_m3_per_K":float(np.max(abs(dV-dS))) if len(ss) else 0.,
    }

def thermodynamic_rate_gate(surface,stress_Pa,temperatures,maximum_occupancy=.80,path_stress_Pa=None):
    stress=np.asarray(stress_Pa,float);records=[];all_valid=True
    path_stress=stress if path_stress_Pa is None else np.asarray(path_stress_Pa,float)
    for T in temperatures:
        G=np.asarray(surface.G_eV(stress,T),float);S=np.asarray(surface.entropy_kB(stress,T),float)
        V=np.asarray(surface.activation_volume_m3(stress,T),float);exponent=-G/(KB_EV_PER_K*T);lograte=math.log(surface.attempt_frequency_s)+exponent
        rate=np.exp(np.clip(lograte,-745.,700.))
        numerical_under=lograte<-745.;numerical_over=lograte>700.
        Gpath=np.asarray(surface.G_eV(path_stress,T),float);path_rate=np.exp(np.clip(math.log(surface.attempt_frequency_s)-Gpath/(KB_EV_PER_K*T),-745.,700.))
        constitutive=gammainc(HITS,np.minimum(path_rate*TAU_S,1e12))>=.99
        floor=np.isclose(Gpath,np.nanmin(G),rtol=0,atol=1e-6)
        floor_fraction=float(np.mean(floor));ceiling_fraction=float(np.mean(constitutive))
        valid=bool(np.all(np.isfinite(np.r_[G,S,V,rate])) and np.min(G)>0 and
                   not np.any(numerical_under) and not np.any(numerical_over) and
                   floor_fraction<maximum_occupancy and ceiling_fraction<maximum_occupancy)
        all_valid &= valid
        records.append(dict(temperature_K=float(T),minimum_G_eV=float(np.min(G)),maximum_G_eV=float(np.max(G)),fraction_of_path_at_floor=floor_fraction,fraction_of_path_at_ceiling=ceiling_fraction,elementary_rate_underflowed=bool(np.any(numerical_under)),elementary_rate_overflowed=bool(np.any(numerical_over)),constitutive_ceiling_occupied=bool(np.any(constitutive)),valid=valid))
    return all_valid,records

def classify_coupled(curve,fidelity:str,accessible_fraction:float,causal:Mapping)->str:
    if fidelity not in {"F1B_FULL_PRODUCTION_STATE","F2R_PRODUCTION_REDUCED_REPLAY"}:
        return f"{fidelity.split('_')[0]}_NO_SYSTEM_CLASS_PROMOTION"
    if accessible_fraction<.80 or not causal.get("all_class_anchors_valid",False):
        return "STATE_UNRESOLVED"
    K=np.asarray(curve,float);ratio=K[-1]/K[0];span=K.max()/K.min();d=np.diff(K)
    if ratio<=.70 and np.all(d<=0) and causal.get("opening_precedes_relaxation",False) and causal.get("emission_admissible",False):return "CERAMIC_LIKE_PROVISIONAL_COUPLED_MODEL_RESPONSE_CLASS"
    if span<=1.25 and causal.get("direct_state_cancellation",False):return "WEAK_T_PROVISIONAL_COUPLED_MODEL_RESPONSE_CLASS"
    imax=int(np.argmax(K))
    if 0<imax<len(K)-1 and K[imax]>=1.15*max(K[0],K[-1]) and np.all(d[:imax]>0) and np.all(d[imax:]<0) and causal.get("state_peak_contribution",False):return "PEAK_T_PROVISIONAL_COUPLED_MODEL_RESPONSE_CLASS"
    if ratio>=1.5 and causal.get("positive_interval_width_K",0)>=100 and causal.get("plastic_state_increase",False) and causal.get("state_transition_contribution_MPa_sqrt_m",0)>=.5 and causal.get("expected_rate_shift",False):return "DBTT_LIKE_PROVISIONAL_COUPLED_MODEL_RESPONSE_CLASS"
    return "ACCESSIBLE_UNCLASSIFIED_COUPLED_CONTROL"

__all__=["ENTROPY_REPRESENTATION","barrier_temperature_derivative_over_kB","sign_explicit_fields","corrected_entropy_from_legacy","paired_parameters","serialize_surface","deserialize_surface","exact_Tref_equal","finite_difference_audit","thermodynamic_rate_gate","classify_coupled","candidate_surface_from_parameters"]
