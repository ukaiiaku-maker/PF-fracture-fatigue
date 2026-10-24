"""Derivatives of the executed native EXP-floor free-energy surface."""
from __future__ import annotations

from math import exp

from .native import PIN

EV_TO_J=1.602176634e-19
KB_J_PER_K=1.380649e-23
KB_EV_PER_K=KB_J_PER_K/EV_TO_J


def surface_diagnostics(surface,stress_Pa:float,T_K:float,nu0_s:float)->dict:
    if stress_Pa<0 or T_K<=0 or nu0_s<=0:
        raise ValueError("physical stress, temperature and attempt frequency required")
    A_raw=surface.G00_eV+surface.gT_eV_per_K*(T_K-surface.Tref_K)
    c_raw=surface.sigc0_Pa+surface.sT_Pa_per_K*(T_K-surface.Tref_K)
    A=max(A_raw,1e-12);dA=surface.gT_eV_per_K if A_raw>1e-12 else 0.0
    c=max(c_raw,1.0);dc=surface.sT_Pa_per_K if c_raw>1.0 else 0.0
    lower=max(surface.floor_min_eV,surface.floor_fraction*A)
    if lower==surface.floor_min_eV:floor_branch="absolute_min";dF=0.0
    else:floor_branch="fraction";dF=surface.floor_fraction*dA
    upper=surface.floor_max_fraction*A
    if lower>=upper:floor_branch="upper_fraction_cap";dF=surface.floor_max_fraction*dA
    F=min(upper,lower)
    a=max(surface.exp_a,0.0);n=max(surface.exp_n,1e-9)
    q=(stress_Pa/c)**n
    B=exp(-a*q) if a*q<745 else 0.0
    raw=F+(A-F)*B
    G=max(raw,0.0)
    if raw<=0:
        S=H=V=0.0;outer_clip=True
    else:
        dGdT=dF+(dA-dF)*B+(A-F)*B*a*n*q*dc/c
        S=-dGdT
        H=G+T_K*S
        if stress_Pa>0:
            V=(A-F)*B*a*n/c*(stress_Pa/c)**(n-1)
        elif n>1:V=0.0
        elif n==1:V=(A-F)*B*a/c
        else:V=float("nan")
        outer_clip=False
    exponent=max(-700.0,min(0.0,-G/(KB_EV_PER_K*T_K)))
    return {"source_commit":PIN,"stress_Pa":stress_Pa,"temperature_K":T_K,
            "G_eV":G,"S_eV_per_K":S,"S_over_kB":S/KB_EV_PER_K,
            "H_eV":H,"V_eV_per_Pa":V,"V_m3":V*EV_TO_J,
            "elementary_rate_s":nu0_s*exp(exponent),"A_eV":A,"c_Pa":c,
            "floor_eV":F,"floor_branch":floor_branch,
            "A_lower_clip":A_raw<=1e-12,"c_lower_clip":c_raw<=1.0,
            "outer_zero_clip":outer_clip,"exponent_clip":exponent==-700.0}
