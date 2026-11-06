"""Task-local independent-floor opening barrier for both pinned native adapters.

Native 0-D and 1-D kinetics both call ``candidate.cleavage.barrier_eV``.
This immutable surface implements that same interface without editing the
pinned source or changing the native rate and three-hit event conventions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


KB_EV_PER_K=1.380649e-23/1.602176634e-19
TMIN_K=183.15
TMAX_K=773.15
EV_TO_J=1.602176634e-19


@dataclass(frozen=True)
class FiniteCutoffOpeningSurface:
    """Isothermal free-energy barrier with a C1 fold cutoff.

    Stress scales are Pa, amplitudes are eV. No thermal extrapolation is
    implied by the single-temperature parameters.
    """
    F_ref_eV: float
    B_ref_eV: float
    s_ref_Pa: float
    sigma_instability_ref_Pa: float
    Tref_K: float = 294.15
    family: str = "finite_cutoff_exp_3half"

    def validate(self):
        values=(self.F_ref_eV,self.B_ref_eV,self.s_ref_Pa,
                self.sigma_instability_ref_Pa,self.Tref_K)
        if (not all(math.isfinite(v) for v in values) or self.F_ref_eV<0 or
            self.B_ref_eV<0 or self.F_ref_eV+self.B_ref_eV<=0 or
            self.s_ref_Pa<=0 or self.sigma_instability_ref_Pa<=0 or
            self.Tref_K<=0 or self.family!="finite_cutoff_exp_3half"):
            raise ValueError("invalid finite-cutoff opening parameters")

    def _check_temperature(self,T_K):
        self.validate()
        if not math.isclose(float(T_K),self.Tref_K,rel_tol=0,abs_tol=1e-9):
            raise ValueError("finite-cutoff opening family is isothermal-only")

    def barrier_eV(self,stress_Pa,T_K):
        self._check_temperature(T_K)
        sigma=np.maximum(np.asarray(stress_Pa,dtype=float),0.)
        q=np.maximum(1.-sigma/self.sigma_instability_ref_Pa,0.)
        value=q**1.5*(self.F_ref_eV+self.B_ref_eV*np.exp(-sigma/self.s_ref_Pa))
        return float(value) if value.ndim==0 else value

    def derivatives(self,stress_Pa,T_K):
        self._check_temperature(T_K)
        sigma=float(stress_Pa)
        if not math.isfinite(sigma) or sigma<0:
            raise ValueError("derivatives require a nonnegative finite stress")
        if sigma>=self.sigma_instability_ref_Pa:
            return {"G_eV":0.,"V_eV_per_Pa":0.,"V_m3":0.,
                    "curvature_eV_per_Pa2":0.,"q":0.,"domain":"cutoff"}
        c=self.sigma_instability_ref_Pa;s=self.s_ref_Pa
        q=1.-sigma/c;E=math.exp(-sigma/s);H=self.F_ref_eV+self.B_ref_eV*E
        G=q**1.5*H
        V=1.5/c*math.sqrt(q)*H+self.B_ref_eV/s*q**1.5*E
        curvature=.75/c**2/math.sqrt(q)*H+3*self.B_ref_eV/(c*s)*math.sqrt(q)*E+self.B_ref_eV/s**2*q**1.5*E
        return {"G_eV":G,"V_eV_per_Pa":V,"V_m3":V*EV_TO_J,
                "curvature_eV_per_Pa2":curvature,"q":q,"domain":"open",
                "S_eV_per_K":None,"H_eV":None}


@dataclass(frozen=True)
class IndependentFloorSurface:
    G0_ref_eV: float
    S0_over_kB: float
    Gf_ref_eV: float
    Sf_over_kB: float
    sigc_ref_Pa: float
    d_sigc_dT_Pa_per_K: float
    alpha: float
    n: float
    Tref_K: float=481.33

    def endpoints(self,T_K:float)->tuple[float,float,float]:
        dt=float(T_K)-self.Tref_K
        return (self.G0_ref_eV-self.S0_over_kB*KB_EV_PER_K*dt,
                self.Gf_ref_eV-self.Sf_over_kB*KB_EV_PER_K*dt,
                self.sigc_ref_Pa+self.d_sigc_dT_Pa_per_K*dt)

    def validate(self)->None:
        if self.alpha<=0 or self.n<=0:
            raise ValueError("opening alpha and exponent must be positive")
        for T in (TMIN_K,TMAX_K):
            G0,Gf,c=self.endpoints(T)
            if not (0<=Gf<=G0 and c>0):
                raise ValueError("independent opening endpoints outside physical domain")

    def barrier_eV(self,stress_Pa,T_K:float):
        self.validate()
        G0,Gf,c=self.endpoints(T_K)
        sigma=np.maximum(np.asarray(stress_Pa,dtype=float),0.0)
        q=(sigma/c)**self.n
        return Gf+(G0-Gf)*np.exp(-self.alpha*q)

    def derivatives(self,stress_Pa:float,T_K:float)->dict:
        """S=-dG/dT, H=G+TS and V=-dG/dsigma at fixed physical stress."""
        self.validate()
        if stress_Pa<0 or T_K<=0:
            raise ValueError("physical stress and temperature required")
        G0,Gf,c=self.endpoints(T_K)
        q=(stress_Pa/c)**self.n
        B=float(np.exp(-self.alpha*q))
        gap=G0-Gf
        G=Gf+gap*B
        S0=self.S0_over_kB*KB_EV_PER_K
        Sf=self.Sf_over_kB*KB_EV_PER_K
        # The stress-scale derivative enters at fixed applied stress.
        S=Sf+(S0-Sf)*B-gap*B*self.alpha*self.n*q*self.d_sigc_dT_Pa_per_K/c
        if stress_Pa>0:
            V=gap*B*self.alpha*self.n*q/stress_Pa
        elif self.n>1:
            V=0.0
        elif self.n==1:
            V=gap*self.alpha/c
        else:
            V=float("inf")
        return {"G_eV":G,"S_eV_per_K":S,"S_over_kB":S/KB_EV_PER_K,
                "H_eV":G+T_K*S,"V_eV_per_Pa":V,"q":q,"B":B,
                "floor_branch":"independent_affine","A_eV":G0,
                "floor_eV":Gf,"c_Pa":c}


def fractional_equivalent(native_surface):
    """Tie endpoint entropies and values on the native fractional-floor branch.

    Equivalence holds only where native absolute-min and upper-cap clips are
    inactive. Those two legacy branches remain native, distinct models.
    """
    f=native_surface.floor_fraction
    S0=-native_surface.gT_eV_per_K/KB_EV_PER_K
    return IndependentFloorSurface(
        native_surface.G00_eV,S0,f*native_surface.G00_eV,f*S0,
        native_surface.sigc0_Pa,native_surface.sT_Pa_per_K,
        native_surface.exp_a,native_surface.exp_n,native_surface.Tref_K)
