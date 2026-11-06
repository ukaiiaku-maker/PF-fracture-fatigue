"""EXP opening floor with constant activation heat-capacity completion.

At reference T0: Gf(T0)=Gf0, Sf(T0)=Sf0.  Constant Cp gives
Gf(T)=Gf0-Sf0*kB*(T-T0)+Cp*kB*((T-T0)-T*log(T/T0)).
The stress dependence and native three-hit clock are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log
import numpy as np

from .opening_surface import IndependentFloorSurface,KB_EV_PER_K


@dataclass(frozen=True)
class HeatCapacityFloorSurface(IndependentFloorSurface):
    Cp_floor_over_kB: float=0.
    T0_K: float=294.15

    def endpoints(self,T_K):
        if T_K<=0:raise ValueError("positive absolute temperature required")
        G0,_,c=super().endpoints(T_K)
        dt=float(T_K)-self.T0_K
        Gf0=self.Gf_ref_eV-self.Sf_over_kB*KB_EV_PER_K*(self.T0_K-self.Tref_K)
        Gf=(Gf0-self.Sf_over_kB*KB_EV_PER_K*dt+
            self.Cp_floor_over_kB*KB_EV_PER_K*(dt-T_K*log(T_K/self.T0_K)))
        return G0,Gf,c

    def validate(self):
        if self.Cp_floor_over_kB<0 or self.T0_K<=0:
            raise ValueError("invalid floor heat-capacity completion")
        for T in (183.15,self.T0_K,773.15):
            G0,Gf,c=self.endpoints(T)
            if not (0<=Gf<=G0 and c>0):
                raise ValueError("thermal floor outside physical endpoints")

    def derivatives(self,stress_Pa,T_K):
        self.validate()
        if stress_Pa<0:raise ValueError("nonnegative stress required")
        G0,Gf,c=self.endpoints(T_K)
        q=(stress_Pa/c)**self.n
        B=float(np.exp(-self.alpha*q));gap=G0-Gf
        G=Gf+gap*B
        Sf=(self.Sf_over_kB+self.Cp_floor_over_kB*log(T_K/self.T0_K))*KB_EV_PER_K
        S0=self.S0_over_kB*KB_EV_PER_K
        S=Sf+(S0-Sf)*B-gap*B*self.alpha*self.n*q*self.d_sigc_dT_Pa_per_K/c
        V=gap*B*self.alpha*self.n*q/stress_Pa if stress_Pa>0 else 0.
        return {"G_eV":G,"S_eV_per_K":S,"S_over_kB":S/KB_EV_PER_K,
                "H_eV":G+T_K*S,"V_eV_per_Pa":V,"q":q,"B":B,
                "floor_branch":"constant_Cp_independent_floor",
                "A_eV":G0,"floor_eV":Gf,"c_Pa":c}
