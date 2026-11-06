"""Versioned material coordinates for the corrected Ast comparisons.

The original seven-coordinate native law remains a named branch. Optional
coordinates are released in stages; the opening stress-scale gauge is fixed
at the inherited alpha and exponent throughout these comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass,replace
from typing import Literal

import numpy as np

from .objective import PARAMETERS,material_from_normalized,physically_admissible
from .opening_surface import IndependentFloorSurface,KB_EV_PER_K
from .surfaces import surface_diagnostics


ENTROPY_STRESSES_PA=(4e9,6e9,8e9,10e9,12e9)
ENTROPY_INTERVAL_KB=(-40.0,-10.0)


@dataclass(frozen=True)
class Parameterization:
    opening_kind: Literal["native_fractional_floor","independent_floor"]="native_fractional_floor"
    opening_stress_slope: Literal["inherited","zero","fitted"]="inherited"
    extended_mobility: bool=False
    entropy_scope: Literal["broad","zero_drive","operating_drive"]="broad"
    entropy_prior_scale_kB: float=5.0

    @property
    def specs(self)->tuple[tuple[str,float,float],...]:
        specs=list(PARAMETERS)
        if self.opening_stress_slope=="fitted":
            specs.append(("opening.d_sigc_dT_over_sigc_per_K",-.002,.002))
        if self.opening_kind=="independent_floor":
            specs.extend((("opening.Gf_ref_eV",0.0,5.0),
                          ("opening.Sf_over_kB",-80.0,80.0)))
        if self.extended_mobility:
            specs.extend((("peierls.activation_entropy_kB",-40.0,40.0),
                          ("taylor.H0_eV",.2,5.0),
                          ("taylor.activation_entropy_kB",-40.0,40.0)))
        return tuple(specs)

    def material(self,base,x):
        values=np.asarray(x,dtype=float)
        if values.shape!=(len(self.specs),) or np.any(~np.isfinite(values)) or np.any(values<0) or np.any(values>1):
            raise ValueError("parameter vector outside declared bounds")
        candidate=material_from_normalized(base,values[:len(PARAMETERS)])
        i=len(PARAMETERS)
        slope=base.cleavage.sT_Pa_per_K
        if self.opening_stress_slope=="zero":slope=0.0
        elif self.opening_stress_slope=="fitted":
            _,lo,hi=self.specs[i]
            slope=(lo+(hi-lo)*values[i])*candidate.cleavage.sigc0_Pa
            i+=1
        if self.opening_kind=="independent_floor":
            Gf=5.0*values[i];Sf=-80.0+160.0*values[i+1];i+=2
            native=candidate.cleavage
            opening=IndependentFloorSurface(
                G0_ref_eV=native.G00_eV,
                S0_over_kB=-native.gT_eV_per_K/KB_EV_PER_K,
                Gf_ref_eV=Gf,Sf_over_kB=Sf,
                sigc_ref_Pa=native.sigc0_Pa,
                d_sigc_dT_Pa_per_K=slope,
                alpha=base.cleavage.exp_a,n=base.cleavage.exp_n,
                Tref_K=base.cleavage.Tref_K)
            candidate=replace(candidate,cleavage=opening)
        elif slope!=candidate.cleavage.sT_Pa_per_K:
            candidate=replace(candidate,cleavage=replace(candidate.cleavage,
                                                          sT_Pa_per_K=slope))
        if self.extended_mobility:
            p_entropy=-40.0+80.0*values[i];i+=1
            t_H=.2+4.8*values[i];i+=1
            t_entropy=-40.0+80.0*values[i];i+=1
            candidate=replace(candidate,
                peierls=replace(candidate.peierls,activation_entropy_kB=p_entropy),
                taylor=replace(candidate.taylor,H0_eV=t_H,
                               activation_entropy_kB=t_entropy))
        assert i==len(values)
        return replace(candidate,candidate_id=f"ast2018_{self.opening_kind}")

    def admissible(self,candidate)->bool:
        return physically_admissible(candidate)

    def entropy_cost(self,candidate,temperatures_K)->dict:
        """Fixed-grid hinge prior, kept separate from the data objective.

        The 5 kB scale is an analyst working scale, not an experimental SD.
        It is applied to the declared grid even when a trajectory terminates.
        """
        if self.entropy_scope=="broad":
            return {"cost":0.0,"n_anchors":0,"max_violation_kB":0.0}
        stresses=(0.0,) if self.entropy_scope=="zero_drive" else ENTROPY_STRESSES_PA
        Tvals=tuple(sorted(set(float(T) for T in temperatures_K)))
        if not Tvals or self.entropy_prior_scale_kB<=0:
            raise ValueError("fixed entropy anchor grid and prior scale required")
        errors=[]
        for T in Tvals:
            for stress in stresses:
                S=surface_diagnostics(candidate.cleavage,stress,T,1e12)["S_over_kB"]
                errors.append(max(ENTROPY_INTERVAL_KB[0]-S,0.0,
                                  S-ENTROPY_INTERVAL_KB[1]))
        e=np.asarray(errors,dtype=float)
        return {"cost":float(.5*np.mean((e/self.entropy_prior_scale_kB)**2)),
                "n_anchors":len(errors),"max_violation_kB":float(np.max(e)),
                "anchor_stresses_Pa":stresses,"anchor_temperatures_K":Tvals,
                "entropy_interval_kB":ENTROPY_INTERVAL_KB,
                "prior_scale_kB":self.entropy_prior_scale_kB}
