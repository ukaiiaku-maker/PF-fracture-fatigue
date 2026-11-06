"""Cold-feasible shared-floor hypothesis for a conditional corrected search.

Stress scales follow the measured 294-to-773 K elastic-modulus secant. This
is a declared physical hypothesis, not an inferred stress-scale measurement.
It prevents the v9 emission scale's near-zero high-temperature collapse.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from .opening_surface import KB_EV_PER_K
from .shared_material_v9 import T0,seed_evaluator
from .surfaces import surface_diagnostics
from .thermal_floor_v10 import HeatCapacityFloorSurface

NAMES=("G0_open_294_eV","Gf_open_294_eV","sigc_open_294_GPa",
       "S0_open_kB","Sf_open_kB","Cp_floor_kB","G0_emit_294_eV",
       "sigc_emit_294_GPa","S_emit_kB")
BOUNDS=np.array([[3.,5.8],[.28,.65],[1.,3.2],[-35.,-1.],[-18.,-5.],[0.,20.],
                 [.9,2.],[1.2,3.5],[-50.,-5.]],float)
E_REF_GPA=411.
E_HOT_GPA=385.
T_HOT_K=773.15
SECANT=(E_HOT_GPA/E_REF_GPA-1)/(T_HOT_K-T0)


def coordinates_from_v9(receipt):
    d=receipt["coordinates"]
    return np.array([d["G0_open_294_eV"],d["Gf_open_294_eV"],
        d["sigc_open_294_GPa"],d["S0_open_kB"],-14.8,7.5,
        d["G0_emit_294_eV"],d["sigc_emit_294_GPa"],-45.],float)


def material_from_coordinates(base,x):
    x=np.asarray(x,float)
    if x.shape!=(9,) or not np.all(np.isfinite(x)) or np.any(x<BOUNDS[:,0]) or np.any(x>BOUNDS[:,1]):
        raise ValueError("outside v10 conditional material bounds")
    c,e=base.cleavage,base.emission
    dTc=c.Tref_K-T0;dTe=e.Tref_K-T0
    sc_slope=x[2]*SECANT;se_slope=x[7]*SECANT
    opening=HeatCapacityFloorSurface(
        G0_ref_eV=x[0]-x[3]*KB_EV_PER_K*dTc,
        Gf_ref_eV=x[1]-x[4]*KB_EV_PER_K*dTc,
        sigc_ref_Pa=(x[2]+sc_slope*dTc)*1e9,
        S0_over_kB=x[3],Sf_over_kB=x[4],
        d_sigc_dT_Pa_per_K=sc_slope*1e9,
        alpha=c.alpha,n=c.n,Tref_K=c.Tref_K,
        Cp_floor_over_kB=x[5],T0_K=T0)
    emission=replace(e,G00_eV=x[6]-x[8]*KB_EV_PER_K*dTe,
        gT_eV_per_K=-x[8]*KB_EV_PER_K,
        sigc0_Pa=(x[7]+se_slope*dTe)*1e9,
        sT_Pa_per_K=se_slope*1e9)
    material=replace(base,cleavage=opening,emission=emission,
                     candidate_id="ast2018_shared_floor_v10_conditional")
    opening.validate()
    # Positive isothermal activation enthalpy at the two zero-stress limits.
    if x[1]+T0*x[4]*KB_EV_PER_K<=.01 or x[6]+T0*x[8]*KB_EV_PER_K<=.01:
        raise ValueError("nonpositive floor or emission activation enthalpy")
    if x[0]+T0*x[3]*KB_EV_PER_K<=.01:
        raise ValueError("nonpositive zero-stress opening enthalpy")
    for T in (183.15,294.15,773.15):
        for stress in (0.,1e9,2e9,3e9,4e9,5e9,6e9,8e9,12e9):
            opening_d=opening.derivatives(stress,T)
            if opening_d["S_over_kB"]>0:
                raise ValueError("opening activation entropy prior violated")
            if opening_d["H_eV"]<0:
                raise ValueError("negative opening activation enthalpy")
            if surface_diagnostics(emission,stress,T,1e12)["S_over_kB"]>0:
                raise ValueError("emission activation entropy prior violated")
    return material


def initial_material():
    import json
    from pathlib import Path
    _,seed=seed_evaluator()
    receipt=json.loads(Path("runs/ast2018_shared_floor_v9/selected_material.json").read_text())
    x=coordinates_from_v9(receipt)
    return seed,x,material_from_coordinates(seed,x)
