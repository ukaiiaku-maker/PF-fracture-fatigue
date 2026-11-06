"""Shared-temperature floor material and provenance-aware Ast objective.

The coordinates are values at 294.15 K.  Entropies are in kB and stress
slopes in GPa/K.  The fixed v8 shape, attempt clocks, release barriers and
all mechanics remain unchanged.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .data import AstData
from .family_fit_v8 import material_for
from .objective import WorkingObjective, physically_admissible
from .opening_surface import KB_EV_PER_K
from .primary_fit_v5 import PrimaryEvaluator


DATA = Path("runs/ast2018_shape_fit_v5/production/input/Ast2018_joint_fit_data")
T0 = 294.15
NAMES = ("G0_open_294_eV", "Gf_open_294_eV", "sigc_open_294_GPa",
         "S0_open_kB", "Sf_open_kB", "dsigc_open_GPa_per_K",
         "G0_emit_294_eV", "sigc_emit_294_GPa", "S_emit_kB",
         "dsigc_emit_GPa_per_K")
# Wide enough to include the unchanged v8 material. Endpoint admissibility is
# checked separately; numerical clipping never makes a candidate acceptable.
BOUNDS = np.array([[3.0,5.8],[.25,.8],[1.0,3.2],[-35.,20.],[-20.,20.],
                   [-.006,.010],[.9,2.0],[1.2,3.5],[-100.,40.],
                   [-.006,.003]],dtype=float)
LINKED = {"Tp21C_rate0p2", "Tp100C_rate0p2", "Tp250C_rate0p2"}


def seed_evaluator():
    primary=PrimaryEvaluator(step=.05)
    floor=np.array([4.3344873374384765,.455,1.5039187322958822,
                    1.2977795363442521,2.1748262661667837])
    material,_,_=material_for(primary,"floor",floor)
    return primary,material


def coordinates(material):
    c,e=material.cleavage,material.emission
    G0,Gf,sc=c.endpoints(T0)
    dt=T0-e.Tref_K
    return np.array([G0,Gf,sc/1e9,c.S0_over_kB,c.Sf_over_kB,
                     c.d_sigc_dT_Pa_per_K/1e9,
                     e.G00_eV+e.gT_eV_per_K*dt,
                     (e.sigc0_Pa+e.sT_Pa_per_K*dt)/1e9,
                     -e.gT_eV_per_K/KB_EV_PER_K,e.sT_Pa_per_K/1e9])


def material_from_coordinates(base,x):
    x=np.asarray(x,dtype=float)
    if (x.shape!=(len(NAMES),) or not np.all(np.isfinite(x)) or
            np.any(x<BOUNDS[:,0]) or np.any(x>BOUNDS[:,1])):
        raise ValueError("shared material outside explicit coordinate bounds")
    c,e=base.cleavage,base.emission
    dt=c.Tref_K-T0
    opening=replace(c,G0_ref_eV=x[0]-x[3]*KB_EV_PER_K*dt,
                    Gf_ref_eV=x[1]-x[4]*KB_EV_PER_K*dt,
                    sigc_ref_Pa=(x[2]+x[5]*dt)*1e9,
                    S0_over_kB=x[3],Sf_over_kB=x[4],
                    d_sigc_dT_Pa_per_K=x[5]*1e9)
    dt=e.Tref_K-T0
    emission=replace(e,G00_eV=x[6]-x[8]*KB_EV_PER_K*dt,
                     gT_eV_per_K=-x[8]*KB_EV_PER_K,
                     sigc0_Pa=(x[7]+x[9]*dt)*1e9,
                     sT_Pa_per_K=x[9]*1e9)
    candidate=replace(base,cleavage=opening,emission=emission,
                      candidate_id="ast2018_shared_floor_v9")
    if not physically_admissible(candidate):
        raise ValueError("material clips or crosses physical endpoints")
    return candidate


def make_objective(primary,material,scenario="TABLE_PRIMARY",subset=None,
                   *,predictor=None,zero=None):
    data=primary.ev.data
    J_conditions={r["condition_id"] for r in data.J_curves
                  if r["sample_for_fit"].lower() in ("true","1","yes")}
    def linked_filter(condition_id,family):
        return (family.name!="nominal_K_trace" or condition_id in LINKED or
                condition_id not in J_conditions)
    def specimen_for_condition(condition):
        cid=condition["condition_id"]
        specimen=primary.ev.assumptions.specimen(condition)
        if cid=="Tp21C_rate0p2":
            return primary.specimen
        if cid not in LINKED:
            return specimen
        rows=sorted((r for r in data.K_displacement if r["condition_id"]==cid
                     and r["fit_role"]=="loading_history_candidate" and
                     float(r["K_MPa_sqrt_m"])<=2.5),
                    key=lambda r:float(r["displacement_um"]))[:5]
        u=np.array([float(r["displacement_um"]) for r in rows])
        K=np.array([float(r["K_MPa_sqrt_m"]) for r in rows])
        slope=float(u@K/(u@u))
        C0=specimen.geometry_factor(specimen.a0_m)/(slope*1e12)
        return replace(specimen,initial_compliance_m_per_N=C0,
                       instrument_compliance_m_per_N=0.)
    kwargs={"condition_subset":subset,"block_weight_policy":"rate_condition",
            "family_filter":linked_filter,"candidate_factory":material_from_coordinates,
            "admissibility":physically_admissible,
            "specimen_factory":specimen_for_condition}
    if predictor is not None:kwargs["predictor"]=predictor
    return WorkingObjective(data,scenario,primary.ev.assumptions,
                            primary.ev.zero if zero is None else zero,
                            material,primary.ev.base_physics,**kwargs)
