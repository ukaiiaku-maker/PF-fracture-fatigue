"""Focused isothermal comparison of EXP-floor and finite-cutoff opening laws.

All material changes are explicit coordinates. The scalar mechanics, native
Arrhenius clocks, source law, linked observations and objective remain shared.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import gammainc, gammaincinv

from .opening_surface import FiniteCutoffOpeningSurface, KB_EV_PER_K
from .primary_fit_v5 import PrimaryEvaluator
from .single_condition import from_isothermal


RUN=Path('runs/ast2018_opening_family_fit_v8')
T_K=294.15
SIGMA_REF_PA=6.e9
CUTOFFS_GPA=(9.,12.,16.)
B_VECTOR=np.array([4.3582858338409505,1653261341.9323406,1.3,
                   2347978468.7400994,.8344988772426564,
                   1.4038469456939304,.455])
A_VECTOR=np.array([3.4000000000000004,1653261341.9323406,1.25,
                   2347978468.7400994,.8344988772426564,
                   1.4038469456939304,.445])
BOUNDS={
    'floor':np.array([[3.0,5.8],[.2,.8],[1.2,3.2],[1.12,1.46],[1.7,3.4]]),
    'cutoff':np.array([[-2.,2.],[0.,1.],[-.1,1.],[1.12,1.46],[1.7,3.4]]),
}
LABELS={
    'floor':('G0_open_eV','Gf_open_eV','sigma_characteristic_GPa',
             'G0_emit_eV','sigma_emit_GPa'),
    'cutoff':('log10_open_rate_at_6GPa_s','w_at_6GPa','log10_s_GPa',
              'G0_emit_eV','sigma_emit_GPa'),
}


def clock_rate_from_barrier(G_eV,physics):
    raw=physics.cleavage_nu0_s*math.exp(-G_eV/(KB_EV_PER_K*T_K))
    tau=physics.cleavage_correlation_time_s
    return float(gammainc(physics.cleavage_hits,raw*tau)/tau)


def barrier_from_clock_rate(rate_s,physics):
    tau=physics.cleavage_correlation_time_s
    ceiling=float(gammainc(physics.cleavage_hits,
                            physics.cleavage_nu0_s*tau)/tau)
    if not (0<rate_s<ceiling):raise ValueError('opening rate outside native clock domain')
    exposure=float(gammaincinv(physics.cleavage_hits,rate_s*tau))
    raw=exposure/tau
    barrier=-KB_EV_PER_K*T_K*math.log(raw/physics.cleavage_nu0_s)
    if not math.isfinite(barrier) or barrier<0:
        raise ValueError('invalid inverse opening barrier')
    return barrier


def material_for(ev,family,coords,cutoff_GPa=None):
    c=np.asarray(coords,float)
    if c.shape!=(5,) or np.any(~np.isfinite(c)) or np.any(c<BOUNDS[family][:,0]) or np.any(c>BOUNDS[family][:,1]):
        raise ValueError('family coordinates outside explicit bounds')
    x=B_VECTOR.copy()
    x[2]=c[3];x[3]=c[4]*1e9
    if family=='floor':
        if cutoff_GPa is not None:raise ValueError('floor family has no cutoff')
        x[0]=c[0];x[6]=c[1];x[1]=c[2]*1e9
        material=from_isothermal(ev.ev.material,x,ev.ev.completion)
        material.cleavage.validate()
        return material,x,{'family':'independent_EXP_floor','coordinates':dict(zip(LABELS[family],c.tolist()))}
    if family!='cutoff' or cutoff_GPa not in CUTOFFS_GPA:
        raise ValueError('unknown cutoff sensitivity point')
    material=from_isothermal(ev.ev.material,x,ev.ev.completion)
    sigma_c=cutoff_GPa*1e9
    if SIGMA_REF_PA>=sigma_c:raise ValueError('reference drive must be below cutoff')
    Gref=barrier_from_clock_rate(10.**c[0],ev.physics)
    q=1.-SIGMA_REF_PA/sigma_c
    Href=Gref/q**1.5
    F=(1.-c[1])*Href
    s=10.**c[2]*1e9
    B=c[1]*Href*math.exp(SIGMA_REF_PA/s)
    if not (.3<=F+B<=7.):
        raise ValueError('zero-stress opening amplitude outside declared bounds')
    opening=FiniteCutoffOpeningSurface(F,B,s,sigma_c)
    opening.validate()
    material=replace(material,cleavage=opening,
                     candidate_id=f'ast2018_cutoff_{cutoff_GPa:g}GPa')
    spec={'family':opening.family,'reference_drive_Pa':SIGMA_REF_PA,
          'cutoff_sensitivity_GPa':cutoff_GPa,
          'coordinates':dict(zip(LABELS[family],c.tolist())),
          'opening_surface':asdict(opening),'G_at_reference_eV':Gref,
          'rate_at_reference_s':10.**c[0]}
    return material,x,spec


def evaluate(family,coords,cutoff_GPa=None,level='reduced',step=.05,save=None):
    ev=PrimaryEvaluator(step=step)
    started=time.monotonic()
    try:
        material,x,spec=material_for(ev,family,coords,cutoff_GPa)
        row=ev.run(x,level,save,material=material)
        row['material_spec']=spec
    except Exception as exc:
        row={'status':'inadmissible','error':f'{type(exc).__name__}: {exc}',
             'loss':1e6,'full_support':False,'acceptable':False,
             'runtime_s':time.monotonic()-started}
    row.update(family=family,cutoff_sensitivity_GPa=cutoff_GPa,
               coordinates=np.asarray(coords,float).tolist(),level=level,step=step)
    return row


def source_hashes():
    names=('family_fit_v8.py','family_search_v8.py','primary_fit_v5.py','opening_surface.py',
           'reduced.py','spatial.py','line_profile.py','mechanics.py',
           'path_policy.py','single_condition.py','objective.py')
    folder=Path(__file__).parent
    return {name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in names}
