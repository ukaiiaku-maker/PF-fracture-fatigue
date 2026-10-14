#!/usr/bin/env python3
"""Run deterministic current-backend monotonic ramps for retained R10 rows."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from arrhenius_fracture.archived_fatigue_material_r9 import build_archived_current_engine
from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.reduced_persistent_reference_r2 import advance_signed, runtime_receipt
from arrhenius_fracture.retained_material_r10 import (
    load_offspring_binding,load_parent_binding,load_strict_original_binding,load_v2_binding,
)
from arrhenius_fracture.stochastic_avalanche_tip import clear_pending_geometry_events,pop_pending_geometry_event


def load(a):
    if a.kind=="original":return load_strict_original_binding(a.original_registry,a.identifier)
    if a.kind=="parent":return load_parent_binding(a.parent_registry,a.identifier)
    if a.kind=="offspring":return load_offspring_binding(a.offspring_registry,a.identifier)
    return load_v2_binding(a.v2_rows,a.v2_manifest,a.identifier)


def state_diagnostic(engine,K_MPa_sqrt_m,T):
    """Serialize the actual hazard-bearing state without changing it."""
    sigma=float(engine.sigma_tip(float(K_MPa_sqrt_m)*1e6))
    lam,raw,barrier=engine.lambda_cleave(sigma,float(T))
    diag=engine.mpz.diagnostics()
    radius=float(engine.mpz.blunted_radius(engine.f.r0,engine.b))
    return {
        "K_MPa_sqrt_m":float(K_MPa_sqrt_m),
        "opening_stress_Pa":sigma,"cleavage_rate_s-1":float(lam),
        "raw_cleavage_rate_s-1":float(raw),"cleavage_barrier_J":float(barrier),
        "hazard_action_current":float(getattr(engine,"hazard_action_current",0.0)),
        "hazard_threshold_action":float(getattr(engine,"hazard_threshold_action",1.0)),
        "remaining_action_to_mark":max(float(getattr(engine,"hazard_threshold_action",1.0))-
                                       float(getattr(engine,"hazard_action_current",0.0)),0.0),
        "emitted_total":float(getattr(engine.mpz,"emitted_total",0.0)),
        "gross_source_activity":float(getattr(engine.mpz,"cumulative_gross_source_activity",0.0)),
        "mobile_count":float(engine.mpz.mobile_count),
        "retained_count":float(engine.mpz.retained_count),
        "gross_return_activity":float(getattr(engine.mpz,"cumulative_gross_return_activity",0.0)),
        "signed_return":float(np.sum(engine.mpz.returned_slip_positive-engine.mpz.returned_slip_negative)),
        "accumulated_source_linked_slip":float(np.sum(engine.mpz.accumulated_slip_positive+engine.mpz.accumulated_slip_negative)),
        "tip_radius_m":radius,
        "reference_tip_radius_m":float(engine.f.r0),
        "radius_over_r0":float(radius/engine.f.r0),
        "K_shield_MPa_sqrt_m":float(engine.K_shield()/1e6),
        "persistent_site_multiplicity_per_system":float(diag["persistent_site_multiplicity_per_system"]),
    }


def run(binding,T,Kdot,Kmax,dK,target_um,max_steps):
    engine=build_archived_current_engine(binding,kernel_family_path=R4_KERNEL_FAMILY)
    clear_pending_geometry_events(); Kleft=0.; events=[]; history=[]; steps=0
    while Kleft<Kmax and engine.checkpoint_advance_total_m*1e6<target_um:
        steps+=1
        if steps>max_steps:break
        increment=min(dK,Kmax-Kleft);Kmid=Kleft+.5*increment;dt=increment/Kdot
        result=advance_signed(engine,K_Pa_sqrt_m=Kmid*1e6,temperature_K=T,dt_s=dt)
        consumed=min(max(float(result.get("kinetic_dt_consumed_s",dt)),0.),dt)
        fired=bool(result.get("fired",False));descriptor=pop_pending_geometry_event() if fired else None
        if fired != (descriptor is not None):raise RuntimeError("native event/descriptor mismatch")
        Kright=Kleft+Kdot*consumed if fired else Kleft+increment
        if descriptor is not None:
            diagnostic=engine.mpz.diagnostics();events.append({
                "event_index":len(events)+1,"K_MPa_sqrt_m":Kright,
                "extension_um":engine.checkpoint_advance_total_m*1e6,
                "event_advance_um":float(descriptor["event_advance_m"])*1e6,
                "K_shield_MPa_sqrt_m":engine.K_shield()/1e6,
                "mobile_count":engine.mpz.mobile_count,"retained_count":engine.mpz.retained_count,
                "source_activity":engine.mpz.cumulative_gross_source_activity,
                "return_activity":engine.mpz.cumulative_gross_return_activity,
                "persistent_site_multiplicity_per_system":diagnostic["persistent_site_multiplicity_per_system"],
                "hazard_bearing_state":state_diagnostic(engine,Kright,T),
            })
        if not history or fired or steps%100==0:
            history.append({"step":steps,"K_MPa_sqrt_m":Kright,"fired":fired,
                "extension_um":engine.checkpoint_advance_total_m*1e6,
                "K_shield_MPa_sqrt_m":engine.K_shield()/1e6,
                "mobile_count":engine.mpz.mobile_count,"retained_count":engine.mpz.retained_count})
        Kleft=float(Kright)
        if fired and consumed<=0.:Kleft+=min(increment,1e-12)
    complete=engine.checkpoint_advance_total_m*1e6>=target_um
    return {"schema":"r10.current-deterministic-monotonic-ramp/1",
        "candidate_id":binding.optimized_row["candidate_id"],"temperature_K":T,
        "status":"complete" if complete else "KMAX_OR_STEP_CENSOR",
        "Kdot_MPa_sqrt_m_s":Kdot,"Kmax_MPa_sqrt_m":Kmax,"dK_MPa_sqrt_m":dK,
        "target_extension_um":target_um,"final_K_MPa_sqrt_m":Kleft,
        "final_extension_um":engine.checkpoint_advance_total_m*1e6,"event_count":len(events),
        "K_init_MPa_sqrt_m":events[0]["K_MPa_sqrt_m"] if events else None,
        "K_final_event_MPa_sqrt_m":events[-1]["K_MPa_sqrt_m"] if events else None,
        "events":events,"history":history,"runtime_receipt":runtime_receipt(engine,stage="r10_monotonic_terminal"),
        "terminal_hazard_bearing_state":state_diagnostic(engine,Kleft,T),
        "binding":dict(binding.provenance),"deterministic_diagnostic_mark":True,
        "fixed_avalanche_length":True,"local_strength_Pa":30e9}


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--kind",choices=("original","parent","offspring","v2"),required=True)
    p.add_argument("--identifier",required=True);p.add_argument("--original-registry",type=Path)
    p.add_argument("--parent-registry",type=Path);p.add_argument("--offspring-registry",type=Path)
    p.add_argument("--v2-rows",type=Path);p.add_argument("--v2-manifest",type=Path)
    p.add_argument("--temperatures",default="300,450,600,750,900,1050,1200")
    p.add_argument("--Kdot",type=float,default=.005);p.add_argument("--Kmax",type=float,default=80.)
    p.add_argument("--dK",type=float,default=.05);p.add_argument("--target-extension-um",type=float,default=50.)
    p.add_argument("--max-steps",type=int,default=2_000_000);p.add_argument("--out-root",type=Path,required=True)
    p.add_argument("--force",action="store_true")
    a=p.parse_args(argv);a.out_root.mkdir(parents=True,exist_ok=True);binding=load(a);cases=[]
    for T in [float(x) for x in a.temperatures.split(",")]:
        path=a.out_root/f"T_{int(T):04d}K.json"
        if path.is_file() and not a.force:result=json.loads(path.read_text())
        else:
            result=run(binding,T,a.Kdot,a.Kmax,a.dK,a.target_extension_um,a.max_steps)
            temp=path.with_suffix(".json.tmp");temp.write_text(json.dumps(result,indent=2,sort_keys=True,default=str)+"\n");temp.replace(path)
        cases.append({k:result[k] for k in ("candidate_id","temperature_K","status","K_init_MPa_sqrt_m","K_final_event_MPa_sqrt_m","final_extension_um","event_count")})
        print(json.dumps(cases[-1],sort_keys=True),flush=True)
    (a.out_root/"panel_summary.json").write_text(json.dumps({"cases":cases},indent=2,sort_keys=True)+"\n");return 0


if __name__=="__main__":raise SystemExit(main())
