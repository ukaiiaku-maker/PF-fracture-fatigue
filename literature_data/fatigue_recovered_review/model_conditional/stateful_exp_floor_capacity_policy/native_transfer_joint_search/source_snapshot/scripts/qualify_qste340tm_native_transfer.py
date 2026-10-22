#!/usr/bin/env python3
"""Paired actual-native/reduced transfer qualification for QSTE340TM.

The native side is a real UnifiedMPZState with the installed continuum source,
exchange, recovery, advection and moving-frame owners.  The comparison uses a
one-system/four-bin lumped geometry with a uniform local stress profile so the
only intended difference is spatial transport versus the aggregate escape map.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from scipy.special import gammainc

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.continuum_source_tip import install_minimal_continuum_source
from arrhenius_fracture.material_manifest import ExpFloorBarrier,MaterialManifest,TransportBarrier,KB_EV_PER_K
from arrhenius_fracture.unified_mpz import MPZConfig,UnifiedMPZState
from scripts.calibrate_qste340tm_stateful_exp_floor import (
    ATTEMPT_FREQUENCY_S,BURGERS_M,COOPERATIVE_HITS,COOPERATIVE_TAU_S,
    EVENT_LENGTH_M,EXP_A,EXP_FLOOR_FRACTION,EXP_GT_EV_PER_K,EXP_ST_PA_PER_K,
    EXP_TREF_K,NOMINAL_FREQUENCY_HZ,PROCESS_ZONE_M,RHO0_M2,SHEAR_MODULUS_PA,
    TEMPERATURE_K,build_records,
)
from scripts.qste340tm_faithful_reduction import PlasticParameters,one_cycle_with_transfers,stress_cycle
from scripts.refit_qste340tm_faithful_reduction import DATA,OUT as PRIOR_OUT,opening_prediction,pack,plastic_parameters,unpack
from scripts.fatigue_informed_joint_barrier_v3_v10230 import SmoothBroadBasis

OUT=DATA/"model_conditional/stateful_exp_floor_capacity_policy/native_transfer_joint_search"


def _barrier(g298:float,s298:float,prefactor:float)->ExpFloorBarrier:
    return ExpFloorBarrier(G00_eV=g298-EXP_GT_EV_PER_K*(TEMPERATURE_K-EXP_TREF_K),
        gT_eV_per_K=EXP_GT_EV_PER_K,sigc0_Pa=s298-EXP_ST_PA_PER_K*(TEMPERATURE_K-EXP_TREF_K),
        sT_Pa_per_K=EXP_ST_PA_PER_K,alpha=EXP_A,exponent=1.,floor_fraction=EXP_FLOOR_FRACTION,
        Tref_K=EXP_TREF_K,attempt_frequency_s=prefactor)


def manifest(parameters:PlasticParameters)->MaterialManifest:
    emission=_barrier(parameters.emission_G0_298_eV,parameters.plastic_sigc_298_Pa,1e11)
    entropy=-EXP_GT_EV_PER_K/KB_EV_PER_K
    h0=parameters.transport_G0_298_eV-EXP_GT_EV_PER_K*(TEMPERATURE_K-EXP_TREF_K)
    transport=lambda pref:TransportBarrier(H0_eV=h0,activation_entropy_kB=entropy,
        alpha=EXP_A,exponent=1.,attempt_frequency_s=pref,stress_ratio=1.)
    return MaterialManifest(name="QSTE340TM_conditional",candidate_id="QSTE_NATIVE_TRANSFER",
        cleavage=_barrier(1.,parameters.plastic_sigc_298_Pa,1e12),emission=emission,
        peierls=transport(1e12),taylor=transport(1e11),taylor_corr_rho_c_m2=1e30,
        taylor_corr_scale=0.,source_sites_per_system=1.,encounter_efficiency=1.,
        retained_recovery_rate_s=parameters.retained_recovery_per_s,
        source_refresh_length_m=PROCESS_ZONE_M,c_blunt=1.,max_K_shield_MPa_sqrt_m=0.)


def native_state(parameters:PlasticParameters,state:np.ndarray)->UnifiedMPZState:
    cfg=MPZConfig(length_m=PROCESS_ZONE_M,n_bins=4,n_systems=1,source_bin_count=4,
        shielding_orientation_factors=(1.,),blunting_length_m=PROCESS_ZONE_M,
        forest_density_floor_m2=RHO0_M2,wake_length_m=PROCESS_ZONE_M,wake_n_bins=4,
        wake_shielding=True)
    native=UnifiedMPZState(manifest(parameters),cfg);install_minimal_continuum_source(native,BURGERS_M,SHEAR_MODULUS_PA)
    native.mobile[:]=float(state[0])/native.n_bins;native.retained[:]=float(state[1])/native.n_bins
    native.accumulated_slip[:]=(float(state[0])+float(state[1]))/native.n_bins
    native.tip_source_activity[:]=float(state[2]);native.available_sites[:]=native.reference_source_multiplicity*float(state[2])
    # Exact matched lumped geometry: retain the actual native owners while
    # suppressing the production spatial stress decay for this correspondence test.
    native.local_stress_profile_Pa=lambda stress:np.full(native.n_bins,max(float(stress),0.))
    native._continuum_tip_radius_m=PROCESS_ZONE_M
    return native


def opening_action(stress:np.ndarray,backstress:np.ndarray,retained:np.ndarray,p:dict)->float:
    basis=SmoothBroadBasis(p["opening_sigma_b_Pa"],p["opening_exponent"],1e-4)
    local=np.maximum(stress-p["cleavage_backstress_fraction"]*backstress,0.)
    raw=p["opening_floor_over_kBT"]+p["opening_amplitude_over_kBT"]*basis.value(local)
    elementary=ATTEMPT_FREQUENCY_S*np.exp(-np.clip(raw,0.,700.))
    completed=gammainc(COOPERATIVE_HITS,np.minimum(elementary*COOPERATIVE_TAU_S,1e12))/COOPERATIVE_TAU_S
    return float(EVENT_LENGTH_M*np.mean(completed)/NOMINAL_FREQUENCY_HZ)


def native_cycle(delta_k:float,r_value:float,initial:np.ndarray,parameters:PlasticParameters,
                 opening:dict,panels:int)->dict:
    native=native_state(parameters,initial);stress,_,_=stress_cycle(delta_k,r_value,initial[0],initial[1],panels)
    dt=1/(NOMINAL_FREQUENCY_HZ*panels);totals={k:0. for k in ("emitted","trapped","released","recovered","escaped")}
    actions=[];backs=[];retained=[]
    for sigma in stress:
        before_escaped=native.escaped_total;result=native.evolve(dt,TEMPERATURE_K,float(sigma),BURGERS_M)
        totals["emitted"]+=result["dN_emit"];totals["trapped"]+=result["dN_trapped"]
        totals["released"]+=result["dN_released"];totals["recovered"]+=result["dN_recovered"]
        totals["escaped"]+=native.escaped_total-before_escaped
        rho=(native.mobile_count+native.retained_count)/PROCESS_ZONE_M**2
        back=SHEAR_MODULUS_PA*BURGERS_M*math.sqrt(max(rho,0.))*math.sqrt(3.)
        backs.append(back);retained.append(native.retained_count)
    action=opening_action(stress,np.asarray(backs),np.asarray(retained),opening)
    final=np.array([native.mobile_count,native.retained_count,float(native.tip_source_activity[0])])
    balance=float(initial[0]+initial[1]+totals["emitted"]-final[0]-final[1]-totals["recovered"]-totals["escaped"])
    return {"final":final.tolist(),"transfers":totals|{"inventory_balance_residual":balance},
            "opening_action_m_per_cycle":action,"native_blunted_radius_m":native.blunted_radius(1e-6,BURGERS_M),
            "native_accumulated_slip":native.local_slip_count(),"native_wake_mobile":native.wake_mobile_count,
            "native_wake_retained":native.wake_retained_count}


def reduced_cycle(delta_k:float,r_value:float,initial:np.ndarray,parameters:PlasticParameters,
                  opening:dict,panels:int)->dict:
    final,transfers=one_cycle_with_transfers(delta_k,r_value,initial,parameters,panels)
    stress,_,back=stress_cycle(delta_k,r_value,initial[0],initial[1],panels)
    action=opening_action(stress,np.full(panels,back),np.full(panels,initial[1]),opening)
    return {"final":final.tolist(),"transfers":transfers,"opening_action_m_per_cycle":action}


def compare_case(label:str,delta_k:float,r_value:float,initial:np.ndarray,parameters:PlasticParameters,
                 opening:dict,panels:int)->dict:
    native=native_cycle(delta_k,r_value,initial,parameters,opening,panels)
    reduced=reduced_cycle(delta_k,r_value,initial,parameters,opening,panels)
    errors={}
    for i,name in enumerate(("mobile","retained","source_activity")):
        errors[name+"_absolute"]=reduced["final"][i]-native["final"][i]
        errors[name+"_relative"]=errors[name+"_absolute"]/max(abs(native["final"][i]),1e-12)
    for name in ("emitted","trapped","released","recovered","escaped"):
        errors[name+"_absolute"]=reduced["transfers"][name]-native["transfers"][name]
        errors[name+"_relative"]=errors[name+"_absolute"]/max(abs(native["transfers"][name]),1e-12)
    errors["opening_action_relative"]=(reduced["opening_action_m_per_cycle"]-native["opening_action_m_per_cycle"])/max(native["opening_action_m_per_cycle"],1e-300)
    return {"label":label,"DeltaK_MPa_sqrt_m":delta_k,"R":r_value,"panels":panels,
            "initial":initial.tolist(),"native":native,"reduced":reduced,"errors":errors}


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--panels",type=int,nargs="+",default=[32,128,512,2048]);args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    fit=json.loads((PRIOR_OUT/"fit_faithful_reduction.json").read_text());vector=np.asarray(fit["parameters_vector"],float);selected=plastic_parameters(vector);opening=unpack(vector)
    records=build_records(DATA);cases=[]
    # Selected, sustained-active, strong-escape and zero-source states.
    states=[("selected",np.array([selected.initial_mobile_count,selected.initial_retained_count,1.]),selected),
            ("sustained_active",np.array([20.,100.,.8]),PlasticParameters(1.2,1.5e9,1.5,1e-3,20.,100.,.8)),
            ("strong_escape",np.array([2.,3.,1.]),PlasticParameters(.3,2e9,.3,0.,2.,3.,1.)),
            ("zero_source",np.array([10.,20.,.4]),PlasticParameters(5.,2e9,1.5,1e-3,10.,20.,.4))]
    targets=[]
    for r in (.1,.3,.5,.7):
        block=[x for x in records if x["R"]==r]
        for pos in (0,len(block)//2,len(block)-1):targets.append((r,float(block[pos]["DeltaK_MPa_sqrt_m"]),pos))
    # Compact matrix: low/mid/high at R=.1 and .7, plus active controls at mid R=.3.
    chosen=[targets[0],targets[1],targets[2],targets[9],targets[10],targets[11]]
    for panels in args.panels:
        for r,dk,pos in chosen:
            label="selected_"+("low" if pos==0 else "high" if pos>200 else "mid")
            cases.append(compare_case(label,dk,r,states[0][1],states[0][2],opening,panels))
        mid=[x for x in records if x["R"]==.3][len([x for x in records if x["R"]==.3])//2]
        for label,state,parameters in states[1:]:cases.append(compare_case(label,float(mid["DeltaK_MPa_sqrt_m"]),.3,state,parameters,opening,panels))
    # Exercise the actual native moving-frame/source-refresh owner separately.
    n=native_state(states[1][2],states[1][1]);before={"mobile":n.mobile_count,"retained":n.retained_count,"slip":n.local_slip_count(),"activity":float(n.tip_source_activity[0])};receipt=n.advance(.25e-6);after={"mobile":n.mobile_count,"retained":n.retained_count,"slip":n.local_slip_count(),"activity":float(n.tip_source_activity[0]),"wake_mobile":n.wake_mobile_count,"wake_retained":n.wake_retained_count}
    payload={"schema":"qste340tm-actual-native-transfer-correspondence/1","native_owner":"UnifiedMPZState + install_minimal_continuum_source",
        "geometry":"one-system/four-bin 1 um lumped comparison; uniform local stress profile",
        "cases":cases,"moving_frame_probe":{"advance_m":.25e-6,"before":before,"receipt":receipt,"after":after},
        "correspondence_scope":{"represented":["continuum source activity","mobile emission","native exchange","retained recovery","native mobile advection","opening hazard observation"],"not_represented_by_aggregate_replay":["spatial stress decay","persistent accumulated-slip radius","wake shielding","signed history","crack-advance remap","source refresh after advance"]}}
    (OUT/"native_transfer_correspondence.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"cases":len(cases),"panels":args.panels,"output":str(OUT/"native_transfer_correspondence.json")},indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
