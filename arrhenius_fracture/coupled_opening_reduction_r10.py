"""State-informed cycle and ramp reductions for R10 cohesive opening modes.

The reduction uses the production EXP_floor plastic half-step and current
finite-compliance cohesive kernels.  It intentionally omits stochastic event
localization; a unit accumulated high-barrier action is the deterministic mean
mixed-event mark.  Unsupported crossover action is never consumed.
"""
from __future__ import annotations
import math
from typing import Any

import numpy as np

from .cohesive_arrhenius import BilinearCohesiveLaw
from .fatigue_v1 import FatigueWaveform
from .opening_alternatives_r10 import OpeningAlternative, evaluate_opening
from .persistent_site_high_cycle_state_v10230 import serialize_active_state

MODEL_ID = "r10.coupled-opening-state-reduction/1"


def score_recurrence(events: list[dict[str, Any]], observation_limit_m: float,
                     unsupported: bool=False, lower_m: float=20e-6,
                     upper_m: float=60e-6) -> dict[str, Any]:
    """Classify event histories without treating bounded bursts as a veto."""
    cycles=sorted({int(math.floor(float(e["cycle"])+1e-12)) for e in events})
    target_hit=bool(events and float(events[-1]["extension_m"])>=observation_limit_m-1e-15)
    lower=next((e for e in events if float(e["extension_m"])>=lower_m),None)
    upper=next((e for e in events if float(e["extension_m"])>=upper_m),None)
    rate=None;window=None
    if lower is not None and upper is not None:
        duration=float(upper["cycle"])-float(lower["cycle"])
        if duration>0. and len(cycles)>=2:
            rate=(float(upper["extension_m"])-float(lower["extension_m"]))/duration
            window={"start_cycle":float(lower["cycle"]),"end_cycle":float(upper["cycle"]),
                    "start_extension_m":float(lower["extension_m"]),
                    "end_extension_m":float(upper["extension_m"]),"duration_cycles":duration}
    if unsupported:
        outcome="UNSUPPORTED_MIXED_CROSSOVER"
        rate=None;window=None
    elif target_hit and len(cycles)<=1:
        outcome="TARGET_TRUNCATED_SINGLE_EXCURSION_CASCADE"
        rate=None;window=None
    elif rate is not None:
        outcome="FINITE_RECURRENT_GROWTH"
    elif len(cycles)>=2:
        outcome="RECURRENT_GROWTH_OUTSIDE_REPORTING_WINDOW"
    elif events:
        outcome="FINITE_SINGLE_BURST"
    else:
        outcome="FINITE_EXPOSURE_NO_GROWTH"
    return {"outcome":outcome,"developed_rate_m_per_cycle":rate,
            "developed_window":window,"event_cycle_count":len(cycles),
            "observation_target_hit":target_hit}


def bilinear_accessibility_coordinates(thermal, temperature_K: float,
                                       environment_stiffness_Pa_per_m: float,
                                       radius_m: float, phases: int) -> dict[str, float]:
    law=BilinearCohesiveLaw(thermal);peak=thermal.t_hat(temperature_K)
    dp=law.peak_separation(temperature_K);df=law.final_separation(temperature_K)
    scale=math.sqrt(2*math.pi*radius_m)/1e6
    endpoint=environment_stiffness_Pa_per_m*df*scale
    loss=(peak+environment_stiffness_Pa_per_m*dp)*scale
    hmax=.5*(1+.1)+.5*(1-.1)*math.cos(math.pi/phases)
    return {"static_completed_branch_K_MPa_sqrt_m":endpoint,
            "occupied_bonded_branch_loss_K_MPa_sqrt_m":loss,
            "discrete_waveform_Kmax_for_branch_loss_MPa_sqrt_m":loss/hmax,
            "waveform_midpoint_hmax":hmax,"delta_peak_m":dp,"delta_final_m":df}


def _advance_plastic(engine, duration_s: float, temperature_K: float,
                     K_Pa_sqrt_m: float) -> dict[str,float]:
    Ksigned=float(K_Pa_sqrt_m);K=max(Ksigned,0.);half=.5*duration_s
    if bool(getattr(engine.mpz,"_reversible_transport_installed",False)):
        engine.mpz._reversible_transport_K_signed_Pa_sqrt_m=Ksigned
        engine.mpz._reversible_tip_radius_m=float(engine.r_eff())
    sigma=max(float(engine.sigma_tip(K)),0.)
    first=engine._plastic_half_step(half,temperature_K,sigma)
    if bool(getattr(engine.mpz,"_reversible_transport_installed",False)):
        engine.mpz._reversible_tip_radius_m=float(engine.r_eff())
    sigma=max(float(engine.sigma_tip(K)),0.)
    second=engine._plastic_half_step(half,temperature_K,sigma)
    engine.t=float(engine.t)+duration_s
    totals={}
    for payload in (first,second):
        for key,value in dict(payload).items():
            if isinstance(value,(int,float,np.integer,np.floating)) and math.isfinite(float(value)):
                totals[key]=totals.get(key,0.)+float(value)
    return totals


def _commit(engine, patch_m: float):
    wake=engine.mpz.advance(patch_m);engine.a_adv=float(engine.a_adv)+patch_m
    engine.n_adv=int(engine.n_adv)+1
    engine.checkpoint_advance_total_m=float(engine.checkpoint_advance_total_m)+patch_m
    return wake


def coupled_cycle_reduction(engine, alternative: OpeningAlternative, thermal,
                            environment_stiffness_Pa_per_m: float, Kmax_MPa_sqrt_m: float,
                            temperature_K: float=300., cycles: int=32, phases: int=8,
                            patch_m: float=1e-6, observation_limit_m: float=120e-6) -> dict[str,Any]:
    wave=FatigueWaveform(Kmax=Kmax_MPa_sqrt_m*1e6,R=.1,frequency_Hz=1000.)
    dt=wave.period_s/phases;branch={"branch":"BONDED","delta_m":0.};action=0.
    events=[];history=[];unsupported=False;mixed_action_crossing=None
    for cycle in range(cycles):
        start=serialize_active_state(engine).vector;cycle_events=0
        for phase in range(phases):
            angle=2*math.pi*(phase+.5)/phases;Ksigned=float(wave.K_phase(angle))
            _advance_plastic(engine,dt,temperature_K,Ksigned)
            stress=max(float(engine.sigma_tip(max(Ksigned,0.))),0.)
            opening=evaluate_opening(alternative,stress/environment_stiffness_Pa_per_m,
                environment_stiffness_Pa_per_m,thermal,temperature_K,branch,phase_duration_s=dt)
            branch=opening["connected_state"];validity=opening["kinetic_validity"]
            unsupported|=validity in {"CROSSOVER_REQUIRES_FIRST_PASSAGE",
                "NONADIABATIC_DISTRIBUTION_PROPAGATION_REQUIRED",
                "STATIC_SADDLE_MFPT_HAZARD_APPROXIMATION"}
            if validity=="QUALIFIED_HIGH_BARRIER" and opening.get("elementary_rate_s-1") is not None:
                action+=float(opening["elementary_rate_s-1"])*dt
            if alternative.opening_mode=="cohesive_mixed" and action>=1.:
                mixed_action_crossing={"cycle":cycle+(phase+.5)/phases,"phase":phase,
                    "action":action,"approximation":"DETERMINISTIC_UNIT_ACTION_ACQUISITION_SCREEN",
                    "commitment_status":"REQUIRES_PREFIX_REMAINDER_NATIVE_CONSUMER"}
                break
            trigger=opening["state"]=="SEPARATION_COMPLETED"
            while trigger and engine.checkpoint_advance_total_m+patch_m<=observation_limit_m+1e-15:
                _commit(engine,patch_m);cycle_events+=1
                events.append({"cycle":cycle+(phase+.5)/phases,"phase":phase,
                    "extension_m":float(engine.checkpoint_advance_total_m),
                    "accepted_opening_drive_Pa":stress,
                    "trigger":"MECHANICAL_COMPLETION" if opening["state"]=="SEPARATION_COMPLETED" else "MEAN_HIGH_BARRIER_ACTION",
                    "mobile_count":float(engine.mpz.mobile_count),"retained_count":float(engine.mpz.retained_count),
                    "tip_radius_m":float(engine.r_eff()),"K_shield_Pa_sqrt_m":float(engine.K_shield())})
                # The advanced material frame owns a new cohesive cell; only
                # that cell starts bonded.  mpz.advance retains/remaps the
                # plastic wake rather than copying the departed cell state.
                branch={"branch":"BONDED","delta_m":0.};action=max(action-1.,0.)
                stress=max(float(engine.sigma_tip(max(Ksigned,0.))),0.)
                opening=evaluate_opening(alternative,stress/environment_stiffness_Pa_per_m,
                    environment_stiffness_Pa_per_m,thermal,temperature_K,branch,phase_duration_s=dt)
                branch=opening["connected_state"]
                trigger=opening["state"]=="SEPARATION_COMPLETED"
            if engine.checkpoint_advance_total_m>=observation_limit_m-1e-15:break
            if mixed_action_crossing is not None:break
        end=serialize_active_state(engine).vector;denom=max(float(np.linalg.norm(start)),1.)
        history.append({"cycle":cycle+1,"events_cycle":cycle_events,
            "extension_m":float(engine.checkpoint_advance_total_m),
            "state_relative_change":float(np.linalg.norm(end-start)/denom),
            "mobile_count":float(engine.mpz.mobile_count),"retained_count":float(engine.mpz.retained_count),
            "tip_radius_m":float(engine.r_eff()),"K_shield_Pa_sqrt_m":float(engine.K_shield())})
        if engine.checkpoint_advance_total_m>=observation_limit_m-1e-15 or mixed_action_crossing is not None:break
    scored=score_recurrence(events,observation_limit_m,unsupported)
    if mixed_action_crossing is not None and not unsupported:
        scored.update({"outcome":"SCREENED_MIXED_ACTION_CROSSING_REQUIRES_NATIVE_LOCALIZATION",
                       "developed_rate_m_per_cycle":None,"developed_window":None})
    return {"schema":MODEL_ID,"loading":"CYCLIC_STATE_REDUCTION","Kmax_MPa_sqrt_m":Kmax_MPa_sqrt_m,
        "DeltaK_MPa_sqrt_m":.9*Kmax_MPa_sqrt_m,"temperature_K":temperature_K,
        "cycles_requested":cycles,"cycles_completed":len(history),"phases":phases,
        "events":events,"history":history,"event_cycle_count":scored["event_cycle_count"],
        "developed_rate_m_per_cycle":scored["developed_rate_m_per_cycle"],
        "developed_window":scored["developed_window"],"outcome":scored["outcome"],
        "observation_target_hit":scored["observation_target_hit"],
        "unsupported_crossover":unsupported,"mixed_action_crossing":mixed_action_crossing,
        "legacy_opening_clock_used":False,"mean_action_mark_used":False,
        "mixed_action_role":"ACQUISITION_SCREEN_ONLY" if alternative.opening_mode=="cohesive_mixed" else None}


def coupled_monotonic_ramp(engine, alternative: OpeningAlternative, thermal,
                           environment_stiffness_Pa_per_m: float, temperature_K: float,
                           Kdot_MPa_sqrt_m_s: float=.005, dK_MPa_sqrt_m: float=.5,
                           Kmax_MPa_sqrt_m: float=200.) -> dict[str,Any]:
    K=0.;branch={"branch":"BONDED","delta_m":0.};action=0.;history=[];unsupported=False
    first_loss=None;first_commit=None
    while K<Kmax_MPa_sqrt_m-1e-14:
        right=min(K+dK_MPa_sqrt_m,Kmax_MPa_sqrt_m);mid=.5*(K+right)
        dt=(right-K)/Kdot_MPa_sqrt_m_s
        _advance_plastic(engine,dt,temperature_K,mid*1e6)
        stress=max(float(engine.sigma_tip(right*1e6)),0.)
        opening=evaluate_opening(alternative,stress/environment_stiffness_Pa_per_m,
            environment_stiffness_Pa_per_m,thermal,temperature_K,branch,phase_duration_s=dt)
        branch=opening["connected_state"];validity=opening["kinetic_validity"]
        unsupported|=validity in {"CROSSOVER_REQUIRES_FIRST_PASSAGE",
            "NONADIABATIC_DISTRIBUTION_PROPAGATION_REQUIRED",
            "STATIC_SADDLE_MFPT_HAZARD_APPROXIMATION"}
        if opening["state"] in {"BONDED_BRANCH_LOST","POST_ESCAPE_TRANSIT"} and first_loss is None:first_loss=right
        if validity=="QUALIFIED_HIGH_BARRIER" and opening.get("elementary_rate_s-1") is not None:
            action+=float(opening["elementary_rate_s-1"])*dt
        committed=(opening["state"]=="SEPARATION_COMPLETED" or
                   (alternative.opening_mode=="cohesive_mixed" and action>=1.))
        history.append({"K_MPa_sqrt_m":right,"state":opening["state"],"kinetic_validity":validity,
            "action":action,"mobile_count":float(engine.mpz.mobile_count),
            "retained_count":float(engine.mpz.retained_count),"tip_radius_m":float(engine.r_eff()),
            "K_shield_Pa_sqrt_m":float(engine.K_shield())})
        K=right
        if committed:first_commit=right;break
    return {"schema":MODEL_ID,"loading":"MONOTONIC_RAMP","temperature_K":temperature_K,
        "Kdot_MPa_sqrt_m_s":Kdot_MPa_sqrt_m_s,"dK_MPa_sqrt_m":dK_MPa_sqrt_m,
        "Kmax_MPa_sqrt_m":Kmax_MPa_sqrt_m,"occupied_branch_loss_K_MPa_sqrt_m":first_loss,
        "first_committed_separation_K_MPa_sqrt_m":first_commit,"unsupported_crossover":unsupported,
        "status":"COMMITTED_SEPARATION" if first_commit is not None else
                 "UNSUPPORTED_MIXED_CROSSOVER" if unsupported else "KMAX_CENSOR",
        "history":history,"legacy_opening_clock_used":False}


__all__=["MODEL_ID","bilinear_accessibility_coordinates","score_recurrence","coupled_cycle_reduction",
         "coupled_monotonic_ramp"]
