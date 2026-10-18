"""CZM-owned adapter for the qualified v10.2.30 high-cycle state maps.

Only numerical waiting-state propagation is reused.  The opening rate, pending
event, material-cell transaction, threshold, and waveform remainder remain
owned here by the selected R10 athermal/mixed cohesive law.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from .opening_alternatives_r10 import OpeningAlternative, evaluate_opening
from .persistent_site_high_cycle_engine_v10230 import (
    MODEL_ID as HIGH_CYCLE_MODEL_ID,
    integrate_state_coupled_waveform,
    invalidate_high_cycle_cache,
)
from .persistent_site_high_cycle_state_v10230 import serialize_active_state
from .reduced_persistent_reference_r2 import ReducedPersistentReversibleTipEngine

MODEL_ID="r10.czm-owned-high-cycle-adapter/1"


class CZMPhaseController:
    def __init__(self, phase_count: int):
        self.phase_count=max(int(phase_count),8)
        self.cfg=SimpleNamespace(n_phase=self.phase_count,min_block_cycles=1e-6,
            block_cycles=1e12,max_block_cycles=1e12,cycle_block_mode="hazard_limited")

    def _phases(self):
        return (np.arange(self.phase_count,dtype=float)+.5)*(2.*math.pi/self.phase_count)


def _opening(engine, sigma_Pa: float, temperature_K: float, duration_s: float | None=None):
    result=evaluate_opening(engine._r10_czm_alternative,max(float(sigma_Pa),0.)/engine._r10_czm_kenv,
        engine._r10_czm_kenv,engine._r10_czm_thermal,float(temperature_K),
        engine._r10_czm_branch,phase_duration_s=duration_s or engine._r10_czm_phase_duration_s)
    engine._r10_czm_last_opening=result
    validity=result["kinetic_validity"]
    if validity in {"CROSSOVER_REQUIRES_FIRST_PASSAGE","NONADIABATIC_DISTRIBUTION_PROPAGATION_REQUIRED",
                    "STATIC_SADDLE_MFPT_HAZARD_APPROXIMATION"}:
        engine._r10_czm_unsupported=True
        engine._r10_czm_unsupported_validities.add(validity)
    return result


def _lambda_czm(self, sigma_Pa: float, temperature_K: float):
    opening=_opening(self,sigma_Pa,temperature_K)
    self._r10_czm_branch=dict(opening["connected_state"])
    if opening["state"]=="SEPARATION_COMPLETED":
        rate=1e300
    elif (self._r10_czm_alternative.opening_mode=="cohesive_mixed" and
          opening["kinetic_validity"]=="QUALIFIED_HIGH_BARRIER" and
          opening.get("elementary_rate_s-1") is not None):
        rate=max(float(opening["elementary_rate_s-1"]),0.)
    else:
        rate=0.
    barrier=float(opening.get("barrier_J") or 0.)
    return rate,rate,barrier


def _integrate_czm(self, K_Pa_sqrt_m: float, temperature_K: float, duration_s: float,
                   stress_override=None, lambda_override=None):
    requested=max(float(duration_s),0.)
    stress=(max(float(stress_override),0.) if stress_override is not None
            else max(float(self.sigma_tip(max(float(K_Pa_sqrt_m),0.))),0.))
    opening=_opening(self,stress,temperature_K,requested)
    if opening["state"]=="SEPARATION_COMPLETED":
        # Mechanical completion is deterministic.  The phase discretization
        # locates it at this accepted phase endpoint; an exponential threshold
        # must not create a fictitious sub-phase waiting time.
        self.t=float(self.t)+requested
        self._r10_czm_pending_event={"threshold_action":None,
            "trigger":"MECHANICAL_COMPLETION","pre_opening":opening,
            "accepted_duration_s":requested,"unused_duration_s":0.,
            "stochastic_threshold":False}
        return {"fired":True,"n_fire":1,"v_crack":0.,"dB":0.,
            "physical_hazard_action_step":0.,"da":0.,"dt_consumed":requested,
            "dt_unused":0.,"packet_mean":0.,"packet_variance_m2":0.,
            "lambda_c":0.,"lambda_c_raw":0.,
            "Gc_J":float(opening.get("barrier_J") or 0.),"sigma_tip":stress,
            "plastic":{},"advance":{},"microsteps":1,"r10_czm_owner":MODEL_ID,
            "r10_czm_accepted_threshold":None,
            "r10_czm_unsupported":bool(self._r10_czm_unsupported)}
    if lambda_override is None:
        if (self._r10_czm_alternative.opening_mode=="cohesive_mixed" and
            opening["kinetic_validity"]=="QUALIFIED_HIGH_BARRIER" and
            opening.get("elementary_rate_s-1") is not None):
            rate=max(float(opening["elementary_rate_s-1"]),0.)
        else:
            rate=0.
    else:
        rate=max(float(lambda_override),0.)
    threshold=max(float(self.hazard_threshold_action),1e-300)
    current=max(float(self.hazard_action_current),0.)
    remaining=max(threshold-current,0.)
    consumed=requested if rate<=0. else min(requested,remaining/rate)
    half=.5*consumed
    if getattr(self.mpz,"_reversible_transport_installed",False):
        self.mpz._reversible_transport_K_signed_Pa_sqrt_m=float(K_Pa_sqrt_m)
        self.mpz._reversible_tip_radius_m=float(self.r_eff())
    first=self._plastic_half_step(half,temperature_K,stress)
    if getattr(self.mpz,"_reversible_transport_installed",False):
        self.mpz._reversible_tip_radius_m=float(self.r_eff())
    stress_end=max(float(self.sigma_tip(max(float(K_Pa_sqrt_m),0.))),0.)
    second=self._plastic_half_step(half,temperature_K,stress_end)
    self.t=float(self.t)+consumed
    action=rate*consumed
    fired=bool(rate>0. and current+action>=threshold-1e-12*max(threshold,1.))
    accepted_threshold=threshold if fired else None
    if fired:
        self.hazard_threshold_history.append(threshold)
        self.hazard_event_index=int(self.hazard_event_index)+1
        self.hazard_action_current=0.;self.B=0.
        self._r10_czm_pending_event={"threshold_action":threshold,
            "trigger":"QUALIFIED_HIGH_BARRIER_FIRST_PASSAGE",
            "pre_opening":opening,"accepted_duration_s":consumed,
            "unused_duration_s":max(requested-consumed,0.),
            "stochastic_threshold":True}
    else:
        self.hazard_action_current=min(current+action,threshold)
        self.B=self.hazard_action_current/threshold
        self._r10_czm_branch=dict(opening["connected_state"])
    plastic={}
    for payload in (first,second):
        for key,value in dict(payload).items():
            if isinstance(value,(int,float,np.integer,np.floating)):
                plastic[key]=plastic.get(key,0.)+float(value)
    return {"fired":fired,"n_fire":int(fired),"v_crack":0.,"dB":action/threshold,
        "physical_hazard_action_step":action,"da":0.,"dt_consumed":consumed,
        "dt_unused":max(requested-consumed,0.),"packet_mean":0.,"packet_variance_m2":0.,
        "lambda_c":rate,"lambda_c_raw":rate,"Gc_J":float(opening.get("barrier_J") or 0.),
        "sigma_tip":stress,"plastic":plastic,"advance":{},"microsteps":1,
        "r10_czm_owner":MODEL_ID,"r10_czm_accepted_threshold":accepted_threshold,
        "r10_czm_unsupported":bool(self._r10_czm_unsupported)}


class CZMOwnedPersistentEngine(ReducedPersistentReversibleTipEngine):
    """Class-level bindings survive the numerical engine's private deep copies."""
    lambda_cleave=_lambda_czm
    _integrate_coupled=_integrate_czm


def install_czm_owner(engine, alternative: OpeningAlternative, thermal,
                      environment_stiffness_Pa_per_m: float, phase_count: int,
                      seed: int, frequency_Hz: float = 1000.) -> Any:
    if not isinstance(engine,ReducedPersistentReversibleTipEngine):
        raise TypeError("CZM high-cycle adapter requires the retained reversible engine")
    engine.__class__=CZMOwnedPersistentEngine
    engine._r10_czm_alternative=alternative
    engine._r10_czm_thermal=thermal
    engine._r10_czm_kenv=float(environment_stiffness_Pa_per_m)
    if frequency_Hz <= 0.:
        raise ValueError("frequency_Hz must be positive")
    engine._r10_czm_phase_duration_s=1./float(frequency_Hz)/max(int(phase_count),8)
    engine._r10_czm_branch={"branch":"BONDED","delta_m":0.}
    engine._r10_czm_unsupported=False
    engine._r10_czm_unsupported_validities=set()
    engine._r10_czm_pending_event=None
    engine._r10_czm_rng=np.random.default_rng(int(seed))
    engine._hazard_rng=engine._r10_czm_rng
    engine.hazard_threshold_action=float(engine._r10_czm_rng.exponential())
    engine.hazard_action_current=0.;engine.B=0.;engine.hazard_event_index=0
    engine.hazard_threshold_history=[]
    invalidate_high_cycle_cache(engine,"install_r10_czm_owner")
    return engine


def commit_pending_event(engine, patch_m: float) -> dict[str,Any]:
    pending=engine._r10_czm_pending_event
    if pending is None:raise RuntimeError("no CZM-owned pending event")
    wake=engine.mpz.advance(float(patch_m));engine.a_adv=float(engine.a_adv)+patch_m
    engine.n_adv=int(engine.n_adv)+1
    engine.checkpoint_advance_total_m=float(engine.checkpoint_advance_total_m)+patch_m
    engine._r10_czm_branch={"branch":"BONDED","delta_m":0.}
    if pending.get("stochastic_threshold",True):
        engine.hazard_threshold_action=float(engine._r10_czm_rng.exponential())
    engine.hazard_action_current=0.;engine.B=0.;engine._r10_czm_pending_event=None
    invalidate_high_cycle_cache(engine,"accepted_r10_czm_event")
    return {**pending,"wake_transaction":wake,"extension_m":float(engine.checkpoint_advance_total_m),
            "successor_threshold_action":float(engine.hazard_threshold_action)}


def opening_snapshot(engine, K_Pa_sqrt_m: float, temperature_K: float) -> dict[str,Any]:
    stress=max(float(engine.sigma_tip(max(float(K_Pa_sqrt_m),0.))),0.)
    opening=_opening(engine,stress,temperature_K)
    connected=opening.get("connected") or {}
    return {"stress_Pa":stress,"environment_imposed_opening_m":stress/engine._r10_czm_kenv,
        "branch_state":opening["state"],"kinetic_validity":opening["kinetic_validity"],
        "barrier_kBT":opening.get("barrier_kBT"),"elementary_rate_s-1":opening.get("elementary_rate_s-1"),
        "cohesive_separation_m":connected.get("delta_m"),"tip_radius_m":float(engine.r_eff()),
        "K_shield_Pa_sqrt_m":float(engine.K_shield()),"mobile_count":float(engine.mpz.mobile_count),
        "retained_count":float(engine.mpz.retained_count),
        "source_backstress_Pa":float(getattr(engine.mpz,"continuum_source_last_sigma_back_Pa",0.))}


def qualify_accelerated_overlap(engine, controller, waveform, temperature_K: float) -> dict[str,Any]:
    from .persistent_site_poincare_v10230 import one_cycle_map
    exact=one_cycle_map(engine,controller,waveform,temperature_K)
    accelerated=copy.deepcopy(engine)
    accelerated.hazard_threshold_action=1e100;accelerated.hazard_action_current=0.;accelerated.B=0.
    result=integrate_state_coupled_waveform(accelerated,controller,waveform,temperature_K,1.)
    left=exact.state_end.vector;right=serialize_active_state(accelerated).vector
    state_error=float(np.max(np.abs(left-right)/np.maximum(np.maximum(np.abs(left),np.abs(right)),1.)))
    action_error=abs(float(exact.hazard_action_per_cycle)-float(result["physical_hazard_action_step"]))
    action_scale=max(abs(float(exact.hazard_action_per_cycle)),abs(float(result["physical_hazard_action_step"])),1e-300)
    return {"passed":state_error<=2e-3 and action_error/action_scale<=2e-3,
        "state_relative_max_error":state_error,"action_relative_error":action_error/action_scale,
        "exact_action_per_cycle":float(exact.hazard_action_per_cycle),
        "accelerated_action_per_cycle":float(result["physical_hazard_action_step"]),
        "high_cycle_model_id":HIGH_CYCLE_MODEL_ID,"adapter_model_id":MODEL_ID}


__all__=["MODEL_ID","CZMPhaseController","install_czm_owner","commit_pending_event",
         "opening_snapshot","qualify_accelerated_overlap"]
