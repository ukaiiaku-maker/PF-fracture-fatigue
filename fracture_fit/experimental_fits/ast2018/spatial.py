"""Coupled cantilever wrapper for the pinned spatial 1-D state engine."""
from __future__ import annotations

import copy
from math import pi, sqrt

import numpy as np

from .mechanics import HistoryPoint, Specimen
from .reduced import Prediction


def _event_rate_audit(state, candidate, physics, K_MPa_sqrt_m, T_K):
    """State-dependent native rate diagnostics at a committed event threshold."""
    rates = state.local_rates(K_MPa_sqrt_m, T_K)
    mobile = np.maximum(state.mobile_m2, 0)
    retained = np.maximum(state.retained_m2, 0)
    forest = state.forest_density_m2()
    spacing = 1 / (2 * np.sqrt(forest))
    jump = physics.jump_fraction_of_forest_spacing * spacing
    phi = spacing / max(physics.b_m, 1e-30)
    if np.isfinite(physics.taylor_phi_max):
        phi = np.minimum(phi, physics.taylor_phi_max)
    p_stress = candidate.peierls.stress_fraction * rates["tau_eff_Pa"]
    t_stress = candidate.taylor.stress_fraction * rates["tau_eff_Pa"] * phi
    emission_stress = np.asarray(rates["emission_effective_stress_Pa"], dtype=float)
    mobile_total = float(np.sum(mobile))
    retained_total = float(np.sum(retained))
    shield = state.K_shield_MPa_sqrt_m()
    cleavage_sigma = (max(K_MPa_sqrt_m - shield, 0) * 1e6 /
                      sqrt(2 * pi * max(state.tip_radius_m(), physics.b_m, 1e-30)))
    multiplicity = float(state.source_geometry()["multiplicity_per_system"])
    return {"rho_mobile_sum_m2":mobile_total,
            "rho_retained_sum_m2":retained_total,
            "shielding_pre_MPa_sqrt_m":float(shield),
            "cleavage_effective_stress_Pa":float(cleavage_sigma),
            "emission_effective_stress_min_Pa":float(np.min(emission_stress)),
            "emission_effective_stress_max_Pa":float(np.max(emission_stress)),
            "peierls_effective_stress_min_Pa":float(np.min(np.abs(p_stress))),
            "peierls_effective_stress_max_Pa":float(np.max(np.abs(p_stress))),
            "taylor_effective_stress_min_Pa":float(np.min(np.abs(t_stress))),
            "taylor_effective_stress_max_Pa":float(np.max(np.abs(t_stress))),
            "cleavage_effective_rate_s":float(state.cleavage_rate_s(K_MPa_sqrt_m,T_K)),
            "emission_effective_activations_s":float(multiplicity*np.sum(
                np.maximum(rates["emission_rate_per_site_s"],0))),
            "peierls_effective_encounter_s":float(np.sum(mobile*rates["encounter_s"][:,None,:])/
                mobile_total) if mobile_total>0 else None,
            "taylor_effective_release_s":float(np.sum(retained*rates["taylor_completion_s"][:,None,:])/
                retained_total) if retained_total>0 else None,
            "peierls_mobile_weighted_signed_rate_s":float(np.sum(
                mobile*rates["peierls_velocity_m_s"][:,None,:])/
                max(np.sum(mobile*jump[:,None,:]),1e-30)) if mobile_total>0 else None}


def _plastic_displacement_rate(state, specimen: Specimen, force_N: float,
                               T_K: float, committed_a_m: float) -> float:
    """The same virtual-work projection used by the mean-field wrapper.

    Signed native Peierls transport supplies v. The spatial integral uses the
    actual mobile populations and dx*state_strip_width for each bin.
    """
    if force_N <= 0 or not np.any(state.mobile_m2 > 0):
        return 0.0
    K = specimen.current_K_MPa_sqrt_m(force_N,committed_a_m)
    rates = state.local_rates(K,T_K)
    mobile = np.sum(np.maximum(state.mobile_m2,0),axis=1)
    velocity = rates["velocity_m_s"]
    x = (np.arange(state.c.n_bins)+0.5)*state.dx
    factor = np.abs(state.emission_drive_factors())[:,None]
    radius=state.tip_radius_m()
    unit_tau=factor*specimen.geometry_factor(committed_a_m)/(
        np.sqrt(2*np.pi*(radius+x))[None,:])
    gamma_dot = state.c.b_m*mobile*velocity
    volume = specimen.B_m*state.dx*state.state_strip_width_m
    return float(np.sum(unit_tau*gamma_dot)*volume)


def _displacement_from_native_work(before, after, specimen: Specimen,
                                   committed_a_m: float, force_N: float) -> float:
    """Project the native internally integrated Orowan work to load-point slip.

    The native energy ledger integrates the same mobile-density/velocity law
    through emission and storage substeps. We convert its per-front-length
    external work using the ratio of the unit-force Mode-I field to native
    applied tip stress. The source zone is the declared scalar quadrature area.
    """
    delta_work=after.external_plastic_work_J_per_m-before.external_plastic_work_J_per_m
    if delta_work==0:return 0.0
    radius=.5*(before.tip_radius_m()+after.tip_radius_m())
    length=after.c.source_zone_length_m
    spatial_mean=2*sqrt(radius)*(sqrt(radius+length)-sqrt(radius))/length
    ratio=spatial_mean/max(force_N,1e-30)
    return specimen.B_m*ratio*delta_work


def predict_spatial(spatial,candidate,physics,specimen:Specimen,T_K:float,
                    Kdot_MPa_sqrt_m_per_s:float,assumptions,
                    *,event_seed:int|None=None,deterministic_threshold:bool=False,
                    required_u_m:float=0.0,required_extension_m:float|None=None,
                    translation_mode:str="hazard_coupled",
                    translation_action_exponent:float=.95)->Prediction:
    """Run native 1-D state steps under shared scalar force equilibrium.

    This retains the native v9.13 action-coupled state translation convention;
    only the observation/mechanics loading callback is added. The committed
    geometric crack length changes after first passage, never during a trial.
    """
    if translation_mode != "hazard_coupled" or translation_action_exponent<=0:
        raise ValueError("Ast confirmation requires declared hazard-coupled translation")
    state = spatial.EmergentGNDState(candidate,physics)
    rng=np.random.default_rng(assumptions.event_seed if event_seed is None else event_seed)
    threshold=1.0 if deterministic_threshold else float(rng.exponential())
    action=0.0;committed_path=0.0;extension=0.0
    u=upl=force=time=0.0
    speed=specimen.initial_displacement_rate(Kdot_MPa_sqrt_m_per_s)
    du_nominal=(assumptions.nominal_K_step_MPa_sqrt_m*1e6*
                (specimen.initial_compliance_m_per_N+specimen.instrument_compliance_m_per_N)/
                specimen.geometry_factor(specimen.a0_m))
    history=[HistoryPoint(0,0,0,specimen.a0_m,0,0)]
    events=[];steps=0;max_dH=0.0
    target=(assumptions.target_extension_m if required_extension_m is None else required_extension_m)

    def trial(fraction,old_state,du,old_committed_path):
        dt=fraction*du/speed
        unew=u+fraction*du
        a=specimen.a0_m+extension
        Fnew=specimen.equilibrium_force(unew,upl,a)
        znew=None;uplnew=upl;da_step=0.0
        for iteration in range(6):
            Fused=Fnew
            znew=copy.deepcopy(old_state)
            K0=specimen.current_K_MPa_sqrt_m(force,a)
            K1=specimen.current_K_MPa_sqrt_m(Fnew,a)
            r0=znew.cleavage_rate_s(K0,T_K)
            r1pred=znew.cleavage_rate_s(K1,T_K)
            Hpred=.5*(r0+r1pred)*dt
            fraction_path=min(max((action+Hpred)/threshold,0.0),1.0)**translation_action_exponent
            da_step=max(assumptions.event_advance_m*fraction_path-old_committed_path,0.0)
            da_step=min(da_step,assumptions.event_advance_m-old_committed_path)
            resolved=spatial._state_advance_is_resolved(
                znew,.5*(K0+K1),T_K,dt,minimum_expected_activations=0.0)
            if resolved and da_step>0:
                znew.advance_coupled_segment(duration_s=dt,da_m=da_step,
                    K_start_MPa_sqrt_m=K0,K_end_MPa_sqrt_m=K1,T_K=T_K,
                    geometry_extension_override_m=extension)
            elif resolved:
                znew.advance_time(dt,.5*(K0+K1),T_K)
            elif da_step>0:
                znew.time_s+=dt;znew.translate_tip(da_step)
            else:
                znew.time_s+=dt
            uplnew=upl+_displacement_from_native_work(
                old_state,znew,specimen,a,.5*(force+Fnew))
            Fnew=specimen.equilibrium_force(unew,uplnew,a)
            if abs(Fnew-Fused)<=1e-5*max(Fnew,1e-12):break
        else:
            raise ArithmeticError("scalar force equilibrium did not converge")
        K1=specimen.current_K_MPa_sqrt_m(Fnew,a)
        dH=.5*(old_state.cleavage_rate_s(K0,T_K)+znew.cleavage_rate_s(K1,T_K))*dt
        return znew,unew,uplnew,Fnew,dt,dH,da_step

    try:
        while extension<target-1e-15 or u<required_u_m-1e-15:
            if steps>=assumptions.maximum_load_steps:
                return Prediction("numerical_maximum_steps",history,events,steps,max_dH)
            if specimen.nominal_K_MPa_sqrt_m(force)>=assumptions.maximum_nominal_K_MPa_sqrt_m:
                return Prediction("numerical_load_domain_limit",history,events,steps,max_dH)
            steps+=1
            old=copy.deepcopy(state)
            du=du_nominal
            znew,unew,uplnew,Fnew,dt,dH,da_step=trial(1,old,du,committed_path)
            if not np.isfinite(dH) or dH<0:raise ArithmeticError("invalid hazard increment")
            max_dH=max(max_dH,dH)
            if action+dH>=threshold:
                lo,hi=0.0,1.0
                for _ in range(assumptions.localization_bisections):
                    mid=.5*(lo+hi)
                    result=trial(mid,old,du,committed_path)
                    if action+result[5]>=threshold:hi=mid
                    else:lo=mid
                znew,unew,uplnew,Fnew,dt,dH,da_step=trial(hi,old,du,committed_path)
                state,u,upl,force=znew,unew,uplnew,Fnew;time+=dt
                K=specimen.current_K_MPa_sqrt_m(force,specimen.a0_m+extension)
                if extension+assumptions.event_advance_m>=specimen.W_m-specimen.a0_m:
                    history.append(HistoryPoint(time,u,force,specimen.a0_m+extension,
                                                upl,K,physical_failure=True))
                    events.append({"index":len(events),"time_s":time,
                                   "threshold_action":threshold,"hazard_action":action+dH,
                                   "terminal_ligament_failure":True,
                                   "proposed_advance_m":assumptions.event_advance_m,
                                   "cumulative_extension_m":extension,
                                   "K_pre_MPa_sqrt_m":K})
                    return Prediction("physical_ligament_failure",history,events,steps,max_dH)
                history.append(HistoryPoint(time,u,force,specimen.a0_m+extension,upl,
                                            max(K-state.K_shield_MPa_sqrt_m(),0)))
                pre=state.source_geometry()
                back=float(np.mean(state.backstress_state()[2]))
                rate_audit=_event_rate_audit(state,candidate,physics,K,T_K)
                remaining=assumptions.event_advance_m-(committed_path+da_step)
                if remaining>1e-15:state.translate_tip(remaining)
                extension+=assumptions.event_advance_m
                force=specimen.equilibrium_force(u,upl,specimen.a0_m+extension)
                Kpost=specimen.current_K_MPa_sqrt_m(force,specimen.a0_m+extension)
                history.append(HistoryPoint(time,u,force,specimen.a0_m+extension,upl,
                                            max(Kpost-state.K_shield_MPa_sqrt_m(),0),event=True))
                events.append({"index":len(events),"time_s":time,"threshold_action":threshold,
                               "hazard_action":action+dH,"cumulative_extension_m":extension,
                               "event_advance_m":assumptions.event_advance_m,
                               "K_pre_MPa_sqrt_m":K,"K_post_MPa_sqrt_m":Kpost,
                               "tip_radius_pre_m":pre["tip_radius_m"],
                               "front_width_pre_m":pre["front_width_m"],
                               "source_multiplicity_pre":pre["multiplicity_per_system"],
                               "backstress_pre_Pa":back,
                               "cumulative_external_plastic_work_J_per_m":
                                   float(state.external_plastic_work_J_per_m),
                               "cumulative_activations":float(np.sum(state.cumulative_source_activations))})
                events[-1].update(rate_audit)
                action=0.0;committed_path=0.0
                threshold=1.0 if deterministic_threshold else float(rng.exponential())
            else:
                state,u,upl,force=znew,unew,uplnew,Fnew;time+=dt
                action+=dH;committed_path+=da_step
                K=specimen.current_K_MPa_sqrt_m(force,specimen.a0_m+extension)
                history.append(HistoryPoint(time,u,force,specimen.a0_m+extension,upl,
                                            max(K-state.K_shield_MPa_sqrt_m(),0)))
        return Prediction("complete",history,events,steps,max_dH)
    except Exception as exc:
        return Prediction("numerical_failure",history,events,steps,max_dH,
                          f"{type(exc).__name__}: {exc}")
