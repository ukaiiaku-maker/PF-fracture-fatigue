"""Coupled scalar cantilever / native stateful 0-D first-passage integration."""
from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass
from math import pi, sqrt

import numpy as np

from .mechanics import (HistoryPoint,Specimen,birth_plastic_displacement,
                        transport_plastic_velocity,unit_force_resolved_stress)
from .conservative_transport import residence_step
from .line_profile import (source_birth,weighted_density,line_content,coupled_source_advance,
                           accumulated_slip_moment,
                           advance as advance_line_profile,translate_to_new_tip)
from .path_policy import partial_path_increment


@dataclass
class Prediction:
    status: str
    history: list[HistoryPoint]
    events: list[dict]
    load_steps: int
    maximum_hazard_step: float
    numerical_failure: str | None = None
    final_state: dict | None = None
    remaining_interval_hazard_action: float = 0.0
    remaining_interval_threshold_action: float = 1.0
    mechanism_history: list[dict] | None = None

    def as_dict(self) -> dict:
        return {"status":self.status,"history":[asdict(p) for p in self.history],
                "events":self.events,"load_steps":self.load_steps,
                "maximum_hazard_step":self.maximum_hazard_step,
                "numerical_failure":self.numerical_failure,
                "remaining_interval_hazard_action":self.remaining_interval_hazard_action,
                "remaining_interval_threshold_action":self.remaining_interval_threshold_action}


def _plastic_displacement_rate(zero, candidate, physics, reduced, state,
                               specimen: Specimen, force_N: float, T_K: float) -> float:
    """Orowan slip projected by a unit-force Mode-I process-zone stress.

    gamma_dot=b*rho_mobile*v_P. Virtual work gives du_pl/dt as the volume
    integral of unit-force resolved stress times gamma_dot. The 0-D state is
    uniform over its source-zone length, so V=B*state_strip_width*L_source.
    The scalar unit-force stress is averaged as kappa/sqrt(2*pi*(r+x)).
    This is an observation closure, not a change to native kinetics.
    """
    if force_N <= 0 or not np.any(state.mobile_m2 > 0):
        return 0.0
    geometry = zero._state_geometry(candidate,physics,reduced,state)
    radius = max(float(geometry["tip_radius_m"]),physics.b_m)
    K = specimen.current_K_MPa_sqrt_m(force_N,specimen.a0_m+state.extension_m)
    sigma = K*1e6/sqrt(2*pi*radius)
    factors = np.abs(zero._emission_factors(physics,state.extension_m))
    forest = max(physics.rho_forest_floor_m2+
                 float(np.sum(state.mobile_m2+state.retained_m2)),1.0)
    spacing = 1/(2*sqrt(forest))
    jump = physics.jump_fraction_of_forest_spacing*spacing
    stress = candidate.peierls.stress_fraction*factors*sigma
    rate = zero._arrhenius_rate(candidate.peierls.surface(candidate.emission),
                                np.maximum(stress,0),T_K,candidate.peierls.nu0_s)
    velocity = physics.mobile_transport_velocity_scale*jump*rate
    mobile=np.broadcast_to(np.maximum(state.mobile_m2,0)[:,None],
                           (len(state.mobile_m2),reduced.source_bins))
    speeds=np.broadcast_to(velocity[:,None],mobile.shape)
    return transport_plastic_velocity(specimen,specimen.a0_m+state.extension_m,
        radius,physics.b_m,reduced.state_strip_width_m,reduced.dx_m,
        mobile,speeds,factors)


def _conservative_residence_after_native(zero,candidate,physics,reduced,
                                          old_state,new_state,specimen,K_mid,T_K,dt):
    """Couple the unchanged source root to conservative residence at frozen drive.

    Native scalar evolution supplies time bookkeeping; this adapter replaces
    its source/transport split with the same Arrhenius root evaluated on the
    transported density response. Coefficients refresh each physical substep.
    """
    if dt<=0:return 0.0
    shape=(physics.n_systems,2,physics.n_bins)
    if not hasattr(old_state,"ast_accumulated_slip_profile_m2"):
        raise RuntimeError("nonzero/restarted reduced state lacks accumulated-slip profile")
    slip=np.array(old_state.ast_accumulated_slip_profile_m2,copy=True)
    if slip.shape!=shape:
        raise ValueError("invalid accumulated-slip profile shape")
    cell_area=reduced.dx_m*reduced.state_strip_width_m
    geometry=zero._state_geometry(candidate,physics,reduced,old_state)
    radius=max(float(geometry["tip_radius_m"]),physics.b_m)
    sigma=max(float(K_mid),0)*1e6/sqrt(2*pi*radius)
    factors=np.abs(zero._emission_factors(physics,old_state.extension_m))
    forest=max(float(physics.rho_forest_floor_m2)+
               float(np.sum(old_state.mobile_m2+old_state.retained_m2)),1.0)
    spacing=1/(2*sqrt(forest))
    jump=physics.jump_fraction_of_forest_spacing*spacing
    external=factors*sigma
    p_surface=candidate.peierls.surface(candidate.emission)
    p_stress=candidate.peierls.stress_fraction*external
    p_rate=zero._arrhenius_rate(p_surface,np.maximum(p_stress,0),T_K,
                               candidate.peierls.nu0_s)
    peierls_velocity=jump*np.asarray(p_rate,float)
    velocity=physics.mobile_transport_velocity_scale*peierls_velocity
    mfp=physics.mean_free_path_coefficient/sqrt(forest)
    encounter=(max(physics.encounter_efficiency,0)*np.abs(peierls_velocity)/
               max(mfp,1e-30))
    t_surface=candidate.taylor.surface(candidate.emission)
    phi=spacing/max(physics.b_m,1e-30)
    if np.isfinite(physics.taylor_phi_max):phi=min(phi,physics.taylor_phi_max)
    t_stress=candidate.taylor.stress_fraction*external*phi
    t_single=np.asarray(zero._arrhenius_rate(t_surface,np.maximum(t_stress,0),
                                  T_K,candidate.taylor.nu0_s),float)
    corr_length=candidate.taylor_corr_scale/(2*sqrt(max(candidate.taylor_corr_rho_c_m2,1e-300)))
    release=t_single/max(1+2*corr_length*sqrt(forest),1)
    encounter=np.broadcast_to(encounter,old_state.mobile_m2.shape)
    release=np.broadcast_to(release,old_state.mobile_m2.shape)
    mobile=np.array(getattr(old_state,"ast_mobile_profile_m2",np.zeros(shape)),copy=True)
    retained=np.array(getattr(old_state,"ast_retained_profile_m2",np.zeros(shape)),copy=True)
    x=(np.arange(physics.n_bins)+.5)*reduced.dx_m
    unit=unit_force_resolved_stress(specimen,specimen.a0_m+old_state.extension_m,
                                    radius,x,factors)
    resolved=max(abs(float(candidate.taylor.stress_fraction)),1e-6)
    kback=max(float(physics.persistent_backstress_scale),0.)*physics.G_Pa*abs(physics.b_m)/resolved
    rates=[lambda stress:float(zero._arrhenius_rate(candidate.emission,stress,T_K,
                    physics.emission_nu0_s)) if stress>0 else 0.
           for _ in range(physics.n_systems)]
    from .implicit_source import solve_backstress_limited_activations
    result=coupled_source_advance(mobile,retained,velocity,encounter,release,dt,
        reduced.dx_m,cell_area,unit,physics.b_m,specimen.B_m,
        physics.activation_to_line_content_per_system,physics.emission_signs,
        reduced.source_bins,float(geometry["multiplicity_per_system"]),
        factors*sigma,kback,rates,solve_backstress_limited_activations)
    dA=result["activations"]
    slip_birth=source_birth(slip,dA,physics.activation_to_line_content_per_system,
                            physics.emission_signs,reduced.source_bins,cell_area)
    new_state.ast_accumulated_slip_profile_m2=slip
    new_state.local_slip_count_by_system=accumulated_slip_moment(
        slip,reduced.dx_m,reduced.state_strip_width_m)
    new_state.ast_slip_total_births=float(getattr(old_state,"ast_slip_total_births",0.))+float(np.sum(slip_birth))
    new_state.ast_slip_translation_losses=float(getattr(old_state,"ast_slip_translation_losses",0.))
    new_state.cumulative_activations=old_state.cumulative_activations+dA
    new_state.ast_mobile_profile_m2=mobile
    new_state.ast_retained_profile_m2=retained
    new_state.mobile_m2,new_state.retained_m2=weighted_density(
        mobile,retained,reduced.dx_m,reduced.state_strip_width_m)
    displacement=result["work"]
    for name,increment in (("ast_emitted_line",result["birth"]),
                           ("ast_escaped_line",result["escape"]),
                           ("ast_captured_line",result["capture"]),
                           ("ast_released_line",result["release"]),
                           ("ast_glide_line_distance_m",result["path"]),
                           ("ast_transport_upl_m",displacement)):
        setattr(new_state,name,float(getattr(old_state,name,0.0))+float(increment))
    new_state.ast_current_line_content=result["resident"]
    return displacement


def predict_reduced(zero, candidate, physics, specimen: Specimen, T_K: float,
                    Kdot_MPa_sqrt_m_per_s: float, assumptions,
                    *, event_seed: int | None = None,
                    deterministic_threshold: bool = False,
                    required_u_m: float = 0.0,
                    required_extension_m: float | None = None) -> Prediction:
    """Run accepted state updates with force equilibrium and localized hazard.

    A fixed physical event length is part of this declared model convention.
    The displacement increment is numerical and can be refined independently.
    Limits return numerical/domain statuses, never experimental censoring.
    """
    if specimen.bending_opening_shape is not None:
        raise ValueError("nominal bending field is qualified only in the native 1-D evaluator")
    physics.validate()
    reduced = zero.reduction_geometry(physics)
    nsys = physics.n_systems
    state = zero.ZeroDState(np.zeros(nsys),np.zeros(nsys),np.zeros(nsys),
                            cumulative_activations=np.zeros(nsys),
                            max_tip_radius_m=physics.r0_m)
    state.ast_mobile_profile_m2=np.zeros((nsys,2,physics.n_bins))
    state.ast_retained_profile_m2=np.zeros((nsys,2,physics.n_bins))
    state.ast_accumulated_slip_profile_m2=np.zeros((nsys,2,physics.n_bins))
    rng = np.random.default_rng(assumptions.event_seed if event_seed is None else event_seed)
    threshold = 1.0 if deterministic_threshold else float(rng.exponential())
    action = 0.0
    accepted_path = 0.0
    u = 0.0
    upl = 0.0
    time = 0.0
    force = 0.0
    extension = 0.0
    speed = specimen.initial_displacement_rate(Kdot_MPa_sqrt_m_per_s)
    du_nominal = (assumptions.nominal_K_step_MPa_sqrt_m*1e6*
                  (specimen.initial_compliance_m_per_N+specimen.instrument_compliance_m_per_N) /
                  specimen.geometry_factor(specimen.a0_m))
    # The declared loading window is measured in the initial elastic K ramp.
    # A plastic trajectory can remain at low force while the actuator advances;
    # lack of a crack event by this finite window is an observed nonattainment.
    horizon_u = (assumptions.maximum_nominal_K_MPa_sqrt_m*1e6*
                 (specimen.initial_compliance_m_per_N+specimen.instrument_compliance_m_per_N) /
                 specimen.geometry_factor(specimen.a0_m))
    history = [HistoryPoint(0,0,0,specimen.a0_m,0,0)]
    events = []
    max_dH = 0.0
    steps = 0
    diagnostic_stride=int(os.environ.get("AST2018_MECHANISM_DIAGNOSTIC_STRIDE","0"))
    if diagnostic_stride<0:raise ValueError("nonnegative mechanism diagnostic stride required")
    mechanism_history=[]

    def record_mechanism(z,phase,dt,old_activations,hazard_value=None):
        """Read-only rates at an accepted state; no update to source or clock."""
        if diagnostic_stride==0:return
        try:
            geometry=zero._state_geometry(candidate,physics,reduced,z)
            radius=max(float(geometry["tip_radius_m"]),physics.b_m)
            K=specimen.current_K_MPa_sqrt_m(force,specimen.a0_m+extension)
            sigma=max(K,0)*1e6/sqrt(2*pi*radius)
            factors=np.abs(zero._emission_factors(physics,z.extension_m))
            external=factors*sigma
            back=np.asarray(geometry["sigma_back_by_system_Pa"],float)
            emission_stress=np.maximum(external-back,0.)
            emission=np.asarray(zero._arrhenius_rate(candidate.emission,emission_stress,
                T_K,physics.emission_nu0_s),float)
            forest=max(physics.rho_forest_floor_m2+
                float(np.sum(z.mobile_m2+z.retained_m2)),1.)
            spacing=1/(2*sqrt(forest))
            jump=physics.jump_fraction_of_forest_spacing*spacing
            p_stress=candidate.peierls.stress_fraction*external
            glide=np.asarray(zero._arrhenius_rate(candidate.peierls.surface(candidate.emission),
                np.maximum(p_stress,0.),T_K,candidate.peierls.nu0_s),float)
            velocity=physics.mobile_transport_velocity_scale*jump*glide
            phi=spacing/max(physics.b_m,1e-30)
            if np.isfinite(physics.taylor_phi_max):phi=min(phi,physics.taylor_phi_max)
            t_stress=candidate.taylor.stress_fraction*external*phi
            release=np.asarray(zero._arrhenius_rate(candidate.taylor.surface(candidate.emission),
                np.maximum(t_stress,0.),T_K,candidate.taylor.nu0_s),float)
            corr=candidate.taylor_corr_scale/(2*sqrt(max(candidate.taylor_corr_rho_c_m2,1e-300)))
            release=release/max(1+2*corr*sqrt(forest),1)
            opening=float(zero._advance_state(candidate,physics,reduced,z,duration_s=0,
                K_MPa_sqrt_m=K,temperature_K=T_K))
            mechanism_history.append({"phase":phase,"step":steps,"time_s":time,
                "u_m":u,"extension_m":extension,"K_local_MPa_sqrt_m":K,
                "tip_radius_m":radius,"front_width_m":float(geometry["front_width_m"]),
                "backstress_mean_Pa":float(np.mean(back)),
                "opening_effective_rate_s":opening,
                "emission_rate_mean_s":float(np.mean(emission)),
                "glide_rate_mean_s":float(np.mean(glide)),
                "release_rate_mean_s":float(np.mean(release)),
                "residence_time_median_s":float(np.median(physics.mpz_length_m/
                    np.maximum(velocity,1e-300))),
                "accepted_source_activation_rate_s":float(np.sum(
                    z.cumulative_activations-old_activations)/max(dt,1e-30)),
                "mobile_density_sum_m2":float(np.sum(z.mobile_m2)),
                "retained_density_sum_m2":float(np.sum(z.retained_m2)),
                "hazard_action":action if hazard_value is None else hazard_value,
                "threshold_action":threshold})
        except Exception as exc:
            mechanism_history.append({"phase":phase,"step":steps,
                "time_s":time,"diagnostic_error":f"{type(exc).__name__}: {exc}"})

    def translate_fields(znew,distance):
        if distance<=0:return
        area=reduced.dx_m*reduced.state_strip_width_m
        before=line_content(znew.ast_mobile_profile_m2,znew.ast_retained_profile_m2,area)
        translate_to_new_tip(znew.ast_mobile_profile_m2,znew.ast_retained_profile_m2,
                             distance,reduced.dx_m)
        after=line_content(znew.ast_mobile_profile_m2,znew.ast_retained_profile_m2,area)
        znew.ast_translation_exit_line=float(getattr(znew,"ast_translation_exit_line",0.))+before-after
        znew.ast_current_line_content=after
        znew.mobile_m2,znew.retained_m2=weighted_density(
            znew.ast_mobile_profile_m2,znew.ast_retained_profile_m2,
            reduced.dx_m,reduced.state_strip_width_m)
        slip_before=float(np.sum(znew.ast_accumulated_slip_profile_m2)*area)
        translate_to_new_tip(znew.ast_accumulated_slip_profile_m2,
                             np.zeros_like(znew.ast_accumulated_slip_profile_m2),
                             distance,reduced.dx_m)
        slip_after=float(np.sum(znew.ast_accumulated_slip_profile_m2)*area)
        znew.ast_slip_translation_losses=float(getattr(znew,"ast_slip_translation_losses",0.))+slip_before-slip_after
        znew.local_slip_count_by_system=accumulated_slip_moment(
            znew.ast_accumulated_slip_profile_m2,reduced.dx_m,reduced.state_strip_width_m)

    def trial(fraction, old_state, du, old_accepted_path):
        dt = fraction*du/speed
        unew = u+fraction*du
        # Picard iteration couples the native state update to scalar force
        # equilibrium. In an elastic step the first pass is already exact.
        elastic_force = specimen.equilibrium_force(unew,upl,specimen.a0_m+extension)
        def evaluate_force(Fguess):
            znew=copy.deepcopy(old_state)
            K0=specimen.current_K_MPa_sqrt_m(force,specimen.a0_m+extension)
            Kguess=specimen.current_K_MPa_sqrt_m(Fguess,specimen.a0_m+extension)
            predicted=.5*(zero._advance_state(candidate,physics,reduced,old_state,
                duration_s=0,K_MPa_sqrt_m=K0,temperature_K=T_K)+
                zero._advance_state(candidate,physics,reduced,old_state,
                duration_s=0,K_MPa_sqrt_m=Kguess,temperature_K=T_K))*dt
            da_step=partial_path_increment(action,predicted,threshold,
                old_accepted_path,assumptions.event_advance_m)
            K_mid = specimen.current_K_MPa_sqrt_m(
                0.5*(force+Fguess),specimen.a0_m+extension)
            max_h=float(os.environ.get("AST2018_REDUCED_MAX_SOURCE_SUBSTEP_S","0.1"))
            if not np.isfinite(max_h) or max_h<=0:
                raise ValueError("invalid reduced source substep")
            transport=birth=0.
            remaining=dt
            elapsed=0.
            while remaining>1e-14:
                h=min(remaining,max_h)
                previous=copy.deepcopy(znew)
                transport+=_conservative_residence_after_native(zero,candidate,physics,
                    reduced,previous,znew,specimen,K_mid,T_K,h)
                znew.time_s=previous.time_s+h
                birth+=birth_plastic_displacement(
                    znew.cumulative_activations-previous.cumulative_activations,
                    physics.activation_to_line_content_per_system,physics,specimen,
                    specimen.a0_m+extension,
                    zero._state_geometry(candidate,physics,reduced,previous)["tip_radius_m"],
                    zero._emission_factors(physics,previous.extension_m))
                znew.ast_source_substeps=int(getattr(previous,"ast_source_substeps",0))+1
                translate_fields(znew,da_step*h/dt)
                geometry=zero._state_geometry(candidate,physics,reduced,znew)
                znew.max_tip_radius_m=max(znew.max_tip_radius_m,float(geometry["tip_radius_m"]))
                znew.max_backstress_Pa=max(znew.max_backstress_Pa,float(np.max(geometry["sigma_back_by_system_Pa"])))
                znew.min_front_width_m=min(znew.min_front_width_m,float(geometry["front_width_m"]))
                znew.max_multiplicity=max(znew.max_multiplicity,float(geometry["multiplicity_per_system"]))
                elapsed+=h
                remaining-=h
            znew.ast_birth_upl_m=float(getattr(old_state,"ast_birth_upl_m",0.0))+birth
            uplnew = upl+transport+birth
            Feq=specimen.equilibrium_force(unew,uplnew,specimen.a0_m+extension)
            return znew,uplnew,Feq,da_step
        Fnew=elastic_force
        for iteration in range(6):
            Fused=Fnew
            znew,uplnew,Fnew,da_step=evaluate_force(Fused)
            if abs(Fnew-Fused)<=1e-5*max(Fnew,1e-12):break
        else:
            # Strong plastic feedback can make Picard oscillate. Solve the
            # bounded scalar equilibrium directly, replaying the same native
            # old state at every trial force; no state is committed early.
            lo,hi=0.0,elastic_force
            zlo,ulo,flo,_=evaluate_force(lo)
            zhi,uhi,fhi,_=evaluate_force(hi)
            if flo<lo or fhi>hi+1e-14:
                raise ArithmeticError("scalar force equilibrium is not bracketed")
            # The bracket can span millinewtons while the plastic force root
            # lies below a micronewton. A fixed 24 halvings then cannot reach
            # the declared relative residual; keep bisecting until it does.
            for _ in range(80):
                mid=.5*(lo+hi)
                zmid,umid,fmid,dmid=evaluate_force(mid)
                if fmid>mid:lo=mid
                else:hi=mid
                if abs(fmid-mid)<=1e-5*max(fmid,1e-12):break
                if mid==lo==hi:
                    raise ArithmeticError("bounded force root lost floating-point resolution")
            znew,uplnew,Fnew,da_step=zmid,umid,fmid,dmid
            if abs(Fnew-mid)>1e-4*max(Fnew,1e-12):
                raise ArithmeticError("bounded force root did not converge")
        K0 = specimen.current_K_MPa_sqrt_m(force,specimen.a0_m+extension)
        K1 = specimen.current_K_MPa_sqrt_m(Fnew,specimen.a0_m+extension)
        r0 = zero._advance_state(candidate,physics,reduced,old_state,
                                 duration_s=0,K_MPa_sqrt_m=K0,temperature_K=T_K)
        r1 = zero._advance_state(candidate,physics,reduced,znew,
                                 duration_s=0,K_MPa_sqrt_m=K1,temperature_K=T_K)
        return znew,unew,uplnew,Fnew,dt,0.5*(r0+r1)*dt,da_step

    try:
        target_extension = (assumptions.target_extension_m if required_extension_m is None
                            else required_extension_m)
        while extension < target_extension-1e-15 or u < required_u_m-1e-15:
            if u >= horizon_u-1e-15:
                return Prediction("model_limit_nominal_K_window",history,events,steps,max_dH,
                    remaining_interval_hazard_action=action,
                    remaining_interval_threshold_action=threshold,
                    mechanism_history=mechanism_history)
            if steps >= assumptions.maximum_load_steps:
                return Prediction("numerical_maximum_steps",history,events,steps,max_dH,
                    remaining_interval_hazard_action=action,
                    remaining_interval_threshold_action=threshold,
                    mechanism_history=mechanism_history)
            if specimen.nominal_K_MPa_sqrt_m(force) >= assumptions.maximum_nominal_K_MPa_sqrt_m:
                return Prediction("load_map_exhausted",history,events,steps,max_dH,
                    remaining_interval_hazard_action=action,
                    remaining_interval_threshold_action=threshold,
                    mechanism_history=mechanism_history)
            steps += 1
            old_state = copy.deepcopy(state)
            displacement_remainder=(required_u_m-u if required_u_m>u+1e-15
                                    else float("inf"))
            du = min(du_nominal,horizon_u-u,displacement_remainder)
            znew,unew,uplnew,Fnew,dt,dH,da_step = trial(1,old_state,du,accepted_path)
            max_dH = max(max_dH,dH)
            if dH < 0 or not np.isfinite(dH):
                raise ArithmeticError("nonfinite or negative hazard increment")
            if action+dH >= threshold:
                lo,hi = 0.0,1.0
                for _ in range(assumptions.localization_bisections):
                    mid = 0.5*(lo+hi)
                    result = trial(mid,old_state,du,accepted_path)
                    if action+result[5] >= threshold: hi = mid
                    else: lo = mid
                znew,unew,uplnew,Fnew,dt,dH,da_step = trial(hi,old_state,du,accepted_path)
                state,u,upl,force = znew,unew,uplnew,Fnew
                time += dt
                if diagnostic_stride:
                    record_mechanism(state,"pre_event",dt,old_state.cumulative_activations,
                                     action+dH)
                K = specimen.current_K_MPa_sqrt_m(force,specimen.a0_m+extension)
                if extension+assumptions.event_advance_m >= specimen.W_m-specimen.a0_m:
                    history.append(HistoryPoint(time,u,force,specimen.a0_m+extension,
                                                upl,K,physical_failure=True))
                    events.append({"index":len(events),"time_s":time,
                                   "threshold_action":threshold,"hazard_action":action+dH,
                                   "terminal_ligament_failure":True,
                                   "proposed_advance_m":assumptions.event_advance_m,
                                   "cumulative_extension_m":extension,
                                   "K_pre_MPa_sqrt_m":K})
                    return Prediction("physical_ligament_failure",history,events,steps,max_dH,
                        remaining_interval_hazard_action=action+dH,
                        remaining_interval_threshold_action=threshold,
                        mechanism_history=mechanism_history)
                history.append(HistoryPoint(time,u,force,specimen.a0_m+extension,upl,K))
                pre = zero._state_geometry(candidate,physics,reduced,state)
                extension += assumptions.event_advance_m
                translate_fields(state,max(assumptions.event_advance_m-
                                           accepted_path-da_step,0.))
                state.extension_m += assumptions.event_advance_m
                force = specimen.equilibrium_force(u,upl,specimen.a0_m+extension)
                Kpost = specimen.current_K_MPa_sqrt_m(force,specimen.a0_m+extension)
                history.append(HistoryPoint(time,u,force,specimen.a0_m+extension,upl,Kpost,event=True))
                events.append({"index":len(events),"time_s":time,"threshold_action":threshold,
                               "hazard_action":action+dH,"cumulative_extension_m":extension,
                               "event_advance_m":assumptions.event_advance_m,
                               "K_pre_MPa_sqrt_m":K,"K_post_MPa_sqrt_m":Kpost,
                               "tip_radius_pre_m":pre["tip_radius_m"],
                               "front_width_pre_m":pre["front_width_m"],
                               "source_multiplicity_pre":pre["multiplicity_per_system"],
                               "backstress_pre_Pa":float(np.mean(pre["sigma_back_by_system_Pa"])),
                               "cumulative_birth_plastic_u_m":float(getattr(state,"ast_birth_upl_m",0.0)),
                               "cumulative_transport_plastic_u_m":float(getattr(state,"ast_transport_upl_m",0.0)),
                               "cumulative_emitted_line":float(getattr(state,"ast_emitted_line",0.0)),
                               "cumulative_escaped_line":float(getattr(state,"ast_escaped_line",0.0)),
                               "cumulative_captured_line":float(getattr(state,"ast_captured_line",0.0)),
                               "cumulative_released_line":float(getattr(state,"ast_released_line",0.0)),
                               "cumulative_glide_line_distance_m":float(getattr(state,"ast_glide_line_distance_m",0.0)),
                               "current_line_content":float(getattr(state,"ast_current_line_content",0.0)),
                               "cumulative_translation_exit_line":float(getattr(state,"ast_translation_exit_line",0.0)),
                               "cumulative_activations":float(np.sum(state.cumulative_activations)),
                               "slip_total_births":float(getattr(state,"ast_slip_total_births",0.)),
                               "slip_remaining_content":float(np.sum(state.ast_accumulated_slip_profile_m2)*
                                   reduced.dx_m*reduced.state_strip_width_m),
                               "slip_translation_losses":float(getattr(state,"ast_slip_translation_losses",0.)),
                               "slip_moment_by_system":state.local_slip_count_by_system.tolist(),
                               "cumulative_source_substeps":int(getattr(state,"ast_source_substeps",0)),
                               "tip_radius_post_m":float(zero._state_geometry(candidate,physics,reduced,state)["tip_radius_m"])})
                action = 0.0
                accepted_path=0.0
                threshold = 1.0 if deterministic_threshold else float(rng.exponential())
            else:
                state,u,upl,force = znew,unew,uplnew,Fnew
                time += dt
                action += dH
                accepted_path += da_step
                if diagnostic_stride and steps%diagnostic_stride==0:
                    record_mechanism(state,"accepted",dt,old_state.cumulative_activations)
                K = specimen.current_K_MPa_sqrt_m(force,specimen.a0_m+extension)
                history.append(HistoryPoint(time,u,force,specimen.a0_m+extension,upl,K))
        geometry=zero._state_geometry(candidate,physics,reduced,state)
        final_state={"mobile_m2":state.ast_mobile_profile_m2.copy(),
                     "retained_m2":state.ast_retained_profile_m2.copy(),
                     "accumulated_slip_m2":state.ast_accumulated_slip_profile_m2.copy(),
                     "cumulative_activations":state.cumulative_activations.copy(),
                     "tip_radius_m":float(geometry["tip_radius_m"]),
                     "source_substeps":int(getattr(state,"ast_source_substeps",0)),
                     "accepted_duration_s":float(state.time_s)}
        return Prediction("complete",history,events,steps,max_dH,final_state=final_state,
            remaining_interval_hazard_action=action,
            remaining_interval_threshold_action=threshold,
            mechanism_history=mechanism_history)
    except Exception as exc:
        return Prediction("numerical_failure",history,events,steps,max_dH,
                          f"{type(exc).__name__}: {exc}",
                          remaining_interval_hazard_action=action,
                          remaining_interval_threshold_action=threshold,
                          mechanism_history=mechanism_history)
