"""Coupled scalar cantilever / native stateful 0-D first-passage integration."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from math import pi, sqrt

import numpy as np

from .mechanics import HistoryPoint, Specimen


@dataclass
class Prediction:
    status: str
    history: list[HistoryPoint]
    events: list[dict]
    load_steps: int
    maximum_hazard_step: float
    numerical_failure: str | None = None

    def as_dict(self) -> dict:
        return {"status":self.status,"history":[asdict(p) for p in self.history],
                "events":self.events,"load_steps":self.load_steps,
                "maximum_hazard_step":self.maximum_hazard_step,
                "numerical_failure":self.numerical_failure}


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
    gamma_dot = physics.b_m*np.maximum(state.mobile_m2,0)*velocity
    source_length=physics.source_zone_length_m
    mean_inverse_root=2*(sqrt(radius+source_length)-sqrt(radius))/source_length
    unit_tau=factors*specimen.geometry_factor(specimen.a0_m+state.extension_m)*(
        mean_inverse_root/sqrt(2*pi))
    volume = specimen.B_m*reduced.state_strip_width_m*physics.source_zone_length_m
    return max(float(np.sum(unit_tau*gamma_dot)*volume),0.0)


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
    physics.validate()
    reduced = zero.reduction_geometry(physics)
    nsys = physics.n_systems
    state = zero.ZeroDState(np.zeros(nsys),np.zeros(nsys),np.zeros(nsys),
                            cumulative_activations=np.zeros(nsys),
                            max_tip_radius_m=physics.r0_m)
    rng = np.random.default_rng(assumptions.event_seed if event_seed is None else event_seed)
    threshold = 1.0 if deterministic_threshold else float(rng.exponential())
    action = 0.0
    u = 0.0
    upl = 0.0
    time = 0.0
    force = 0.0
    extension = 0.0
    speed = specimen.initial_displacement_rate(Kdot_MPa_sqrt_m_per_s)
    du_nominal = (assumptions.nominal_K_step_MPa_sqrt_m*1e6*
                  (specimen.initial_compliance_m_per_N+specimen.instrument_compliance_m_per_N) /
                  specimen.geometry_factor(specimen.a0_m))
    history = [HistoryPoint(0,0,0,specimen.a0_m,0,0)]
    events = []
    max_dH = 0.0
    steps = 0

    def trial(fraction, old_state, du):
        dt = fraction*du/speed
        unew = u+fraction*du
        # Picard iteration couples the native state update to scalar force
        # equilibrium. In an elastic step the first pass is already exact.
        elastic_force = specimen.equilibrium_force(unew,upl,specimen.a0_m+extension)
        def evaluate_force(Fguess):
            znew=copy.deepcopy(old_state)
            K_mid = specimen.current_K_MPa_sqrt_m(
                0.5*(force+Fguess),specimen.a0_m+extension)
            zero._advance_state(candidate,physics,reduced,znew,
                                duration_s=dt,K_MPa_sqrt_m=K_mid,
                                temperature_K=T_K)
            v0 = _plastic_displacement_rate(zero,candidate,physics,reduced,old_state,
                                             specimen,force,T_K)
            v1 = _plastic_displacement_rate(zero,candidate,physics,reduced,znew,
                                             specimen,Fguess,T_K)
            uplnew = upl+0.5*(v0+v1)*dt
            Feq=specimen.equilibrium_force(unew,uplnew,specimen.a0_m+extension)
            return znew,uplnew,Feq
        Fnew=elastic_force
        for iteration in range(6):
            Fused=Fnew
            znew,uplnew,Fnew=evaluate_force(Fused)
            if abs(Fnew-Fused)<=1e-5*max(Fnew,1e-12):break
        else:
            # Strong plastic feedback can make Picard oscillate. Solve the
            # bounded scalar equilibrium directly, replaying the same native
            # old state at every trial force; no state is committed early.
            lo,hi=0.0,elastic_force
            zlo,ulo,flo=evaluate_force(lo)
            zhi,uhi,fhi=evaluate_force(hi)
            if flo<lo or fhi>hi+1e-14:
                raise ArithmeticError("scalar force equilibrium is not bracketed")
            for _ in range(24):
                mid=.5*(lo+hi)
                zmid,umid,fmid=evaluate_force(mid)
                if fmid>mid:lo=mid
                else:hi=mid
                if abs(fmid-mid)<=1e-5*max(fmid,1e-12):break
            znew,uplnew,Fnew=zmid,umid,fmid
            if abs(Fnew-mid)>1e-4*max(Fnew,1e-12):
                raise ArithmeticError("bounded force root did not converge")
        K0 = specimen.current_K_MPa_sqrt_m(force,specimen.a0_m+extension)
        K1 = specimen.current_K_MPa_sqrt_m(Fnew,specimen.a0_m+extension)
        r0 = zero._advance_state(candidate,physics,reduced,old_state,
                                 duration_s=0,K_MPa_sqrt_m=K0,temperature_K=T_K)
        r1 = zero._advance_state(candidate,physics,reduced,znew,
                                 duration_s=0,K_MPa_sqrt_m=K1,temperature_K=T_K)
        return znew,unew,uplnew,Fnew,dt,0.5*(r0+r1)*dt

    try:
        target_extension = (assumptions.target_extension_m if required_extension_m is None
                            else required_extension_m)
        while extension < target_extension-1e-15 or u < required_u_m-1e-15:
            if steps >= assumptions.maximum_load_steps:
                return Prediction("numerical_maximum_steps",history,events,steps,max_dH)
            if specimen.nominal_K_MPa_sqrt_m(force) >= assumptions.maximum_nominal_K_MPa_sqrt_m:
                return Prediction("numerical_load_domain_limit",history,events,steps,max_dH)
            steps += 1
            old_state = copy.deepcopy(state)
            du = du_nominal
            znew,unew,uplnew,Fnew,dt,dH = trial(1,old_state,du)
            max_dH = max(max_dH,dH)
            if dH < 0 or not np.isfinite(dH):
                raise ArithmeticError("nonfinite or negative hazard increment")
            if action+dH >= threshold:
                lo,hi = 0.0,1.0
                for _ in range(assumptions.localization_bisections):
                    mid = 0.5*(lo+hi)
                    result = trial(mid,old_state,du)
                    if action+result[-1] >= threshold: hi = mid
                    else: lo = mid
                znew,unew,uplnew,Fnew,dt,dH = trial(hi,old_state,du)
                state,u,upl,force = znew,unew,uplnew,Fnew
                time += dt
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
                    return Prediction("physical_ligament_failure",history,events,steps,max_dH)
                history.append(HistoryPoint(time,u,force,specimen.a0_m+extension,upl,K))
                pre = zero._state_geometry(candidate,physics,reduced,state)
                extension += assumptions.event_advance_m
                retention = zero._translation_retention_factor(
                    assumptions.event_advance_m,physics,reduced)
                state.mobile_m2 *= retention
                state.retained_m2 *= retention
                state.local_slip_count_by_system *= retention
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
                               "cumulative_activations":float(np.sum(state.cumulative_activations))})
                action = 0.0
                threshold = 1.0 if deterministic_threshold else float(rng.exponential())
            else:
                state,u,upl,force = znew,unew,uplnew,Fnew
                time += dt
                action += dH
                K = specimen.current_K_MPa_sqrt_m(force,specimen.a0_m+extension)
                history.append(HistoryPoint(time,u,force,specimen.a0_m+extension,upl,K))
        return Prediction("complete",history,events,steps,max_dH)
    except Exception as exc:
        return Prediction("numerical_failure",history,events,steps,max_dH,
                          f"{type(exc).__name__}: {exc}")
