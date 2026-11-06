"""Coupled cantilever wrapper for the pinned spatial 1-D state engine."""
from __future__ import annotations

import copy
from math import ceil, pi, sqrt

import numpy as np
from scipy.linalg import solve_banded
from .implicit_source import (solve_backstress_limited_activations,
                              mechanically_blocked_at_source_bound)

from .mechanics import (HistoryPoint,Specimen,birth_plastic_displacement,
                        transport_plastic_velocity,unit_force_resolved_stress)
from .reduced import Prediction
from .path_policy import partial_path_increment
from .section_plasticity import (SectionPlasticity, SectionPlasticityConfig,
                                 SectionSaturation)


class SpatialForceRootFailure(ArithmeticError):
    """A trial force root failed; caller may retry at a smaller physical step."""


def _advance_coupled_fixed_schedule(state, duration_s, da_m, K0, K1, T_K,
                                    extension_m, steps):
    """Replay the pinned native operator order with one fixed CFL partition.

    The native routine chooses ``ceil(da / moving_tip_cfl / dx)`` separately
    for each force guess. The wrapper also used a separate no-translation
    branch when ``da`` reached zero. A shared partition and operator path
    makes the equilibrium map consistent across neighboring force guesses.
    """
    max_translation=max(float(state.c.moving_tip_cfl)*float(state.dx),1e-30)
    if steps<1 or da_m/steps>max_translation*(1+1e-12):
        raise SpatialForceRootFailure("fixed spatial translation schedule violates CFL")
    dt=max(float(duration_s),0.0)/steps
    da=max(float(da_m),0.0)/steps
    old_override=state._emission_geometry_extension_override_m
    state._emission_geometry_extension_override_m=max(float(extension_m),0.0)
    state.begin_diagnostic_interval()
    try:
        for j in range(steps):
            Kmid=K0+(j+.5)/steps*(K1-K0)
            if dt>0:state.advance_time(dt,Kmid,T_K)
            if da>0:state.translate_tip(da)
    finally:
        state._emission_geometry_extension_override_m=old_override
    state.interval_tip_radius_end_m=state.tip_radius_m()
    state.interval_tip_radius_max_m=max(
        float(state.interval_tip_radius_max_m),
        float(state.interval_tip_radius_end_m))


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
    cleavage_sigma = (max(state._elastic_external_K(K_MPa_sqrt_m) - shield, 0) * 1e6 /
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
    factor = np.abs(state.emission_drive_factors())[:,None]
    radius=state.tip_radius_m()
    return transport_plastic_velocity(specimen,committed_a_m,radius,
        state.c.b_m,state.state_strip_width_m,state.dx,mobile,velocity,factor[:,0])


TIP_INTERACTIONS = ("native", "signed_pileup")


def _mechanical_state_type(native_state, specimen:Specimen, tip_interaction:str="native"):
    """Integrate the same unit-force Orowan projection at native substeps.

    ``tip_interaction="signed_pileup"`` separates same-sign and mixed-sign
    content. Only mixed-sign content (2 min(+,-) per system) forms the random
    forest that sets the Taylor spacing, jump length, mean free path and
    capture; a same-sign pile-up on the emission plane does not interact
    through the Taylor mechanism, so its motion is Peierls-limited. Its net
    signed content, mobile as well as retained, sets the GND back stress and
    crack-tip shielding. The Taylor amplification stays uncapped.
    """
    if tip_interaction not in TIP_INTERACTIONS:
        raise ValueError(f"unknown tip interaction {tip_interaction!r}")
    class AstMechanicalState(native_state):
        def __init__(self,candidate,physics):
            super().__init__(candidate,physics)
            self.ast_transport_upl_m=0.0
            self.ast_birth_upl_m=0.0
            self.ast_emitted_line_per_m=0.0
            self.ast_escaped_line_per_m=0.0
            self.ast_captured_line_per_m=0.0
            self.ast_released_line_per_m=0.0
            self.ast_recovered_line_per_m=0.0
            self.ast_annihilated_line_per_m=0.0
            self.ast_translation_exit_line_per_m=0.0
            self.ast_glide_distance_density_integral_per_m=0.0
            self.ast_max_line_balance_residual_per_m=0.0
            self.ast_committed_a_m=specimen.a0_m

        def signed_gnd_m2(self):
            if tip_interaction=="native":
                return super().signed_gnd_m2()
            content=self.mobile_m2+self.retained_m2
            return content[:,1,:]-content[:,0,:]

        def forest_density_m2(self):
            if tip_interaction=="native":
                return super().forest_density_m2()
            content=np.maximum(self.mobile_m2+self.retained_m2,0)
            mixed=2*np.minimum(content[:,0,:],content[:,1,:])
            interaction=np.asarray(self.c.forest_interaction_matrix,float)
            return np.maximum(self.c.rho_forest_floor_m2+interaction@mixed,1.0)

        def _elastic_external_K(self,K_MPa_sqrt_m,*,neutral=False):
            radius=self.c.r0_m if neutral else self.tip_radius_m()
            return (max(float(K_MPa_sqrt_m),0.0)*
                    specimen.opening_shape(self.ast_committed_a_m,radius))

        def local_rates(self,K_MPa_sqrt_m,T_K):
            if specimen.bending_opening_shape is None:
                return super().local_rates(K_MPa_sqrt_m,T_K)
            # The tip-source Arrhenius law retains the native one-activation
            # treatment, driven by the FE-normalized stress at the tip. The
            # same field's signed ahead-tip profile drives native glide and
            # release in every spatial bin. All internal kernels are native.
            rates=super().local_rates(self._elastic_external_K(K_MPa_sqrt_m),T_K)
            radius=self.tip_radius_m()
            x=np.asarray(self.x,float)
            shape=np.array([specimen.opening_shape(self.ast_committed_a_m,radius+xi)
                            for xi in x])
            applied=max(float(K_MPa_sqrt_m),0.)*1e6*shape/np.sqrt(2*pi*(radius+x))
            shield=self.K_shield_MPa_sqrt_m()*1e6/sqrt(2*pi*radius)
            factors=self.emission_drive_factors()[:,None]
            tau_applied=factors*applied[None,:]
            tau_shielded=tau_applied-factors*shield
            tau_eff=tau_shielded+rates["tau_gnd_Pa"]
            forest=self.forest_density_m2()
            spacing=1/(2*np.sqrt(forest))
            jump=self.c.jump_fraction_of_forest_spacing*spacing
            p_surface=self.p.peierls.surface(self.p.emission)
            peierls=self._signed_rate(p_surface,
                self.p.peierls.stress_fraction*tau_eff,T_K,self.p.peierls.nu0_s)
            peierls_velocity=jump*peierls
            velocity=max(float(self.c.mobile_transport_velocity_scale),0.)*peierls_velocity
            t_surface=self.p.taylor.surface(self.p.emission)
            phi=spacing/max(self.c.b_m,1e-30)
            if np.isfinite(self.c.taylor_phi_max):
                phi=np.minimum(phi,self.c.taylor_phi_max)
            taylor_single=np.maximum(self._signed_rate(t_surface,
                self.p.taylor.stress_fraction*tau_eff*phi,T_K,
                self.p.taylor.nu0_s),0.)
            corr=self.p.taylor_corr_scale/(2*sqrt(max(self.p.taylor_corr_rho_c_m2,1e-300)))
            taylor=taylor_single/np.maximum(1+2*corr*np.sqrt(forest),1.)
            mfp=self.c.mean_free_path_coefficient/np.sqrt(forest)
            encounter=max(float(self.c.encounter_efficiency),0.)*np.abs(peierls_velocity)/np.maximum(mfp,1e-30)
            rates.update({"velocity_m_s":velocity,
                "peierls_velocity_m_s":peierls_velocity,
                "taylor_completion_s":taylor,"encounter_s":encounter,
                "tau_external_Pa":tau_applied,
                "tau_nonlocal_shielding_Pa":tau_shielded-tau_applied,
                "tau_eff_Pa":tau_eff})
            return rates

        def cleavage_rate_s(self,K_MPa_sqrt_m,T_K,*,neutral=False):
            if specimen.bending_opening_shape is None:
                if neutral:
                    return super().cleavage_rate_s(K_MPa_sqrt_m,T_K,neutral=True)
                return super().cleavage_rate_s(K_MPa_sqrt_m,T_K)
            return super().cleavage_rate_s(
                self._elastic_external_K(K_MPa_sqrt_m,neutral=neutral),
                T_K,neutral=neutral)

        def _plastic_power_snapshot(self,rates):
            power=super()._plastic_power_snapshot(rates)
            mobile=np.sum(np.maximum(self.mobile_m2,0),axis=1)
            power["ast_transport_velocity_m_s"]=transport_plastic_velocity(
                specimen,self.ast_committed_a_m,self.tip_radius_m(),self.c.b_m,
                self.state_strip_width_m,self.dx,mobile,
                np.asarray(rates["velocity_m_s"]),self.emission_drive_factors())
            return power

        def _accumulate_work_interval(self,start,end,duration_s):
            # The inherited endpoint trapezoid is inconsistent with the
            # backward-Euler transport solve when mobile content escapes in
            # less than a feedback step. Rewards are integrated below with
            # the same solved occupancy as the conservative state operator.
            del start,end,duration_s

        def _emit_exact(self,rates,dt):
            added=super()._emit_exact(rates,dt)
            self.ast_emitted_line_per_m+=added
            return added

        def _annihilate_exact(self,rates,dt):
            removed=super()._annihilate_exact(rates,dt)
            self.ast_annihilated_line_per_m+=removed
            return removed

        def translate_tip(self,da_m):
            track=all(hasattr(self,name) for name in ("mobile_m2","retained_m2","dx"))
            before=float(np.sum(self.mobile_m2+self.retained_m2)*self.dx) if track else 0.0
            super().translate_tip(da_m)
            if track:
                after=float(np.sum(self.mobile_m2+self.retained_m2)*self.dx)
                self.ast_translation_exit_line_per_m+=max(before-after,0.0)

        def _coupled_mobile_retained(self,rates,dt):
            """Use each BE substep's mobile occupancy for slip and work."""
            if dt<=0:return
            n=max(int(self.coupled_operator_substeps),1)
            h=float(dt)/n
            recovery=max(float(rates["recovery_rate_s"]),0.0)
            velocity_base=np.asarray(rates["velocity_m_s"],float)
            encounter=np.asarray(rates["encounter_s"],float)
            release=np.asarray(rates["taylor_completion_s"],float)
            factors=self.emission_drive_factors()
            unit=unit_force_resolved_stress(specimen,self.ast_committed_a_m,
                self.tip_radius_m(),self.x,factors)
            volume=specimen.B_m*self.state_strip_width_m*self.dx
            area=self.cell_area_m2
            work_fields=(("external_plastic_work_J_per_m","tau_external_Pa"),
                         ("nonlocal_shielding_work_J_per_m","tau_nonlocal_shielding_Pa"),
                         ("internal_stress_work_J_per_m","tau_gnd_Pa"),
                         ("effective_plastic_work_J_per_m","tau_eff_Pa"))
            for system in range(self.c.n_systems):
                for q in range(2):
                    sign=-1.0 if q==0 else 1.0
                    velocity=sign*velocity_base[system]
                    banded=self._coupled_banded_matrix(velocity,encounter[system],
                        release[system],recovery,self.dx,h)
                    state=np.empty(2*self.c.n_bins,float)
                    state[0::2]=np.maximum(self.mobile_m2[system,q],0)
                    state[1::2]=np.maximum(self.retained_m2[system,q],0)
                    face=np.empty(self.c.n_bins+1,float)
                    face[0]=velocity[0];face[-1]=velocity[-1]
                    if self.c.n_bins>1:face[1:-1]=.5*(velocity[:-1]+velocity[1:])
                    for _ in range(n):
                        old_total=float(np.sum(state)*self.dx)
                        state=solve_banded((2,2),banded,state,
                            overwrite_ab=False,overwrite_b=False,check_finite=False)
                        state=np.maximum(state,0.0)
                        mobile=state[0::2];retained=state[1::2]
                        mobile_time=h*mobile
                        signed_travel=velocity_base[system]*mobile_time
                        self.ast_transport_upl_m+=float(np.sum(
                            unit[system]*self.c.b_m*signed_travel)*volume)
                        self.ast_glide_distance_density_integral_per_m+=float(
                            np.sum(np.abs(signed_travel))*self.dx)
                        for attr,key in work_fields:
                            increment=float(np.sum(np.asarray(rates[key])[system]*
                                self.c.b_m*signed_travel)*area)
                            setattr(self,attr,float(getattr(self,attr))+increment)
                        effective=np.asarray(rates["tau_eff_Pa"])[system]
                        density=effective*self.c.b_m*signed_travel
                        self.effective_plastic_dissipation_J_per_m+=float(
                            np.sum(np.maximum(density,0.0))*area)
                        escaped=h*(max(face[-1],0.0)*mobile[-1]+
                                   max(-face[0],0.0)*mobile[0])
                        captured=float(np.sum(h*encounter[system]*mobile)*self.dx)
                        released=float(np.sum(h*release[system]*retained)*self.dx)
                        recovered=float(h*recovery*np.sum(retained)*self.dx)
                        self.ast_escaped_line_per_m+=escaped
                        self.ast_captured_line_per_m+=captured
                        self.ast_released_line_per_m+=released
                        self.ast_recovered_line_per_m+=recovered
                        residual=old_total-float(np.sum(state)*self.dx)-escaped-recovered
                        self.ast_max_line_balance_residual_per_m=max(
                            self.ast_max_line_balance_residual_per_m,abs(residual))
                        if abs(residual)>1e-8*max(old_total,1e-30):
                            raise ArithmeticError("spatial transport lost line-content balance")
                    self.mobile_m2[system,q]=state[0::2]
                    self.retained_m2[system,q]=state[1::2]
            if not (np.all(np.isfinite(self.mobile_m2)) and
                    np.all(np.isfinite(self.retained_m2))):
                raise ArithmeticError("nonfinite conservative transport state")

        def _advance_spatial_step(self,dt,K_MPa_sqrt_m,T_K):
            """Complete frozen-coefficient source/transport steps with native fields."""
            totals={"emitted_per_m":0.,"annihilated_per_m":0.}
            if dt<=0:return totals
            n=max(int(self.coupled_operator_substeps),1)
            h=float(dt)/n
            for _ in range(n):
                rates=self.local_rates(K_MPa_sqrt_m,T_K)
                totals["annihilated_per_m"]+=self._annihilate_exact(rates,.5*h)
                # Annihilation changes forest, GND and source feedback.
                rates=self.local_rates(K_MPa_sqrt_m,T_K)
                geometry=self.source_geometry()
                nsrc=self._source_zone_bin_count()
                weights,norm=self._tip_weights()
                drive=np.asarray(rates["emission_drive_Pa"],float)
                conversion=np.asarray(self.c.activation_to_line_content_per_system,float)
                resolved=max(abs(float(self.p.taylor.stress_fraction)),1e-6)
                kback=(max(float(self.c.persistent_backstress_scale),0.)*
                       self.c.G_Pa*abs(self.c.b_m)/resolved)
                if kback<=0:raise ArithmeticError("nonpositive source backstress")
                M=float(geometry["multiplicity_per_system"])
                base=np.empty((self.c.n_systems,2,2*self.c.n_bins))
                response=np.zeros_like(base)
                rho_base=np.zeros(self.c.n_systems)
                rho_response=np.zeros(self.c.n_systems)
                velocity_base=np.asarray(rates["velocity_m_s"],float)
                encounter=np.asarray(rates["encounter_s"],float)
                release=np.asarray(rates["taylor_completion_s"],float)
                recovery=max(float(rates["recovery_rate_s"]),0.)
                for system in range(self.c.n_systems):
                    source_q=1 if self.c.emission_signs[system]>0 else 0
                    for q in range(2):
                        v=(-1. if q==0 else 1.)*velocity_base[system]
                        matrix=self._coupled_banded_matrix(v,encounter[system],
                            release[system],recovery,self.dx,h)
                        old=np.empty(2*self.c.n_bins)
                        old[0::2]=np.maximum(self.mobile_m2[system,q],0.)
                        old[1::2]=np.maximum(self.retained_m2[system,q],0.)
                        base[system,q]=solve_banded((2,2),matrix,old,
                            overwrite_ab=False,overwrite_b=False,check_finite=False)
                        if q==source_q:
                            packet=np.zeros_like(old)
                            packet[0:2*nsrc:2]=conversion[system]/(nsrc*self.cell_area_m2)
                            response[system,q]=solve_banded((2,2),matrix,packet,
                                overwrite_ab=False,overwrite_b=False,check_finite=False)
                    rho_base[system]=np.sum((base[system,:,0::2]+base[system,:,1::2])*
                                             weights[None,:])/norm
                    rho_response[system]=np.sum((response[system,:,0::2]+
                                               response[system,:,1::2])*weights[None,:])/norm
                A=np.zeros(self.c.n_systems)
                rate_initial=np.zeros_like(A);rate_final=np.zeros_like(A)
                sigma_final=np.zeros_like(A)
                for system in range(self.c.n_systems):
                    def rate_at(stress):
                        return (float(self._arrhenius_rate(
                            self.p.emission.barrier_eV(stress,T_K),T_K,
                            self.c.emission_nu0_s)) if stress>0 else 0.)
                    sigma0=max(float(drive[system])-kback*np.sqrt(max(rho_base[system],0.)),0.)
                    rate_initial[system]=rate_at(sigma0)
                    if M>0 and sigma0>0:
                        A[system]=solve_backstress_limited_activations(
                            multiplicity=M,dt_s=h,drive_stress_Pa=float(drive[system]),
                            rho_initial_m2=float(rho_base[system]),
                            rho_increment_per_activation_m2=float(rho_response[system]),
                            backstress_prefactor_Pa_sqrt_m2=kback,
                            rate_function=rate_at,tolerance=self.c.implicit_tolerance,
                            max_iterations=self.c.implicit_max_iterations)
                    sigma_final[system]=max(float(drive[system])-kback*np.sqrt(
                        max(rho_base[system]+A[system]*rho_response[system],0.)),0.)
                    blocked=mechanically_blocked_at_source_bound(
                        activations=A[system],drive_stress_Pa=drive[system],
                        rho_initial_m2=rho_base[system],
                        rho_increment_per_activation_m2=rho_response[system],
                        backstress_prefactor_Pa_sqrt_m2=kback)
                    if blocked:
                        sigma_final[system]=0.0
                    rate_final[system]=rate_at(sigma_final[system])
                    residual=A[system]-M*h*rate_final[system]
                    if (A[system]>0 and not blocked and
                        abs(residual)>2e-8*max(A[system],M*h*rate_initial[system],1e-30)):
                        raise ArithmeticError("spatial joint source root residual")
                new=base+A[:,None,None]*response
                if np.any(new< -1e-10) or not np.all(np.isfinite(new)):
                    raise ArithmeticError("invalid joint source/transport state")
                new=np.maximum(new,0.)
                factors=self.emission_drive_factors()
                unit=unit_force_resolved_stress(specimen,self.ast_committed_a_m,
                    self.tip_radius_m(),self.x,factors)
                volume=specimen.B_m*self.state_strip_width_m*self.dx
                area=self.cell_area_m2
                work_fields=(("external_plastic_work_J_per_m","tau_external_Pa"),
                             ("nonlocal_shielding_work_J_per_m","tau_nonlocal_shielding_Pa"),
                             ("internal_stress_work_J_per_m","tau_gnd_Pa"),
                             ("effective_plastic_work_J_per_m","tau_eff_Pa"))
                before=float(np.sum(self.mobile_m2+self.retained_m2)*self.dx)
                escape=recovered=0.
                for system in range(self.c.n_systems):
                    for q in range(2):
                        v=velocity_base[system]
                        mobile=new[system,q,0::2];retained=new[system,q,1::2]
                        signed_travel=h*v*mobile
                        self.ast_transport_upl_m+=float(np.sum(unit[system]*self.c.b_m*
                                                             signed_travel)*volume)
                        self.ast_glide_distance_density_integral_per_m+=float(
                            np.sum(np.abs(signed_travel))*self.dx)
                        for attr,key in work_fields:
                            setattr(self,attr,float(getattr(self,attr))+float(np.sum(
                                np.asarray(rates[key])[system]*self.c.b_m*signed_travel)*area))
                        diss=np.asarray(rates["tau_eff_Pa"])[system]*self.c.b_m*signed_travel
                        self.effective_plastic_dissipation_J_per_m+=float(np.sum(np.maximum(diss,0.))*area)
                        face=np.empty(self.c.n_bins+1)
                        face[0]=v[0];face[-1]=v[-1]
                        if self.c.n_bins>1:face[1:-1]=.5*(v[:-1]+v[1:])
                        signed_face=(-1. if q==0 else 1.)*face
                        escape+=h*(max(signed_face[-1],0.)*mobile[-1]+max(-signed_face[0],0.)*mobile[0])
                        self.ast_captured_line_per_m+=float(np.sum(h*encounter[system]*mobile)*self.dx)
                        self.ast_released_line_per_m+=float(np.sum(h*release[system]*retained)*self.dx)
                        recovered+=float(h*recovery*np.sum(retained)*self.dx)
                self.mobile_m2[:]=new[:,: ,0::2]
                self.retained_m2[:]=new[:,: ,1::2]
                born_line=A*conversion
                for system,sign in enumerate(self.c.emission_signs):
                    q=1 if sign>0 else 0
                    self.accumulated_slip_m2[system,q,:nsrc]+=(
                        born_line[system]/(nsrc*self.cell_area_m2))
                after=float(np.sum(self.mobile_m2+self.retained_m2)*self.dx)
                balance=before+float(np.sum(born_line))/self.state_strip_width_m-after-escape-recovered
                self.ast_max_line_balance_residual_per_m=max(
                    self.ast_max_line_balance_residual_per_m,abs(balance))
                if abs(balance)>2e-8*max(before+float(np.sum(born_line))/self.state_strip_width_m,1e-30):
                    raise ArithmeticError("spatial joint line-content balance")
                emitted=float(np.sum(born_line))/self.state_strip_width_m
                self.ast_emitted_line_per_m+=emitted
                self.ast_escaped_line_per_m+=escape
                self.ast_recovered_line_per_m+=recovered
                totals["emitted_per_m"]+=emitted
                self.last_source_activations=A
                self.last_line_content=born_line
                self.last_emission_drive_Pa=drive.copy()
                self.last_sigma_back_initial_Pa=kback*np.sqrt(np.maximum(rho_base,0.))
                self.last_sigma_effective_final_Pa=sigma_final
                self.last_rate_initial_s=rate_initial
                self.last_rate_final_s=rate_final
                self.cumulative_source_activations+=A
                self.cumulative_line_content+=born_line
                self.interval_source_activations+=A
                self.interval_line_content+=born_line
                self.interval_tip_radius_max_m=max(float(self.interval_tip_radius_max_m),
                                                   float(self.tip_radius_m()))
                self.source_available_m2[...]=self.p.rho_source0_m2
                rates_end=self.local_rates(K_MPa_sqrt_m,T_K)
                totals["annihilated_per_m"]+=self._annihilate_exact(rates_end,.5*h)
                self.ast_joint_substeps=int(getattr(self,"ast_joint_substeps",0))+1
                self.ast_joint_duration_s=float(getattr(self,"ast_joint_duration_s",0.))+h
            return totals
    return AstMechanicalState


def predict_spatial(spatial,candidate,physics,specimen:Specimen,T_K:float,
                    Kdot_MPa_sqrt_m_per_s:float,assumptions,
                    *,event_seed:int|None=None,deterministic_threshold:bool=False,
                    required_u_m:float=0.0,required_extension_m:float|None=None,
                    translation_mode:str="hazard_coupled",
                    translation_action_exponent:float=.95,
                    section:SectionPlasticityConfig|None=None,
                    tip_interaction:str="native")->Prediction:
    """Run native 1-D state steps under shared scalar force equilibrium.

    This retains the native v9.13 action-coupled state translation convention;
    only the observation/mechanics loading callback is added. The committed
    geometric crack length changes after first passage, never during a trial.
    ``section`` selects tip-only plasticity (default) or the full-section
    option, in which the same native mechanisms also act throughout the beam.
    ``tip_interaction`` selects the native crack-tip state or the signed
    pile-up variant (see ``_mechanical_state_type``); the bulk section is
    unchanged by it.
    """
    if translation_mode != "hazard_coupled" or translation_action_exponent<=0:
        raise ValueError("Ast confirmation requires declared hazard-coupled translation")
    state = _mechanical_state_type(spatial.EmergentGNDState,specimen,
                                   tip_interaction)(candidate,physics)
    state.ast_section=(SectionPlasticity(candidate,physics,specimen,section,
                                         spatial.EmergentGNDState)
                       if section is not None and section.active else None)
    state.ast_section_saturated=False
    rng=np.random.default_rng(assumptions.event_seed if event_seed is None else event_seed)
    threshold=1.0 if deterministic_threshold else float(rng.exponential())
    action=0.0;committed_path=0.0;extension=0.0
    u=upl=force=time=0.0
    speed=specimen.initial_displacement_rate(Kdot_MPa_sqrt_m_per_s)
    du_nominal=(assumptions.nominal_K_step_MPa_sqrt_m*1e6*
                (specimen.initial_compliance_m_per_N+specimen.instrument_compliance_m_per_N)/
                specimen.geometry_factor(specimen.a0_m))
    horizon_u=(assumptions.maximum_nominal_K_MPa_sqrt_m*1e6*
               (specimen.initial_compliance_m_per_N+specimen.instrument_compliance_m_per_N)/
               specimen.geometry_factor(specimen.a0_m))
    history=[HistoryPoint(0,0,0,specimen.a0_m,0,0)]
    events=[];steps=0;max_dH=0.0
    target=(assumptions.target_extension_m if required_extension_m is None else required_extension_m)

    def trial(fraction,old_state,du,old_committed_path):
        dt=fraction*du/speed
        unew=u+fraction*du
        a=specimen.a0_m+extension
        K0=specimen.current_K_MPa_sqrt_m(force,a)
        elastic_force=specimen.equilibrium_force(unew,upl,a)
        def evaluate_force(Fused,fixed_translation_steps=None):
            znew=copy.deepcopy(old_state)
            znew.ast_committed_a_m=a
            znew.ast_section_saturated=False
            K1=specimen.current_K_MPa_sqrt_m(Fused,a)
            r0=znew.cleavage_rate_s(K0,T_K)
            r1pred=znew.cleavage_rate_s(K1,T_K)
            Hpred=.5*(r0+r1pred)*dt
            da_step=partial_path_increment(action,Hpred,threshold,old_committed_path,
                                           assumptions.event_advance_m,
                                           translation_action_exponent)
            resolved=spatial._state_advance_is_resolved(
                znew,.5*(K0+K1),T_K,dt,minimum_expected_activations=0.0)
            if resolved and (da_step>0 or fixed_translation_steps is not None):
                if fixed_translation_steps is None:
                    znew.advance_coupled_segment(duration_s=dt,da_m=da_step,
                        K_start_MPa_sqrt_m=K0,K_end_MPa_sqrt_m=K1,T_K=T_K,
                        geometry_extension_override_m=extension)
                else:
                    _advance_coupled_fixed_schedule(znew,dt,da_step,K0,K1,T_K,
                                                    extension,fixed_translation_steps)
            elif resolved:
                znew.advance_time(dt,.5*(K0+K1),T_K)
            elif da_step>0:
                znew.time_s+=dt;znew.translate_tip(da_step)
            else:
                znew.time_s+=dt
            birth=birth_plastic_displacement(
                znew.cumulative_source_activations-old_state.cumulative_source_activations,
                physics.activation_to_line_content_per_system,physics,specimen,a,
                old_state.tip_radius_m(),old_state.emission_drive_factors())
            znew.ast_birth_upl_m+=birth
            uplnew=upl+(znew.ast_transport_upl_m-old_state.ast_transport_upl_m)+birth
            if znew.ast_section is not None:
                try:
                    uplnew+=znew.ast_section.advance(dt,force,Fused,a,T_K,
                                                     du_limit_m=max(unew-uplnew,0.0))
                except SectionSaturation:
                    # This trial force cannot be a root: its bulk flow alone
                    # exceeds the imposed displacement, so equilibrium F is 0.
                    znew.ast_section_saturated=True
                    uplnew=max(uplnew,unew)
            Fnew=specimen.equilibrium_force(unew,uplnew,a)
            return znew,uplnew,Fnew,da_step
        Fnew=elastic_force
        for iteration in range(6):
            Fused=Fnew
            znew,uplnew,Fnew,da_step=evaluate_force(Fused)
            if abs(Fnew-Fused)<=1e-5*max(Fnew,1e-12):break
        else:
            # Replay the same old spatial state at each trial force. Strong
            # source/transport feedback can oscillate under Picard iteration;
            # a bounded root is needed before committing any MPZ evolution.
            lo,hi=0.0,elastic_force
            _,_,_,dlo=evaluate_force(lo)
            _,_,_,dhi=evaluate_force(hi)
            max_translation=max(float(old_state.c.moving_tip_cfl)*float(old_state.dx),1e-30)
            fixed_steps=max(ceil(max(dlo,dhi)/max_translation),1)
            zlo,ulo,flo,_=evaluate_force(lo,fixed_steps)
            zhi,uhi,fhi,_=evaluate_force(hi,fixed_steps)
            if flo<lo or fhi>hi+1e-14:
                raise SpatialForceRootFailure("spatial scalar force equilibrium is not bracketed")
            for _ in range(80):
                mid=.5*(lo+hi)
                zmid,umid,fmid,dmid=evaluate_force(mid,fixed_steps)
                if abs(fmid-mid)<=1e-5*max(fmid,1e-12):
                    break
                if fmid>mid:lo=mid
                else:hi=mid
                if mid==lo==hi:
                    raise SpatialForceRootFailure("spatial force root lost floating-point resolution")
            znew,uplnew,Fnew,da_step=zmid,umid,fmid,dmid
            if abs(Fnew-mid)>1e-4*max(Fnew,1e-12):
                raise SpatialForceRootFailure(
                    f"spatial bounded force root did not converge: "
                    f"trial={mid:.9g} N, equilibrium={Fnew:.9g} N, "
                    f"bracket=[{lo:.9g},{hi:.9g}] N")
        if znew.ast_section_saturated:
            raise SpatialForceRootFailure("accepted force root has saturated bulk flow")
        K1=specimen.current_K_MPa_sqrt_m(Fnew,a)
        dH=.5*(old_state.cleavage_rate_s(K0,T_K)+znew.cleavage_rate_s(K1,T_K))*dt
        return znew,unew,uplnew,Fnew,dt,dH,da_step

    try:
        while extension<target-1e-15 or u<required_u_m-1e-15:
            if u>=horizon_u-1e-15:
                return Prediction("model_limit_nominal_K_window",history,events,steps,max_dH)
            if steps>=assumptions.maximum_load_steps:
                return Prediction("numerical_maximum_steps",history,events,steps,max_dH)
            if specimen.nominal_K_MPa_sqrt_m(force)>=assumptions.maximum_nominal_K_MPa_sqrt_m:
                return Prediction("load_map_exhausted",history,events,steps,max_dH)
            steps+=1
            old=copy.deepcopy(state)
            displacement_remainder=(required_u_m-u if required_u_m>u+1e-15
                                    else float("inf"))
            du=min(du_nominal,horizon_u-u,displacement_remainder)
            for reduction in range(13):
                try:
                    znew,unew,uplnew,Fnew,dt,dH,da_step=trial(1,old,du,committed_path)
                    break
                except SpatialForceRootFailure as exc:
                    if reduction==12:
                        raise SpatialForceRootFailure(
                            f"force root failed after 12 replayed step halvings "
                            f"at u={u:.9g} m: {exc}") from exc
                    du*=.5
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
                                            max(state._elastic_external_K(K)-state.K_shield_MPa_sqrt_m(),0)))
                pre=state.source_geometry()
                back=float(np.mean(state.backstress_state()[2]))
                rate_audit=_event_rate_audit(state,candidate,physics,K,T_K)
                remaining=assumptions.event_advance_m-(committed_path+da_step)
                if remaining>1e-15:state.translate_tip(remaining)
                extension+=assumptions.event_advance_m
                state.ast_committed_a_m=specimen.a0_m+extension
                force=specimen.equilibrium_force(u,upl,specimen.a0_m+extension)
                Kpost=specimen.current_K_MPa_sqrt_m(force,specimen.a0_m+extension)
                history.append(HistoryPoint(time,u,force,specimen.a0_m+extension,upl,
                                            max(state._elastic_external_K(Kpost)-state.K_shield_MPa_sqrt_m(),0),event=True))
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
                               "cumulative_birth_plastic_u_m":float(state.ast_birth_upl_m),
                               "cumulative_transport_plastic_u_m":float(state.ast_transport_upl_m),
                               "cumulative_emitted_line_per_m":float(state.ast_emitted_line_per_m),
                               "cumulative_escaped_line_per_m":float(state.ast_escaped_line_per_m),
                               "cumulative_captured_line_per_m":float(state.ast_captured_line_per_m),
                               "cumulative_released_line_per_m":float(state.ast_released_line_per_m),
                               "cumulative_recovered_line_per_m":float(state.ast_recovered_line_per_m),
                               "cumulative_annihilated_line_per_m":float(state.ast_annihilated_line_per_m),
                               "cumulative_translation_exit_line_per_m":float(state.ast_translation_exit_line_per_m),
                               "cumulative_glide_distance_density_integral_per_m":
                                   float(state.ast_glide_distance_density_integral_per_m),
                               "maximum_line_balance_residual_per_m":
                                   float(state.ast_max_line_balance_residual_per_m),
                               "cumulative_activations":float(np.sum(state.cumulative_source_activations))})
                events[-1].update(rate_audit)
                if state.ast_section is not None:
                    events[-1].update(state.ast_section.snapshot(
                        force,specimen.a0_m+extension))
                action=0.0;committed_path=0.0
                threshold=1.0 if deterministic_threshold else float(rng.exponential())
            else:
                state,u,upl,force=znew,unew,uplnew,Fnew;time+=dt
                action+=dH;committed_path+=da_step
                K=specimen.current_K_MPa_sqrt_m(force,specimen.a0_m+extension)
                history.append(HistoryPoint(time,u,force,specimen.a0_m+extension,upl,
                                            max(state._elastic_external_K(K)-state.K_shield_MPa_sqrt_m(),0)))
        final_state=None
        if all(hasattr(state,name) for name in ("mobile_m2","retained_m2",
                                              "accumulated_slip_m2")):
            final_state={"mobile_m2":state.mobile_m2.copy(),
                         "retained_m2":state.retained_m2.copy(),
                         "accumulated_slip_m2":state.accumulated_slip_m2.copy(),
                         "cumulative_activations":state.cumulative_source_activations.copy(),
                         "tip_radius_m":float(state.tip_radius_m()),
                         "joint_substeps":int(getattr(state,"ast_joint_substeps",0)),
                         "accepted_joint_duration_s":float(getattr(state,"ast_joint_duration_s",0.)),
                         "state_time_s":float(state.time_s)}
            if state.ast_section is not None:
                section_state=state.ast_section
                final_state["section"]={**section_state.snapshot(
                        force,specimen.a0_m+extension),
                    "eps_p":section_state.eps_p.copy(),
                    "mobile_m2":section_state.mobile_m2.copy(),
                    "retained_m2":section_state.retained_m2.copy(),
                    "station_s_m":section_state.s.copy(),
                    "fibre_y_m":section_state.y.copy(),
                    "config":section.as_dict()}
        return Prediction("complete",history,events,steps,max_dH,final_state=final_state)
    except Exception as exc:
        return Prediction("numerical_failure",history,events,steps,max_dH,
                          f"{type(exc).__name__}: {exc}")
