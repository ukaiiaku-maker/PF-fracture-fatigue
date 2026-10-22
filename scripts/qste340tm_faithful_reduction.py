#!/usr/bin/env python3
"""Minimal faithful QSTE reduction of the native continuum-source MPZ owner.

Emission enters mobile content. Encounter/Taylor exchange transfers content
between mobile and retained inventories. Recovery and forward transport then
remove their supported populations. The source activity follows the native
aggregate-channel occupancy equation and is reactivated by Peierls clearing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np

from scripts.calibrate_qste340tm_stateful_exp_floor import (
    BETA_BACK, BURGERS_M, C_BLUNT, NOMINAL_FREQUENCY_HZ, PHASE_STEPS,
    POISSON_RATIO, PROCESS_ZONE_M, RHO0_M2, SHEAR_MODULUS_PA,
    TEMPERATURE_K, TIP_RADIUS_M, V_EMB_B3, exp_floor_barrier,
)
from arrhenius_fracture.config import EV_TO_J

RESOLVED_STRESS_FRACTION = 1.0 / math.sqrt(3.0)
SOURCE_MULTIPLICITY = 1.0
ENCOUNTER_EFFICIENCY = 1.0
JUMP_FRACTION = 1.0


@dataclass(frozen=True)
class PlasticParameters:
    emission_G0_298_eV: float
    plastic_sigc_298_Pa: float
    transport_G0_298_eV: float
    retained_recovery_per_s: float
    initial_mobile_count: float
    initial_retained_count: float
    initial_source_activity: float = 1.0
    exp_floor_n: float = 1.0


def embrittlement_eV(retained: float) -> float:
    rho = RHO0_M2 + max(float(retained), 0.0) / PROCESS_ZONE_M**2
    stored = 0.5 * SHEAR_MODULUS_PA * BURGERS_M**2 * rho
    return stored * V_EMB_B3 * BURGERS_M**3 / EV_TO_J


def stress_cycle(delta_k: float, r_value: float, mobile: float, retained: float,
                 n_phase: int = PHASE_STEPS) -> tuple[np.ndarray, float, float]:
    phase = 2.0 * math.pi * (np.arange(n_phase) + 0.5) / n_phase
    fraction = 0.5 * (1.0 + r_value) + 0.5 * (1.0 - r_value) * np.cos(phase)
    radius = TIP_RADIUS_M + C_BLUNT * BURGERS_M * max(retained, 0.0)
    stress = delta_k / (1.0-r_value) * 1.0e6 * fraction / math.sqrt(2.0*math.pi*radius)
    rho_tip = max(mobile + retained, 0.0) / PROCESS_ZONE_M**2
    backstress = (BETA_BACK * SHEAR_MODULUS_PA * BURGERS_M
                  * math.sqrt(max(rho_tip, 0.0)) / RESOLVED_STRESS_FRACTION)
    return stress, radius, backstress


def phase_rates(stress: np.ndarray, mobile: float, retained: float,
                parameters: PlasticParameters) -> dict[str, np.ndarray]:
    effective = np.maximum(stress - stress_cycle_backstress(mobile, retained), 0.0)
    emission = exp_floor_barrier(
        "qste_emit", parameters.emission_G0_298_eV,
        parameters.plastic_sigc_298_Pa, 1.0e11,
        exp_n=parameters.exp_floor_n,
    ).rate(effective, TEMPERATURE_K)
    peierls = exp_floor_barrier(
        "qste_peierls", parameters.transport_G0_298_eV,
        parameters.plastic_sigc_298_Pa, 1.0e12,
        exp_n=parameters.exp_floor_n,
    ).rate(RESOLVED_STRESS_FRACTION*effective, TEMPERATURE_K)
    forest = RHO0_M2 + max(retained, 0.0)/PROCESS_ZONE_M**2
    spacing = 1.0/(2.0*math.sqrt(max(forest, 1.0)))
    # Native UnifiedMPZState resolves Taylor stress and then scales it by the
    # obstacle-spacing/Burgers ratio (phi).  Omitting phi turns release into a
    # second Peierls clock and is not the original transport owner.
    taylor_stress = RESOLVED_STRESS_FRACTION*effective*spacing/max(BURGERS_M,1e-30)
    taylor_single = exp_floor_barrier(
        "qste_taylor", parameters.transport_G0_298_eV,
        parameters.plastic_sigc_298_Pa, 1.0e11,
        exp_n=parameters.exp_floor_n,
    ).rate(taylor_stress, TEMPERATURE_K)
    velocity = JUMP_FRACTION*spacing*peierls
    encounter = ENCOUNTER_EFFICIENCY*velocity*math.sqrt(max(forest, 0.0))
    release = -np.expm1(-np.minimum(np.maximum(taylor_single, 0.0), 700.0))
    escape = velocity/PROCESS_ZONE_M
    radius = TIP_RADIUS_M + C_BLUNT*BURGERS_M*max(retained, 0.0)
    clear = velocity/max(radius, BURGERS_M)
    return {"emission":np.asarray(emission), "peierls":np.asarray(peierls),
            "taylor_single":np.asarray(taylor_single), "release":np.asarray(release),
            "encounter":np.asarray(encounter), "escape":np.asarray(escape),
            "clear":np.asarray(clear), "velocity":np.asarray(velocity)}


def stress_cycle_backstress(mobile: float, retained: float) -> float:
    rho_tip = max(float(mobile)+float(retained), 0.0)/PROCESS_ZONE_M**2
    return (BETA_BACK*SHEAR_MODULUS_PA*BURGERS_M
            * math.sqrt(max(rho_tip, 0.0))/RESOLVED_STRESS_FRACTION)


def _exchange_matrix(k_enc: float, k_rel: float, dt: float) -> np.ndarray:
    rate = max(k_enc, 0.0) + max(k_rel, 0.0)
    if rate <= 1.0e-300:
        return np.eye(2)
    f = max(k_enc, 0.0)/rate
    decay = math.exp(-min(rate*dt, 700.0))
    retained_from_mobile = f*(1.0-decay)
    retained_from_retained = f + (1.0-f)*decay
    return np.array([[1.0-retained_from_mobile, 1.0-retained_from_retained],
                     [retained_from_mobile, retained_from_retained]])


def phase_operator(hazard: float, clear: float, encounter: float, release: float,
                   escape: float, retained_recovery: float, dt: float) -> np.ndarray:
    """Return the affine native-order phase map on [mobile, retained, q, 1]."""
    hazard=max(hazard,0.0); clear=max(clear,0.0); total=hazard+clear
    if total > 1.0e-300:
        decay=math.exp(-min(total*dt,700.0)); qeq=clear/total
        alpha=hazard*(1.0-decay)/total
        beta=hazard*qeq*(dt-(1.0-decay)/total)
        qconst=qeq*(1.0-decay)
    else:
        decay=1.0;alpha=beta=qconst=0.0
    source=np.eye(4);source[0,2]=alpha;source[0,3]=beta;source[2,2]=decay;source[2,3]=qconst
    exchange=np.eye(4);exchange[:2,:2]=_exchange_matrix(encounter,release,dt)
    recovery=np.eye(4);recovery[1,1]=math.exp(-min(max(retained_recovery,0.0)*dt,700.0))
    transport=np.eye(4);transport[0,0]=math.exp(-min(max(escape,0.0)*dt,700.0))
    return transport@recovery@exchange@source


def cycle_operator(delta_k: float, r_value: float, state: np.ndarray,
                   parameters: PlasticParameters, n_phase: int=PHASE_STEPS) -> tuple[np.ndarray,dict]:
    mobile,retained,_activity=map(float,state)
    stress,radius,backstress=stress_cycle(delta_k,r_value,mobile,retained,n_phase)
    rates=phase_rates(stress,mobile,retained,parameters)
    dt=1.0/(NOMINAL_FREQUENCY_HZ*n_phase)
    operator=np.eye(4)
    for j in range(n_phase):
        panel=phase_operator(SOURCE_MULTIPLICITY*rates["emission"][j],rates["clear"][j],
            rates["encounter"][j],rates["release"][j],rates["escape"][j],
            parameters.retained_recovery_per_s,dt)
        operator=panel@operator
    return operator,{"stress":stress,"radius":radius,"backstress":backstress,"rates":rates}


def apply_cycles(operator: np.ndarray, state: np.ndarray, cycles: float) -> np.ndarray:
    if cycles <= 0.0:return np.asarray(state,float).copy()
    # The homogeneous affine map always has a unit eigenvalue and can become
    # nearly defective.  Raising all eigenvalues to a large real power magnifies
    # roundoff into fictitious population growth.  Use stable integer repeated
    # squaring and reserve a general fractional power for the remainder < 1.
    # Cycle counts in the experimental replay are interpolated from an integer
    # crack-history coordinate.  Round the inferred interval to the nearest
    # physical cycle rather than taking a non-unique fractional power of this
    # singular stochastic/affine map (maximum timing error: half a cycle).
    integer=max(int(round(float(cycles))),0)
    powered=np.linalg.matrix_power(operator,integer)
    out=np.real_if_close(powered@np.r_[state,1.0],tol=1000)
    if np.iscomplexobj(out):raise FloatingPointError("complex affine cycle power")
    result=np.asarray(out[:3],float)
    result[:2]=np.maximum(result[:2],0.0);result[2]=float(np.clip(result[2],0.0,1.0))
    return result


def one_cycle_with_transfers(delta_k: float,r_value: float,state: np.ndarray,
                             parameters:PlasticParameters,n_phase:int=PHASE_STEPS):
    current=np.r_[np.asarray(state,float),1.0];dt=1.0/(NOMINAL_FREQUENCY_HZ*n_phase)
    emitted=trapped=released=recovered=escaped=0.0
    stress,_,_=stress_cycle(delta_k,r_value,current[0],current[1],n_phase)
    rates=phase_rates(stress,current[0],current[1],parameters)
    endpoint_survivor=0.0; subsequent=0.0
    for j in range(n_phase-1,-1,-1):
        E=rates["escape"][j]*dt
        endpoint_survivor += rates["emission"][j]*dt*(-math.expm1(-E)/E if E>1e-12 else 1.0)*math.exp(-min(subsequent,700.0))
        subsequent += E
    for j in range(n_phase):
        before=current.copy();H=SOURCE_MULTIPLICITY*rates["emission"][j];kc=rates["clear"][j]
        total=H+kc;dec=math.exp(-min(total*dt,700.0)) if total>0 else 1.0;qeq=kc/total if total>0 else 1.0
        integral_q=qeq*dt+(before[2]-qeq)*(1.0-dec)/total if total>0 else before[2]*dt
        em=H*integral_q;emitted+=em
        after_source=before.copy();after_source[0]+=em;after_source[2]=qeq+(before[2]-qeq)*dec if total>0 else before[2]
        ex=_exchange_matrix(rates["encounter"][j],rates["release"][j],dt);after_exchange=after_source.copy();after_exchange[:2]=ex@after_source[:2]
        trapped+=max(after_exchange[1]-after_source[1],0.0);released+=max(after_source[1]-after_exchange[1],0.0)
        after_recovery=after_exchange.copy();after_recovery[1]*=math.exp(-min(parameters.retained_recovery_per_s*dt,700.0));recovered+=after_exchange[1]-after_recovery[1]
        current=after_recovery.copy();current[0]*=math.exp(-min(rates["escape"][j]*dt,700.0));escaped+=after_recovery[0]-current[0]
    balance=state[0]+state[1]+emitted-current[0]-current[1]-recovered-escaped
    return current[:3],{"emitted":emitted,"trapped":trapped,"released":released,
        "recovered":recovered,"escaped":escaped,"inventory_balance_residual":balance,
        "endpoint_mobile_survivor_diagnostic":endpoint_survivor,
        "mobile_final":current[0],"retained_final":current[1],"source_activity_final":current[2],
        "mean_peierls_rate_s":float(np.mean(rates["peierls"])),
        "mean_release_rate_s":float(np.mean(rates["release"])),
        "mean_encounter_rate_s":float(np.mean(rates["encounter"])),
        "mean_escape_rate_s":float(np.mean(rates["escape"]))}
