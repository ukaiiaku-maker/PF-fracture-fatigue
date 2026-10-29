#!/usr/bin/env python3
"""Emission-owned blunting reduction for the QSTE340TM calibration.

The historical sharp-front model and the current spatial MPZ owner both make
tip blunting depend on source-linked accumulated slip, not on the instantaneous
retained population.  This module restores that ownership to the analytical
QSTE reduction while keeping population backstress and stored energy separate.

State coordinates are::

    [mobile, retained, source_activity,
     local_source_slip, local_forward_slip]

``local_source_slip`` receives every emitted line.  ``local_forward_slip``
receives mobile content transported through the forward boundary.  Both are
convected out of the exponentially weighted tip neighborhood as the crack
advances.  Recovery or escape can therefore remove a dislocation population
without undoing the plastic slip that already blunted the tip.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from scripts.calibrate_qste340tm_stateful_exp_floor import (
    BETA_BACK,
    BURGERS_M,
    NOMINAL_FREQUENCY_HZ,
    PHASE_STEPS,
    PROCESS_ZONE_M,
    RHO0_M2,
    SHEAR_MODULUS_PA,
    TEMPERATURE_K,
    TIP_RADIUS_M,
    exp_floor_barrier,
)
from scripts.qste340tm_faithful_reduction import (
    ENCOUNTER_EFFICIENCY,
    JUMP_FRACTION,
    RESOLVED_STRESS_FRACTION,
    SOURCE_MULTIPLICITY,
    PlasticParameters,
    _exchange_matrix,
)


MOBILE = 0
RETAINED = 1
SOURCE_ACTIVITY = 2
LOCAL_SOURCE_SLIP = 3
LOCAL_FORWARD_SLIP = 4
WAKE_MOBILE = 5
WAKE_RETAINED = 6
STATE_SIZE = 7
HOMOGENEOUS_SIZE = STATE_SIZE + 1


@dataclass(frozen=True)
class BluntingParameters:
    """Shared, R-independent geometric conversion of plastic slip to radius."""

    source_slip_coefficient: float = 1.0
    forward_slip_coefficient: float = 1.0
    blunting_length_m: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.source_slip_coefficient < 0.0:
            raise ValueError("source-slip blunting coefficient must be nonnegative")
        if self.forward_slip_coefficient < 0.0:
            raise ValueError("forward-slip blunting coefficient must be nonnegative")
        if not math.isfinite(self.blunting_length_m) or self.blunting_length_m <= 0.0:
            raise ValueError("blunting length must be positive and finite")


def initial_state(
    plastic: PlasticParameters,
    *,
    initial_source_slip: float = 0.0,
    initial_forward_slip: float = 0.0,
) -> np.ndarray:
    return np.asarray(
        [
            plastic.initial_mobile_count,
            plastic.initial_retained_count,
            plastic.initial_source_activity,
            initial_source_slip,
            initial_forward_slip,
            0.0,
            0.0,
        ],
        dtype=float,
    )


def effective_radius(state: np.ndarray, blunting: BluntingParameters) -> float:
    state = np.asarray(state, dtype=float)
    slip = (
        blunting.source_slip_coefficient * max(float(state[LOCAL_SOURCE_SLIP]), 0.0)
        + blunting.forward_slip_coefficient * max(float(state[LOCAL_FORWARD_SLIP]), 0.0)
    )
    return max(TIP_RADIUS_M + BURGERS_M * slip, TIP_RADIUS_M)


def population_backstress(mobile: float, retained: float) -> float:
    """Population-owned Taylor backstress; plastic ledgers do not enter."""
    rho_tip = max(float(mobile) + float(retained), 0.0) / PROCESS_ZONE_M**2
    return (
        BETA_BACK
        * SHEAR_MODULUS_PA
        * BURGERS_M
        * math.sqrt(max(rho_tip, 0.0))
        / RESOLVED_STRESS_FRACTION
    )


def _state_with_wake(state: np.ndarray) -> np.ndarray:
    """Upgrade historical five-coordinate states without changing their meaning."""
    values = np.asarray(state, dtype=float).reshape(-1)
    if values.size == STATE_SIZE:
        return values.copy()
    if values.size == 5:
        return np.r_[values, 0.0, 0.0]
    raise ValueError(f"expected a five- or {STATE_SIZE}-coordinate QSTE state")


def wake_tip_projection(blunting: BluntingParameters) -> float:
    """Geometric attenuation of the one-moment wake at the moving tip.

    The wake moment represents content centered one half blunting length behind
    the tip.  The square-root crack-tip kernel supplies the attenuation.  This
    is fixed by existing lengths and is not an independently fitted fraction.
    """
    return math.sqrt(
        PROCESS_ZONE_M
        / (PROCESS_ZONE_M + 0.5 * blunting.blunting_length_m)
    )


def effective_population(state: np.ndarray, blunting: BluntingParameters) -> tuple[float, float]:
    values = _state_with_wake(state)
    projection = wake_tip_projection(blunting)
    return (
        float(values[MOBILE] + projection * values[WAKE_MOBILE]),
        float(values[RETAINED] + projection * values[WAKE_RETAINED]),
    )


def stress_cycle(
    delta_k: float,
    r_value: float,
    state: np.ndarray,
    blunting: BluntingParameters,
    n_phase: int = PHASE_STEPS,
) -> tuple[np.ndarray, float, float]:
    phase = 2.0 * math.pi * (np.arange(int(n_phase)) + 0.5) / int(n_phase)
    fraction = 0.5 * (1.0 + r_value) + 0.5 * (1.0 - r_value) * np.cos(phase)
    radius = effective_radius(state, blunting)
    stress = (
        float(delta_k)
        / (1.0 - float(r_value))
        * 1.0e6
        * fraction
        / math.sqrt(2.0 * math.pi * radius)
    )
    mobile, retained = effective_population(state, blunting)
    backstress = population_backstress(mobile, retained)
    return stress, radius, backstress


def phase_rates(
    stress: np.ndarray,
    state: np.ndarray,
    plastic: PlasticParameters,
    radius_m: float,
    blunting: BluntingParameters,
) -> dict[str, np.ndarray]:
    mobile, retained = effective_population(state, blunting)
    mobile = max(mobile, 0.0)
    retained = max(retained, 0.0)
    effective = np.maximum(stress - population_backstress(mobile, retained), 0.0)
    emission = exp_floor_barrier(
        "qste_emit",
        plastic.emission_G0_298_eV,
        plastic.plastic_sigc_298_Pa,
        1.0e11,
        exp_n=plastic.exp_floor_n,
    ).rate(effective, TEMPERATURE_K)
    peierls = exp_floor_barrier(
        "qste_peierls",
        plastic.transport_G0_298_eV,
        plastic.plastic_sigc_298_Pa,
        1.0e12,
        exp_n=plastic.exp_floor_n,
    ).rate(RESOLVED_STRESS_FRACTION * effective, TEMPERATURE_K)
    forest = RHO0_M2 + retained / PROCESS_ZONE_M**2
    spacing = 1.0 / (2.0 * math.sqrt(max(forest, 1.0)))
    taylor_stress = (
        RESOLVED_STRESS_FRACTION
        * effective
        * spacing
        / max(BURGERS_M, 1.0e-30)
    )
    taylor_single = exp_floor_barrier(
        "qste_taylor",
        plastic.transport_G0_298_eV,
        plastic.plastic_sigc_298_Pa,
        1.0e11,
        exp_n=plastic.exp_floor_n,
    ).rate(taylor_stress, TEMPERATURE_K)
    velocity = JUMP_FRACTION * spacing * peierls
    encounter = ENCOUNTER_EFFICIENCY * velocity * math.sqrt(max(forest, 0.0))
    release = -np.expm1(-np.minimum(np.maximum(taylor_single, 0.0), 700.0))
    escape = velocity / PROCESS_ZONE_M
    clear = velocity / max(float(radius_m), BURGERS_M)
    return {
        "emission": np.asarray(emission),
        "peierls": np.asarray(peierls),
        "taylor_single": np.asarray(taylor_single),
        "release": np.asarray(release),
        "encounter": np.asarray(encounter),
        "escape": np.asarray(escape),
        "clear": np.asarray(clear),
        "velocity": np.asarray(velocity),
    }


def phase_operator(
    hazard: float,
    clear: float,
    encounter: float,
    release: float,
    escape: float,
    retained_recovery: float,
    dt: float,
) -> np.ndarray:
    """Affine phase map with exact gross-source and forward-slip ledgers."""
    hazard = max(float(hazard), 0.0)
    clear = max(float(clear), 0.0)
    total = hazard + clear
    if total > 1.0e-300:
        decay = math.exp(-min(total * dt, 700.0))
        equilibrium = clear / total
        alpha = hazard * (1.0 - decay) / total
        beta = hazard * equilibrium * (dt - (1.0 - decay) / total)
        activity_constant = equilibrium * (1.0 - decay)
    else:
        decay = 1.0
        alpha = beta = activity_constant = 0.0

    source = np.eye(HOMOGENEOUS_SIZE)
    source[MOBILE, SOURCE_ACTIVITY] = alpha
    source[MOBILE, -1] = beta
    source[SOURCE_ACTIVITY, SOURCE_ACTIVITY] = decay
    source[SOURCE_ACTIVITY, -1] = activity_constant
    source[LOCAL_SOURCE_SLIP, SOURCE_ACTIVITY] = alpha
    source[LOCAL_SOURCE_SLIP, -1] = beta

    exchange_map = np.eye(HOMOGENEOUS_SIZE)
    exchange_map[:2, :2] = _exchange_matrix(encounter, release, dt)

    recovery = np.eye(HOMOGENEOUS_SIZE)
    recovery[RETAINED, RETAINED] = math.exp(
        -min(max(float(retained_recovery), 0.0) * dt, 700.0)
    )
    recovery[WAKE_RETAINED, WAKE_RETAINED] = recovery[RETAINED, RETAINED]

    transport = np.eye(HOMOGENEOUS_SIZE)
    mobile_survival = math.exp(-min(max(float(escape), 0.0) * dt, 700.0))
    transport[MOBILE, MOBILE] = mobile_survival
    transport[LOCAL_FORWARD_SLIP, MOBILE] = 1.0 - mobile_survival
    return transport @ recovery @ exchange_map @ source


def cycle_operator(
    delta_k: float,
    r_value: float,
    state: np.ndarray,
    plastic: PlasticParameters,
    blunting: BluntingParameters,
    n_phase: int = PHASE_STEPS,
) -> tuple[np.ndarray, dict[str, object]]:
    stress, radius, backstress = stress_cycle(
        delta_k, r_value, state, blunting, n_phase
    )
    rates = phase_rates(stress, state, plastic, radius, blunting)
    dt = 1.0 / (NOMINAL_FREQUENCY_HZ * int(n_phase))
    operator = np.eye(HOMOGENEOUS_SIZE)
    for phase in range(int(n_phase)):
        panel = phase_operator(
            SOURCE_MULTIPLICITY * rates["emission"][phase],
            rates["clear"][phase],
            rates["encounter"][phase],
            rates["release"][phase],
            rates["escape"][phase],
            plastic.retained_recovery_per_s,
            dt,
        )
        operator = panel @ operator
    return operator, {
        "stress": stress,
        "radius": radius,
        "backstress": backstress,
        "rates": rates,
    }


def apply_cycles(operator: np.ndarray, state: np.ndarray, cycles: float) -> np.ndarray:
    state = _state_with_wake(state)
    integer = max(int(round(float(cycles))), 0)
    if integer == 0:
        return np.asarray(state, dtype=float).copy()
    powered = np.linalg.matrix_power(operator, integer)
    output = np.real_if_close(powered @ np.r_[state, 1.0], tol=1000)
    if np.iscomplexobj(output):
        raise FloatingPointError("complex affine cycle power")
    result = np.asarray(output[:STATE_SIZE], dtype=float)
    nonnegative = [MOBILE, RETAINED, LOCAL_SOURCE_SLIP, LOCAL_FORWARD_SLIP, WAKE_MOBILE, WAKE_RETAINED]
    result[nonnegative] = np.maximum(
        result[nonnegative], 0.0
    )
    result[SOURCE_ACTIVITY] = float(np.clip(result[SOURCE_ACTIVITY], 0.0, 1.0))
    return result


def convect_local_plasticity(
    state: np.ndarray,
    crack_advance_m: float,
    blunting: BluntingParameters,
) -> np.ndarray:
    """Translate the exponentially weighted slip neighborhood with the tip."""
    output = np.asarray(state, dtype=float).copy()
    survival = math.exp(
        -max(float(crack_advance_m), 0.0) / blunting.blunting_length_m
    )
    output[LOCAL_SOURCE_SLIP] *= survival
    output[LOCAL_FORWARD_SLIP] *= survival
    return output


def resolve_increment_advection(
    state: np.ndarray,
    crack_advance_m: float,
    blunting: BluntingParameters,
    *,
    maximum_step_m: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Resolve the moving-tip active-to-wake transaction in small increments.

    This is the two-spatial-moment reduction of ``UnifiedMPZState.advance``:
    active content crossed by the tip is deposited in the wake; the existing
    wake and local slip ledgers are translated farther behind the tip.  The
    first resolved transaction therefore determines the reusable coarse
    operator entirely from the existing process-zone and blunting lengths.
    No retained/advection fraction is fitted.
    """
    current = _state_with_wake(state)
    distance = max(float(crack_advance_m), 0.0)
    if distance == 0.0:
        return current, {
            "advection_substeps": 0.0,
            "active_to_wake_mobile": 0.0,
            "active_to_wake_retained": 0.0,
            "wake_mobile_discarded": 0.0,
            "wake_retained_discarded": 0.0,
            "population_balance_residual": 0.0,
        }
    step_limit = (
        0.25 * min(PROCESS_ZONE_M, blunting.blunting_length_m)
        if maximum_step_m is None
        else max(float(maximum_step_m), 1.0e-30)
    )
    n_steps = max(int(math.ceil(distance / step_limit - 1.0e-12)), 1)
    step = distance / n_steps
    crossed_m = crossed_r = discarded_m = discarded_r = 0.0
    initial_population = float(
        current[MOBILE] + current[RETAINED] + current[WAKE_MOBILE] + current[WAKE_RETAINED]
    )
    for _ in range(n_steps):
        active_survival = math.exp(-step / max(PROCESS_ZONE_M, 1.0e-30))
        wake_survival = math.exp(-step / max(blunting.blunting_length_m, 1.0e-30))
        old_wm = float(current[WAKE_MOBILE])
        old_wr = float(current[WAKE_RETAINED])
        dm = float(current[MOBILE]) * (1.0 - active_survival)
        dr = float(current[RETAINED]) * (1.0 - active_survival)
        current[MOBILE] *= active_survival
        current[RETAINED] *= active_survival
        current[WAKE_MOBILE] = old_wm * wake_survival + dm
        current[WAKE_RETAINED] = old_wr * wake_survival + dr
        crossed_m += dm
        crossed_r += dr
        discarded_m += old_wm * (1.0 - wake_survival)
        discarded_r += old_wr * (1.0 - wake_survival)
        current = convect_local_plasticity(current, step, blunting)
    final_population = float(
        current[MOBILE] + current[RETAINED] + current[WAKE_MOBILE] + current[WAKE_RETAINED]
    )
    return current, {
        "advection_substeps": float(n_steps),
        "active_to_wake_mobile": crossed_m,
        "active_to_wake_retained": crossed_r,
        "wake_mobile_discarded": discarded_m,
        "wake_retained_discarded": discarded_r,
        "population_balance_residual": initial_population - final_population - discarded_m - discarded_r,
        "active_population_survival": math.exp(-distance / max(PROCESS_ZONE_M, 1.0e-30)),
        "local_blunting_survival": math.exp(-distance / max(blunting.blunting_length_m, 1.0e-30)),
        "wake_tip_projection": wake_tip_projection(blunting),
    }


def advance_interval(
    delta_k: float,
    r_value: float,
    state: np.ndarray,
    cycles: float,
    crack_advance_m: float,
    plastic: PlasticParameters,
    blunting: BluntingParameters,
    *,
    n_phase: int = PHASE_STEPS,
    radius_fraction_per_chunk: float = 0.05,
    maximum_chunk_cycles: int = 100_000,
    maximum_chunks: int = 20_000,
    resolved_population_advection: bool = False,
    advection_step_m: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Advance nonlinear radius feedback with adaptive frozen-coefficient chunks."""
    remaining = max(int(round(float(cycles))), 0)
    current = np.asarray(state, dtype=float).copy()
    total_cycles = max(remaining, 1)
    chunks = 0
    transaction = {
        "advection_substeps": 0.0,
        "active_to_wake_mobile": 0.0,
        "active_to_wake_retained": 0.0,
        "wake_mobile_discarded": 0.0,
        "wake_retained_discarded": 0.0,
        "population_balance_residual": 0.0,
    }
    while remaining:
        operator, diagnostic = cycle_operator(
            delta_k, r_value, current, plastic, blunting, n_phase
        )
        one = apply_cycles(operator, current, 1)
        radius = effective_radius(current, blunting)
        radius_one = effective_radius(one, blunting)
        radius_rate = max(radius_one - radius, 0.0)
        if radius_rate > 0.0:
            radius_limited = max(
                int(radius_fraction_per_chunk * radius / radius_rate), 1
            )
        else:
            radius_limited = maximum_chunk_cycles
        chunk = min(remaining, maximum_chunk_cycles, radius_limited)
        current = apply_cycles(operator, current, chunk)
        increment = max(float(crack_advance_m), 0.0) * chunk / total_cycles
        if resolved_population_advection:
            current, resolved = resolve_increment_advection(
                current, increment, blunting, maximum_step_m=advection_step_m
            )
            for key in transaction:
                transaction[key] += float(resolved.get(key, 0.0))
        else:
            current = convect_local_plasticity(current, increment, blunting)
        remaining -= chunk
        chunks += 1
        if chunks > maximum_chunks:
            raise RuntimeError("emission-blunting interval exceeded adaptive chunk limit")
    mobile_eff, retained_eff = effective_population(current, blunting)
    return current, {
        "chunks": float(chunks),
        "tip_radius_m": effective_radius(current, blunting),
        "backstress_Pa": population_backstress(mobile_eff, retained_eff),
        "local_source_slip": float(current[LOCAL_SOURCE_SLIP]),
        "local_forward_slip": float(current[LOCAL_FORWARD_SLIP]),
        "active_mobile": float(current[MOBILE]),
        "active_retained": float(current[RETAINED]),
        "wake_mobile": float(current[WAKE_MOBILE]),
        "wake_retained": float(current[WAKE_RETAINED]),
        **transaction,
    }

