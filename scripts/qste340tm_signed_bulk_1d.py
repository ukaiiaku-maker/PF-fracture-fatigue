#!/usr/bin/env python3
"""Signed bulk-GND transaction for the QSTE340TM moving-state 1-D model.

This is an additive owner layered on the established emission/blunting
reduction.  Forward passage creates the antisymmetric Mode-I signed content
on the two crack faces.  Crack advance sweeps a geometrically determined
fraction of that content into the tip.  Only the outward-opening projection
blunts the tip, through the existing source-slip coefficient; no independent
bulk efficacy or R-dependent coordinate is introduced.

The owner is deliberately a two-face spatial moment.  It is the 1-D analogue
of the signed-GND area transaction used by the native 2-D calculation, not a
claim to resolve the complete 2-D GND field.
"""
from __future__ import annotations

import math

import numpy as np

import scripts.qste340tm_emission_blunting_reduction as base


# The first seven coordinates retain the established state identity.
MOBILE = base.MOBILE
RETAINED = base.RETAINED
SOURCE_ACTIVITY = base.SOURCE_ACTIVITY
LOCAL_SOURCE_SLIP = base.LOCAL_SOURCE_SLIP
LOCAL_FORWARD_SLIP = base.LOCAL_FORWARD_SLIP
WAKE_MOBILE = base.WAKE_MOBILE
WAKE_RETAINED = base.WAKE_RETAINED

BULK_LOWER_SIGNED = 7
BULK_UPPER_SIGNED = 8
BULK_TIP_OPENING = 9
STATE_SIZE = 10
HOMOGENEOUS_SIZE = STATE_SIZE + 1

PlasticParameters = base.PlasticParameters
BluntingParameters = base.BluntingParameters
PROCESS_ZONE_M = base.PROCESS_ZONE_M
TIP_RADIUS_M = base.TIP_RADIUS_M
BURGERS_M = base.BURGERS_M


def initial_state(plastic: PlasticParameters) -> np.ndarray:
    return np.r_[base.initial_state(plastic), 0.0, 0.0, 0.0]


def _state(state: np.ndarray) -> np.ndarray:
    values = np.asarray(state, dtype=float).reshape(-1)
    if values.size == STATE_SIZE:
        return values.copy()
    if values.size in (5, base.STATE_SIZE):
        return np.r_[base._state_with_wake(values), 0.0, 0.0, 0.0]
    raise ValueError(f"expected a five-, seven-, or {STATE_SIZE}-coordinate QSTE state")


def effective_radius(state: np.ndarray, blunting: BluntingParameters) -> float:
    values = _state(state)
    source_owned = (
        max(float(values[LOCAL_SOURCE_SLIP]), 0.0)
        + max(float(values[BULK_TIP_OPENING]), 0.0)
    )
    forward_owned = max(float(values[LOCAL_FORWARD_SLIP]), 0.0)
    slip = (
        blunting.source_slip_coefficient * source_owned
        + blunting.forward_slip_coefficient * forward_owned
    )
    return max(TIP_RADIUS_M + BURGERS_M * slip, TIP_RADIUS_M)


def effective_population(state: np.ndarray, blunting: BluntingParameters) -> tuple[float, float]:
    return base.effective_population(_state(state)[:base.STATE_SIZE], blunting)


def stress_cycle(delta_k: float, r_value: float, state: np.ndarray,
                 blunting: BluntingParameters, n_phase: int = base.PHASE_STEPS):
    values = _state(state)
    phase = 2.0 * math.pi * (np.arange(int(n_phase)) + 0.5) / int(n_phase)
    fraction = 0.5 * (1.0 + r_value) + 0.5 * (1.0 - r_value) * np.cos(phase)
    radius = effective_radius(values, blunting)
    stress = (
        float(delta_k) / (1.0 - float(r_value)) * 1.0e6 * fraction
        / math.sqrt(2.0 * math.pi * radius)
    )
    mobile, retained = effective_population(values, blunting)
    backstress = base.population_backstress(mobile, retained)
    return stress, radius, backstress


def _embed_base_panel(panel: np.ndarray) -> np.ndarray:
    """Embed the accepted seven-state affine panel and expose its passage flux."""
    output = np.eye(HOMOGENEOUS_SIZE)
    output[:base.STATE_SIZE, :base.STATE_SIZE] = panel[:base.STATE_SIZE, :base.STATE_SIZE]
    output[:base.STATE_SIZE, -1] = panel[:base.STATE_SIZE, -1]

    # The exact gross forward-passage increment is the change in the existing
    # forward-slip ledger during this panel.  Mode-I antisymmetry assigns equal
    # and opposite signed content to the lower and upper faces.  Their outward
    # projections have the same opening sign after absorption.
    passage = panel[LOCAL_FORWARD_SLIP].copy()
    passage[LOCAL_FORWARD_SLIP] -= 1.0
    output[BULK_LOWER_SIGNED, :base.STATE_SIZE] += 0.5 * passage[:base.STATE_SIZE]
    output[BULK_LOWER_SIGNED, -1] += 0.5 * passage[-1]
    output[BULK_UPPER_SIGNED, :base.STATE_SIZE] -= 0.5 * passage[:base.STATE_SIZE]
    output[BULK_UPPER_SIGNED, -1] -= 0.5 * passage[-1]
    return output


def cycle_operator(delta_k: float, r_value: float, state: np.ndarray,
                   plastic: PlasticParameters, blunting: BluntingParameters,
                   n_phase: int = base.PHASE_STEPS):
    values = _state(state)
    stress, radius, backstress = stress_cycle(
        delta_k, r_value, values, blunting, n_phase
    )
    rates = base.phase_rates(
        stress, values[:base.STATE_SIZE], plastic, radius, blunting
    )
    dt = 1.0 / (base.NOMINAL_FREQUENCY_HZ * int(n_phase))
    operator = np.eye(HOMOGENEOUS_SIZE)
    for phase in range(int(n_phase)):
        panel = base.phase_operator(
            base.SOURCE_MULTIPLICITY * rates["emission"][phase],
            rates["clear"][phase], rates["encounter"][phase],
            rates["release"][phase], rates["escape"][phase],
            plastic.retained_recovery_per_s, dt,
        )
        operator = _embed_base_panel(panel) @ operator
    return operator, {
        "stress": stress, "radius": radius, "backstress": backstress,
        "rates": rates,
    }


def apply_cycles(operator: np.ndarray, state: np.ndarray, cycles: float) -> np.ndarray:
    values = _state(state)
    integer = max(int(round(float(cycles))), 0)
    if integer == 0:
        return values
    powered = np.linalg.matrix_power(operator, integer)
    raw = np.real_if_close(powered @ np.r_[values, 1.0], tol=1000)
    if np.iscomplexobj(raw):
        raise FloatingPointError("complex signed-bulk affine cycle power")
    output = np.asarray(raw[:STATE_SIZE], dtype=float)
    for index in (MOBILE, RETAINED, LOCAL_SOURCE_SLIP, LOCAL_FORWARD_SLIP,
                  WAKE_MOBILE, WAKE_RETAINED, BULK_TIP_OPENING):
        output[index] = max(float(output[index]), 0.0)
    output[SOURCE_ACTIVITY] = float(np.clip(output[SOURCE_ACTIVITY], 0.0, 1.0))
    return output


def sweep_increment(state: np.ndarray, crack_advance_m: float,
                    blunting: BluntingParameters,
                    *, maximum_step_m: float | None = None):
    """Sweep signed forward content into the moving tip with exact balances."""
    current = _state(state)
    distance = max(float(crack_advance_m), 0.0)
    if distance == 0.0:
        return current, {
            "bulk_advection_substeps": 0.0,
            "bulk_lower_signed_swept": 0.0,
            "bulk_upper_signed_swept": 0.0,
            "bulk_opening_absorbed": 0.0,
            "bulk_signed_balance_residual": 0.0,
        }
    step_limit = (
        0.25 * min(PROCESS_ZONE_M, blunting.blunting_length_m)
        if maximum_step_m is None else max(float(maximum_step_m), 1.0e-30)
    )
    steps = max(int(math.ceil(distance / step_limit - 1.0e-12)), 1)
    step = distance / steps
    lower0 = float(current[BULK_LOWER_SIGNED])
    upper0 = float(current[BULK_UPPER_SIGNED])
    lower_swept = upper_swept = opening = 0.0
    for _ in range(steps):
        crossed = 1.0 - math.exp(-step / max(PROCESS_ZONE_M, 1.0e-30))
        dl = float(current[BULK_LOWER_SIGNED]) * crossed
        du = float(current[BULK_UPPER_SIGNED]) * crossed
        current[BULK_LOWER_SIGNED] -= dl
        current[BULK_UPPER_SIGNED] -= du
        # Lower outward normal is -y and upper outward normal is +y.  The sign
        # convention above therefore makes +lower/-upper opening-effective.
        absorbed = max(dl, 0.0) + max(-du, 0.0)
        current[BULK_TIP_OPENING] *= math.exp(
            -step / max(blunting.blunting_length_m, 1.0e-30)
        )
        current[BULK_TIP_OPENING] += absorbed
        lower_swept += dl
        upper_swept += du
        opening += absorbed
    residual = (
        lower0 - float(current[BULK_LOWER_SIGNED]) - lower_swept
        + upper0 - float(current[BULK_UPPER_SIGNED]) - upper_swept
    )
    return current, {
        "bulk_advection_substeps": float(steps),
        "bulk_lower_signed_swept": lower_swept,
        "bulk_upper_signed_swept": upper_swept,
        "bulk_opening_absorbed": opening,
        "bulk_signed_balance_residual": residual,
    }


def advance_interval(delta_k: float, r_value: float, state: np.ndarray,
                     cycles: float, crack_advance_m: float,
                     plastic: PlasticParameters, blunting: BluntingParameters,
                     *, n_phase: int = base.PHASE_STEPS,
                     radius_fraction_per_chunk: float = 0.05,
                     maximum_chunk_cycles: int = 100_000,
                     maximum_chunks: int = 20_000,
                     advection_step_m: float | None = None):
    remaining = max(int(round(float(cycles))), 0)
    current = _state(state)
    total_cycles = max(remaining, 1)
    chunks = 0
    transaction = {
        "active_to_wake_mobile": 0.0, "active_to_wake_retained": 0.0,
        "wake_mobile_discarded": 0.0, "wake_retained_discarded": 0.0,
        "advection_substeps": 0.0, "bulk_advection_substeps": 0.0,
        "bulk_lower_signed_swept": 0.0, "bulk_upper_signed_swept": 0.0,
        "bulk_opening_absorbed": 0.0, "bulk_signed_balance_residual": 0.0,
    }
    while remaining:
        operator, _ = cycle_operator(
            delta_k, r_value, current, plastic, blunting, n_phase
        )
        one = apply_cycles(operator, current, 1)
        radius = effective_radius(current, blunting)
        radius_rate = max(effective_radius(one, blunting) - radius, 0.0)
        radius_limited = (
            max(int(radius_fraction_per_chunk * radius / radius_rate), 1)
            if radius_rate > 0.0 else maximum_chunk_cycles
        )
        chunk = min(remaining, maximum_chunk_cycles, radius_limited)
        current = apply_cycles(operator, current, chunk)
        increment = max(float(crack_advance_m), 0.0) * chunk / total_cycles

        base_state, population = base.resolve_increment_advection(
            current[:base.STATE_SIZE], increment, blunting,
            maximum_step_m=advection_step_m,
        )
        current[:base.STATE_SIZE] = base_state
        current, bulk = sweep_increment(
            current, increment, blunting, maximum_step_m=advection_step_m
        )
        for key in transaction:
            transaction[key] += float((population | bulk).get(key, 0.0))
        remaining -= chunk
        chunks += 1
        if chunks > maximum_chunks:
            raise RuntimeError("signed-bulk interval exceeded adaptive chunk limit")

    mobile, retained = effective_population(current, blunting)
    return current, {
        "chunks": float(chunks),
        "tip_radius_m": effective_radius(current, blunting),
        "backstress_Pa": base.population_backstress(mobile, retained),
        "local_source_slip": float(current[LOCAL_SOURCE_SLIP]),
        "local_forward_slip": float(current[LOCAL_FORWARD_SLIP]),
        "bulk_lower_signed": float(current[BULK_LOWER_SIGNED]),
        "bulk_upper_signed": float(current[BULK_UPPER_SIGNED]),
        "bulk_tip_opening": float(current[BULK_TIP_OPENING]),
        **transaction,
    }

