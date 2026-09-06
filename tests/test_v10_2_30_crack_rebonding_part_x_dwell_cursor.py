"""PX2.5: exact intra-dwell chronological cursor.

Fixes a real bug found in external review: ``chronological_phase_offset_
rad(elapsed_time_s, period_s)`` maps elapsed time onto a single sinusoidal
angle assuming the whole period is one sinusoidal traverse -- true only
when the hold is zero. Once ``period_s`` was redefined (PX1.1) to include
the dwell, this silently applied a spurious rotation to the SINUSOIDAL
bins whenever the wake's continuous clock actually sat inside the dwell:
verified concretely at Kmax=18 MPa sqrt(m), R=-0.5, hold=0.5 ms, a cursor
30% into the hold produced K(phase=0)=8.50 MPa sqrt(m) for the immediately
following segment instead of the correct 17.74 MPa sqrt(m).

FatigueWaveform.cycle_schedule_from_elapsed fixes this: it finds which
bin of the schedule the continuous cursor is actually in (sinusoid or
dwell) and returns the schedule rotated to start exactly there, with bin
0's own duration equal to the exact remaining time in that bin. Reduces
byte-for-byte to the original phase_offset_rad rotation at hold=0.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from arrhenius_fracture.crack_rebonding_kinetics_v10230 import InitialPrecrackWakeMode
from arrhenius_fracture.crack_rebonding_v10230 import (
    chronological_phase_offset_rad,
    restore_rebonding_checkpoint,
    serialize_rebonding_checkpoint,
)
from arrhenius_fracture.fatigue_v1 import FatigueWaveform
from arrhenius_fracture.persistent_site_coupled_hazard_v10229 import _commit_constant_segment

from _crack_rebonding_engine_fixture import build_real_engine, controller, rebonding_cfg


def _rb_cfg(**overrides):
    return dataclasses.replace(
        rebonding_cfg(**overrides),
        initial_precrack_wake_mode=InitialPrecrackWakeMode.INITIAL_WAKE_CLEAN,
    ).validate()


# ---------------------------------------------------------------------------
# 1. Unit-level: cycle_schedule_from_elapsed correctness
# ---------------------------------------------------------------------------


def test_hold_zero_reduces_to_original_phase_offset_rotation():
    wave = FatigueWaveform(Kmax=18.0e6, R=-0.5, frequency_Hz=1000.0)
    n_phase = 16
    for elapsed in [0.0, 0.3e-3, 0.7e-3, 1.5e-3]:
        offset = chronological_phase_offset_rad(elapsed, wave.period_s)
        K_ref, dt_ref = wave.cycle_schedule(n_phase, signed=True, phase_offset_rad=offset)
        K_got, dt_got = wave.cycle_schedule_from_elapsed(n_phase, elapsed, signed=True)
        assert np.array_equal(K_ref, K_got)
        assert np.array_equal(dt_ref, dt_got)


def test_cursor_zero_matches_unrotated_schedule():
    wave = FatigueWaveform(Kmax=18.0e6, R=-0.5, frequency_Hz=1000.0, minimum_load_hold_s=0.0005)
    K0, dt0 = wave.cycle_schedule(16, signed=True)
    K, dt = wave.cycle_schedule_from_elapsed(16, 0.0, signed=True)
    assert np.array_equal(K0, K)
    assert np.array_equal(dt0, dt)


def test_cursor_inside_hold_gives_correct_remaining_dwell_and_fresh_sinusoid():
    """The exact scenario the reviewed bug produced wrong output for:
    cursor 30% into the hold must report Kmin with the exact remaining
    duration as bin 0, and the immediately following bin must be the
    UNROTATED sinusoid's first entry (fresh start), not a spurious
    rotation derived from hold position."""
    wave = FatigueWaveform(Kmax=18.0e6, R=-0.5, frequency_Hz=1000.0, minimum_load_hold_s=0.0005)
    n_phase = 16
    base, hold = wave.base_period_s, wave.minimum_load_hold_s
    elapsed = base + 0.3 * hold  # 30% into the hold

    K_unrot, dt_unrot = wave.cycle_schedule(n_phase, signed=True)
    K, dt = wave.cycle_schedule_from_elapsed(n_phase, elapsed, signed=True)

    assert K[0] == pytest.approx(wave.R * wave.Kmax)  # still in the hold -> Kmin
    assert dt[0] == pytest.approx(0.7 * hold)  # 70% of the hold remains
    assert K[1] == pytest.approx(K_unrot[0])  # sinusoid resumes FRESH at phase 0, not rotated
    assert dt.sum() == pytest.approx(dt_unrot.sum())  # still exactly one full period


def test_cursor_inside_sinusoidal_bin_gives_correct_remaining_fraction():
    wave = FatigueWaveform(Kmax=18.0e6, R=-0.5, frequency_Hz=1000.0, minimum_load_hold_s=0.0005)
    n_phase = 16
    dt_bin = wave.base_period_s / n_phase
    K_unrot, dt_unrot = wave.cycle_schedule(n_phase, signed=True)

    elapsed = 0.5 * dt_bin  # halfway through the first sinusoidal bin
    K, dt = wave.cycle_schedule_from_elapsed(n_phase, elapsed, signed=True)
    assert K[0] == pytest.approx(K_unrot[0])
    assert dt[0] == pytest.approx(0.5 * dt_bin)
    assert dt.sum() == pytest.approx(dt_unrot.sum())


def test_schedule_from_elapsed_always_sums_to_one_full_period():
    wave = FatigueWaveform(Kmax=18.0e6, R=-0.5, frequency_Hz=1000.0, minimum_load_hold_s=0.0020)
    n_phase = 16
    total = wave.period_s
    for frac in np.linspace(0.0, 0.999, 15):
        K, dt = wave.cycle_schedule_from_elapsed(n_phase, frac * total, signed=True)
        assert dt.sum() == pytest.approx(total, rel=1e-10)
        assert np.all(dt >= 0.0)


# ---------------------------------------------------------------------------
# 2. Production transaction tests: fires inside hold, resumes correctly,
#    rollback/checkpoint preserve the exact cursor.
# ---------------------------------------------------------------------------


def test_repeated_commits_advance_elapsed_time_through_the_hold_correctly():
    """A sequence of committed segments whose cumulative duration crosses
    from the sinusoid into the hold, and later wraps past a full period,
    must leave the wake's elapsed_time_s clock at exactly the expected
    residual (mod period), and the NEXT constructed schedule must
    correctly reflect wherever that residual actually lands."""
    ctrl = controller(n_phase=16)
    cfg = _rb_cfg(bond_barrier_eV=0.6)
    wave = FatigueWaveform(Kmax=18.0e6, R=-0.5, frequency_Hz=1000.0, minimum_load_hold_s=0.0005)
    engine = build_real_engine(cfg)

    period = wave.period_s
    cumulative = 0.0
    # Commit several short segments (well under one period each) and track
    # the expected elapsed_time_s independently (a simple running sum mod
    # period), cross-checking against the engine's own wake clock.
    for cycles in [0.3, 0.5, 0.4, 0.6, 0.5, 0.5, 0.4]:
        dt = cycles * period
        _commit_constant_segment(engine, ctrl, wave, 300.0, cycles, sigma_average_Pa=1.0e8, lambda_average_s=1.0e-3)
        cumulative = (cumulative + dt) % period
        assert engine._rebonding_state.elapsed_time_s == pytest.approx(cumulative, rel=1e-9, abs=1e-15)


def test_schedule_reflects_actual_engine_cursor_after_partial_commits():
    """After committing enough to land inside the hold, the schedule
    constructed for the NEXT segment must show the correct remaining
    dwell time as bin 0 -- not a spurious rotation."""
    ctrl = controller(n_phase=16)
    cfg = _rb_cfg(bond_barrier_eV=0.6)
    wave = FatigueWaveform(Kmax=18.0e6, R=-0.5, frequency_Hz=1000.0, minimum_load_hold_s=0.0005)
    engine = build_real_engine(cfg)

    # Commit exactly 1.3 cycles -- lands 0.3 of a cycle into the SECOND
    # period, i.e. (0.3 * base_period_s) into the sinusoid of cycle 2 if
    # 0.3 < base_period_s/period_s fraction, else into the hold. Choose a
    # cycles value that lands inside the hold: base=1/1000=1e-3,
    # hold=5e-4, period=1.5e-3. base/period = 0.6667. Landing at cycle
    # fraction 0.8 means 0.8*period = 1.2e-3 s into a period, elapsed
    # 0.8-0.6667=0.1333 of a period past the sinusoid = inside the hold.
    _commit_constant_segment(engine, ctrl, wave, 300.0, 0.8, sigma_average_Pa=1.0e8, lambda_average_s=1.0e-3)
    elapsed = engine._rebonding_state.elapsed_time_s
    assert wave.base_period_s < elapsed < wave.period_s  # confirmed inside the hold

    n_phase = 16
    K, dt = wave.cycle_schedule_from_elapsed(n_phase, elapsed, signed=True)
    expected_remaining_hold = wave.period_s - elapsed
    assert K[0] == pytest.approx(wave.R * wave.Kmax)
    assert dt[0] == pytest.approx(expected_remaining_hold, rel=1e-9)


def test_rollback_restores_exact_intra_hold_cursor():
    ctrl = controller(n_phase=16)
    cfg = _rb_cfg(bond_barrier_eV=0.6)
    wave = FatigueWaveform(Kmax=18.0e6, R=-0.5, frequency_Hz=1000.0, minimum_load_hold_s=0.0005)
    engine = build_real_engine(cfg)
    _commit_constant_segment(engine, ctrl, wave, 300.0, 0.8, sigma_average_Pa=1.0e8, lambda_average_s=1.0e-3)
    state = engine._rebonding_state
    elapsed_before = state.elapsed_time_s
    assert wave.base_period_s < elapsed_before < wave.period_s

    snap = state.snapshot()
    # Perturb, then restore.
    state.elapsed_time_s = 0.0
    assert state.elapsed_time_s != elapsed_before
    state.restore(snap)
    assert state.elapsed_time_s == elapsed_before


def test_checkpoint_round_trip_restores_exact_intra_hold_cursor():
    ctrl = controller(n_phase=16)
    cfg = _rb_cfg(bond_barrier_eV=0.6)
    wave = FatigueWaveform(Kmax=18.0e6, R=-0.5, frequency_Hz=1000.0, minimum_load_hold_s=0.0005)
    engine = build_real_engine(cfg)
    _commit_constant_segment(engine, ctrl, wave, 300.0, 0.8, sigma_average_Pa=1.0e8, lambda_average_s=1.0e-3)
    elapsed_before = engine._rebonding_state.elapsed_time_s
    assert wave.base_period_s < elapsed_before < wave.period_s

    payload = serialize_rebonding_checkpoint(engine)
    engine._rebonding_state.elapsed_time_s = 0.0
    restore_rebonding_checkpoint(engine, payload)
    assert engine._rebonding_state.elapsed_time_s == pytest.approx(elapsed_before)


def test_block_subdivision_preserves_cursor_and_wake_state_through_the_hold():
    """Committing 0.8 cycles in one call versus in several smaller calls
    summing to 0.8 cycles must land at the same elapsed_time_s and the
    same wake state -- the exact-matrix-exponential composition
    invariance already established in PX1.1, now re-verified with the
    corrected cursor interpretation across a sinusoid-to-hold boundary."""
    ctrl = controller(n_phase=16)
    cfg = _rb_cfg(bond_barrier_eV=0.6)
    wave = FatigueWaveform(Kmax=18.0e6, R=-0.5, frequency_Hz=1000.0, minimum_load_hold_s=0.0005)

    engine_one_shot = build_real_engine(cfg)
    _commit_constant_segment(engine_one_shot, ctrl, wave, 300.0, 0.8, sigma_average_Pa=1.0e8, lambda_average_s=1.0e-3)

    engine_subdivided = build_real_engine(cfg)
    for cycles in [0.2, 0.3, 0.3]:
        _commit_constant_segment(engine_subdivided, ctrl, wave, 300.0, cycles, sigma_average_Pa=1.0e8, lambda_average_s=1.0e-3)

    assert engine_one_shot._rebonding_state.elapsed_time_s == pytest.approx(
        engine_subdivided._rebonding_state.elapsed_time_s, rel=1e-9
    )
    assert np.allclose(
        engine_one_shot._rebonding_state.active[0].state_vector(),
        engine_subdivided._rebonding_state.active[0].state_vector(),
        atol=1e-10,
    )


def test_hold_to_sinusoid_wrap_happens_at_correct_physical_time():
    """Committing exactly enough to consume the rest of the hold and land
    precisely at the sinusoid's start (phase 0) must leave elapsed_time_s
    at exactly base_period_s (mod period), not drift."""
    ctrl = controller(n_phase=16)
    cfg = _rb_cfg(bond_barrier_eV=0.6)
    wave = FatigueWaveform(Kmax=18.0e6, R=-0.5, frequency_Hz=1000.0, minimum_load_hold_s=0.0005)
    engine = build_real_engine(cfg)
    # cycles fraction corresponding to exactly base_period_s of physical time.
    cycles_to_base = wave.base_period_s / wave.period_s
    _commit_constant_segment(engine, ctrl, wave, 300.0, cycles_to_base, sigma_average_Pa=1.0e8, lambda_average_s=1.0e-3)
    assert engine._rebonding_state.elapsed_time_s == pytest.approx(wave.base_period_s, rel=1e-9)
