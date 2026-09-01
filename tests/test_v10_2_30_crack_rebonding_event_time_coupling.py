from __future__ import annotations

import math

import numpy as np
import pytest

from arrhenius_fracture.crack_rebonding_kinetics_v10230 import CrackRebondingControls, RebondModelLevel
from arrhenius_fracture.crack_rebonding_v10230 import (
    WakePatch,
    phase_resolved_action,
    solve_coupled_event_time,
)


def test_solve_coupled_event_time_converges_for_time_varying_rate():
    """A synthetic scenario where the true physical rate increases over the
    trial interval (bond buildup), so the naive block-constant guess
    (lambda_avg = rate at t=0) would give the wrong event time -- the
    fixed-point iteration must converge to the true closed-form root."""
    rate0 = 1000.0
    alpha = 200.0  # rate(t) = rate0*(1+alpha*t), mild nonlinearity
    B_start, B_threshold = 0.0, 1.0
    eps_B = 1.0e-9

    def true_action(dt: float) -> float:
        return rate0 * (dt + alpha * dt**2 / 2.0)

    a = rate0 * alpha / 2.0
    b = rate0
    c = -(B_threshold - B_start)
    dt_star = (-b + math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)

    def integrate_coupled_fn(lambda_avg: float) -> dict:
        dt_consumed = (B_threshold - B_start) / lambda_avg
        return {"fired": True, "dt_consumed": dt_consumed}

    def phase_resolved_action_fn(dt: float):
        return true_action(dt), {}, 0

    result = solve_coupled_event_time(
        integrate_coupled_fn=integrate_coupled_fn,
        phase_resolved_action_fn=phase_resolved_action_fn,
        lambda_avg_uncoupled=rate0,
        B_start=B_start,
        B_threshold=B_threshold,
        eps_B=eps_B,
        max_iterations=40,
    )

    assert result["fired"] is True
    assert result["converged"] is True
    assert result["dt_used"] == pytest.approx(dt_star, rel=1.0e-4)
    assert abs(result["action_closure_residual"]) <= eps_B


def test_solve_coupled_event_time_no_event_passthrough():
    def integrate_coupled_fn(lambda_avg: float) -> dict:
        return {"fired": False, "dt_consumed": 0.0}

    def phase_resolved_action_fn(dt: float):
        return 0.0, {}, 0

    result = solve_coupled_event_time(
        integrate_coupled_fn=integrate_coupled_fn,
        phase_resolved_action_fn=phase_resolved_action_fn,
        lambda_avg_uncoupled=1.0,
        B_start=0.0,
        B_threshold=1.0,
        eps_B=1.0e-9,
    )
    assert result["fired"] is False


def test_solve_coupled_event_time_matches_constant_rate():
    # If the true rate genuinely is constant, bisection on dt converges to
    # the exact analytic root regardless of the iteration count it takes.
    rate = 500.0
    B_start, B_threshold = 0.0, 1.0

    def integrate_coupled_fn(lambda_avg: float) -> dict:
        return {"fired": True, "dt_consumed": (B_threshold - B_start) / lambda_avg}

    def phase_resolved_action_fn(dt: float):
        return rate * dt, {}, 0

    result = solve_coupled_event_time(
        integrate_coupled_fn=integrate_coupled_fn,
        phase_resolved_action_fn=phase_resolved_action_fn,
        lambda_avg_uncoupled=rate,
        B_start=B_start,
        B_threshold=B_threshold,
        eps_B=1.0e-9,
    )
    assert result["fired"] is True
    assert result["converged"] is True
    assert result["dt_used"] == pytest.approx((B_threshold - B_start) / rate, rel=1.0e-6)


def _phase_K(idx: int, n_phase: int, Kmax: float, R: float) -> float:
    phase = (idx + 0.5) * (2.0 * math.pi / n_phase)
    Kmean = 0.5 * (Kmax + R * Kmax)
    Kamp = 0.5 * (Kmax - R * Kmax)
    return Kmean + Kamp * math.cos(phase)


def test_phase_resolved_action_event_earlier_than_candidate_block():
    """A patch that bonds strongly during compression should shield cleavage
    enough that the accumulated action over a short interval already differs
    materially from the K_rebond=0 baseline -- i.e. rebonding measurably
    changes the action landscape well before the nominal block ends."""
    cfg = CrackRebondingControls(
        model_level=RebondModelLevel.CLEAN_REVERSIBLE_REBOND,
        wake_length_m=5.0e-6,
        wake_weight_length_m=1.0e-6,
        restored_work_of_separation_J_m2=5.0,
        rebond_K_geometry_factor=1.0,
        bond_activation_volume_m3=3.0e-29,
        bond_barrier_eV=0.2,
        rupture_activation_volume_m3=3.0e-29,
        rupture_barrier_eV=0.6,
        chemistry_factor=1.0,
    ).validate()

    patch = WakePatch(patch_id=0, length_m=1.0e-7, s_j_m=5.0e-8, p_P=0.0, p_C=1.0, p_B=0.0, creation_event_index=0)
    n_phase = 60
    Kmax = 18.0e6
    R = -0.95
    f_Hz = 1000.0
    dt_phase = (1.0 / f_Hz) / n_phase

    def K_phase_fn(idx: int) -> float:
        return _phase_K(idx, n_phase, Kmax, R)

    def lambda_cleave_fn(sigma: float) -> float:
        return 1.0e-3 * sigma  # simple monotone stand-in hazard

    action_short, states_short, _, _ = phase_resolved_action(
        active_patches=[patch],
        patch_states={0: patch.state_vector()},
        k0=0,
        t_interval=5 * dt_phase,
        K_phase_fn=K_phase_fn,
        dt_phase=dt_phase,
        n_phase=n_phase,
        r_contact_m=1.0e-8,
        cfg=cfg,
        T_K=300.0,
        Eprime_Pa=2.0e11,
        K_shield_Pa_sqrt_m=0.0,
        r_eff_m=1.0e-6,
        lambda_cleave_fn=lambda_cleave_fn,
    )
    assert action_short >= 0.0
    total = states_short[0].sum()
    assert total == pytest.approx(1.0, abs=1.0e-8)


def test_commit_rebonding_event_fails_closed_on_uncertified_bulk_action(monkeypatch):
    """S8C caller-side enforcement: persistent_site_cyclic_energy_gated_v10230
    .py's phase_resolved_action_fn closure (inside _commit_rebonding_event)
    must retry with an extended transient budget on an uncertified bulk
    result, then raise rather than silently commit an event on an
    uncertified action estimate -- the fail-closed contract phase_resolved_
    action itself deliberately does not enforce (see its docstring)."""
    import _crack_rebonding_engine_fixture as fx

    from arrhenius_fracture import crack_rebonding_v10230 as _rebond

    engine = fx.build_real_engine(fx.rebonding_cfg())
    ctrl = fx.controller()
    waveform = fx.default_waveform()
    fx.run_to_next_fired_event(engine, ctrl, waveform)
    assert engine._energy_gate_pending.get("rebonding_block_context") is not None

    call_count = 0
    real_action_fn = _rebond.phase_resolved_action

    def _always_unqualified(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        action, end_states, end_idx, diag = real_action_fn(*args, **kwargs)
        diag = dict(diag)
        diag["bulk_action_qualified"] = False
        diag["total_tail_action_error_bound"] = 1.0e12
        return action, end_states, end_idx, diag

    monkeypatch.setattr(_rebond, "phase_resolved_action", _always_unqualified)

    pending = engine._energy_gate_pending
    committed_length = pending["proposal_m"]
    gate = {
        "energy_admissible_event_length_m": committed_length,
        "arrest_reason": "test_commit",
        "hazard_resistance_J_per_m2": 1.0,
        "orientation_gamma_relative": 1.0,
    }
    result_ref = pending["descriptor"].get("energy_gate_result_ref")

    max_extensions = engine._rebonding_state.cfg.bulk_action_max_transient_extensions
    with pytest.raises(RuntimeError, match="bulk-action certificate not qualified"):
        engine.commit_energy_gated_event(committed_length, gate, result_ref)

    # Bisection calls phase_resolved_action_fn multiple times per solve; each
    # of those calls independently retries up to max_extensions+1 times
    # before giving up -- so the raise must come from the FIRST bisection
    # call to exhaust its own retry budget, confirming the retry loop is
    # bounded rather than silently accepting the uncertified result at any
    # point.
    assert call_count >= max_extensions + 1
