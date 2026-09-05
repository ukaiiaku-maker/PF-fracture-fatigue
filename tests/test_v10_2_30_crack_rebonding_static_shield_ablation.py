"""Committed regression coverage for the v10.2.30 static-shield-
attribution study's PRESCRIBED_POST_FIRST_EVENT_COHESIVE_SHIELD ablation
(persistent_site_coupled_hazard_v10229.py::_phase_statistics).

Replaces the scratch debugging scripts used during development (which
first caught the dead-code injection-point bug at persistent_site_
cyclic_v10229.py::preview_cycle_waveform) with permanent coverage, per
review: "A future edit could again move the mechanism to a dead path
while all archived-data verification continued to pass."

Uses the real, fully-composed production engine class
(CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine) via the shared
tests/_crack_rebonding_engine_fixture.py construction recipe -- the same
recipe the S7/S8 qualification tests use -- not an isolated/simplified
engine, so these tests exercise the actual call chain a real run uses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from arrhenius_fracture import crack_rebonding_causal_pilot_v2_v10230 as pilot  # noqa: E402
from arrhenius_fracture import persistent_site_coupled_hazard_v10229 as _coupled_hazard  # noqa: E402
from arrhenius_fracture import persistent_site_cyclic_v10229 as _legacy_cyclic  # noqa: E402
from arrhenius_fracture.fatigue_v1 import FatigueWaveform  # noqa: E402
from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (  # noqa: E402
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine as Engine,
)

from _crack_rebonding_engine_fixture import (  # noqa: E402
    build_real_engine,
    controller,
    rebonding_cfg,
)

K_B_STATIC_Pa_sqrt_m = 900000.0
WAVEFORM = FatigueWaveform(Kmax=18.0e6, R=-0.95, frequency_Hz=1000.0)


def _phase_stats(engine, ctrl, static_control=None):
    if static_control is not None:
        engine._static_shield_control = static_control
    elif hasattr(engine, "_static_shield_control"):
        del engine._static_shield_control
    return _coupled_hazard._phase_statistics(engine, ctrl, WAVEFORM, 300.0)


def _make_controller(n_phase):
    return controller(n_phase)


def _build_engine(cfg):
    return build_real_engine(cfg), {}


def _run(seed: int, static_control: dict | None, *, rb_cfg=None, max_accepted_events: int = 4):
    Engine.configure_hazard(mode="exponential", seed=seed)
    Engine.reset_audit()
    return pilot.run_trajectory(
        name="test", build_engine=_build_engine, make_controller=_make_controller,
        waveform_cls=FatigueWaveform, rebonding_cfg=rb_cfg, R=-0.95,
        reset_engine_registry=Engine.reset_audit, Kmax_Pa_sqrt_m=18.0e6,
        hazard_rng_seed=seed, max_accepted_events=max_accepted_events,
        max_projected_extension_m=1.0, max_wall_seconds=120.0,
        static_shield_control=static_control,
    )


# --- 1. static_shield_control=None is bit-identical to the pre-existing baseline ---

def test_disabled_static_shield_matches_undecorated_engine():
    engine = build_real_engine(None)
    ctrl = controller(16)
    baseline = _phase_stats(engine, ctrl, static_control=None)

    engine2 = build_real_engine(None)
    disabled = _phase_stats(engine2, ctrl, static_control={"enabled": False, "K_b_static_Pa_sqrt_m": K_B_STATIC_Pa_sqrt_m, "first_event_fired": True})
    assert disabled["lambda_avg_s"] == pytest.approx(baseline["lambda_avg_s"], rel=1.0e-12)
    assert disabled["sigma_avg_Pa"] == pytest.approx(baseline["sigma_avg_Pa"], rel=1.0e-12)


# --- 2 & 3. K_b step function: zero for event 0, prescribed for every later event ---

def test_k_b_step_function_zero_then_prescribed():
    result = _run(
        seed=123456,
        static_control={"enabled": True, "K_b_static_Pa_sqrt_m": K_B_STATIC_Pa_sqrt_m, "first_event_fired": False},
    )
    assert result["n_accepted_events"] >= 2
    k_b_seq = [e["K_b_applied_Pa_sqrt_m"] for e in result["events"]]
    assert k_b_seq[0] == 0.0
    assert all(v == K_B_STATIC_Pa_sqrt_m for v in k_b_seq[1:])


# --- 4. K_b=0.9e6 measurably increases post-first-event waiting time ---

def test_k_b_increases_post_first_event_waiting_time():
    off = _run(seed=555555, static_control=None)
    on = _run(
        seed=555555,
        static_control={"enabled": True, "K_b_static_Pa_sqrt_m": K_B_STATIC_Pa_sqrt_m, "first_event_fired": False},
    )
    n = min(len(off["events"]), len(on["events"]))
    assert n >= 3
    # Event 0 (K_b=0 in both) must match exactly (common-random-numbers).
    assert on["events"][0]["waiting_time_s_this_event"] == pytest.approx(
        off["events"][0]["waiting_time_s_this_event"], rel=1.0e-9
    )
    # Every subsequent event must be MEASURABLY slower under the shield.
    for i in range(1, n):
        wt_on = on["events"][i]["waiting_time_s_this_event"]
        wt_off = off["events"][i]["waiting_time_s_this_event"]
        assert wt_on > wt_off * 1.05, f"event {i}: shielded waiting time not measurably larger"


# --- 5. An extreme K_b strongly suppresses later firing ---

def test_extreme_k_b_suppresses_later_events():
    result = _run(
        seed=555555,
        static_control={"enabled": True, "K_b_static_Pa_sqrt_m": 1.4e7, "first_event_fired": False},
        max_accepted_events=5,
    )
    # The hazard becomes so suppressed after event 0 that firing again
    # within a modest wall-time/block budget should not reach the full
    # requested event count.
    assert result["n_accepted_events"] < 5


# --- 6. Structural guard: the legacy (confirmed dead-for-the-real-engine)
# injection point must never silently regain a static-shield branch, and
# the confirmed-correct location must keep carrying it. A single fresh-
# engine numerical probe of _phase_statistics is NOT used here: for this
# fixture engine's (DBTT_A0003837, not A_NATIVE) calibration at a fresh
# (B=0, K_shield=0) state, lambda_cleave saturates at an identical ceiling
# whether or not K_b_static is subtracted -- a degenerate sample point for
# THIS fixture, not evidence the mechanism is broken (the real A_NATIVE
# engine, and this same fixture across a full multi-event trajectory in
# test_k_b_increases_post_first_event_waiting_time above, both show clean,
# non-degenerate sensitivity). A source-level structural check is robust
# to that per-engine saturation quirk and directly guards against the
# regression the review specifically warned about: "a future edit could
# again move the mechanism to a dead path while all archived-data
# verification continued to pass."

def test_static_shield_wired_only_at_confirmed_correct_location():
    import inspect

    legacy_source = inspect.getsource(_legacy_cyclic)
    assert "_static_shield_control" not in legacy_source, (
        "persistent_site_cyclic_v10229.py (confirmed dead code for the real "
        "production engine hierarchy per docs/v10_2_30_crack_rebonding_"
        "equation_lineage.md) must never carry a static-shield branch again "
        "-- this is exactly the trap the first implementation attempt fell "
        "into, caught only by a K_b sensitivity sweep before it silently "
        "under-tested the real engine"
    )

    coupled_source = inspect.getsource(_coupled_hazard._phase_statistics)
    assert "_static_shield_control" in coupled_source
    assert "static_shield_active" in coupled_source


# --- 7. Ordinary K_values/plastic/emission channel (sigma_avg/min/max,
# r_eff) is unaffected by the static-shield branch; verified both at the
# _phase_statistics level (structural: sigma comes from `sig`, computed
# before the hazard_coupled/static_shield_active branch, never from
# sig_cleave) and via the full multi-event pipeline (final crack
# advance/accepted-length sequence, which the emission/geometry channel
# alone determines, is IDENTICAL with the shield on or off). ---

def test_only_cleavage_channel_is_modified_not_emission_channel():
    engine_off = build_real_engine(None)
    engine_on = build_real_engine(None)
    ctrl = controller(16)
    off = _phase_stats(engine_off, ctrl, static_control=None)
    on = _phase_stats(
        engine_on, ctrl,
        static_control={"enabled": True, "K_b_static_Pa_sqrt_m": K_B_STATIC_Pa_sqrt_m, "first_event_fired": True},
    )
    assert on["sigma_avg_Pa"] == pytest.approx(off["sigma_avg_Pa"], rel=1.0e-12)
    assert on["sigma_min_Pa"] == pytest.approx(off["sigma_min_Pa"], rel=1.0e-12)
    assert on["sigma_max_Pa"] == pytest.approx(off["sigma_max_Pa"], rel=1.0e-12)
    assert on["r_eff_m"] == pytest.approx(off["r_eff_m"], rel=1.0e-12)

    # Full-pipeline confirmation: accepted event lengths (the emission/
    # geometry-gate-determined quantity) are unaffected by the shield,
    # even though waiting times (the cleavage-hazard-determined quantity,
    # see test_k_b_increases_post_first_event_waiting_time) clearly are.
    off_traj = _run(seed=555555, static_control=None)
    on_traj = _run(
        seed=555555,
        static_control={"enabled": True, "K_b_static_Pa_sqrt_m": K_B_STATIC_Pa_sqrt_m, "first_event_fired": False},
    )
    n = min(len(off_traj["events"]), len(on_traj["events"]))
    assert n >= 3
    for i in range(n):
        assert on_traj["events"][i]["accepted_length_m"] == pytest.approx(
            off_traj["events"][i]["accepted_length_m"], rel=1.0e-9
        )


# --- 8. Dynamic rebonding and the prescribed-static control are mutually
# exclusive: dynamic rebonding takes priority, static shield is ignored.
# Verified via the full multi-event pipeline (robust to the single-shot
# _phase_statistics saturation quirk noted above): installing BOTH must
# reproduce the dynamic-only waiting-time sequence exactly, not the
# static-only sequence. ---

def test_dynamic_rebonding_takes_priority_over_static_shield():
    seed = 555555
    dynamic_only = _run(seed=seed, static_control=None, rb_cfg=rebonding_cfg())
    both = _run(
        seed=seed, rb_cfg=rebonding_cfg(),
        static_control={"enabled": True, "K_b_static_Pa_sqrt_m": K_B_STATIC_Pa_sqrt_m, "first_event_fired": False},
    )
    static_only = _run(
        seed=seed, static_control={"enabled": True, "K_b_static_Pa_sqrt_m": K_B_STATIC_Pa_sqrt_m, "first_event_fired": False},
    )
    n = min(len(dynamic_only["events"]), len(both["events"]), len(static_only["events"]))
    assert n >= 3
    for i in range(n):
        wt_both = both["events"][i]["waiting_time_s_this_event"]
        wt_dynamic = dynamic_only["events"][i]["waiting_time_s_this_event"]
        wt_static = static_only["events"][i]["waiting_time_s_this_event"]
        assert wt_both == pytest.approx(wt_dynamic, rel=1.0e-9)
        if i >= 1:
            assert wt_both != pytest.approx(wt_static, rel=1.0e-6)


# --- Section C: synthetic-input regression for the localization-parity
# audit's core formulas. Directly guards against the exact units bug
# found while first writing that audit (an erroneous extra period_s
# factor in the "constant segment" firing-time formula, which produced a
# spurious ~99.9% discrepancy against the exact integrator before being
# caught and fixed here). Uses a hand-computable constant-lambda case
# (not the real engine, so this runs in milliseconds) rather than
# re-deriving the physical audit numbers, which live in scripts/audit_
# static_shield_localization_parity.py's own committed JSON report. ---

def test_localization_parity_formulas_agree_for_constant_lambda():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from audit_static_shield_localization_parity import (
        _constant_segment_firing_time,
        _exact_firing_time,
    )
    import numpy as np

    # A perfectly CONSTANT lambda sequence: the constant-segment method
    # and the exact integrator must agree EXACTLY (to float precision),
    # regardless of B_start -- this is the case the units bug silently
    # broke by a factor of exactly `period_s`.
    n_phase = 80
    period_s = 1.0e-3
    dt_phase = period_s / n_phase
    lam_constant = 500.0
    lambdas = np.full(n_phase, lam_constant, dtype=float)

    for B_start in (0.0, 0.3, 0.7, 0.95):
        t_constant_segment = _constant_segment_firing_time(lam_constant, B_start, period_s)
        t_exact = _exact_firing_time(lambdas, B_start, dt_phase, period_s)
        assert t_constant_segment == pytest.approx(t_exact, rel=1.0e-6), (
            "for a truly constant lambda, the constant-segment formula must exactly "
            "reproduce the phase-resolved integrator -- a mismatch here almost always "
            "means a units error (e.g. a stray period_s factor), not a real physical "
            "discrepancy"
        )
        # Sanity: t = remaining_B / lambda for a constant rate.
        expected = (1.0 - B_start) / lam_constant
        assert t_exact == pytest.approx(expected, rel=1.0e-6)
